#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.stats.kstwo`, the exact finite-n two-sided
//! Kolmogorov–Smirnov law (frankenscipy-1ksfv.16).
//!
//! fsci's `Kstwo` ports SciPy 1.17.1's `kolmogn` (and `kolmognp` for the density). The bead's
//! grid: n ∈ {1, 2, 5, 10, 20, 100, 1000, 10⁵}, with x over the support [1/(2n), 1], including
//! both tails. The mass sits on the 1/√n scale, so x is c/√n for c from 0.3 to 3.5, clipped into
//! the support, plus the lower edge and x = 1, where sf is exactly 0 and must match exactly.
//! Tolerances as the bead sets them: cdf rel 1e-12, sf rel 1e-10 (its far tail). pdf is held
//! to kstwo.pdf at 1e-10 relative, or within its stencil noise (see `scipy_stencil_noise`). In
//! the deep tail (sf < 1e-3), where fsci and SciPy differentiate different things, it is held
//! to the derivative of SciPy's own sf at 1e-6.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_stats::{ContinuousDistribution, Kstwo};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const CDF_REL_TOL: f64 = 1e-12;
const SF_REL_TOL: f64 = 1e-10;
const PDF_REL_TOL: f64 = 1e-10;
/// Against the central difference of SciPy's sf (h = 1e-5·x), whose own accuracy is ~1e-7.
const PDF_FROM_SF_REL_TOL: f64 = 1e-6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const ARMS: [&str; 3] = ["cdf", "sf", "pdf"];
const NS: [usize; 8] = [1, 2, 5, 10, 20, 100, 1000, 100_000];
const C: [f64; 14] = [
    0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.4, 1.6, 1.9, 2.2, 2.6, 3.0, 3.5,
];

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    arm: String,
    n: usize,
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    value: Option<f64>,
    /// pdf cases only: SciPy's `kstwo.sf(x, n)`.
    #[serde(default)]
    sf: Option<f64>,
    /// pdf cases only: the central difference −(sf(x + h) − sf(x − h))/2h of SciPy's own sf,
    /// h = 1e-5·x.
    #[serde(default)]
    pdf_from_sf: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    arm: String,
    rel_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    max_rel_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    fs::create_dir_all(output_dir()).expect("create kstwo diff output dir");
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize kstwo diff log");
    fs::write(path, json).expect("write kstwo diff log");
}

/// The x grid for one n: the lower edge (just inside), c/√n clipped into the support, and 1.
fn grid(n: usize) -> Vec<f64> {
    let lower = 0.5 / n as f64;
    let mut xs = vec![lower * 1.0001, 1.0];
    for &c in &C {
        let x = c / (n as f64).sqrt();
        if x > lower && x < 1.0 {
            xs.push(x);
        }
    }
    xs.sort_by(f64::total_cmp);
    xs.dedup();
    xs
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    for &n in &NS {
        for x in grid(n) {
            for arm in ARMS {
                points.push(PointCase {
                    case_id: format!("{arm}_n{n}_x{x}"),
                    arm: arm.into(),
                    n,
                    x,
                });
            }
        }
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy.stats import kstwo

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    try:
        x = float(case["x"]); n = int(case["n"])
        v = float(getattr(kstwo, case["arm"])(x, n))
        row = {"case_id": cid, "value": v if math.isfinite(v) else None}
        if case["arm"] == "pdf":
            h = 1e-5 * x
            row["sf"] = float(kstwo.sf(x, n))
            row["pdf_from_sf"] = float(-(kstwo.sf(x + h, n) - kstwo.sf(x - h, n)) / (2 * h))
        points.append(row)
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize kstwo query");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for kstwo oracle: {e}"
            );
            eprintln!("skipping kstwo oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open kstwo oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "kstwo oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping kstwo oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for kstwo oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "kstwo oracle failed: {stderr}"
        );
        eprintln!("skipping kstwo oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse kstwo oracle JSON"))
}

/// SciPy's `kstwo.pdf` differentiates its CDF with a five-point stencil of step
/// δ = min(x/65536, x − 1/n, ½ − x) wherever 1 < n·x < n − 1 and x < ½. Its value there carries
/// rounding noise of a few eps/δ, absolute. In the upper tail that noise is the whole value:
/// pdf(0.3; 1000) = 9.1e-12, where the density is ~1e-78. fsci differentiates the sf there. So
/// in that region the pdf arm allows the fsci-stats unit test's STENCIL_NOISE·eps/δ absolute,
/// and PDF_REL_TOL relative everywhere else.
const STENCIL_NOISE: f64 = 4.0;

fn scipy_stencil_noise(n: usize, x: f64) -> Option<f64> {
    let nf = n as f64;
    let t = nf * x;
    (t > 1.0 && t < nf - 1.0 && x < 0.5).then(|| {
        let delta = (x / 65536.0).min(x - 1.0 / nf).min(0.5 - x);
        STENCIL_NOISE * f64::EPSILON / delta
    })
}

/// Relative difference; a zero SciPy value (sf at x = 1) must be matched exactly.
fn rel_diff(fsci: f64, scipy: f64) -> f64 {
    if scipy == 0.0 {
        if fsci == 0.0 { 0.0 } else { f64::INFINITY }
    } else {
        ((fsci - scipy) / scipy).abs()
    }
}

#[test]
fn diff_stats_kstwo() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());
    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;
    let mut ledger = CompareLedger::new("diff_stats_kstwo", &ARMS);

    for case in &query.points {
        let oracle_arm = pmap.get(&case.case_id).expect("validated oracle");
        let stencil = scipy_stencil_noise(case.n, case.x);
        // Where fsci differentiates the sf (the stencil region with sf < 1e-3, see
        // `kolmogn_p`), the reference is the derivative of SciPy's own sf. There SciPy's
        // kstwo.pdf differentiates a CDF within 1e-3 of 1 and, for n > 140, a Pelz–Good CDF that
        // its own sf does not match: 2.5e-4 off at (0.0949, 1000), 8% at (0.3, 141).
        let from_sf =
            case.arm == "pdf" && stencil.is_some() && oracle_arm.sf.is_some_and(|sf| sf < 1e-3);
        let scipy = if from_sf {
            oracle_arm.pdf_from_sf
        } else {
            oracle_arm.value
        };
        let dist = Kstwo::new(case.n).expect("n >= 1");
        // An arm outside ARMS is skipped here and then fails the per-arm count below.
        let Some((fsci, tol)) = (match case.arm.as_str() {
            "cdf" => Some((dist.cdf(case.x), CDF_REL_TOL)),
            "sf" => Some((dist.sf(case.x), SF_REL_TOL)),
            "pdf" if from_sf => Some((dist.pdf(case.x), PDF_FROM_SF_REL_TOL)),
            "pdf" => Some((dist.pdf(case.x), PDF_REL_TOL)),
            _ => None,
        }) else {
            continue;
        };
        let Some((s, f)) = ledger.pair(&case.arm, &case.case_id, scipy, Some(fsci)) else {
            continue;
        };
        let d = rel_diff(f, s);
        let noise = (case.arm == "pdf" && !from_sf).then_some(stencil).flatten();
        let pass = d <= tol || noise.is_some_and(|noise| (f - s).abs() <= noise);
        if noise.is_none() {
            max_overall = max_overall.max(d);
        }
        ledger.compared(&case.arm, &case.case_id, pass);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            arm: case.arm.clone(),
            rel_diff: d,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_stats_kstwo".into(),
        category: "scipy.stats.kstwo".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        max_rel_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "kstwo {} mismatch: {} rel_diff={}",
                d.arm, d.case_id, d.rel_diff
            );
        }
    }
    assert!(
        all_pass,
        "scipy.stats.kstwo conformance failed: {} cases, max_rel_diff={}",
        diffs.len(),
        max_overall
    );
    let counts = ledger.finish(NS.len());
    for arm in ARMS {
        let cases = query.points.iter().filter(|c| c.arm == arm).count();
        assert_eq!(
            counts[arm].compared_cases, cases,
            "arm `{arm}` must compare all {cases} of its cases"
        );
    }
}
