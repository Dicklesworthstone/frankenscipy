#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.stats.vonmises`.
//!
//! Resolves [frankenscipy-q291n]. VonMises (circular Gaussian
//! analogue) has anchor tests in `fsci-stats/src/lib.rs` but no
//! dedicated scipy diff harness. 6 (kappa, loc) pairs × 9
//! x-values × 2 families (pdf, cdf) via subprocess.
//!
//! pdf is closed-form via the modified Bessel I0 helper.
//! cdf uses 2048-step trapezoidal integration over one period;
//! the O(h²) floor at h ≈ 2π/2048 ≈ 3e-3 yields ~1e-5 abs.
//!
//! ppf is intentionally omitted — VonMises has no explicit
//! ppf, so it would inherit the trait-default `ppf_bisection`
//! over the 1e-5-noisy cdf, producing meaningless tolerances.
//!
//! x-grid is bounded within one period [loc−π, loc+π] so we
//! exercise the within-period base_cdf branch (both fsci and
//! scipy use the same "cycles + within-period cdf" convention,
//! verified at vonmises.cdf(2π) ≈ 1.5).

use std::collections::HashMap;
use std::f64::consts::PI;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{ContinuousDistribution, VonMises};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const PDF_TOL: f64 = 1.0e-12;
const CDF_TOL: f64 = 1.0e-5;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    kappa: f64,
    loc: f64,
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    pdf: Option<f64>,
    cdf: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    family: String,
    abs_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_abs_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create vonmises diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize vonmises diff log");
    fs::write(path, json).expect("write vonmises diff log");
}

fn generate_query() -> OracleQuery {
    // kappa = 0 ⇒ uniform on circle. kappa = ∞ ⇒ delta at loc.
    // Pick small/medium/large kappa and a few loc shifts.
    let pairs: [(f64, f64); 6] = [
        (0.5, 0.0),
        (1.0, 0.0),
        (3.0, 0.0),
        (10.0, 0.0),
        (1.0, PI / 4.0),
        (5.0, -PI / 3.0),
    ];
    let mut points = Vec::new();
    for &(kappa, loc) in &pairs {
        // 9 x-values within one period [loc-π, loc+π]
        let frac = [-0.95_f64, -0.7, -0.4, -0.15, 0.0, 0.15, 0.4, 0.7, 0.95];
        for &f in &frac {
            let x = loc + f * PI;
            points.push(PointCase {
                case_id: format!("k{kappa}_l{loc}_f{f}"),
                kappa,
                loc,
                x,
            });
        }
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
from scipy.stats import vonmises

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    kappa = float(case["kappa"]); loc = float(case["loc"]); x = float(case["x"])
    try:
        points.append({
            "case_id": cid,
            "pdf": float(vonmises.pdf(x, kappa, loc=loc)),
            "cdf": float(vonmises.cdf(x, kappa, loc=loc)),
        })
    except Exception:
        points.append({"case_id": cid, "pdf": None, "cdf": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize vonmises query");
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
                "failed to spawn python3 for vonmises oracle: {e}"
            );
            eprintln!("skipping vonmises oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open vonmises oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "vonmises oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping vonmises oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for vonmises oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "vonmises oracle failed: {stderr}"
        );
        eprintln!("skipping vonmises oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse vonmises oracle JSON"))
}

#[test]
fn diff_stats_vonmises() {
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

    for case in &query.points {
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        let dist = VonMises::new(case.kappa, case.loc);
        if let Some(spdf) = oracle.pdf {
            let d = (dist.pdf(case.x) - spdf).abs();
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: "pdf".into(),
                abs_diff: d,
                pass: d <= PDF_TOL,
            });
        }
        if let Some(scdf) = oracle.cdf {
            let d = (dist.cdf(case.x) - scdf).abs();
            max_overall = max_overall.max(d);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: "cdf".into(),
                abs_diff: d,
                pass: d <= CDF_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_vonmises".into(),
        category: "scipy.stats.vonmises".into(),
        case_count: diffs.len(),
        max_abs_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };

    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "vonmises {} mismatch: {} abs_diff={}",
                d.family, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.stats.vonmises conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
