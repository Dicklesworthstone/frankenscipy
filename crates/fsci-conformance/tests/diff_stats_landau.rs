#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.stats.landau` (frankenscipy-1ksfv.16).
//!
//! SciPy 1.17.1 computes the Landau law with Boost.Math's `landau_distribution`; fsci's
//! `Landau` ports Boost's double-precision branches. The bead's grid: x over [−5, 1e4] for
//! pdf/cdf/sf (rel 1e-13) and probabilities from 1e-300 to 1 − 1e-9 for ppf/isf (rel 1e-12),
//! at the standard law and two (loc, scale) pairs. Points are placed in every interval of
//! Boost's branch chains, including the far left tail where the density underflows to 0, which
//! must then match exactly. Non-finite inputs and p ∈ {0, 1} are pinned by the fsci-stats unit
//! tests; JSON cannot carry the infinities they return.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_stats::{ContinuousDistribution, Landau};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const PDF_REL_TOL: f64 = 1.0e-13;
const CDF_REL_TOL: f64 = 1.0e-13;
const PPF_REL_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const ARMS: [&str; 5] = ["pdf", "cdf", "sf", "ppf", "isf"];

/// (loc, scale) pairs: the standard law and two shifted/scaled ones.
const PARAMS: [(f64, f64); 3] = [(0.0, 1.0), (1.5, 2.0), (-3.0, 0.5)];

/// Standardized points z: every interval of Boost's pdf/cdf chains, x < 0 down to the zero
/// tail and x ≥ 0 out to the 2/(πx²) tail, within the bead's [−5, 1e4] (after loc/scale).
const Z: [f64; 24] = [
    -5.0, -4.6, -4.2, -3.5, -2.8, -2.2, -1.5, -0.9, -0.4, -0.05, 0.0, 0.3, 0.7, 1.2, 1.8, 3.0, 5.5,
    11.0, 25.0, 50.0, 120.0, 700.0, 3000.0, 9000.0,
];

/// Probabilities for ppf and isf, both tails and the centre.
const P: [f64; 16] = [
    1e-300,
    1e-100,
    1e-40,
    1e-20,
    1e-8,
    1e-4,
    0.01,
    0.1,
    0.3,
    0.365,
    0.5,
    0.7,
    0.9,
    0.99,
    0.999_999,
    0.999_999_999,
];

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    arm: String,
    loc: f64,
    scale: f64,
    /// x for pdf/cdf/sf, p for ppf/isf.
    arg: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    value: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create landau diff output dir");
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize landau diff log");
    fs::write(path, json).expect("write landau diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    for &(loc, scale) in &PARAMS {
        for arm in ["pdf", "cdf", "sf"] {
            for &z in &Z {
                points.push(PointCase {
                    case_id: format!("{arm}_l{loc}_s{scale}_z{z}"),
                    arm: arm.into(),
                    loc,
                    scale,
                    arg: z.mul_add(scale, loc),
                });
            }
        }
        for arm in ["ppf", "isf"] {
            for &p in &P {
                points.push(PointCase {
                    case_id: format!("{arm}_l{loc}_s{scale}_p{p}"),
                    arm: arm.into(),
                    loc,
                    scale,
                    arg: p,
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
from scipy.stats import landau

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    d = landau(loc=float(case["loc"]), scale=float(case["scale"]))
    try:
        v = float(getattr(d, case["arm"])(float(case["arg"])))
        points.append({"case_id": cid, "value": v if math.isfinite(v) else None})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize landau query");
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
                "failed to spawn python3 for landau oracle: {e}"
            );
            eprintln!("skipping landau oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open landau oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "landau oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping landau oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for landau oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "landau oracle failed: {stderr}"
        );
        eprintln!("skipping landau oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse landau oracle JSON"))
}

/// Relative difference; a zero SciPy value (the underflowed left tail) must be matched exactly.
fn rel_diff(fsci: f64, scipy: f64) -> f64 {
    if scipy == 0.0 {
        if fsci == 0.0 { 0.0 } else { f64::INFINITY }
    } else {
        ((fsci - scipy) / scipy).abs()
    }
}

#[test]
fn diff_stats_landau() {
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
    let mut ledger = CompareLedger::new("diff_stats_landau", &ARMS);

    for case in &query.points {
        let scipy = pmap.get(&case.case_id).expect("validated oracle").value;
        let dist = Landau::new(case.loc, case.scale);
        // An arm outside ARMS is skipped here and then fails the per-arm count below.
        let Some((fsci, tol)) = (match case.arm.as_str() {
            "pdf" => Some((dist.pdf(case.arg), PDF_REL_TOL)),
            "cdf" => Some((dist.cdf(case.arg), CDF_REL_TOL)),
            "sf" => Some((dist.sf(case.arg), CDF_REL_TOL)),
            "ppf" => Some((dist.ppf(case.arg), PPF_REL_TOL)),
            "isf" => Some((dist.isf(case.arg), PPF_REL_TOL)),
            _ => None,
        }) else {
            continue;
        };
        let Some((s, f)) = ledger.pair(&case.arm, &case.case_id, scipy, Some(fsci)) else {
            continue;
        };
        let d = rel_diff(f, s);
        max_overall = max_overall.max(d);
        ledger.compared(&case.arm, &case.case_id, d <= tol);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            arm: case.arm.clone(),
            rel_diff: d,
            pass: d <= tol,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_stats_landau".into(),
        category: "scipy.stats.landau".into(),
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
                "landau {} mismatch: {} rel_diff={}",
                d.arm, d.case_id, d.rel_diff
            );
        }
    }
    assert!(
        all_pass,
        "scipy.stats.landau conformance failed: {} cases, max_rel_diff={}",
        diffs.len(),
        max_overall
    );
    let counts = ledger.finish(Z.len().min(P.len()));
    for arm in ARMS {
        let cases = query.points.iter().filter(|c| c.arm == arm).count();
        assert_eq!(
            counts[arm].compared_cases, cases,
            "arm `{arm}` must compare all {cases} of its cases"
        );
    }
}
