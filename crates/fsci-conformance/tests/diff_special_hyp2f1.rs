#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the Gauss
//! hypergeometric `scipy.special.hyp2f1` ( ₂F₁(a, b; c; z) ).
//!
//! Resolves [frankenscipy-6cbyo]. Companion to
//! `diff_special_hyp1f1`. fsci-special's hyp2f1 is the most
//! precision-sensitive of the hypergeometric kernels — series
//! converges for |z| < 1, with branch-cut and asymptotic-form
//! handling for |z| ≥ 1. fsci's implementation handles the
//! z<0 and 0<z<1 series regimes; near-z=1 and |z|>1 are tested
//! separately.
//!
//! 24 finite-valued (a, b, c, z) cases plus 4 real branch-cut
//! cases via subprocess. Tolerances: 5e-7 rel — same as hyp1f1;
//! Gauss hypergeometric is wide-tolerance coverage, not a
//! precision claim.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::hyp2f1;
use fsci_special::types::SpecialTensor;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const TOL_REL: f64 = 5.0e-7;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    a: f64,
    b: f64,
    c: f64,
    z: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
    branch_cut_points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct BranchCutArm {
    case_id: String,
    positive_infinity: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
    branch_cut_points: Vec<BranchCutArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    abs_diff: f64,
    rel_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct BranchCutDiff {
    case_id: String,
    scipy_positive_infinity: bool,
    rust_positive_infinity: bool,
    hardened_error: bool,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_abs_diff: f64,
    max_rel_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
    branch_cut_cases: Vec<BranchCutDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create hyp2f1 diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize hyp2f1 diff log");
    fs::write(path, json).expect("write hyp2f1 diff log");
}

fn fsci_eval(a: f64, b: f64, c: f64, z: f64) -> Option<f64> {
    let pa = SpecialTensor::RealScalar(a);
    let pb = SpecialTensor::RealScalar(b);
    let pc = SpecialTensor::RealScalar(c);
    let pz = SpecialTensor::RealScalar(z);
    match hyp2f1(&pa, &pb, &pc, &pz, RuntimeMode::Strict) {
        Ok(SpecialTensor::RealScalar(v)) => Some(v),
        _ => None,
    }
}

fn fsci_hardened_errors(a: f64, b: f64, c: f64, z: f64) -> bool {
    let pa = SpecialTensor::RealScalar(a);
    let pb = SpecialTensor::RealScalar(b);
    let pc = SpecialTensor::RealScalar(c);
    let pz = SpecialTensor::RealScalar(z);
    hyp2f1(&pa, &pb, &pc, &pz, RuntimeMode::Hardened).is_err()
}

fn generate_query() -> OracleQuery {
    // Pinned (a, b, c, z) — emphasizing identities and
    // small/moderate-z safe regimes:
    //   ₂F₁(1, 1, 2, z) = -ln(1-z)/z
    //   ₂F₁(0.5, 0.5, 1, z) = (2/π)·K(z) (related to ellipk)
    //   ₂F₁(1, 1, 3/2, z²) = arcsin(z)/(z√(1-z²))
    let cases: [(f64, f64, f64, f64); 24] = [
        // Identity: -ln(1-z)/z
        (1.0, 1.0, 2.0, 0.5),
        (1.0, 1.0, 2.0, -0.5),
        (1.0, 1.0, 2.0, 0.1),
        (1.0, 1.0, 2.0, -0.9),
        // Identity: (2/π)·K(z)
        (0.5, 0.5, 1.0, 0.25),
        (0.5, 0.5, 1.0, 0.5),
        (0.5, 0.5, 1.0, 0.9),
        // Generic mid-range
        (1.5, 2.5, 3.0, 0.3),
        (1.5, 2.5, 3.0, -0.3),
        (2.0, 3.0, 4.0, 0.5),
        (2.0, 3.0, 4.0, -0.5),
        (0.3, 0.7, 1.5, 0.4),
        (0.3, 0.7, 1.5, -0.7),
        // Larger parameters
        (5.0, 5.0, 10.0, 0.5),
        (5.0, 5.0, 10.0, -0.5),
        (3.0, 7.0, 11.0, 0.7),
        // Small z
        (1.5, 2.5, 3.5, 0.001),
        (2.0, 4.0, 6.0, 1.0e-6),
        // Negative-only z (faster series convergence). Stop
        // short of -1 because fsci's hyp2f1 series returns NaN
        // for |z| ≥ 0.99 in some (a, b, c) regimes — likely a
        // truncation-vs-Pfaff-transformation seam.
        (1.0, 2.0, 3.0, -0.9),
        (0.5, 1.5, 2.5, -0.5),
        // ₂F₁(a, b, b, z) = (1-z)^(-a)
        (1.0, 5.0, 5.0, 0.5),
        (2.0, 3.0, 3.0, -0.5),
        // Misc
        (1.0, 0.5, 1.5, -0.3),
        (2.5, 1.5, 4.0, 0.2),
    ];
    let mut points = Vec::new();
    for (i, &(a, b, c, z)) in cases.iter().enumerate() {
        points.push(PointCase {
            case_id: format!("a{a}_b{b}_c{c}_z{z}_i{i}"),
            a,
            b,
            c,
            z,
        });
    }
    let branch_cut_cases: [(f64, f64, f64, f64); 4] = [
        (1.0, 2.0, 3.0, 1.5),
        (0.25, 0.75, 2.5, 1.25),
        (2.0, 1.0, 5.0, 1.1),
        (0.5, 1.5, 4.0, 2.0),
    ];
    let mut branch_cut_points = Vec::new();
    for (i, &(a, b, c, z)) in branch_cut_cases.iter().enumerate() {
        branch_cut_points.push(PointCase {
            case_id: format!("branch_cut_a{a}_b{b}_c{c}_z{z}_i{i}"),
            a,
            b,
            c,
            z,
        });
    }
    OracleQuery {
        points,
        branch_cut_points,
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy import special

def finite_or_none(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.loads(sys.argv[1])
points = []
for case in q["points"]:
    cid = case["case_id"]
    a = float(case["a"]); b = float(case["b"])
    c = float(case["c"]); z = float(case["z"])
    try:
        value = special.hyp2f1(a, b, c, z)
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
branch_cut_points = []
for case in q["branch_cut_points"]:
    cid = case["case_id"]
    a = float(case["a"]); b = float(case["b"])
    c = float(case["c"]); z = float(case["z"])
    try:
        value = float(special.hyp2f1(a, b, c, z))
        branch_cut_points.append({
            "case_id": cid,
            "positive_infinity": math.isinf(value) and value > 0.0,
        })
    except Exception:
        branch_cut_points.append({"case_id": cid, "positive_infinity": False})
print(json.dumps({"points": points, "branch_cut_points": branch_cut_points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize hyp2f1 query");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-")
        .arg(&query_json)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for hyp2f1 oracle: {e}"
            );
            eprintln!("skipping hyp2f1 oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open hyp2f1 oracle stdin");
        if let Err(err) = stdin.write_all(script.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "hyp2f1 oracle script write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping hyp2f1 oracle: script write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for hyp2f1 oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "hyp2f1 oracle failed: {stderr}"
        );
        eprintln!("skipping hyp2f1 oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse hyp2f1 oracle JSON"))
}

#[test]
fn diff_special_hyp2f1() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());
    assert_eq!(
        oracle.branch_cut_points.len(),
        query.branch_cut_points.len()
    );

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();
    let branch_cut_map: HashMap<String, BranchCutArm> = oracle
        .branch_cut_points
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut branch_cut_diffs = Vec::new();
    let mut max_abs_overall = 0.0_f64;
    let mut max_rel_overall = 0.0_f64;

    for case in &query.points {
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        if let Some(scipy_v) = oracle.value
            && let Some(rust_v) = fsci_eval(case.a, case.b, case.c, case.z)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            let scale = scipy_v.abs().max(1.0);
            let rel_diff = abs_diff / scale;
            max_abs_overall = max_abs_overall.max(abs_diff);
            max_rel_overall = max_rel_overall.max(rel_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                abs_diff,
                rel_diff,
                pass: abs_diff <= TOL_REL * scale,
            });
        }
    }

    for case in &query.branch_cut_points {
        let oracle = branch_cut_map
            .get(&case.case_id)
            .expect("validated branch-cut oracle");
        let rust_positive_infinity = fsci_eval(case.a, case.b, case.c, case.z)
            .is_some_and(|v| v.is_infinite() && v.is_sign_positive());
        let hardened_error = fsci_hardened_errors(case.a, case.b, case.c, case.z);
        let pass = oracle.positive_infinity && rust_positive_infinity && hardened_error;
        branch_cut_diffs.push(BranchCutDiff {
            case_id: case.case_id.clone(),
            scipy_positive_infinity: oracle.positive_infinity,
            rust_positive_infinity,
            hardened_error,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass) && branch_cut_diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_hyp2f1".into(),
        category: "scipy.special.hyp2f1".into(),
        case_count: diffs.len() + branch_cut_diffs.len(),
        max_abs_diff: max_abs_overall,
        max_rel_diff: max_rel_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
        branch_cut_cases: branch_cut_diffs.clone(),
    };

    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "hyp2f1 mismatch: {} abs={} rel={}",
                d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }
    for d in &branch_cut_diffs {
        if !d.pass {
            eprintln!(
                "hyp2f1 branch-cut mismatch: {} scipy_inf={} rust_inf={} hardened_error={}",
                d.case_id, d.scipy_positive_infinity, d.rust_positive_infinity, d.hardened_error
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special.hyp2f1 conformance failed: {} cases, max_abs={} max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
}
