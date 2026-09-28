#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the modified cylindrical
//! Bessel family I_0, I_1, K_0, K_1 and their exponentially scaled
//! forms (`scipy.special.i0/i1/k0/k1/i0e/i1e/k0e/k1e`).
//!
//! Resolves [frankenscipy-k4hhh]; the scaled arms are frankenscipy-wyn06.
//! Companion to `diff_special_bessel` (J_n / Y_n). 11 x-values × 2
//! (i0, i1), 11 x-values × 2 (k0, k1, x>0 only), and 9 x-values × 4
//! scaled plus x = -3.5 for i0e/i1e = 82 cases via subprocess.
//!
//! Tolerance: 1e-15 relative for every arm; all eight are
//! bit-identical Cephes ports.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_runtime::RuntimeMode;
use fsci_special::types::SpecialTensor;
use fsci_special::{i0, i0e, i1, i1e, k0, k0e, k1, k1e};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
// All eight are Cephes' Chebyshev kernels, bit-identical to SciPy. This is a TRUE relative
// tolerance, a few ulp. It was 1e-7 absolute (1e-9 relative above 1), which could not see the
// last-bit differences of the old ln(x) - ln 2 and exp·(cheb/sqrt) forms (frankenscipy-wyn06).
const REL_TOL: f64 = 1.0e-15;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
/// One ledger arm per SciPy function compared.
const ARMS: [&str; 8] = ["i0", "i1", "k0", "k1", "i0e", "i1e", "k0e", "k1e"];

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
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
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    func: String,
    abs_diff: f64,
    rel_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    max_abs_diff: f64,
    max_rel_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create modified-bessel diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize modified-bessel diff log");
    fs::write(path, json).expect("write modified-bessel diff log");
}

fn fsci_eval(func: &str, x: f64) -> Option<f64> {
    let arg = SpecialTensor::RealScalar(x);
    let result = match func {
        "i0" => i0(&arg, RuntimeMode::Strict),
        "i1" => i1(&arg, RuntimeMode::Strict),
        "k0" => k0(&arg, RuntimeMode::Strict),
        "k1" => k1(&arg, RuntimeMode::Strict),
        "i0e" => i0e(&arg, RuntimeMode::Strict),
        "i1e" => i1e(&arg, RuntimeMode::Strict),
        "k0e" => k0e(&arg, RuntimeMode::Strict),
        "k1e" => k1e(&arg, RuntimeMode::Strict),
        _ => return None,
    };
    match result {
        Ok(SpecialTensor::RealScalar(v)) => Some(v),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    // I_n is even/odd-order parity; K_n is x>0. Walk small,
    // moderate, and large arguments. K_n diverges as x→0, so
    // the smallest K_n probe stays at x=0.01.
    let xs_in = [
        1.0e-12_f64,
        0.001,
        0.01,
        0.1,
        0.5,
        1.0,
        2.0,
        3.5,
        5.0,
        7.5,
        10.0,
    ];
    let xs_kn = [
        0.01_f64, 0.05, 0.1, 0.5, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0, 20.0,
    ];
    let mut points = Vec::new();
    for &x in &xs_in {
        for func in ["i0", "i1"] {
            points.push(PointCase {
                case_id: format!("{func}_x{x}"),
                func: func.to_string(),
                x,
            });
        }
    }
    for &x in &xs_kn {
        for func in ["k0", "k1"] {
            points.push(PointCase {
                case_id: format!("{func}_x{x}"),
                func: func.to_string(),
                x,
            });
        }
    }
    // The scaled forms across both kernel switches (2 and 8) and out to 1000, where
    // I·exp(-x) and K·exp(x) built from the unscaled values overflow or underflow.
    let xs_scaled = [0.01_f64, 0.5, 1.9, 2.1, 7.9, 8.5, 50.0, 700.0, 1000.0];
    for &x in &xs_scaled {
        for func in ["i0e", "i1e", "k0e", "k1e"] {
            points.push(PointCase {
                case_id: format!("{func}_x{x}"),
                func: func.to_string(),
                x,
            });
        }
    }
    for func in ["i0e", "i1e"] {
        points.push(PointCase {
            case_id: format!("{func}_x-3.5"),
            func: func.to_string(),
            x: -3.5,
        });
    }
    OracleQuery { points }
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

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    func = case["func"]; x = float(case["x"])
    try:
        if func == "i0":   value = special.i0(x)
        elif func == "i1": value = special.i1(x)
        elif func == "k0": value = special.k0(x)
        elif func == "k1": value = special.k1(x)
        elif func == "i0e": value = special.i0e(x)
        elif func == "i1e": value = special.i1e(x)
        elif func == "k0e": value = special.k0e(x)
        elif func == "k1e": value = special.k1e(x)
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize modified-bessel query");
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
                "failed to spawn python3 for modified-bessel oracle: {e}"
            );
            eprintln!("skipping modified-bessel oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open modified-bessel oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "modified-bessel oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping modified-bessel oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for modified-bessel oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "modified-bessel oracle failed: {stderr}"
        );
        eprintln!("skipping modified-bessel oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse modified-bessel oracle JSON"))
}

#[test]
fn diff_special_bessel_modified() {
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
    let mut max_abs_overall = 0.0_f64;
    let mut max_rel_overall = 0.0_f64;
    let mut ledger = CompareLedger::new("diff_special_bessel_modified", &ARMS);

    for case in &query.points {
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        let arm = case.func.as_str();
        let Some((scipy_v, rust_v)) = ledger.pair(
            arm,
            &case.case_id,
            oracle.value,
            fsci_eval(&case.func, case.x),
        ) else {
            continue;
        };
        let abs_diff = (rust_v - scipy_v).abs();
        let rel_diff = if scipy_v.abs() > 1.0 {
            abs_diff / scipy_v.abs()
        } else {
            abs_diff
        };
        max_abs_overall = max_abs_overall.max(abs_diff);
        max_rel_overall = max_rel_overall.max(rel_diff);
        let pass = abs_diff <= REL_TOL * scipy_v.abs();
        ledger.compared(arm, &case.case_id, pass);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            func: case.func.clone(),
            abs_diff,
            rel_diff,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_bessel_modified".into(),
        category: "scipy.special.i0/i1/k0/k1".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        max_abs_diff: max_abs_overall,
        max_rel_diff: max_rel_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };

    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "modified-bessel {} mismatch: {} abs={} rel={}",
                d.func, d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special modified-bessel conformance failed: {} cases, max_abs={} max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
    let min_per_arm = ARMS
        .iter()
        .map(|arm| query.points.iter().filter(|c| c.func == *arm).count())
        .min()
        .expect("ARMS is non-empty");
    ledger.finish(min_per_arm);
}
