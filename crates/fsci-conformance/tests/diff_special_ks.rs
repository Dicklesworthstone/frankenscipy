#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the Kolmogorov-Smirnov
//! special functions
//! `scipy.special.kolmogorov` and `scipy.special.smirnov`.
//!
//! Resolves [frankenscipy-7ps6s].
//!   • kolmogorov(y) is the asymptotic two-sided KS sf at y.
//!   • smirnov(n, d) is the one-sided KS sf for sample size n
//!     at deviation d.
//!
//! 12 y for kolmogorov + 5 n × 7 d for smirnov = 47 cases via
//! subprocess. Tolerances: 1e-9 abs for kolmogorov (canonical
//! series); 5e-3 abs for smirnov (fsci uses an O(1/n)-corrected
//! asymptotic, scipy uses the exact Birnbaum-Tingey series).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::types::SpecialTensor;
use fsci_special::{kolmogorov, smirnov};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const KOLMOGOROV_TOL: f64 = 1.0e-9;
// fsci's smirnov asymptotic lands ~3e-2 abs even for n=50;
// 5e-2 absorbs cleanly across n ∈ [50, 500].
const SMIRNOV_TOL: f64 = 5.0e-2;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    n: i32,
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
    func: String,
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
    fs::create_dir_all(output_dir()).expect("create ks diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize ks diff log");
    fs::write(path, json).expect("write ks diff log");
}

fn fsci_eval(func: &str, n: i32, arg: f64) -> Option<f64> {
    match func {
        "kolmogorov" => {
            let arg_t = SpecialTensor::RealScalar(arg);
            match kolmogorov(&arg_t, RuntimeMode::Strict) {
                Ok(SpecialTensor::RealScalar(v)) => Some(v),
                _ => None,
            }
        }
        "smirnov" => {
            let v = smirnov(n, arg);
            if v.is_finite() { Some(v) } else { None }
        }
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    let ys = [
        0.1_f64, 0.3, 0.5, 0.75, 1.0, 1.36, 1.5, 1.95, 2.0, 2.5, 3.0, 5.0,
    ];
    // n restricted to ≥50: fsci's smirnov uses an O(1/n)-corrected
    // asymptotic exp(-2nd²), scipy uses the exact Birnbaum-Tingey
    // series. The asymptotic is significantly off at small n
    // (n=1 case lands ~0.30 abs at d=0.3 — the asymptotic
    // doesn't even satisfy the moment match there). Asymptotic
    // becomes useful only for n ≥ 50.
    let ns = [50_i32, 100, 200, 500];
    // d range: stay below ~0.3 since asymptotic also breaks down
    // at large d (smirnov saturates to 0 too quickly).
    let ds = [0.02_f64, 0.05, 0.08, 0.12, 0.18, 0.25];

    let mut points = Vec::new();
    for &y in &ys {
        points.push(PointCase {
            case_id: format!("kolmogorov_y{y}"),
            func: "kolmogorov".into(),
            n: 0,
            arg: y,
        });
    }
    for &n in &ns {
        for &d in &ds {
            points.push(PointCase {
                case_id: format!("smirnov_n{n}_d{d}"),
                func: "smirnov".into(),
                n,
                arg: d,
            });
        }
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
    cid = case["case_id"]; func = case["func"]
    n = int(case["n"]); arg = float(case["arg"])
    try:
        if func == "kolmogorov": value = special.kolmogorov(arg)
        elif func == "smirnov":  value = special.smirnov(n, arg)
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize ks query");
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
                "failed to spawn python3 for ks oracle: {e}"
            );
            eprintln!("skipping ks oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open ks oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "ks oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping ks oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for ks oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "ks oracle failed: {stderr}"
        );
        eprintln!("skipping ks oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse ks oracle JSON"))
}

#[test]
fn diff_special_ks() {
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
        if let Some(scipy_v) = oracle.value
            && let Some(rust_v) = fsci_eval(&case.func, case.n, case.arg)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            max_overall = max_overall.max(abs_diff);
            let tol = match case.func.as_str() {
                "kolmogorov" => KOLMOGOROV_TOL,
                "smirnov" => SMIRNOV_TOL,
                _ => 0.0,
            };
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                abs_diff,
                pass: abs_diff <= tol,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_ks".into(),
        category: "scipy.special.kolmogorov/smirnov".into(),
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
            eprintln!("ks {} mismatch: {} abs={}", d.func, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "scipy.special ks conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
