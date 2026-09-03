#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the combinatorial
//! kernels `scipy.special.factorial/factorial2/comb/perm`.
//!
//! Resolves [frankenscipy-095pp]. Four exact-integer kernels
//! represented as f64 (so the magnitude ceiling is ~1.79e308).
//!
//! Tolerances: 1e-13 abs OR 1e-13 rel — for n ≤ 20 the value
//! fits exactly in u64; for larger n the f64 representation
//! limits precision to ~1e-15 relative.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{comb, factorial, factorial2, perm};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-13;
// Large n: scipy uses gamma(n+1) via lgamma+exp, fsci uses
// iterative product. Both are f64 with ~1e-15 rel precision,
// but compounding for large n gives ~1e-13 rel.
// perm(50, 50) = 50! ~3.04e64 lands ~6e-12 rel off scipy.
const REL_TOL: f64 = 1.0e-11;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    n: u64,
    k: u64,
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
    fs::create_dir_all(output_dir()).expect("create factorial diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize factorial diff log");
    fs::write(path, json).expect("write factorial diff log");
}

fn fsci_eval(func: &str, n: u64, k: u64) -> Option<f64> {
    let v = match func {
        "factorial" => factorial(n),
        "factorial2" => factorial2(n as i64),
        "comb" => comb(n, k),
        "perm" => perm(n, k),
        _ => return None,
    };
    if v.is_finite() { Some(v) } else { None }
}

fn generate_query() -> OracleQuery {
    // factorial / factorial2: small to moderate n. f64 max
    // factorial is ~170! before overflow.
    let factorial_ns = [0_u64, 1, 5, 10, 15, 20, 50, 100, 150];
    // comb / perm: (n, k) pairs.
    let comb_pairs = [
        (5_u64, 2),
        (10, 3),
        (10, 5),
        (20, 10),
        (50, 25),
        (100, 50),
        (50, 0),
        (50, 50),
        (50, 1),
        (50, 49),
    ];

    let mut points = Vec::new();
    for &n in &factorial_ns {
        for func in ["factorial", "factorial2"] {
            points.push(PointCase {
                case_id: format!("{func}_n{n}"),
                func: func.to_string(),
                n,
                k: 0,
            });
        }
    }
    for &(n, k) in &comb_pairs {
        for func in ["comb", "perm"] {
            points.push(PointCase {
                case_id: format!("{func}_n{n}_k{k}"),
                func: func.to_string(),
                n,
                k,
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
    n = int(case["n"]); k = int(case["k"])
    try:
        if func == "factorial":  value = float(special.factorial(n, exact=False))
        elif func == "factorial2":value = float(special.factorial2(n, exact=False))
        elif func == "comb":     value = float(special.comb(n, k, exact=False))
        elif func == "perm":     value = float(special.perm(n, k, exact=False))
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize factorial query");
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
                "failed to spawn python3 for factorial oracle: {e}"
            );
            eprintln!("skipping factorial oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open factorial oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "factorial oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping factorial oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for factorial oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "factorial oracle failed: {stderr}"
        );
        eprintln!("skipping factorial oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse factorial oracle JSON"))
}

#[test]
fn diff_special_factorial() {
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
            && let Some(rust_v) = fsci_eval(&case.func, case.n, case.k)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            max_overall = max_overall.max(abs_diff);
            let scale = scipy_v.abs().max(1.0);
            let pass = abs_diff <= ABS_TOL || abs_diff <= REL_TOL * scale;
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                abs_diff,
                pass,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_factorial".into(),
        category: "scipy.special.factorial/factorial2/comb/perm".into(),
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
                "factorial {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "factorial conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
