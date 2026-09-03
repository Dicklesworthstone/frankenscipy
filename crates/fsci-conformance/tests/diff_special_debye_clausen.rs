#![forbid(unsafe_code)]
//! Live formula-derived parity for fsci_special::{debye, clausen}.
//!
//! Resolves [frankenscipy-2o6hw].
//!
//! debye(n, x) = (n/x^n) ∫_0^x t^n / (e^t − 1) dt
//! clausen(θ)  = Σ_{k=1}^∞ sin(kθ) / k^2
//!
//! References computed via scipy.integrate.quad (debye) and a
//! truncated series with N=100000 terms (clausen). Tolerance: 5e-6
//! abs (Simpson-rule debye + finite Clausen series).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{clausen, debye};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 5.0e-6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct DebyeCase {
    case_id: String,
    n: usize,
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ClausenCase {
    case_id: String,
    theta: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    debye: Vec<DebyeCase>,
    clausen: Vec<ClausenCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct ArmScalar {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    debye: Vec<ArmScalar>,
    clausen: Vec<ArmScalar>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
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
    fs::create_dir_all(output_dir()).expect("create debye_clausen diff dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

fn generate_query() -> OracleQuery {
    let xs = [0.5_f64, 1.0, 2.0, 5.0, 10.0];
    let mut debye_cases = Vec::new();
    for n in 1..=4 {
        for &x in &xs {
            debye_cases.push(DebyeCase {
                case_id: format!("n{n}_x{x}").replace('.', "p"),
                n,
                x,
            });
        }
    }

    // Clausen function: 0 at multiples of π, max around θ ≈ π/3.
    use std::f64::consts::PI;
    let thetas: &[f64] = &[
        0.1,
        PI / 6.0,
        PI / 4.0,
        PI / 3.0,
        PI / 2.0,
        2.0 * PI / 3.0,
        3.0 * PI / 4.0,
        5.0 * PI / 6.0,
        -PI / 3.0,
        -PI / 2.0,
        1.5,
        2.5,
    ];
    let clausen_cases: Vec<ClausenCase> = thetas
        .iter()
        .enumerate()
        .map(|(i, &th)| ClausenCase {
            case_id: format!("c{i:02}"),
            theta: th,
        })
        .collect();

    OracleQuery {
        debye: debye_cases,
        clausen: clausen_cases,
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy.integrate import quad

def debye(n, x):
    if x == 0.0:
        return 1.0
    if x < 0.0:
        return float("nan")
    def integrand(t):
        if t < 1e-15:
            return t**(n-1)
        return t**n / (math.exp(t) - 1.0)
    val, _ = quad(integrand, 0.0, x, epsabs=1e-12, epsrel=1e-12, limit=200)
    return (n / x**n) * val

def clausen(theta):
    # series sum to N=200000 (good to ~1e-10 for moderate theta)
    s = 0.0
    for k in range(1, 200000):
        s += math.sin(k * theta) / (k * k)
    return s

q = json.load(sys.stdin)

debye_out = []
for case in q["debye"]:
    cid = case["case_id"]
    n = int(case["n"]); x = float(case["x"])
    try:
        v = debye(n, x)
        if math.isfinite(v):
            debye_out.append({"case_id": cid, "value": float(v)})
        else:
            debye_out.append({"case_id": cid, "value": None})
    except Exception as e:
        sys.stderr.write(f"debye {cid}: {e}\n")
        debye_out.append({"case_id": cid, "value": None})

clausen_out = []
for case in q["clausen"]:
    cid = case["case_id"]
    th = float(case["theta"])
    try:
        v = clausen(th)
        if math.isfinite(v):
            clausen_out.append({"case_id": cid, "value": float(v)})
        else:
            clausen_out.append({"case_id": cid, "value": None})
    except Exception as e:
        sys.stderr.write(f"clausen {cid}: {e}\n")
        clausen_out.append({"case_id": cid, "value": None})

print(json.dumps({"debye": debye_out, "clausen": clausen_out}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize query");
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
                "failed to spawn python3 for debye/clausen oracle: {e}"
            );
            eprintln!("skipping debye/clausen oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "debye/clausen oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping debye/clausen oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for debye/clausen oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "debye/clausen oracle failed: {stderr}"
        );
        eprintln!("skipping debye/clausen oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse debye/clausen oracle JSON"))
}

#[test]
fn diff_special_debye_clausen() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let debye_map: HashMap<String, ArmScalar> = oracle
        .debye
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let clausen_map: HashMap<String, ArmScalar> = oracle
        .clausen
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.debye {
        let Some(arm) = debye_map.get(&case.case_id) else {
            continue;
        };
        let Some(expected) = arm.value else {
            continue;
        };
        let actual = debye(case.n, case.x);
        let abs_d = (actual - expected).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "debye".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    for case in &query.clausen {
        let Some(arm) = clausen_map.get(&case.case_id) else {
            continue;
        };
        let Some(expected) = arm.value else {
            continue;
        };
        let actual = clausen(case.theta);
        let abs_d = (actual - expected).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "clausen".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_debye_clausen".into(),
        category: "fsci_special::debye + clausen vs scipy.integrate / series".into(),
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
            eprintln!("{} mismatch: {} abs_diff={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "debye/clausen conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
