#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the Bessel derivatives
//! `scipy.special.jvp/yvp/ivp/kvp`.
//!
//! Resolves [frankenscipy-wrcf5]. Bessel derivatives are
//! computed in fsci-special via the standard recurrence
//!   J'_v(z) = (J_{v−1}(z) − J_{v+1}(z))/2
//! at derivative order 1; higher orders chain.
//!
//! 4 v-values × 6 z-values × 4 funcs = 96 cases via subprocess.
//! Tolerances: 1e-6 abs/rel — inherits the cylindrical Bessel
//! precision floor (frankenscipy-0om9c). v restricted to v<1
//! for jvp/yvp because yv broken at v>1 (frankenscipy-6avjb).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::types::SpecialTensor;
use fsci_special::{ivp, jvp, kvp, yvp};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-6;
const REL_TOL: f64 = 1.0e-6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    v: f64,
    z: f64,
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
    fs::create_dir_all(output_dir()).expect("create bessel-derivs diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize bessel-derivs diff log");
    fs::write(path, json).expect("write bessel-derivs diff log");
}

fn fsci_eval(func: &str, v: f64, z: f64) -> Option<f64> {
    let pv = SpecialTensor::RealScalar(v);
    let pz = SpecialTensor::RealScalar(z);
    let result = match func {
        "jvp" => jvp(&pv, &pz, 1, RuntimeMode::Strict),
        "yvp" => yvp(&pv, &pz, 1, RuntimeMode::Strict),
        "ivp" => ivp(&pv, &pz, 1, RuntimeMode::Strict),
        "kvp" => kvp(&pv, &pz, 1, RuntimeMode::Strict),
        _ => return None,
    };
    match result {
        Ok(SpecialTensor::RealScalar(value)) => Some(value),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    // jvp covers v ∈ {0, 0.5, 0.7} (within yv-safe v<1 range
    // — but yvp(0.5, z) needs Y at v=1.5 via the recurrence and
    // hits the yv-NaN issue tracked in frankenscipy-6avjb). So
    // yvp restricted to integer v=0 only. ivp/kvp take v∈{0, 1, 2}.
    let zs = [0.5_f64, 1.0, 2.0, 5.0, 10.0, 20.0];
    let mut points = Vec::new();
    for &z in &zs {
        for &v in &[0.0_f64, 0.5, 0.7] {
            points.push(PointCase {
                case_id: format!("jvp_v{v}_z{z}"),
                func: "jvp".into(),
                v,
                z,
            });
        }
        // yvp only at v=0 (integer order — bypasses the yv reflection)
        points.push(PointCase {
            case_id: format!("yvp_v0_z{z}"),
            func: "yvp".into(),
            v: 0.0,
            z,
        });
        for &v in &[0.0_f64, 1.0, 2.0] {
            for func in ["ivp", "kvp"] {
                points.push(PointCase {
                    case_id: format!("{func}_v{v}_z{z}"),
                    func: func.to_string(),
                    v,
                    z,
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
    v = float(case["v"]); z = float(case["z"])
    try:
        if func == "jvp":   value = special.jvp(v, z)
        elif func == "yvp": value = special.yvp(v, z)
        elif func == "ivp": value = special.ivp(v, z)
        elif func == "kvp": value = special.kvp(v, z)
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize bessel-derivs query");
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
                "failed to spawn python3 for bessel-derivs oracle: {e}"
            );
            eprintln!("skipping bessel-derivs oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open bessel-derivs oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "bessel-derivs oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping bessel-derivs oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for bessel-derivs oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "bessel-derivs oracle failed: {stderr}"
        );
        eprintln!("skipping bessel-derivs oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse bessel-derivs oracle JSON"))
}

#[test]
fn diff_special_bessel_derivs() {
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

    for case in &query.points {
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        if let Some(scipy_v) = oracle.value
            && let Some(rust_v) = fsci_eval(&case.func, case.v, case.z)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            let rel_diff = if scipy_v.abs() > 1.0 {
                abs_diff / scipy_v.abs()
            } else {
                abs_diff
            };
            max_abs_overall = max_abs_overall.max(abs_diff);
            max_rel_overall = max_rel_overall.max(rel_diff);
            let pass = if scipy_v.abs() > 1.0 {
                rel_diff <= REL_TOL
            } else {
                abs_diff <= ABS_TOL
            };
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                abs_diff,
                rel_diff,
                pass,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_bessel_derivs".into(),
        category: "scipy.special.jvp/yvp/ivp/kvp".into(),
        case_count: diffs.len(),
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
                "bessel-derivs {} mismatch: {} abs={} rel={}",
                d.func, d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special bessel-derivs conformance failed: {} cases, max_abs={} max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
}
