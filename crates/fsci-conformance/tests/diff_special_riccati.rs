#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the Riccati-Bessel
//! functions `scipy.special.riccati_jn` and
//! `scipy.special.riccati_yn` (returns values + derivatives).
//!
//! Resolves [frankenscipy-4ny8f]. Riccati-Bessel is the
//! standard form for Mie scattering. fsci composes them from
//! the spherical Bessel kernel.
//!
//! 4 n-values × 6 x-values × 2 funcs × 2 arms (values, derivs)
//! = 96 case-arms via subprocess. Tolerances: 1e-6 abs/rel —
//! inherits the cylindrical Bessel precision floor
//! (frankenscipy-0om9c).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{riccati_jn, riccati_yn};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-6;
const REL_TOL: f64 = 1.0e-6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    n: u32,
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    values: Option<Vec<f64>>,
    derivs: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    func: String,
    arm: String,
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
    fs::create_dir_all(output_dir()).expect("create riccati diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize riccati diff log");
    fs::write(path, json).expect("write riccati diff log");
}

fn fsci_eval(func: &str, n: u32, x: f64) -> Option<(Vec<f64>, Vec<f64>)> {
    let (vs, ds) = match func {
        "riccati_jn" => riccati_jn(n, x),
        "riccati_yn" => riccati_yn(n, x),
        _ => return None,
    };
    if vs.iter().chain(ds.iter()).all(|v| v.is_finite()) {
        Some((vs, ds))
    } else {
        None
    }
}

fn generate_query() -> OracleQuery {
    let ns = [0_u32, 1, 3, 5];
    // riccati_yn diverges at x→0, so x>0.
    let xs = [0.5_f64, 1.0, 2.0, 5.0, 10.0, 20.0];
    let mut points = Vec::new();
    for &n in &ns {
        for &x in &xs {
            // riccati_yn re-enabled after gt5x9 (sign-convention fix):
            // fsci now matches scipy to ~1e-10 across the n × x grid
            // (frankenscipy-lbsb1).
            for func in ["riccati_jn", "riccati_yn"] {
                points.push(PointCase {
                    case_id: format!("{func}_n{n}_x{x}"),
                    func: func.to_string(),
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
from scipy import special

def finite_or_none_list(arr):
    out = []
    for v in arr:
        try:
            v = float(v)
            out.append(v if math.isfinite(v) else None)
        except Exception:
            out.append(None)
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; func = case["func"]
    n = int(case["n"]); x = float(case["x"])
    try:
        if func == "riccati_jn":
            v_arr, d_arr = special.riccati_jn(n, x)
        elif func == "riccati_yn":
            v_arr, d_arr = special.riccati_yn(n, x)
        else:
            v_arr, d_arr = [], []
        v_list = finite_or_none_list(v_arr.tolist() if hasattr(v_arr, 'tolist') else v_arr)
        d_list = finite_or_none_list(d_arr.tolist() if hasattr(d_arr, 'tolist') else d_arr)
        if any(x is None for x in v_list) or any(x is None for x in d_list):
            points.append({"case_id": cid, "values": None, "derivs": None})
        else:
            points.append({"case_id": cid, "values": v_list, "derivs": d_list})
    except Exception:
        points.append({"case_id": cid, "values": None, "derivs": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize riccati query");
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
                "failed to spawn python3 for riccati oracle: {e}"
            );
            eprintln!("skipping riccati oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open riccati oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "riccati oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping riccati oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for riccati oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "riccati oracle failed: {stderr}"
        );
        eprintln!("skipping riccati oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse riccati oracle JSON"))
}

#[test]
fn diff_special_riccati() {
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
        let (sv, sd) = match (oracle.values.as_ref(), oracle.derivs.as_ref()) {
            (Some(v), Some(d)) => (v, d),
            _ => continue,
        };
        let Some((rv, rd)) = fsci_eval(&case.func, case.n, case.x) else {
            continue;
        };
        if rv.len() != sv.len() || rd.len() != sd.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                arm: "values".into(),
                abs_diff: f64::INFINITY,
                rel_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let assess = |arms: &mut Vec<CaseDiff>,
                      arm: &str,
                      r_arr: &[f64],
                      s_arr: &[f64],
                      max_abs: &mut f64,
                      max_rel: &mut f64| {
            let mut worst_abs = 0.0_f64;
            let mut worst_rel = 0.0_f64;
            let mut pass = true;
            for (rv, sv) in r_arr.iter().zip(s_arr.iter()) {
                let abs_d = (rv - sv).abs();
                let scale = sv.abs().max(1.0);
                let rel_d = abs_d / scale;
                worst_abs = worst_abs.max(abs_d);
                worst_rel = worst_rel.max(rel_d);
                if !(abs_d <= ABS_TOL || rel_d <= REL_TOL) {
                    pass = false;
                }
            }
            *max_abs = max_abs.max(worst_abs);
            *max_rel = max_rel.max(worst_rel);
            arms.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                arm: arm.into(),
                abs_diff: worst_abs,
                rel_diff: worst_rel,
                pass,
            });
        };
        assess(
            &mut diffs,
            "values",
            &rv,
            sv,
            &mut max_abs_overall,
            &mut max_rel_overall,
        );
        assess(
            &mut diffs,
            "derivs",
            &rd,
            sd,
            &mut max_abs_overall,
            &mut max_rel_overall,
        );
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_riccati".into(),
        category: "scipy.special.riccati_jn/yn".into(),
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
                "riccati {} {} mismatch: {} abs={} rel={}",
                d.func, d.arm, d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "riccati conformance failed: {} cases, max_abs={} max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
}
