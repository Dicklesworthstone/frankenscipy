#![forbid(unsafe_code)]
//! Live SciPy differential coverage for non-integer-order
//! Bessel `scipy.special.jv` (J_v) and `scipy.special.yv` (Y_v).
//!
//! Resolves [frankenscipy-ucedr]. Companion to
//! `diff_special_bessel` (J_0, J_1, Y_0, Y_1) and
//! `diff_special_bessel_modified` (I_0, I_1, K_0, K_1).
//! Both x>0 and integer-collision orders are tested separately;
//! the harness pins half-integer and irrational orders.
//!
//! 6 v-values × 7 x-values × 2 functions = 84 grid cases via
//! subprocess, at 1e-6 abs / rel. A further 14 cases at 5 ≤ x < 20
//! and at integer-adjacent orders (frankenscipy-5v5lj) are gated
//! purely relatively at `REGIME_REL_TOL`: there fsci's jv summed a
//! power series that cancels like I_v(x)/|J_v(x)| and its yv divided
//! that by sin(vπ), each 2e-11..3e-8 relative off SciPy, which is
//! within 3e-14 of mpmath on every one of them.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_runtime::RuntimeMode;
use fsci_special::types::SpecialTensor;
use fsci_special::{jv, yv};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-6;
const REL_TOL: f64 = 1.0e-6;
/// |fsci − scipy| ≤ REGIME_REL_TOL·|scipy| for the frankenscipy-5v5lj cases.
const REGIME_REL_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
/// One ledger arm per SciPy function compared.
const ARMS: [&str; 2] = ["jv", "yv"];

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    v: f64,
    x: f64,
    /// Gated at `REGIME_REL_TOL` (relative) rather than the grid's abs/rel pair.
    regime: bool,
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
    fs::create_dir_all(output_dir()).expect("create jv/yv diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize jv/yv diff log");
    fs::write(path, json).expect("write jv/yv diff log");
}

fn fsci_eval(func: &str, v: f64, x: f64) -> Option<f64> {
    let pv = SpecialTensor::RealScalar(v);
    let px = SpecialTensor::RealScalar(x);
    let result = match func {
        "jv" => jv(&pv, &px, RuntimeMode::Strict),
        "yv" => yv(&pv, &px, RuntimeMode::Strict),
        _ => return None,
    };
    match result {
        Ok(SpecialTensor::RealScalar(value)) => Some(value),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    // Non-integer order coverage expanded after 6avjb (jv_series
    // sign + Γ recurrence fix). fsci jv/yv now return finite values
    // matching scipy across v ∈ {0.5, 0.7, 1.3, 1.5, 1.7, 2.5} on
    // the x grid below. The asymptotic-seam at z ≥ 30 still has
    // ~7e-4 drift (separate follow-up); the harness keeps x ≤ 15
    // where the agreement is within the 1e-6 tolerance.
    let vs = [0.5_f64, 0.7, 1.3, 1.5, 1.7, 2.5];
    let xs = [0.5_f64, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0];
    let mut points = Vec::new();
    for &v in &vs {
        for &x in &xs {
            for func in ["jv", "yv"] {
                points.push(PointCase {
                    case_id: format!("{func}_v{v}_x{x}"),
                    func: func.to_string(),
                    v,
                    x,
                    regime: false,
                });
            }
        }
    }
    // frankenscipy-5v5lj: points of the perf_special_vs_scipy fixture away from zeros
    // (|f| ≥ 0.3·√(J² + Y²)). Old fsci vs mpmath on each: jv 2.2e-11, 7.5e-11, 2.0e-10,
    // 2.2e-10, 1.6e-10; yv 2.1e-10, 1.0e-8, 3.4e-10, 1.1e-10, 3.9e-10, 1.2e-8, 9.3e-11,
    // 2.7e-8, 1.6e-9.
    let regime: [(&str, f64, f64); 14] = [
        ("jv", 0.186555440333679, 11.696838909483272),
        ("jv", 3.0677140968577095, 12.793734118797643),
        ("jv", 1.7010132969601093, 13.786288141135577),
        ("jv", -2.4191175426473723, 13.732172803481589),
        ("jv", -1.065722302833091, 12.892622322133032),
        ("yv", 3.999796300611098, 2.735650793047621),
        ("yv", 7.0000713997858, 6.317146548560355),
        ("yv", 9.999406001781995, 9.87902486292541),
        ("yv", 2.980564658306025, 10.975920072239782),
        ("yv", 3.1207581377255873, 13.900157799526601),
        ("yv", 2.999551001346996, 12.83214300357099),
        ("yv", -2.9994934015197954, 5.730020809937571),
        ("yv", -1.0006794979615066, 13.789424631726105),
        ("yv", -3.999717100848697, 11.43905768282695),
    ];
    for (func, v, x) in regime {
        points.push(PointCase {
            case_id: format!("{func}_v{v}_x{x}"),
            func: func.to_string(),
            v,
            x,
            regime: true,
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
    func = case["func"]; v = float(case["v"]); x = float(case["x"])
    try:
        if func == "jv":   value = special.jv(v, x)
        elif func == "yv": value = special.yv(v, x)
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize jv/yv query");
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
                "failed to spawn python3 for jv/yv oracle: {e}"
            );
            eprintln!("skipping jv/yv oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open jv/yv oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "jv/yv oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping jv/yv oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for jv/yv oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "jv/yv oracle failed: {stderr}"
        );
        eprintln!("skipping jv/yv oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse jv/yv oracle JSON"))
}

#[test]
fn diff_special_bessel_jv_yv() {
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
    let mut ledger = CompareLedger::new("diff_special_bessel_jv_yv", &ARMS);

    for case in &query.points {
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        let arm = case.func.as_str();
        let Some((scipy_v, rust_v)) = ledger.pair(
            arm,
            &case.case_id,
            oracle.value,
            fsci_eval(&case.func, case.v, case.x),
        ) else {
            continue;
        };
        let abs_diff = (rust_v - scipy_v).abs();
        let rel_diff = if case.regime || scipy_v.abs() > 1.0 {
            abs_diff / scipy_v.abs()
        } else {
            abs_diff
        };
        max_abs_overall = max_abs_overall.max(abs_diff);
        max_rel_overall = max_rel_overall.max(rel_diff);
        let pass = if case.regime {
            rel_diff <= REGIME_REL_TOL
        } else if scipy_v.abs() > 1.0 {
            rel_diff <= REL_TOL
        } else {
            abs_diff <= ABS_TOL
        };
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
        test_id: "diff_special_bessel_jv_yv".into(),
        category: "scipy.special.jv/yv".into(),
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
                "jv/yv {} mismatch: {} abs={} rel={}",
                d.func, d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special jv/yv conformance failed: {} cases, max_abs={} max_rel={}",
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
