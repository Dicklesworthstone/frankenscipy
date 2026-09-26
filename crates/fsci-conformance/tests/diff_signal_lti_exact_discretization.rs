#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `fsci_signal::Lti::{step, impulse, lsim}`
//! (frankenscipy-1ksfv.15): the exact matrix-exponential discretization against
//! `scipy.signal.step` / `impulse` / `lsim` on first- to third-order SISO systems, a stiff
//! system sampled far outside an explicit integrator's stability region, an undamped
//! oscillator, feedthrough D ≠ 0, `interp` both ways, an initial state with `t[0] > 0`, and a
//! zero-input free response.
//!
//! Per case every output sample must agree to `OUT_ABS_TOL + OUT_REL_TOL·|y_scipy|`. Every
//! case must be compared: a SciPy failure or an fsci error is a FAILED case (olv0j.1).

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_signal::Lti;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-011";
const OUT_ABS_TOL: f64 = 1.0e-11;
const OUT_REL_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String,
    num: Vec<f64>,
    den: Vec<f64>,
    t: Vec<f64>,
    u: Vec<f64>,
    x0: Option<Vec<f64>>,
    interp: bool,
}

fn grid(t0: f64, dt: f64, n: usize) -> Vec<f64> {
    (0..n).map(|i| t0 + i as f64 * dt).collect()
}

fn case(id: &str, op: &str, num: &[f64], den: &[f64], t: Vec<f64>) -> Case {
    Case {
        case_id: id.to_string(),
        op: op.to_string(),
        num: num.to_vec(),
        den: den.to_vec(),
        u: Vec::new(),
        t,
        x0: None,
        interp: true,
    }
}

fn lsim_case(
    id: &str,
    num: &[f64],
    den: &[f64],
    t: Vec<f64>,
    u: Vec<f64>,
    x0: Option<Vec<f64>>,
    interp: bool,
) -> Case {
    Case {
        u,
        x0,
        interp,
        ..case(id, "lsim", num, den, t)
    }
}

fn cases() -> Vec<Case> {
    let t10 = grid(0.0, 0.05, 201);
    let sin_u: Vec<f64> = t10.iter().map(|v| (1.3 * v).sin()).collect();
    let bumpy: Vec<f64> = t10
        .iter()
        .map(|v| (0.7 * v).cos() + 0.5 * (3.1 * v).sin())
        .collect();
    let t_late = grid(0.5, 0.2, 11);
    let late_u: Vec<f64> = t_late.iter().map(|v| v.sin()).collect();
    vec![
        case(
            "step_first_order",
            "step",
            &[1.0],
            &[1.0, 1.0],
            grid(0.0, 0.1, 71),
        ),
        case(
            "step_underdamped",
            "step",
            &[1.0],
            &[1.0, 0.4, 1.0],
            t10.clone(),
        ),
        case(
            "step_stiff",
            "step",
            &[1e4],
            &[1.0, 10001.0, 1e4],
            grid(0.0, 0.01, 500),
        ),
        case(
            "impulse_undamped_oscillator",
            "impulse",
            &[1.0],
            &[1.0, 0.0, 1.0],
            t10.clone(),
        ),
        case(
            "step_feedthrough",
            "step",
            &[1.0, 2.0],
            &[1.0, 1.0],
            grid(0.0, 0.1, 31),
        ),
        case(
            "impulse_feedthrough",
            "impulse",
            &[1.0, 2.0],
            &[1.0, 1.0],
            grid(0.0, 0.1, 31),
        ),
        case(
            "impulse_third_order",
            "impulse",
            &[1.0, 2.0, 3.0],
            &[1.0, 4.0, 5.0, 2.0],
            t10.clone(),
        ),
        lsim_case(
            "lsim_sin_interp",
            &[1.0],
            &[1.0, 0.4, 1.0],
            t10.clone(),
            sin_u.clone(),
            None,
            true,
        ),
        lsim_case(
            "lsim_sin_zoh",
            &[1.0],
            &[1.0, 0.4, 1.0],
            t10.clone(),
            sin_u,
            None,
            false,
        ),
        lsim_case(
            "lsim_third_order_interp",
            &[1.0, 2.0, 3.0],
            &[1.0, 4.0, 5.0, 2.0],
            t10.clone(),
            bumpy,
            None,
            true,
        ),
        lsim_case(
            "lsim_x0_late_start",
            &[1.0],
            &[1.0, 3.0, 2.0],
            t_late.clone(),
            late_u,
            Some(vec![0.3, -0.2]),
            true,
        ),
        lsim_case(
            "lsim_free_response",
            &[1.0],
            &[1.0, 0.4, 1.0],
            t10.clone(),
            vec![0.0; t10.len()],
            Some(vec![1.0, -0.5]),
            true,
        ),
    ]
}

#[derive(Debug, Clone, Deserialize)]
struct OracleArm {
    case_id: String,
    y: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    max_abs_diff: f64,
    pass: bool,
    reason: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    pass: bool,
    timestamp_ms: u128,
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

fn scipy_oracle_or_skip(query: &[Case]) -> Option<Vec<OracleArm>> {
    let script = r#"
import json, sys, warnings
import numpy as np
from scipy import signal

out = []
for case in json.load(sys.stdin):
    arm = {"case_id": case["case_id"], "y": None}
    try:
        system = (case["num"], case["den"])
        t = np.array(case["t"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if case["op"] == "step":
                _, y = signal.step(system, T=t)
            elif case["op"] == "impulse":
                _, y = signal.impulse(system, T=t)
            else:
                _, y, _ = signal.lsim(system, np.array(case["u"]), t, X0=case["x0"],
                                      interp=case["interp"])
        arm["y"] = [float(v) for v in np.atleast_1d(y)]
    except Exception:
        pass
    out.append(arm)
print(json.dumps(out))
"#;
    let query_json = serde_json::to_string(query).expect("serialize lti query");
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
                "failed to spawn python3 for the lti oracle: {e}"
            );
            eprintln!("skipping lti oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open lti oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "lti oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping lti oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for lti oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "lti oracle failed: {stderr}"
        );
        eprintln!("skipping lti oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse lti oracle JSON"))
}

#[test]
fn diff_signal_lti_exact_discretization() {
    let cases = cases();
    let Some(oracle) = scipy_oracle_or_skip(&cases) else {
        return;
    };
    let arms: HashMap<String, OracleArm> = oracle
        .into_iter()
        .map(|arm| (arm.case_id.clone(), arm))
        .collect();

    let mut diffs = Vec::new();
    let op_arms = ["step", "impulse", "lsim"];
    let mut ledger = CompareLedger::new("diff_signal_lti_exact_discretization", &op_arms);
    for case in &cases {
        let arm = &arms[&case.case_id];
        let fsci = Lti::new(case.num.clone(), case.den.clone())
            .and_then(|sys| match case.op.as_str() {
                "step" => sys.step(&case.t),
                "impulse" => sys.impulse(&case.t),
                _ => sys.lsim(&case.u, &case.t, case.x0.as_deref(), case.interp),
            })
            .ok();
        let Some((want, y)) =
            ledger.slices(&case.op, &case.case_id, arm.y.as_deref(), fsci.as_deref())
        else {
            continue;
        };
        let max_abs_diff = y
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let bad = y.iter().zip(want).position(|(a, b)| {
            let d = (a - b).abs();
            d.is_nan() || d > OUT_ABS_TOL + OUT_REL_TOL * b.abs()
        });
        let reason = bad.map_or_else(String::new, |i| {
            format!("sample {i}: {} vs {}", y[i], want[i])
        });
        ledger.compared(&case.op, &case.case_id, bad.is_none());
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            max_abs_diff,
            pass: bad.is_none(),
            reason,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_signal_lti_exact_discretization".into(),
        category: "scipy.signal.step / impulse / lsim (exact discretization)".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        cases: diffs.clone(),
    };
    fs::create_dir_all(output_dir()).expect("create lti diff dir");
    fs::write(
        output_dir().join("diff_signal_lti_exact_discretization.json"),
        serde_json::to_string_pretty(&log).expect("serialize lti log"),
    )
    .expect("write lti log");
    for d in &diffs {
        println!(
            "{} max_abs_diff={:e} {}",
            d.case_id, d.max_abs_diff, d.reason
        );
    }
    assert_eq!(diffs.len(), cases.len(), "every case must be compared");
    assert!(all_pass, "Lti step/impulse/lsim vs scipy.signal failed");
    ledger.finish(
        op_arms
            .iter()
            .map(|op| cases.iter().filter(|c| c.op == *op).count())
            .min()
            .unwrap_or(0),
    );
}
