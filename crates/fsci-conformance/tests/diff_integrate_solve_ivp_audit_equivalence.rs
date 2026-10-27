#![forbid(unsafe_code)]
//! Property test: fsci_integrate::solve_ivp_with_audit produces
//! numerically identical output to solve_ivp for valid inputs.
//!
//! Resolves [frankenscipy-unfeq]. Audit codepath only logs to the
//! ledger; t and y trajectories must be bit-identical at 1e-15 abs.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_integrate::{
    SolveIvpOptions, SolverKind, ToleranceValue, solve_ivp, solve_ivp_with_audit,
    solve_ivp_with_casp, solve_ivp_with_casp_portfolio,
};
use fsci_runtime::{AuditLedger, OdeSolverAction, OdeSolverPortfolio, RuntimeMode};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-15;

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
    compared: BTreeMap<String, ArmCounts>,
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
    fs::create_dir_all(output_dir()).expect("create solve_ivp_audit diff dir");
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

fn vec_max_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn diff_integrate_solve_ivp_audit_equivalence() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut max_overall = 0.0_f64;
    let ledger = AuditLedger::shared();
    // The reference side is plain `solve_ivp`, the compared side `solve_ivp_with_audit`; one arm
    // per method (the CaseDiff `op`).
    let mut compare_ledger = CompareLedger::new(
        "diff_integrate_solve_ivp_audit_equivalence",
        &["Rk45", "Rk23", "Dop853"],
    );

    // Three test ODEs:
    let probes: &[(&str, (f64, f64), Vec<f64>, fn(f64, &[f64]) -> Vec<f64>)] = &[
        // Exponential decay: y' = -y, y(0)=1 → y(t)=exp(-t)
        ("expdecay", (0.0, 2.0), vec![1.0], |_t, y| vec![-y[0]]),
        // Harmonic: y'' = -y, as system [y'=v, v'=-y], y(0)=1, v(0)=0
        (
            "harmonic",
            (0.0, std::f64::consts::PI),
            vec![1.0, 0.0],
            |_t, y| vec![y[1], -y[0]],
        ),
        // Lotka-Volterra-ish: u'=u*(1-v), v'=v*(u-1), small horizon
        ("lv_small", (0.0, 1.0), vec![1.5, 1.0], |_t, y| {
            vec![y[0] * (1.0 - y[1]), y[1] * (y[0] - 1.0)]
        }),
    ];

    for &(label, t_span, ref y0, f) in probes {
        for method in [SolverKind::Rk45, SolverKind::Rk23, SolverKind::Dop853] {
            let opts = SolveIvpOptions {
                t_span,
                y0,
                method,
                t_eval: None,
                dense_output: false,
                events: None,
                rtol: 1e-6,
                atol: fsci_integrate::ToleranceValue::Scalar(1e-9),
                first_step: None,
                max_step: f64::INFINITY,
                mode: RuntimeMode::Strict,
                jac: None,
                jac_sparsity: None,
            };
            let mut f1 = f;
            let mut f2 = f;
            let case_id = format!("solve_ivp_{label}_{method:?}");
            let arm = format!("{method:?}");
            let plain = solve_ivp(&mut f1, &opts).ok();
            let audited = solve_ivp_with_audit(&mut f2, &opts, &ledger).ok();
            let Some((plain, audited)) = compare_ledger.both(&arm, &case_id, plain, audited) else {
                continue;
            };
            let d_t = vec_max_diff(&plain.t, &audited.t);
            // Flatten y for comparison
            let plain_y: Vec<f64> = plain.y.iter().flatten().copied().collect();
            let audit_y: Vec<f64> = audited.y.iter().flatten().copied().collect();
            let d_y = vec_max_diff(&plain_y, &audit_y);
            let abs_d = d_t.max(d_y);
            max_overall = max_overall.max(abs_d);
            // vec_max_diff's max-fold reads a NaN element as a zero difference, so a NaN on
            // either side must fail the case explicitly.
            let no_nan = [&plain.t, &audited.t, &plain_y, &audit_y]
                .iter()
                .all(|v| !v.iter().any(|x| x.is_nan()));
            let pass = abs_d <= ABS_TOL && no_nan;
            compare_ledger.compared(&arm, &case_id, pass);
            diffs.push(CaseDiff {
                case_id,
                op: arm,
                abs_diff: abs_d,
                pass,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_integrate_solve_ivp_audit_equivalence".into(),
        category: "fsci_integrate::solve_ivp_with_audit equivalent to solve_ivp".into(),
        case_count: diffs.len(),
        compared: compare_ledger.counts().clone(),
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
        "solve_ivp_audit_equiv conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    compare_ledger.finish(probes.len());
}

#[test]
fn diff_integrate_solve_ivp_with_casp_portfolio() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut portfolio = OdeSolverPortfolio::new(RuntimeMode::Strict, 16);
    let mut compare_ledger = CompareLedger::new(
        "diff_integrate_solve_ivp_with_casp_portfolio",
        &["RK45", "BDF", "Radau"],
    );

    let mut f = |_t: f64, y: &[f64]| vec![-y[0]];
    let opts = SolveIvpOptions {
        t_span: (0.0, 1.0),
        y0: &[1.0],
        method: SolverKind::Rk45,
        t_eval: None,
        dense_output: false,
        events: None,
        rtol: 1e-6,
        atol: fsci_integrate::ToleranceValue::Scalar(1e-9),
        first_step: None,
        max_step: f64::INFINITY,
        mode: RuntimeMode::Strict,
        jac: None,
        jac_sparsity: None,
    };

    // 1. Non-stiff system routes to RK45
    let res_nonstiff = solve_ivp_with_casp_portfolio(&mut f, &opts, &mut portfolio, 1.0, false)
        .expect("casp rk45 solve");
    assert_eq!(res_nonstiff.chosen_action, OdeSolverAction::RK45);
    let exact = (-1.0_f64).exp();
    let last_y = *res_nonstiff.result.y.last().unwrap().first().unwrap();
    let d1 = (last_y - exact).abs();
    compare_ledger.compared("RK45", "casp_nonstiff_routes_to_rk45", d1 < 1e-5);
    diffs.push(CaseDiff {
        case_id: "casp_nonstiff_routes_to_rk45".into(),
        op: "RK45".into(),
        abs_diff: d1,
        pass: d1 < 1e-5,
    });

    // 2. Stiff system routes to BDF
    let res_stiff = solve_ivp_with_casp_portfolio(&mut f, &opts, &mut portfolio, 1e6, false)
        .expect("casp bdf solve");
    assert_eq!(res_stiff.chosen_action, OdeSolverAction::BDF);
    compare_ledger.compared("BDF", "casp_stiff_routes_to_bdf", res_stiff.result.success);
    diffs.push(CaseDiff {
        case_id: "casp_stiff_routes_to_bdf".into(),
        op: "BDF".into(),
        abs_diff: 0.0,
        pass: res_stiff.result.success,
    });

    // 3. Algebraic / DAE constrained system routes to Radau
    let res_radau = solve_ivp_with_casp_portfolio(&mut f, &opts, &mut portfolio, 10.0, true)
        .expect("casp radau solve");
    assert_eq!(res_radau.chosen_action, OdeSolverAction::Radau);
    compare_ledger.compared(
        "Radau",
        "casp_algebraic_routes_to_radau",
        res_radau.result.success,
    );
    diffs.push(CaseDiff {
        case_id: "casp_algebraic_routes_to_radau".into(),
        op: "Radau".into(),
        abs_diff: 0.0,
        pass: res_radau.result.success,
    });

    assert_eq!(
        portfolio.evidence_len(),
        3,
        "portfolio must accumulate 3 evidence entries"
    );

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_integrate_solve_ivp_with_casp_portfolio".into(),
        category: "fsci_integrate::solve_ivp_with_casp_portfolio CASP routing".into(),
        case_count: diffs.len(),
        compared: compare_ledger.counts().clone(),
        max_abs_diff: d1,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    assert!(all_pass, "all casp portfolio routing cases must pass");
    // One routing case per arm.
    compare_ledger.finish(1);
}

#[derive(Debug, Deserialize)]
struct ScipyIvpEntry {
    case_id: String,
    y_final: Vec<f64>,
    success: bool,
}

#[test]
fn diff_integrate_solve_ivp_with_casp_live_scipy() {
    let start = Instant::now();
    let mut portfolio = OdeSolverPortfolio::new(RuntimeMode::Strict, 32);

    // 1. Harmonic oscillator (non-stiff -> RK45)
    let opts_harmonic = SolveIvpOptions {
        t_span: (0.0, 2.0),
        y0: &[1.0, 0.0],
        rtol: 1e-6,
        atol: ToleranceValue::Scalar(1e-9),
        ..SolveIvpOptions::default()
    };
    let mut harmonic = |_t: f64, y: &[f64]| vec![y[1], -y[0]];
    let res_harmonic =
        solve_ivp_with_casp(&mut harmonic, &opts_harmonic, &mut portfolio).expect("harmonic");
    assert_eq!(res_harmonic.chosen_action, OdeSolverAction::RK45);

    // 2. Van der Pol mu=1 (non-stiff -> RK45)
    let opts_vdp1 = SolveIvpOptions {
        t_span: (0.0, 5.0),
        y0: &[2.0, 0.0],
        rtol: 1e-6,
        atol: ToleranceValue::Scalar(1e-9),
        ..SolveIvpOptions::default()
    };
    let mut vdp1 = |_t: f64, y: &[f64]| vec![y[1], (1.0 - y[0] * y[0]) * y[1] - y[0]];
    let res_vdp1 = solve_ivp_with_casp(&mut vdp1, &opts_vdp1, &mut portfolio).expect("vdp_mu1");
    assert_eq!(res_vdp1.chosen_action, OdeSolverAction::RK45);

    // 3. Van der Pol mu=1000 (stiff -> BDF)
    let opts_vdp1000 = SolveIvpOptions {
        t_span: (0.0, 5.0),
        y0: &[2.0, 0.0],
        rtol: 1e-6,
        atol: ToleranceValue::Scalar(1e-9),
        ..SolveIvpOptions::default()
    };
    let mut vdp1000 = |_t: f64, y: &[f64]| vec![y[1], 1000.0 * (1.0 - y[0] * y[0]) * y[1] - y[0]];
    let res_vdp1000 =
        solve_ivp_with_casp(&mut vdp1000, &opts_vdp1000, &mut portfolio).expect("vdp_mu1000");
    assert_eq!(res_vdp1000.chosen_action, OdeSolverAction::BDF);

    // 4. Robertson chemical kinetics (stiff with zero initial y1 -> BDF)
    let opts_robertson = SolveIvpOptions {
        t_span: (0.0, 40.0),
        y0: &[1.0, 0.0, 0.0],
        rtol: 1e-6,
        atol: ToleranceValue::Vector(vec![1e-8, 1e-11, 1e-6]),
        ..SolveIvpOptions::default()
    };
    let mut robertson = |_t: f64, y: &[f64]| {
        let y0 = y[0];
        let y1 = y[1];
        let y2 = y[2];
        vec![
            -0.04 * y0 + 1.0e4 * y1 * y2,
            0.04 * y0 - 1.0e4 * y1 * y2 - 3.0e7 * y1 * y1,
            3.0e7 * y1 * y1,
        ]
    };
    let res_robertson =
        solve_ivp_with_casp(&mut robertson, &opts_robertson, &mut portfolio).expect("robertson");
    assert!(matches!(
        res_robertson.chosen_action,
        OdeSolverAction::BDF | OdeSolverAction::Radau
    ));
    assert_eq!(
        portfolio.calibrator().total_predictions(),
        4,
        "calibrator must be fed once per top-level call (4 total)"
    );

    let action_name = |a: OdeSolverAction| match a {
        OdeSolverAction::RK45 => "RK45",
        OdeSolverAction::RK23 => "RK23",
        OdeSolverAction::DOP853 => "DOP853",
        OdeSolverAction::BDF => "BDF",
        OdeSolverAction::Radau => "Radau",
    };

    let py_script = format!(
        r#"
import json
import numpy as np
from scipy.integrate import solve_ivp

out = []

r1 = solve_ivp(lambda t, y: [y[1], -y[0]], (0.0, 2.0), [1.0, 0.0], method="{m1}", rtol=1e-6, atol=1e-9)
out.append({{"case_id": "harmonic", "y_final": r1.y[:, -1].tolist(), "success": bool(r1.success)}})

r2 = solve_ivp(lambda t, y: [y[1], (1.0 - y[0]**2)*y[1] - y[0]], (0.0, 5.0), [2.0, 0.0], method="{m2}", rtol=1e-6, atol=1e-9)
out.append({{"case_id": "vdp_mu1", "y_final": r2.y[:, -1].tolist(), "success": bool(r2.success)}})

r3 = solve_ivp(lambda t, y: [y[1], 1000.0*(1.0 - y[0]**2)*y[1] - y[0]], (0.0, 5.0), [2.0, 0.0], method="{m3}", rtol=1e-6, atol=1e-9)
out.append({{"case_id": "vdp_mu1000", "y_final": r3.y[:, -1].tolist(), "success": bool(r3.success)}})

def robertson(t, y):
    return [
        -0.04 * y[0] + 1e4 * y[1] * y[2],
        0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1]**2,
        3e7 * y[1]**2,
    ]
r4 = solve_ivp(robertson, (0.0, 40.0), [1.0, 0.0, 0.0], method="{m4}", rtol=1e-6, atol=[1e-8, 1e-11, 1e-6])
out.append({{"case_id": "robertson", "y_final": r4.y[:, -1].tolist(), "success": bool(r4.success)}})

print(json.dumps(out))
"#,
        m1 = action_name(res_harmonic.chosen_action),
        m2 = action_name(res_vdp1.chosen_action),
        m3 = action_name(res_vdp1000.chosen_action),
        m4 = action_name(res_robertson.chosen_action),
    );

    let output = match fsci_conformance::scipy_oracle_command()
        .args(["-c", &py_script])
        .output()
    {
        Ok(out) => out,
        Err(e) => {
            assert!(
                std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").is_err(),
                "scipy oracle required but spawn failed: {e}"
            );
            return;
        }
    };
    if !output.status.success() {
        assert!(
            std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").is_err(),
            "scipy failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let scipy_entries: Vec<ScipyIvpEntry> =
        serde_json::from_slice(&output.stdout).expect("parse scipy JSON");

    let rust_results = [
        ("harmonic", &res_harmonic, 1e-4),
        ("vdp_mu1", &res_vdp1, 1e-4),
        ("vdp_mu1000", &res_vdp1000, 1e-3),
        ("robertson", &res_robertson, 1e-3),
    ];

    let mut compare_ledger = CompareLedger::new(
        "diff_integrate_solve_ivp_with_casp_live_scipy",
        &["harmonic", "vdp_mu1", "vdp_mu1000", "robertson"],
    );
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for ((case_id, rust_res, tol), scipy_res) in rust_results.iter().zip(scipy_entries.iter()) {
        assert_eq!(*case_id, scipy_res.case_id);
        let rust_final = rust_res.result.y.last().expect("rust y_final");
        let Some((r_y, s_y)) = compare_ledger.both(
            case_id,
            case_id,
            rust_res.result.success.then_some(rust_final),
            scipy_res.success.then_some(&scipy_res.y_final),
        ) else {
            continue;
        };
        let d = vec_max_diff(r_y, s_y);
        max_overall = max_overall.max(d);
        let pass = d <= *tol && !r_y.iter().any(|v| v.is_nan());
        compare_ledger.compared(case_id, case_id, pass);
        diffs.push(CaseDiff {
            case_id: (*case_id).to_string(),
            op: format!("{:?}", rust_res.chosen_action),
            abs_diff: d,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_integrate_solve_ivp_with_casp_live_scipy".into(),
        category: "fsci_integrate::solve_ivp_with_casp vs live scipy.integrate.solve_ivp".into(),
        case_count: diffs.len(),
        compared: compare_ledger.counts().clone(),
        max_abs_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        assert!(
            d.pass,
            "live scipy mismatch on {} ({}): abs_diff={}",
            d.case_id, d.op, d.abs_diff
        );
    }
    compare_ledger.finish(1);
}
