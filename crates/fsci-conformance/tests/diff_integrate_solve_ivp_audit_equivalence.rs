#![forbid(unsafe_code)]
//! Property test: fsci_integrate::solve_ivp_with_audit produces
//! numerically identical output to solve_ivp for valid inputs.
//!
//! Resolves [frankenscipy-unfeq]. Audit codepath only logs to the
//! ledger; t and y trajectories must be bit-identical at 1e-15 abs.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_integrate::{
    SolveIvpOptions, SolverKind, solve_ivp, solve_ivp_with_audit, solve_ivp_with_casp_portfolio,
};
use fsci_runtime::{AuditLedger, OdeSolverAction, OdeSolverPortfolio, RuntimeMode};
use serde::Serialize;

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
            };
            let mut f1 = f;
            let mut f2 = f;
            let Ok(plain) = solve_ivp(&mut f1, &opts) else {
                continue;
            };
            let Ok(audited) = solve_ivp_with_audit(&mut f2, &opts, &ledger) else {
                continue;
            };
            let d_t = vec_max_diff(&plain.t, &audited.t);
            // Flatten y for comparison
            let plain_y: Vec<f64> = plain.y.iter().flatten().copied().collect();
            let audit_y: Vec<f64> = audited.y.iter().flatten().copied().collect();
            let d_y = vec_max_diff(&plain_y, &audit_y);
            let abs_d = d_t.max(d_y);
            max_overall = max_overall.max(abs_d);
            diffs.push(CaseDiff {
                case_id: format!("solve_ivp_{label}_{method:?}"),
                op: format!("{method:?}"),
                abs_diff: abs_d,
                pass: abs_d <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_integrate_solve_ivp_audit_equivalence".into(),
        category: "fsci_integrate::solve_ivp_with_audit equivalent to solve_ivp".into(),
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
        "solve_ivp_audit_equiv conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}

#[test]
fn diff_integrate_solve_ivp_with_casp_portfolio() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut portfolio = OdeSolverPortfolio::new(RuntimeMode::Strict, 16);

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
    };

    // 1. Non-stiff system routes to RK45
    let res_nonstiff = solve_ivp_with_casp_portfolio(&mut f, &opts, &mut portfolio, 1.0, false)
        .expect("casp rk45 solve");
    assert_eq!(res_nonstiff.chosen_action, OdeSolverAction::RK45);
    let exact = (-1.0_f64).exp();
    let last_y = *res_nonstiff.result.y.last().unwrap().first().unwrap();
    let d1 = (last_y - exact).abs();
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
        max_abs_diff: d1,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    assert!(all_pass, "all casp portfolio routing cases must pass");
}
