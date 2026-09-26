#![forbid(unsafe_code)]
//! Analytic-solution parity for fsci_integrate::odeint on ODEs with
//! closed-form solutions.
//!
//! Resolves [frankenscipy-yuofx]. 1e-4 abs at sampled t points.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_integrate::odeint;
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-4;
/// The closed-form cases below, one per block.
const CASE_IDS: [&str; 4] = [
    "exp_decay",
    "exp_growth",
    "harmonic_y1_is_cos",
    "exp_2t_half",
];

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create odeint diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize odeint diff log");
    fs::write(path, json).expect("write odeint diff log");
}

/// fsci's y[0] at every sample time. A short or empty solution then reaches the ledger as a
/// length mismatch, not as a zip that compares nothing; an empty row is a NaN, an fsci failure.
fn first_component(sol: &[Vec<f64>]) -> Vec<f64> {
    sol.iter()
        .map(|row| row.first().copied().unwrap_or(f64::NAN))
        .collect()
}

/// The closed form at every sample time.
fn exact_at(t: &[f64], exact: impl Fn(f64) -> f64) -> Vec<f64> {
    t.iter().map(|&ti| exact(ti)).collect()
}

fn max_abs_diff(fsci: &[f64], exact: &[f64]) -> f64 {
    fsci.iter()
        .zip(exact)
        .map(|(got, want)| (got - want).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn diff_integrate_odeint_closed_form() {
    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;
    let mut ledger = CompareLedger::new("diff_integrate_odeint_closed_form", &["odeint"]);

    // Sample times
    let t: Vec<f64> = (0..=10).map(|i| (i as f64) * 0.2).collect();

    // === exp decay: dy/dt = -y, y(0) = 1 → y(t) = e^(-t) ===
    {
        let case_id = CASE_IDS[0];
        let mut f = |y: &[f64], _t: f64| vec![-y[0]];
        let y0 = vec![1.0_f64];
        let fsci = odeint(&mut f, &y0, &t)
            .ok()
            .map(|sol| first_component(&sol));
        let exact = exact_at(&t, |ti| (-ti).exp());
        if let Some((want, got)) =
            ledger.slices("odeint", case_id, Some(exact.as_slice()), fsci.as_deref())
        {
            let max_d = max_abs_diff(got, want);
            max_overall = max_overall.max(max_d);
            ledger.compared("odeint", case_id, max_d <= ABS_TOL);
            diffs.push(CaseDiff {
                case_id: case_id.into(),
                abs_diff: max_d,
                pass: max_d <= ABS_TOL,
            });
        }
    }

    // === exp growth: dy/dt = y, y(0) = 1 → y(t) = e^t ===
    {
        let case_id = CASE_IDS[1];
        let mut f = |y: &[f64], _t: f64| vec![y[0]];
        let y0 = vec![1.0_f64];
        let fsci = odeint(&mut f, &y0, &t)
            .ok()
            .map(|sol| first_component(&sol));
        let exact = exact_at(&t, f64::exp);
        if let Some((want, got)) =
            ledger.slices("odeint", case_id, Some(exact.as_slice()), fsci.as_deref())
        {
            let max_d = max_abs_diff(got, want);
            max_overall = max_overall.max(max_d);
            ledger.compared("odeint", case_id, max_d <= ABS_TOL);
            diffs.push(CaseDiff {
                case_id: case_id.into(),
                abs_diff: max_d,
                pass: max_d <= ABS_TOL,
            });
        }
    }

    // === harmonic oscillator: y1' = y2, y2' = -y1; y(0) = (1, 0) → y1 = cos(t) ===
    {
        let case_id = CASE_IDS[2];
        let mut f = |y: &[f64], _t: f64| vec![y[1], -y[0]];
        let y0 = vec![1.0_f64, 0.0];
        let fsci = odeint(&mut f, &y0, &t)
            .ok()
            .map(|sol| first_component(&sol));
        let exact = exact_at(&t, f64::cos);
        if let Some((want, got)) =
            ledger.slices("odeint", case_id, Some(exact.as_slice()), fsci.as_deref())
        {
            let max_d = max_abs_diff(got, want);
            max_overall = max_overall.max(max_d);
            ledger.compared("odeint", case_id, max_d <= ABS_TOL);
            diffs.push(CaseDiff {
                case_id: case_id.into(),
                abs_diff: max_d,
                pass: max_d <= ABS_TOL,
            });
        }
    }

    // === Linear ODE: dy/dt = 2*y, y(0) = 0.5 → y(t) = 0.5*e^(2t) ===
    {
        let case_id = CASE_IDS[3];
        let mut f = |y: &[f64], _t: f64| vec![2.0 * y[0]];
        let y0 = vec![0.5_f64];
        let fsci = odeint(&mut f, &y0, &t)
            .ok()
            .map(|sol| first_component(&sol));
        let exact = exact_at(&t, |ti| 0.5 * (2.0 * ti).exp());
        if let Some((want, got)) =
            ledger.slices("odeint", case_id, Some(exact.as_slice()), fsci.as_deref())
        {
            let max_d = max_abs_diff(got, want);
            max_overall = max_overall.max(max_d);
            ledger.compared("odeint", case_id, max_d <= ABS_TOL);
            diffs.push(CaseDiff {
                case_id: case_id.into(),
                abs_diff: max_d,
                pass: max_d <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_integrate_odeint_closed_form".into(),
        category: "fsci_integrate::odeint vs analytic ODE solutions".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        max_abs_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("odeint mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "odeint conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    ledger.finish(CASE_IDS.len());
}
