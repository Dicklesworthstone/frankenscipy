#![forbid(unsafe_code)]
//! Analytic-derivative parity for fsci_opt::derivative on hard-coded
//! model identifiers (square, cube, sin, cos, exp, log, sqrt). Uses the
//! known analytic derivative as the oracle; no scipy needed.
//!
//! Resolves [frankenscipy-s4h6q]. 1e-8 abs.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_opt::{DifferentiateOptions, derivative};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-8;
const ARMS: [&str; 7] = ["square", "cube", "sin", "cos", "exp", "log", "sqrt"];

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    model: String,
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
    fs::create_dir_all(output_dir()).expect("create derivative diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize derivative diff log");
    fs::write(path, json).expect("write derivative diff log");
}

#[test]
fn diff_opt_derivative() {
    let opts = DifferentiateOptions::default();
    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    // (model, fsci closure, analytic derivative closure, x value)
    let models: Vec<(&str, Box<dyn Fn(f64) -> f64>, Box<dyn Fn(f64) -> f64>)> = vec![
        (
            "square",
            Box::new(|x: f64| x * x),
            Box::new(|x: f64| 2.0 * x),
        ),
        (
            "cube",
            Box::new(|x: f64| x * x * x),
            Box::new(|x: f64| 3.0 * x * x),
        ),
        (
            "sin",
            Box::new(|x: f64| x.sin()),
            Box::new(|x: f64| x.cos()),
        ),
        (
            "cos",
            Box::new(|x: f64| x.cos()),
            Box::new(|x: f64| -x.sin()),
        ),
        (
            "exp",
            Box::new(|x: f64| x.exp()),
            Box::new(|x: f64| x.exp()),
        ),
        ("log", Box::new(|x: f64| x.ln()), Box::new(|x: f64| 1.0 / x)),
        (
            "sqrt",
            Box::new(|x: f64| x.sqrt()),
            Box::new(|x: f64| 0.5 / x.sqrt()),
        ),
    ];

    let xs_default = [0.5_f64, 1.0, 1.5, 2.5, 5.0];
    let mut ledger = CompareLedger::new("diff_opt_derivative", &ARMS);
    for (model, f, df_true) in &models {
        for x in xs_default {
            // log/sqrt need x > 0; use positive x set
            if (*model == "log" || *model == "sqrt") && x <= 0.0 {
                continue;
            }
            let case_id = format!("{model}_x{x}");
            let fsci_df = derivative(|t: f64| f(t), x, opts).ok().map(|res| res.df);
            // SciPy 1.17.1's own derivative(np.log, 0.5) fails (status -3: its default stencil
            // evaluates log at 0), so fsci must refuse too.
            if *model == "log" && x == 0.5 {
                ledger.expected_raise(model, &case_id, fsci_df.is_none());
                continue;
            }
            // SciPy recovers from a non-finite evaluation at these points and fsci does not.
            if (*model == "log" && x == 1.0) || (*model == "sqrt" && x == 0.5) {
                ledger.allowlisted(
                    model,
                    &case_id,
                    "frankenscipy-3vcjx",
                    "fsci refuses where scipy.differentiate.derivative succeeds",
                );
                continue;
            }
            let Some((expected, df)) = ledger.pair(model, &case_id, Some(df_true(x)), fsci_df)
            else {
                continue;
            };
            let abs_d = (df - expected).abs();
            max_overall = max_overall.max(abs_d);
            ledger.compared(model, &case_id, abs_d <= ABS_TOL);
            diffs.push(CaseDiff {
                case_id,
                model: (*model).into(),
                abs_diff: abs_d,
                pass: abs_d <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_derivative".into(),
        category: "fsci_opt::derivative vs analytic".into(),
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
            eprintln!(
                "{} mismatch: {} abs_diff={}",
                d.model, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "derivative conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    // log/sqrt admit only x > 0, and each has one case allowlisted under frankenscipy-3vcjx.
    ledger.finish(xs_default.iter().filter(|&&x| x > 0.0).count() - 1);
}
