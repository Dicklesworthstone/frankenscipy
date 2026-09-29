#![forbid(unsafe_code)]
//! Cover fsci_opt::validate_wolfe_params + fsci_opt::root dispatcher.
//!
//! Resolves [frankenscipy-729wa]. validate_wolfe_params has four
//! error branches plus the OK case. The root() dispatcher routes to
//! five distinct multivariate root methods (Hybr/Broyden1/Broyden2/
//! Anderson/Lm); we exercise all five on a small 2D rosenbrock-style
//! gradient system and verify each finds the same root.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_opt::linesearch::validate_wolfe_params;
use fsci_opt::{MultivariateRootMethod, MultivariateRootOptions, WolfeParams, root};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    pass: bool,
    note: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create wolfe_root diff dir");
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

#[test]
fn diff_opt_validate_wolfe_root_dispatch() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut check = |id: &str, ok: bool, note: String| {
        diffs.push(CaseDiff {
            case_id: id.into(),
            pass: ok,
            note,
        });
    };

    // === validate_wolfe_params: OK path ===
    {
        let ok = validate_wolfe_params(WolfeParams::default()).is_ok();
        check("wolfe_default_ok", ok, String::new());
    }

    // === validate_wolfe_params: c1 >= c2 → error ===
    {
        let bad = WolfeParams {
            c1: 0.9,
            c2: 0.5,
            ..WolfeParams::default()
        };
        let err = validate_wolfe_params(bad).is_err();
        check("wolfe_c1_ge_c2_errors", err, String::new());
    }

    // === validate_wolfe_params: c1 == 0 → error ===
    {
        let bad = WolfeParams {
            c1: 0.0,
            ..WolfeParams::default()
        };
        let err = validate_wolfe_params(bad).is_err();
        check("wolfe_c1_zero_errors", err, String::new());
    }

    // === validate_wolfe_params: c2 == 1 → error ===
    {
        let bad = WolfeParams {
            c2: 1.0,
            ..WolfeParams::default()
        };
        let err = validate_wolfe_params(bad).is_err();
        check("wolfe_c2_one_errors", err, String::new());
    }

    // === validate_wolfe_params: amin >= amax → error ===
    {
        let bad = WolfeParams {
            amin: 100.0,
            amax: 50.0,
            ..WolfeParams::default()
        };
        let err = validate_wolfe_params(bad).is_err();
        check("wolfe_amin_ge_amax_errors", err, String::new());
    }

    // === validate_wolfe_params: amin <= 0 → error ===
    {
        let bad = WolfeParams {
            amin: 0.0,
            ..WolfeParams::default()
        };
        let err = validate_wolfe_params(bad).is_err();
        check("wolfe_amin_zero_errors", err, String::new());
    }

    // === validate_wolfe_params: maxiter == 0 → error ===
    {
        let bad = WolfeParams {
            maxiter: 0,
            ..WolfeParams::default()
        };
        let err = validate_wolfe_params(bad).is_err();
        check("wolfe_maxiter_zero_errors", err, String::new());
    }

    // === root() dispatcher: 2D nonlinear system with a known root at (1, 0) ===
    // F(x) = [(x0-1)² + x1², x0 - 1 - x1]  → only solution is (1, 0)
    // Simpler still: F(x) = [x0 - 1, x1]   → unique root (1, 0)
    let f = |x: &[f64]| -> Vec<f64> { vec![x[0] - 1.0, x[1]] };
    let x0 = vec![0.5, 0.5];
    let expected_root = [1.0, 0.0];

    // The root() arm compares against the analytic root; the validate_wolfe_params checks above
    // are fail-closed assertions with no reference value and stay outside the ledger.
    let mut ledger = CompareLedger::new("diff_opt_validate_wolfe_root_dispatch", &["root"]);
    let methods = [
        MultivariateRootMethod::Hybr,
        MultivariateRootMethod::Broyden1,
        MultivariateRootMethod::Broyden2,
        MultivariateRootMethod::Anderson,
        MultivariateRootMethod::Lm,
    ];
    for method in methods {
        let opts = MultivariateRootOptions {
            method,
            tol: 1.0e-8,
            max_iter: 200,
        };
        let res = root(f, &x0, opts);
        let case_id = format!("root_method_{method:?}");
        // The analytic root always exists: an error or an unconverged run is an fsci failure.
        let fsci_x = res
            .as_ref()
            .ok()
            .filter(|r| r.converged)
            .map(|r| r.x.as_slice());
        let Some((_, x)) = ledger.slices("root", &case_id, Some(expected_root.as_slice()), fsci_x)
        else {
            let note = match &res {
                Ok(r) => format!("x={:?} converged={}", r.x, r.converged),
                Err(e) => format!("error: {e:?}"),
            };
            check(&case_id, false, note);
            continue;
        };
        let close = (x[0] - 1.0).abs() < 1.0e-4 && x[1].abs() < 1.0e-4;
        ledger.compared("root", &case_id, close);
        check(&case_id, close, format!("x={x:?} converged=true"));
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_opt_validate_wolfe_root_dispatch".into(),
        category: "fsci_opt::validate_wolfe_params + root() dispatch coverage".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("validate_wolfe_root mismatch: {} — {}", d.case_id, d.note);
        }
    }

    assert!(
        all_pass,
        "validate_wolfe + root dispatch coverage failed: {} cases",
        diffs.len(),
    );
    ledger.finish(methods.len());
}
