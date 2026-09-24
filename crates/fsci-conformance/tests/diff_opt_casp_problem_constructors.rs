#![forbid(unsafe_code)]
//! Verify OptCaspProblem::unbounded_from_x0 and OptCaspProblem::from_x0_and_options
//! populate the documented fields from x0 length and MinimizeOptions inputs.
//!
//! Resolves [frankenscipy-oh7zo]. The constructors compute dimension from
//! x0.len(), variable_scale_ratio from x0 magnitudes, has_box_bounds from
//! options.bounds (only if any limit is finite), has_general_constraints
//! from a non-empty options.constraints, gradient_available from
//! options.gradient_available, and hessian_product_available from
//! options.hessp.is_some(). The unbounded constructor must always report
//! has_box_bounds=false and has_general_constraints=false regardless of
//! options.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::minimize::OptCaspProblem;
use fsci_opt::types::{Bound, Constraint, MinimizeOptions};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    field: String,
    expected: String,
    actual: String,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create casp_ctor diff dir");
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

fn check<T: PartialEq + std::fmt::Debug>(
    diffs: &mut Vec<CaseDiff>,
    case_id: &str,
    field: &str,
    expected: T,
    actual: T,
) {
    let pass = expected == actual;
    diffs.push(CaseDiff {
        case_id: case_id.into(),
        field: field.into(),
        expected: format!("{expected:?}"),
        actual: format!("{actual:?}"),
        pass,
    });
}

fn hessp_stub(_x: &[f64], p: &[f64]) -> Vec<f64> {
    p.to_vec()
}

// Shared box-bound slices.
static BOUNDS_FINITE: &[Bound] = &[(Some(-1.0), Some(1.0)), (Some(0.0), Some(10.0))];
static BOUNDS_ALL_NONE: &[Bound] = &[(None, None), (None, None)];
static BOUNDS_PARTIAL: &[Bound] = &[(Some(0.0), None), (None, None)];

#[test]
fn diff_opt_casp_problem_constructors() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();

    // Probe 1: unbounded_from_x0 with default options
    //   - dimension == x0.len()
    //   - variable_scale_ratio finite >= 1
    //   - has_box_bounds = false (constructor forces this)
    //   - has_general_constraints = false
    //   - gradient_available = options.gradient_available (default true)
    //   - hessian_product_available = false (default options.hessp = None)
    {
        let x0 = vec![1.0_f64, 2.0, 3.0];
        let opts = MinimizeOptions::default();
        let p = OptCaspProblem::unbounded_from_x0(&x0, opts);
        check(
            &mut diffs,
            "unbounded_default",
            "dimension",
            3usize,
            p.dimension,
        );
        check(
            &mut diffs,
            "unbounded_default",
            "has_box_bounds",
            false,
            p.has_box_bounds,
        );
        check(
            &mut diffs,
            "unbounded_default",
            "has_general_constraints",
            false,
            p.has_general_constraints,
        );
        check(
            &mut diffs,
            "unbounded_default",
            "gradient_available",
            true,
            p.gradient_available,
        );
        check(
            &mut diffs,
            "unbounded_default",
            "hessian_product_available",
            false,
            p.hessian_product_available,
        );
        check(
            &mut diffs,
            "unbounded_default",
            "scale_finite",
            true,
            p.variable_scale_ratio.is_finite(),
        );
        check(
            &mut diffs,
            "unbounded_default",
            "scale_ge_one",
            true,
            p.variable_scale_ratio >= 1.0,
        );
    }

    // Probe 2: unbounded_from_x0 ignores options.bounds and options.constraints
    {
        let x0 = vec![0.5_f64, -0.25];
        let constraints = [Constraint::ineq(|v: &[f64]| vec![v[0]])];
        let opts = MinimizeOptions {
            bounds: Some(BOUNDS_FINITE),
            constraints: &constraints,
            hessp: Some(hessp_stub),
            gradient_available: false,
            ..MinimizeOptions::default()
        };
        let p = OptCaspProblem::unbounded_from_x0(&x0, opts);
        check(
            &mut diffs,
            "unbounded_overrides",
            "dimension",
            2usize,
            p.dimension,
        );
        check(
            &mut diffs,
            "unbounded_overrides",
            "has_box_bounds",
            false,
            p.has_box_bounds,
        );
        check(
            &mut diffs,
            "unbounded_overrides",
            "has_general_constraints",
            false,
            p.has_general_constraints,
        );
        check(
            &mut diffs,
            "unbounded_overrides",
            "gradient_available",
            false,
            p.gradient_available,
        );
        check(
            &mut diffs,
            "unbounded_overrides",
            "hessian_product_available",
            true,
            p.hessian_product_available,
        );
    }

    // Probe 3: from_x0_and_options with finite box bounds => has_box_bounds=true
    {
        let x0 = vec![1.0_f64, 2.0];
        let opts = MinimizeOptions {
            bounds: Some(BOUNDS_FINITE),
            ..MinimizeOptions::default()
        };
        let p = OptCaspProblem::from_x0_and_options(&x0, opts);
        check(
            &mut diffs,
            "fxo_finite_bounds",
            "dimension",
            2usize,
            p.dimension,
        );
        check(
            &mut diffs,
            "fxo_finite_bounds",
            "has_box_bounds",
            true,
            p.has_box_bounds,
        );
        check(
            &mut diffs,
            "fxo_finite_bounds",
            "has_general_constraints",
            false,
            p.has_general_constraints,
        );
        check(
            &mut diffs,
            "fxo_finite_bounds",
            "hessian_product_available",
            false,
            p.hessian_product_available,
        );
        check(
            &mut diffs,
            "fxo_finite_bounds",
            "gradient_available",
            true,
            p.gradient_available,
        );
    }

    // Probe 4: from_x0_and_options with Some(bounds) but all (None, None) => has_box_bounds=false
    // (bounds_have_finite_limit requires at least one Some endpoint)
    {
        let x0 = vec![3.0_f64, 4.0];
        let opts = MinimizeOptions {
            bounds: Some(BOUNDS_ALL_NONE),
            ..MinimizeOptions::default()
        };
        let p = OptCaspProblem::from_x0_and_options(&x0, opts);
        check(
            &mut diffs,
            "fxo_none_bounds",
            "has_box_bounds",
            false,
            p.has_box_bounds,
        );
    }

    // Probe 5: from_x0_and_options with partial bound (one Some endpoint) => has_box_bounds=true
    {
        let x0 = vec![0.0_f64, 0.0];
        let opts = MinimizeOptions {
            bounds: Some(BOUNDS_PARTIAL),
            ..MinimizeOptions::default()
        };
        let p = OptCaspProblem::from_x0_and_options(&x0, opts);
        check(
            &mut diffs,
            "fxo_partial_bounds",
            "has_box_bounds",
            true,
            p.has_box_bounds,
        );
    }

    // Probe 6: from_x0_and_options with bounds=None => has_box_bounds=false
    {
        let x0 = vec![1.0_f64];
        let opts = MinimizeOptions::default();
        let p = OptCaspProblem::from_x0_and_options(&x0, opts);
        check(
            &mut diffs,
            "fxo_no_bounds",
            "dimension",
            1usize,
            p.dimension,
        );
        check(
            &mut diffs,
            "fxo_no_bounds",
            "has_box_bounds",
            false,
            p.has_box_bounds,
        );
        check(
            &mut diffs,
            "fxo_no_bounds",
            "has_general_constraints",
            false,
            p.has_general_constraints,
        );
    }

    // Probe 7: from_x0_and_options with a constraint and hessp=Some
    {
        let x0 = vec![1.0_f64, 2.0, 3.0, 4.0];
        let constraints = [Constraint::eq(|v: &[f64]| vec![v[0] - 1.0])];
        let opts = MinimizeOptions {
            constraints: &constraints,
            hessp: Some(hessp_stub),
            gradient_available: false,
            ..MinimizeOptions::default()
        };
        let p = OptCaspProblem::from_x0_and_options(&x0, opts);
        check(
            &mut diffs,
            "fxo_general_constr_hessp",
            "dimension",
            4usize,
            p.dimension,
        );
        check(
            &mut diffs,
            "fxo_general_constr_hessp",
            "has_box_bounds",
            false,
            p.has_box_bounds,
        );
        check(
            &mut diffs,
            "fxo_general_constr_hessp",
            "has_general_constraints",
            true,
            p.has_general_constraints,
        );
        check(
            &mut diffs,
            "fxo_general_constr_hessp",
            "gradient_available",
            false,
            p.gradient_available,
        );
        check(
            &mut diffs,
            "fxo_general_constr_hessp",
            "hessian_product_available",
            true,
            p.hessian_product_available,
        );
    }

    // Probe 8: variable_scale_ratio reflects x0 magnitude spread
    // For x0=[1.0, 1.0e6], scales = max(|v|, 1.0) = [1.0, 1.0e6], ratio = 1.0e6.
    // For x0=[1.0, 1.0], ratio = 1.0.
    {
        let x0_a = vec![1.0_f64, 1.0e6];
        let p_a = OptCaspProblem::unbounded_from_x0(&x0_a, MinimizeOptions::default());
        let x0_b = vec![1.0_f64, 1.0];
        let p_b = OptCaspProblem::unbounded_from_x0(&x0_b, MinimizeOptions::default());
        check(
            &mut diffs,
            "scale_spread",
            "ratio_a_eq_1e6",
            true,
            (p_a.variable_scale_ratio - 1.0e6).abs() < 1.0e-9,
        );
        check(
            &mut diffs,
            "scale_spread",
            "ratio_b_eq_1",
            true,
            (p_b.variable_scale_ratio - 1.0).abs() < 1.0e-12,
        );
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_casp_problem_constructors".into(),
        category: "OptCaspProblem::{unbounded_from_x0, from_x0_and_options} field population"
            .into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "casp_ctor mismatch: {}.{} got {} (expected {})",
                d.case_id, d.field, d.actual, d.expected
            );
        }
    }

    assert!(
        all_pass,
        "OptCaspProblem constructor coverage failed: {} cases",
        diffs.len(),
    );
}
