#![forbid(unsafe_code)]
//! Branch-coverage property test for fsci_opt::select_minimize_method.
//!
//! Resolves [frankenscipy-dh3h1]. The CASP selector is documented to
//! route by constraint class, gradient availability, Hessian product,
//! dimension, and scale ratio. This test exercises each documented
//! branch and verifies the returned method.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_opt::OptimizeMethod;
use fsci_opt::minimize::{OptCaspProblem, select_minimize_method};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create casp_select diff dir");
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
fn diff_opt_select_minimize_method() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();

    // 1. General constraints → Slsqp: scipy.optimize.minimize's own routing when constraints
    // are given and no method is (frankenscipy-1ksfv.1). The old TrustConstr route led to a
    // kernel that has to refuse constraints (frankenscipy-1ksfv.2).
    let p1 = OptCaspProblem {
        dimension: 3,
        variable_scale_ratio: 1.0,
        has_box_bounds: false,
        has_general_constraints: true,
        gradient_available: true,
        hessian_product_available: false,
    };
    let d1 = select_minimize_method(p1).expect("p1 ok");
    diffs.push(CaseDiff {
        case_id: "general_constr".into(),
        expected: format!("{:?}", OptimizeMethod::Slsqp),
        actual: format!("{:?}", d1.method),
        pass: d1.method == OptimizeMethod::Slsqp,
    });

    // 2. Box bounds (no general constr) → LBfgsB
    let p2 = OptCaspProblem {
        dimension: 5,
        variable_scale_ratio: 1.0,
        has_box_bounds: true,
        has_general_constraints: false,
        gradient_available: true,
        hessian_product_available: false,
    };
    let d2 = select_minimize_method(p2).expect("p2 ok");
    diffs.push(CaseDiff {
        case_id: "box_bounds".into(),
        expected: format!("{:?}", OptimizeMethod::LBfgsB),
        actual: format!("{:?}", d2.method),
        pass: d2.method == OptimizeMethod::LBfgsB,
    });

    // 3. No gradient → NelderMead
    let p3 = OptCaspProblem {
        dimension: 3,
        variable_scale_ratio: 1.0,
        has_box_bounds: false,
        has_general_constraints: false,
        gradient_available: false,
        hessian_product_available: false,
    };
    let d3 = select_minimize_method(p3).expect("p3 ok");
    diffs.push(CaseDiff {
        case_id: "no_grad".into(),
        expected: format!("{:?}", OptimizeMethod::NelderMead),
        actual: format!("{:?}", d3.method),
        pass: d3.method == OptimizeMethod::NelderMead,
    });

    // 4. Hessian products available + ill-scaled small → trust-ncg, which uses the products
    //    (the BFGS-model trust-exact this used to pick never called hessp; frankenscipy-1ksfv.4)
    let p4 = OptCaspProblem {
        dimension: 3,
        variable_scale_ratio: 1.0e5,
        has_box_bounds: false,
        has_general_constraints: false,
        gradient_available: true,
        hessian_product_available: true,
    };
    let d4 = select_minimize_method(p4).expect("p4 ok");
    diffs.push(CaseDiff {
        case_id: "hessp_small_illscaled".into(),
        expected: format!("{:?}", OptimizeMethod::TrustNcg),
        actual: format!("{:?}", d4.method),
        pass: d4.method == OptimizeMethod::TrustNcg,
    });

    // 5. Hessian product, well-scaled → NewtonCg
    let p5 = OptCaspProblem {
        dimension: 8,
        variable_scale_ratio: 1.0,
        has_box_bounds: false,
        has_general_constraints: false,
        gradient_available: true,
        hessian_product_available: true,
    };
    let d5 = select_minimize_method(p5).expect("p5 ok");
    diffs.push(CaseDiff {
        case_id: "hessp_well_scaled".into(),
        expected: format!("{:?}", OptimizeMethod::NewtonCg),
        actual: format!("{:?}", d5.method),
        pass: d5.method == OptimizeMethod::NewtonCg,
    });

    // 6. Large dim without Hessian → ConjugateGradient
    let p6 = OptCaspProblem {
        dimension: 50,
        variable_scale_ratio: 1.0,
        has_box_bounds: false,
        has_general_constraints: false,
        gradient_available: true,
        hessian_product_available: false,
    };
    let d6 = select_minimize_method(p6).expect("p6 ok");
    diffs.push(CaseDiff {
        case_id: "large_dim_no_hessp".into(),
        expected: format!("{:?}", OptimizeMethod::ConjugateGradient),
        actual: format!("{:?}", d6.method),
        pass: d6.method == OptimizeMethod::ConjugateGradient,
    });

    // 7. Dimension 0 → error
    let p7 = OptCaspProblem {
        dimension: 0,
        variable_scale_ratio: 1.0,
        has_box_bounds: false,
        has_general_constraints: false,
        gradient_available: true,
        hessian_product_available: false,
    };
    let r7 = select_minimize_method(p7);
    diffs.push(CaseDiff {
        case_id: "dim_zero_err".into(),
        expected: "Err".into(),
        actual: if r7.is_err() {
            "Err".into()
        } else {
            "Ok".into()
        },
        pass: r7.is_err(),
    });

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_opt_select_minimize_method".into(),
        category: "fsci_opt::select_minimize_method branch coverage".into(),
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
                "casp mismatch: {} got {} (expected {})",
                d.case_id, d.actual, d.expected
            );
        }
    }

    assert!(
        all_pass,
        "casp branch coverage failed: {} cases",
        diffs.len(),
    );
}
