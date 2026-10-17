#![forbid(unsafe_code)]
//! Branch-coverage for fsci_special::select_hypergeometric_branch.
//!
//! Resolves [frankenscipy-fm05w]. The CASP selector routes each
//! hypergeometric call (0F1/1F1/2F1) to the appropriate evaluation
//! branch (DirectSeries / KummerTransform / TerminatingPolynomial /
//! GaussSummation / PfaffTransform / DivergentAtUnitArgument /
//! ParameterGuard / UnsupportedAnalyticContinuation). This harness
//! exercises each documented branch once and verifies the returned
//! HypergeometricBranch matches expectation.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::{HyperSolverPortfolio, RuntimeMode};
use fsci_special::{
    HyperCaspProblem, HypergeometricBranch, select_hypergeometric_branch,
    select_hypergeometric_branch_with_casp,
};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    expected: String,
    actual: String,
    pass: bool,
    note: String,
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
    fs::create_dir_all(output_dir()).expect("create hyper_casp diff dir");
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
fn diff_special_select_hypergeometric_branch_casp() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut probe =
        |id: &str, problem: HyperCaspProblem, mode: RuntimeMode, expected: HypergeometricBranch| {
            let actual_res = select_hypergeometric_branch(problem, mode);
            let (actual_str, pass, note) = match actual_res {
                Ok(d) => (
                    format!("{:?}", d.branch),
                    d.branch == expected,
                    String::new(),
                ),
                Err(e) => ("Err".into(), false, format!("err: {e:?}")),
            };
            diffs.push(CaseDiff {
                case_id: id.into(),
                expected: format!("{expected:?}"),
                actual: actual_str,
                pass,
                note,
            });
        };

    // === Hyp0f1 ===
    // Small z_abs → DirectSeries
    probe(
        "hyp0f1_direct_series_small_z",
        HyperCaspProblem::hyp0f1(2.0, 1.0, 1.0e-10),
        RuntimeMode::Strict,
        HypergeometricBranch::DirectSeries,
    );
    // Large negative z → AsymptoticExpansion (the oscillatory J side)
    probe(
        "hyp0f1_asymptotic_large_negative_z",
        HyperCaspProblem::hyp0f1(2.0, -100.0, 1.0e-10),
        RuntimeMode::Strict,
        HypergeometricBranch::AsymptoticExpansion,
    );
    // Large positive z → DirectSeries: past n = -b its terms have one sign, so it cannot
    // cancel; the ten-term I_nu asymptotic that used to serve it was 5.2e-6 off at
    // (9.59, 55.4) (frankenscipy-n5ub1).
    probe(
        "hyp0f1_direct_series_large_positive_z",
        HyperCaspProblem::hyp0f1(2.0, 100.0, 1.0e-10),
        RuntimeMode::Strict,
        HypergeometricBranch::DirectSeries,
    );

    // === Hyp1f1 ===
    // Moderate z → DirectSeries
    probe(
        "hyp1f1_direct_moderate_z",
        HyperCaspProblem::hyp1f1(1.0, 2.0, 0.5, 1.0e-10),
        RuntimeMode::Strict,
        HypergeometricBranch::DirectSeries,
    );
    // Large negative z → KummerTransform
    probe(
        "hyp1f1_kummer_large_negative_z",
        HyperCaspProblem::hyp1f1(1.0, 2.0, -50.0, 1.0e-10),
        RuntimeMode::Strict,
        HypergeometricBranch::KummerTransform,
    );

    // === Hyp2f1 ===
    // Terminating polynomial: a is non-positive integer
    probe(
        "hyp2f1_terminating_neg_int_a",
        HyperCaspProblem::hyp2f1(-3.0, 1.5, 2.0, 0.5, 1.0e-10),
        RuntimeMode::Strict,
        HypergeometricBranch::TerminatingPolynomial,
    );
    // Direct series inside unit disk with z > 0
    probe(
        "hyp2f1_direct_in_disk_positive",
        HyperCaspProblem::hyp2f1(1.0, 1.5, 2.0, 0.5, 1.0e-10),
        RuntimeMode::Strict,
        HypergeometricBranch::DirectSeries,
    );
    // Pfaff transform inside unit disk with z < 0
    probe(
        "hyp2f1_pfaff_in_disk_negative",
        HyperCaspProblem::hyp2f1(1.0, 1.5, 2.0, -0.5, 1.0e-10),
        RuntimeMode::Strict,
        HypergeometricBranch::PfaffTransform,
    );
    // Gauss summation at z = 1 with c - a - b > 0
    probe(
        "hyp2f1_gauss_z_equal_one_converges",
        HyperCaspProblem::hyp2f1(0.5, 0.5, 2.0, 1.0, 1.0e-10),
        RuntimeMode::Strict,
        HypergeometricBranch::GaussSummation,
    );
    // Divergent at z = 1 when c - a - b <= 0
    probe(
        "hyp2f1_divergent_at_unit",
        HyperCaspProblem::hyp2f1(1.0, 2.0, 2.0, 1.0, 1.0e-10),
        RuntimeMode::Strict,
        HypergeometricBranch::DivergentAtUnitArgument,
    );
    // Pfaff transform outside unit disk with z < 0
    probe(
        "hyp2f1_pfaff_outside_disk_negative",
        HyperCaspProblem::hyp2f1(1.0, 1.5, 2.0, -5.0, 1.0e-10),
        RuntimeMode::Strict,
        HypergeometricBranch::PfaffTransform,
    );

    // === ParameterGuard branch: parameter_stability_margin ≤ precision_target ===
    // Pick b very close to a nonpositive integer pole (b = -3 + 1e-12)
    // so that lower_parameter_stability_margin(b) ≈ 1e-12 < precision_target = 1e-10
    {
        let mut p = HyperCaspProblem::hyp1f1(1.0, -3.0 + 1.0e-12, 0.5, 1.0e-10);
        // Force the margin field to a value below precision_target — the
        // helper uses parameter_stability_margin directly, so we set it
        // explicitly to guarantee the branch fires.
        p.parameter_stability_margin = 1.0e-12;
        probe(
            "parameter_guard_low_margin",
            p,
            RuntimeMode::Hardened,
            HypergeometricBranch::ParameterGuard,
        );
        // Strict evaluates a 1F1 lower parameter next to (not on) a pole, as SciPy does
        // (frankenscipy-tbhnl): the guard fires in Hardened only.
        probe(
            "parameter_near_pole_strict_evaluates",
            p,
            RuntimeMode::Strict,
            HypergeometricBranch::DirectSeries,
        );
    }

    // === Strict-mode non-finite input → UnsupportedAnalyticContinuation ===
    // (Hardened mode would error; Strict delegates)
    {
        let mut p = HyperCaspProblem::hyp1f1(1.0, 2.0, f64::NAN, 1.0e-10);
        // Keep margin healthy so the non-finite path is the trigger
        p.parameter_stability_margin = 1.0;
        probe(
            "unsupported_strict_non_finite_z",
            p,
            RuntimeMode::Strict,
            HypergeometricBranch::UnsupportedAnalyticContinuation,
        );
    }

    // === Hardened-mode non-finite input → error (no decision) ===
    {
        let mut p = HyperCaspProblem::hyp1f1(1.0, 2.0, f64::NAN, 1.0e-10);
        p.parameter_stability_margin = 1.0;
        let result = select_hypergeometric_branch(p, RuntimeMode::Hardened);
        let pass = result.is_err();
        diffs.push(CaseDiff {
            case_id: "hardened_non_finite_z_errors".into(),
            expected: "Err".into(),
            actual: if pass { "Err".into() } else { "Ok".into() },
            pass,
            note: String::new(),
        });
    }

    // === Invalid precision_target → DomainError ===
    {
        let p = HyperCaspProblem::hyp1f1(1.0, 2.0, 0.5, -1.0);
        let result = select_hypergeometric_branch(p, RuntimeMode::Strict);
        let pass = result.is_err();
        diffs.push(CaseDiff {
            case_id: "invalid_precision_target_errors".into(),
            expected: "Err".into(),
            actual: if pass { "Err".into() } else { "Ok".into() },
            pass,
            note: String::new(),
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_special_select_hypergeometric_branch_casp".into(),
        category: "fsci_special::select_hypergeometric_branch CASP routing".into(),
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
                "hyper_casp mismatch: {} expected={} actual={} {}",
                d.case_id, d.expected, d.actual, d.note
            );
        }
    }

    assert!(
        all_pass,
        "hypergeometric CASP branch coverage failed: {} cases",
        diffs.len(),
    );
}

#[test]
fn diff_special_select_hypergeometric_branch_with_casp_portfolio() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut portfolio = HyperSolverPortfolio::new(RuntimeMode::Strict, 128);

    let mut probe_casp = |id: &str,
                          problem: HyperCaspProblem,
                          mode: RuntimeMode,
                          portfolio: &mut HyperSolverPortfolio,
                          expected: HypergeometricBranch| {
        let actual_res = select_hypergeometric_branch_with_casp(problem, mode, portfolio);
        let (actual_str, pass, note) = match actual_res {
            Ok(d) => (
                format!("{:?}", d.branch),
                d.branch == expected,
                String::new(),
            ),
            Err(e) => ("Err".into(), false, format!("err: {e:?}")),
        };
        diffs.push(CaseDiff {
            case_id: id.into(),
            expected: format!("{expected:?}"),
            actual: actual_str,
            pass,
            note,
        });
    };

    // 1. Direct series near zero
    probe_casp(
        "casp_hyp2f1_direct_series",
        HyperCaspProblem::hyp2f1(1.0, 2.0, 3.0, 0.2, 1.0e-14),
        RuntimeMode::Strict,
        &mut portfolio,
        HypergeometricBranch::DirectSeries,
    );

    // 2. Kummer transform for negative real confluent argument (1F1)
    probe_casp(
        "casp_hyp1f1_kummer_transform",
        HyperCaspProblem::hyp1f1(1.0, 2.0, -3.0, 1.0e-14),
        RuntimeMode::Strict,
        &mut portfolio,
        HypergeometricBranch::KummerTransform,
    );

    // 3. Pfaff transform for negative real 2F1 argument (never Kummer)
    probe_casp(
        "casp_hyp2f1_pfaff_transform",
        HyperCaspProblem::hyp2f1(1.0, 2.0, 3.0, -3.0, 1.0e-14),
        RuntimeMode::Strict,
        &mut portfolio,
        HypergeometricBranch::PfaffTransform,
    );

    // 4. Asymptotic expansion for large 1F1 argument
    probe_casp(
        "casp_hyp1f1_asymptotic",
        HyperCaspProblem::hyp1f1(1.0, 2.0, 250.0, 1.0e-14),
        RuntimeMode::Strict,
        &mut portfolio,
        HypergeometricBranch::AsymptoticExpansion,
    );

    // 5. Real branch cut for 2F1 z > 1 (never 1F1 asymptotic)
    probe_casp(
        "casp_hyp2f1_real_branch_cut",
        HyperCaspProblem::hyp2f1(1.0, 2.0, 3.0, 25.0, 1.0e-14),
        RuntimeMode::Strict,
        &mut portfolio,
        HypergeometricBranch::RealBranchCutInfinity,
    );

    // 6. Divergent at z = 1 when c - a - b <= 0 (hyp2f1(1, 2, 3, 1))
    probe_casp(
        "casp_hyp2f1_unit_divergent",
        HyperCaspProblem::hyp2f1(1.0, 2.0, 3.0, 1.0, 1.0e-14),
        RuntimeMode::Strict,
        &mut portfolio,
        HypergeometricBranch::DivergentAtUnitArgument,
    );

    // 7. Gauss summation at z = 1 when c - a - b > 0
    probe_casp(
        "casp_hyp2f1_unit_gauss",
        HyperCaspProblem::hyp2f1(0.5, 0.5, 2.0, 1.0, 1.0e-14),
        RuntimeMode::Strict,
        &mut portfolio,
        HypergeometricBranch::GaussSummation,
    );

    // 8. Parameter guard near lower-parameter pole
    probe_casp(
        "casp_hyp2f1_boundary_near_pole",
        HyperCaspProblem::hyp2f1(1.0, 2.0, -2.0 + 1.0e-15, 0.2, 1.0e-14),
        RuntimeMode::Strict,
        &mut portfolio,
        HypergeometricBranch::ParameterGuard,
    );

    // 9. Negative-case latch check: 50 consecutive benign calls must never latch GuardedFallback
    let benign = HyperCaspProblem::hyp2f1(1.0, 2.0, 3.0, 0.2, 1.0e-14);
    for idx in 0..50 {
        probe_casp(
            &format!("casp_benign_non_latch_{idx}"),
            benign,
            RuntimeMode::Strict,
            &mut portfolio,
            HypergeometricBranch::DirectSeries,
        );
    }
    assert!(
        !portfolio.calibrator().should_fallback(),
        "50 benign calls must not trigger conformal fallback"
    );

    assert_eq!(
        portfolio.evidence_len(),
        58,
        "portfolio must accumulate 58 evidence entries"
    );

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_special_select_hypergeometric_branch_with_casp_portfolio".into(),
        category: "fsci_special::select_hypergeometric_branch_with_casp CASP portfolio routing"
            .into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    assert!(all_pass, "all casp portfolio routing cases must pass");
}

fn ulp_distance(actual: f64, reference: f64) -> u64 {
    if actual == reference {
        return 0;
    }
    if !actual.is_finite() || !reference.is_finite() {
        return u64::MAX;
    }
    let scale = reference.abs().max(f64::MIN_POSITIVE);
    let ulp = scale * f64::EPSILON;
    ((actual - reference).abs() / ulp).round() as u64
}

/// Legacy `z in [0.9, 1.0)` formula before `frankenscipy-7tb8d.10` near-integer pole pairing,
/// used to measure the before/after accuracy delta on the `|z| in [0.9, 1.1]` band in the same
/// test invocation.
fn legacy_hyp2f1_unit_band(a: f64, b: f64, c: f64, z: f64) -> f64 {
    use fsci_special::{gamma, hyp2f1, SpecialTensor};
    if (0.9..1.0).contains(&z) {
        let cab = c - a - b;
        if (cab - cab.round()).abs() > 1e-6 {
            let g = |x: f64| match gamma(&SpecialTensor::RealScalar(x), RuntimeMode::Strict) {
                Ok(SpecialTensor::RealScalar(v)) => v,
                _ => f64::NAN,
            };
            let omz = 1.0 - z;
            let pref1 = g(c) * g(cab) / (g(c - a) * g(c - b));
            let pref2 = g(c) * g(-cab) / (g(a) * g(b));
            let mut s1 = 1.0_f64;
            let mut t1 = 1.0_f64;
            let mut s2 = 1.0_f64;
            let mut t2 = 1.0_f64;
            for n in 0..500 {
                let nf = n as f64;
                t1 *= (a + nf) * (b + nf) * omz / ((1.0 - cab + nf) * (nf + 1.0));
                t2 *= (c - a + nf) * (c - b + nf) * omz / ((1.0 + cab + nf) * (nf + 1.0));
                s1 += t1;
                s2 += t2;
            }
            return pref1 * s1 + omz.powf(cab) * pref2 * s2;
        }
    }
    match hyp2f1(
        &SpecialTensor::RealScalar(a),
        &SpecialTensor::RealScalar(b),
        &SpecialTensor::RealScalar(c),
        &SpecialTensor::RealScalar(z),
        RuntimeMode::Strict,
    ) {
        Ok(SpecialTensor::RealScalar(v)) => v,
        _ => f64::NAN,
    }
}

#[test]
fn diff_special_hyper_casp_2000_point_mpmath_accuracy_grid() {
    use fsci_special::{hyp0f1, hyp1f1, hyp2f1, SpecialTensor};
    use std::process::Command;

    let mut script = String::from(
        "import json, mpmath as mp\nmp.mp.dps = 50\nout = []\n",
    );

    // 500 points of hyp0f1(b, z)
    let mut hyp0f1_pts = Vec::with_capacity(500);
    for i in 0..25 {
        let b = 0.35 + (i as f64) * 0.65;
        for j in 0..20 {
            let z = -35.0 + (j as f64) * 4.5;
            hyp0f1_pts.push((b, z));
        }
    }

    // 500 points of hyp1f1(a, b, z)
    let mut hyp1f1_pts = Vec::with_capacity(500);
    for i in 0..25 {
        let a = -2.25 + (i as f64) * 0.45;
        let b = 0.65 + ((i % 7) as f64) * 0.85;
        for j in 0..20 {
            let z = -25.0 + (j as f64) * 3.0;
            hyp1f1_pts.push((a, b, z));
        }
    }

    // 1000 points of hyp2f1(a, b, c, z), including 400 points in the |z| in [0.9, 1.1] band
    // with near-integer c - a - b offsets (where legacy 15.8.4 pole cancellation occurred).
    let mut hyp2f1_pts = Vec::with_capacity(1000);
    for i in 0..30 {
        let a = -1.35 + (i as f64) * 0.25;
        let b = 0.45 + ((i % 6) as f64) * 0.55;
        let c = 1.25 + ((i % 5) as f64) * 0.75;
        for j in 0..20 {
            let z = -4.5 + (j as f64) * 0.26;
            hyp2f1_pts.push((a, b, c, z));
        }
    }
    let band_zs = [
        -1.08, -1.04, -1.01, -0.98, -0.95, -0.91, 0.905, 0.925, 0.95, 0.975,
    ];
    let eps_offsets = [-1.0e-3, -1.0e-4, 1.0e-4, 1.0e-3, 0.15];
    for m_idx in 0..8 {
        let a = 0.35 + (m_idx as f64) * 0.3;
        let b = 0.65 + ((m_idx % 4) as f64) * 0.4;
        let m = (m_idx % 3) as f64;
        for &eps in &eps_offsets {
            let c = a + b + m + eps;
            for &z in &band_zs {
                hyp2f1_pts.push((a, b, c, z));
            }
        }
    }
    assert_eq!(hyp0f1_pts.len() + hyp1f1_pts.len() + hyp2f1_pts.len(), 2000);

    let payload = serde_json::json!({
        "hyp0f1": hyp0f1_pts,
        "hyp1f1": hyp1f1_pts,
        "hyp2f1": hyp2f1_pts,
    });
    script.push_str(&format!(
        "data = json.loads({:?})\n",
        payload.to_string()
    ));
    script.push_str(
        "r0 = [float(mp.hyp0f1(b, z)) for b, z in data['hyp0f1']]\n\
         r1 = [float(mp.hyp1f1(a, b, z)) for a, b, z in data['hyp1f1']]\n\
         r2 = [float(mp.hyp2f1(a, b, c, z)) for a, b, c, z in data['hyp2f1']]\n\
         print(json.dumps({'hyp0f1': r0, 'hyp1f1': r1, 'hyp2f1': r2}))\n",
    );

    let output = Command::new("/home/ubuntu/.local/bin/python3.13")
        .arg("-c")
        .arg(&script)
        .output()
        .expect("invoke python3.13 with mpmath");
    assert!(
        output.status.success(),
        "mpmath script failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let refs: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("parse mpmath json");

    let mut max_ulp_0f1 = 0_u64;
    for ((b, z), r_val) in hyp0f1_pts
        .iter()
        .zip(refs["hyp0f1"].as_array().expect("hyp0f1 array"))
    {
        let expected = r_val.as_f64().expect("f64");
        let actual = match hyp0f1(
            &SpecialTensor::RealScalar(*b),
            &SpecialTensor::RealScalar(*z),
            RuntimeMode::Strict,
        )
        .expect("hyp0f1")
        {
            SpecialTensor::RealScalar(v) => v,
            other => panic!("unexpected tensor {other:?}"),
        };
        max_ulp_0f1 = max_ulp_0f1.max(ulp_distance(actual, expected));
    }

    let mut max_ulp_1f1 = 0_u64;
    for ((a, b, z), r_val) in hyp1f1_pts
        .iter()
        .zip(refs["hyp1f1"].as_array().expect("hyp1f1 array"))
    {
        let expected = r_val.as_f64().expect("f64");
        let actual = match hyp1f1(
            &SpecialTensor::RealScalar(*a),
            &SpecialTensor::RealScalar(*b),
            &SpecialTensor::RealScalar(*z),
            RuntimeMode::Strict,
        )
        .expect("hyp1f1")
        {
            SpecialTensor::RealScalar(v) => v,
            other => panic!("unexpected tensor {other:?}"),
        };
        max_ulp_1f1 = max_ulp_1f1.max(ulp_distance(actual, expected));
    }

    let mut max_ulp_2f1_before = 0_u64;
    let mut max_ulp_2f1_after = 0_u64;
    let mut max_ulp_band_before = 0_u64;
    let mut max_ulp_band_after = 0_u64;
    for ((a, b, c, z), r_val) in hyp2f1_pts
        .iter()
        .zip(refs["hyp2f1"].as_array().expect("hyp2f1 array"))
    {
        let expected = r_val.as_f64().expect("f64");
        let actual = match hyp2f1(
            &SpecialTensor::RealScalar(*a),
            &SpecialTensor::RealScalar(*b),
            &SpecialTensor::RealScalar(*c),
            &SpecialTensor::RealScalar(*z),
            RuntimeMode::Strict,
        )
        .expect("hyp2f1")
        {
            SpecialTensor::RealScalar(v) => v,
            other => panic!("unexpected tensor {other:?}"),
        };
        let before = legacy_hyp2f1_unit_band(*a, *b, *c, *z);
        let ulp_before = ulp_distance(before, expected);
        let ulp_after = ulp_distance(actual, expected);
        max_ulp_2f1_before = max_ulp_2f1_before.max(ulp_before);
        max_ulp_2f1_after = max_ulp_2f1_after.max(ulp_after);
        if (0.9..=1.1).contains(&z.abs()) {
            max_ulp_band_before = max_ulp_band_before.max(ulp_before);
            max_ulp_band_after = max_ulp_band_after.max(ulp_after);
        }
    }

    println!(
        "2000-point mpmath (dps=50) grid: hyp0f1 max_ulp={max_ulp_0f1}, hyp1f1 max_ulp={max_ulp_1f1}, \
         hyp2f1 max_ulp before={max_ulp_2f1_before} after={max_ulp_2f1_after}, \
         |z| in [0.9, 1.1] band max_ulp before={max_ulp_band_before} after={max_ulp_band_after}"
    );

    assert!(
        max_ulp_2f1_after <= max_ulp_2f1_before,
        "hyp2f1 max ULP regressed: after={max_ulp_2f1_after} > before={max_ulp_2f1_before}"
    );
    assert!(
        max_ulp_band_after < max_ulp_band_before,
        "|z| in [0.9, 1.1] band must be strictly better: after={max_ulp_band_after} vs before={max_ulp_band_before}"
    );
}
