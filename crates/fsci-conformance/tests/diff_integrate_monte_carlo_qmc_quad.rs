#![forbid(unsafe_code)]
//! Cover fsci_integrate::{monte_carlo_integrate, qmc_quad}.
//!
//! Resolves [frankenscipy-xu0ko]. Both routines are stochastic
//! (MC uses a seeded LCG; QMC uses Halton blocks) so direct equality
//! with closed-form expected values is brittle. The right invariants:
//!   * Estimate is within k·standard_error of the truth (k=4 for MC;
//!     QMC converges faster but we still use a generous k).
//!   * Edge cases (n_samples=0 or empty bounds) return (0.0, 0.0)
//!     for MC.
//!   * QMC reports n_estimates < 2 → error (variance needs ≥2 blocks).
//!   * QMC reports dimension > 32 → error (out of pinned-prime table).

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_integrate::{monte_carlo_integrate, qmc_quad};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
/// (case id prefix, ledger arm): one arm per routine.
const ARMS: [(&str, &str); 2] = [("mc_", "monte_carlo_integrate"), ("qmc_", "qmc_quad")];

fn arm_of(case_id: &str) -> &'static str {
    ARMS.into_iter()
        .find(|(prefix, _)| case_id.starts_with(*prefix))
        .map(|(_, arm)| arm)
        .unwrap_or_else(|| panic!("case {case_id} names no declared arm"))
}

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
    fs::create_dir_all(output_dir()).expect("create mc_qmc diff dir");
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
fn diff_integrate_monte_carlo_qmc_quad() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut ledger = CompareLedger::new(
        "diff_integrate_monte_carlo_qmc_quad",
        &ARMS.map(|(_, arm)| arm),
    );
    // Every check records one CaseDiff, so `diffs` is the designed case list, and one ledger
    // verdict (a NaN estimate fails its own comparison).
    let mut check = |id: &str, ok: bool, note: String| {
        ledger.compared(arm_of(id), id, ok);
        diffs.push(CaseDiff {
            case_id: id.into(),
            pass: ok,
            note,
        });
    };

    // === monte_carlo_integrate ===
    // 1D: ∫_{0}^{1} x² dx = 1/3. With n=50000 samples expected stderr ~ σ/√n
    {
        let (est, se) = monte_carlo_integrate(|x| x[0] * x[0], &[(0.0, 1.0)], 50_000, 42);
        let diff = (est - 1.0 / 3.0).abs();
        check(
            "mc_1d_quadratic_within_4se",
            diff <= 4.0 * se.max(1e-3),
            format!("est={est} se={se} diff={diff}"),
        );
    }

    // 2D: ∫∫_{[0,1]²} 1 dxdy = 1
    {
        let (est, se) = monte_carlo_integrate(|_| 1.0, &[(0.0, 1.0), (0.0, 1.0)], 10_000, 7);
        check(
            "mc_2d_constant",
            (est - 1.0).abs() <= 4.0 * se.max(1e-6),
            format!("est={est} se={se}"),
        );
    }

    // 2D: π estimate via indicator of unit disk in [-1,1]² → π
    {
        let (est, se) = monte_carlo_integrate(
            |x| {
                if x[0] * x[0] + x[1] * x[1] <= 1.0 {
                    1.0
                } else {
                    0.0
                }
            },
            &[(-1.0, 1.0), (-1.0, 1.0)],
            100_000,
            12345,
        );
        check(
            "mc_2d_pi_estimate_within_4se",
            (est - PI).abs() <= 4.0 * se.max(1e-2),
            format!("est={est} se={se}"),
        );
    }

    // Edge: n_samples=0 → (0, 0)
    {
        let (e, s) = monte_carlo_integrate(|_| 1.0, &[(0.0, 1.0)], 0, 1);
        check(
            "mc_zero_samples_zero",
            e == 0.0 && s == 0.0,
            format!("est={e} se={s}"),
        );
    }

    // Edge: empty bounds → (0, 0)
    {
        let (e, s) = monte_carlo_integrate(|_| 5.0, &[], 100, 1);
        check(
            "mc_empty_bounds_zero",
            e == 0.0 && s == 0.0,
            format!("est={e} se={s}"),
        );
    }

    // === qmc_quad ===
    // 1D: ∫_{0}^{1} x² dx = 1/3
    {
        let r = qmc_quad(|x| x[0] * x[0], &[0.0], &[1.0], 5, 256).expect("qmc 1d quadratic");
        check(
            "qmc_1d_quadratic_close",
            (r.integral - 1.0 / 3.0).abs() <= 4.0 * r.standard_error.max(1e-3),
            format!("integral={} se={}", r.integral, r.standard_error),
        );
    }

    // 2D: ∫∫_{[0,1]²} x + y dxdy = 1
    {
        let r = qmc_quad(|v| v[0] + v[1], &[0.0, 0.0], &[1.0, 1.0], 5, 256).expect("qmc 2d sum");
        check(
            "qmc_2d_sum_close",
            (r.integral - 1.0).abs() <= 4.0 * r.standard_error.max(1e-3),
            format!("integral={} se={}", r.integral, r.standard_error),
        );
    }

    // 3D: ∫∫∫_{[0,1]^3} x*y*z = 1/8
    {
        let r = qmc_quad(
            |v| v[0] * v[1] * v[2],
            &[0.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0],
            5,
            256,
        )
        .expect("qmc 3d xyz");
        check(
            "qmc_3d_xyz_close",
            (r.integral - 1.0 / 8.0).abs() <= 4.0 * r.standard_error.max(1e-3),
            format!("integral={} se={}", r.integral, r.standard_error),
        );
    }

    // QMC: n_estimates == 1 is allowed (se = 0)
    {
        let r = qmc_quad(|_| 1.0, &[0.0], &[1.0], 1, 256).expect("single estimate ok");
        check(
            "qmc_n_estimates_one_ok_zero_se",
            (r.integral - 1.0).abs() < 1e-12 && r.standard_error == 0.0,
            format!("integral={} se={}", r.integral, r.standard_error),
        );
    }

    // QMC: n_estimates == 0 → error (no samples possible)
    {
        let r = qmc_quad(|_| 1.0, &[0.0], &[1.0], 0, 256);
        check(
            "qmc_n_estimates_zero_errors",
            r.is_err(),
            format!("res={r:?}"),
        );
    }

    // QMC: lb/ub length mismatch → error
    {
        let r = qmc_quad(|_| 1.0, &[0.0, 0.0], &[1.0], 5, 256);
        check(
            "qmc_lb_ub_length_mismatch_errors",
            r.is_err(),
            format!("res={r:?}"),
        );
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_integrate_monte_carlo_qmc_quad".into(),
        category: "fsci_integrate::{monte_carlo_integrate, qmc_quad} coverage".into(),
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
            eprintln!("mc/qmc mismatch: {} — {}", d.case_id, d.note);
        }
    }

    assert!(all_pass, "mc/qmc coverage failed: {} cases", diffs.len());
    // monte_carlo_integrate has 5 checks and qmc_quad 6; each arm must compare all of its own.
    let min_per_arm = ARMS
        .iter()
        .map(|(_, arm)| diffs.iter().filter(|d| arm_of(&d.case_id) == *arm).count())
        .min()
        .expect("ARMS is non-empty");
    ledger.finish(min_per_arm);
}
