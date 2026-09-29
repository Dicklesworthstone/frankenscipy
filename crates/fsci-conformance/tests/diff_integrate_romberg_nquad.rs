#![forbid(unsafe_code)]
//! Cover fsci_integrate::{romberg, nquad}.
//!
//! Resolves [frankenscipy-lpt9x]. scipy.integrate.romberg was
//! removed in scipy >= 1.12, so we use closed-form expected values
//! for romberg. nquad still exists in scipy but we compare against
//! closed-form analytical integrals for multi-D test cases, which is
//! tighter than scipys oracle (both implementations chain quad and
//! drift similarly at the inner-loop tolerance).

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_integrate::{QuadOptions, nquad, romberg};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
/// One ledger arm per routine; a case's arm is its id prefix.
const ARMS: [&str; 2] = ["romberg", "nquad"];

fn arm_of(case_id: &str) -> &'static str {
    ARMS.into_iter()
        .find(|arm| case_id.starts_with(&format!("{arm}_")))
        .unwrap_or_else(|| panic!("case {case_id} names no declared arm"))
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    actual: f64,
    expected: f64,
    abs_diff: f64,
    rel_diff: f64,
    converged: bool,
    tol: f64,
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
    fs::create_dir_all(output_dir()).expect("create romberg_nquad diff dir");
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
fn diff_integrate_romberg_nquad() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut ledger = CompareLedger::new("diff_integrate_romberg_nquad", &ARMS);

    // Every case pushes exactly one CaseDiff, so `diffs` is the designed case list; the ledger
    // records whether each was compared.
    let push = |ledger: &mut CompareLedger,
                diffs: &mut Vec<CaseDiff>,
                case_id: &str,
                actual: f64,
                expected: f64,
                converged: bool,
                tol: f64| {
        let abs_diff = (actual - expected).abs();
        let denom = expected.abs().max(1.0e-300);
        let rel_diff = abs_diff / denom;
        let pass = converged && (abs_diff <= tol || rel_diff <= tol);
        let arm = arm_of(case_id);
        // A non-finite integral against a finite closed form is recorded as an fsci failure.
        if ledger
            .pair(arm, case_id, Some(expected), Some(actual))
            .is_some()
        {
            ledger.compared(arm, case_id, pass);
        }
        diffs.push(CaseDiff {
            case_id: case_id.into(),
            actual,
            expected,
            abs_diff,
            rel_diff,
            converged,
            tol,
            pass,
            note: String::new(),
        });
    };

    // === romberg: closed-form integrals ===
    // (a) constant: ∫_{0}^{2} 5 dx = 10
    {
        let r = romberg(|_| 5.0, 0.0, 2.0, 1.0e-10, 12);
        push(
            &mut ledger,
            &mut diffs,
            "romberg_constant",
            r.integral,
            10.0,
            r.converged,
            1e-10,
        );
    }
    // (b) linear: ∫_{0}^{3} (2x + 1) dx = x² + x |_0^3 = 9 + 3 = 12
    {
        let r = romberg(|x| 2.0 * x + 1.0, 0.0, 3.0, 1.0e-10, 12);
        push(
            &mut ledger,
            &mut diffs,
            "romberg_linear",
            r.integral,
            12.0,
            r.converged,
            1e-10,
        );
    }
    // (c) quadratic: ∫_{0}^{1} x² dx = 1/3
    {
        let r = romberg(|x| x * x, 0.0, 1.0, 1.0e-10, 12);
        push(
            &mut ledger,
            &mut diffs,
            "romberg_quadratic",
            r.integral,
            1.0 / 3.0,
            r.converged,
            1e-10,
        );
    }
    // (d) cubic: ∫_{0}^{2} x³ dx = 4
    {
        let r = romberg(|x| x * x * x, 0.0, 2.0, 1.0e-10, 12);
        push(
            &mut ledger,
            &mut diffs,
            "romberg_cubic",
            r.integral,
            4.0,
            r.converged,
            1e-10,
        );
    }
    // (e) sine: ∫_{0}^{π} sin(x) dx = 2
    {
        let r = romberg(|x: f64| x.sin(), 0.0, PI, 1.0e-10, 12);
        push(
            &mut ledger,
            &mut diffs,
            "romberg_sin",
            r.integral,
            2.0,
            r.converged,
            1e-9,
        );
    }
    // (f) cosine: ∫_{0}^{π} cos(x) dx = 0
    {
        let r = romberg(|x: f64| x.cos(), 0.0, PI, 1.0e-10, 12);
        push(
            &mut ledger,
            &mut diffs,
            "romberg_cos",
            r.integral,
            0.0,
            r.converged,
            1e-9,
        );
    }
    // (g) exp: ∫_{0}^{1} exp(x) dx = e - 1
    {
        let r = romberg(|x: f64| x.exp(), 0.0, 1.0, 1.0e-10, 12);
        let expected = std::f64::consts::E - 1.0;
        push(
            &mut ledger,
            &mut diffs,
            "romberg_exp",
            r.integral,
            expected,
            r.converged,
            1e-9,
        );
    }
    // (h) NaN tolerance returns non-converged with NaN integral
    {
        let r = romberg(|_| 1.0, 0.0, 1.0, f64::NAN, 12);
        let pass = !r.converged && r.integral.is_nan();
        ledger.compared("romberg", "romberg_nan_tol_errors", pass);
        diffs.push(CaseDiff {
            case_id: "romberg_nan_tol_errors".into(),
            actual: r.integral,
            expected: f64::NAN,
            abs_diff: 0.0,
            rel_diff: 0.0,
            converged: r.converged,
            tol: 0.0,
            pass,
            note: format!(
                "NaN tol → converged={} integral={}",
                r.converged, r.integral
            ),
        });
    }

    // === nquad: multi-D closed-form integrals ===
    let opts = QuadOptions::default();

    // 2D: ∫∫_{[0,1]^2} 1 dxdy = 1
    {
        let r = nquad(|_x| 1.0, &[(0.0, 1.0), (0.0, 1.0)], opts).expect("nquad const");
        push(
            &mut ledger,
            &mut diffs,
            "nquad_2d_const",
            r.integral,
            1.0,
            r.converged,
            1e-8,
        );
    }
    // 2D: ∫∫_{[0,1]^2} (x + y) dxdy = 1
    // ∫₀¹ ∫₀¹ x + y dxdy = 0.5 + 0.5 = 1
    {
        let r = nquad(|v| v[0] + v[1], &[(0.0, 1.0), (0.0, 1.0)], opts).expect("nquad sum");
        push(
            &mut ledger,
            &mut diffs,
            "nquad_2d_xpy",
            r.integral,
            1.0,
            r.converged,
            1e-8,
        );
    }
    // 2D: ∫∫_{[0,π]×[0,π]} sin(x)*sin(y) dxdy = 2 * 2 = 4
    {
        let r = nquad(
            |v: &[f64]| v[0].sin() * v[1].sin(),
            &[(0.0, PI), (0.0, PI)],
            opts,
        )
        .expect("nquad sin*sin");
        push(
            &mut ledger,
            &mut diffs,
            "nquad_2d_sin_sin",
            r.integral,
            4.0,
            r.converged,
            1e-7,
        );
    }
    // 3D: ∫∫∫_{[0,1]^3} xyz dxdydz = (1/2)^3 = 1/8
    {
        let r = nquad(
            |v| v[0] * v[1] * v[2],
            &[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            opts,
        )
        .expect("nquad xyz");
        push(
            &mut ledger,
            &mut diffs,
            "nquad_3d_xyz",
            r.integral,
            1.0 / 8.0,
            r.converged,
            1e-7,
        );
    }
    // 0D (empty ranges) returns func() with neval=1
    {
        let r = nquad(|_| 7.5, &[], opts).expect("nquad 0d");
        push(
            &mut ledger,
            &mut diffs,
            "nquad_0d_const",
            r.integral,
            7.5,
            r.converged,
            1e-12,
        );
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_integrate_romberg_nquad".into(),
        category: "fsci_integrate::{romberg, nquad} closed-form coverage".into(),
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
            eprintln!(
                "romberg/nquad mismatch: {} actual={} expected={} abs={} rel={} converged={} note={}",
                d.case_id, d.actual, d.expected, d.abs_diff, d.rel_diff, d.converged, d.note
            );
        }
    }

    assert!(
        all_pass,
        "romberg/nquad coverage failed: {} cases",
        diffs.len()
    );
    // romberg has 8 cases and nquad 5; each arm must compare all of its own.
    let min_per_arm = ARMS
        .iter()
        .map(|arm| diffs.iter().filter(|d| arm_of(&d.case_id) == *arm).count())
        .min()
        .expect("ARMS is non-empty");
    ledger.finish(min_per_arm);
}
