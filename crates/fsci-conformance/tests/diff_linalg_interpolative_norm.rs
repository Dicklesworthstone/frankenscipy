#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `fsci_linalg::interpolative::estimate_spectral_norm`
//! and `estimate_spectral_norm_diff`, against `scipy.linalg.interpolative`.
//!
//! ## Why these two and not `estimate_rank`
//!
//! All three are randomized, but they are not randomized in the same way, and only two of them
//! estimate a well-defined number.
//!
//! The norm estimators converge to `σ₁`, which does not depend on the random start. Measured on
//! scipy 1.17.1: repeated calls on the same matrix agree to twelve decimals and match the SVD's
//! `σ₁` to machine precision. That is a quantity two independent implementations can be held to.
//!
//! `estimate_rank` is different. It returned `n` for EVERY case tried on small matrices —
//! including an exactly rank-1 8×6 matrix at `eps = 1e-2` — and on larger ones it returns a
//! coarse over-estimate that is INSENSITIVE to `eps`: 12 for a true rank of 5, 17 for 10, 9 for
//! 2, identical at `eps = 1e-3` and `eps = 1e-8`. It is a block-granularity upper bound, so its
//! exact integer is a function of SciPy's internal block size and Fortran RNG. Reproducing that
//! number would be cloning an implementation detail rather than implementing a capability, and
//! a test pinning "12" would be pinning a block size. It is deliberately not implemented.
//!
//! ## How the tolerances are derived rather than chosen
//!
//! The power method's error falls as `(σ₂/σ₁)^(2·its)`, so accuracy is set by the spectral GAP,
//! not by the iteration count alone. A single fixed tolerance would therefore be either too
//! loose for well-separated matrices or unattainable for near-degenerate ones. Each case is
//! judged against its own predicted bound, computed from the gap the oracle reports, with a
//! floor at machine precision. Measured on scipy: a gap of 0.183 gives agreement to 1.5e-16,
//! while a gap of 0.90 gives 3.2e-7 — both exactly as the bound predicts.
//!
//! One assertion needs no tolerance at all and is applied to BOTH arms: the estimate can never
//! EXCEED the true norm. That is a property of the method, and it catches a wrong normalisation
//! even when the number looks reasonable.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{SystemTime, UNIX_EPOCH};

use fsci_linalg::interpolative::{
    DEFAULT_SPECTRAL_NORM_ITERATIONS, estimate_spectral_norm, estimate_spectral_norm_diff,
};
use serde::{Deserialize, Serialize};

use fsci_runtime::scipy_incumbent::ScipyIncumbent;

/// Submodules the oracle actually uses.
const SCIPY_REQUIRED_MODULES: &[&str] = &["scipy.linalg.interpolative"];

/// The live-SciPy incumbent this test compares against, resolved once and PROVEN by running
/// the import rather than by a name resolving on `PATH`.
///
/// These two test targets could not BUILD until frankenscipy-a623e, so they were invisible to
/// the sweep that retired the bare-`python3` spawn everywhere else (frankenscipy-m5s54). On
/// `thinkstation1` a bare `python3` is 3.14 with no SciPy, so this oracle would have skipped
/// on the one host that actually carries the pinned incumbent -- while `FSCI_REQUIRE_SCIPY_ORACLE=1`
/// turned that same skip into a hard failure for a reason unrelated to the code under test.
///
/// `None` when the host has no importable SciPy, which is the honest state on most rch
/// workers and is what the existing skip/assert pair below is written to handle.
fn incumbent() -> Option<&'static ScipyIncumbent> {
    static INCUMBENT: std::sync::OnceLock<Option<ScipyIncumbent>> = std::sync::OnceLock::new();
    INCUMBENT
        .get_or_init(
            || match ScipyIncumbent::resolve_with(&[], SCIPY_REQUIRED_MODULES) {
                Ok(found) => {
                    eprintln!("{}", found.provenance_line());
                    Some(found)
                }
                Err(error) => {
                    eprintln!("scipy_incumbent: unresolved -- {error}");
                    None
                }
            },
        )
        .as_ref()
}

const PACKET_ID: &str = "FSCI-P2C-007";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

/// Floor for the derived tolerances: no comparison of two independently-rounded reductions is
/// meaningful below this.
const TOLERANCE_FLOOR: f64 = 1e-13;

/// Safety factor on the predicted `(σ₂/σ₁)^(2·its)` bound, covering the constant the asymptotic
/// rate omits. Kept small on purpose: measured errors sit far inside the bound (3.2e-7 against
/// a predicted 1.5e-2 at a gap of 0.90), so a large factor would only weaken the test.
const BOUND_SAFETY: f64 = 4.0;

/// Sweeps below which the asymptotic bound does not apply, so no case may use fewer.
///
/// `(σ₂/σ₁)^(2·its)` is the ASYMPTOTIC rate; it omits the initial alignment `|⟨x₀, v₁⟩|`, which
/// dominates for the first sweep or two. Measured: on a rank-one matrix — where the gap is zero
/// and the bound is therefore the machine-precision floor — a single sweep gives a relative
/// error of 2.6e-1, because one sweep estimates `σ₁·√|⟨x₀, v₁⟩|` rather than `σ₁`. From the
/// second sweep the iterate is aligned and the bound holds. Cases use more than that.
const MINIMUM_MEANINGFUL_SWEEPS: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// `1/(1 + i + 2j)` — smoothly decaying, a large spectral gap.
    Decay,
    /// Hilbert-like `1/(1 + i + j)`.
    Hilbert,
    /// Exactly rank 2, so `σ₂/σ₁` is small and `σ₃` is zero.
    Rank2,
    /// Diagonally dominant, a DELIBERATELY NARROW gap (~0.9) — the case that exercises the
    /// derived tolerance rather than the machine-precision floor.
    Narrow,
    /// Rank one: `σ₂` is exactly zero and the answer is analytic.
    RankOne,
}

impl Kind {
    fn eval(self, i: usize, j: usize) -> f64 {
        let (x, y) = (i as f64, j as f64);
        match self {
            Self::Decay => 1.0 / (1.0 + x + 2.0 * y),
            Self::Hilbert => 1.0 / (1.0 + x + y),
            Self::Rank2 => (x + 1.0) * (y + 1.0) + (x - 1.0) * (2.0 * y - 3.0),
            Self::Narrow => {
                if i == j {
                    x + 2.0
                } else {
                    0.5 / (1.0 + (x - y).abs())
                }
            }
            Self::RankOne => (x + 1.0) * (2.0 * y - 1.0),
        }
    }

    fn tag(self) -> &'static str {
        match self {
            Self::Decay => "decay",
            Self::Hilbert => "hilbert",
            Self::Rank2 => "rank2",
            Self::Narrow => "narrow",
            Self::RankOne => "rankone",
        }
    }
}

fn build(rows: usize, cols: usize, kind: Kind, scale: f64) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|i| (0..cols).map(|j| scale * kind.eval(i, j)).collect())
        .collect()
}

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    rows: usize,
    cols: usize,
    /// Row-major bit patterns for `a`, and for `b` when the case tests the difference form.
    a_bits: Vec<u64>,
    b_bits: Option<Vec<u64>>,
    its: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// The incumbent's estimate.
    estimate_bits: Option<u64>,
    /// The TRUE largest singular value, from a full SVD of the same operator.
    true_norm_bits: Option<u64>,
    /// `σ₂/σ₁` of the same operator, which sets the achievable accuracy.
    gap_bits: Option<u64>,
    error: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    rows: usize,
    cols: usize,
    is_difference_form: bool,
    gap: f64,
    derived_tolerance: f64,
    ours_relative_error: f64,
    scipy_relative_error: f64,
    arms_relative_difference: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared_cases: usize,
    difference_form_cases: usize,
    tight_cases: usize,
    worst_arms_relative_difference: f64,
    pass: bool,
    timestamp_ms: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn emit_log(log: &DiffLog) {
    fs::create_dir_all(output_dir()).expect("create interpolative norm diff output dir");
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize interpolative norm diff log");
    fs::write(path, json).expect("write interpolative norm diff log");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn flat_bits(m: &[Vec<f64>]) -> Vec<u64> {
    m.iter()
        .flat_map(|row| row.iter().map(|v| v.to_bits()))
        .collect()
}

fn single(case_id: &str, rows: usize, cols: usize, kind: Kind, its: usize) -> Case {
    Case {
        case_id: format!("{case_id}-{}", kind.tag()),
        rows,
        cols,
        a_bits: flat_bits(&build(rows, cols, kind, 1.0)),
        b_bits: None,
        its,
    }
}

fn difference(
    case_id: &str,
    rows: usize,
    cols: usize,
    a_kind: Kind,
    b_kind: Kind,
    b_scale: f64,
) -> Case {
    Case {
        case_id: format!("diff-{case_id}"),
        rows,
        cols,
        a_bits: flat_bits(&build(rows, cols, a_kind, 1.0)),
        b_bits: Some(flat_bits(&build(rows, cols, b_kind, b_scale))),
        its: DEFAULT_SPECTRAL_NORM_ITERATIONS,
    }
}

fn generate_query() -> OracleQuery {
    let its = DEFAULT_SPECTRAL_NORM_ITERATIONS;
    OracleQuery {
        points: vec![
            // Wide spectral gaps: both arms should reach machine precision.
            single("8x6", 8, 6, Kind::Decay, its),
            single("12x10", 12, 10, Kind::Decay, its),
            single("20x4", 20, 4, Kind::Decay, its),
            single("4x14", 4, 14, Kind::Decay, its),
            single("9x9", 9, 9, Kind::Hilbert, its),
            single("15x11", 15, 11, Kind::Hilbert, its),
            single("8x6", 8, 6, Kind::Rank2, its),
            single("12x10", 12, 10, Kind::Rank2, its),
            // σ₂ is exactly zero, so the iterate aligns after the FIRST sweep and every sweep
            // from the second on is exact. (Not after ONE sweep — see
            // MINIMUM_MEANINGFUL_SWEEPS.)
            single("7x5", 7, 5, Kind::RankOne, its),
            single("6x6", 6, 6, Kind::RankOne, MINIMUM_MEANINGFUL_SWEEPS),
            // NARROW gaps: these are the cases the derived tolerance exists for. A fixed
            // machine-precision tolerance would fail here for reasons that are the method's,
            // not the implementation's.
            single("8x6", 8, 6, Kind::Narrow, its),
            single("12x10", 12, 10, Kind::Narrow, its),
            // Fewer iterations, which must LOOSEN the achievable accuracy in step with the
            // bound rather than break it.
            single("short-8x6", 8, 6, Kind::Decay, MINIMUM_MEANINGFUL_SWEEPS),
            single("short-12x10", 12, 10, Kind::Narrow, 5),
            // The difference form, which must not form `a - b`.
            difference("scaled", 9, 7, Kind::Decay, Kind::Decay, 0.25),
            difference("mixed", 8, 8, Kind::Hilbert, Kind::Decay, 1.0),
            difference("rank2-vs-decay", 10, 6, Kind::Rank2, Kind::Decay, 3.0),
            difference("narrow", 7, 7, Kind::Narrow, Kind::Decay, 0.5),
            // a - a is the zero operator; both arms must report zero, not NaN.
            difference("self", 6, 5, Kind::Decay, Kind::Decay, 1.0),
        ],
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import struct
import sys
import numpy as np
import scipy.linalg.interpolative as ii

def bits(values):
    return [struct.unpack("<Q", struct.pack("<d", float(v)))[0] for v in values]

def unbits(values):
    return [struct.unpack("<d", struct.pack("<Q", int(v)))[0] for v in values]

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    try:
        m, n, its = int(case["rows"]), int(case["cols"]), int(case["its"])
        A = np.array(unbits(case["a_bits"]), dtype=float).reshape(m, n)
        if case.get("b_bits") is not None:
            B = np.array(unbits(case["b_bits"]), dtype=float).reshape(m, n)
            est = ii.estimate_spectral_norm_diff(A, B, its=its)
            operator = A - B
        else:
            est = ii.estimate_spectral_norm(A, its=its)
            operator = A
        s = np.linalg.svd(operator, compute_uv=False)
        true_norm = float(s[0]) if s.size else 0.0
        gap = float(s[1] / s[0]) if s.size > 1 and s[0] > 0.0 else 0.0
        if not all(math.isfinite(v) for v in (float(est), true_norm, gap)):
            raise ValueError("case produced a non-finite value")
        points.append({
            "case_id": cid,
            "estimate_bits": bits([est])[0],
            "true_norm_bits": bits([true_norm])[0],
            "gap_bits": bits([gap])[0],
            "error": None,
        })
    except Exception as exc:
        points.append({
            "case_id": cid, "estimate_bits": None, "true_norm_bits": None,
            "gap_bits": None, "error": f"{type(exc).__name__}: {exc}",
        })
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize interpolative norm query");
    let Some(incumbent) = incumbent() else {
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "no interpreter on this host can import scipy.linalg.interpolative, so the \
             differential arm cannot run"
        );
        eprintln!("skipping interpolative oracle: no live SciPy incumbent on this host");
        return None;
    };
    let mut child = match incumbent
        .command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for interpolative norm oracle: {e}"
            );
            eprintln!("skipping interpolative norm oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open interpolative norm oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "interpolative norm oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping interpolative norm oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for interpolative norm oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "interpolative norm oracle failed: {stderr}"
        );
        eprintln!("skipping interpolative norm oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse interpolative norm oracle JSON"))
}

fn reshape(bits: &[u64], rows: usize, cols: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|i| {
            bits[i * cols..(i + 1) * cols]
                .iter()
                .copied()
                .map(f64::from_bits)
                .collect()
        })
        .collect()
}

/// The accuracy the power method can actually reach on an operator with this spectral gap after
/// `its` sweeps, floored at machine precision.
///
/// Derived, not chosen: the error of the power method on `AᵀA` falls as `(σ₂/σ₁)^(2·its)`.
fn derived_tolerance(gap: f64, its: usize) -> f64 {
    let exponent = 2u32.saturating_mul(u32::try_from(its).unwrap_or(u32::MAX));
    let predicted = BOUND_SAFETY * gap.powi(exponent.min(i32::MAX as u32) as i32);
    predicted.max(TOLERANCE_FLOOR)
}

#[test]
fn diff_linalg_interpolative_spectral_norm() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(
        oracle.points.len(),
        query.points.len(),
        "the oracle must answer every case"
    );

    let mut cases = Vec::new();
    let mut compared = 0usize;
    let mut difference_cases = 0usize;
    let mut tight_cases = 0usize;
    let mut worst = 0.0f64;

    for (case, arm) in query.points.iter().zip(&oracle.points) {
        assert_eq!(
            case.case_id, arm.case_id,
            "oracle answers must stay in order"
        );
        assert!(
            arm.error.is_none(),
            "case {} raised on the incumbent: {:?}",
            case.case_id,
            arm.error
        );
        let (Some(estimate_bits), Some(true_norm_bits), Some(gap_bits)) =
            (arm.estimate_bits, arm.true_norm_bits, arm.gap_bits)
        else {
            panic!(
                "case {} came back with a null field and no error; a half-populated arm \
                 would compare vacuously",
                case.case_id
            );
        };
        let scipy_estimate = f64::from_bits(estimate_bits);
        let true_norm = f64::from_bits(true_norm_bits);
        let gap = f64::from_bits(gap_bits);

        let a = reshape(&case.a_bits, case.rows, case.cols);
        let ours = match case.b_bits.as_ref() {
            Some(b_bits) => {
                difference_cases += 1;
                let b = reshape(b_bits, case.rows, case.cols);
                estimate_spectral_norm_diff(&a, &b, case.its)
                    .unwrap_or_else(|e| panic!("case {}: {e}", case.case_id))
            }
            None => estimate_spectral_norm(&a, case.its)
                .unwrap_or_else(|e| panic!("case {}: {e}", case.case_id)),
        };

        // The zero operator: both arms must report exactly zero rather than a NaN from
        // normalising by a zero magnitude.
        if true_norm == 0.0 {
            assert_eq!(ours, 0.0, "case {}: zero operator", case.case_id);
            assert_eq!(
                scipy_estimate, 0.0,
                "case {}: incumbent on the zero operator",
                case.case_id
            );
            compared += 1;
            cases.push(CaseDiff {
                case_id: case.case_id.clone(),
                rows: case.rows,
                cols: case.cols,
                is_difference_form: case.b_bits.is_some(),
                gap: 0.0,
                derived_tolerance: 0.0,
                ours_relative_error: 0.0,
                scipy_relative_error: 0.0,
                arms_relative_difference: 0.0,
                pass: true,
            });
            continue;
        }

        // NO TOLERANCE: the power method cannot exceed the largest singular value. Applied to
        // both arms, because it is a property of the method rather than of either code.
        assert!(
            ours <= true_norm * (1.0 + 1e-12),
            "case {}: our estimate {ours} exceeds the true norm {true_norm}",
            case.case_id
        );
        assert!(
            scipy_estimate <= true_norm * (1.0 + 1e-12),
            "case {}: the incumbent's estimate {scipy_estimate} exceeds the true norm \
             {true_norm} — the oracle, not us",
            case.case_id
        );

        // ENFORCED, not documented: a case with too few sweeps would be judged against a bound
        // that does not apply to it, and would then either fail for the method's reasons or
        // pass a loosened one. Either way the row would be misleading.
        assert!(
            case.its >= MINIMUM_MEANINGFUL_SWEEPS,
            "case {} uses {} sweeps, below the {MINIMUM_MEANINGFUL_SWEEPS} the derived bound \
             is valid from",
            case.case_id,
            case.its
        );
        let tolerance = derived_tolerance(gap, case.its);
        let ours_error = (ours - true_norm).abs() / true_norm;
        let scipy_error = (scipy_estimate - true_norm).abs() / true_norm;
        let arms_difference = (ours - scipy_estimate).abs() / true_norm;

        assert!(
            ours_error <= tolerance,
            "case {}: our relative error {ours_error:e} exceeds the bound {tolerance:e} \
             derived from gap {gap} over {} sweeps",
            case.case_id,
            case.its
        );
        assert!(
            scipy_error <= tolerance,
            "case {}: the INCUMBENT's relative error {scipy_error:e} exceeds the same derived \
             bound {tolerance:e}; the bound, not the implementation, is what to look at",
            case.case_id
        );
        assert!(
            arms_difference <= 2.0 * tolerance,
            "case {}: the two arms differ by {arms_difference:e}, more than twice the derived \
             bound {tolerance:e}",
            case.case_id
        );

        if tolerance <= TOLERANCE_FLOOR {
            tight_cases += 1;
        }
        worst = worst.max(arms_difference);
        compared += 1;
        cases.push(CaseDiff {
            case_id: case.case_id.clone(),
            rows: case.rows,
            cols: case.cols,
            is_difference_form: case.b_bits.is_some(),
            gap,
            derived_tolerance: tolerance,
            ours_relative_error: ours_error,
            scipy_relative_error: scipy_error,
            arms_relative_difference: arms_difference,
            pass: true,
        });
    }

    emit_log(&DiffLog {
        test_id: "diff_linalg_interpolative_spectral_norm".to_string(),
        category: "linalg.interpolative".to_string(),
        case_count: query.points.len(),
        compared_cases: compared,
        difference_form_cases: difference_cases,
        tight_cases,
        worst_arms_relative_difference: worst,
        pass: true,
        timestamp_ms: timestamp_ms(),
        cases,
    });

    // GREEN-BY-SKIPPING GUARD, plus a guard against the suite drifting into only easy cases.
    eprintln!(
        "interpolative norm diff: compared={compared} difference_form={difference_cases} \
         tight={tight_cases} worst_arms_diff={worst:e}"
    );
    assert_eq!(
        compared,
        query.points.len(),
        "every case must be compared, not skipped"
    );
    assert!(
        difference_cases >= 4,
        "only {difference_cases} cases exercised the difference form"
    );
    // If EVERY case were narrow-gapped, every tolerance would be loose and the suite would
    // prove little. At least half must be at the machine-precision floor.
    assert!(
        tight_cases * 2 >= compared,
        "only {tight_cases} of {compared} cases were held to the machine-precision floor; \
         the suite has drifted toward cases its tolerances cannot constrain"
    );
}

/// MUST-HIT / MUST-MISS control for the derived tolerance, which is the only thing standing
/// between this suite and a tolerance loose enough to pass anything.
#[test]
fn derived_tolerance_tightens_with_the_spectral_gap() {
    let its = DEFAULT_SPECTRAL_NORM_ITERATIONS;

    // MUST-HIT: a wide gap collapses to the machine-precision floor, so those cases are held
    // to the tightest standard available.
    assert_eq!(derived_tolerance(0.2, its), TOLERANCE_FLOOR);
    assert_eq!(derived_tolerance(0.0, its), TOLERANCE_FLOOR);
    // A gap of exactly 0.5 does NOT quite reach the floor: 4·0.5^40 is 3.6e-12. Asserted as
    // the number it is rather than rounded down to the floor, because the boundary between
    // "held at the floor" and "held at a derived value" is the thing this control exists to
    // pin.
    let half = derived_tolerance(0.5, its);
    assert!(
        half > TOLERANCE_FLOOR && half < 1e-11,
        "a 0.5 gap should be just above the floor, got {half:e}"
    );

    // MUST-HIT: a narrow gap loosens, in the direction and roughly the amount the method's
    // convergence rate predicts.
    let narrow = derived_tolerance(0.9, its);
    assert!(
        narrow > TOLERANCE_FLOOR,
        "a 0.9 gap must not be held to machine precision, got {narrow:e}"
    );
    assert!(
        narrow < 1.0,
        "even a 0.9 gap must still constrain the answer, got {narrow:e}"
    );

    // MUST-HIT: fewer sweeps loosen the bound.
    assert!(derived_tolerance(0.9, 2) > derived_tolerance(0.9, its));

    // MUST-MISS: the tolerance must never reach a value that would accept a wholly wrong
    // answer. A gap of 0.99 is nearly degenerate and still bounds the error below 100%.
    assert!(
        derived_tolerance(0.99, its) < 10.0,
        "got {:e}",
        derived_tolerance(0.99, its)
    );
    // And it is monotone in the gap, which is what makes it a bound rather than a fudge.
    let mut previous = 0.0;
    for step in 0..=10 {
        let value = derived_tolerance(f64::from(step) / 10.0, its);
        assert!(
            value >= previous,
            "tolerance must not decrease as the gap widens"
        );
        previous = value;
    }
}
