#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `fsci_opt::find_root` against
//! `scipy.optimize.elementwise.find_root` (Chandrupatla's method).
//!
//! NOTHING IS PERMITTED TO DIFFER. Unlike `bracket_root`, whose two interleaved searches make
//! SciPy pay for one evaluation ours can skip, this is a single deterministic iteration: the
//! sampling schedule, the returned root, `f(x)`, the final bracket, `nit`, `nfev` and the
//! termination status are all compared exactly.
//!
//! WHY THE SCHEDULE AND NOT JUST THE ROOT. Every bracketing method converges to the same root;
//! that is what makes them all correct and what makes "the answers agree" nearly worthless as
//! evidence. What distinguishes Chandrupatla's method is WHERE it samples — interpolating when
//! its admissibility test passes and bisecting when it does not. A version that bisected every
//! time would agree on the root of every case here and disagree on the second abscissa of all
//! of them. So the abscissae are the comparison.
//!
//! FLOATS CROSS THE PROCESS BOUNDARY AS IEEE-754 BIT PATTERNS, never decimal text. `serde_json`
//! without `float_roundtrip` is not correctly rounded on every input — measured on
//! `"0.0017144775390624983"`, which it reads one ULP high — and that is large enough both to
//! invent a disagreement and, worse, to erase a real one.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use fsci_opt::{FindRootOptions, FindRootStatus, find_root};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

const SCIPY_CONVERGED: i64 = 0;
const SCIPY_SIGN_ERROR: i64 = -1;
const SCIPY_MAX_ITERATIONS: i64 = -2;
const SCIPY_NON_FINITE: i64 = -3;

/// Objectives both arms implement identically, in exact `f64` arithmetic so that any
/// disagreement is the SEARCH differing rather than the function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Objective {
    /// `x - c`, whose root the method should reach almost immediately.
    Linear,
    /// `x*x - c`
    Quadratic,
    /// `x*x*x - c`
    Cubic,
    /// `x**5 - c`, flat near the origin and steep away from it.
    Quintic,
    /// `x*x + c` — no root at all for positive `c`.
    NoRoot,
}

impl Objective {
    fn eval(self, x: f64, c: f64) -> f64 {
        match self {
            Self::Linear => x - c,
            Self::Quadratic => x * x - c,
            Self::Cubic => x * x * x - c,
            Self::Quintic => x * x * x * x * x - c,
            Self::NoRoot => x * x + c,
        }
    }

    fn tag(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Quadratic => "quadratic",
            Self::Cubic => "cubic",
            Self::Quintic => "quintic",
            Self::NoRoot => "noroot",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    objective: String,
    c: f64,
    xl0: f64,
    xr0: f64,
    xatol: Option<f64>,
    xrtol: Option<f64>,
    fatol: Option<f64>,
    frtol: Option<f64>,
    maxiter: Option<usize>,
    #[serde(skip)]
    kind: Objective,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

/// One case's answer. Every float arrives as its IEEE-754 bit pattern.
#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    sampled_bits: Option<Vec<u64>>,
    x_bits: Option<u64>,
    f_x_bits: Option<u64>,
    bracket_bits: Option<Vec<u64>>,
    f_bracket_bits: Option<Vec<u64>>,
    nit: Option<u64>,
    nfev: Option<u64>,
    status: Option<i64>,
    success: Option<bool>,
    error: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    objective: String,
    status: String,
    nit: u64,
    nfev: u64,
    samples_compared: usize,
    interpolated_at_least_once: bool,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared_cases: usize,
    total_samples_compared: usize,
    interpolating_cases: usize,
    statuses_seen: Vec<String>,
    pass: bool,
    timestamp_ms: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn emit_log(log: &DiffLog) {
    fs::create_dir_all(output_dir()).expect("create find_root diff output dir");
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize find_root diff log");
    fs::write(path, json).expect("write find_root diff log");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn case(case_id: &str, kind: Objective, c: f64, xl0: f64, xr0: f64) -> Case {
    Case {
        case_id: case_id.to_string(),
        objective: kind.tag().to_string(),
        c,
        xl0,
        xr0,
        xatol: None,
        xrtol: None,
        fatol: None,
        frtol: None,
        maxiter: None,
        kind,
    }
}

fn generate_query() -> OracleQuery {
    use Objective::{Cubic, Linear, NoRoot, Quadratic, Quintic};
    let points = vec![
        // The worked example: the first step bisects to 1.5, the second interpolates to
        // 1.230644178..., where plain bisection would have gone to 1.25.
        case("cubic", Cubic, 2.0, 1.0, 2.0),
        case("cubic-wide", Cubic, 2.0, -5.0, 40.0),
        case("quadratic", Quadratic, 2.0, 1.0, 2.0),
        case("quadratic-asym", Quadratic, 10.0, 0.5, 100.0),
        // A root exactly at an endpoint. THE case Rust's `signum` would break: with it,
        // sign(0.0) == sign(1.0) and the search reports "no bracket" on an interval whose left
        // end IS the root.
        case("root-at-left", Linear, 1.0, 1.0, 2.0),
        case("root-at-right", Linear, 2.0, 1.0, 2.0),
        // Reversed interval: the search must handle it and still report ascending.
        case("reversed", Cubic, 2.0, 2.0, 1.0),
        // Straddling zero, so signs are unambiguous but the root is at the origin.
        case("root-at-origin", Cubic, 0.0, -1.0, 3.0),
        // Flat-then-steep, which is where interpolation earns its keep.
        case("quintic", Quintic, 1e-6, 0.0, 10.0),
        case("quintic-neg", Quintic, -32.0, -10.0, 1.0),
        // Large and small scales, to exercise the RELATIVE width tolerance.
        case("large-scale", Quadratic, 1e12, 1.0, 1e8),
        case("small-scale", Quadratic, 1e-12, 1e-9, 1.0),
        // NO SIGN CHANGE: must report a sign error and NaN rather than inventing a root.
        case("no-root", NoRoot, 1.0, 1.0, 2.0),
        // Tolerances, each stopping the search on a different condition.
        Case {
            frtol: Some(1e-3),
            ..case("frtol-loose", Cubic, 2.0, 1.0, 2.0)
        },
        Case {
            xatol: Some(1e-6),
            xrtol: Some(0.0),
            fatol: Some(0.0),
            frtol: Some(0.0),
            ..case("xatol-loose", Cubic, 2.0, 1.0, 2.0)
        },
        Case {
            xatol: Some(0.0),
            xrtol: Some(1e-4),
            fatol: Some(0.0),
            frtol: Some(0.0),
            ..case("xrtol-only", Quadratic, 3.0, 1.0, 3.0)
        },
        Case {
            fatol: Some(1e-8),
            frtol: Some(0.0),
            ..case("fatol-loose", Quintic, 3.0, 1.0, 2.0)
        },
        // BUDGET EXHAUSTED partway, which must report max-iterations rather than the root it
        // happens to be near.
        Case {
            maxiter: Some(2),
            ..case("budget-2", Cubic, 2.0, 1.0, 2.0)
        },
        Case {
            maxiter: Some(1),
            ..case("budget-1", Quintic, 1e-6, 0.0, 10.0)
        },
    ];
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import struct
import sys
import numpy as np
from scipy.optimize import elementwise

def bits(values):
    # IEEE-754 bit patterns: decimal text is not a lossless channel for f64 between these two
    # processes, so the numbers travel as integers.
    return [struct.unpack("<Q", struct.pack("<d", float(v)))[0] for v in values]

def objective(tag, c):
    if tag == "linear":
        return lambda x: x - c
    if tag == "quadratic":
        return lambda x: x * x - c
    if tag == "cubic":
        return lambda x: x * x * x - c
    if tag == "quintic":
        return lambda x: x * x * x * x * x - c
    if tag == "noroot":
        return lambda x: x * x + c
    raise ValueError(tag)

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    sampled = []
    base = objective(case["objective"], float(case["c"]))

    def f(x, _base=base, _sampled=sampled):
        a = np.asarray(x, dtype=float)
        _sampled.extend(np.atleast_1d(a).ravel().tolist())
        return _base(a)

    tolerances = {}
    for key in ("xatol", "xrtol", "fatol", "frtol"):
        if case.get(key) is not None:
            tolerances[key] = float(case[key])
    kwargs = {}
    if tolerances:
        kwargs["tolerances"] = tolerances
    if case.get("maxiter") is not None:
        kwargs["maxiter"] = int(case["maxiter"])
    try:
        r = elementwise.find_root(f, (float(case["xl0"]), float(case["xr0"])), **kwargs)
        # NaN is expected for a failed search and is representable as a bit pattern, so it is
        # carried like any other value; only the ABSCISSAE must be finite for the schedule to
        # mean anything.
        if not all(math.isfinite(v) for v in sampled):
            raise ValueError("case sampled a non-finite abscissa")
        points.append({
            "case_id": cid,
            "sampled_bits": bits(sampled),
            "x_bits": bits([r.x])[0],
            "f_x_bits": bits([r.f_x])[0],
            "bracket_bits": bits(r.bracket),
            "f_bracket_bits": bits(r.f_bracket),
            "nit": int(r.nit),
            "nfev": int(r.nfev),
            "status": int(r.status),
            "success": bool(r.success),
            "error": None,
        })
    except Exception as exc:
        points.append({
            "case_id": cid, "sampled_bits": None, "x_bits": None, "f_x_bits": None,
            "bracket_bits": None, "f_bracket_bits": None,
            "nit": None, "nfev": None, "status": None, "success": None,
            "error": f"{type(exc).__name__}: {exc}",
        })
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize find_root query");
    let mut child = match Command::new("python3")
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
                "failed to spawn python3 for find_root oracle: {e}"
            );
            eprintln!("skipping find_root oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open find_root oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "find_root oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping find_root oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for find_root oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "find_root oracle failed: {stderr}"
        );
        eprintln!("skipping find_root oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse find_root oracle JSON"))
}

/// Bitwise sequence equality against transported bit patterns.
fn same_bits(ours: &[f64], theirs: &[u64]) -> bool {
    ours.len() == theirs.len() && ours.iter().zip(theirs).all(|(x, y)| x.to_bits() == *y)
}

fn as_floats(bits: &[u64]) -> Vec<f64> {
    bits.iter().copied().map(f64::from_bits).collect()
}

fn status_name(status: FindRootStatus) -> &'static str {
    match status {
        FindRootStatus::Converged => "converged",
        FindRootStatus::SignError => "sign_error",
        FindRootStatus::MaxIterations => "max_iterations",
        FindRootStatus::NonFinite => "non_finite",
    }
}

fn scipy_status_name(status: i64) -> String {
    match status {
        SCIPY_CONVERGED => "converged".to_string(),
        SCIPY_SIGN_ERROR => "sign_error".to_string(),
        SCIPY_MAX_ITERATIONS => "max_iterations".to_string(),
        SCIPY_NON_FINITE => "non_finite".to_string(),
        other => format!("unmapped({other})"),
    }
}

/// Did this schedule ever take a step that plain bisection would not have?
///
/// The point of the method. Every abscissa after the first is compared against the midpoint of
/// the interval the method was working from; if they always coincided, this test would be
/// passing on a bisector and would say nothing about Chandrupatla's method specifically.
fn took_an_interpolated_step(sampled: &[f64]) -> bool {
    // sampled[0], sampled[1] are the interval ends; sampled[2] is always the bisection.
    sampled.len() > 3 && {
        let midpoint = 0.5 * (sampled[0] + sampled[1]);
        // The third abscissa is the bisection of the ORIGINAL interval by construction; any
        // later point differing from a naive continued bisection means interpolation ran.
        (sampled[2] - midpoint).abs() < f64::EPSILON * midpoint.abs().max(1.0)
            && sampled[3] != 0.5 * (sampled[2] + sampled[0])
            && sampled[3] != 0.5 * (sampled[2] + sampled[1])
    }
}

#[test]
fn diff_optimize_find_root() {
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
    let mut total_samples = 0usize;
    let mut interpolating = 0usize;
    let mut statuses_seen: Vec<String> = Vec::new();

    for (case, arm) in query.points.iter().zip(&oracle.points) {
        assert_eq!(
            case.case_id, arm.case_id,
            "oracle answers must stay in order"
        );
        assert!(
            arm.error.is_none(),
            "case {} raised on the incumbent: {:?}. Every case here is one scipy supports; \
             an error means the query is malformed, not that the case may be skipped.",
            case.case_id,
            arm.error
        );
        let (
            Some(sampled),
            Some(x_bits),
            Some(f_x_bits),
            Some(bracket),
            Some(f_bracket),
            Some(nit),
            Some(nfev),
            Some(status),
            Some(success),
        ) = (
            arm.sampled_bits.as_ref(),
            arm.x_bits,
            arm.f_x_bits,
            arm.bracket_bits.as_ref(),
            arm.f_bracket_bits.as_ref(),
            arm.nit,
            arm.nfev,
            arm.status,
            arm.success,
        )
        else {
            panic!(
                "case {} came back with a null field and no error; a half-populated arm \
                 would compare vacuously",
                case.case_id
            );
        };

        let visited = std::cell::RefCell::new(Vec::new());
        let kind = case.kind;
        let c = case.c;
        let result = find_root(
            |x| {
                visited.borrow_mut().push(x);
                kind.eval(x, c)
            },
            (case.xl0, case.xr0),
            FindRootOptions {
                xatol: case.xatol,
                xrtol: case.xrtol,
                fatol: case.fatol,
                frtol: case.frtol,
                maxiter: case.maxiter,
            },
        )
        .unwrap_or_else(|e| panic!("case {} failed on our arm: {e}", case.case_id));
        let ours = visited.into_inner();

        assert!(
            same_bits(&ours, sampled),
            "case {}: sampling schedule differs.\n  ours ({} pts): {:?}\n  scipy ({} pts): {:?}",
            case.case_id,
            ours.len(),
            &ours[..ours.len().min(20)],
            sampled.len(),
            as_floats(&sampled[..sampled.len().min(20)])
        );
        assert_eq!(
            status_name(result.status),
            scipy_status_name(status),
            "case {}: termination status differs",
            case.case_id
        );
        assert_eq!(
            result.success, success,
            "case {}: success differs",
            case.case_id
        );
        assert_eq!(
            u64::try_from(result.nit).expect("nit fits u64"),
            nit,
            "case {}: iteration count differs",
            case.case_id
        );
        assert_eq!(
            u64::try_from(result.nfev).expect("nfev fits u64"),
            nfev,
            "case {}: evaluation count differs",
            case.case_id
        );
        // `to_bits` compares NaN to NaN as equal, which is what is wanted: a failed search must
        // report NaN on BOTH arms, and `==` would call that a mismatch.
        assert_eq!(
            result.x.to_bits(),
            x_bits,
            "case {}: root differs; ours {}, scipy {}",
            case.case_id,
            result.x,
            f64::from_bits(x_bits)
        );
        assert_eq!(
            result.f_x.to_bits(),
            f_x_bits,
            "case {}: f(x) differs; ours {}, scipy {}",
            case.case_id,
            result.f_x,
            f64::from_bits(f_x_bits)
        );
        assert!(
            same_bits(&[result.bracket.0, result.bracket.1], bracket),
            "case {}: bracket differs; ours {:?}, scipy {:?}",
            case.case_id,
            result.bracket,
            as_floats(bracket)
        );
        assert!(
            same_bits(&[result.f_bracket.0, result.f_bracket.1], f_bracket),
            "case {}: f_bracket differs; ours {:?}, scipy {:?}",
            case.case_id,
            result.f_bracket,
            as_floats(f_bracket)
        );

        // A successful search must actually have found a root, not merely stopped.
        if result.success {
            assert!(
                result.x.is_finite() && result.f_x.is_finite(),
                "case {}: reported success with a non-finite answer",
                case.case_id
            );
            assert!(
                result.bracket.0 <= result.x && result.x <= result.bracket.1,
                "case {}: root {} lies outside its own bracket {:?}",
                case.case_id,
                result.x,
                result.bracket
            );
        }

        let interpolated = took_an_interpolated_step(&ours);
        if interpolated {
            interpolating += 1;
        }
        compared += 1;
        total_samples += ours.len();
        let name = status_name(result.status).to_string();
        if !statuses_seen.contains(&name) {
            statuses_seen.push(name.clone());
        }
        cases.push(CaseDiff {
            case_id: case.case_id.clone(),
            objective: case.objective.clone(),
            status: name,
            nit,
            nfev,
            samples_compared: ours.len(),
            interpolated_at_least_once: interpolated,
            pass: true,
        });
    }

    statuses_seen.sort();
    emit_log(&DiffLog {
        test_id: "diff_optimize_find_root".to_string(),
        category: "optimize.elementwise".to_string(),
        case_count: query.points.len(),
        compared_cases: compared,
        total_samples_compared: total_samples,
        interpolating_cases: interpolating,
        statuses_seen: statuses_seen.clone(),
        pass: true,
        timestamp_ms: timestamp_ms(),
        cases,
    });

    // GREEN-BY-SKIPPING GUARD. Every assertion above passes by vacuity on an empty comparison.
    eprintln!(
        "find_root diff: compared={compared} samples={total_samples} \
         interpolating={interpolating} statuses={statuses_seen:?}"
    );
    assert_eq!(
        compared,
        query.points.len(),
        "every case must be compared, not skipped"
    );
    assert!(
        total_samples > 100,
        "only {total_samples} abscissae compared; the deep cases did not run"
    );
    // Without this the suite could be passing on a bisector — see `took_an_interpolated_step`.
    assert!(
        interpolating >= 5,
        "only {interpolating} cases took an interpolated step; this suite would not \
         distinguish Chandrupatla's method from plain bisection"
    );
    assert_eq!(
        statuses_seen,
        vec![
            "converged".to_string(),
            "max_iterations".to_string(),
            "sign_error".to_string()
        ],
        "every reachable termination status must be exercised"
    );
}

/// MUST-HIT / MUST-MISS control for the bitwise comparator every assertion above depends on.
#[test]
fn bit_comparator_rejects_the_differences_it_exists_to_catch() {
    let w = |xs: &[f64]| -> Vec<u64> { xs.iter().map(|x| x.to_bits()).collect() };

    assert!(same_bits(&[1.0, 2.0, 0.5], &w(&[1.0, 2.0, 0.5])));
    assert!(same_bits(&[], &w(&[])));
    assert!(!same_bits(&[1.0, 2.0], &w(&[1.0, 2.0, 3.0])));
    assert!(!same_bits(&[1.0, 2.5], &w(&[1.0, 2.0])));
    // `-0.0 == 0.0` is true in Rust; these are different abscissae.
    assert!(!same_bits(&[0.0], &w(&[-0.0])));
    // NaN must compare EQUAL here — a failed search reports NaN on both arms.
    assert!(same_bits(&[f64::NAN], &w(&[f64::NAN])));
    // One ULP, built by incrementing the bit pattern: a hand-typed decimal would round back.
    let one_ulp_up = 1.0f64.to_bits() + 1;
    assert_ne!(one_ulp_up, 1.0f64.to_bits());
    assert!(!same_bits(&[1.0], &[one_ulp_up]));
}

/// MUST-HIT / MUST-MISS control for the interpolation detector, which is the thing standing
/// between this suite and silently certifying a bisector.
#[test]
fn interpolation_detector_separates_a_real_step_from_a_bisection() {
    // MUST-HIT: the measured scipy schedule for `x**3 - 2` on (1, 2). Third point is the
    // bisection 1.5; fourth is the interpolated 1.230644178, not the bisection 1.25.
    assert!(took_an_interpolated_step(&[
        1.0,
        2.0,
        1.5,
        1.230_644_178_012_599_2,
        1.261_922_195_4
    ]));

    // MUST-MISS: a pure bisector. Fourth point is the midpoint of (1.0, 1.5).
    assert!(!took_an_interpolated_step(&[1.0, 2.0, 1.5, 1.25, 1.3125]));
    // MUST-MISS: a bisector that happened to go the other way.
    assert!(!took_an_interpolated_step(&[1.0, 2.0, 1.5, 1.75, 1.625]));
    // MUST-MISS: too short to tell, which must not count as evidence of interpolation.
    assert!(!took_an_interpolated_step(&[1.0, 2.0, 1.5]));
    assert!(!took_an_interpolated_step(&[1.0, 2.0]));
}

/// MUST-MISS control for the status mapping: an unmapped code must be visibly unmapped.
#[test]
fn status_mapping_names_every_code_it_claims_and_flags_the_rest() {
    assert_eq!(scipy_status_name(SCIPY_CONVERGED), "converged");
    assert_eq!(scipy_status_name(SCIPY_SIGN_ERROR), "sign_error");
    assert_eq!(scipy_status_name(SCIPY_MAX_ITERATIONS), "max_iterations");
    assert_eq!(scipy_status_name(SCIPY_NON_FINITE), "non_finite");
    assert_eq!(scipy_status_name(-77), "unmapped(-77)");
    assert_eq!(status_name(FindRootStatus::Converged), "converged");
    assert_eq!(status_name(FindRootStatus::SignError), "sign_error");
}
