#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `fsci_opt::find_minimum` against
//! `scipy.optimize.elementwise.find_minimum` (Chandrupatla's minimization method).
//!
//! NOTHING IS PERMITTED TO DIFFER: sampling schedule, minimizer, `f(x)`, the final trio, `nit`,
//! `nfev` and the termination status are all compared exactly, with floats carried across the
//! process boundary as IEEE-754 bit patterns rather than decimal text (`serde_json` without
//! `float_roundtrip` misreads some 17-digit decimals by one ULP, which would both invent and
//! erase differences).
//!
//! ## Why the schedule is the comparison, concretely
//!
//! For `(x - 2)²` on the trio `(0, 1, 5)` the parabola through those three points has its
//! vertex at EXACTLY 2.0 — the answer, in one step. SciPy does not take it: Chandrupatla's
//! condition (7) compares this iteration's vertex against the PREVIOUS one, `q0`, which starts
//! at `x3 = 5`, and `|2 - 5|` is not less than `|x2 - x1| / 2`. So the first step is the golden
//! section `2.52786404500042` instead, and the search takes four iterations.
//!
//! An implementation that initialised `q0` to anything near the vertex would converge in ONE
//! iteration on this case and agree on the minimizer to the last bit. It would look better and
//! be a different algorithm. Only the schedule catches it.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{SystemTime, UNIX_EPOCH};

use fsci_opt::{FindMinimumOptions, FindMinimumStatus, find_minimum};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

const SCIPY_CONVERGED: i64 = 0;
const SCIPY_BRACKET_ERROR: i64 = -1;
const SCIPY_MAX_ITERATIONS: i64 = -2;
const SCIPY_NON_FINITE: i64 = -3;

/// Objectives both arms implement identically, in exact `f64` arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Objective {
    /// `(x - c)**2`
    Parabola,
    /// `(x - c)**4`, flat at its minimum — the case that motivates the loose default `xrtol`.
    Quartic,
    /// `(x - c)**2 * ((x - c) * (x - c) + 1)`, asymmetric about its minimum.
    Skewed,
    /// `x - c` — monotone, so no interior minimum exists.
    Monotone,
    /// `-(x - c)**2` — a maximum where a minimum is wanted.
    Inverted,
}

impl Objective {
    fn eval(self, x: f64, c: f64) -> f64 {
        let d = x - c;
        match self {
            Self::Parabola => d * d,
            Self::Quartic => d * d * d * d,
            Self::Skewed => d * d * (d * d + 1.0),
            Self::Monotone => d,
            Self::Inverted => -(d * d),
        }
    }

    fn tag(self) -> &'static str {
        match self {
            Self::Parabola => "parabola",
            Self::Quartic => "quartic",
            Self::Skewed => "skewed",
            Self::Monotone => "monotone",
            Self::Inverted => "inverted",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    objective: String,
    c: f64,
    x1: f64,
    x2: f64,
    x3: f64,
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
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared_cases: usize,
    total_samples_compared: usize,
    statuses_seen: Vec<String>,
    pass: bool,
    timestamp_ms: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn emit_log(log: &DiffLog) {
    fs::create_dir_all(output_dir()).expect("create find_minimum diff output dir");
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize find_minimum diff log");
    fs::write(path, json).expect("write find_minimum diff log");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn case(case_id: &str, kind: Objective, c: f64, x1: f64, x2: f64, x3: f64) -> Case {
    Case {
        case_id: case_id.to_string(),
        objective: kind.tag().to_string(),
        c,
        x1,
        x2,
        x3,
        xatol: None,
        xrtol: None,
        fatol: None,
        frtol: None,
        maxiter: None,
        kind,
    }
}

fn generate_query() -> OracleQuery {
    use Objective::{Inverted, Monotone, Parabola, Quartic, Skewed};
    let points = vec![
        // The worked example: the vertex is exactly 2.0 and is NOT taken on step one.
        case("parabola", Parabola, 2.0, 0.0, 1.0, 5.0),
        case("parabola-tight", Parabola, 2.0, 1.9, 2.05, 2.2),
        case("parabola-wide", Parabola, -30.0, -1000.0, 0.0, 500.0),
        // The trio given UNSORTED and REVERSED: both must be sorted on entry, and the initial
        // three samples must still follow the caller's argument order.
        case("unsorted", Parabola, 2.0, 5.0, 1.0, 0.0),
        case("unsorted-middle-last", Parabola, 2.0, 0.0, 5.0, 1.0),
        // Flat at the minimum, which is what the default `xrtol = sqrt(eps)` exists for.
        case("quartic", Quartic, 0.75, -3.0, 0.0, 10.0),
        case("quartic-offcentre", Quartic, -2.5, -8.0, -1.0, 1.0),
        // Asymmetric, so the parabola fit is wrong in a direction that changes each iteration.
        case("skewed", Skewed, 1.25, -2.0, 0.5, 6.0),
        case("skewed-far", Skewed, 40.0, 0.0, 10.0, 200.0),
        // Minimum sitting exactly on the middle point already.
        case("already-centred", Parabola, 1.0, 0.0, 1.0, 2.0),
        // Scale extremes, to exercise the RELATIVE width tolerance in both directions.
        case("large-scale", Parabola, 1e6, 0.0, 1e5, 1e7),
        case("small-scale", Parabola, 1e-7, -1e-5, 0.0, 1e-4),
        // NOT A BRACKET: the middle point is not the lowest. Must report a bracket error and
        // NaN rather than inventing a minimum.
        case("monotone", Monotone, 0.0, 0.0, 1.0, 5.0),
        case("inverted", Inverted, 2.0, 0.0, 1.0, 5.0),
        // Tolerances, each able to stop the search early on a different condition.
        Case {
            xrtol: Some(1e-3),
            ..case("xrtol-loose", Parabola, 2.0, 0.0, 1.0, 5.0)
        },
        Case {
            xatol: Some(1e-2),
            xrtol: Some(0.0),
            ..case("xatol-loose", Parabola, 2.0, 0.0, 1.0, 5.0)
        },
        Case {
            fatol: Some(1e-4),
            frtol: Some(0.0),
            ..case("fatol-loose", Skewed, 1.25, -2.0, 0.5, 6.0)
        },
        Case {
            frtol: Some(1e-2),
            ..case("frtol-loose", Parabola, 3.0, 0.0, 1.0, 9.0)
        },
        // BUDGET EXHAUSTED, which must keep the iterate rather than NaN it.
        Case {
            maxiter: Some(2),
            ..case("budget-2", Parabola, 2.0, 0.0, 1.0, 5.0)
        },
        Case {
            maxiter: Some(1),
            ..case("budget-1", Quartic, 0.75, -3.0, 0.0, 10.0)
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
    return [struct.unpack("<Q", struct.pack("<d", float(v)))[0] for v in values]

def objective(tag, c):
    if tag == "parabola":
        return lambda x: (x - c) * (x - c)
    if tag == "quartic":
        return lambda x: (x - c) * (x - c) * (x - c) * (x - c)
    if tag == "skewed":
        return lambda x: (x - c) * (x - c) * ((x - c) * (x - c) + 1.0)
    if tag == "monotone":
        return lambda x: x - c
    if tag == "inverted":
        return lambda x: -((x - c) * (x - c))
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
    init = (float(case["x1"]), float(case["x2"]), float(case["x3"]))
    try:
        r = elementwise.find_minimum(f, init, **kwargs)
        # NaN in the RESULT is expected for a failed search and travels fine as a bit pattern;
        # only the abscissae must be finite for the schedule to mean anything.
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
    let query_json = serde_json::to_string(query).expect("serialize find_minimum query");
    let mut child = match fsci_conformance::scipy_oracle_command()
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
                "failed to spawn python3 for find_minimum oracle: {e}"
            );
            eprintln!("skipping find_minimum oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open find_minimum oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "find_minimum oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping find_minimum oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for find_minimum oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "find_minimum oracle failed: {stderr}"
        );
        eprintln!("skipping find_minimum oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse find_minimum oracle JSON"))
}

fn same_bits(ours: &[f64], theirs: &[u64]) -> bool {
    ours.len() == theirs.len() && ours.iter().zip(theirs).all(|(x, y)| x.to_bits() == *y)
}

fn as_floats(bits: &[u64]) -> Vec<f64> {
    bits.iter().copied().map(f64::from_bits).collect()
}

fn status_name(status: FindMinimumStatus) -> &'static str {
    match status {
        FindMinimumStatus::Converged => "converged",
        FindMinimumStatus::BracketError => "bracket_error",
        FindMinimumStatus::MaxIterations => "max_iterations",
        FindMinimumStatus::NonFinite => "non_finite",
    }
}

fn scipy_status_name(status: i64) -> String {
    match status {
        SCIPY_CONVERGED => "converged".to_string(),
        SCIPY_BRACKET_ERROR => "bracket_error".to_string(),
        SCIPY_MAX_ITERATIONS => "max_iterations".to_string(),
        SCIPY_NON_FINITE => "non_finite".to_string(),
        other => format!("unmapped({other})"),
    }
}

#[test]
fn diff_optimize_find_minimum() {
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
        let result = find_minimum(
            |x| {
                visited.borrow_mut().push(x);
                kind.eval(x, c)
            },
            (case.x1, case.x2, case.x3),
            FindMinimumOptions {
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
        // Compared by bits so that NaN matches NaN, which is the expected answer on both arms
        // for a failed search.
        assert_eq!(
            result.x.to_bits(),
            x_bits,
            "case {}: minimizer differs; ours {}, scipy {}",
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
            same_bits(
                &[result.bracket.0, result.bracket.1, result.bracket.2],
                bracket
            ),
            "case {}: bracket differs; ours {:?}, scipy {:?}",
            case.case_id,
            result.bracket,
            as_floats(bracket)
        );
        assert!(
            same_bits(
                &[result.f_bracket.0, result.f_bracket.1, result.f_bracket.2],
                f_bracket
            ),
            "case {}: f_bracket differs; ours {:?}, scipy {:?}",
            case.case_id,
            result.f_bracket,
            as_floats(f_bracket)
        );

        // A converged trio must really bracket a minimum, so `success` is worth trusting.
        if result.success {
            let (xl, xm, xr) = result.bracket;
            let (fl, fm, fr) = result.f_bracket;
            assert!(
                xl <= xm && xm <= xr,
                "case {}: bracket out of order: {:?}",
                case.case_id,
                result.bracket
            );
            assert!(
                fm <= fl && fm <= fr,
                "case {}: reported success with the middle point not the lowest: {:?}",
                case.case_id,
                result.f_bracket
            );
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
            pass: true,
        });
    }

    statuses_seen.sort();
    emit_log(&DiffLog {
        test_id: "diff_optimize_find_minimum".to_string(),
        category: "optimize.elementwise".to_string(),
        case_count: query.points.len(),
        compared_cases: compared,
        total_samples_compared: total_samples,
        statuses_seen: statuses_seen.clone(),
        pass: true,
        timestamp_ms: timestamp_ms(),
        cases,
    });

    // GREEN-BY-SKIPPING GUARD.
    eprintln!(
        "find_minimum diff: compared={compared} samples={total_samples} \
         statuses={statuses_seen:?}"
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
    assert_eq!(
        statuses_seen,
        vec![
            "bracket_error".to_string(),
            "converged".to_string(),
            "max_iterations".to_string()
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
    assert!(!same_bits(&[0.0], &w(&[-0.0])), "a lost sign on zero");
    assert!(same_bits(&[f64::NAN], &w(&[f64::NAN])), "NaN matches NaN");
    let one_ulp_up = 2.0f64.to_bits() + 1;
    assert_ne!(one_ulp_up, 2.0f64.to_bits());
    assert!(!same_bits(&[2.0], &[one_ulp_up]));
}

/// MUST-MISS control for the status mapping.
#[test]
fn status_mapping_names_every_code_it_claims_and_flags_the_rest() {
    assert_eq!(scipy_status_name(SCIPY_CONVERGED), "converged");
    assert_eq!(scipy_status_name(SCIPY_BRACKET_ERROR), "bracket_error");
    assert_eq!(scipy_status_name(SCIPY_MAX_ITERATIONS), "max_iterations");
    assert_eq!(scipy_status_name(SCIPY_NON_FINITE), "non_finite");
    assert_eq!(scipy_status_name(-42), "unmapped(-42)");
    assert_eq!(status_name(FindMinimumStatus::Converged), "converged");
    assert_eq!(
        status_name(FindMinimumStatus::BracketError),
        "bracket_error"
    );
}
