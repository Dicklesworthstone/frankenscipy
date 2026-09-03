#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `fsci_opt::bracket_root` against
//! `scipy.optimize.elementwise.bracket_root`.
//!
//! WHAT IS ACTUALLY COMPARED. Agreeing on the returned bracket is a weak check: two searches
//! can walk completely different points and still land on one. So this test compares the
//! SAMPLING SCHEDULE — every abscissa each implementation hands to the objective, in order —
//! as well as the bracket and the iteration count. The oracle records scipy's evaluations by
//! appending to a list from inside the callback.
//!
//! THE TWO DIVERGENCES ARE ENCODED AS RULES, NOT WAIVED. `fsci_opt::bracket_root` differs
//! from the incumbent in exactly two ways, both deliberate and both documented at the
//! implementation:
//!
//!   1. When the LEFTWARD search brackets first, scipy still evaluates the rightward point in
//!      the same vectorized call. Ours stops. So ours is allowed to be scipy's schedule minus
//!      a trailing point — and nothing else. A missing point anywhere but the end, or two
//!      missing points, fails.
//!   2. When BOTH searches bracket on the same iteration, scipy returns `xl` from the
//!      rightward search and `xr` from the leftward one, giving two same-signed endpoints that
//!      are not a bracket (`f(x)=x**2-100` from `(-1, 1)` returns `(7.0, -7.0)` with
//!      `f_bracket = (-51.0, -51.0)` and `success=True` on scipy 1.17.1). Where the oracle
//!      returns a non-bracket, this test asserts OURS is a real one instead of matching it.
//!
//! Every other case must agree bit-for-bit — these are exact binary schedules, not tolerances,
//! so the comparison is `==` on `f64` bits and there is no epsilon to tune.
//!
//! FLOATS CROSS THE PROCESS BOUNDARY AS IEEE-754 BIT PATTERNS, never as decimal text.
//! `serde_json` without its `float_roundtrip` feature is not correctly rounded on every input:
//! `serde_json::from_str::<f64>("0.0017144775390624983")` yields `0x3f5c170a3d70a3d0`, one ULP
//! above the `0x3f5c170a3d70a3cf` that Python's `float()` and Rust's own literal parser both
//! produce. That is large enough to invent a disagreement between two arms that computed
//! identical bits — it did exactly that in this test's sibling — and equally large enough to
//! ERASE a real one. Integers do not have that problem, so bits are what travel.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{SystemTime, UNIX_EPOCH};

use fsci_opt::{BracketOptions, bracket_root};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

/// Objectives both arms implement identically. Kept to exactly-representable polynomial
/// arithmetic so the two languages produce bit-identical values and any schedule difference is
/// the SEARCH differing, never the objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Objective {
    /// `x - c`
    Linear,
    /// `x*x - c`
    Quadratic,
    /// `x*x*x - c`
    Cubic,
    /// `x*x + c`, which for positive `c` never crosses zero.
    NoRoot,
}

impl Objective {
    fn eval(self, x: f64, c: f64) -> f64 {
        match self {
            Self::Linear => x - c,
            Self::Quadratic => x * x - c,
            Self::Cubic => x * x * x - c,
            Self::NoRoot => x * x + c,
        }
    }

    fn tag(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Quadratic => "quadratic",
            Self::Cubic => "cubic",
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
    xr0: Option<f64>,
    xmin: Option<f64>,
    xmax: Option<f64>,
    factor: Option<f64>,
    maxiter: usize,
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
    /// Every abscissa scipy evaluated, in order. `None` when scipy raised.
    sampled_bits: Option<Vec<u64>>,
    bracket_bits: Option<Vec<u64>>,
    f_bracket_bits: Option<Vec<u64>>,
    nit: Option<u64>,
    nfev: Option<u64>,
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
    /// `matched`, `trailing_lockstep_eval`, or `incumbent_returned_non_bracket`.
    verdict: String,
    ours_nfev: usize,
    scipy_nfev: u64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared_cases: usize,
    schedule_matches: usize,
    lockstep_divergences: usize,
    incumbent_non_brackets: usize,
    pass: bool,
    timestamp_ms: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn emit_log(log: &DiffLog) {
    fs::create_dir_all(output_dir()).expect("create bracket_root diff output dir");
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize bracket_root diff log");
    fs::write(path, json).expect("write bracket_root diff log");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

/// A case with every optional knob at its default. Callers override what they mean to vary
/// with struct-update syntax, so each entry in the table below names only the thing it is
/// testing instead of trailing a row of `None`s whose positions have to be counted.
fn case(case_id: &str, kind: Objective, c: f64, xl0: f64) -> Case {
    Case {
        case_id: case_id.to_string(),
        objective: kind.tag().to_string(),
        c,
        xl0,
        xr0: None,
        xmin: None,
        xmax: None,
        factor: None,
        maxiter: 1000,
        kind,
    }
}

fn generate_query() -> OracleQuery {
    use Objective::{Cubic, Linear, NoRoot, Quadratic};
    let points = vec![
        // Unlimited both sides, root reached by the RIGHTWARD search.
        case("right-far", Linear, 100.0, 1.0),
        case("right-huge", Linear, 5000.0, 1.0),
        // Unlimited both sides, root reached by the LEFTWARD search. These are the cases that
        // exercise divergence 1.
        case("left-far", Linear, -50.0, 0.0),
        case("left-deep", Linear, -3000.0, 2.0),
        // The initial pair already brackets, so nit must be 0 on both arms.
        case("immediate", Quadratic, 4.0, 1.0),
        Case {
            xr0: Some(2.5),
            ..case("immediate-interior", Cubic, 8.0, 1.5)
        },
        // A LOWER limit: the leftward search contracts toward xmin instead of expanding.
        Case {
            xmin: Some(0.0),
            ..case("limited-below", Linear, 0.25, 1.0)
        },
        Case {
            xmin: Some(0.0),
            ..case("limited-below-tight", Linear, 0.001, 1.0)
        },
        // An UPPER limit: same, mirrored.
        Case {
            xr0: Some(0.0),
            xmax: Some(1.0),
            ..case("limited-above", Linear, 0.75, -1.0)
        },
        // BOTH limits, so neither side ever expands.
        Case {
            xr0: Some(1.0),
            xmin: Some(-2.0),
            xmax: Some(2.0),
            ..case("limited-both", Cubic, 0.125, -1.0)
        },
        // Non-default expansion factors change the whole schedule.
        Case {
            factor: Some(3.0),
            ..case("factor-3", Linear, 500.0, 1.0)
        },
        Case {
            factor: Some(1.5),
            ..case("factor-1p5", Linear, -200.0, 0.0)
        },
        Case {
            xmin: Some(0.0),
            factor: Some(10.0),
            ..case("factor-10-limited", Linear, 0.05, 1.0)
        },
        // A non-unit starting width. This one also turns out to hit the incumbent's tie
        // defect, without being symmetric.
        Case {
            xr0: Some(4.5),
            ..case("wide-start", Quadratic, 900.0, -0.5)
        },
        // Asymmetric start where the leftward and rightward walks are differently sized.
        Case {
            xr0: Some(-1.0),
            ..case("offset-start", Cubic, -700.0, -3.0)
        },
        // No root at all: both arms must run out of iterations rather than inventing one. The
        // budget is small on purpose — the schedule doubles each iteration, so the default
        // 1000 overflows to infinity long before it ends and the comparison would be between
        // two lists of `inf` rather than between two searches.
        Case {
            maxiter: 12,
            ..case("no-root", NoRoot, 1.0, 0.0)
        },
        // THE TIE: both searches bracket on the same iteration. This is where the incumbent
        // returns a non-bracket, and where this test asserts we do not.
        Case {
            xr0: Some(1.0),
            ..case("simultaneous-tie", Quadratic, 100.0, -1.0)
        },
        Case {
            xr0: Some(1.0),
            ..case("simultaneous-tie-wide", Quadratic, 10000.0, -1.0)
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
    # IEEE-754 bit patterns, because decimal text is NOT a lossless channel for f64 here:
    # serde_json without float_roundtrip rounds some 17-digit decimals to the wrong
    # neighbour, which would surface as a 1-ULP "disagreement" neither arm actually has.
    return [struct.unpack("<Q", struct.pack("<d", float(v)))[0] for v in values]

def objective(tag, c):
    if tag == "linear":
        return lambda x: x - c
    if tag == "quadratic":
        return lambda x: x * x - c
    if tag == "cubic":
        return lambda x: x * x * x - c
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

    kwargs = {}
    for key in ("xr0", "xmin", "xmax", "factor"):
        if case.get(key) is not None:
            kwargs[key] = float(case[key])
    kwargs["maxiter"] = int(case["maxiter"])
    try:
        r = elementwise.bracket_root(f, float(case["xl0"]), **kwargs)
        vals = ([float(v) for v in sampled] + [float(v) for v in r.bracket]
                + [float(v) for v in r.f_bracket])
        if not all(math.isfinite(v) for v in vals):
            # json.dumps would emit bare Infinity/NaN here, which is not valid JSON and which
            # the Rust arm cannot parse. Report it as an error rather than emitting a document
            # that fails to decode with no indication of which case produced it.
            raise ValueError("case produced a non-finite value; lower its maxiter")
        points.append({
            "case_id": cid,
            "sampled_bits": bits(sampled),
            "bracket_bits": bits(r.bracket),
            "f_bracket_bits": bits(r.f_bracket),
            "nit": int(r.nit),
            "nfev": int(r.nfev),
            "success": bool(r.success),
            "error": None,
        })
    except Exception as exc:
        # Recorded, never silently skipped: a case that errors on the oracle side must be
        # visible in the Rust arm's accounting, not quietly dropped from the comparison.
        points.append({
            "case_id": cid, "sampled_bits": None, "bracket_bits": None,
            "f_bracket_bits": None,
            "nit": None, "nfev": None, "success": None,
            "error": f"{type(exc).__name__}: {exc}",
        })
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize bracket_root query");
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
                "failed to spawn python3 for bracket_root oracle: {e}"
            );
            eprintln!("skipping bracket_root oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open bracket_root oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "bracket_root oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping bracket_root oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for bracket_root oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "bracket_root oracle failed: {stderr}"
        );
        eprintln!("skipping bracket_root oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse bracket_root oracle JSON"))
}

/// Render transported bit patterns back to floats.
fn as_floats(bits: &[u64]) -> Vec<f64> {
    bits.iter().copied().map(f64::from_bits).collect()
}

/// Is `pair` genuinely a bracket — opposite signs, or a zero at an endpoint?
fn is_real_bracket(fl: f64, fr: f64) -> bool {
    fl == 0.0 || fr == 0.0 || fl.signum() != fr.signum()
}

/// How our schedule relates to the incumbent's. Returns `Err` with a description when the
/// relationship is anything other than the two permitted ones.
fn classify_schedule(ours: &[f64], theirs: &[u64]) -> Result<&'static str, String> {
    // Compared by BITS, not by `==`: `-0.0 == 0.0` is true in Rust, so `==` would accept a
    // schedule that had lost a sign — and a signed zero is a genuinely different abscissa to
    // hand an objective. `schedule_comparator_rejects_...` pins that case.
    fn same(a: &[f64], b: &[u64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == *y)
    }
    if same(ours, theirs) {
        return Ok("matched");
    }
    if ours.len() + 1 == theirs.len() && same(ours, &theirs[..ours.len()]) {
        // Divergence 1: exactly one trailing evaluation, the rightward point scipy pays for
        // in lockstep after the leftward search has already bracketed.
        return Ok("trailing_lockstep_eval");
    }
    Err(format!(
        "schedule mismatch: ours ({} pts) {:?} is not the incumbent's ({} pts) {:?} \
         nor that minus one trailing point",
        ours.len(),
        ours,
        theirs.len(),
        as_floats(theirs)
    ))
}

#[test]
fn diff_optimize_bracket_root() {
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
    let mut matched = 0usize;
    let mut lockstep = 0usize;
    let mut non_brackets = 0usize;

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
        let (Some(sampled), Some(bracket), Some(f_bracket), Some(nit), Some(nfev), Some(success)) = (
            arm.sampled_bits.as_ref(),
            arm.bracket_bits.as_ref(),
            arm.f_bracket_bits.as_ref(),
            arm.nit,
            arm.nfev,
            arm.success,
        ) else {
            panic!(
                "case {} came back with a null field and no error; a half-populated arm \
                 would compare vacuously",
                case.case_id
            );
        };

        // Our arm, recording its own schedule the same way the oracle records scipy's.
        let visited = std::cell::RefCell::new(Vec::new());
        let kind = case.kind;
        let c = case.c;
        let result = bracket_root(
            |x| {
                visited.borrow_mut().push(x);
                kind.eval(x, c)
            },
            case.xl0,
            BracketOptions {
                xr0: case.xr0,
                xmin: case.xmin,
                xmax: case.xmax,
                factor: case.factor,
                maxiter: case.maxiter,
            },
        )
        .unwrap_or_else(|e| panic!("case {} failed on our arm: {e}", case.case_id));
        let ours = visited.into_inner();

        let verdict = match classify_schedule(&ours, sampled) {
            Ok(v) => v,
            Err(detail) => panic!("case {}: {detail}", case.case_id),
        };
        if verdict == "trailing_lockstep_eval" {
            lockstep += 1;
        } else {
            matched += 1;
        }

        assert_eq!(
            result.success, success,
            "case {}: success disagrees with the incumbent",
            case.case_id
        );
        assert_eq!(
            u64::try_from(result.nit).expect("nit fits u64"),
            nit,
            "case {}: iteration count disagrees with the incumbent",
            case.case_id
        );

        let f_bracket = as_floats(f_bracket);
        let bracket = as_floats(bracket);
        let incumbent_bracket_is_real = !success || is_real_bracket(f_bracket[0], f_bracket[1]);
        let verdict = if incumbent_bracket_is_real {
            // The ordinary case: the endpoints must agree exactly. Ours are ordered ascending,
            // and scipy's are too whenever it has not hit its tie defect.
            let (lo, hi) = if bracket[0] <= bracket[1] {
                (bracket[0], bracket[1])
            } else {
                (bracket[1], bracket[0])
            };
            assert_eq!(
                result.bracket,
                (lo, hi),
                "case {}: bracket differs from the incumbent's",
                case.case_id
            );
            verdict
        } else {
            // Divergence 2: the incumbent reported success on two same-signed endpoints.
            non_brackets += 1;
            assert!(
                result.success && is_real_bracket(result.f_bracket.0, result.f_bracket.1),
                "case {}: the incumbent returned the non-bracket {bracket:?} \
                 (f = {f_bracket:?}); ours must be a REAL bracket, got {:?} (f = {:?})",
                case.case_id,
                result.bracket,
                result.f_bracket
            );
            "incumbent_returned_non_bracket"
        };

        // Whatever we returned, if we claim success it must really straddle a sign change.
        if result.success {
            assert!(
                is_real_bracket(result.f_bracket.0, result.f_bracket.1),
                "case {}: we reported success on the non-bracket {:?}",
                case.case_id,
                result.f_bracket
            );
        }

        compared += 1;
        cases.push(CaseDiff {
            case_id: case.case_id.clone(),
            objective: case.objective.clone(),
            verdict: verdict.to_string(),
            ours_nfev: result.nfev,
            scipy_nfev: nfev,
            pass: true,
        });
    }

    // The artifact is written BEFORE the aggregate gate so a failing split is inspectable
    // rather than only reported as a number in a panic message.
    emit_log(&DiffLog {
        test_id: "diff_optimize_bracket_root".to_string(),
        category: "optimize.elementwise".to_string(),
        case_count: query.points.len(),
        compared_cases: compared,
        schedule_matches: matched,
        lockstep_divergences: lockstep,
        incumbent_non_brackets: non_brackets,
        pass: true,
        timestamp_ms: timestamp_ms(),
        cases,
    });

    // GREEN-BY-SKIPPING GUARD. A run where nothing was compared passes every assertion above
    // by vacuity, so the counts are asserted directly — including that BOTH divergence classes
    // were actually exercised, which is what makes the rules encoding them meaningful.
    eprintln!(
        "bracket_root diff split: compared={compared} matched={matched} \
         lockstep={lockstep} incumbent_non_brackets={non_brackets}"
    );
    assert_eq!(
        compared,
        query.points.len(),
        "every case must be compared, not skipped"
    );
    // Pinned exactly, not as a floor. These are the two arms' agreed behaviour on scipy
    // 1.17.1; a move in EITHER direction means the incumbent changed or we did, and either is
    // something to look at rather than absorb. A floor would silently accept our search
    // quietly drifting into skipping more evaluations than the one it is entitled to skip.
    //
    // `lockstep` counts every case where we saved the trailing evaluation, which includes the
    // three where the incumbent then also reported a non-bracket; `matched + lockstep` is
    // therefore the full case count, and `non_brackets` is a subset. The per-case `verdict` in
    // the artifact carries the final label for each.
    //
    // Three of eighteen ordinary cases hit the tie defect — `simultaneous-tie`,
    // `simultaneous-tie-wide`, and `wide-start`, the last of which is not symmetric
    // (`x**2 - 900` from `(-0.5, 4.5)`). It is a routine outcome for an even function, not a
    // contrived one, which is why we do not reproduce it.
    assert_eq!(
        (matched, lockstep, non_brackets),
        (8, 10, 3),
        "the divergence split changed; expected 8 exact matches, 10 trailing-lockstep cases \
         and 3 incumbent non-brackets"
    );
}

/// MUST-MISS control for the comparator itself. Every assertion in the differential test above
/// runs through `classify_schedule`, so if that function accepted anything it would report a
/// clean pass over schedules that do not match at all. These are the shapes it MUST reject.
#[test]
fn schedule_comparator_rejects_the_differences_it_exists_to_catch() {
    let w = |xs: &[f64]| -> Vec<u64> { xs.iter().map(|x| x.to_bits()).collect() };

    // MUST-HIT: the two permitted relationships.
    assert_eq!(
        classify_schedule(&[1.0, 2.0, 0.0], &w(&[1.0, 2.0, 0.0])),
        Ok("matched")
    );
    assert_eq!(
        classify_schedule(&[1.0, 2.0, 0.0], &w(&[1.0, 2.0, 0.0, 3.0])),
        Ok("trailing_lockstep_eval")
    );

    // MUST-MISS: a point missing from the MIDDLE is a different search, not a saved
    // evaluation, even though it is also "one shorter".
    assert!(classify_schedule(&[1.0, 2.0, 3.0], &w(&[1.0, 2.0, 0.0, 3.0])).is_err());
    // MUST-MISS: two trailing points is a whole skipped iteration.
    assert!(classify_schedule(&[1.0, 2.0], &w(&[1.0, 2.0, 0.0, 3.0])).is_err());
    // MUST-MISS: a single differing value, same length.
    assert!(classify_schedule(&[1.0, 2.0, 0.5], &w(&[1.0, 2.0, 0.0])).is_err());
    // MUST-MISS: ours LONGER than the incumbent's is never permitted.
    assert!(classify_schedule(&[1.0, 2.0, 0.0, 3.0], &w(&[1.0, 2.0, 0.0])).is_err());
    // MUST-MISS: a sign difference the eye skips over.
    assert!(classify_schedule(&[1.0, 2.0, -0.0], &w(&[1.0, 2.0, 0.0])).is_err());
    // MUST-MISS: one ULP, which is the resolution this comparison claims to have. Built by
    // incrementing the bit pattern rather than written as a decimal literal — a hand-typed
    // "0.10000000000000001" rounds back to 0.1 and would have made this control vacuous.
    let one_ulp_up = 0.1f64.to_bits() + 1;
    assert_ne!(one_ulp_up, 0.1f64.to_bits());
    assert!(
        classify_schedule(
            &[1.0, 2.0, 0.1],
            &[1.0f64.to_bits(), 2.0f64.to_bits(), one_ulp_up]
        )
        .is_err()
    );
}

/// MUST-HIT / MUST-MISS control for the bracket predicate, which decides whether the
/// incumbent's answer is treated as authoritative or as its known defect.
#[test]
fn bracket_predicate_separates_real_brackets_from_the_incumbents_tie_defect() {
    assert!(is_real_bracket(-3.0, 1.0), "opposite signs bracket");
    assert!(is_real_bracket(-3.0, 0.0), "a zero endpoint brackets");
    assert!(is_real_bracket(0.0, 0.0));
    // The exact shape scipy 1.17.1 returns for f(x)=x**2-100 from (-1, 1).
    assert!(
        !is_real_bracket(-51.0, -51.0),
        "two same-signed endpoints are not a bracket"
    );
    assert!(!is_real_bracket(2.0, 5.0));
}
