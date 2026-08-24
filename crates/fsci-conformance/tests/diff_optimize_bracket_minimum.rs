#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `fsci_opt::bracket_minimum` against
//! `scipy.optimize.elementwise.bracket_minimum`.
//!
//! STRICTER THAN ITS `bracket_root` SIBLING, on purpose. A root bracket search runs two
//! interleaved searches and scipy's vectorization makes it pay for one evaluation ours can
//! skip; that test therefore permits a single trailing difference. A MINIMUM bracket search is
//! ONE search walking one direction, so there is no lockstep to diverge over and nothing is
//! permitted to differ: the sampling schedule, the returned trio, `nit`, `nfev` and the
//! termination status must all agree exactly.
//!
//! Comparison is on `f64` bits, not `==`, so a `-0.0`/`0.0` difference cannot pass as equality.
//! The objectives are exact polynomial arithmetic in both languages, so any disagreement is the
//! SEARCH differing rather than the arithmetic.
//!
//! ## Floats cross the process boundary as BIT PATTERNS, and they have to
//!
//! `serde_json` without its `float_roundtrip` feature is not correctly rounded on every input.
//! Measured here, not assumed: `serde_json::from_str::<f64>("0.0017144775390624983")` yields
//! `0x3f5c170a3d70a3d0`, one ULP above the `0x3f5c170a3d70a3cf` that both Python's `float()`
//! and Rust's own literal parser produce for that same text. The first version of this test
//! failed on exactly that — a 1-ULP disagreement invented entirely by the transport, in a case
//! where both implementations had computed identical bits.
//!
//! So every float is carried as its IEEE-754 `u64` bit pattern, which JSON represents exactly
//! and both `json` and `serde_json` parse without rounding. Decimal transport would have made
//! this test flaky in one direction and BLIND in the other, since a real 1-ULP difference could
//! equally have been erased. `bit_transport_survives_values_decimal_json_would_corrupt` pins
//! the exact value that exposed it.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use fsci_opt::{MinimumBracketOptions, MinimumBracketStatus, bracket_minimum};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

/// SciPy's integer status codes for this routine, from `scipy.optimize._elementwise_iterative_method`
/// plus the local `_ELIMITS`. Mapped explicitly rather than compared as bare integers so a
/// change in either arm's meaning shows up as a name mismatch instead of an arithmetic one.
const SCIPY_CONVERGED: i64 = 0;
const SCIPY_LIMIT_REACHED: i64 = -1;
const SCIPY_MAX_ITERATIONS: i64 = -2;
const SCIPY_NON_FINITE: i64 = -3;

/// Objectives both arms implement identically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Objective {
    /// `(x - c)**2` — an ordinary unimodal minimum at `c`.
    ShiftedSquare,
    /// `x - c` — monotone, so the search descends forever in one direction.
    Linear,
    /// `-(x - c)**2` — unbounded below; never brackets.
    InvertedSquare,
    /// `c` — a plateau, which is deliberately NOT a bracket.
    Constant,
}

impl Objective {
    fn eval(self, x: f64, c: f64) -> f64 {
        match self {
            Self::ShiftedSquare => (x - c) * (x - c),
            Self::Linear => x - c,
            Self::InvertedSquare => -((x - c) * (x - c)),
            Self::Constant => c,
        }
    }

    fn tag(self) -> &'static str {
        match self {
            Self::ShiftedSquare => "shifted_square",
            Self::Linear => "linear",
            Self::InvertedSquare => "inverted_square",
            Self::Constant => "constant",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    objective: String,
    c: f64,
    xm0: f64,
    xl0: Option<f64>,
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

/// One case's answer from the incumbent. Every float arrives as its IEEE-754 bit pattern —
/// see the module comment for the measured reason decimals cannot be trusted here.
#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    sampled_bits: Option<Vec<u64>>,
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
    fs::create_dir_all(output_dir()).expect("create bracket_minimum diff output dir");
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize bracket_minimum diff log");
    fs::write(path, json).expect("write bracket_minimum diff log");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

/// A case with every optional knob at its default; callers override only what they vary.
fn case(case_id: &str, kind: Objective, c: f64, xm0: f64) -> Case {
    Case {
        case_id: case_id.to_string(),
        objective: kind.tag().to_string(),
        c,
        xm0,
        xl0: None,
        xr0: None,
        xmin: None,
        xmax: None,
        factor: None,
        maxiter: 1000,
        kind,
    }
}

fn generate_query() -> OracleQuery {
    use Objective::{Constant, InvertedSquare, Linear, ShiftedSquare};
    let points = vec![
        // The initial trio already brackets: nit must be 0 on both arms.
        case("immediate", ShiftedSquare, 0.0, 0.0),
        // Walking RIGHT, with the steps measured from a fixed anchor. Getting the anchor wrong
        // (compounding off the moving end instead) changes the schedule from 1,2,4,8,16 to
        // something that still converges, which is exactly why the schedule is compared.
        case("walk-right", ShiftedSquare, 10.0, 1.0),
        case("walk-right-far", ShiftedSquare, 300.0, 1.0),
        // Walking LEFT. The result must still be reported ascending.
        case("walk-left", ShiftedSquare, -8.0, 0.0),
        case("walk-left-far", ShiftedSquare, -250.0, 2.0),
        // A LOWER limit, which also exercises the limit-aware default endpoint
        // `xl0 = xm0 - min((xm0 - xmin)/16, 0.5)` rather than a flat 0.5 back-off.
        Case {
            xmin: Some(0.0),
            ..case("limited-below", ShiftedSquare, 0.1, 1.0)
        },
        Case {
            xmin: Some(-1.0),
            ..case("limited-below-loose", ShiftedSquare, -0.5, 3.0)
        },
        // An UPPER limit, mirrored.
        Case {
            xmax: Some(1.0),
            ..case("limited-above", ShiftedSquare, 0.9, 0.0)
        },
        // BOTH limits.
        Case {
            xmin: Some(-4.0),
            xmax: Some(4.0),
            ..case("limited-both", ShiftedSquare, 3.0, 0.0)
        },
        // Explicit asymmetric starting trio instead of the defaults.
        Case {
            xl0: Some(-2.0),
            xr0: Some(0.5),
            ..case("explicit-trio", ShiftedSquare, 7.0, 0.0)
        },
        // Non-default factors reshape the whole schedule.
        Case {
            factor: Some(3.0),
            ..case("factor-3", ShiftedSquare, 100.0, 0.0)
        },
        Case {
            factor: Some(1.25),
            ..case("factor-1p25", ShiftedSquare, -40.0, 0.0)
        },
        Case {
            xmin: Some(0.0),
            factor: Some(4.0),
            ..case("factor-4-limited", ShiftedSquare, 0.02, 1.0)
        },
        // DESCENDING INTO A BOUND. At the default budget this is max-iterations; it only lands
        // on the bound after the gap underflows, which takes 1075 iterations. Both outcomes are
        // covered so the status mapping is exercised in each direction.
        Case {
            xmin: Some(0.0),
            maxiter: 40,
            ..case("into-bound-budgeted", Linear, 0.0, 1.0)
        },
        Case {
            xmin: Some(0.0),
            maxiter: 1200,
            ..case("into-bound-reached", Linear, 0.0, 1.0)
        },
        // A PLATEAU is deliberately not a bracket: `fm <= fl && fm <= fr` holds everywhere, but
        // scipy's condition requires a STRICT inequality on one side, and so do we.
        Case {
            maxiter: 10,
            ..case("plateau", Constant, 1.0, 0.0)
        },
        // Unbounded below: the search runs away and must exhaust its budget rather than
        // inventing a bracket. Kept short so the schedule stays finite and the JSON stays valid.
        Case {
            maxiter: 15,
            ..case("runaway", InvertedSquare, 0.0, 1.0)
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
    # IEEE-754 bit patterns, because decimal text is not a lossless channel for f64 here:
    # serde_json (without float_roundtrip) rounds some 17-digit decimals to the wrong
    # neighbour, which would show up as a 1-ULP "disagreement" neither arm actually has.
    return [struct.unpack('<Q', struct.pack('<d', float(v)))[0] for v in values]

def objective(tag, c):
    if tag == "shifted_square":
        return lambda x: (x - c) * (x - c)
    if tag == "linear":
        return lambda x: x - c
    if tag == "inverted_square":
        return lambda x: -((x - c) * (x - c))
    if tag == "constant":
        return lambda x: np.full_like(np.asarray(x, dtype=float), c)
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
    for key in ("xl0", "xr0", "xmin", "xmax", "factor"):
        if case.get(key) is not None:
            kwargs[key] = float(case[key])
    kwargs["maxiter"] = int(case["maxiter"])
    try:
        r = elementwise.bracket_minimum(f, float(case["xm0"]), **kwargs)
        vals = ([float(v) for v in sampled] + [float(v) for v in r.bracket]
                + [float(v) for v in r.f_bracket])
        if not all(math.isfinite(v) for v in vals):
            # json.dumps would emit bare Infinity/NaN, which is not valid JSON. Fail loudly
            # with the case named rather than producing a document that cannot be decoded.
            raise ValueError("case produced a non-finite value; lower its maxiter")
        points.append({
            "case_id": cid,
            "sampled_bits": bits(sampled),
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
            "case_id": cid, "sampled_bits": None, "bracket_bits": None,
            "f_bracket_bits": None,
            "nit": None, "nfev": None, "status": None, "success": None,
            "error": f"{type(exc).__name__}: {exc}",
        })
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize bracket_minimum query");
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
                "failed to spawn python3 for bracket_minimum oracle: {e}"
            );
            eprintln!("skipping bracket_minimum oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open bracket_minimum oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "bracket_minimum oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping bracket_minimum oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for bracket_minimum oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "bracket_minimum oracle failed: {stderr}"
        );
        eprintln!("skipping bracket_minimum oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse bracket_minimum oracle JSON"))
}

/// Bitwise sequence equality against the incumbent's transported bit patterns. `==` on `f64`
/// would accept `-0.0` where `0.0` was expected, which is a genuinely different abscissa to
/// hand an objective.
fn same_bits(ours: &[f64], theirs: &[u64]) -> bool {
    ours.len() == theirs.len() && ours.iter().zip(theirs).all(|(x, y)| x.to_bits() == *y)
}

/// Render transported bits back to floats, for failure messages only.
fn as_floats(bits: &[u64]) -> Vec<f64> {
    bits.iter().copied().map(f64::from_bits).collect()
}

fn status_name(status: MinimumBracketStatus) -> &'static str {
    match status {
        MinimumBracketStatus::Converged => "converged",
        MinimumBracketStatus::LimitReached => "limit_reached",
        MinimumBracketStatus::MaxIterations => "max_iterations",
        MinimumBracketStatus::NonFinite => "non_finite",
    }
}

fn scipy_status_name(status: i64) -> String {
    match status {
        SCIPY_CONVERGED => "converged".to_string(),
        SCIPY_LIMIT_REACHED => "limit_reached".to_string(),
        SCIPY_MAX_ITERATIONS => "max_iterations".to_string(),
        SCIPY_NON_FINITE => "non_finite".to_string(),
        other => format!("unmapped({other})"),
    }
}

#[test]
fn diff_optimize_bracket_minimum() {
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
            Some(bracket),
            Some(f_bracket),
            Some(nit),
            Some(nfev),
            Some(status),
            Some(success),
        ) = (
            arm.sampled_bits.as_ref(),
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
        let result = bracket_minimum(
            |x| {
                visited.borrow_mut().push(x);
                kind.eval(x, c)
            },
            case.xm0,
            MinimumBracketOptions {
                xl0: case.xl0,
                xr0: case.xr0,
                xmin: case.xmin,
                xmax: case.xmax,
                factor: case.factor,
                maxiter: case.maxiter,
            },
        )
        .unwrap_or_else(|e| panic!("case {} failed on our arm: {e}", case.case_id));
        let ours = visited.into_inner();

        // NOTHING is permitted to differ here — see the module comment.
        assert!(
            same_bits(&ours, sampled),
            "case {}: sampling schedule differs.\n  ours ({} pts): {:?}\n  scipy ({} pts): {:?}",
            case.case_id,
            ours.len(),
            &ours[..ours.len().min(24)],
            sampled.len(),
            &as_floats(&sampled[..sampled.len().min(24)])
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

        // A converged trio must really be a bracket, so `success` is worth trusting.
        if result.success {
            let (fl, fm, fr) = result.f_bracket;
            assert!(
                (fl >= fm && fr > fm) || (fl > fm && fr >= fm),
                "case {}: reported success on the non-bracket {:?}",
                case.case_id,
                result.f_bracket
            );
            let (xl, xm, xr) = result.bracket;
            assert!(
                xl < xm && xm < xr,
                "case {}: bracket is not strictly ordered: {:?}",
                case.case_id,
                result.bracket
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
        test_id: "diff_optimize_bracket_minimum".to_string(),
        category: "optimize.elementwise".to_string(),
        case_count: query.points.len(),
        compared_cases: compared,
        total_samples_compared: total_samples,
        statuses_seen: statuses_seen.clone(),
        pass: true,
        timestamp_ms: timestamp_ms(),
        cases,
    });

    // GREEN-BY-SKIPPING GUARD. Every assertion above passes by vacuity on an empty comparison,
    // so what was actually compared is asserted directly.
    eprintln!(
        "bracket_minimum diff: compared={compared} samples={total_samples} \
         statuses={statuses_seen:?}"
    );
    assert_eq!(
        compared,
        query.points.len(),
        "every case must be compared, not skipped"
    );
    assert!(
        total_samples > 1000,
        "only {total_samples} abscissae compared; the deep cases did not run"
    );
    // All three reachable termination statuses must appear, or the mapping is only half tested.
    assert_eq!(
        statuses_seen,
        vec![
            "converged".to_string(),
            "limit_reached".to_string(),
            "max_iterations".to_string()
        ],
        "every reachable termination status must be exercised"
    );
}

/// MUST-HIT / MUST-MISS control for the bitwise comparator every assertion above depends on.
#[test]
fn bit_comparator_rejects_the_differences_it_exists_to_catch() {
    let bits = |xs: &[f64]| -> Vec<u64> { xs.iter().map(|x| x.to_bits()).collect() };

    assert!(same_bits(&[1.0, 2.0, 0.5], &bits(&[1.0, 2.0, 0.5])));
    assert!(same_bits(&[], &bits(&[])));

    assert!(
        !same_bits(&[1.0, 2.0], &bits(&[1.0, 2.0, 3.0])),
        "length differs"
    );
    assert!(
        !same_bits(&[1.0, 2.0, 3.0], &bits(&[1.0, 2.0])),
        "length differs"
    );
    assert!(!same_bits(&[1.0, 2.5], &bits(&[1.0, 2.0])), "value differs");
    // The one a plain `==` would wave through: `-0.0 == 0.0` is true in Rust.
    assert!(
        !same_bits(&[0.0], &bits(&[-0.0])),
        "a lost sign on zero must not compare equal"
    );
    // And the one `==` rejects even when the bits agree.
    assert!(same_bits(&[f64::NAN], &bits(&[f64::NAN])));
    // A one-ULP difference is exactly the size this test must be able to see.
    assert!(!same_bits(
        &[0.001_714_477_539_062_498_3],
        &[0x3f5c_170a_3d70_a3d0]
    ));
}

/// The transport bug that made the first version of this test fail, pinned as a control.
///
/// Both arms had computed identical bits; `serde_json` invented the difference while parsing
/// the incumbent's decimal text. If this ever starts passing — because the dependency gained
/// `float_roundtrip`, say — the bit transport is still correct and this test simply documents
/// history; if it keeps failing, the transport is still load-bearing.
#[test]
fn bit_transport_survives_values_decimal_json_would_corrupt() {
    const TEXT: &str = "0.0017144775390624983";
    let via_rust_literal: f64 = 0.001_714_477_539_062_498_3;
    let via_json: f64 = serde_json::from_str(TEXT).expect("parse a bare JSON float");

    assert_eq!(
        via_rust_literal.to_bits(),
        0x3f5c_170a_3d70_a3cf,
        "Rust's own parser is correctly rounded for this text"
    );
    assert_ne!(
        via_json.to_bits(),
        via_rust_literal.to_bits(),
        "if serde_json ever becomes correctly rounded here, decimal transport would be safe \
         again — but the bit transport this test uses does not depend on that"
    );

    // The channel this test actually uses is exact for the same value.
    let transported: Vec<u64> =
        serde_json::from_str(&format!("[{}]", via_rust_literal.to_bits())).expect("parse bits");
    assert!(same_bits(&[via_rust_literal], &transported));
}

/// MUST-MISS control for the status mapping: an unmapped code must be visibly unmapped rather
/// than silently colliding with a real status name.
#[test]
fn status_mapping_names_every_code_it_claims_and_flags_the_rest() {
    assert_eq!(scipy_status_name(SCIPY_CONVERGED), "converged");
    assert_eq!(scipy_status_name(SCIPY_LIMIT_REACHED), "limit_reached");
    assert_eq!(scipy_status_name(SCIPY_MAX_ITERATIONS), "max_iterations");
    assert_eq!(scipy_status_name(SCIPY_NON_FINITE), "non_finite");
    assert_eq!(scipy_status_name(-99), "unmapped(-99)");
    assert_eq!(status_name(MinimumBracketStatus::Converged), "converged");
    assert_eq!(
        status_name(MinimumBracketStatus::LimitReached),
        "limit_reached"
    );
}
