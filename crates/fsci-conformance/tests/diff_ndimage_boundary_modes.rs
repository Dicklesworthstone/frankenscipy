#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `fsci_ndimage::BoundaryMode` semantics under
//! INTERPOLATION, against `scipy.ndimage`.
//!
//! ## Why this exists: a surface diff cannot see a missing MODE
//!
//! Every function in `scipy.ndimage` is present in this workspace — 76 of 76 by name. But
//! `scipy.ndimage` accepts EIGHT boundary modes and `fsci_ndimage::BoundaryMode` has five, and
//! a name-level surface diff is blind to that. The three extra SciPy spellings are
//! `grid-constant`, `grid-mirror` and `grid-wrap`.
//!
//! For FILTERS they are pure aliases — measured on scipy 1.17.1 with `correlate1d`,
//! `wrap ≡ grid-wrap`, `reflect ≡ grid-mirror`, `constant ≡ grid-constant`, all bit-identical —
//! so the five modes here cover all eight and nothing is missing on that path.
//!
//! For INTERPOLATION they are NOT aliases. Measured with `shift(a, 0.5, order=1)` on
//! `[1, 2, 3, 4, 5]`, the leading sample differs:
//!
//! ```text
//!     wrap           4.5      grid-wrap        3.0
//!     constant       0.0      grid-constant    0.5
//!     reflect        1.0      grid-mirror      1.0     (these two ARE the same)
//! ```
//!
//! So two genuine behaviours are unreachable from this crate, and — the part a surface diff
//! would never surface — the mode we spell `Wrap` has to be pinned to whichever SciPy spelling
//! it actually implements, rather than to the one it shares a name with.
//!
//! ## This test states the mapping and proves it
//!
//! `EXPECTED_MAPPING` below names, for each of our modes, the SciPy mode it is claimed to
//! equal. The test asserts that claim against the live incumbent for every case, and — the part
//! that makes it more than a rubber stamp — ALSO asserts that our mode differs from the SciPy
//! spellings it is claimed NOT to equal, wherever the incumbent itself distinguishes them. A
//! mapping that merely matched something would otherwise pass while pointing at the wrong row.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{SystemTime, UNIX_EPOCH};

use fsci_ndimage::{BoundaryMode, NdArray, shift};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

/// Interpolation is exact arithmetic on both sides for these orders, but the two arms reach it
/// by different spline paths, so this is relative rather than bitwise.
const REL_TOL: f64 = 1e-11;

/// Every boundary mode SciPy accepts.
const SCIPY_MODES: [&str; 8] = [
    "reflect",
    "grid-mirror",
    "constant",
    "grid-constant",
    "nearest",
    "mirror",
    "wrap",
    "grid-wrap",
];

/// Our modes, and the SciPy spelling each is claimed to implement.
///
/// `Wrap` targets SciPy's `wrap`, whose period is `n - 1` with the endpoints identified — NOT
/// `grid-wrap`, whose period is `n`. I first claimed `grid-wrap` after reading
/// `BoundaryMode::Wrap => Some(i.rem_euclid(n))` at `lib.rs:3838`; that line is the FILTER index
/// path, and the interpolation path is a different one whose own comment states the `n - 1`
/// intent. Measuring against all eight SciPy spellings is what settled it. Reading picked the
/// wrong row; the incumbent picked the right one.
const EXPECTED_MAPPING: [(&str, &str); 5] = [
    ("Reflect", "reflect"),
    ("Constant", "constant"),
    ("Nearest", "nearest"),
    ("Mirror", "mirror"),
    ("Wrap", "wrap"),
];

/// Cells where this crate does NOT match the incumbent today, measured not assumed.
///
/// THIS IS A RATCHET, NOT A WAIVER. The test asserts every listed cell still diverges and every
/// unlisted cell still matches, so it fails in BOTH directions: a regression in a working cell,
/// and a FIX of a broken one. Fixing the defect therefore requires deleting the row, which is
/// the point — a known-divergence list nobody is forced to update is just a way of forgetting.
///
/// The shape of the list is the diagnosis. Eighteen of the twenty-three sit at spline orders
/// 2, 4 and 5, which are precisely the orders the interpolation path leaves to a generic
/// `make_interp_spline` coefficient solve. Orders 0 and 1 need no prefilter at all, and order 3
/// has its own hand-written cardinal cubic path (`cubic_constant_wrap_coefficients`), which is
/// why those largely work. SciPy instead prefilters with a mode-aware recursive B-spline filter
/// for every order, so ours diverges wherever the boundary is reached and the order is one
/// nobody special-cased. `Nearest` never appears below: clamping is order-independent, so it is
/// immune by construction.
const KNOWN_DIVERGENCES_TO_FIX: [(&str, &str); 15] = [
    // Order 0 tie-breaks: scipy maps the coordinate THEN rounds, we round then map, so the two
    // disagree at exact half-integers.
    ("1d-5-order0", "Mirror"),
    ("1d-5-order0", "Wrap"),
    // Orders 2, 4, 5 — the generic prefilter path.
    ("1d-5-order2", "Constant"),
    ("1d-5-order2", "Wrap"),
    ("1d-8-neg-order2", "Constant"),
    ("1d-8-neg-order2", "Wrap"),
    ("1d-5-order4", "Constant"),
    ("1d-5-order4", "Wrap"),
    ("1d-8-neg-order4", "Constant"),
    ("1d-8-neg-order4", "Wrap"),
    ("1d-5-order5", "Constant"),
    ("1d-5-order5", "Reflect"),
    ("1d-5-order5", "Wrap"),
    ("1d-8-neg-order5", "Constant"),
    ("1d-8-neg-order5", "Wrap"),
];
// The eight Reflect entries removed on 2026-09-04: the short-axis stencil
// degradation (frankenscipy-0y8z8) was their cause — at full order with the
// per-tap fold, Reflect matches the incumbent at orders 3, 4 and 5 on every
// case the ratchet previously excused.

/// Cells where this crate REFUSES an input the incumbent accepts. EMPTY since
/// 2026-09-04: the sole entry ("1d-5-order5", "Mirror" — a length-5 axis at
/// spline order 5, which scipy computes and we refused with "mirror boundary
/// requires each axis length > spline order") was produced by the fail-closed
/// removed with frankenscipy-0y8z8; the case now interpolates and matches.
const KNOWN_REFUSALS_TO_FIX: [(&str, &str); 0] = [];

fn is_known_divergence(case_id: &str, our_mode: &str) -> bool {
    KNOWN_DIVERGENCES_TO_FIX
        .iter()
        .any(|(c, m)| *c == case_id && *m == our_mode)
}

fn is_known_refusal(case_id: &str, our_mode: &str) -> bool {
    KNOWN_REFUSALS_TO_FIX
        .iter()
        .any(|(c, m)| *c == case_id && *m == our_mode)
}

fn our_mode(name: &str) -> BoundaryMode {
    match name {
        "Reflect" => BoundaryMode::Reflect,
        "Constant" => BoundaryMode::Constant,
        "Nearest" => BoundaryMode::Nearest,
        "Mirror" => BoundaryMode::Mirror,
        "Wrap" => BoundaryMode::Wrap,
        other => panic!("unknown mode name {other}"),
    }
}

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    shape: Vec<usize>,
    data_bits: Vec<u64>,
    shift_bits: Vec<u64>,
    order: usize,
    cval_bits: u64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// One entry per SciPy mode, in `SCIPY_MODES` order; each is the flattened result.
    results_bits: Option<Vec<Vec<u64>>>,
    error: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct ModeFinding {
    our_mode: String,
    claimed_scipy_mode: String,
    matched: bool,
    max_relative_difference: f64,
    /// SciPy modes this one is distinguishable from on this case, proving the claim is not
    /// merely one of several that would have fit.
    distinguished_from: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    shape: Vec<usize>,
    order: usize,
    findings: Vec<ModeFinding>,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared_cases: usize,
    total_comparisons: usize,
    discriminating_comparisons: usize,
    max_relative_difference: f64,
    pass: bool,
    timestamp_ms: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn emit_log(log: &DiffLog) {
    fs::create_dir_all(output_dir()).expect("create ndimage boundary diff output dir");
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize ndimage boundary diff log");
    fs::write(path, json).expect("write ndimage boundary diff log");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn as_floats(b: &[u64]) -> Vec<f64> {
    b.iter().copied().map(f64::from_bits).collect()
}

fn case(case_id: &str, shape: Vec<usize>, order: usize, shifts: Vec<f64>, cval: f64) -> Case {
    let total: usize = shape.iter().product();
    // A deterministic ramp with structure, so a wrong boundary shows up as a wrong VALUE and
    // not merely as a different rounding of the same one.
    let data: Vec<f64> = (0..total)
        .map(|i| (i as f64) + 1.0 + 0.25 * ((i * 3) as f64).sin())
        .collect();
    Case {
        case_id: case_id.to_string(),
        shape,
        data_bits: bits(&data),
        shift_bits: bits(&shifts),
        order,
        cval_bits: cval.to_bits(),
    }
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    // Orders 0 through 5, since the boundary handling differs by spline order and the
    // cubic path in particular is special-cased in this crate.
    for order in 0..=5 {
        points.push(case(
            &format!("1d-5-order{order}"),
            vec![5],
            order,
            vec![0.5],
            0.0,
        ));
        points.push(case(
            &format!("1d-8-neg-order{order}"),
            vec![8],
            order,
            vec![-1.25],
            0.0,
        ));
    }
    // Larger shifts, which push further into the extension region where the modes diverge most.
    points.push(case("1d-6-big", vec![6], 1, vec![3.5], 0.0));
    points.push(case("1d-6-bign<", vec![6], 3, vec![-4.5], 0.0));
    // A non-zero cval, which separates constant from grid-constant more sharply.
    points.push(case("1d-5-cval", vec![5], 1, vec![0.5], 7.5));
    points.push(case("1d-5-cval-o3", vec![5], 3, vec![1.5], -2.25));
    // Two dimensions, where each axis is wrapped independently.
    points.push(case("2d-4x5", vec![4, 5], 1, vec![0.5, -0.5], 0.0));
    points.push(case("2d-4x5-o3", vec![4, 5], 3, vec![1.25, 0.75], 0.0));
    points.push(case("2d-3x3-big", vec![3, 3], 1, vec![2.5, -2.5], 0.0));
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import struct
import sys
import numpy as np
import scipy.ndimage as nd

MODES = ["reflect", "grid-mirror", "constant", "grid-constant",
         "nearest", "mirror", "wrap", "grid-wrap"]

def bits(values):
    return [struct.unpack("<Q", struct.pack("<d", float(v)))[0] for v in values]

def unbits(values):
    return [struct.unpack("<d", struct.pack("<Q", int(v)))[0] for v in values]

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    try:
        shape = tuple(int(s) for s in case["shape"])
        a = np.array(unbits(case["data_bits"]), dtype=float).reshape(shape)
        sh = unbits(case["shift_bits"])
        order = int(case["order"])
        cval = unbits([case["cval_bits"]])[0]
        results = []
        for m in MODES:
            r = nd.shift(a, sh, order=order, mode=m, cval=cval)
            flat = [float(v) for v in np.asarray(r).ravel()]
            if not all(math.isfinite(v) for v in flat):
                raise ValueError(f"mode {m} produced a non-finite value")
            results.append(bits(flat))
        points.append({"case_id": cid, "results_bits": results, "error": None})
    except Exception as exc:
        points.append({
            "case_id": cid, "results_bits": None,
            "error": f"{type(exc).__name__}: {exc}",
        })
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize ndimage boundary query");
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
                "failed to spawn python3 for ndimage boundary oracle: {e}"
            );
            eprintln!("skipping ndimage boundary oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open ndimage boundary oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "ndimage boundary oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping ndimage boundary oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for ndimage boundary oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "ndimage boundary oracle failed: {stderr}"
        );
        eprintln!("skipping ndimage boundary oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse ndimage boundary oracle JSON"))
}

/// Largest relative difference, scaled by the magnitude present. Infinite on a length mismatch.
fn max_relative_difference(ours: &[f64], theirs: &[f64]) -> f64 {
    if ours.len() != theirs.len() {
        return f64::INFINITY;
    }
    let scale = theirs
        .iter()
        .fold(0.0f64, |acc, v| acc.max(v.abs()))
        .max(1.0);
    ours.iter()
        .zip(theirs)
        .fold(0.0f64, |acc, (a, b)| acc.max((a - b).abs() / scale))
}

#[test]
fn diff_ndimage_boundary_modes() {
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
    let mut comparisons = 0usize;
    let mut discriminating = 0usize;
    let mut worst = 0.0f64;
    let mut mismatches: Vec<String> = Vec::new();
    let mut divergences_seen = 0usize;
    let mut refusals_seen = 0usize;
    let mut agreements = 0usize;

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
        let Some(results) = arm.results_bits.as_ref() else {
            panic!(
                "case {} came back with a null field and no error; a half-populated arm \
                 would compare vacuously",
                case.case_id
            );
        };
        assert_eq!(
            results.len(),
            SCIPY_MODES.len(),
            "case {}: oracle must answer every mode",
            case.case_id
        );

        let data = as_floats(&case.data_bits);
        let shifts = as_floats(&case.shift_bits);
        let cval = f64::from_bits(case.cval_bits);
        let input = NdArray::new(data, case.shape.clone()).expect("valid array");

        let mut findings = Vec::new();
        for (our_name, claimed) in EXPECTED_MAPPING {
            // A refusal is itself a divergence worth recording: SciPy accepts every
            // (order, mode, shape) combination in this query, so an error here is a capability
            // restriction we impose and the incumbent does not.
            let ours_flat = match shift(&input, &shifts, case.order, our_mode(our_name), cval) {
                Ok(result) => result.data,
                Err(error) => {
                    // SciPy accepts every combination in this query, so a refusal is a
                    // restriction we impose. Ratcheted like a divergence: an unlisted refusal
                    // fails, and a listed one that starts succeeding fails too.
                    if !is_known_refusal(&case.case_id, our_name) {
                        mismatches.push(format!(
                            "case {} [shape {:?}, order {}]: BoundaryMode::{our_name} REFUSED \
                             where scipy '{claimed}' succeeded, and it is NOT in \
                             KNOWN_REFUSALS_TO_FIX: {error}",
                            case.case_id, case.shape, case.order
                        ));
                    } else {
                        refusals_seen += 1;
                    }
                    comparisons += 1;
                    continue;
                }
            };

            let claimed_index = SCIPY_MODES
                .iter()
                .position(|m| *m == claimed)
                .expect("claimed mode is one scipy accepts");
            let claimed_values = as_floats(&results[claimed_index]);
            let difference = max_relative_difference(&ours_flat, &claimed_values);
            let known = is_known_divergence(&case.case_id, our_name);
            // Collected rather than panicked on, so ONE run reports the whole picture: failing
            // at the first cell hides how widespread a divergence is, which is the first thing
            // worth knowing about one.
            if difference > REL_TOL && !known {
                mismatches.push(format!(
                    "REGRESSION case {} [shape {:?}, order {}]: BoundaryMode::{our_name} vs \
                     scipy '{claimed}' differs by {difference:e} and is NOT a known \
                     divergence\n    ours:  {:?}\n    scipy: {:?}",
                    case.case_id,
                    case.shape,
                    case.order,
                    &ours_flat[..ours_flat.len().min(6)],
                    &claimed_values[..claimed_values.len().min(6)]
                ));
            } else if difference <= REL_TOL && known {
                // The good failure: someone fixed it. The ledger row must go, or the next
                // reader will believe a defect exists that does not.
                mismatches.push(format!(
                    "FIXED case {} [shape {:?}, order {}]: BoundaryMode::{our_name} now MATCHES \
                     scipy '{claimed}'. Remove (\"{}\", \"{our_name}\") from \
                     KNOWN_DIVERGENCES_TO_FIX and drop its length by one.",
                    case.case_id, case.shape, case.order, case.case_id
                ));
            }
            if known {
                divergences_seen += 1;
            } else {
                agreements += 1;
            }
            worst = worst.max(difference);
            comparisons += 1;

            // THE PART THAT MAKES THE MAPPING MEAN SOMETHING. Matching one row proves little
            // if several rows are identical on this case. Record every OTHER scipy mode that
            // the incumbent itself distinguishes from the claimed one, and require ours to
            // differ from those too.
            let mut distinguished = Vec::new();
            for (index, other) in SCIPY_MODES.iter().enumerate() {
                if index == claimed_index {
                    continue;
                }
                let other_values = as_floats(&results[index]);
                let incumbent_separates =
                    max_relative_difference(&claimed_values, &other_values) > REL_TOL;
                if !incumbent_separates {
                    // scipy itself treats these as the same here, so there is nothing to
                    // discriminate and no claim to make.
                    continue;
                }
                let ours_vs_other = max_relative_difference(&ours_flat, &other_values);
                if ours_vs_other <= REL_TOL && !known {
                    mismatches.push(format!(
                        "case {}: BoundaryMode::{our_name} is claimed to be '{claimed}' but \
                         ALSO matches '{other}', which scipy distinguishes from '{claimed}'; \
                         the mapping is ambiguous here",
                        case.case_id
                    ));
                    continue;
                }
                distinguished.push((*other).to_string());
                discriminating += 1;
            }

            findings.push(ModeFinding {
                our_mode: our_name.to_string(),
                claimed_scipy_mode: claimed.to_string(),
                matched: difference <= REL_TOL,
                max_relative_difference: difference,
                distinguished_from: distinguished,
            });
        }

        compared += 1;
        cases.push(CaseDiff {
            case_id: case.case_id.clone(),
            shape: case.shape.clone(),
            order: case.order,
            findings,
            pass: true,
        });
    }

    emit_log(&DiffLog {
        test_id: "diff_ndimage_boundary_modes".to_string(),
        category: "ndimage".to_string(),
        case_count: query.points.len(),
        compared_cases: compared,
        total_comparisons: comparisons,
        discriminating_comparisons: discriminating,
        max_relative_difference: worst,
        pass: true,
        timestamp_ms: timestamp_ms(),
        cases,
    });

    // GREEN-BY-SKIPPING GUARD, plus a guard that the mapping was actually pinned down rather
    // than merely not contradicted.
    eprintln!(
        "ndimage boundary diff: compared={compared} comparisons={comparisons} \
         discriminating={discriminating} max_rel_diff={worst:e}"
    );
    assert!(
        mismatches.is_empty(),
        "{} mode/case comparisons disagree with the incumbent:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
    assert_eq!(
        compared,
        query.points.len(),
        "every case must be compared, not skipped"
    );
    assert_eq!(
        comparisons,
        compared * EXPECTED_MAPPING.len(),
        "every mode must be compared on every case"
    );
    assert!(
        discriminating > 100,
        "only {discriminating} discriminating comparisons; the mapping is not pinned down"
    );
    // EVERY ledger row must have been reached. A stale row naming a case that no longer
    // exists would otherwise sit there forever looking like a live defect.
    assert_eq!(
        divergences_seen,
        KNOWN_DIVERGENCES_TO_FIX.len(),
        "the known-divergence ledger has {} rows but {divergences_seen} were exercised; a row \
         names a case/mode this query no longer runs",
        KNOWN_DIVERGENCES_TO_FIX.len()
    );
    assert_eq!(
        refusals_seen,
        KNOWN_REFUSALS_TO_FIX.len(),
        "the known-refusal ledger has {} rows but {refusals_seen} were exercised",
        KNOWN_REFUSALS_TO_FIX.len()
    );
    assert_eq!(
        agreements + divergences_seen + refusals_seen,
        comparisons,
        "every comparison must be accounted for as agreement, known divergence or known refusal"
    );
    assert!(
        agreements >= 70,
        "only {agreements} cells verified against the incumbent; coverage has shrunk"
    );
}

/// MUST-HIT / MUST-MISS control for the comparator.
#[test]
fn relative_comparator_rejects_the_differences_it_exists_to_catch() {
    assert_eq!(max_relative_difference(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
    assert!(max_relative_difference(&[1.0, 2.0], &[1.0, 2.0 + 1e-15]) < REL_TOL);
    assert!(
        max_relative_difference(&[4.5], &[3.0]) > REL_TOL,
        "wrap vs grid-wrap"
    );
    assert!(
        max_relative_difference(&[0.0], &[0.5]) > REL_TOL,
        "constant vs grid-constant"
    );
    assert_eq!(
        max_relative_difference(&[1.0], &[1.0, 2.0]),
        f64::INFINITY,
        "a shape mismatch must never read as agreement"
    );
}
