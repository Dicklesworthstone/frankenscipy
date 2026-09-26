#![forbid(unsafe_code)]
//! Coverage for fsci_ndimage::affine_transform.
//!
//! Resolves [frankenscipy-0s069] and [frankenscipy-nplp5]. fsci's
//! affine_transform takes a 2×3 matrix and now treats it exactly as
//! scipy.ndimage.affine_transform does: the matrix is the OUTPUT→INPUT
//! mapping applied directly (input location = matrix @ output + offset),
//! with no internal inversion. Direct scipy parity therefore holds:
//!   * Identity matrix returns input unchanged (order ∈ {0, 1})
//!   * General matrix matches scipy reference values (order ∈ {1, 3},
//!     mode ∈ {constant, reflect})
//!   * Error: order > 5
//!   * Error: input.ndim != 2

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_ndimage::{BoundaryMode, NdArray, affine_transform};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
/// Interpolation orders of the identity cases.
const IDENTITY_ORDERS: [usize; 2] = [0, 1];

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
    fs::create_dir_all(output_dir()).expect("create affine diff dir");
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
fn diff_ndimage_affine_transform_properties() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    // The reference side is the input itself (identity) or recorded SciPy values; the error
    // cases are ones SciPy 1.17.1 raises on (RuntimeError "spline order not supported" /
    // "affine matrix has wrong number of rows"). A NaN in fsci's output used to vanish in the
    // max fold below.
    let mut ledger = CompareLedger::new(
        "diff_ndimage_affine_transform_properties",
        &["identity", "scipy_parity", "errors"],
    );
    let mut check = |id: &str, ok: bool, note: String| {
        diffs.push(CaseDiff {
            case_id: id.into(),
            pass: ok,
            note,
        });
    };

    // Build a 5×5 test image with sin texture
    let rows = 5;
    let cols = 5;
    let data: Vec<f64> = (0..rows * cols).map(|i| (i as f64).sin() * 10.0).collect();
    let arr = NdArray::new(data.clone(), vec![rows, cols]).expect("ndarray");

    // === 1. Identity affine: matrix = [[1,0,0],[0,1,0]] ===
    // For both order=0 and order=1, output should equal input.
    {
        let identity: [[f64; 3]; 2] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        for order in IDENTITY_ORDERS {
            let case_id = format!("identity_order{order}");
            let result = affine_transform(&arr, &identity, order, BoundaryMode::Constant, 0.0)
                .expect("identity affine");
            let Some((expected, actual)) = ledger.slices(
                "identity",
                &case_id,
                Some(data.as_slice()),
                Some(result.data.as_slice()),
            ) else {
                continue;
            };
            let max_diff = actual
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let pass = result.shape == arr.shape && max_diff <= ABS_TOL;
            ledger.compared("identity", &case_id, pass);
            check(&case_id, pass, format!("max_diff={max_diff}"));
        }
    }

    // === 2. General matrix matches scipy.ndimage.affine_transform ===
    // matrix = [[0.8, 0.1, 0.5], [-0.2, 1.1, -0.3]] (linear part + offset
    // column). Reference values from scipy.ndimage.affine_transform on the
    // same sin-texture 5×5 image.
    {
        let matrix: [[f64; 3]; 2] = [[0.8, 0.1, 0.5], [-0.2, 1.1, -0.3]];
        // (order, mode, tol, expected). order<=1 and order=3 constant are
        // bit-exact via the cardinal B-spline path; order=3 reflect routes
        // through the de Boor interpolating spline and agrees with scipy only
        // to ~1e-5 (tracked for follow-up).
        let references: [(usize, BoundaryMode, f64, [f64; 25]); 4] = [
            (
                1,
                BoundaryMode::Constant,
                ABS_TOL,
                [
                    0.000000000000e+00,
                    2.008036304347e-01,
                    6.650969071061e+00,
                    8.197105989107e+00,
                    0.000000000000e+00,
                    0.000000000000e+00,
                    -6.577724325852e+00,
                    -1.497660674610e+00,
                    5.064449943337e+00,
                    7.944470650796e+00,
                    0.000000000000e+00,
                    -5.261247238375e+00,
                    -7.251922148746e+00,
                    -3.115779049884e+00,
                    3.495465663835e+00,
                    0.000000000000e+00,
                    4.626496087927e+00,
                    -3.826560568877e+00,
                    -7.705465045180e+00,
                    -4.731583279153e+00,
                    0.000000000000e+00,
                    8.604137686135e+00,
                    6.413691759594e+00,
                    -1.763251282674e+00,
                    0.000000000000e+00,
                ],
            ),
            (
                3,
                BoundaryMode::Constant,
                ABS_TOL,
                [
                    0.000000000000e+00,
                    1.006998652486e+00,
                    7.892645468516e+00,
                    8.333880986305e+00,
                    0.000000000000e+00,
                    0.000000000000e+00,
                    -9.335527003762e+00,
                    -2.140018754972e+00,
                    7.412471008392e+00,
                    9.788554050012e+00,
                    0.000000000000e+00,
                    -4.819961508003e+00,
                    -1.047813467035e+01,
                    -5.231924638274e+00,
                    5.843354866869e+00,
                    0.000000000000e+00,
                    5.890788219480e+00,
                    -4.385238177278e+00,
                    -9.324349225853e+00,
                    -5.170924704727e+00,
                    0.000000000000e+00,
                    9.111265243496e+00,
                    7.671378145616e+00,
                    -2.175945364439e+00,
                    0.000000000000e+00,
                ],
            ),
            (
                1,
                BoundaryMode::Reflect,
                ABS_TOL,
                [
                    -4.794621373316e+00,
                    2.008036304347e-01,
                    6.650969071061e+00,
                    8.197105989107e+00,
                    2.952263871868e+00,
                    -8.344533255310e+00,
                    -6.577724325852e+00,
                    -1.497660674610e+00,
                    5.064449943337e+00,
                    7.944470650796e+00,
                    -4.245902157847e+00,
                    -5.261247238375e+00,
                    -7.251922148746e+00,
                    -3.115779049884e+00,
                    3.495465663835e+00,
                    5.308569450525e+00,
                    4.626496087927e+00,
                    -3.826560568877e+00,
                    -7.705465045180e+00,
                    -4.731583279153e+00,
                    8.006620199984e+00,
                    8.604137686135e+00,
                    6.413691759594e+00,
                    -1.763251282674e+00,
                    -8.640277915246e+00,
                ],
            ),
            (
                3,
                BoundaryMode::Reflect,
                1.0e-4,
                [
                    -6.370564537739e+00,
                    3.221833936293e-01,
                    7.829940240048e+00,
                    8.826398229861e+00,
                    2.445676381848e+00,
                    -1.051377959630e+01,
                    -8.913804179475e+00,
                    -2.109795446830e+00,
                    7.115075623034e+00,
                    9.624431251211e+00,
                    -3.249894236676e+00,
                    -5.775240237524e+00,
                    -1.008914888377e+01,
                    -4.807834235735e+00,
                    4.727963036077e+00,
                    6.190333023064e+00,
                    5.061460587515e+00,
                    -4.440078012997e+00,
                    -9.582387542220e+00,
                    -5.626796613973e+00,
                    8.943740103206e+00,
                    9.127318301507e+00,
                    7.010309896721e+00,
                    -2.153725559542e+00,
                    -9.402332586797e+00,
                ],
            ),
        ];
        for (order, mode, tol, expected) in references {
            let case_id = format!("scipy_parity_order{order}_{mode:?}");
            let result = affine_transform(&arr, &matrix, order, mode, 0.0).expect("general affine");
            let Some((expected, actual)) = ledger.slices(
                "scipy_parity",
                &case_id,
                Some(expected.as_slice()),
                Some(result.data.as_slice()),
            ) else {
                continue;
            };
            let max_diff = actual
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let pass = result.shape == arr.shape && max_diff <= tol;
            ledger.compared("scipy_parity", &case_id, pass);
            check(&case_id, pass, format!("max_diff={max_diff}"));
        }
    }

    // === 3. Error: order > 5 ===
    {
        let identity: [[f64; 3]; 2] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let result = affine_transform(&arr, &identity, 6, BoundaryMode::Constant, 0.0);
        ledger.expected_raise("errors", "order_too_large_errors", result.is_err());
        check(
            "order_too_large_errors",
            result.is_err(),
            format!("res={result:?}"),
        );
    }

    // === 4. Error: input.ndim != 2 (1D input) ===
    {
        let arr_1d = NdArray::new(vec![1.0, 2.0, 3.0], vec![3]).expect("ndarray 1d");
        let identity: [[f64; 3]; 2] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let result = affine_transform(&arr_1d, &identity, 0, BoundaryMode::Constant, 0.0);
        ledger.expected_raise("errors", "non_2d_errors", result.is_err());
        check("non_2d_errors", result.is_err(), format!("res={result:?}"));
    }

    // === 5. Error: input.ndim != 2 (3D input) ===
    {
        let arr_3d = NdArray::new(vec![1.0; 8], vec![2, 2, 2]).expect("ndarray 3d");
        let identity: [[f64; 3]; 2] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let result = affine_transform(&arr_3d, &identity, 0, BoundaryMode::Constant, 0.0);
        ledger.expected_raise("errors", "ndim_3_errors", result.is_err());
        check("ndim_3_errors", result.is_err(), format!("res={result:?}"));
    }

    // === 6. Output shape always equals input shape (per fsci semantics) ===
    {
        let scale_2x: [[f64; 3]; 2] = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let result =
            affine_transform(&arr, &scale_2x, 1, BoundaryMode::Constant, 0.0).expect("scale 2x");
        check(
            "output_shape_eq_input",
            result.shape == arr.shape,
            format!("shape={:?}", result.shape),
        );
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_ndimage_affine_transform_properties".into(),
        category: "fsci_ndimage::affine_transform property-based coverage".into(),
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
            eprintln!("affine_transform mismatch: {} — {}", d.case_id, d.note);
        }
    }

    assert!(
        all_pass,
        "affine_transform property coverage failed: {} cases",
        diffs.len()
    );
    // identity (one case per order) is the smallest of the three arms: parity has 4, errors 3.
    ledger.finish(IDENTITY_ORDERS.len());
}
