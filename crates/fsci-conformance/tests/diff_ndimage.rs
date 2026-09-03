#![forbid(unsafe_code)]
//! Live SciPy differential coverage for FSCI-P2C-015 ndimage filters.
//!
//! The existing ndimage E2E tests exercise FrankenSciPy outputs against
//! invariant checks. This harness adds a process-based SciPy oracle for concrete
//! filter semantics across deterministic input families.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{
    BoundaryMode, NdArray, binary_dilation, binary_dilation_axes, binary_erosion,
    binary_erosion_axes, gaussian_filter, laplace, maximum_filter, median_filter, minimum_filter,
    prewitt, sobel, uniform_filter,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-015";
const TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct NdimageCase {
    case_id: String,
    op: String,
    shape: [usize; 2],
    data: Vec<f64>,
    size: Option<usize>,
    sigma: Option<f64>,
    mode: String,
    cval: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleCase {
    case_id: String,
    values: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
    shape: [usize; 2],
    mode: String,
    parameter: String,
    max_abs_diff: f64,
    tolerance: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_abs_diff: f64,
    tolerance: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create ndimage diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize ndimage diff log");
    fs::write(path, json).expect("write ndimage diff log");
}

fn deterministic_data(shape: [usize; 2], seed: usize) -> Vec<f64> {
    let [rows, cols] = shape;
    (0..rows * cols)
        .map(|idx| {
            let row = idx / cols;
            let col = idx % cols;
            let plane = (row as f64 + 1.0) * 0.5 - (col as f64) * 0.25;
            let seed_bias = (seed % 7) as f64 * 0.125 - 0.375;
            let ripple = ((idx + seed * 3) % 5) as f64 * 0.2 - 0.4;
            plane + seed_bias + ripple
        })
        .collect()
}

fn ndimage_cases() -> Vec<NdimageCase> {
    let shapes = [[1, 1], [1, 7], [2, 5], [4, 4], [5, 3]];
    let modes = ["constant", "nearest"];
    let mut cases = Vec::new();

    for (shape_idx, shape) in shapes.iter().copied().enumerate() {
        for mode in modes {
            for size in [1, 3, 5] {
                let cval = if mode == "constant" { -1.5 } else { 0.0 };
                let seed = cases.len() + shape_idx;
                cases.push(NdimageCase {
                    case_id: format!(
                        "uniform_shape{}x{}_mode_{mode}_size_{size}",
                        shape[0], shape[1]
                    ),
                    op: String::from("uniform_filter"),
                    shape,
                    data: deterministic_data(shape, seed),
                    size: Some(size),
                    sigma: None,
                    mode: mode.to_string(),
                    cval,
                });
            }
        }
    }

    for (shape_idx, shape) in shapes.iter().copied().enumerate() {
        for mode in modes {
            for sigma in [0.5, 1.0, 1.25] {
                let cval = if mode == "constant" { 2.25 } else { 0.0 };
                let seed = 100 + cases.len() + shape_idx;
                cases.push(NdimageCase {
                    case_id: format!(
                        "gaussian_shape{}x{}_mode_{mode}_sigma_{sigma:.2}",
                        shape[0], shape[1]
                    ),
                    op: String::from("gaussian_filter"),
                    shape,
                    data: deterministic_data(shape, seed),
                    size: None,
                    sigma: Some(sigma),
                    mode: mode.to_string(),
                    cval,
                });
            }
        }
    }

    cases
}

fn ndimage_rank_cases() -> Vec<NdimageCase> {
    let shapes = [[1, 1], [1, 8], [3, 4], [4, 4], [6, 2]];
    let modes = ["constant", "nearest"];
    let ops = ["median_filter", "minimum_filter", "maximum_filter"];
    let mut cases = Vec::new();

    for (shape_idx, shape) in shapes.iter().copied().enumerate() {
        for mode in modes {
            for op in ops {
                for size in [1, 3, 5] {
                    let cval = if mode == "constant" { -2.75 } else { 0.0 };
                    let seed = 500 + cases.len() + shape_idx;
                    cases.push(NdimageCase {
                        case_id: format!(
                            "{op}_shape{}x{}_mode_{mode}_size_{size}",
                            shape[0], shape[1]
                        ),
                        op: op.to_string(),
                        shape,
                        data: deterministic_data(shape, seed),
                        size: Some(size),
                        sigma: None,
                        mode: mode.to_string(),
                        cval,
                    });
                }
            }
        }
    }

    cases
}

fn boundary_mode(mode: &str) -> Option<BoundaryMode> {
    match mode {
        "constant" => Some(BoundaryMode::Constant),
        "nearest" => Some(BoundaryMode::Nearest),
        _ => None,
    }
}

fn rust_output(case: &NdimageCase) -> Option<Vec<f64>> {
    let array = NdArray::new(case.data.clone(), case.shape.to_vec()).ok()?;
    let mode = boundary_mode(&case.mode)?;
    let output = match case.op.as_str() {
        "uniform_filter" => uniform_filter(&array, case.size?, mode, case.cval),
        "gaussian_filter" => gaussian_filter(&array, case.sigma?, mode, case.cval),
        "median_filter" => median_filter(&array, case.size?, mode, case.cval),
        "minimum_filter" => minimum_filter(&array, case.size?, mode, case.cval),
        "maximum_filter" => maximum_filter(&array, case.size?, mode, case.cval),
        _ => return None,
    }
    .ok()?;
    Some(output.data)
}

fn max_abs_diff(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len(), "oracle output length mismatch");
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| {
            if lhs == rhs || (lhs.is_nan() && rhs.is_nan()) {
                0.0
            } else if !lhs.is_finite() || !rhs.is_finite() {
                f64::INFINITY
            } else {
                (lhs - rhs).abs()
            }
        })
        .fold(0.0_f64, f64::max)
}

#[test]
fn max_abs_diff_flags_nonfinite_mismatches() {
    assert_eq!(max_abs_diff(&[f64::NAN], &[f64::NAN]), 0.0);
    assert_eq!(max_abs_diff(&[f64::INFINITY], &[f64::INFINITY]), 0.0);
    assert!(max_abs_diff(&[f64::NAN], &[1.0]).is_infinite());
    assert!(max_abs_diff(&[f64::INFINITY], &[1.0]).is_infinite());
}

fn run_scipy_oracle(cases: &[NdimageCase]) -> Option<Vec<OracleCase>> {
    let script = r#"
import json
import sys

import numpy as np
from scipy import ndimage

cases = json.load(sys.stdin)
output = []
for case in cases:
    arr = np.array(case["data"], dtype=np.float64).reshape(case["shape"])
    if case["op"] == "uniform_filter":
        values = ndimage.uniform_filter(
            arr,
            size=int(case["size"]),
            mode=case["mode"],
            cval=float(case["cval"]),
        )
    elif case["op"] == "median_filter":
        values = ndimage.median_filter(
            arr,
            size=int(case["size"]),
            mode=case["mode"],
            cval=float(case["cval"]),
        )
    elif case["op"] == "minimum_filter":
        values = ndimage.minimum_filter(
            arr,
            size=int(case["size"]),
            mode=case["mode"],
            cval=float(case["cval"]),
        )
    elif case["op"] == "maximum_filter":
        values = ndimage.maximum_filter(
            arr,
            size=int(case["size"]),
            mode=case["mode"],
            cval=float(case["cval"]),
        )
    elif case["op"] == "gaussian_filter":
        values = ndimage.gaussian_filter(
            arr,
            sigma=float(case["sigma"]),
            mode=case["mode"],
            cval=float(case["cval"]),
            truncate=4.0,
        )
    else:
        raise ValueError(f"unsupported op: {case['op']}")
    output.append({
        "case_id": case["case_id"],
        "values": [float(value) for value in values.ravel(order="C")],
    })
print(json.dumps(output))
"#;

    let mut child = fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;

    {
        let stdin = child.stdin.as_mut()?;
        let input = serde_json::to_vec(cases).expect("serialize ndimage oracle input");
        stdin.write_all(&input).expect("write ndimage oracle input");
    }

    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        eprintln!(
            "SciPy ndimage oracle unavailable: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }

    serde_json::from_slice(&output.stdout).ok()
}

fn scipy_oracle_or_skip(test_id: &str, cases: &[NdimageCase]) -> Option<Vec<OracleCase>> {
    let oracle = run_scipy_oracle(cases);
    if oracle.is_none() {
        assert!(
            std::env::var_os(REQUIRE_SCIPY_ENV).is_none(),
            "{REQUIRE_SCIPY_ENV}=1 but SciPy ndimage is unavailable for {test_id}"
        );
        eprintln!(
            "SciPy ndimage not available; skipping {test_id}. Set {REQUIRE_SCIPY_ENV}=1 to fail closed when the oracle is missing"
        );
    }
    oracle
}

#[test]
fn diff_001_ndimage_filters_live_scipy() {
    let cases = ndimage_cases();
    assert_eq!(cases.len(), 60, "ndimage diff case inventory changed");

    let start = Instant::now();
    let Some(oracle_cases) = scipy_oracle_or_skip("diff_001_ndimage_filters_live_scipy", &cases)
    else {
        return;
    };
    assert_eq!(
        oracle_cases.len(),
        cases.len(),
        "SciPy ndimage oracle case count mismatch"
    );

    let mut case_diffs = Vec::with_capacity(cases.len());
    for (case, oracle) in cases.iter().zip(oracle_cases.iter()) {
        assert_eq!(
            case.case_id, oracle.case_id,
            "ndimage oracle case id mismatch"
        );
        let actual = rust_output(case).unwrap_or_default();
        let diff = max_abs_diff(&actual, &oracle.values);
        let parameter = match case.op.as_str() {
            "uniform_filter" => format!("size={}", case.size.expect("uniform size")),
            "gaussian_filter" => format!("sigma={:.2}", case.sigma.expect("gaussian sigma")),
            "median_filter" | "minimum_filter" | "maximum_filter" => {
                format!("size={}", case.size.expect("rank filter size"))
            }
            _ => String::from("unknown"),
        };
        case_diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            shape: case.shape,
            mode: case.mode.clone(),
            parameter,
            max_abs_diff: diff,
            tolerance: TOL,
            pass: diff <= TOL,
        });
    }

    let max_diff = case_diffs
        .iter()
        .map(|case| case.max_abs_diff)
        .fold(0.0_f64, f64::max);
    let pass = case_diffs.iter().all(|case| case.pass);
    let log = DiffLog {
        test_id: String::from("diff_001_ndimage_filters_live_scipy"),
        category: String::from("live_scipy_differential"),
        case_count: cases.len(),
        max_abs_diff: max_diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: case_diffs,
    };
    emit_log(&log);

    assert!(
        pass,
        "ndimage live SciPy diff max_abs_diff={max_diff:.3e} exceeds tolerance {TOL:.3e}"
    );
}

#[test]
fn diff_002_ndimage_rank_filters_live_scipy() {
    let cases = ndimage_rank_cases();
    assert_eq!(
        cases.len(),
        90,
        "ndimage rank-filter diff case inventory changed"
    );

    let start = Instant::now();
    let Some(oracle_cases) =
        scipy_oracle_or_skip("diff_002_ndimage_rank_filters_live_scipy", &cases)
    else {
        return;
    };
    assert_eq!(
        oracle_cases.len(),
        cases.len(),
        "SciPy ndimage rank-filter oracle case count mismatch"
    );

    let mut case_diffs = Vec::with_capacity(cases.len());
    for (case, oracle) in cases.iter().zip(oracle_cases.iter()) {
        assert_eq!(
            case.case_id, oracle.case_id,
            "ndimage rank-filter oracle case id mismatch"
        );
        let actual = rust_output(case).unwrap_or_default();
        let diff = max_abs_diff(&actual, &oracle.values);
        case_diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            shape: case.shape,
            mode: case.mode.clone(),
            parameter: format!("size={}", case.size.expect("rank filter size")),
            max_abs_diff: diff,
            tolerance: TOL,
            pass: diff <= TOL,
        });
    }

    let max_diff = case_diffs
        .iter()
        .map(|case| case.max_abs_diff)
        .fold(0.0_f64, f64::max);
    let pass = case_diffs.iter().all(|case| case.pass);
    let log = DiffLog {
        test_id: String::from("diff_002_ndimage_rank_filters_live_scipy"),
        category: String::from("live_scipy_differential"),
        case_count: cases.len(),
        max_abs_diff: max_diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: case_diffs,
    };
    emit_log(&log);

    assert!(
        pass,
        "ndimage live SciPy rank-filter diff max_abs_diff={max_diff:.3e} exceeds tolerance {TOL:.3e}"
    );
}

#[derive(Debug, Clone, Serialize)]
struct EdgeCase {
    case_id: String,
    op: String,
    shape: [usize; 2],
    data: Vec<f64>,
    axis: Option<usize>,
    mode: String,
    cval: f64,
}

fn edge_detection_cases() -> Vec<EdgeCase> {
    let shapes = [[3, 3], [4, 5], [5, 4], [6, 6], [2, 8]];
    let modes = ["constant", "nearest"];
    let mut cases = Vec::new();

    for (shape_idx, shape) in shapes.iter().copied().enumerate() {
        for mode in modes {
            let cval = 0.0;
            for axis in 0..2 {
                let seed = 1000 + cases.len() + shape_idx;
                cases.push(EdgeCase {
                    case_id: format!(
                        "sobel_shape{}x{}_mode_{mode}_axis_{axis}",
                        shape[0], shape[1]
                    ),
                    op: String::from("sobel"),
                    shape,
                    data: deterministic_data(shape, seed),
                    axis: Some(axis),
                    mode: mode.to_string(),
                    cval,
                });
                cases.push(EdgeCase {
                    case_id: format!(
                        "prewitt_shape{}x{}_mode_{mode}_axis_{axis}",
                        shape[0], shape[1]
                    ),
                    op: String::from("prewitt"),
                    shape,
                    data: deterministic_data(shape, seed + 50),
                    axis: Some(axis),
                    mode: mode.to_string(),
                    cval,
                });
            }
            let seed = 2000 + cases.len() + shape_idx;
            cases.push(EdgeCase {
                case_id: format!("laplace_shape{}x{}_mode_{mode}", shape[0], shape[1]),
                op: String::from("laplace"),
                shape,
                data: deterministic_data(shape, seed),
                axis: None,
                mode: mode.to_string(),
                cval,
            });
        }
    }

    cases
}

fn rust_edge_output(case: &EdgeCase) -> Option<Vec<f64>> {
    let array = NdArray::new(case.data.clone(), case.shape.to_vec()).ok()?;
    let mode = boundary_mode(&case.mode)?;
    let output = match case.op.as_str() {
        "sobel" => sobel(&array, case.axis?, mode, case.cval),
        "prewitt" => prewitt(&array, case.axis?, mode, case.cval),
        "laplace" => laplace(&array, mode, case.cval),
        _ => return None,
    }
    .ok()?;
    Some(output.data)
}

fn run_scipy_edge_oracle(cases: &[EdgeCase]) -> Option<Vec<OracleCase>> {
    let script = r#"
import json
import sys

import numpy as np
from scipy import ndimage

cases = json.load(sys.stdin)
output = []
for case in cases:
    arr = np.array(case["data"], dtype=np.float64).reshape(case["shape"])
    if case["op"] == "sobel":
        values = ndimage.sobel(
            arr,
            axis=int(case["axis"]),
            mode=case["mode"],
            cval=float(case["cval"]),
        )
    elif case["op"] == "prewitt":
        values = ndimage.prewitt(
            arr,
            axis=int(case["axis"]),
            mode=case["mode"],
            cval=float(case["cval"]),
        )
    elif case["op"] == "laplace":
        values = ndimage.laplace(
            arr,
            mode=case["mode"],
            cval=float(case["cval"]),
        )
    else:
        raise ValueError(f"unsupported op: {case['op']}")
    output.append({
        "case_id": case["case_id"],
        "values": [float(value) for value in values.ravel(order="C")],
    })
print(json.dumps(output))
"#;

    let mut child = fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;

    {
        let stdin = child.stdin.as_mut()?;
        let input = serde_json::to_vec(cases).expect("serialize edge oracle input");
        stdin.write_all(&input).expect("write edge oracle input");
    }

    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        eprintln!(
            "SciPy edge oracle unavailable: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }

    serde_json::from_slice(&output.stdout).ok()
}

fn scipy_edge_oracle_or_skip(test_id: &str, cases: &[EdgeCase]) -> Option<Vec<OracleCase>> {
    let oracle = run_scipy_edge_oracle(cases);
    if oracle.is_none() {
        assert!(
            std::env::var_os(REQUIRE_SCIPY_ENV).is_none(),
            "{REQUIRE_SCIPY_ENV}=1 but SciPy ndimage edge detection is unavailable for {test_id}"
        );
        eprintln!(
            "SciPy ndimage not available; skipping {test_id}. Set {REQUIRE_SCIPY_ENV}=1 to fail closed"
        );
    }
    oracle
}

#[test]
fn diff_003_ndimage_edge_detection_live_scipy() {
    let cases = edge_detection_cases();
    assert_eq!(
        cases.len(),
        50,
        "edge detection diff case inventory changed"
    );

    let start = Instant::now();
    let Some(oracle_cases) =
        scipy_edge_oracle_or_skip("diff_003_ndimage_edge_detection_live_scipy", &cases)
    else {
        return;
    };
    assert_eq!(
        oracle_cases.len(),
        cases.len(),
        "SciPy edge detection oracle case count mismatch"
    );

    let mut case_diffs = Vec::with_capacity(cases.len());
    for (case, oracle) in cases.iter().zip(oracle_cases.iter()) {
        assert_eq!(case.case_id, oracle.case_id, "edge oracle case id mismatch");
        let actual = rust_edge_output(case).unwrap_or_default();
        let diff = max_abs_diff(&actual, &oracle.values);
        let parameter = match case.op.as_str() {
            "sobel" | "prewitt" => format!("axis={}", case.axis.unwrap_or(0)),
            "laplace" => String::from("n/a"),
            _ => String::from("unknown"),
        };
        case_diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            shape: case.shape,
            mode: case.mode.clone(),
            parameter,
            max_abs_diff: diff,
            tolerance: TOL,
            pass: diff <= TOL,
        });
    }

    let max_diff = case_diffs
        .iter()
        .map(|case| case.max_abs_diff)
        .fold(0.0_f64, f64::max);
    let pass = case_diffs.iter().all(|case| case.pass);
    let log = DiffLog {
        test_id: String::from("diff_003_ndimage_edge_detection_live_scipy"),
        category: String::from("live_scipy_differential"),
        case_count: cases.len(),
        max_abs_diff: max_diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: case_diffs,
    };
    emit_log(&log);

    assert!(
        pass,
        "ndimage edge detection diff max_abs_diff={max_diff:.3e} exceeds tolerance {TOL:.3e}"
    );
}

#[derive(Debug, Clone, Serialize)]
struct BinaryMorphCase {
    case_id: String,
    op: String,
    shape: [usize; 2],
    data: Vec<f64>,
    structure_size: usize,
    iterations: usize,
    axes: Option<Vec<isize>>,
}

fn binary_morph_cases() -> Vec<BinaryMorphCase> {
    let shapes = [[3, 3], [4, 5], [5, 5], [6, 4], [8, 8]];
    let mut cases = Vec::new();

    for (shape_idx, shape) in shapes.iter().copied().enumerate() {
        for structure_size in [3, 5] {
            for iterations in [1, 2] {
                let seed = 3000 + cases.len() + shape_idx;
                let binary_data: Vec<f64> = deterministic_data(shape, seed)
                    .iter()
                    .map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
                    .collect();

                cases.push(BinaryMorphCase {
                    case_id: format!(
                        "binary_erosion_shape{}x{}_size_{structure_size}_iter_{iterations}",
                        shape[0], shape[1]
                    ),
                    op: String::from("binary_erosion"),
                    shape,
                    data: binary_data.clone(),
                    structure_size,
                    iterations,
                    axes: None,
                });
                cases.push(BinaryMorphCase {
                    case_id: format!(
                        "binary_dilation_shape{}x{}_size_{structure_size}_iter_{iterations}",
                        shape[0], shape[1]
                    ),
                    op: String::from("binary_dilation"),
                    shape,
                    data: binary_data,
                    structure_size,
                    iterations,
                    axes: None,
                });
            }
        }
    }

    #[rustfmt::skip]
    let axes_data = vec![
        2.0, 0.0, -1.0,
        0.0, 3.0, 4.0,
    ];
    for axes in [vec![-1], vec![-2], vec![-2, -1], vec![]] {
        let axes_label = if axes.is_empty() {
            String::from("empty")
        } else {
            axes.iter()
                .map(isize::to_string)
                .collect::<Vec<_>>()
                .join("_")
        };
        cases.push(BinaryMorphCase {
            case_id: format!("binary_erosion_axes_{axes_label}_size_2_iter_1"),
            op: String::from("binary_erosion"),
            shape: [2, 3],
            data: axes_data.clone(),
            structure_size: 2,
            iterations: 1,
            axes: Some(axes.clone()),
        });
        cases.push(BinaryMorphCase {
            case_id: format!("binary_dilation_axes_{axes_label}_size_2_iter_1"),
            op: String::from("binary_dilation"),
            shape: [2, 3],
            data: axes_data.clone(),
            structure_size: 2,
            iterations: 1,
            axes: Some(axes),
        });
    }

    cases
}

fn rust_binary_morph_output(case: &BinaryMorphCase) -> Option<Vec<f64>> {
    let array = NdArray::new(case.data.clone(), case.shape.to_vec()).ok()?;
    let output = match case.op.as_str() {
        "binary_erosion" => match case.axes.as_deref() {
            Some(axes) => binary_erosion_axes(&array, case.structure_size, axes, case.iterations),
            None => binary_erosion(&array, case.structure_size, case.iterations),
        },
        "binary_dilation" => match case.axes.as_deref() {
            Some(axes) => binary_dilation_axes(&array, case.structure_size, axes, case.iterations),
            None => binary_dilation(&array, case.structure_size, case.iterations),
        },
        _ => return None,
    }
    .ok()?;
    Some(output.data)
}

fn run_scipy_binary_morph_oracle(cases: &[BinaryMorphCase]) -> Option<Vec<OracleCase>> {
    let script = r#"
import json
import sys

import numpy as np
from scipy import ndimage

cases = json.load(sys.stdin)
output = []
for case in cases:
    arr = np.array(case["data"], dtype=np.float64).reshape(case["shape"])
    arr_bool = arr.astype(bool)
    size = int(case["structure_size"])
    iterations = int(case["iterations"])
    axes = case.get("axes")
    if axes is None:
        structure = np.ones((size, size), dtype=bool)
        kwargs = {}
    else:
        axes = tuple(int(axis) for axis in axes)
        structure = np.ones((size,) * len(axes), dtype=bool)
        kwargs = {"axes": axes}

    if case["op"] == "binary_erosion":
        values = ndimage.binary_erosion(
            arr_bool, structure=structure, iterations=iterations, **kwargs
        )
    elif case["op"] == "binary_dilation":
        values = ndimage.binary_dilation(
            arr_bool, structure=structure, iterations=iterations, **kwargs
        )
    else:
        raise ValueError(f"unsupported op: {case['op']}")
    output.append({
        "case_id": case["case_id"],
        "values": [float(value) for value in values.ravel(order="C")],
    })
print(json.dumps(output))
"#;

    let mut child = fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;

    {
        let stdin = child.stdin.as_mut()?;
        let input = serde_json::to_vec(cases).expect("serialize binary morph oracle input");
        stdin
            .write_all(&input)
            .expect("write binary morph oracle input");
    }

    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        eprintln!(
            "SciPy binary morph oracle unavailable: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }

    serde_json::from_slice(&output.stdout).ok()
}

fn scipy_binary_morph_oracle_or_skip(
    test_id: &str,
    cases: &[BinaryMorphCase],
) -> Option<Vec<OracleCase>> {
    let oracle = run_scipy_binary_morph_oracle(cases);
    if oracle.is_none() {
        assert!(
            std::env::var_os(REQUIRE_SCIPY_ENV).is_none(),
            "{REQUIRE_SCIPY_ENV}=1 but SciPy binary morph is unavailable for {test_id}"
        );
        eprintln!(
            "SciPy ndimage not available; skipping {test_id}. Set {REQUIRE_SCIPY_ENV}=1 to fail closed"
        );
    }
    oracle
}

#[test]
fn diff_004_ndimage_binary_morphology_live_scipy() {
    let cases = binary_morph_cases();
    assert_eq!(cases.len(), 48, "binary morph diff case inventory changed");

    let start = Instant::now();
    let Some(oracle_cases) =
        scipy_binary_morph_oracle_or_skip("diff_004_ndimage_binary_morphology_live_scipy", &cases)
    else {
        return;
    };
    assert_eq!(
        oracle_cases.len(),
        cases.len(),
        "SciPy binary morph oracle case count mismatch"
    );

    let mut case_diffs = Vec::with_capacity(cases.len());
    for (case, oracle) in cases.iter().zip(oracle_cases.iter()) {
        assert_eq!(
            case.case_id, oracle.case_id,
            "binary morph oracle case id mismatch"
        );
        let actual = rust_binary_morph_output(case).unwrap_or_default();
        let diff = max_abs_diff(&actual, &oracle.values);
        case_diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            shape: case.shape,
            mode: String::from("n/a"),
            parameter: format!(
                "size={},iter={},axes={:?}",
                case.structure_size, case.iterations, case.axes
            ),
            max_abs_diff: diff,
            tolerance: TOL,
            pass: diff <= TOL,
        });
    }

    let max_diff = case_diffs
        .iter()
        .map(|case| case.max_abs_diff)
        .fold(0.0_f64, f64::max);
    let pass = case_diffs.iter().all(|case| case.pass);
    let log = DiffLog {
        test_id: String::from("diff_004_ndimage_binary_morphology_live_scipy"),
        category: String::from("live_scipy_differential"),
        case_count: cases.len(),
        max_abs_diff: max_diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: case_diffs,
    };
    emit_log(&log);

    assert!(
        pass,
        "ndimage binary morph diff max_abs_diff={max_diff:.3e} exceeds tolerance {TOL:.3e}"
    );
}
