#![forbid(unsafe_code)]
//! Live SciPy parity for `scipy.fft.fftshift` and `ifftshift` axis handling.
//!
//! Covers duplicated axes and empty axes, which NumPy/SciPy intentionally
//! apply in sequence rather than deduplicating.

use std::io::Write;
use std::process::{Command, Stdio};

use fsci_fft::{fftshift, ifftshift};
use serde::{Deserialize, Serialize};

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Copy, Serialize)]
struct ShiftCase {
    case_id: &'static str,
    op: &'static str,
    shape: &'static [usize],
    axes: &'static [usize],
}

#[derive(Debug, Clone, Deserialize)]
struct ShiftArm {
    case_id: String,
    values: Option<Vec<i32>>,
    error: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<ShiftArm>,
}

const SHAPE_2X3: [usize; 2] = [2, 3];
const AXES_EMPTY: [usize; 0] = [];
const AXES_DUPLICATE_LAST: [usize; 2] = [1, 1];
const AXES_DUPLICATE_FIRST: [usize; 2] = [0, 0];
const AXES_MIXED_DUPLICATE: [usize; 3] = [0, 1, 1];

const CASES: [ShiftCase; 8] = [
    ShiftCase {
        case_id: "fftshift_empty_axes",
        op: "fftshift",
        shape: &SHAPE_2X3,
        axes: &AXES_EMPTY,
    },
    ShiftCase {
        case_id: "ifftshift_empty_axes",
        op: "ifftshift",
        shape: &SHAPE_2X3,
        axes: &AXES_EMPTY,
    },
    ShiftCase {
        case_id: "fftshift_duplicate_last_axis",
        op: "fftshift",
        shape: &SHAPE_2X3,
        axes: &AXES_DUPLICATE_LAST,
    },
    ShiftCase {
        case_id: "ifftshift_duplicate_last_axis",
        op: "ifftshift",
        shape: &SHAPE_2X3,
        axes: &AXES_DUPLICATE_LAST,
    },
    ShiftCase {
        case_id: "fftshift_duplicate_first_axis",
        op: "fftshift",
        shape: &SHAPE_2X3,
        axes: &AXES_DUPLICATE_FIRST,
    },
    ShiftCase {
        case_id: "ifftshift_duplicate_first_axis",
        op: "ifftshift",
        shape: &SHAPE_2X3,
        axes: &AXES_DUPLICATE_FIRST,
    },
    ShiftCase {
        case_id: "fftshift_mixed_duplicate_axes",
        op: "fftshift",
        shape: &SHAPE_2X3,
        axes: &AXES_MIXED_DUPLICATE,
    },
    ShiftCase {
        case_id: "ifftshift_mixed_duplicate_axes",
        op: "ifftshift",
        shape: &SHAPE_2X3,
        axes: &AXES_MIXED_DUPLICATE,
    },
];

fn scipy_oracle_or_skip(query: &[ShiftCase]) -> Result<Option<OracleResult>, String> {
    let script = r#"
import json
import sys
import numpy as np
from scipy import fft

q = json.loads(sys.argv[1])
points = []
for case in q:
    cid = case["case_id"]
    shape = tuple(int(v) for v in case["shape"])
    axes = tuple(int(v) for v in case["axes"])
    arr = np.arange(int(np.prod(shape)), dtype=np.int32).reshape(shape)
    try:
        if case["op"] == "fftshift":
            out = fft.fftshift(arr, axes=axes)
        elif case["op"] == "ifftshift":
            out = fft.ifftshift(arr, axes=axes)
        else:
            raise ValueError(f"unknown op {case['op']}")
        points.append({"case_id": cid, "values": out.ravel().astype(int).tolist(), "error": None})
    except Exception as exc:
        points.append({"case_id": cid, "values": None, "error": type(exc).__name__ + ": " + str(exc)})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query)
        .map_err(|err| format!("failed to serialize fft shift query: {err}"))?;
    let mut child = match Command::new("python3")
        .arg("-")
        .arg(query_json)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            if std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
                return Err(format!(
                    "failed to spawn python3 for fft shift oracle: {err}"
                ));
            }
            return Ok(None);
        }
    };
    {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| String::from("failed to open fft shift oracle stdin"))?;
        if let Err(err) = stdin.write_all(script.as_bytes()) {
            drop(stdin);
            let output = child
                .wait_with_output()
                .map_err(|wait_err| format!("fft shift oracle wait failed: {wait_err}"))?;
            let stderr = String::from_utf8_lossy(&output.stderr);
            if std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
                return Err(format!(
                    "fft shift oracle stdin write failed: {err}; stderr: {stderr}"
                ));
            }
            return Ok(None);
        }
    }
    let output = child
        .wait_with_output()
        .map_err(|err| format!("fft shift oracle wait failed: {err}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
            return Err(format!("fft shift oracle failed: {stderr}"));
        }
        return Ok(None);
    }
    serde_json::from_slice(&output.stdout)
        .map(Some)
        .map_err(|err| format!("failed to parse fft shift oracle JSON: {err}"))
}

fn fsci_shift(case: &ShiftCase) -> Result<Vec<i32>, String> {
    let len = case.shape.iter().product();
    let input: Vec<i32> = (0..len)
        .map(i32::try_from)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("input value does not fit i32: {err}"))?;
    match case.op {
        "fftshift" => {
            fftshift(&input, case.shape, Some(case.axes)).map_err(|err| format!("{err:?}"))
        }
        "ifftshift" => {
            ifftshift(&input, case.shape, Some(case.axes)).map_err(|err| format!("{err:?}"))
        }
        other => Err(format!("unknown shift op {other}")),
    }
}

#[test]
fn fftshift_duplicate_and_empty_axes_match_scipy() -> Result<(), String> {
    let Some(oracle) = scipy_oracle_or_skip(&CASES)? else {
        return Ok(());
    };

    for case in &CASES {
        let arm = oracle
            .points
            .iter()
            .find(|point| point.case_id == case.case_id)
            .ok_or_else(|| missing_oracle_case(case.case_id))?;
        let expected = arm
            .values
            .as_ref()
            .ok_or_else(|| oracle_error(case.case_id, arm.error.as_deref()))?;
        let actual = fsci_shift(case)?;
        if actual != *expected {
            return Err(mismatch_error(case.case_id, &actual, expected));
        }
    }
    Ok(())
}

fn missing_oracle_case(case_id: &str) -> String {
    format!("oracle did not return fft shift case {case_id}")
}

fn oracle_error(case_id: &str, error: Option<&str>) -> String {
    format!("oracle errored for {case_id}: {error:?}")
}

fn mismatch_error(case_id: &str, actual: &[i32], expected: &[i32]) -> String {
    format!("{case_id} mismatch: fsci={actual:?} scipy={expected:?}")
}
