#![forbid(unsafe_code)]
//! Live SciPy differential coverage for Hermitian n-D FFT entrypoints.
//!
//! Covers `scipy.fft.hfft2`, `ihfft2`, `hfftn`, and `ihfftn`.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_fft::{Complex64, FftOptions, hfft2, hfftn, ihfft2, ihfftn};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-005";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const TOL: f64 = 1e-8;
/// One ledger arm per SciPy entrypoint compared.
const ARMS: [&str; 4] = ["ihfft2", "hfft2", "ihfftn", "hfftn"];

#[derive(Debug, Clone, Serialize)]
struct CasePoint {
    case_id: &'static str,
    op: &'static str,
    shape: Vec<usize>,
    values: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    cases: Vec<CasePoint>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleArm {
    case_id: String,
    input_complex: Option<Vec<[f64; 2]>>,
    real: Option<Vec<f64>>,
    complex: Option<Vec<[f64; 2]>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    cases: Vec<OracleArm>,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct CaseDiff<'a> {
    case_id: &'a str,
    op: &'a str,
    shape: &'a [usize],
    max_abs_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog<'a> {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff<'a>>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog<'_>) -> Result<(), String> {
    fs::create_dir_all(output_dir()).map_err(|err| err.to_string())?;
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).map_err(|err| err.to_string())?;
    fs::write(path, json).map_err(|err| err.to_string())
}

fn generate_query() -> OracleQuery {
    OracleQuery {
        cases: vec![
            CasePoint {
                case_id: "ihfft2_2x3",
                op: "ihfft2",
                shape: vec![2, 3],
                values: (0..6).map(|v| v as f64).collect(),
            },
            CasePoint {
                case_id: "hfft2_3x4",
                op: "hfft2",
                shape: vec![3, 4],
                values: (0..12).map(|v| v as f64 * 0.5 - 2.0).collect(),
            },
            CasePoint {
                case_id: "ihfftn_2x3x4",
                op: "ihfftn",
                shape: vec![2, 3, 4],
                values: (0..24).map(|v| v as f64 - 7.0).collect(),
            },
            CasePoint {
                case_id: "hfftn_2x2x3",
                op: "hfftn",
                shape: vec![2, 2, 3],
                values: (0..12)
                    .map(|v| (v as f64).sin() + v as f64 * 0.25)
                    .collect(),
            },
        ],
    }
}

fn scipy_required() -> bool {
    std::env::var(REQUIRE_SCIPY_ENV).is_ok()
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Result<Option<OracleResult>, String> {
    let script = r#"
import json
import os
import sys

try:
    import numpy as np
    from scipy import fft
except Exception as exc:
    print(f"scipy import failed: {exc}", file=sys.stderr)
    sys.exit(2)

def cflat(arr):
    return [[float(z.real), float(z.imag)] for z in np.asarray(arr).reshape(-1)]

def rflat(arr):
    return [float(v) for v in np.asarray(arr).reshape(-1)]

q = json.loads(os.environ["FSCI_HFFT_ND_QUERY"])
out = []
for case in q["cases"]:
    shape = tuple(int(v) for v in case["shape"])
    x = np.array(case["values"], dtype=float).reshape(shape)
    op = case["op"]
    try:
        if op == "ihfft2":
            y = fft.ihfft2(x, s=shape)
            out.append({"case_id": case["case_id"], "input_complex": None, "real": None, "complex": cflat(y)})
        elif op == "ihfftn":
            y = fft.ihfftn(x, s=shape)
            out.append({"case_id": case["case_id"], "input_complex": None, "real": None, "complex": cflat(y)})
        elif op == "hfft2":
            spectrum = fft.ihfft2(x, s=shape)
            y = fft.hfft2(spectrum, s=shape)
            out.append({"case_id": case["case_id"], "input_complex": cflat(spectrum), "real": rflat(y), "complex": None})
        elif op == "hfftn":
            spectrum = fft.ihfftn(x, s=shape)
            y = fft.hfftn(spectrum, s=shape)
            out.append({"case_id": case["case_id"], "input_complex": cflat(spectrum), "real": rflat(y), "complex": None})
    except Exception as exc:
        print(f"case {case['case_id']} failed: {exc}", file=sys.stderr)
        sys.exit(3)

print(json.dumps({"cases": out}))
"#;
    let query_json = serde_json::to_string(query).map_err(|err| err.to_string())?;
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-")
        .env("FSCI_HFFT_ND_QUERY", query_json)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            if scipy_required() {
                return Err(format!("failed to spawn python3 for hfft n-d oracle: {e}"));
            }
            eprintln!("skipping hfft n-d oracle: python3 not available ({e})");
            return Ok(None);
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| "open hfft n-d oracle stdin".to_string())?;
        if let Err(err) = stdin.write_all(script.as_bytes()) {
            let output = child
                .wait_with_output()
                .map_err(|wait_err| wait_err.to_string())?;
            let stderr = String::from_utf8_lossy(&output.stderr);
            if scipy_required() {
                return Err(format!(
                    "hfft n-d oracle stdin write failed: {err}; stderr: {stderr}"
                ));
            }
            eprintln!("skipping hfft n-d oracle: stdin write failed ({err})\n{stderr}");
            return Ok(None);
        }
    }
    let output = child.wait_with_output().map_err(|err| err.to_string())?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if scipy_required() {
            return Err(format!("hfft n-d oracle failed: {stderr}"));
        }
        eprintln!("skipping hfft n-d oracle: scipy not available\n{stderr}");
        return Ok(None);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout)
        .map(Some)
        .map_err(|err| format!("parse hfft n-d oracle JSON: {err}; stdout: {stdout}"))
}

fn complex_from_pairs(values: &[[f64; 2]]) -> Vec<Complex64> {
    values.iter().map(|&[re, im]| (re, im)).collect()
}

fn shape2(shape: &[usize]) -> Result<(usize, usize), String> {
    let [rows, cols] = shape else {
        return Err(format!("expected 2-D shape, got {shape:?}"));
    };
    Ok((*rows, *cols))
}

fn max_abs_diff_real(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
}

/// Interleaved `[re, im, re, im, ...]`: the max of the element-wise |diff| over this is the
/// max over elements of `max(|d_re|, |d_im|)`, and `ledger.slices` sees every component.
fn interleave_complex(values: &[Complex64]) -> Vec<f64> {
    values.iter().flat_map(|&(re, im)| [re, im]).collect()
}

fn interleave_pairs(values: &[[f64; 2]]) -> Vec<f64> {
    values.iter().flatten().copied().collect()
}

#[test]
fn diff_fft_hfft_nd() -> Result<(), String> {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query)? else {
        return Ok(());
    };
    assert_eq!(oracle.cases.len(), query.cases.len());

    let opts = FftOptions::default();
    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut ledger = CompareLedger::new("diff_fft_hfft_nd", &ARMS);

    for (case, expected) in query.cases.iter().zip(oracle.cases.iter()) {
        assert_eq!(case.case_id, expected.case_id);
        // (SciPy's output, fsci's output), both flattened to f64 so the ledger checks length and
        // every component. The hfft ops take SciPy's own ihfft spectrum as their input.
        let (scipy_out, fsci_out): (Option<Vec<f64>>, Option<Vec<f64>>) = match case.op {
            "ihfft2" => {
                let shape = shape2(&case.shape)?;
                let actual = ihfft2(&case.values, shape, &opts).ok();
                (
                    expected.complex.as_deref().map(interleave_pairs),
                    actual.as_deref().map(interleave_complex),
                )
            }
            "ihfftn" => {
                let actual = ihfftn(&case.values, &case.shape, &opts).ok();
                (
                    expected.complex.as_deref().map(interleave_pairs),
                    actual.as_deref().map(interleave_complex),
                )
            }
            "hfft2" => {
                let shape = shape2(&case.shape)?;
                let actual = expected
                    .input_complex
                    .as_deref()
                    .and_then(|input| hfft2(&complex_from_pairs(input), shape, &opts).ok());
                (expected.real.clone(), actual)
            }
            "hfftn" => {
                let actual = expected
                    .input_complex
                    .as_deref()
                    .and_then(|input| hfftn(&complex_from_pairs(input), &case.shape, &opts).ok());
                (expected.real.clone(), actual)
            }
            other => return Err(format!("unknown hfft n-d op: {other}")),
        };
        let Some((scipy_v, fsci_v)) = ledger.slices(
            case.op,
            case.case_id,
            scipy_out.as_deref(),
            fsci_out.as_deref(),
        ) else {
            continue;
        };
        let max_abs_diff = max_abs_diff_real(fsci_v, scipy_v);
        ledger.compared(case.op, case.case_id, max_abs_diff <= TOL);
        diffs.push(CaseDiff {
            case_id: case.case_id,
            op: case.op,
            shape: &case.shape,
            max_abs_diff,
            pass: max_abs_diff <= TOL,
        });
    }

    let all_pass = diffs.iter().all(|diff| diff.pass);
    let log = DiffLog {
        test_id: "diff_fft_hfft_nd".into(),
        category: "scipy.fft hfft2/ihfft2/hfftn/ihfftn".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log)?;

    for diff in &diffs {
        if !diff.pass {
            eprintln!(
                "hfft n-d mismatch: {} op={} shape={:?} max_abs_diff={}",
                diff.case_id, diff.op, diff.shape, diff.max_abs_diff
            );
        }
    }

    if !all_pass {
        return Err(format!(
            "scipy.fft Hermitian n-D conformance failed: {} cases",
            diffs.len()
        ));
    }
    // Each op has its own case set; every arm must compare all of its cases.
    let min_per_arm = ARMS
        .iter()
        .map(|arm| query.cases.iter().filter(|c| c.op == *arm).count())
        .min()
        .expect("ARMS is non-empty");
    ledger.finish(min_per_arm);
    Ok(())
}
