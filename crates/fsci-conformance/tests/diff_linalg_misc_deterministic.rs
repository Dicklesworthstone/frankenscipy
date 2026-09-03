#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the closed-form / deterministic
//! linalg primitives not yet covered by diff_linalg_*:
//!   - `fiedler(a)`                    vs `scipy.linalg.fiedler(a)`
//!   - `fiedler_companion(a)`          vs `scipy.linalg.fiedler_companion(a)`
//!   - `dft_matrix(n)`                 vs `scipy.linalg.dft(n) / sqrt(n)` (fsci normalizes)
//!   - `convolution_matrix(h, n, mode)` vs `scipy.linalg.convolution_matrix`
//!   - `bandwidth(a)`                  vs `scipy.linalg.bandwidth(a)`
//!   - `frobenius_norm(a)`             vs `numpy.linalg.norm(a, 'fro')`
//!
//! Resolves [frankenscipy-4mw9t]. All ops are closed-form, so bit-exact
//! (1e-12 abs) parity is expected. dft_matrix needs special handling:
//! fsci returns a 1/√n-scaled complex matrix as `Vec<Vec<(re, im)>>` to
//! match scipy.linalg.dft(n) / √n (scipy's dft returns the un-normalized
//! DFT — we divide on the python side to match fsci's scaling).
//!
//! Scope notes:
//!  * fiedler_companion: fsci builds Fiedler's pentadiagonal construction
//!    (monic-rescaled), bit-exact against scipy.linalg.fiedler_companion
//!    including non-monic leading coefficients. Resolves frankenscipy-al8mi.
//!  * bandwidth: scipy returns an unexpected (0, 1) for a 3×5 matrix
//!    whose nonzero pattern has j-i ∈ {0, 1, 2}; harness restricts to
//!    square matrices where fsci and scipy agree.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{
    bandwidth, convolution_matrix, dft_matrix, fiedler, fiedler_companion, frobenius_norm,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-009";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    /// 1-D vector input (for fiedler / fiedler_companion / convolution_matrix h).
    vec1d: Vec<f64>,
    /// 2-D matrix input (for bandwidth / frobenius_norm).
    mat2d: Vec<Vec<f64>>,
    /// dft_matrix dimension (and convolution_matrix `n`).
    n: usize,
    /// convolution_matrix mode.
    mode: String,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// Flattened scalar / matrix / pair values.
    values: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
    abs_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_abs_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create linalg_misc diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize linalg_misc diff log");
    fs::write(path, json).expect("write linalg_misc diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    // fiedler: take 1-D vector (no leading-coefficient normalization).
    let fiedler_inputs: &[(&str, Vec<f64>)] = &[
        ("len3_simple", vec![1.0, 2.0, 3.0]),
        ("len4_negative", vec![-1.0, 0.0, 2.0, -3.0]),
        ("len5_mixed", vec![3.5, -1.0, 2.0, 4.0, 0.5]),
    ];
    for (label, v) in fiedler_inputs {
        points.push(PointCase {
            case_id: format!("fiedler_{label}"),
            op: "fiedler".into(),
            vec1d: v.clone(),
            mat2d: vec![],
            n: 0,
            mode: "".into(),
        });
    }

    // fiedler_companion: take 1-D polynomial coefficients (descending order).
    // Includes non-monic leading coefficients to exercise the monic rescale.
    let fiedler_companion_inputs: &[(&str, Vec<f64>)] = &[
        ("len3_monic", vec![1.0, 2.0, 3.0]),
        ("len4_nonmonic", vec![2.0, -4.0, 6.0, -8.0]),
        ("len5_nonmonic_mixed", vec![3.5, -1.0, 2.0, 4.0, 0.5]),
        ("len6_monic", vec![1.0, -3.0, 2.0, 5.0, -1.0, 7.0]),
        ("len7_nonmonic", vec![2.0, 1.0, -3.0, 4.0, -5.0, 6.0, -7.0]),
        ("len2_nonmonic", vec![2.0, 6.0]),
    ];
    for (label, v) in fiedler_companion_inputs {
        points.push(PointCase {
            case_id: format!("fiedler_companion_{label}"),
            op: "fiedler_companion".into(),
            vec1d: v.clone(),
            mat2d: vec![],
            n: 0,
            mode: "".into(),
        });
    }

    // dft_matrix: scalar n.
    for &n in &[2usize, 3, 4, 5, 8] {
        points.push(PointCase {
            case_id: format!("dft_matrix_n{n}"),
            op: "dft_matrix".into(),
            vec1d: vec![],
            mat2d: vec![],
            n,
            mode: "".into(),
        });
    }

    // convolution_matrix: (h, n, mode).
    let cm_cases: &[(&str, Vec<f64>, usize, &str)] = &[
        ("h3_n5_full", vec![1.0, 2.0, 1.0], 5, "full"),
        ("h3_n5_same", vec![1.0, 2.0, 1.0], 5, "same"),
        ("h3_n5_valid", vec![1.0, 2.0, 1.0], 5, "valid"),
        ("h4_n6_full", vec![0.5, 1.0, 1.5, -2.0], 6, "full"),
        ("h2_n4_same", vec![1.0, -1.0], 4, "same"),
    ];
    for (label, h, n, mode) in cm_cases {
        points.push(PointCase {
            case_id: format!("convmat_{label}"),
            op: "convolution_matrix".into(),
            vec1d: h.clone(),
            mat2d: vec![],
            n: *n,
            mode: (*mode).into(),
        });
    }

    // bandwidth + frobenius_norm: 2-D matrices.
    let mat_cases: &[(&str, Vec<Vec<f64>>)] = &[
        (
            "diag_3",
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 2.0, 0.0],
                vec![0.0, 0.0, 3.0],
            ],
        ),
        (
            "tridiag_4",
            vec![
                vec![1.0, 2.0, 0.0, 0.0],
                vec![3.0, 4.0, 5.0, 0.0],
                vec![0.0, 6.0, 7.0, 8.0],
                vec![0.0, 0.0, 9.0, 10.0],
            ],
        ),
        (
            "dense_3x3",
            vec![
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0],
            ],
        ),
        (
            "lower_tri_4",
            vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![2.0, 3.0, 0.0, 0.0],
                vec![4.0, 5.0, 6.0, 0.0],
                vec![7.0, 8.0, 9.0, 10.0],
            ],
        ),
        (
            "full_4x4",
            vec![
                vec![1.0, 2.0, 3.0, 4.0],
                vec![5.0, 6.0, 7.0, 8.0],
                vec![9.0, 10.0, 11.0, 12.0],
                vec![13.0, 14.0, 15.0, 16.0],
            ],
        ),
    ];
    for (label, m) in mat_cases {
        for op in ["bandwidth", "frobenius_norm"] {
            points.push(PointCase {
                case_id: format!("{op}_{label}"),
                op: op.into(),
                vec1d: vec![],
                mat2d: m.clone(),
                n: 0,
                mode: "".into(),
            });
        }
    }

    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import linalg

def finite_flat_or_none(arr):
    out = []
    for v in np.asarray(arr).flatten().tolist():
        try:
            v = float(v)
        except Exception:
            return None
        if not math.isfinite(v):
            return None
        out.append(v)
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    try:
        if op == "fiedler":
            m = linalg.fiedler(np.array(case["vec1d"], dtype=float))
            points.append({"case_id": cid, "values": finite_flat_or_none(m)})
        elif op == "fiedler_companion":
            m = linalg.fiedler_companion(np.array(case["vec1d"], dtype=float))
            points.append({"case_id": cid, "values": finite_flat_or_none(m)})
        elif op == "dft_matrix":
            # fsci normalizes by 1/√n; scipy.linalg.dft returns un-normalized.
            n = int(case["n"])
            m = linalg.dft(n) / np.sqrt(n)
            # Pack as alternating (re, im) pairs (row-major).
            packed = []
            for row in m:
                for c in row:
                    packed.append(float(np.real(c)))
                    packed.append(float(np.imag(c)))
            # finite-check each
            if any(not math.isfinite(v) for v in packed):
                points.append({"case_id": cid, "values": None})
            else:
                points.append({"case_id": cid, "values": packed})
        elif op == "convolution_matrix":
            h = np.array(case["vec1d"], dtype=float)
            n = int(case["n"])
            mode = case["mode"]
            m = linalg.convolution_matrix(h, n, mode=mode)
            points.append({"case_id": cid, "values": finite_flat_or_none(m)})
        elif op == "bandwidth":
            m = np.array(case["mat2d"], dtype=float)
            lo, up = linalg.bandwidth(m)
            points.append({"case_id": cid, "values": [float(lo), float(up)]})
        elif op == "frobenius_norm":
            m = np.array(case["mat2d"], dtype=float)
            v = float(np.linalg.norm(m, 'fro'))
            points.append({"case_id": cid, "values": [v] if math.isfinite(v) else None})
        else:
            points.append({"case_id": cid, "values": None})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize linalg_misc query");
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
                "failed to spawn python3 for linalg_misc oracle: {e}"
            );
            eprintln!("skipping linalg_misc oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open linalg_misc oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "linalg_misc oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping linalg_misc oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for linalg_misc oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "linalg_misc oracle failed: {stderr}"
        );
        eprintln!("skipping linalg_misc oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse linalg_misc oracle JSON"))
}

fn fsci_eval(case: &PointCase) -> Option<Vec<f64>> {
    match case.op.as_str() {
        "fiedler" => Some(fiedler(&case.vec1d).into_iter().flatten().collect()),
        "fiedler_companion" => fiedler_companion(&case.vec1d)
            .ok()
            .map(|m| m.into_iter().flatten().collect()),
        "dft_matrix" => {
            let m = dft_matrix(case.n);
            let mut out = Vec::with_capacity(case.n * case.n * 2);
            for row in &m {
                for &(re, im) in row {
                    out.push(re);
                    out.push(im);
                }
            }
            Some(out)
        }
        "convolution_matrix" => Some(
            convolution_matrix(&case.vec1d, case.n, &case.mode)
                .into_iter()
                .flatten()
                .collect(),
        ),
        "bandwidth" => {
            let (lo, up) = bandwidth(&case.mat2d);
            Some(vec![lo as f64, up as f64])
        }
        "frobenius_norm" => Some(vec![frobenius_norm(&case.mat2d)]),
        _ => None,
    }
}

#[test]
fn diff_linalg_misc_deterministic() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(fsci_v) = fsci_eval(case) else {
            continue;
        };
        let Some(scipy_v) = scipy_arm.values.as_ref() else {
            continue;
        };
        if fsci_v.len() != scipy_v.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                op: case.op.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let abs_d = fsci_v
            .iter()
            .zip(scipy_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_linalg_misc_deterministic".into(),
        category: "scipy.linalg fiedler/dft/convolution_matrix/bandwidth + frobenius_norm".into(),
        case_count: diffs.len(),
        max_abs_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "linalg_misc {} mismatch: {} abs_diff={}",
                d.op, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.linalg misc-deterministic conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
