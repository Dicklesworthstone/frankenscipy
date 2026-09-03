#![forbid(unsafe_code)]
//! Live numpy.fft parity for fsci_fft::{fft2, ifft2}.
//!
//! Resolves [frankenscipy-lxo9j]. Compares 2D complex FFT and its
//! inverse against numpy.fft.{fft2, ifft2} on a range of shapes
//! (square + rectangular, power-of-two + non-power-of-two, including
//! a prime row count). Audit-variant equivalence is already covered
//! by diff_fft_audit_variants_nd_equivalence; this fills the missing
//! numerical-parity gap against the canonical numpy reference.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_fft::{Complex64, FftOptions, fft2, ifft2};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct CasePoint {
    case_id: String,
    /// "fft2" | "ifft2"
    op: String,
    rows: usize,
    cols: usize,
    /// Flattened row-major, real parts then imag parts interleaved
    real: Vec<f64>,
    imag: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<CasePoint>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
struct OraclePoint {
    case_id: String,
    real: Option<Vec<f64>>,
    imag: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<OraclePoint>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
    rows: usize,
    cols: usize,
    max_abs_diff: f64,
    pass: bool,
    note: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create fft2 diff dir");
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

/// Build a deterministic signal of (real, imag) values for a given shape.
fn make_signal(rows: usize, cols: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    // Cheap deterministic generator (LCG)
    let mut state = seed
        .wrapping_mul(2_862_933_555_777_941_757_u64)
        .wrapping_add(3037000493);
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64) / ((1u64 << 31) as f64) - 1.0
    };
    let n = rows * cols;
    let mut real = Vec::with_capacity(n);
    let mut imag = Vec::with_capacity(n);
    for _ in 0..n {
        real.push(next());
        imag.push(next());
    }
    (real, imag)
}

fn build_query() -> OracleQuery {
    let mut pts = Vec::new();
    let shapes: &[(usize, usize)] = &[
        (4, 4),   // power-of-two square
        (8, 8),   // larger PoT square
        (3, 4),   // non-PoT rectangular
        (5, 7),   // both dims non-PoT (5 is prime)
        (16, 4),  // mixed PoT
        (6, 8),   // non-PoT × PoT
        (1, 16),  // degenerate row
        (10, 10), // mid-size non-PoT square
    ];
    for (i, &(rows, cols)) in shapes.iter().enumerate() {
        let (re, im) = make_signal(rows, cols, 12345 + i as u64);
        // fft2
        pts.push(CasePoint {
            case_id: format!("fft2_{rows}x{cols}"),
            op: "fft2".into(),
            rows,
            cols,
            real: re.clone(),
            imag: im.clone(),
        });
        // ifft2
        pts.push(CasePoint {
            case_id: format!("ifft2_{rows}x{cols}"),
            op: "ifft2".into(),
            rows,
            cols,
            real: re,
            imag: im,
        });
    }
    OracleQuery { points: pts }
}

fn scipy_oracle_or_skip(q: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json, math, sys
import numpy as np

q = json.load(sys.stdin)
out = []
for c in q["points"]:
    cid = c["case_id"]
    try:
        rows, cols = int(c["rows"]), int(c["cols"])
        real = np.array(c["real"], dtype=float).reshape(rows, cols)
        imag = np.array(c["imag"], dtype=float).reshape(rows, cols)
        signal = real + 1j * imag
        if c["op"] == "fft2":
            r = np.fft.fft2(signal)
        elif c["op"] == "ifft2":
            r = np.fft.ifft2(signal)
        else:
            r = None
        if r is None or not np.all(np.isfinite(r.real)) or not np.all(np.isfinite(r.imag)):
            out.append({"case_id": cid, "real": None, "imag": None})
        else:
            out.append({
                "case_id": cid,
                "real": [float(v) for v in r.real.flatten()],
                "imag": [float(v) for v in r.imag.flatten()],
            })
    except Exception:
        out.append({"case_id": cid, "real": None, "imag": None})

print(json.dumps({"points": out}))
"#;
    let query_json = serde_json::to_string(q).expect("serialize");
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
                "python3 spawn failed: {e}"
            );
            eprintln!("skipping fft2 oracle: python3 unavailable ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping fft2 oracle: stdin write failed");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "oracle failed: {stderr}"
        );
        eprintln!("skipping fft2 oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse JSON"))
}

#[test]
fn diff_fft_fft2_ifft2_numpy_parity() {
    let query = build_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let opts = FftOptions::default();

    for (case, o) in query.points.iter().zip(oracle.points.iter()) {
        assert_eq!(case.case_id, o.case_id);
        let (Some(exp_re), Some(exp_im)) = (o.real.as_ref(), o.imag.as_ref()) else {
            continue;
        };

        let signal: Vec<Complex64> = case
            .real
            .iter()
            .zip(case.imag.iter())
            .map(|(&r, &i)| (r, i))
            .collect();

        let result = match case.op.as_str() {
            "fft2" => fft2(&signal, (case.rows, case.cols), &opts),
            "ifft2" => ifft2(&signal, (case.rows, case.cols), &opts),
            other => panic!("unknown op {other}"),
        };
        let out = match result {
            Ok(v) => v,
            Err(e) => {
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    op: case.op.clone(),
                    rows: case.rows,
                    cols: case.cols,
                    max_abs_diff: f64::INFINITY,
                    pass: false,
                    note: format!("fft2/ifft2 error: {e:?}"),
                });
                continue;
            }
        };

        if out.len() != exp_re.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                op: case.op.clone(),
                rows: case.rows,
                cols: case.cols,
                max_abs_diff: f64::INFINITY,
                pass: false,
                note: format!("length mismatch: fsci={} numpy={}", out.len(), exp_re.len()),
            });
            continue;
        }

        let mut max_abs = 0.0_f64;
        for (idx, &(re, im)) in out.iter().enumerate() {
            let dr = (re - exp_re[idx]).abs();
            let di = (im - exp_im[idx]).abs();
            max_abs = max_abs.max(dr.max(di));
        }
        let pass = max_abs <= ABS_TOL;
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            rows: case.rows,
            cols: case.cols,
            max_abs_diff: max_abs,
            pass,
            note: String::new(),
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_fft_fft2_ifft2_numpy_parity".into(),
        category: "fsci_fft::{fft2, ifft2} vs numpy.fft".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "fft2/ifft2 mismatch: {} ({}) shape={}x{} max_abs={} note={}",
                d.case_id, d.op, d.rows, d.cols, d.max_abs_diff, d.note
            );
        }
    }

    assert!(
        all_pass,
        "fft2/ifft2 numpy parity failed: {} cases",
        diffs.len()
    );
}
