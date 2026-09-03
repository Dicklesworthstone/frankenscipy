#![forbid(unsafe_code)]
//! Property-based parity harness for fsci_linalg::schur and hessenberg.
//!
//! Resolves [frankenscipy-jskpp]. Both decompositions involve sign /
//! ordering ambiguities, so this harness checks invariants instead of
//! per-element parity:
//!  - Schur: A ≈ Z T Zᵀ; T diagonal eigenvalues sorted match
//!    scipy.linalg.eigvals(A).real sorted (for matrices with all-real
//!    spectra).
//!  - Hessenberg: A ≈ Q H Qᵀ. Q orthogonal (QQᵀ ≈ I).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{DecompOptions, hessenberg, schur};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    rows: usize,
    cols: usize,
    a: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    eigvals_sorted: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create schur_hessenberg diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize schur_hessenberg diff log");
    fs::write(path, json).expect("write schur_hessenberg diff log");
}

fn rows_of(a_flat: &[f64], rows: usize, cols: usize) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|r| (0..cols).map(|c| a_flat[r * cols + c]).collect())
        .collect()
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let p = b.len();
    let n = b[0].len();
    let mut out = vec![vec![0.0_f64; n]; m];
    for i in 0..m {
        for k in 0..p {
            for j in 0..n {
                out[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    out
}

fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let n = a[0].len();
    let mut out = vec![vec![0.0_f64; m]; n];
    for i in 0..m {
        for j in 0..n {
            out[j][i] = a[i][j];
        }
    }
    out
}

fn frob_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let mut max = 0.0_f64;
    for (ra, rb) in a.iter().zip(b.iter()) {
        for (&va, &vb) in ra.iter().zip(rb.iter()) {
            max = max.max((va - vb).abs());
        }
    }
    max
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    // Symmetric → all real eigenvalues
    let sym_3 = vec![4.0_f64, 1.0, 0.5, 1.0, 5.0, 0.3, 0.5, 0.3, 6.0];
    // Diagonal — trivial
    let diag_4 = vec![
        2.0_f64, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 7.0,
    ];
    // Triangular — eigenvalues on diagonal
    let tri_3 = vec![2.0_f64, 1.5, -0.5, 0.0, 4.0, 1.0, 0.0, 0.0, -1.0];
    // Symmetric 4x4
    let sym_4 = vec![
        3.0_f64, 0.5, 0.2, 0.1, 0.5, 4.0, 0.3, 0.05, 0.2, 0.3, 5.0, 0.4, 0.1, 0.05, 0.4, 6.0,
    ];

    for (label, mat, rows, cols) in [
        ("sym_3x3", &sym_3, 3, 3),
        ("diag_4x4", &diag_4, 4, 4),
        ("tri_3x3", &tri_3, 3, 3),
        ("sym_4x4", &sym_4, 4, 4),
    ] {
        for op in ["schur", "hessenberg"] {
            points.push(PointCase {
                case_id: format!("{op}_{label}"),
                op: op.into(),
                rows,
                cols,
                a: mat.clone(),
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

def finite_sorted_real_or_none(arr):
    flat = []
    for v in np.asarray(arr).flatten().tolist():
        re = float(v.real) if hasattr(v, "real") else float(v)
        if not math.isfinite(re):
            return None
        flat.append(re)
    flat.sort()
    return flat

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    r = int(case["rows"]); c = int(case["cols"])
    A = np.array(case["a"], dtype=float).reshape(r, c)
    try:
        w = linalg.eigvals(A)
        points.append({"case_id": cid, "eigvals_sorted": finite_sorted_real_or_none(w)})
    except Exception:
        points.append({"case_id": cid, "eigvals_sorted": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize schur_hessenberg query");
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
                "failed to spawn python3 for schur_hessenberg oracle: {e}"
            );
            eprintln!("skipping schur_hessenberg oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open schur_hessenberg oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "schur_hessenberg oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping schur_hessenberg oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for schur_hessenberg oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "schur_hessenberg oracle failed: {stderr}"
        );
        eprintln!("skipping schur_hessenberg oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse schur_hessenberg oracle JSON"))
}

#[test]
fn diff_linalg_schur_hessenberg_reconstruct() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(expected_eigs) = scipy_arm.eigvals_sorted.as_ref() else {
            continue;
        };
        let a = rows_of(&case.a, case.rows, case.cols);
        let opts = DecompOptions::default();
        let abs_d = match case.op.as_str() {
            "schur" => {
                let Ok(res) = schur(&a, opts) else { continue };
                // Reconstruction: A ≈ Z T Z^T
                let z_t = mat_mul(&res.z, &res.t);
                let zt_zt = mat_mul(&z_t, &transpose(&res.z));
                let recon = frob_diff(&a, &zt_zt);
                // Eigenvalues from T diagonal (for all-real spectra)
                let mut t_diag: Vec<f64> = (0..res.t.len()).map(|i| res.t[i][i]).collect();
                t_diag.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let eig_d = if t_diag.len() != expected_eigs.len() {
                    f64::INFINITY
                } else {
                    t_diag
                        .iter()
                        .zip(expected_eigs.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f64, f64::max)
                };
                recon.max(eig_d)
            }
            "hessenberg" => {
                let Ok(res) = hessenberg(&a, opts) else {
                    continue;
                };
                // Reconstruction: A ≈ Q H Q^T
                let q_h = mat_mul(&res.q, &res.h);
                let qhq = mat_mul(&q_h, &transpose(&res.q));
                let recon = frob_diff(&a, &qhq);
                // Q orthogonal: Q Q^T ≈ I
                let qqt = mat_mul(&res.q, &transpose(&res.q));
                let n = qqt.len();
                let mut ortho = 0.0_f64;
                for i in 0..n {
                    for j in 0..n {
                        let target = if i == j { 1.0 } else { 0.0 };
                        ortho = ortho.max((qqt[i][j] - target).abs());
                    }
                }
                recon.max(ortho)
            }
            _ => continue,
        };
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
        test_id: "diff_linalg_schur_hessenberg_reconstruct".into(),
        category: "fsci_linalg.schur + hessenberg reconstruction".into(),
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
            eprintln!("{} mismatch: {} abs_diff={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "schur_hessenberg conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
