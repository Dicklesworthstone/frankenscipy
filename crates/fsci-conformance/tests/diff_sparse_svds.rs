#![forbid(unsafe_code)]
//! Live scipy.sparse.linalg.svds parity for fsci_sparse::svds.
//!
//! Krylov–Schur plus projected SVD against SciPy's ARPACK path, covering
//! largest/smallest triplets, tall/wide inputs and rank deficiency. Singular
//! vectors are checked through their two residual equations and orthonormality
//! rather than their arbitrary signs or the basis of a repeated singular space.
//! Zero-matrix factors are checked analytically in fsci-sparse: SciPy 1.17.1's
//! ARPACK zero-matrix test is skipped upstream (gh-21110), so it is not a live oracle.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_sparse::{CooMatrix, EigsOptions, FormatConvertible, LinearOperator, Shape2D, svds};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-4;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    rows: usize,
    cols: usize,
    triplets: Vec<(usize, usize, f64)>,
    k: usize,
    which: String,
    v0: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    sigma_sorted: Option<Vec<f64>>,
    orthogonality_error: Option<f64>,
    residual_error: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    abs_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
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
    fs::create_dir_all(output_dir()).expect("create svds diff dir");
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

fn rect_dense(rows: usize, cols: usize, seed: u64) -> Vec<(usize, usize, f64)> {
    let mut out = Vec::new();
    let mut s = seed;
    for r in 0..rows {
        for c in 0..cols {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 11) as f64) / (1u64 << 53) as f64;
            let v = (u - 0.5) * 4.0;
            // Skip occasional zeros to make it sparse-ish but ensure full rank.
            if r == c || (((r * 7 + c * 13) % 5) != 0) {
                out.push((r, c, v + if r == c { 5.0 } else { 0.0 }));
            }
        }
    }
    out
}

fn generate_query() -> OracleQuery {
    OracleQuery {
        points: vec![
            Case {
                case_id: "tall_8x5_k2".into(),
                rows: 8,
                cols: 5,
                triplets: rect_dense(8, 5, 0xdead_beef_cafe_babe),
                k: 2,
                which: "LM".into(),
                v0: None,
            },
            Case {
                case_id: "wide_6x10_k2".into(),
                rows: 6,
                cols: 10,
                triplets: rect_dense(6, 10, 0x1234_5678_90ab_cdef),
                k: 2,
                which: "LM".into(),
                v0: None,
            },
            Case {
                case_id: "square_10x10_k3".into(),
                rows: 10,
                cols: 10,
                triplets: rect_dense(10, 10, 0xfeed_face_dead_beef),
                k: 3,
                which: "LM".into(),
                v0: None,
            },
            Case {
                case_id: "tall_11x5_smallest_start_vector".into(),
                rows: 11,
                cols: 5,
                triplets: rect_dense(11, 5, 0xf00d_cafe),
                k: 2,
                which: "SM".into(),
                v0: Some(vec![1.0, -2.0, 3.0, -4.0, 5.0]),
            },
            Case {
                case_id: "wide_5x200_smallest_start_vector".into(),
                rows: 5,
                cols: 200,
                triplets: rect_dense(5, 200, 0xface_beef),
                k: 2,
                which: "SM".into(),
                v0: Some(vec![1.0, -2.0, 3.0, -4.0, 5.0]),
            },
            Case {
                case_id: "tall_rank_two_k4".into(),
                rows: 8,
                cols: 5,
                triplets: vec![(0, 0, 6.0), (1, 1, 3.0)],
                k: 4,
                which: "LM".into(),
                v0: None,
            },
            Case {
                case_id: "wide_rank_two_k4".into(),
                rows: 5,
                cols: 8,
                triplets: vec![(0, 0, 6.0), (1, 1, 3.0)],
                k: 4,
                which: "LM".into(),
                v0: None,
            },
            Case {
                case_id: "tall_rank_one_dense_k3".into(),
                rows: 7,
                cols: 5,
                triplets: (0..7)
                    .flat_map(|row| (0..5).map(move |col| (row, col, 1.0)))
                    .collect(),
                k: 3,
                which: "LM".into(),
                v0: None,
            },
            Case {
                case_id: "wide_5x200_largest_start_vector".into(),
                rows: 5,
                cols: 200,
                triplets: rect_dense(5, 200, 0xface_beef),
                k: 2,
                which: "LM".into(),
                v0: Some(vec![1.0, -2.0, 3.0, -4.0, 5.0]),
            },
            Case {
                case_id: "clustered_smallest_triplets".into(),
                rows: 9,
                cols: 6,
                triplets: [3.0, 3.0 + 1e-8, 3.0 + 2e-8, 1.0, 1.0 + 1e-8, 1.0 + 2e-8]
                    .into_iter()
                    .enumerate()
                    .map(|(i, value)| (i, i, value))
                    .collect(),
                k: 3,
                which: "SM".into(),
                v0: None,
            },
        ],
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    m = int(case["rows"]); n = int(case["cols"])
    rs = [int(t[0]) for t in case["triplets"]]
    cs = [int(t[1]) for t in case["triplets"]]
    vs = [float(t[2]) for t in case["triplets"]]
    k = int(case["k"])
    try:
        A = csr_matrix((vs, (rs, cs)), shape=(m, n)).astype(float)
        v0 = None if case["v0"] is None else np.asarray(case["v0"], dtype=float)
        u, s, vt = svds(A, k=k, which=case["which"], v0=v0,
                        rng=np.random.default_rng(170))
        sorted_s = sorted([float(x) for x in s.tolist()], reverse=True)
        if all(math.isfinite(x) for x in sorted_s):
            orth = max(float(np.max(np.abs(u.T @ u - np.eye(k)))),
                       float(np.max(np.abs(vt @ vt.T - np.eye(k)))))
            scale = max(1.0, float(np.linalg.norm(A.data)))
            residual = max(float(np.linalg.norm(A @ vt.T - u * s)),
                           float(np.linalg.norm(A.T @ u - vt.T * s))) / scale
            points.append({"case_id": cid, "sigma_sorted": sorted_s,
                           "orthogonality_error": orth, "residual_error": residual})
        else:
            points.append({"case_id": cid, "sigma_sorted": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "sigma_sorted": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize query");
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
                "failed to spawn python3 for svds oracle: {e}"
            );
            eprintln!("skipping svds oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "svds oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping svds oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for svds oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "svds oracle failed: {stderr}"
        );
        eprintln!("skipping svds oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse svds oracle JSON"))
}

/// Check both singular equations and both orthonormality conditions before
/// making the singular values available to the compared-case ledger.
fn fsci_sigma_sorted(case: &Case, opts: EigsOptions) -> Option<Vec<f64>> {
    let mut data = Vec::new();
    let mut rs = Vec::new();
    let mut cs = Vec::new();
    for &(r, c, v) in &case.triplets {
        data.push(v);
        rs.push(r);
        cs.push(c);
    }
    let coo =
        CooMatrix::from_triplets(Shape2D::new(case.rows, case.cols), data, rs, cs, true).ok()?;
    let csr = coo.to_csr().ok()?;
    let res = svds(
        &csr,
        case.k,
        EigsOptions {
            which: case.which.parse().ok()?,
            v0: case.v0.as_deref(),
            ..opts
        },
    )
    .ok()?;
    assert!(res.converged, "{}: unconverged {res:?}", case.case_id);
    assert_eq!(res.singular_values.len(), case.k);
    assert_eq!(res.u.len(), case.k);
    assert_eq!(res.vt.len(), case.k);
    let scale = csr.data().iter().fold(0.0_f64, |n, &v| n.hypot(v)).max(1.0);
    for i in 0..case.k {
        let sigma = res.singular_values[i];
        let av = csr.matvec(&res.vt[i]).expect("A v");
        let atu = csr.rmatvec(&res.u[i]).expect("A^T u");
        for (image, vector) in [(&av, &res.u[i]), (&atu, &res.vt[i])] {
            let error = image
                .iter()
                .zip(vector)
                .fold(0.0_f64, |e, (&x, &y)| e.hypot(x - sigma * y));
            assert!(error / scale < 1e-9, "{}: residual {error:e}", case.case_id);
        }
        for j in 0..=i {
            let want = if i == j { 1.0 } else { 0.0 };
            for vectors in [&res.u, &res.vt] {
                let inner: f64 = vectors[i].iter().zip(&vectors[j]).map(|(x, y)| x * y).sum();
                assert!(
                    (inner - want).abs() < 1e-10,
                    "{}: inner product {inner:e}",
                    case.case_id
                );
            }
        }
    }
    let mut sigs = res.singular_values.clone();
    sigs.sort_by(|a, b| b.total_cmp(a));
    Some(sigs)
}

#[test]
fn diff_sparse_svds() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let opts = EigsOptions::default();
    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;
    let mut ledger = CompareLedger::new("diff_sparse_svds", &["svds"]);

    for case in &query.points {
        if let Some(arm) = pmap.get(&case.case_id)
            && arm.sigma_sorted.is_some()
        {
            assert!(
                arm.orthogonality_error.is_some_and(|error| error < 1e-10),
                "{}: {arm:?}",
                case.case_id
            );
            assert!(
                arm.residual_error.is_some_and(|error| error < 1e-9),
                "{}: {arm:?}",
                case.case_id
            );
        }
        let scipy_sigma = pmap
            .get(&case.case_id)
            .and_then(|arm| arm.sigma_sorted.as_deref());
        let fsci_sigma = fsci_sigma_sorted(case, opts);
        let Some((expected, sigs)) =
            ledger.slices("svds", &case.case_id, scipy_sigma, fsci_sigma.as_deref())
        else {
            continue;
        };
        let abs_d = sigs
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        max_overall = max_overall.max(abs_d);
        ledger.compared("svds", &case.case_id, abs_d <= ABS_TOL);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_sparse_svds".into(),
        category: "fsci_sparse::svds vs scipy.sparse.linalg.svds (LM/SM triplets)".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        max_abs_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("svds mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "svds conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    ledger.finish(query.points.len());
}
