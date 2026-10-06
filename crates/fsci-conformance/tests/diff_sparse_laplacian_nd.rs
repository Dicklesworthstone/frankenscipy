#![forbid(unsafe_code)]
//! Live scipy.sparse.linalg parity for `fsci_sparse::LaplacianNd` (frankenscipy-6j5tz).
//!
//! Every case builds `scipy.sparse.linalg.LaplacianNd(grid_shape, boundary_conditions=bc)` and
//! compares, arm by arm:
//! - `eigenvalues`: `eigenvalues(m)`, within 1e-13 absolute (closed-form `-4 sin²` sums; NumPy's
//!   vectorized `sin` may differ from libm's by an ulp).
//! - `eigenvectors`: `eigenvectors(m)`. Each eigenvalue cluster (equal to 1e-12) that both sides
//!   return in full must hold the same closed-form vectors, matched as a SET to 1e-13 (NumPy's
//!   `argsort` orders ties by an unspecified rule, fsci's is stable); a cluster cut by `m` only
//!   checks that each fsci vector is a unit eigenvector of SciPy's matvec to 1e-12.
//! - `toarray`, `tosparse`: exact equality of the matrix (SciPy's quirks included: a length-1
//!   axis puts -1 on the diagonal, a length-2 periodic axis has off-diagonal 2).
//! - `matvec`: SciPy's roll-based `_matvec` on a fixed vector, within 4 ulp of the scale.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_sparse::{LaplacianBoundary, LaplacianNd, LinearOperator};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const EIGENVALUE_TOL: f64 = 1.0e-13;
const CLUSTER_TOL: f64 = 1.0e-12;
const EIGENVECTOR_TOL: f64 = 1.0e-13;
const RESIDUAL_TOL: f64 = 1.0e-12;
const ARMS: [&str; 5] = [
    "eigenvalues",
    "eigenvectors",
    "toarray",
    "tosparse",
    "matvec",
];

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    grid: Vec<usize>,
    bc: String,
    m: Option<usize>,
    x: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    cases: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleCase {
    case_id: String,
    eigenvalues: Option<Vec<f64>>,
    eigenvectors: Option<Vec<Vec<f64>>>,
    toarray: Option<Vec<Vec<f64>>>,
    tosparse: Option<Vec<Vec<f64>>>,
    matvec: Option<Vec<f64>>,
    error: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    cases: Vec<OracleCase>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    arm: String,
    metric: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    max_metric: BTreeMap<String, f64>,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn emit_log(log: &DiffLog) {
    fs::create_dir_all(output_dir()).expect("create diff dir");
    let path = output_dir().join(format!("{}.json", log.test_id));
    fs::write(path, serde_json::to_string_pretty(log).expect("serialize")).expect("write log");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn generate_query() -> OracleQuery {
    let grids: [(&str, &[usize], Option<usize>); 11] = [
        ("g5", &[5], None),
        ("g2", &[2], None),
        ("g1x3", &[1, 3], None),
        ("g4x3", &[4, 3], None),
        ("g3x4x2", &[3, 4, 2], None),
        ("g6x5_m3", &[6, 5], Some(3)),
        ("g2x5_m3", &[2, 5], Some(3)),
        ("g4x4_m5", &[4, 4], Some(5)),
        ("g3x3_m20", &[3, 3], Some(20)),
        ("g3x2_m0", &[3, 2], Some(0)),
        ("g7x6", &[7, 6], None),
    ];
    let mut cases = Vec::new();
    for (name, grid, m) in grids {
        let n: usize = grid.iter().product();
        let x: Vec<f64> = (0..n)
            .map(|i| ((i as f64) * 0.913).sin() * 3.0 + 0.25 * i as f64)
            .collect();
        for bc in ["dirichlet", "neumann", "periodic"] {
            cases.push(Case {
                case_id: format!("{name}_{bc}"),
                grid: grid.to_vec(),
                bc: bc.to_string(),
                m,
                x: x.clone(),
            });
        }
    }
    OracleQuery { cases }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
import numpy as np
from scipy.sparse.linalg import LaplacianNd

q = json.load(sys.stdin)
out = []
for case in q["cases"]:
    cid = case["case_id"]
    row = {"case_id": cid, "eigenvalues": None, "eigenvectors": None, "toarray": None,
           "tosparse": None, "matvec": None, "error": None}
    try:
        L = LaplacianNd(tuple(case["grid"]), boundary_conditions=case["bc"])
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e!r}\n")
        row["error"] = repr(e)
        out.append(row)
        continue
    m = case["m"]
    x = np.array(case["x"], dtype=float)
    arms = {
        "eigenvalues": lambda: [float(v) for v in L.eigenvalues(m)],
        "toarray": lambda: np.asarray(L.toarray(), dtype=float).tolist(),
        "tosparse": lambda: np.asarray(L.tosparse().toarray(), dtype=float).tolist(),
        "matvec": lambda: [float(v) for v in L.matvec(x)],
    }
    if m != 0:
        # eigenvectors(0) raises in SciPy (np.column_stack of no columns); not compared.
        arms["eigenvectors"] = lambda: [[float(v) for v in col]
                                        for col in np.asarray(L.eigenvectors(m)).T]
    for name, fn in arms.items():
        try:
            row[name] = fn()
        except Exception as e:
            sys.stderr.write(f"oracle {cid} {name}: {e!r}\n")
    out.append(row)
print(json.dumps({"cases": out}))
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
                "failed to spawn python for LaplacianNd oracle: {e}"
            );
            eprintln!("skipping LaplacianNd oracle: python not available ({e})");
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
                "LaplacianNd oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping LaplacianNd oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "LaplacianNd oracle failed: {stderr}"
        );
        eprintln!("skipping LaplacianNd oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse oracle JSON"))
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// The eigenvector check described in the module docs; returns the worst metric.
fn eigenvector_metric(
    fsci_values: &[f64],
    fsci_vectors: &[Vec<f64>],
    scipy_values: &[f64],
    scipy_vectors: &[Vec<f64>],
    lap: &LaplacianNd,
    all_eigenvalues: &[f64],
) -> f64 {
    let mut worst = 0.0_f64;
    for (&lambda, v) in fsci_values.iter().zip(fsci_vectors) {
        // How many eigenvalues equal to λ exist in the full spectrum, and in each output?
        let in_spectrum = all_eigenvalues
            .iter()
            .filter(|&&mu| (mu - lambda).abs() <= CLUSTER_TOL)
            .count();
        let fsci_cluster: Vec<usize> = (0..fsci_values.len())
            .filter(|&i| (fsci_values[i] - lambda).abs() <= CLUSTER_TOL)
            .collect();
        let scipy_cluster: Vec<usize> = (0..scipy_values.len())
            .filter(|&i| (scipy_values[i] - lambda).abs() <= CLUSTER_TOL)
            .collect();
        if fsci_cluster.len() == in_spectrum && scipy_cluster.len() == in_spectrum {
            // Full cluster on both sides: the same set of closed-form vectors.
            let best = scipy_cluster
                .iter()
                .map(|&i| max_abs_diff(v, &scipy_vectors[i]))
                .fold(f64::INFINITY, f64::min);
            worst = worst.max(best / EIGENVECTOR_TOL);
        } else {
            // Cut by m: v must be a unit eigenvector of the operator for λ.
            let av = lap.matvec(v).expect("matvec");
            let residual = av
                .iter()
                .zip(v)
                .map(|(a, b)| (a - lambda * b).powi(2))
                .sum::<f64>()
                .sqrt();
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            worst = worst.max(residual.max((norm - 1.0).abs()) / RESIDUAL_TOL);
        }
    }
    worst
}

#[test]
fn diff_sparse_laplacian_nd() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    let by_id: HashMap<String, OracleCase> = oracle
        .cases
        .into_iter()
        .map(|c| (c.case_id.clone(), c))
        .collect();

    let start = Instant::now();
    let mut ledger = CompareLedger::new("diff_sparse_laplacian_nd", &ARMS);
    let mut diffs = Vec::new();
    let mut max_metric: BTreeMap<String, f64> = BTreeMap::new();
    let mut record = |ledger: &mut CompareLedger, case_id: &str, arm: &str, metric: f64| {
        let pass = metric <= 1.0;
        ledger.compared(arm, case_id, pass);
        let worst = max_metric.entry(arm.to_string()).or_insert(0.0);
        *worst = worst.max(metric);
        diffs.push(CaseDiff {
            case_id: case_id.to_string(),
            arm: arm.to_string(),
            metric,
            pass,
        });
    };

    for case in &query.cases {
        let id = case.case_id.as_str();
        let scipy = by_id.get(id);
        if let Some(error) = scipy.and_then(|s| s.error.as_ref()) {
            for arm in ARMS {
                ledger.oracle_missing(arm, id, error);
            }
            continue;
        }
        let bc: LaplacianBoundary = case.bc.parse().expect("boundary");
        let lap = match LaplacianNd::new(&case.grid, bc) {
            Ok(lap) => lap,
            Err(e) => {
                for arm in ARMS {
                    ledger.rust_failed(arm, id, &e.to_string());
                }
                continue;
            }
        };

        let fsci_values = lap.eigenvalues(case.m);
        if let Some((s, f)) = ledger.slices(
            "eigenvalues",
            id,
            scipy.and_then(|s| s.eigenvalues.as_deref()),
            Some(fsci_values.as_slice()),
        ) {
            record(
                &mut ledger,
                id,
                "eigenvalues",
                max_abs_diff(s, f) / EIGENVALUE_TOL,
            );
        }

        let fsci_vectors = lap.eigenvectors(case.m);
        if case.m == Some(0) {
            // Not an eigenvectors case: SciPy's eigenvectors(0) raises (np.column_stack of no
            // columns) where fsci returns the empty set.
        } else if let Some((sv, fv)) = ledger.both(
            "eigenvectors",
            id,
            scipy.and_then(|s| Some((s.eigenvalues.as_ref()?, s.eigenvectors.as_ref()?))),
            Some(&fsci_vectors),
        ) {
            let (scipy_values, scipy_vectors) = sv;
            if scipy_vectors.len() == fv.len() && scipy_values.len() == fsci_values.len() {
                let metric = eigenvector_metric(
                    &fsci_values,
                    fv,
                    scipy_values,
                    scipy_vectors,
                    &lap,
                    &lap.eigenvalues(None),
                );
                record(&mut ledger, id, "eigenvectors", metric);
            } else {
                record(&mut ledger, id, "eigenvectors", f64::INFINITY);
            }
        }

        let dense = lap.toarray();
        for (arm, scipy_matrix, fsci_matrix) in [
            (
                "toarray",
                scipy.and_then(|s| s.toarray.as_ref()),
                Some(dense.clone()),
            ),
            (
                "tosparse",
                scipy.and_then(|s| s.tosparse.as_ref()),
                lap.tosparse().ok().and_then(|m| m.to_dense().ok()),
            ),
        ] {
            if let Some((s, f)) = ledger.both(arm, id, scipy_matrix, fsci_matrix) {
                let exact = s.len() == f.len()
                    && s.iter()
                        .zip(&f)
                        .all(|(a, b)| a.len() == b.len() && a.iter().zip(b).all(|(p, q)| p == q));
                record(
                    &mut ledger,
                    id,
                    arm,
                    if exact { 0.0 } else { f64::INFINITY },
                );
            }
        }

        let fsci_matvec = lap.matvec(&case.x).ok();
        if let Some((s, f)) = ledger.slices(
            "matvec",
            id,
            scipy.and_then(|s| s.matvec.as_deref()),
            fsci_matvec.as_deref(),
        ) {
            let scale = case.x.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
                * (2.0 * case.grid.len() as f64 + 2.0);
            record(
                &mut ledger,
                id,
                "matvec",
                max_abs_diff(s, f) / (4.0 * f64::EPSILON * scale),
            );
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_sparse_laplacian_nd".into(),
        category: "fsci_sparse::LaplacianNd vs scipy.sparse.linalg.LaplacianNd".into(),
        case_count: query.cases.len(),
        compared: ledger.counts().clone(),
        max_metric: max_metric.clone(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);
    for d in diffs.iter().filter(|d| !d.pass) {
        eprintln!(
            "LaplacianNd mismatch: {} {} metric={}",
            d.case_id, d.arm, d.metric
        );
    }
    eprintln!("LaplacianNd worst metric per arm (1.0 = tolerance): {max_metric:?}");
    // Every arm covers every case except eigenvectors at m = 0 (see above).
    let eigenvector_cases = query.cases.iter().filter(|c| c.m != Some(0)).count();
    ledger.finish(eigenvector_cases);
}
