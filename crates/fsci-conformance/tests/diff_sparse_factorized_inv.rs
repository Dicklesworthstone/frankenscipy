#![forbid(unsafe_code)]
//! Live scipy.sparse.linalg parity for `fsci_sparse::factorized` and `fsci_sparse::inv`
//! (frankenscipy-6j5tz).
//!
//! Arms:
//! - `factorized`: `factorized(A)(b)` for two right-hand sides per matrix, relative to
//!   `max|x|`, within 1e-12.
//! - `inv`: the dense values of `inv(A)` relative to `max|A⁻¹|`, within 1e-12, for CSR and CSC
//!   input (the result keeps the input's format on both sides).
//! - `inv_pattern`: the stored-entry pattern of `inv(A)` (SciPy keeps `np.flatnonzero` of each
//!   column) must be identical where the inverse's zeros are structural (diagonal, triangular,
//!   permutation, block-diagonal and 1x1 matrices), so no rounding decides it.
//!
//! An exactly singular or rectangular `A` makes SciPy raise ("Factor is exactly singular",
//! "can only factor square matrices", "matrix must be square"); fsci must refuse it too, and a
//! value from fsci there is a failing case. For a 1x1 `A` SciPy's `inv` returns a dense 1-D
//! array instead of a sparse matrix (its `spsolve` reads the one-column identity as a vector);
//! its value is compared like any other.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_sparse::{
    CooMatrix, CscMatrix, CsrMatrix, FormatConvertible, LinearOperator, Shape2D, factorized, inv,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const REL_TOL: f64 = 1.0e-12;
const ARMS: [&str; 3] = ["factorized", "inv", "inv_pattern"];

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    shape: (usize, usize),
    triplets: Vec<(usize, usize, f64)>,
    format: String,
    rhs: Vec<Vec<f64>>,
    structural_zeros: bool,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    cases: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleCase {
    case_id: String,
    factorized: Option<Vec<Vec<f64>>>,
    factorized_error: Option<String>,
    inv: Option<Vec<Vec<f64>>>,
    inv_format: Option<String>,
    inv_pattern: Option<Vec<(usize, usize)>>,
    inv_error: Option<String>,
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

/// A 64-bit LCG in [0, 1).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn tridiagonal(n: usize) -> Vec<(usize, usize, f64)> {
    let mut t = Vec::new();
    for i in 0..n {
        t.push((i, i, 4.0 + (i % 3) as f64));
        if i + 1 < n {
            t.push((i, i + 1, -1.25));
            t.push((i + 1, i, -2.0));
        }
    }
    t
}

fn random_dominant(n: usize, density: f64, seed: u64) -> Vec<(usize, usize, f64)> {
    let mut rng = Lcg(seed);
    let mut t = Vec::new();
    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n {
            if i != j && rng.next() < density {
                let v = rng.next() * 2.0 - 1.0;
                row_sum += v.abs();
                t.push((i, j, v));
            }
        }
        t.push((i, i, row_sum + 0.5 + rng.next()));
    }
    t
}

fn lower_triangular(n: usize, seed: u64) -> Vec<(usize, usize, f64)> {
    let mut rng = Lcg(seed);
    let mut t = Vec::new();
    for i in 0..n {
        for j in 0..i {
            if rng.next() < 0.3 {
                t.push((i, j, rng.next() * 2.0 - 1.0));
            }
        }
        t.push((i, i, 1.0 + rng.next()));
    }
    t
}

fn permutation(n: usize) -> Vec<(usize, usize, f64)> {
    (0..n)
        .map(|i| (i, (i * 5 + 3) % n, 0.5 + i as f64))
        .collect()
}

fn block_diagonal(blocks: usize) -> Vec<(usize, usize, f64)> {
    let mut t = Vec::new();
    for b in 0..blocks {
        let o = 3 * b;
        let base = [[4.0, 1.0, 0.5], [-1.0, 3.0, 0.25], [0.75, -0.5, 5.0]];
        for (i, row) in base.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                t.push((o + i, o + j, v * (1.0 + 0.1 * b as f64)));
            }
        }
    }
    t
}

fn generate_query() -> OracleQuery {
    let mut specs: Vec<(String, (usize, usize), Vec<(usize, usize, f64)>, bool)> = vec![
        ("tridiag_8".into(), (8, 8), tridiagonal(8), false),
        ("tridiag_40".into(), (40, 40), tridiagonal(40), false),
        ("tridiag_120".into(), (120, 120), tridiagonal(120), false),
        (
            "random_10".into(),
            (10, 10),
            random_dominant(10, 0.3, 11),
            false,
        ),
        (
            "random_30".into(),
            (30, 30),
            random_dominant(30, 0.15, 12),
            false,
        ),
        (
            "random_60".into(),
            (60, 60),
            random_dominant(60, 0.08, 13),
            false,
        ),
        ("lower_20".into(), (20, 20), lower_triangular(20, 14), true),
        ("permutation_12".into(), (12, 12), permutation(12), true),
        ("block_diag_15".into(), (15, 15), block_diagonal(5), true),
        (
            "diagonal_9".into(),
            (9, 9),
            (0..9).map(|i| (i, i, 1.0 + i as f64 * 0.5)).collect(),
            true,
        ),
        ("one_by_one".into(), (1, 1), vec![(0, 0, -3.5)], true),
        // SciPy raises: exactly singular (two equal rows), and rectangular.
        (
            "singular_4".into(),
            (4, 4),
            vec![
                (0, 0, 1.0),
                (0, 1, 2.0),
                (1, 0, 1.0),
                (1, 1, 2.0),
                (2, 2, 3.0),
                (3, 3, 4.0),
            ],
            true,
        ),
        (
            "rectangular_3x4".into(),
            (3, 4),
            vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0), (0, 3, 1.0)],
            true,
        ),
    ];
    let mut cases = Vec::new();
    for (name, shape, triplets, structural_zeros) in specs.drain(..) {
        let rhs: Vec<Vec<f64>> = (0..2)
            .map(|k| {
                (0..shape.0)
                    .map(|i| ((i + 1) as f64 * (0.37 + k as f64)).sin() + k as f64)
                    .collect()
            })
            .collect();
        for format in ["csr", "csc"] {
            cases.push(Case {
                case_id: format!("{name}_{format}"),
                shape,
                triplets: triplets.clone(),
                format: format.to_string(),
                rhs: rhs.clone(),
                structural_zeros,
            });
        }
    }
    OracleQuery { cases }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
import warnings
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized, inv

warnings.simplefilter("ignore")
q = json.load(sys.stdin)
out = []
for case in q["cases"]:
    cid = case["case_id"]
    rows, cols = case["shape"]
    r = [int(t[0]) for t in case["triplets"]]
    c = [int(t[1]) for t in case["triplets"]]
    v = [float(t[2]) for t in case["triplets"]]
    A = sp.coo_array((v, (r, c)), shape=(rows, cols)).asformat(case["format"])
    row = {"case_id": cid, "factorized": None, "factorized_error": None, "inv": None,
           "inv_format": None, "inv_pattern": None, "inv_error": None}
    try:
        solve = factorized(A)
        row["factorized"] = [[float(x) for x in solve(np.array(b, dtype=float))]
                             for b in case["rhs"]]
    except Exception as e:
        row["factorized_error"] = f"{type(e).__name__}: {e}"
    try:
        Ai = inv(A)
        if sp.issparse(Ai):
            row["inv"] = Ai.toarray().tolist()
            row["inv_format"] = Ai.format
            coo = Ai.tocoo()
            row["inv_pattern"] = sorted([int(i), int(j)] for i, j in zip(coo.row, coo.col))
        else:
            # A 1x1 A: spsolve reads the one-column identity as a vector and returns a dense
            # 1-D array. Same value; reported as such.
            dense = np.asarray(Ai, dtype=float).reshape(rows, cols)
            row["inv"] = dense.tolist()
            row["inv_format"] = "ndarray"
            row["inv_pattern"] = sorted([int(i), int(j)] for i, j in zip(*np.nonzero(dense)))
    except Exception as e:
        row["inv_error"] = f"{type(e).__name__}: {e}"
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
                "failed to spawn python for factorized/inv oracle: {e}"
            );
            eprintln!("skipping factorized/inv oracle: python not available ({e})");
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
                "factorized/inv oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping factorized/inv oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "factorized/inv oracle failed: {stderr}"
        );
        eprintln!("skipping factorized/inv oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse oracle JSON"))
}

fn build(case: &Case) -> (CsrMatrix, CscMatrix) {
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for &(i, j, x) in &case.triplets {
        r.push(i);
        c.push(j);
        v.push(x);
    }
    let coo = CooMatrix::from_triplets(Shape2D::new(case.shape.0, case.shape.1), v, r, c, true)
        .expect("coo");
    (coo.to_csr().expect("csr"), coo.to_csc().expect("csc"))
}

/// The inverse's dense values and stored-entry pattern, through the input's own format.
fn fsci_inverse(case: &Case) -> Result<(Vec<Vec<f64>>, BTreeSet<(usize, usize)>), String> {
    let (csr, csc) = build(case);
    let (dense, pattern) = if case.format == "csr" {
        let inverse: CsrMatrix = inv(&csr).map_err(|e| e.to_string())?;
        let mut pattern = BTreeSet::new();
        for row in 0..inverse.shape().rows {
            for idx in inverse.indptr()[row]..inverse.indptr()[row + 1] {
                pattern.insert((row, inverse.indices()[idx]));
            }
        }
        (inverse.to_dense().map_err(|e| e.to_string())?, pattern)
    } else {
        let inverse: CscMatrix = inv(&csc).map_err(|e| e.to_string())?;
        let mut pattern = BTreeSet::new();
        for col in 0..inverse.shape().cols {
            for idx in inverse.indptr()[col]..inverse.indptr()[col + 1] {
                pattern.insert((inverse.indices()[idx], col));
            }
        }
        (inverse.to_dense().map_err(|e| e.to_string())?, pattern)
    };
    Ok((dense, pattern))
}

fn fsci_factorized(case: &Case) -> Result<Vec<Vec<f64>>, String> {
    let (_, csc) = build(case);
    let solve = factorized(&csc).map_err(|e| e.to_string())?;
    case.rhs
        .iter()
        .map(|b| solve.solve(b).map_err(|e| e.to_string()))
        .collect()
}

fn relative_max_diff(scipy: &[Vec<f64>], fsci: &[Vec<f64>]) -> f64 {
    if scipy.len() != fsci.len() || scipy.iter().zip(fsci).any(|(a, b)| a.len() != b.len()) {
        return f64::INFINITY;
    }
    let scale = scipy
        .iter()
        .flatten()
        .fold(0.0_f64, |m, v| m.max(v.abs()))
        .max(f64::MIN_POSITIVE);
    let diff = scipy
        .iter()
        .flatten()
        .zip(fsci.iter().flatten())
        .map(|(a, b)| {
            if a.is_finite() && b.is_finite() {
                (a - b).abs()
            } else {
                f64::INFINITY
            }
        })
        .fold(0.0_f64, f64::max);
    diff / scale
}

#[test]
fn diff_sparse_factorized_inv() {
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
    let mut ledger = CompareLedger::new("diff_sparse_factorized_inv", &ARMS);
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
    let mut pattern_cases = 0;

    for case in &query.cases {
        let id = case.case_id.as_str();
        let Some(scipy) = by_id.get(id) else {
            for arm in ARMS {
                ledger.oracle_missing(arm, id, "case missing from oracle output");
            }
            continue;
        };

        // factorized
        let fsci = fsci_factorized(case);
        if scipy.factorized_error.is_some() {
            ledger.expected_raise("factorized", id, fsci.is_err());
        } else {
            match (scipy.factorized.as_ref(), fsci) {
                (Some(s), Ok(f)) => {
                    let metric = relative_max_diff(s, &f) / REL_TOL;
                    record(&mut ledger, id, "factorized", metric);
                }
                (None, _) => ledger.oracle_missing("factorized", id, "no value"),
                (Some(_), Err(e)) => ledger.rust_failed("factorized", id, &e),
            }
        }

        // inv and its pattern
        let fsci = fsci_inverse(case);
        if case.structural_zeros {
            pattern_cases += 1;
        }
        if scipy.inv_error.is_some() {
            ledger.expected_raise("inv", id, fsci.is_err());
            if case.structural_zeros {
                ledger.expected_raise("inv_pattern", id, fsci.is_err());
            }
            continue;
        }
        // SciPy keeps the input format, except that a 1x1 A comes back as a dense 1-D array
        // (its spsolve reads the one-column identity as a vector); fsci keeps the format there
        // too, and the value is compared all the same.
        let expected_format = if case.shape == (1, 1) {
            "ndarray"
        } else {
            case.format.as_str()
        };
        assert_eq!(
            scipy.inv_format.as_deref(),
            Some(expected_format),
            "{id}: SciPy's inv result container"
        );
        match (scipy.inv.as_ref(), fsci) {
            (Some(s), Ok((dense, pattern))) => {
                record(
                    &mut ledger,
                    id,
                    "inv",
                    relative_max_diff(s, &dense) / REL_TOL,
                );
                if case.structural_zeros {
                    let scipy_pattern: BTreeSet<(usize, usize)> = scipy
                        .inv_pattern
                        .as_ref()
                        .map(|p| p.iter().copied().collect())
                        .unwrap_or_default();
                    let same = scipy_pattern == pattern;
                    record(
                        &mut ledger,
                        id,
                        "inv_pattern",
                        if same { 0.0 } else { f64::INFINITY },
                    );
                }
            }
            (None, _) => {
                ledger.oracle_missing("inv", id, "no value");
                if case.structural_zeros {
                    ledger.oracle_missing("inv_pattern", id, "no value");
                }
            }
            (Some(_), Err(e)) => {
                ledger.rust_failed("inv", id, &e);
                if case.structural_zeros {
                    ledger.rust_failed("inv_pattern", id, &e);
                }
            }
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_sparse_factorized_inv".into(),
        category: "fsci_sparse::{factorized, inv} vs scipy.sparse.linalg.{factorized, inv}".into(),
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
            "factorized/inv mismatch: {} {} metric={}",
            d.case_id, d.arm, d.metric
        );
    }
    eprintln!("factorized/inv worst metric per arm (1.0 = tolerance): {max_metric:?}");
    ledger.finish(pattern_cases.min(query.cases.len()));
}
