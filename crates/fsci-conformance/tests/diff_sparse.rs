#![forbid(unsafe_code)]
//! Differential oracle + metamorphic + adversarial tests for FSCI-P2C-004 (Sparse).
//!
//! Implements bd-3jh.15.6 acceptance criteria:
//!   Differential: >=15 cases comparing Rust sparse ops vs dense reference
//!   Metamorphic:  >=6 relation tests
//!   Adversarial:  >=8 edge-case / hostile-input tests
//!
//! All tests emit structured JSON logs to
//! `fixtures/artifacts/FSCI-P2C-004/diff/`.

use fsci_runtime::RuntimeMode;
use fsci_sparse::{
    BsrMatrix, CooArray, CooMatrix, CsrMatrix, DiaMatrix, DokMatrix, FormatConvertible, IndexArray,
    IndexArrayRef, IndexDtype, Shape2D, SolveOptions, SparseArrayOutput, SparseError, SparseFormat,
    SparseIndexArrays, SparseMatrixOutput, SparseNpz, SparseObject, add_csr, coo_to_csr_with_mode,
    csr_to_csc_with_mode, diags, expand_dims, eye, get_index_dtype, issparse, isspmatrix,
    load_npz_from_reader, permute_dims, random, safely_cast_index_arrays, save_npz_to_writer,
    scale_coo, scale_csc, scale_csr, spmv_coo, spmv_csc, spmv_csr, spsolve, sub_csr, swapaxes,
};
use serde::Serialize;
use std::fs;
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// ───────────────────── Structured log types ─────────────────────

#[derive(Debug, Clone, Serialize)]
struct DiffTestLog {
    test_id: String,
    category: String,
    input_summary: String,
    expected: String,
    actual: String,
    diff: f64,
    tolerance: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
}

// ───────────────────── Helpers ─────────────────────

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/artifacts/FSCI-P2C-004/diff")
}

fn ensure_output_dir() {
    let dir = output_dir();
    if !dir.exists() {
        fs::create_dir_all(&dir).expect("create diff output dir");
    }
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffTestLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

const TOL: f64 = 1e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

fn dense_from_coo(coo: &CooMatrix) -> Vec<Vec<f64>> {
    let shape = coo.shape();
    let mut dense = vec![vec![0.0; shape.cols]; shape.rows];
    for idx in 0..coo.nnz() {
        dense[coo.row_indices()[idx]][coo.col_indices()[idx]] += coo.data()[idx];
    }
    dense
}

fn dense_matvec(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.iter().zip(vector).map(|(a, b)| a * b).sum())
        .collect()
}

fn max_abs_diff_vec(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        })
}

fn max_abs_diff_matrix(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(ra, rb)| max_abs_diff_vec(ra, rb))
        .fold(0.0_f64, |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        })
}

fn dense_add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b.iter())
        .map(|(ra, rb)| ra.iter().zip(rb.iter()).map(|(x, y)| x + y).collect())
        .collect()
}

fn dense_sub(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b.iter())
        .map(|(ra, rb)| ra.iter().zip(rb.iter()).map(|(x, y)| x - y).collect())
        .collect()
}

fn dense_scale(a: &[Vec<f64>], s: f64) -> Vec<Vec<f64>> {
    a.iter()
        .map(|row| row.iter().map(|v| v * s).collect())
        .collect()
}

fn make_test_coo(rows: usize, cols: usize, triplets: &[(usize, usize, f64)]) -> CooMatrix {
    let (r, c, d): (Vec<_>, Vec<_>, Vec<_>) = triplets.iter().copied().fold(
        (Vec::new(), Vec::new(), Vec::new()),
        |(mut rs, mut cs, mut ds), (r, c, v)| {
            rs.push(r);
            cs.push(c);
            ds.push(v);
            (rs, cs, ds)
        },
    );
    CooMatrix::from_triplets(Shape2D::new(rows, cols), d, r, c, false).expect("valid test coo")
}

fn append_coo_array_contract(values: &mut Vec<f64>, array: &CooArray) {
    let exact_integer =
        |value: usize| f64::from(u32::try_from(value).expect("small oracle integer must fit u32"));
    values.push(1.0); // `scipy.sparse.coo_array` output family.
    values.push(exact_integer(array.ndim()));
    values.extend(array.shape().iter().copied().map(exact_integer));
    values.push(exact_integer(array.nnz()));
    for axis_coords in array.coords() {
        values.extend(axis_coords.iter().copied().map(exact_integer));
    }
    values.extend_from_slice(array.data());
}

fn run_scipy_sparse_oracle(label: &str, script: &str) -> Option<Vec<f64>> {
    let required = std::env::var_os(REQUIRE_SCIPY_ENV).is_some();
    let mut child = match Command::new("python3")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            assert!(
                !required,
                "{label} oracle required but failed to spawn python3: {err}"
            );
            eprintln!("skipping {label} oracle: python3 not available ({err})");
            return None;
        }
    };

    let Some(stdin) = child.stdin.as_mut() else {
        assert!(
            !required,
            "{label} oracle required but stdin was unavailable"
        );
        eprintln!("skipping {label} oracle: stdin unavailable");
        return None;
    };
    if let Err(err) = stdin.write_all(script.as_bytes()) {
        let output = child.wait_with_output().ok();
        let stderr = output
            .as_ref()
            .map(|output| String::from_utf8_lossy(&output.stderr).into_owned())
            .unwrap_or_default();
        assert!(
            !required,
            "{label} oracle required but stdin write failed: {err}; stderr: {stderr}"
        );
        eprintln!("skipping {label} oracle: stdin write failed ({err})\n{stderr}");
        return None;
    }

    let output = match child.wait_with_output() {
        Ok(output) => output,
        Err(err) => {
            assert!(!required, "{label} oracle required but wait failed: {err}");
            eprintln!("skipping {label} oracle: wait failed ({err})");
            return None;
        }
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !required,
            "{label} oracle required but SciPy exited nonzero:\n{stderr}"
        );
        eprintln!("skipping {label} oracle: SciPy unavailable\n{stderr}");
        return None;
    }

    match serde_json::from_slice::<Vec<f64>>(&output.stdout) {
        Ok(values) => Some(values),
        Err(err) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                !required,
                "{label} oracle required but returned invalid JSON: {err}\nstdout:\n{stdout}\nstderr:\n{stderr}"
            );
            eprintln!(
                "skipping {label} oracle: invalid JSON ({err})\nstdout:\n{stdout}\nstderr:\n{stderr}"
            );
            None
        }
    }
}

fn scipy_spsolve_tridiagonal_4x4() -> Option<Vec<f64>> {
    run_scipy_sparse_oracle(
        "sparse spsolve 4x4",
        r#"
import json
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

row = np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3], dtype=np.int64)
col = np.array([0, 1, 0, 1, 2, 1, 2, 3, 2, 3], dtype=np.int64)
data = np.array([4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 3.0], dtype=np.float64)
rhs = np.array([15.0, 10.0, 10.0, 10.0], dtype=np.float64)
matrix = sparse.coo_matrix((data, (row, col)), shape=(4, 4)).tocsr()
solution = splinalg.spsolve(matrix, rhs)
print(json.dumps([float(value) for value in np.atleast_1d(solution).tolist()]))
"#,
    )
}

fn scipy_spsolve_large_diagonal_samples() -> Option<Vec<f64>> {
    run_scipy_sparse_oracle(
        "large sparse spsolve",
        r#"
import json
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

n = 32769
idx = np.arange(n, dtype=np.float64)
diag = 2.0 + (idx % 7.0) * 0.25
expected = 1.0 + (idx % 11.0) * 0.5
rhs = diag * expected
matrix = sparse.diags(diag, offsets=0, format="csr")
solution = splinalg.spsolve(matrix, rhs)
sample_indices = [0, 1, 17, n // 2, n - 1]
print(json.dumps([float(solution[i]) for i in sample_indices]))
"#,
    )
}

fn scipy_sparse_index_array_contract() -> Option<Vec<f64>> {
    run_scipy_sparse_oracle(
        "sparse index array utilities",
        r#"
import json
import numpy as np
from scipy import sparse

def width(dtype):
    return float(np.dtype(dtype).itemsize * 8)

result = [
    width(sparse.get_index_dtype()),
    width(sparse.get_index_dtype(np.array([0, 1], dtype=np.int32))),
    width(sparse.get_index_dtype(np.array([0, 1], dtype=np.int64))),
    width(sparse.get_index_dtype(np.array([0, 1], dtype=np.int64), check_contents=True)),
    width(sparse.get_index_dtype(
        np.array([np.iinfo(np.int32).min - 1], dtype=np.int64),
        check_contents=True,
    )),
    width(sparse.get_index_dtype(np.array([], dtype=np.uint64), check_contents=True)),
    width(sparse.get_index_dtype(maxval=np.iinfo(np.int32).max)),
    width(sparse.get_index_dtype(maxval=np.iinfo(np.int32).max + 1)),
]

csr = sparse.csr_array((
    np.array([1.0, 2.0, 3.0]),
    np.array([0, 2, 1], dtype=np.int64),
    np.array([0, 2, 3], dtype=np.int64),
), shape=(2, 3))
indices, indptr = sparse.safely_cast_index_arrays(csr, np.int32)
result.append(float(
    indices.dtype == np.dtype(np.int32)
    and indptr.dtype == np.dtype(np.int32)
    and indices.tolist() == [0, 2, 1]
    and indptr.tolist() == [0, 2, 3]
))

limit = np.iinfo(np.int32).max
small_coords = sparse.coo_array((
    np.array([4.0]),
    (np.array([1], dtype=np.int64), np.array([0], dtype=np.int64)),
), shape=(int(limit) + 2, 2))
coords = sparse.safely_cast_index_arrays(small_coords, np.int32)
result.append(float(
    all(coord.dtype == np.dtype(np.int32) for coord in coords)
    and [coord.tolist() for coord in coords] == [[1], [0]]
))

large_coords = sparse.coo_array((
    np.array([4.0]),
    (np.array([int(limit) + 1], dtype=np.int64), np.array([0], dtype=np.int64)),
), shape=(int(limit) + 2, 1))
try:
    sparse.safely_cast_index_arrays(large_coords, np.int32)
except ValueError:
    result.append(1.0)
else:
    result.append(0.0)

dia = sparse.dia_array((
    np.array([[0.0, 2.0, 3.0], [4.0, 5.0, 0.0]]),
    np.array([-1, 1], dtype=np.int64),
), shape=(3, 3))
offsets = sparse.safely_cast_index_arrays(dia, np.int32)
result.append(float(offsets.dtype == np.dtype(np.int32) and offsets.tolist() == [-1, 1]))

try:
    sparse.safely_cast_index_arrays(sparse.dok_array((2, 2)), np.int32)
except TypeError:
    result.append(1.0)
else:
    result.append(0.0)

print(json.dumps(result))
"#,
    )
}

fn scipy_sparse_nd_array_contract() -> Option<Vec<f64>> {
    run_scipy_sparse_oracle(
        "N-dimensional sparse array shape operations",
        r#"
import json
import numpy as np
from scipy import sparse

data = np.array([10.0, 20.0, 30.0], dtype=np.float64)
coords = (
    np.array([0, 1, 1], dtype=np.int64),
    np.array([1, 0, 1], dtype=np.int64),
    np.array([2, 1, 0], dtype=np.int64),
)
base = sparse.coo_array((data, coords), shape=(2, 2, 3))
result = []

def append_array(array):
    result.append(float(isinstance(array, sparse.coo_array)))
    result.append(float(array.ndim))
    result.extend(float(extent) for extent in array.shape)
    result.append(float(array.nnz))
    result.extend(float(index) for axis in array.coords for index in axis.tolist())
    result.extend(float(value) for value in array.data.tolist())

append_array(sparse.expand_dims(base, axis=1))
append_array(sparse.expand_dims(base, axis=-4))
append_array(sparse.expand_dims(base, axis=-1))
append_array(sparse.permute_dims(base, axes=(2, 0, 1), copy=True))
append_array(sparse.permute_dims(base.copy(), axes=(-1, 0, 1), copy=False))
append_array(sparse.permute_dims(base, copy=True))
append_array(sparse.swapaxes(base, 0, -1))

def raises(call):
    try:
        call()
    except Exception:
        return 1.0
    return 0.0

result.extend([
    raises(lambda: sparse.expand_dims(base, axis=4)),
    raises(lambda: sparse.expand_dims(base, axis=-5)),
    raises(lambda: sparse.permute_dims(base, axes=(0, 1))),
    raises(lambda: sparse.permute_dims(base, axes=(0, 0, 2))),
    raises(lambda: sparse.permute_dims(base, axes=(0, 1, 3))),
    raises(lambda: sparse.swapaxes(base, 3, 0)),
    raises(lambda: sparse.swapaxes(base, -4, 0)),
])

legacy = sparse.coo_matrix(([1.0], ([0], [0])), shape=(1, 1))
result.extend([
    float(sparse.issparse(base)),
    float(sparse.isspmatrix(base)),
    float(isinstance(base, sparse.sparray)),
    float(sparse.issparse(legacy)),
    float(sparse.isspmatrix(legacy)),
    float(isinstance(legacy, sparse.sparray)),
])

print(json.dumps(result))
"#,
    )
}

fn scipy_sparse_npz_archives() -> Option<Vec<Vec<u8>>> {
    let encoded = run_scipy_sparse_oracle(
        "SciPy-written sparse NPZ archives",
        r#"
import io
import json
import numpy as np
from scipy import sparse

base = np.array([
    [1.0, 0.0, 2.0, 0.0],
    [0.0, 3.0, 0.0, 4.0],
    [5.0, 0.0, 6.0, 0.0],
    [0.0, 7.0, 0.0, 8.0],
], dtype=np.float64)
objects = [
    sparse.csr_matrix(base),
    sparse.csc_matrix(base),
    sparse.coo_matrix(base),
    sparse.bsr_matrix(base, blocksize=(2, 2)),
    sparse.dia_matrix(base),
]
coords = np.array([
    [0, 1, 1],
    [1, 0, 2],
    [0, 1, 1],
], dtype=np.int64)
objects.append(sparse.coo_array(
    (np.array([1.0, 2.0, 3.0], dtype=np.float64), coords),
    shape=(2, 3, 2),
))

result = []
for value in objects:
    stream = io.BytesIO()
    sparse.save_npz(stream, value, compressed=True)
    archive = stream.getvalue()
    result.append(float(len(archive)))
    result.extend(float(byte) for byte in archive)
print(json.dumps(result))
"#,
    )?;

    let mut archives = Vec::with_capacity(6);
    let mut position = 0;
    for _ in 0..6 {
        let raw_len = *encoded.get(position)?;
        if !raw_len.is_finite() || raw_len < 0.0 || raw_len.fract() != 0.0 {
            return None;
        }
        let len = raw_len as usize;
        position += 1;
        let end = position.checked_add(len)?;
        let encoded_archive = encoded.get(position..end)?;
        let mut archive = Vec::with_capacity(len);
        for &byte in encoded_archive {
            if !(0.0..=255.0).contains(&byte) || byte.fract() != 0.0 {
                return None;
            }
            archive.push(byte as u8);
        }
        archives.push(archive);
        position = end;
    }
    (position == encoded.len()).then_some(archives)
}

fn scipy_contract_for_npz_archives(archives: &[Vec<u8>]) -> Option<Vec<f64>> {
    let archives = serde_json::to_string(archives).expect("serialize in-memory NPZ archives");
    let script = format!(
        r#"
import io
import json
import numpy as np
from scipy import sparse

archives = {archives}
format_code = {{"csr": 1.0, "csc": 2.0, "coo": 3.0, "bsr": 4.0, "dia": 5.0}}
result = []
for archive in archives:
    value = sparse.load_npz(io.BytesIO(bytes(archive)))
    dense = np.asarray(value.toarray()).reshape(-1)
    result.extend([
        float(isinstance(value, sparse.sparray)),
        format_code[value.format],
        float(value.ndim),
    ])
    result.extend(float(extent) for extent in value.shape)
    result.append(float(dense.size))
    result.extend(float(element) for element in dense.tolist())
print(json.dumps(result))
"#
    );
    run_scipy_sparse_oracle("Rust-written sparse NPZ archives", &script)
}

fn sparse_npz_fixtures() -> (Vec<SparseMatrixOutput>, CooArray) {
    let coo = make_test_coo(
        4,
        4,
        &[
            (0, 0, 1.0),
            (0, 2, 2.0),
            (1, 1, 3.0),
            (1, 3, 4.0),
            (2, 0, 5.0),
            (2, 2, 6.0),
            (3, 1, 7.0),
            (3, 3, 8.0),
        ],
    );
    let matrices = vec![
        SparseMatrixOutput::Csr(coo.to_csr().expect("NPZ CSR fixture")),
        SparseMatrixOutput::Csc(coo.to_csc().expect("NPZ CSC fixture")),
        SparseMatrixOutput::Coo(coo.clone()),
        SparseMatrixOutput::Bsr(
            BsrMatrix::from_triplets(
                coo.shape(),
                Shape2D::new(2, 2),
                coo.data().to_vec(),
                coo.row_indices().to_vec(),
                coo.col_indices().to_vec(),
            )
            .expect("NPZ BSR fixture"),
        ),
        SparseMatrixOutput::Dia(
            DiaMatrix::from_triplets(
                coo.shape(),
                coo.data().to_vec(),
                coo.row_indices().to_vec(),
                coo.col_indices().to_vec(),
            )
            .expect("NPZ DIA fixture"),
        ),
    ];
    let array = CooArray::from_coords(
        vec![2, 3, 2],
        vec![1.0, 2.0, 3.0],
        vec![vec![0, 1, 1], vec![1, 0, 2], vec![0, 1, 1]],
        false,
    )
    .expect("NPZ N-D COO fixture");
    (matrices, array)
}

fn append_sparse_npz_contract(values: &mut Vec<f64>, sparse: &SparseNpz) {
    let exact_integer =
        |value: usize| f64::from(u32::try_from(value).expect("small NPZ integer must fit u32"));
    let format_code = match sparse.format() {
        SparseFormat::Csr => 1.0,
        SparseFormat::Csc => 2.0,
        SparseFormat::Coo => 3.0,
        SparseFormat::Bsr => 4.0,
        SparseFormat::Dia => 5.0,
        SparseFormat::Dok | SparseFormat::Lil => {
            unreachable!("unsupported formats cannot enter NPZ contract")
        }
    };
    let coo = sparse.to_coo_array().expect("NPZ object converts to COO");
    let dense = coo.to_dense().expect("NPZ object materializes densely");
    values.extend([
        f64::from(u8::from(!sparse.is_matrix())),
        format_code,
        exact_integer(coo.ndim()),
    ]);
    values.extend(coo.shape().iter().copied().map(exact_integer));
    values.push(exact_integer(dense.len()));
    values.extend(dense);
}

// ═══════════════════════════════════════════════════════════════
// DIFFERENTIAL ORACLE TESTS (>=15 cases)
// Compare sparse operations against dense reference computations.
// ═══════════════════════════════════════════════════════════════

#[test]
fn diff_001_spmv_csr_vs_dense_3x3() {
    let start = Instant::now();
    let coo = make_test_coo(
        3,
        3,
        &[
            (0, 0, 2.0),
            (0, 2, -1.0),
            (1, 1, 3.0),
            (2, 0, 1.5),
            (2, 2, 4.0),
        ],
    );
    let csr = coo.to_csr().expect("csr");
    let x = vec![1.0, -2.0, 3.0];
    let sparse_result = spmv_csr(&csr, &x).expect("spmv_csr");
    let dense_result = dense_matvec(&dense_from_coo(&coo), &x);
    let diff = max_abs_diff_vec(&sparse_result, &dense_result);
    let pass = diff <= TOL;
    emit_log(&DiffTestLog {
        test_id: "diff_001_spmv_csr_vs_dense_3x3".into(),
        category: "differential".into(),
        input_summary: "3x3 sparse, 5 nnz, x=[1,-2,3]".into(),
        expected: format!("{dense_result:?}"),
        actual: format!("{sparse_result:?}"),
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "spmv_csr diff={diff} > tol={TOL}");
}

#[test]
fn diff_002_spmv_csc_vs_dense_3x3() {
    let start = Instant::now();
    let coo = make_test_coo(
        3,
        3,
        &[
            (0, 0, 2.0),
            (0, 2, -1.0),
            (1, 1, 3.0),
            (2, 0, 1.5),
            (2, 2, 4.0),
        ],
    );
    let csc = coo.to_csc().expect("csc");
    let x = vec![1.0, -2.0, 3.0];
    let sparse_result = spmv_csc(&csc, &x).expect("spmv_csc");
    let dense_result = dense_matvec(&dense_from_coo(&coo), &x);
    let diff = max_abs_diff_vec(&sparse_result, &dense_result);
    let pass = diff <= TOL;
    emit_log(&DiffTestLog {
        test_id: "diff_002_spmv_csc_vs_dense_3x3".into(),
        category: "differential".into(),
        input_summary: "3x3 sparse via CSC, 5 nnz".into(),
        expected: format!("{dense_result:?}"),
        actual: format!("{sparse_result:?}"),
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "spmv_csc diff={diff}");
}

#[test]
fn diff_003_spmv_coo_vs_dense_3x3() {
    let start = Instant::now();
    let coo = make_test_coo(
        3,
        3,
        &[
            (0, 0, 2.0),
            (0, 2, -1.0),
            (1, 1, 3.0),
            (2, 0, 1.5),
            (2, 2, 4.0),
        ],
    );
    let x = vec![1.0, -2.0, 3.0];
    let sparse_result = spmv_coo(&coo, &x).expect("spmv_coo");
    let dense_result = dense_matvec(&dense_from_coo(&coo), &x);
    let diff = max_abs_diff_vec(&sparse_result, &dense_result);
    let pass = diff <= TOL;
    emit_log(&DiffTestLog {
        test_id: "diff_003_spmv_coo_vs_dense_3x3".into(),
        category: "differential".into(),
        input_summary: "3x3 sparse via COO, 5 nnz".into(),
        expected: format!("{dense_result:?}"),
        actual: format!("{sparse_result:?}"),
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "spmv_coo diff={diff}");
}

#[test]
fn diff_004_add_csr_vs_dense() {
    let start = Instant::now();
    let a_coo = make_test_coo(3, 3, &[(0, 0, 1.0), (1, 2, -2.0), (2, 1, 3.0)]);
    let b_coo = make_test_coo(3, 3, &[(0, 0, 4.0), (1, 1, 5.0), (2, 2, -1.0)]);
    let a_csr = a_coo.to_csr().expect("a csr");
    let b_csr = b_coo.to_csr().expect("b csr");
    let sum_csr = add_csr(&a_csr, &b_csr).expect("add");
    let sparse_dense = dense_from_coo(&sum_csr.to_coo().expect("coo"));
    let expected = dense_add(&dense_from_coo(&a_coo), &dense_from_coo(&b_coo));
    let diff = max_abs_diff_matrix(&sparse_dense, &expected);
    let pass = diff <= TOL;
    emit_log(&DiffTestLog {
        test_id: "diff_004_add_csr_vs_dense".into(),
        category: "differential".into(),
        input_summary: "3x3 A+B, 3 nnz each".into(),
        expected: format!("{expected:?}"),
        actual: format!("{sparse_dense:?}"),
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "add_csr diff={diff}");
}

#[test]
fn diff_005_sub_csr_vs_dense() {
    let start = Instant::now();
    let a_coo = make_test_coo(2, 4, &[(0, 0, 5.0), (0, 3, -2.0), (1, 1, 7.0)]);
    let b_coo = make_test_coo(2, 4, &[(0, 0, 1.0), (1, 1, 3.0), (1, 3, 4.0)]);
    let a_csr = a_coo.to_csr().expect("a");
    let b_csr = b_coo.to_csr().expect("b");
    let result = sub_csr(&a_csr, &b_csr).expect("sub");
    let sparse_dense = dense_from_coo(&result.to_coo().expect("coo"));
    let expected = dense_sub(&dense_from_coo(&a_coo), &dense_from_coo(&b_coo));
    let diff = max_abs_diff_matrix(&sparse_dense, &expected);
    let pass = diff <= TOL;
    emit_log(&DiffTestLog {
        test_id: "diff_005_sub_csr_vs_dense".into(),
        category: "differential".into(),
        input_summary: "2x4 A-B".into(),
        expected: format!("{expected:?}"),
        actual: format!("{sparse_dense:?}"),
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "sub_csr diff={diff}");
}

#[test]
fn diff_006_scale_csr_vs_dense() {
    let start = Instant::now();
    let coo = make_test_coo(3, 2, &[(0, 0, 2.0), (1, 1, -3.0), (2, 0, 1.5)]);
    let csr = coo.to_csr().expect("csr");
    let alpha = -2.5;
    let scaled = scale_csr(&csr, alpha).expect("scale");
    let sparse_dense = dense_from_coo(&scaled.to_coo().expect("coo"));
    let expected = dense_scale(&dense_from_coo(&coo), alpha);
    let diff = max_abs_diff_matrix(&sparse_dense, &expected);
    let pass = diff <= TOL;
    emit_log(&DiffTestLog {
        test_id: "diff_006_scale_csr_vs_dense".into(),
        category: "differential".into(),
        input_summary: format!("3x2 scale by {alpha}"),
        expected: format!("{expected:?}"),
        actual: format!("{sparse_dense:?}"),
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "scale_csr diff={diff}");
}

#[test]
fn diff_007_eye_identity_vs_dense() {
    let start = Instant::now();
    let id = eye(5).expect("eye");
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = spmv_csr(&id, &x).expect("spmv");
    let diff = max_abs_diff_vec(&result, &x);
    let pass = diff <= TOL;
    emit_log(&DiffTestLog {
        test_id: "diff_007_eye_identity_vs_dense".into(),
        category: "differential".into(),
        input_summary: "5x5 identity * x".into(),
        expected: format!("{x:?}"),
        actual: format!("{result:?}"),
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "eye spmv diff={diff}");
}

#[test]
fn diff_008_diags_tridiagonal_vs_dense() {
    let start = Instant::now();
    let csr = diags(
        &[
            vec![-1.0, -1.0, -1.0],
            vec![2.0, 2.0, 2.0, 2.0],
            vec![-1.0, -1.0, -1.0],
        ],
        &[-1, 0, 1],
        Some(Shape2D::new(4, 4)),
    )
    .expect("tridiag");
    let x = vec![1.0, 0.0, 0.0, 1.0];
    let result = spmv_csr(&csr, &x).expect("spmv");
    // Expected: tridiag * [1,0,0,1] = [2,-1,0,-1] wait let me compute:
    // Row 0: 2*1 + (-1)*0 = 2
    // Row 1: (-1)*1 + 2*0 + (-1)*0 = -1
    // Row 2: (-1)*0 + 2*0 + (-1)*1 = -1
    // Row 3: (-1)*0 + 2*1 = 2
    let expected = vec![2.0, -1.0, -1.0, 2.0];
    let diff = max_abs_diff_vec(&result, &expected);
    let pass = diff <= TOL;
    emit_log(&DiffTestLog {
        test_id: "diff_008_diags_tridiagonal_vs_dense".into(),
        category: "differential".into(),
        input_summary: "4x4 tridiag [-1,2,-1] * x".into(),
        expected: format!("{expected:?}"),
        actual: format!("{result:?}"),
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "tridiag spmv diff={diff}");
}

#[test]
fn diff_009_coo_to_csr_to_csc_roundtrip_vs_dense() {
    let start = Instant::now();
    let coo = make_test_coo(
        4,
        4,
        &[
            (0, 1, 1.0),
            (1, 0, -2.0),
            (1, 3, 3.5),
            (2, 2, 4.0),
            (3, 1, -1.0),
            (3, 3, 2.0),
        ],
    );
    let original_dense = dense_from_coo(&coo);
    let roundtrip = coo
        .to_csr()
        .expect("csr")
        .to_csc()
        .expect("csc")
        .to_coo()
        .expect("coo");
    let roundtrip_dense = dense_from_coo(&roundtrip);
    let diff = max_abs_diff_matrix(&original_dense, &roundtrip_dense);
    let pass = diff <= TOL;
    emit_log(&DiffTestLog {
        test_id: "diff_009_roundtrip_vs_dense".into(),
        category: "differential".into(),
        input_summary: "4x4 COO->CSR->CSC->COO roundtrip".into(),
        expected: format!("{original_dense:?}"),
        actual: format!("{roundtrip_dense:?}"),
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "roundtrip diff={diff}");
}

#[test]
fn diff_010_spmv_rectangular_4x6_vs_dense() {
    let start = Instant::now();
    let coo = make_test_coo(
        4,
        6,
        &[
            (0, 0, 1.0),
            (0, 5, -1.0),
            (1, 2, 2.0),
            (1, 3, 3.0),
            (2, 1, -4.0),
            (3, 4, 5.0),
            (3, 5, -2.0),
        ],
    );
    let csr = coo.to_csr().expect("csr");
    let x = vec![1.0, 2.0, 3.0, -1.0, 0.5, 4.0];
    let sparse_result = spmv_csr(&csr, &x).expect("spmv");
    let expected = dense_matvec(&dense_from_coo(&coo), &x);
    let diff = max_abs_diff_vec(&sparse_result, &expected);
    let pass = diff <= TOL;
    emit_log(&DiffTestLog {
        test_id: "diff_010_spmv_rectangular_4x6".into(),
        category: "differential".into(),
        input_summary: "4x6 rectangular spmv".into(),
        expected: format!("{expected:?}"),
        actual: format!("{sparse_result:?}"),
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "rect spmv diff={diff}");
}

#[test]
fn diff_011_spmv_duplicate_entries_sum() {
    let start = Instant::now();
    let coo = CooMatrix::from_triplets(
        Shape2D::new(2, 2),
        vec![1.0, 2.0, 3.0],
        vec![0, 0, 1],
        vec![0, 0, 1],
        true,
    )
    .expect("coo with dups");
    let csr = coo.to_csr().expect("csr");
    let x = vec![2.0, 3.0];
    let sparse_result = spmv_csr(&csr, &x).expect("spmv");
    // After sum_duplicates: (0,0)=3.0, (1,1)=3.0
    let expected = vec![6.0, 9.0];
    let diff = max_abs_diff_vec(&sparse_result, &expected);
    let pass = diff <= TOL;
    emit_log(&DiffTestLog {
        test_id: "diff_011_spmv_duplicate_entries_sum".into(),
        category: "differential".into(),
        input_summary: "2x2 with duplicate entries summed".into(),
        expected: format!("{expected:?}"),
        actual: format!("{sparse_result:?}"),
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "dup sum diff={diff}");
}

#[test]
fn diff_012_random_deterministic_spmv() {
    let start = Instant::now();
    let coo1 = random(Shape2D::new(8, 8), 0.3, 42).expect("random1");
    let coo2 = random(Shape2D::new(8, 8), 0.3, 42).expect("random2");
    let csr1 = coo1.to_csr().expect("csr1");
    let csr2 = coo2.to_csr().expect("csr2");
    let x = vec![1.0, -1.0, 0.5, 2.0, -0.5, 3.0, 0.0, -2.0];
    let r1 = spmv_csr(&csr1, &x).expect("spmv1");
    let r2 = spmv_csr(&csr2, &x).expect("spmv2");
    let diff = max_abs_diff_vec(&r1, &r2);
    let pass = diff <= TOL;
    emit_log(&DiffTestLog {
        test_id: "diff_012_random_deterministic_spmv".into(),
        category: "differential".into(),
        input_summary: "8x8 random(seed=42) deterministic pair".into(),
        expected: format!("{r1:?}"),
        actual: format!("{r2:?}"),
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "random deterministic diff={diff}");
}

#[test]
fn diff_013_csc_spmv_tall_matrix() {
    let start = Instant::now();
    let coo = make_test_coo(
        5,
        2,
        &[
            (0, 0, 1.0),
            (1, 1, 2.0),
            (2, 0, -1.0),
            (3, 1, 3.0),
            (4, 0, 0.5),
        ],
    );
    let csc = coo.to_csc().expect("csc");
    let x = vec![2.0, -1.0];
    let sparse_result = spmv_csc(&csc, &x).expect("spmv");
    let expected = dense_matvec(&dense_from_coo(&coo), &x);
    let diff = max_abs_diff_vec(&sparse_result, &expected);
    let pass = diff <= TOL;
    emit_log(&DiffTestLog {
        test_id: "diff_013_csc_spmv_tall_matrix".into(),
        category: "differential".into(),
        input_summary: "5x2 tall CSC spmv".into(),
        expected: format!("{expected:?}"),
        actual: format!("{sparse_result:?}"),
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "tall spmv diff={diff}");
}

#[test]
fn diff_014_mode_strict_conversion_preserves_values() {
    let start = Instant::now();
    let coo = make_test_coo(3, 3, &[(0, 0, 1.0), (1, 2, -2.0), (2, 1, 3.0)]);
    let original_dense = dense_from_coo(&coo);
    let (csr, _log) = coo_to_csr_with_mode(&coo, RuntimeMode::Strict, "diff-014").expect("csr");
    let (csc, _log2) = csr_to_csc_with_mode(&csr, RuntimeMode::Strict, "diff-014-2").expect("csc");
    let csc_coo: CooMatrix = csc.to_coo().expect("coo");
    let result_dense = dense_from_coo(&csc_coo);
    let diff = max_abs_diff_matrix(&original_dense, &result_dense);
    let pass = diff <= TOL;
    emit_log(&DiffTestLog {
        test_id: "diff_014_strict_conversion".into(),
        category: "differential".into(),
        input_summary: "3x3 strict COO->CSR->CSC roundtrip".into(),
        expected: format!("{original_dense:?}"),
        actual: format!("{result_dense:?}"),
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "strict conv diff={diff}");
}

#[test]
fn diff_015_scale_coo_and_csc_match_csr() {
    let start = Instant::now();
    let coo = make_test_coo(3, 3, &[(0, 1, 2.0), (1, 0, -3.0), (2, 2, 1.5)]);
    let alpha = 3.125;
    let scaled_coo_dense = dense_from_coo(&scale_coo(&coo, alpha).expect("coo scale"));
    let csr = coo.to_csr().expect("csr");
    let csc = coo.to_csc().expect("csc");
    let scaled_csr_dense = dense_from_coo(
        &scale_csr(&csr, alpha)
            .expect("csr scale")
            .to_coo()
            .expect("coo"),
    );
    let scaled_csc_dense = dense_from_coo(
        &scale_csc(&csc, alpha)
            .expect("csc scale")
            .to_coo()
            .expect("coo"),
    );
    let diff1 = max_abs_diff_matrix(&scaled_coo_dense, &scaled_csr_dense);
    let diff2 = max_abs_diff_matrix(&scaled_coo_dense, &scaled_csc_dense);
    let diff = diff1.max(diff2);
    let pass = diff <= TOL;
    emit_log(&DiffTestLog {
        test_id: "diff_015_scale_all_formats_match".into(),
        category: "differential".into(),
        input_summary: format!("3x3 scale({alpha}) COO vs CSR vs CSC"),
        expected: format!("{scaled_coo_dense:?}"),
        actual: format!("csr_diff={diff1}, csc_diff={diff2}"),
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "scale format diff={diff}");
}

#[test]
fn diff_016_spmv_wide_matrix_6x2() {
    let start = Instant::now();
    let coo = make_test_coo(2, 6, &[(0, 0, 1.0), (0, 5, -1.0), (1, 3, 2.0)]);
    let csr = coo.to_csr().expect("csr");
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let sparse_result = spmv_csr(&csr, &x).expect("spmv");
    let expected = dense_matvec(&dense_from_coo(&coo), &x);
    let diff = max_abs_diff_vec(&sparse_result, &expected);
    let pass = diff <= TOL;
    emit_log(&DiffTestLog {
        test_id: "diff_016_spmv_wide_matrix".into(),
        category: "differential".into(),
        input_summary: "2x6 wide spmv".into(),
        expected: format!("{expected:?}"),
        actual: format!("{sparse_result:?}"),
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "wide spmv diff={diff}");
}

#[test]
fn diff_017_spsolve_vs_scipy_superlu_4x4() {
    let start = Instant::now();
    let Some(scipy_result) = scipy_spsolve_tridiagonal_4x4() else {
        eprintln!("SciPy sparse spsolve oracle unavailable; skipping diff_017");
        return;
    };

    let coo = make_test_coo(
        4,
        4,
        &[
            (0, 0, 4.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 4.0),
            (1, 2, -1.0),
            (2, 1, -1.0),
            (2, 2, 4.0),
            (2, 3, -1.0),
            (3, 2, -1.0),
            (3, 3, 3.0),
        ],
    );
    let csr = coo.to_csr().expect("csr");
    let rhs = vec![15.0, 10.0, 10.0, 10.0];
    let rust_result = spsolve(&csr, &rhs, SolveOptions::default())
        .expect("rust spsolve")
        .solution;
    let diff = max_abs_diff_vec(&rust_result, &scipy_result);
    let tolerance = 1e-10;
    let pass = diff <= tolerance;
    emit_log(&DiffTestLog {
        test_id: "diff_017_spsolve_vs_scipy_superlu_4x4".into(),
        category: "scipy_differential".into(),
        input_summary: "4x4 SPD tridiagonal CSR solve vs scipy.sparse.linalg.spsolve".into(),
        expected: format!("scipy={scipy_result:?}"),
        actual: format!("rust={rust_result:?}"),
        diff,
        tolerance,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "spsolve SciPy oracle diff={diff} > tol={tolerance}");
}

#[test]
fn diff_018_large_spsolve_native_sparse_direct_vs_scipy() {
    let start = Instant::now();
    let Some(scipy_samples) = scipy_spsolve_large_diagonal_samples() else {
        eprintln!("SciPy sparse spsolve oracle unavailable; skipping diff_018");
        return;
    };

    let n = 32_769;
    let sample_indices = [0, 1, 17, n / 2, n - 1];
    let mut data = Vec::with_capacity(n);
    let mut rhs = Vec::with_capacity(n);
    for idx in 0..n {
        let diag = 2.0 + (idx % 7) as f64 * 0.25;
        let expected = 1.0 + (idx % 11) as f64 * 0.5;
        data.push(diag);
        rhs.push(diag * expected);
    }
    let csr = CsrMatrix::from_components(
        Shape2D::new(n, n),
        data,
        (0..n).collect(),
        (0..=n).collect(),
        false,
    )
    .expect("large diagonal CSR");
    let rust_result = spsolve(&csr, &rhs, SolveOptions::default())
        .expect("native sparse direct spsolve above dense fallback threshold")
        .solution;
    let rust_samples: Vec<f64> = sample_indices.iter().map(|&idx| rust_result[idx]).collect();
    let diff = max_abs_diff_vec(&rust_samples, &scipy_samples);
    let tolerance = 1e-10;
    let pass = diff <= tolerance;
    emit_log(&DiffTestLog {
        test_id: "diff_018_large_spsolve_native_sparse_direct_vs_scipy".into(),
        category: "scipy_differential".into(),
        input_summary:
            "32769x32769 diagonal CSR solve above dense fallback guard vs scipy.sparse.linalg.spsolve"
                .into(),
        expected: format!("scipy_samples={scipy_samples:?}"),
        actual: format!("rust_samples={rust_samples:?}"),
        diff,
        tolerance,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(
        pass,
        "large spsolve SciPy oracle diff={diff} > tol={tolerance}"
    );
}

#[test]
fn diff_019_sparse_index_array_utilities_vs_scipy() {
    let start = Instant::now();
    let Some(scipy_values) = scipy_sparse_index_array_contract() else {
        eprintln!("SciPy sparse index-array oracle unavailable; skipping diff_019");
        return;
    };

    let dtype_width = |dtype: IndexDtype| match dtype {
        IndexDtype::Int32 => 32.0,
        IndexDtype::Int64 => 64.0,
    };
    let int32_min = i64::from(i32::MIN);
    let int32_max = u64::from(i32::MAX.unsigned_abs());
    let mut rust_values = vec![
        dtype_width(get_index_dtype(&[], None, false).expect("empty dtype selection")),
        dtype_width(
            get_index_dtype(&[IndexArrayRef::I32(&[0, 1])], None, false)
                .expect("int32 dtype selection"),
        ),
        dtype_width(
            get_index_dtype(&[IndexArrayRef::I64(&[0, 1])], None, false)
                .expect("int64 declared-width selection"),
        ),
        dtype_width(
            get_index_dtype(&[IndexArrayRef::I64(&[0, 1])], None, true)
                .expect("int64 content selection"),
        ),
        dtype_width(
            get_index_dtype(&[IndexArrayRef::I64(&[int32_min - 1])], None, true)
                .expect("out-of-int32 content selection"),
        ),
        dtype_width(
            get_index_dtype(&[IndexArrayRef::U64(&[])], None, true)
                .expect("empty uint64 content selection"),
        ),
        dtype_width(
            get_index_dtype(&[], Some(int32_max), false).expect("int32 boundary max selection"),
        ),
        dtype_width(
            get_index_dtype(&[], Some(int32_max + 1), false).expect("int64 boundary max selection"),
        ),
    ];

    let csr = CsrMatrix::from_components(
        Shape2D::new(2, 3),
        vec![1.0, 2.0, 3.0],
        vec![0, 2, 1],
        vec![0, 2, 3],
        true,
    )
    .expect("valid CSR fixture");
    let expected_csr = SparseIndexArrays::Compressed {
        indices: IndexArray::Int32(vec![0, 2, 1]),
        indptr: IndexArray::Int32(vec![0, 2, 3]),
    };
    rust_values.push(f64::from(u8::from(
        safely_cast_index_arrays(&csr, IndexDtype::Int32, "").expect("CSR int32 cast")
            == expected_csr,
    )));

    let int32_max_usize = usize::try_from(i32::MAX).expect("i32 maximum fits usize");
    let small_coords = CooMatrix::from_triplets(
        Shape2D::new(int32_max_usize + 2, 2),
        vec![4.0],
        vec![1],
        vec![0],
        false,
    )
    .expect("large-shape COO with small coordinates");
    let expected_coords = SparseIndexArrays::Coordinates(vec![
        IndexArray::Int32(vec![1]),
        IndexArray::Int32(vec![0]),
    ]);
    rust_values.push(f64::from(u8::from(
        safely_cast_index_arrays(&small_coords, IndexDtype::Int32, "")
            .expect("small COO coordinates fit int32")
            == expected_coords,
    )));

    let large_coords = CooMatrix::from_triplets(
        Shape2D::new(int32_max_usize + 2, 1),
        vec![4.0],
        vec![int32_max_usize + 1],
        vec![0],
        false,
    )
    .expect("large COO coordinate fixture");
    rust_values.push(f64::from(u8::from(matches!(
        safely_cast_index_arrays(&large_coords, IndexDtype::Int32, ""),
        Err(SparseError::IndexOverflow { .. })
    ))));

    let dia = DiaMatrix::from_diagonals(
        Shape2D::new(3, 3),
        vec![-1, 1],
        vec![vec![2.0, 3.0], vec![4.0, 5.0]],
    )
    .expect("valid DIA fixture");
    rust_values.push(f64::from(u8::from(
        safely_cast_index_arrays(&dia, IndexDtype::Int32, "").expect("DIA int32 cast")
            == SparseIndexArrays::Offsets(IndexArray::Int32(vec![-1, 1])),
    )));

    let dok = DokMatrix::new(Shape2D::new(2, 2));
    rust_values.push(f64::from(u8::from(matches!(
        safely_cast_index_arrays(&dok, IndexDtype::Int32, ""),
        Err(SparseError::Unsupported { .. })
    ))));

    let diff = max_abs_diff_vec(&rust_values, &scipy_values);
    let pass = diff == 0.0;
    emit_log(&DiffTestLog {
        test_id: "diff_019_sparse_index_array_utilities_vs_scipy".into(),
        category: "scipy_differential".into(),
        input_summary:
            "get_index_dtype boundaries plus checked CSR/COO/DIA/DOK index-array casting".into(),
        expected: format!("scipy={scipy_values:?}"),
        actual: format!("rust={rust_values:?}"),
        diff,
        tolerance: 0.0,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "sparse index-array utility diff={diff}");
}

#[test]
fn diff_020_nd_sparse_array_shape_operations_vs_scipy() {
    let start = Instant::now();
    let Some(scipy_values) = scipy_sparse_nd_array_contract() else {
        eprintln!("SciPy N-dimensional sparse-array oracle unavailable; skipping diff_020");
        return;
    };

    let base = CooArray::from_coords(
        vec![2, 2, 3],
        vec![10.0, 20.0, 30.0],
        vec![vec![0, 1, 1], vec![1, 0, 1], vec![2, 1, 0]],
        false,
    )
    .expect("valid 3-D COO array");
    let mut rust_values = Vec::new();
    append_coo_array_contract(
        &mut rust_values,
        &expand_dims(&base, 1).expect("expand axis 1"),
    );
    append_coo_array_contract(
        &mut rust_values,
        &expand_dims(&base, -4).expect("expand leading axis"),
    );
    append_coo_array_contract(
        &mut rust_values,
        &expand_dims(&base, -1).expect("expand trailing axis"),
    );
    append_coo_array_contract(
        &mut rust_values,
        &permute_dims(&base, Some(&[2, 0, 1]), true).expect("explicit permutation"),
    );
    append_coo_array_contract(
        &mut rust_values,
        &permute_dims(&base, Some(&[-1, 0, 1]), false).expect("negative permutation axis"),
    );
    append_coo_array_contract(
        &mut rust_values,
        &permute_dims(&base, None, true).expect("default reverse permutation"),
    );
    append_coo_array_contract(
        &mut rust_values,
        &swapaxes(&base, 0, -1).expect("swap first and last axes"),
    );

    rust_values.extend([
        f64::from(u8::from(expand_dims(&base, 4).is_err())),
        f64::from(u8::from(expand_dims(&base, -5).is_err())),
        f64::from(u8::from(permute_dims(&base, Some(&[0, 1]), false).is_err())),
        f64::from(u8::from(
            permute_dims(&base, Some(&[0, 0, 2]), false).is_err(),
        )),
        f64::from(u8::from(
            permute_dims(&base, Some(&[0, 1, 3]), false).is_err(),
        )),
        f64::from(u8::from(swapaxes(&base, 3, 0).is_err())),
        f64::from(u8::from(swapaxes(&base, -4, 0).is_err())),
    ]);

    let legacy = CooMatrix::from_triplets(Shape2D::new(1, 1), vec![1.0], vec![0], vec![0], false)
        .expect("valid legacy COO matrix");
    rust_values.extend([
        f64::from(u8::from(issparse(&base))),
        f64::from(u8::from(isspmatrix(&base))),
        f64::from(u8::from(!base.is_matrix())),
        f64::from(u8::from(issparse(&legacy))),
        f64::from(u8::from(isspmatrix(&legacy))),
        f64::from(u8::from(!legacy.is_matrix())),
    ]);

    let diff = max_abs_diff_vec(&rust_values, &scipy_values);
    let pass = diff == 0.0;
    emit_log(&DiffTestLog {
        test_id: "diff_020_nd_sparse_array_shape_operations_vs_scipy".into(),
        category: "scipy_differential".into(),
        input_summary: "3-D COO expand_dims/permute_dims/swapaxes plus array/matrix predicates"
            .into(),
        expected: format!("scipy={scipy_values:?}"),
        actual: format!("rust={rust_values:?}"),
        diff,
        tolerance: 0.0,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "N-dimensional sparse-array contract diff={diff}");
}

#[test]
fn diff_021_sparse_npz_wire_compatibility_vs_scipy() {
    let start = Instant::now();
    let Some(scipy_archives) = scipy_sparse_npz_archives() else {
        eprintln!("SciPy sparse NPZ writer oracle unavailable; skipping diff_021");
        return;
    };
    let (matrices, array) = sparse_npz_fixtures();

    let mut expected = Vec::new();
    for matrix in &matrices {
        append_sparse_npz_contract(&mut expected, &SparseNpz::Matrix(matrix.clone()));
    }
    append_sparse_npz_contract(
        &mut expected,
        &SparseNpz::Array(SparseArrayOutput::Coo(array.clone())),
    );

    let mut rust_loaded_scipy = Vec::new();
    for archive in scipy_archives {
        let loaded = load_npz_from_reader(Cursor::new(archive))
            .expect("Rust must load SciPy-written sparse NPZ");
        append_sparse_npz_contract(&mut rust_loaded_scipy, &loaded);
    }

    let mut rust_archives = Vec::with_capacity(matrices.len() + 1);
    for matrix in &matrices {
        let mut archive = Cursor::new(Vec::new());
        save_npz_to_writer(&mut archive, matrix, true)
            .expect("Rust must write SciPy-compatible matrix NPZ");
        rust_archives.push(archive.into_inner());
    }
    let mut archive = Cursor::new(Vec::new());
    save_npz_to_writer(&mut archive, &array, true)
        .expect("Rust must write SciPy-compatible N-D COO NPZ");
    rust_archives.push(archive.into_inner());

    let Some(scipy_loaded_rust) = scipy_contract_for_npz_archives(&rust_archives) else {
        eprintln!("SciPy sparse NPZ reader oracle unavailable; skipping diff_021");
        return;
    };

    let scipy_to_rust_diff = max_abs_diff_vec(&rust_loaded_scipy, &expected);
    let rust_to_scipy_diff = max_abs_diff_vec(&scipy_loaded_rust, &expected);
    let diff = scipy_to_rust_diff.max(rust_to_scipy_diff);
    let pass = diff == 0.0;
    emit_log(&DiffTestLog {
        test_id: "diff_021_sparse_npz_wire_compatibility_vs_scipy".into(),
        category: "scipy_differential".into(),
        input_summary:
            "two-way compressed NPZ wire compatibility for CSR/CSC/COO/BSR/DIA and N-D COO array"
                .into(),
        expected: format!("contract={expected:?}"),
        actual: format!("scipy_to_rust={rust_loaded_scipy:?}; rust_to_scipy={scipy_loaded_rust:?}"),
        diff,
        tolerance: 0.0,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(
        pass,
        "sparse NPZ wire diff={diff}; scipy->rust={scipy_to_rust_diff}; rust->scipy={rust_to_scipy_diff}"
    );
}

// ═══════════════════════════════════════════════════════════════
// METAMORPHIC RELATION TESTS (>=6 cases)
// Verify mathematical invariants without a reference oracle.
// ═══════════════════════════════════════════════════════════════

#[test]
fn meta_001_scaling_invariance_spmv() {
    // spmv(scale(A, α), x) == α * spmv(A, x)
    let start = Instant::now();
    let coo = make_test_coo(3, 3, &[(0, 0, 2.0), (0, 2, -1.0), (1, 1, 3.0), (2, 0, 1.5)]);
    let csr = coo.to_csr().expect("csr");
    let alpha = -2.5;
    let x = vec![1.0, 2.0, -3.0];

    let lhs = spmv_csr(&scale_csr(&csr, alpha).expect("scale"), &x).expect("spmv(αA, x)");
    let rhs: Vec<f64> = spmv_csr(&csr, &x)
        .expect("spmv(A,x)")
        .iter()
        .map(|v| v * alpha)
        .collect();
    let diff = max_abs_diff_vec(&lhs, &rhs);
    let pass = diff <= TOL;
    emit_log(&DiffTestLog {
        test_id: "meta_001_scaling_invariance_spmv".into(),
        category: "metamorphic".into(),
        input_summary: "spmv(αA,x) == α·spmv(A,x)".into(),
        expected: format!("{rhs:?}"),
        actual: format!("{lhs:?}"),
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "scaling invariance diff={diff}");
}

#[test]
fn meta_002_addition_commutativity() {
    // A + B == B + A
    let start = Instant::now();
    let a = make_test_coo(3, 3, &[(0, 0, 1.0), (1, 2, -2.0)])
        .to_csr()
        .expect("a");
    let b = make_test_coo(3, 3, &[(0, 2, 3.0), (2, 1, 4.0)])
        .to_csr()
        .expect("b");
    let ab = dense_from_coo(&add_csr(&a, &b).expect("a+b").to_coo().expect("coo"));
    let ba = dense_from_coo(&add_csr(&b, &a).expect("b+a").to_coo().expect("coo"));
    let diff = max_abs_diff_matrix(&ab, &ba);
    let pass = diff <= TOL;
    emit_log(&DiffTestLog {
        test_id: "meta_002_addition_commutativity".into(),
        category: "metamorphic".into(),
        input_summary: "A+B == B+A".into(),
        expected: format!("{ab:?}"),
        actual: format!("{ba:?}"),
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "commutativity diff={diff}");
}

#[test]
fn meta_003_additive_identity() {
    // A + 0 == A
    let start = Instant::now();
    let a_coo = make_test_coo(3, 3, &[(0, 0, 5.0), (1, 2, -3.0), (2, 1, 7.0)]);
    let a = a_coo.to_csr().expect("a");
    let zero = CooMatrix::from_triplets(Shape2D::new(3, 3), vec![], vec![], vec![], false)
        .expect("zero")
        .to_csr()
        .expect("zero csr");
    let result = dense_from_coo(&add_csr(&a, &zero).expect("a+0").to_coo().expect("coo"));
    let expected = dense_from_coo(&a_coo);
    let diff = max_abs_diff_matrix(&result, &expected);
    let pass = diff <= TOL;
    emit_log(&DiffTestLog {
        test_id: "meta_003_additive_identity".into(),
        category: "metamorphic".into(),
        input_summary: "A + 0 == A".into(),
        expected: format!("{expected:?}"),
        actual: format!("{result:?}"),
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "additive identity diff={diff}");
}

#[test]
fn meta_004_additive_inverse() {
    // A + (-A) == 0
    let start = Instant::now();
    let a = make_test_coo(3, 3, &[(0, 0, 5.0), (1, 2, -3.0), (2, 1, 7.0)])
        .to_csr()
        .expect("a");
    let neg_a = scale_csr(&a, -1.0).expect("neg");
    let result_coo = add_csr(&a, &neg_a).expect("a+(-a)").to_coo().expect("coo");
    let result = dense_from_coo(&result_coo);
    let zeros = vec![vec![0.0; 3]; 3];
    let diff = max_abs_diff_matrix(&result, &zeros);
    let pass = diff <= TOL;
    emit_log(&DiffTestLog {
        test_id: "meta_004_additive_inverse".into(),
        category: "metamorphic".into(),
        input_summary: "A + (-A) == 0".into(),
        expected: format!("{zeros:?}"),
        actual: format!("{result:?}"),
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "additive inverse diff={diff}");
}

#[test]
fn meta_005_subtraction_self_is_zero() {
    // A - A == 0
    let start = Instant::now();
    let a = make_test_coo(
        4,
        4,
        &[
            (0, 0, 1.0),
            (1, 1, 2.0),
            (2, 2, 3.0),
            (3, 3, 4.0),
            (0, 3, -1.5),
            (2, 0, 0.5),
        ],
    )
    .to_csr()
    .expect("a");
    let result = dense_from_coo(&sub_csr(&a, &a).expect("a-a").to_coo().expect("coo"));
    let zeros = vec![vec![0.0; 4]; 4];
    let diff = max_abs_diff_matrix(&result, &zeros);
    let pass = diff <= TOL;
    emit_log(&DiffTestLog {
        test_id: "meta_005_subtraction_self_zero".into(),
        category: "metamorphic".into(),
        input_summary: "A - A == 0".into(),
        expected: format!("{zeros:?}"),
        actual: format!("{result:?}"),
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "self-subtraction diff={diff}");
}

#[test]
fn meta_006_spmv_linearity() {
    // spmv(A, αx + βy) == α·spmv(A,x) + β·spmv(A,y)
    let start = Instant::now();
    let coo = make_test_coo(3, 3, &[(0, 0, 2.0), (0, 2, -1.0), (1, 1, 3.0), (2, 0, 4.0)]);
    let csr = coo.to_csr().expect("csr");
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![-1.0, 0.5, 2.0];
    let alpha = 2.0;
    let beta = -1.5;

    let combined: Vec<f64> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| alpha * a + beta * b)
        .collect();
    let lhs = spmv_csr(&csr, &combined).expect("A(αx+βy)");
    let ax: Vec<f64> = spmv_csr(&csr, &x)
        .expect("Ax")
        .iter()
        .map(|v| v * alpha)
        .collect();
    let ay: Vec<f64> = spmv_csr(&csr, &y)
        .expect("Ay")
        .iter()
        .map(|v| v * beta)
        .collect();
    let rhs: Vec<f64> = ax.iter().zip(ay.iter()).map(|(a, b)| a + b).collect();
    let diff = max_abs_diff_vec(&lhs, &rhs);
    let pass = diff <= TOL;
    emit_log(&DiffTestLog {
        test_id: "meta_006_spmv_linearity".into(),
        category: "metamorphic".into(),
        input_summary: "A(αx+βy) == α·Ax + β·Ay".into(),
        expected: format!("{rhs:?}"),
        actual: format!("{lhs:?}"),
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "linearity diff={diff}");
}

// ═══════════════════════════════════════════════════════════════
// ADVERSARIAL / EDGE-CASE TESTS (>=8 cases)
// Hostile inputs, boundary conditions, expected error behavior.
// ═══════════════════════════════════════════════════════════════

#[test]
fn adv_001_zero_nnz_matrix_spmv() {
    let start = Instant::now();
    let empty =
        CooMatrix::from_triplets(Shape2D::new(4, 4), vec![], vec![], vec![], false).expect("empty");
    let csr = empty.to_csr().expect("csr");
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let result = spmv_csr(&csr, &x).expect("spmv");
    let expected = vec![0.0; 4];
    let diff = max_abs_diff_vec(&result, &expected);
    let pass = diff <= TOL;
    emit_log(&DiffTestLog {
        test_id: "adv_001_zero_nnz_spmv".into(),
        category: "adversarial".into(),
        input_summary: "4x4 zero-nnz matrix spmv".into(),
        expected: format!("{expected:?}"),
        actual: format!("{result:?}"),
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "zero-nnz diff={diff}");
}

#[test]
fn adv_002_hardened_mode_rejects_unsorted_csr() {
    let start = Instant::now();
    // Unsorted indices within a row
    let csr = CsrMatrix::from_components(
        Shape2D::new(2, 3),
        vec![1.0, 2.0, 3.0],
        vec![2, 0, 1],
        vec![0, 2, 3],
        false,
    )
    .expect("unsorted csr");
    let err = csr_to_csc_with_mode(&csr, RuntimeMode::Hardened, "adv-002");
    let pass = err.is_err();
    let error_desc = match &err {
        Err(e) => format!("{e}"),
        Ok(_) => "unexpected success".into(),
    };
    emit_log(&DiffTestLog {
        test_id: "adv_002_hardened_rejects_unsorted".into(),
        category: "adversarial".into(),
        input_summary: "unsorted CSR in hardened mode".into(),
        expected: "SparseError::InvalidSparseStructure".into(),
        actual: error_desc,
        diff: 0.0,
        tolerance: 0.0,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "hardened must reject unsorted CSR");
}

#[test]
fn adv_003_spsolve_non_square_is_rejected() {
    let start = Instant::now();
    let coo = make_test_coo(2, 3, &[(0, 0, 1.0), (1, 2, 2.0)]);
    let csr = coo.to_csr().expect("csr");
    let err = spsolve(&csr, &[1.0, 2.0], SolveOptions::default());
    let pass = matches!(err, Err(SparseError::InvalidShape { .. }));
    emit_log(&DiffTestLog {
        test_id: "adv_003_spsolve_nonsquare_rejected".into(),
        category: "adversarial".into(),
        input_summary: "spsolve with 2x3 non-square matrix".into(),
        expected: "SparseError::InvalidShape".into(),
        actual: format!("{err:?}"),
        diff: 0.0,
        tolerance: 0.0,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "non-square spsolve must be rejected");
}

#[test]
fn adv_004_nan_in_rhs_rejected_by_spsolve() {
    let start = Instant::now();
    let coo = make_test_coo(2, 2, &[(0, 0, 1.0), (1, 1, 2.0)]);
    let csr = coo.to_csr().expect("csr");
    let err = spsolve(&csr, &[f64::NAN, 1.0], SolveOptions::default());
    let pass = matches!(err, Err(SparseError::NonFiniteInput { .. }));
    emit_log(&DiffTestLog {
        test_id: "adv_004_nan_rhs_rejected".into(),
        category: "adversarial".into(),
        input_summary: "spsolve with NaN in rhs".into(),
        expected: "SparseError::NonFiniteInput".into(),
        actual: format!("{err:?}"),
        diff: 0.0,
        tolerance: 0.0,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "NaN must be rejected");
}

#[test]
fn adv_005_extreme_fill_dense_like_sparse() {
    // Dense-like sparse matrix (every position filled)
    let start = Instant::now();
    let n = 16;
    let coo = random(Shape2D::new(n, n), 1.0, 99).expect("dense random");
    let csr = coo.to_csr().expect("csr");
    let x: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();
    let sparse_result = spmv_csr(&csr, &x).expect("spmv");
    let expected = dense_matvec(&dense_from_coo(&coo), &x);
    let diff = max_abs_diff_vec(&sparse_result, &expected);
    let pass = diff <= TOL;
    emit_log(&DiffTestLog {
        test_id: "adv_005_extreme_fill_dense".into(),
        category: "adversarial".into(),
        input_summary: format!("{n}x{n} density=1.0 (full fill-in)"),
        expected: format!("len={}", expected.len()),
        actual: format!("max_diff={diff}"),
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "extreme fill diff={diff}");
}

#[test]
fn adv_006_coo_duplicate_handling_all_same_position() {
    // All entries at (0,0) with sum_duplicates
    let start = Instant::now();
    let coo = CooMatrix::from_triplets(
        Shape2D::new(2, 2),
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0],
        true,
    )
    .expect("all dups");
    let expected_val = 15.0; // 1+2+3+4+5
    let csr = coo.to_csr().expect("csr");
    let result = spmv_csr(&csr, &[1.0, 0.0]).expect("spmv");
    let diff = (result[0] - expected_val).abs();
    let pass = diff <= TOL;
    emit_log(&DiffTestLog {
        test_id: "adv_006_all_same_position_dups".into(),
        category: "adversarial".into(),
        input_summary: "5 entries all at (0,0), sum=15".into(),
        expected: format!("[{expected_val}, 0.0]"),
        actual: format!("{result:?}"),
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "all-same-pos diff={diff}");
}

#[test]
fn adv_007_hardened_rejects_empty_structural_row_spsolve() {
    let start = Instant::now();
    // Row 1 has no entries -> structurally singular
    let csr =
        CsrMatrix::from_components(Shape2D::new(2, 2), vec![1.0], vec![0], vec![0, 1, 1], true)
            .expect("csr with empty row");
    let options = SolveOptions {
        mode: RuntimeMode::Hardened,
        ..SolveOptions::default()
    };
    let err = spsolve(&csr, &[1.0, 0.0], options);
    let pass = matches!(err, Err(SparseError::SingularMatrix { .. }));
    emit_log(&DiffTestLog {
        test_id: "adv_007_hardened_empty_row_singular".into(),
        category: "adversarial".into(),
        input_summary: "spsolve with empty structural row in hardened mode".into(),
        expected: "SparseError::SingularMatrix".into(),
        actual: format!("{err:?}"),
        diff: 0.0,
        tolerance: 0.0,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "empty row must be rejected in hardened mode");
}

#[test]
fn adv_008_boundary_single_element_matrix_all_formats() {
    let start = Instant::now();
    let coo = make_test_coo(1, 1, &[(0, 0, 42.0)]);
    let csr = coo.to_csr().expect("csr");
    let csc = coo.to_csc().expect("csc");
    let x = vec![2.0];
    let r_coo = spmv_coo(&coo, &x).expect("coo spmv");
    let r_csr = spmv_csr(&csr, &x).expect("csr spmv");
    let r_csc = spmv_csc(&csc, &x).expect("csc spmv");
    let expected = vec![84.0];
    let diff = max_abs_diff_vec(&r_coo, &expected)
        .max(max_abs_diff_vec(&r_csr, &expected))
        .max(max_abs_diff_vec(&r_csc, &expected));
    let pass = diff <= TOL;
    emit_log(&DiffTestLog {
        test_id: "adv_008_single_element_all_formats".into(),
        category: "adversarial".into(),
        input_summary: "1x1 matrix [42] * [2] across all formats".into(),
        expected: format!("{expected:?}"),
        actual: format!("coo={r_coo:?} csr={r_csr:?} csc={r_csc:?}"),
        diff,
        tolerance: TOL,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "single-element diff={diff}");
}

#[test]
fn adv_009_index_out_of_bounds_rejected() {
    let start = Instant::now();
    let err = CooMatrix::from_triplets(Shape2D::new(2, 2), vec![1.0], vec![5], vec![0], false);
    let pass = matches!(err, Err(SparseError::IndexOutOfBounds { .. }));
    emit_log(&DiffTestLog {
        test_id: "adv_009_index_oob_rejected".into(),
        category: "adversarial".into(),
        input_summary: "COO with row index 5 in 2x2 matrix".into(),
        expected: "SparseError::IndexOutOfBounds".into(),
        actual: format!("{err:?}"),
        diff: 0.0,
        tolerance: 0.0,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "OOB index must be rejected");
}

#[test]
fn adv_010_vector_length_mismatch_rejected() {
    let start = Instant::now();
    let coo = make_test_coo(3, 3, &[(0, 0, 1.0)]);
    let csr = coo.to_csr().expect("csr");
    let err = spmv_csr(&csr, &[1.0, 2.0]); // length 2 != 3
    let pass = matches!(err, Err(SparseError::IncompatibleShape { .. }));
    emit_log(&DiffTestLog {
        test_id: "adv_010_vector_length_mismatch".into(),
        category: "adversarial".into(),
        input_summary: "spmv with 3x3 matrix and length-2 vector".into(),
        expected: "SparseError::IncompatibleShape".into(),
        actual: format!("{err:?}"),
        diff: 0.0,
        tolerance: 0.0,
        pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
    });
    assert!(pass, "length mismatch must be rejected");
}
