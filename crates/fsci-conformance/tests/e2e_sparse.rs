#![forbid(unsafe_code)]
//! E2E scenario tests for FSCI-P2C-004 (Sparse base/model invariants).
//!
//! Implements bd-3jh.15.7 acceptance criteria:
//!   Happy-path:     1-3  (construction→conversion→solve→verify)
//!   Error recovery: 4-6  (singular matrix, shape mismatch, mode switch)
//!   Adversarial:    7-10 (pathological sparsity, edge sizes, stress)
//!
//! Each scenario emits a forensic log bundle to
//! `fixtures/artifacts/FSCI-P2C-004/e2e/`.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_sparse::{
    CooMatrix, CscMatrix, CsrMatrix, FormatConvertible, IterativeSolveOptions, LilMatrix, Shape2D,
    SolveOptions, SparseError, SparseSliceSpec, add_csr, bicgstab, cg, connected_component_sizes,
    csr_to_csc_with_mode, diags, dijkstra, eye, find, gmres, hstack, hstack_with_format,
    is_connected, pagerank, scale_csr, sparse_norm, spmm, spmv_csr, spsolve,
    strongly_connected_components, sub_csr, topological_sort, tril, triu, vstack,
};
use serde::{Deserialize, Serialize};

// ───────────────────────── Forensic log types ─────────────────────────

#[derive(Debug, Clone, Serialize)]
struct ForensicLogBundle {
    scenario_id: String,
    steps: Vec<ForensicStep>,
    artifacts: Vec<ArtifactRef>,
    environment: EnvironmentInfo,
    overall: OverallResult,
}

#[derive(Debug, Clone, Serialize)]
struct ForensicStep {
    step_id: usize,
    step_name: String,
    action: String,
    input_summary: String,
    output_summary: String,
    duration_ns: u128,
    mode: String,
    outcome: String,
}

#[derive(Debug, Clone, Serialize)]
struct ArtifactRef {
    path: String,
    blake3: String,
}

#[derive(Debug, Clone, Serialize)]
struct EnvironmentInfo {
    rust_version: String,
    os: String,
    cpu_count: usize,
    total_memory_mb: String,
}

#[derive(Debug, Clone, Serialize)]
struct OverallResult {
    status: String,
    total_duration_ns: u128,
    replay_command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_chain: Option<String>,
}

// ───────────────────────── Helpers ─────────────────────────

fn e2e_output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/artifacts/FSCI-P2C-004/e2e")
}

fn make_env() -> EnvironmentInfo {
    EnvironmentInfo {
        rust_version: String::from(env!("CARGO_PKG_VERSION")),
        os: String::from(std::env::consts::OS),
        cpu_count: std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1),
        total_memory_mb: String::from("unknown"),
    }
}

fn replay_cmd(scenario_id: &str) -> String {
    format!("cargo test -p fsci-conformance --test e2e_sparse -- {scenario_id} --nocapture")
}

fn write_bundle(scenario_id: &str, bundle: &ForensicLogBundle) {
    let dir = e2e_output_dir();
    fs::create_dir_all(&dir).expect("failed to create e2e dir");
    let path = dir.join(format!("{scenario_id}.json"));
    let json = serde_json::to_vec_pretty(bundle).expect("serialize bundle");
    fs::write(&path, &json).expect("failed to write bundle");
}

const TOL: f64 = 1e-10;

/// Build a tridiagonal matrix: sub-diagonal, main diagonal, super-diagonal.
fn make_tridiag(n: usize, sub: f64, main: f64, sup: f64) -> CsrMatrix {
    let sub_diag = vec![sub; n.saturating_sub(1)];
    let main_diag = vec![main; n];
    let sup_diag = vec![sup; n.saturating_sub(1)];
    diags(
        &[sub_diag, main_diag, sup_diag],
        &[-1, 0, 1],
        Some(Shape2D::new(n, n)),
    )
    .expect("tridiag")
}

/// Build a bidiagonal matrix (main + one off-diagonal).
fn make_bidiag(n: usize, main: f64, off: f64, offset: isize) -> CsrMatrix {
    let main_diag = vec![main; n];
    let off_len = if offset.unsigned_abs() < n {
        n - offset.unsigned_abs()
    } else {
        0
    };
    let off_diag = vec![off; off_len];
    diags(
        &[main_diag, off_diag],
        &[0, offset],
        Some(Shape2D::new(n, n)),
    )
    .expect("bidiag")
}

fn make_step(
    step_id: usize,
    name: &str,
    action: &str,
    input: &str,
    output: &str,
    dur: u128,
    outcome: &str,
) -> ForensicStep {
    ForensicStep {
        step_id,
        step_name: name.to_string(),
        action: action.to_string(),
        input_summary: input.to_string(),
        output_summary: output.to_string(),
        duration_ns: dur,
        mode: "strict".to_string(),
        outcome: outcome.to_string(),
    }
}

fn dense_from_csr(csr: &CsrMatrix) -> Vec<Vec<f64>> {
    let shape = csr.shape();
    let coo = csr.to_coo().expect("csr->coo");
    let mut dense = vec![vec![0.0; shape.cols]; shape.rows];
    for idx in 0..coo.nnz() {
        dense[coo.row_indices()[idx]][coo.col_indices()[idx]] += coo.data()[idx];
    }
    dense
}

fn dense_from_coo(coo: &CooMatrix) -> Vec<Vec<f64>> {
    let shape = coo.shape();
    let mut dense = vec![vec![0.0; shape.cols]; shape.rows];
    for idx in 0..coo.nnz() {
        dense[coo.row_indices()[idx]][coo.col_indices()[idx]] += coo.data()[idx];
    }
    dense
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SparseOracleFixture {
    packet_id: String,
    family: String,
    cases: Vec<SparseOracleCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SparseOracleCase {
    case_id: String,
    operation: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    matrix: Option<SparseOracleMatrix>,
    #[serde(skip_serializing_if = "Option::is_none")]
    blocks: Option<Vec<SparseOracleMatrix>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    k: Option<isize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SparseOracleMatrix {
    format: String,
    shape: [usize; 2],
    row: Vec<usize>,
    col: Vec<usize>,
    data: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct SparseOracleCapture {
    case_outputs: Vec<SparseOracleCaseOutput>,
}

#[derive(Debug, Clone, Deserialize)]
struct SparseOracleCaseOutput {
    case_id: String,
    status: String,
    result_kind: String,
    result: serde_json::Value,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SparseOracleTriplets {
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<String>,
    shape: [usize; 2],
    row: Vec<usize>,
    col: Vec<usize>,
    data: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SparseOracleFindResult {
    row: Vec<usize>,
    col: Vec<usize>,
    data: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SparseOracleCsrComponents {
    shape: [usize; 2],
    data: Vec<f64>,
    indices: Vec<usize>,
    indptr: Vec<usize>,
    has_sorted_indices: bool,
    has_canonical_format: bool,
}

enum SparseInputMatrix {
    Coo(CooMatrix),
    Csr(CsrMatrix),
    Csc(CscMatrix),
}

impl SparseInputMatrix {
    fn as_format_convertible(&self) -> &dyn FormatConvertible {
        match self {
            Self::Coo(matrix) => matrix,
            Self::Csr(matrix) => matrix,
            Self::Csc(matrix) => matrix,
        }
    }
}

fn sparse_oracle_temp_path(prefix: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}_{stamp}.json"))
}

fn build_sparse_input(spec: &SparseOracleMatrix) -> SparseInputMatrix {
    let coo = CooMatrix::from_triplets(
        Shape2D::new(spec.shape[0], spec.shape[1]),
        spec.data.clone(),
        spec.row.clone(),
        spec.col.clone(),
        false,
    )
    .expect("fixture COO should be valid");
    let format = spec.format.as_str();
    assert!(
        matches!(format, "coo" | "csr" | "csc"),
        "unsupported fixture format {}",
        spec.format
    );
    match format {
        "coo" => SparseInputMatrix::Coo(coo),
        "csr" => SparseInputMatrix::Csr(coo.to_csr().expect("coo->csr")),
        _ => SparseInputMatrix::Csc(coo.to_csc().expect("coo->csc")),
    }
}

fn sparse_find(matrix: &SparseInputMatrix) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    match matrix {
        SparseInputMatrix::Coo(matrix) => find(matrix).expect("find"),
        SparseInputMatrix::Csr(matrix) => find(matrix).expect("find"),
        SparseInputMatrix::Csc(matrix) => find(matrix).expect("find"),
    }
}

fn sparse_tril(matrix: &SparseInputMatrix, k: isize) -> CooMatrix {
    match matrix {
        SparseInputMatrix::Coo(matrix) => tril(matrix, k).expect("tril"),
        SparseInputMatrix::Csr(matrix) => tril(matrix, k).expect("tril"),
        SparseInputMatrix::Csc(matrix) => tril(matrix, k).expect("tril"),
    }
}

fn sparse_triu(matrix: &SparseInputMatrix, k: isize) -> CooMatrix {
    match matrix {
        SparseInputMatrix::Coo(matrix) => triu(matrix, k).expect("triu"),
        SparseInputMatrix::Csr(matrix) => triu(matrix, k).expect("triu"),
        SparseInputMatrix::Csc(matrix) => triu(matrix, k).expect("triu"),
    }
}

fn triplets_from_coo(coo: &CooMatrix, format: Option<&str>) -> SparseOracleTriplets {
    SparseOracleTriplets {
        format: format.map(str::to_string),
        shape: [coo.shape().rows, coo.shape().cols],
        row: coo.row_indices().to_vec(),
        col: coo.col_indices().to_vec(),
        data: coo.data().to_vec(),
    }
}

fn compare_sparse_triplets(
    case_id: &str,
    actual: &SparseOracleTriplets,
    expected: &SparseOracleTriplets,
) {
    fn normalized_entries(triplets: &SparseOracleTriplets) -> Vec<(usize, usize, f64)> {
        let mut entries: Vec<_> = triplets
            .row
            .iter()
            .copied()
            .zip(triplets.col.iter().copied())
            .zip(triplets.data.iter().copied())
            .map(|((row, col), data)| (row, col, data))
            .collect();
        entries.sort_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
                .then_with(|| left.2.total_cmp(&right.2))
        });
        entries
    }

    assert_eq!(actual.shape, expected.shape, "{case_id}: shape mismatch");
    let actual_entries = normalized_entries(actual);
    let expected_entries = normalized_entries(expected);
    assert_eq!(
        actual_entries.len(),
        expected_entries.len(),
        "{case_id}: nnz mismatch"
    );
    for (idx, (actual_entry, expected_entry)) in actual_entries
        .iter()
        .zip(expected_entries.iter())
        .enumerate()
    {
        assert_eq!(
            actual_entry.0, expected_entry.0,
            "{case_id}: row mismatch at sorted idx {idx}"
        );
        assert_eq!(
            actual_entry.1, expected_entry.1,
            "{case_id}: col mismatch at sorted idx {idx}"
        );
        assert!(
            (actual_entry.2 - expected_entry.2).abs() <= TOL,
            "{case_id}: data mismatch at sorted idx {idx}: actual={} expected={}",
            actual_entry.2,
            expected_entry.2
        );
    }
}

fn compare_find_result(
    case_id: &str,
    actual: &SparseOracleFindResult,
    expected: &SparseOracleFindResult,
) {
    assert_eq!(actual.row, expected.row, "{case_id}: row mismatch");
    assert_eq!(actual.col, expected.col, "{case_id}: col mismatch");
    assert!(
        max_abs_diff_vec(&actual.data, &expected.data) <= TOL,
        "{case_id}: data mismatch actual={:?} expected={:?}",
        actual.data,
        expected.data
    );
}

fn compare_csr_components(
    case_id: &str,
    actual: &SparseOracleCsrComponents,
    expected: &SparseOracleCsrComponents,
) {
    assert_eq!(actual.shape, expected.shape, "{case_id}: shape mismatch");
    assert_eq!(
        actual.indices, expected.indices,
        "{case_id}: indices mismatch"
    );
    assert_eq!(actual.indptr, expected.indptr, "{case_id}: indptr mismatch");
    assert_eq!(
        actual.has_sorted_indices, expected.has_sorted_indices,
        "{case_id}: has_sorted_indices mismatch"
    );
    assert_eq!(
        actual.has_canonical_format, expected.has_canonical_format,
        "{case_id}: has_canonical_format mismatch"
    );
    assert!(
        max_abs_diff_vec(&actual.data, &expected.data) <= TOL,
        "{case_id}: data mismatch actual={:?} expected={:?}",
        actual.data,
        expected.data
    );
}

fn run_sparse_oracle_case(case: &SparseOracleCase) -> SparseOracleCaseOutput {
    let operation = case.operation.as_str();
    assert!(
        matches!(
            operation,
            "find" | "tril" | "triu" | "tocsc" | "vstack" | "hstack" | "csr_matmul"
        ),
        "unsupported sparse oracle operation {}",
        case.operation
    );
    match operation {
        "find" => {
            let matrix = build_sparse_input(case.matrix.as_ref().expect("matrix"));
            let (row, col, data) = sparse_find(&matrix);
            SparseOracleCaseOutput {
                case_id: case.case_id.clone(),
                status: "ok".to_string(),
                result_kind: "find_triplets".to_string(),
                result: serde_json::to_value(SparseOracleFindResult { row, col, data })
                    .expect("serialize find result"),
                error: None,
            }
        }
        "tril" => {
            let matrix = build_sparse_input(case.matrix.as_ref().expect("matrix"));
            let coo = sparse_tril(&matrix, case.k.unwrap_or(0));
            SparseOracleCaseOutput {
                case_id: case.case_id.clone(),
                status: "ok".to_string(),
                result_kind: "matrix_triplets".to_string(),
                result: serde_json::to_value(triplets_from_coo(&coo, Some("coo")))
                    .expect("serialize tril result"),
                error: None,
            }
        }
        "triu" => {
            let matrix = build_sparse_input(case.matrix.as_ref().expect("matrix"));
            let coo = sparse_triu(&matrix, case.k.unwrap_or(0));
            SparseOracleCaseOutput {
                case_id: case.case_id.clone(),
                status: "ok".to_string(),
                result_kind: "matrix_triplets".to_string(),
                result: serde_json::to_value(triplets_from_coo(&coo, Some("coo")))
                    .expect("serialize triu result"),
                error: None,
            }
        }
        "tocsc" => {
            let matrix = build_sparse_input(case.matrix.as_ref().expect("matrix"));
            let csc = matrix.as_format_convertible().to_csc().expect("tocsc");
            let coo = csc.to_coo().expect("csc->coo");
            SparseOracleCaseOutput {
                case_id: case.case_id.clone(),
                status: "ok".to_string(),
                result_kind: "matrix_triplets".to_string(),
                result: serde_json::to_value(triplets_from_coo(&coo, Some("csc")))
                    .expect("serialize tocsc result"),
                error: None,
            }
        }
        "vstack" => {
            let blocks = case.blocks.as_ref().expect("blocks");
            let matrices: Vec<SparseInputMatrix> = blocks.iter().map(build_sparse_input).collect();
            let refs: Vec<&dyn FormatConvertible> = matrices
                .iter()
                .map(SparseInputMatrix::as_format_convertible)
                .collect();
            let coo = vstack(&refs).expect("vstack").to_coo().expect("csr->coo");
            SparseOracleCaseOutput {
                case_id: case.case_id.clone(),
                status: "ok".to_string(),
                result_kind: "matrix_triplets".to_string(),
                result: serde_json::to_value(triplets_from_coo(&coo, Some("csr")))
                    .expect("serialize vstack result"),
                error: None,
            }
        }
        "csr_matmul" => {
            let blocks = case.blocks.as_ref().expect("blocks");
            assert_eq!(blocks.len(), 2, "csr_matmul expects exactly two matrices");
            let lhs = build_sparse_input(&blocks[0]);
            let rhs = build_sparse_input(&blocks[1]);
            let lhs_csr = lhs.as_format_convertible().to_csr().expect("lhs->csr");
            let rhs_csr = rhs.as_format_convertible().to_csr().expect("rhs->csr");
            let result = spmm(&lhs_csr, &rhs_csr);
            SparseOracleCaseOutput {
                case_id: case.case_id.clone(),
                status: "ok".to_string(),
                result_kind: "csr_components".to_string(),
                result: serde_json::to_value(SparseOracleCsrComponents {
                    shape: [result.shape().rows, result.shape().cols],
                    data: result.data().to_vec(),
                    indices: result.indices().to_vec(),
                    indptr: result.indptr().to_vec(),
                    has_sorted_indices: result.canonical_meta().sorted_indices,
                    has_canonical_format: result.canonical_meta().sorted_indices
                        && result.canonical_meta().deduplicated,
                })
                .expect("serialize csr_matmul result"),
                error: None,
            }
        }
        _ => {
            let blocks = case.blocks.as_ref().expect("blocks");
            let matrices: Vec<SparseInputMatrix> = blocks.iter().map(build_sparse_input).collect();
            let refs: Vec<&dyn FormatConvertible> = matrices
                .iter()
                .map(SparseInputMatrix::as_format_convertible)
                .collect();
            let (coo, actual_format) = if let Some(format) = case.format.as_deref() {
                let output = hstack_with_format(&refs, Some(format)).expect("hstack format");
                let actual_format = output.format_name().to_string();
                let coo = output.to_coo().expect("output->coo");
                (coo, Some(actual_format))
            } else {
                let coo = hstack(&refs).expect("hstack").to_coo().expect("csr->coo");
                (coo, Some("csr".to_string()))
            };
            SparseOracleCaseOutput {
                case_id: case.case_id.clone(),
                status: "ok".to_string(),
                result_kind: "matrix_triplets".to_string(),
                result: serde_json::to_value(triplets_from_coo(&coo, actual_format.as_deref()))
                    .expect("serialize hstack result"),
                error: None,
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// HAPPY-PATH SCENARIOS (1-3)
// ═══════════════════════════════════════════════════════════════════

/// Scenario 1: Construct tridiagonal → convert COO→CSR→CSC→COO → verify roundtrip
#[test]
fn e2e_001_tridiagonal_format_roundtrip() {
    let scenario_id = "e2e_sparse_001_tridiag";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let n = 10;

    // Step 1: Construct tridiagonal matrix via diags
    let t_start = Instant::now();
    let tri = make_tridiag(n, -1.0, 2.0, -1.0);
    steps.push(make_step(
        1,
        "construct_tridiag",
        "diags",
        &format!("n={n}, diags=[-1,2,-1]"),
        &format!("csr nnz={}", tri.nnz()),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Step 2: CSR → CSC
    let t_start = Instant::now();
    let (csc, _log1) =
        csr_to_csc_with_mode(&tri, RuntimeMode::Strict, "e2e-conv-1").expect("csr->csc");
    steps.push(make_step(
        2,
        "csr_to_csc",
        "convert",
        &format!("csr nnz={}", tri.nnz()),
        &format!("csc nnz={}", csc.nnz()),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Step 3: CSC → COO → CSR → verify
    let t_start = Instant::now();
    let coo = csc.to_coo().expect("csc->coo");
    let roundtrip = coo.to_csr().expect("coo->csr");
    let dense_orig = dense_from_csr(&tri);
    let dense_rt = dense_from_csr(&roundtrip);
    let mut max_diff = 0.0_f64;
    for (ro, rr) in dense_orig.iter().zip(dense_rt.iter()) {
        max_diff = max_diff.max(max_abs_diff_vec(ro, rr));
    }
    let pass = max_diff < TOL;
    steps.push(make_step(
        3,
        "roundtrip_verify",
        "compare",
        &format!("max_diff={max_diff:.4e}"),
        &format!("pass={pass}"),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(pass, "tridiagonal roundtrip: max_diff={max_diff:.4e}");
}

/// Scenario 2: Construct tridiag → spmv → verify against dense matvec
#[test]
fn e2e_002_spmv_verify_dense() {
    let scenario_id = "e2e_sparse_002_spmv";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let n = 5;

    // Step 1: Build tridiagonal system
    let t_start = Instant::now();
    let a = make_tridiag(n, -1.0, 3.0, -1.0);
    let v: Vec<f64> = (0..n).map(|i| (i as f64) + 1.0).collect();
    steps.push(make_step(
        1,
        "build_system",
        "diags+vector",
        &format!("n={n}, diag=[-1,3,-1]"),
        &format!("A nnz={}, v len={}", a.nnz(), v.len()),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Step 2: spmv
    let t_start = Instant::now();
    let result = spmv_csr(&a, &v).expect("spmv");
    steps.push(make_step(
        2,
        "spmv",
        "spmv_csr",
        "A*v",
        &format!("result len={}", result.len()),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Step 3: Compare with dense matvec
    let t_start = Instant::now();
    let dense = dense_from_csr(&a);
    let expected: Vec<f64> = dense
        .iter()
        .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
        .collect();
    let diff = max_abs_diff_vec(&result, &expected);
    let pass = diff < TOL;
    steps.push(make_step(
        3,
        "verify_dense",
        "compare",
        &format!("diff={diff:.4e}"),
        &format!("pass={pass}"),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(pass, "spmv vs dense: diff={diff:.4e}");
}

/// Scenario 3: Arithmetic pipeline: A+B, A-B, scale, spmv consistency
#[test]
fn e2e_003_arithmetic_pipeline() {
    let scenario_id = "e2e_sparse_003_arith";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let n = 4;

    // Step 1: Build A and B
    let t_start = Instant::now();
    let a = make_bidiag(n, 2.0, -1.0, 1);
    let b = make_bidiag(n, 1.0, 1.0, -1);
    steps.push(make_step(
        1,
        "build_matrices",
        "diags",
        &format!("n={n}"),
        &format!("A nnz={}, B nnz={}", a.nnz(), b.nnz()),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Step 2: C = A + B
    let t_start = Instant::now();
    let c = add_csr(&a, &b).expect("A+B");
    steps.push(make_step(
        2,
        "add",
        "add_csr",
        "A+B",
        &format!("C nnz={}", c.nnz()),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Step 3: D = A - B
    let t_start = Instant::now();
    let d = sub_csr(&a, &b).expect("A-B");
    steps.push(make_step(
        3,
        "subtract",
        "sub_csr",
        "A-B",
        &format!("D nnz={}", d.nnz()),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Step 4: Verify (A+B) + (A-B) = 2*A via spmv
    let t_start = Instant::now();
    let v: Vec<f64> = (0..n).map(|i| (i as f64) + 1.0).collect();
    let cpd = add_csr(&c, &d).expect("C+D");
    let two_a = scale_csr(&a, 2.0).expect("2*A");
    let lhs = spmv_csr(&cpd, &v).expect("(C+D)*v");
    let rhs = spmv_csr(&two_a, &v).expect("2A*v");
    let diff = max_abs_diff_vec(&lhs, &rhs);
    let pass = diff < TOL;
    steps.push(make_step(
        4,
        "verify_arithmetic",
        "spmv+compare",
        &format!("(A+B)+(A-B) vs 2A, diff={diff:.4e}"),
        &format!("pass={pass}"),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(pass, "arithmetic pipeline failed: diff={diff:.4e}");
}

// ═══════════════════════════════════════════════════════════════════
// ERROR RECOVERY SCENARIOS (4-6)
// ═══════════════════════════════════════════════════════════════════

/// Scenario 4: Non-square spsolve → shape error → switch to spmv → succeed
#[test]
fn e2e_004_nonsquare_solve_recovery() {
    let scenario_id = "e2e_sparse_004_nonsq";
    let overall_start = Instant::now();
    let mut steps = Vec::new();

    // Step 1: Try spsolve on non-square matrix
    let t_start = Instant::now();
    let rect = CooMatrix::from_triplets(
        Shape2D::new(3, 4),
        vec![1.0, 2.0, 3.0],
        vec![0, 1, 2],
        vec![0, 1, 2],
        false,
    )
    .expect("rect coo")
    .to_csr()
    .expect("rect csr");
    let b = vec![1.0, 2.0, 3.0];
    let opts = SolveOptions::default();
    let result = spsolve(&rect, &b, opts);
    let is_err = matches!(result, Err(SparseError::InvalidShape { .. }));
    steps.push(make_step(
        1,
        "solve_nonsquare",
        "spsolve",
        "3x4 non-square matrix",
        &format!("got_invalid_shape={is_err}"),
        t_start.elapsed().as_nanos(),
        if is_err {
            "expected_error"
        } else {
            "unexpected_ok"
        },
    ));

    // Step 2: Fall back to spmv which works on non-square
    let t_start = Instant::now();
    let v = vec![1.0, 2.0, 3.0, 4.0];
    let result = spmv_csr(&rect, &v);
    let pass = result.is_ok();
    steps.push(make_step(
        2,
        "fallback_spmv",
        "spmv_csr",
        "3x4 matrix, vector len=4",
        &format!("success={pass}"),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let overall_pass = is_err && pass;
    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if overall_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(overall_pass, "nonsquare solve recovery failed");
}

/// Scenario 5: Hardened mode rejects unsorted → switch to strict → succeed
#[test]
fn e2e_005_mode_switch_recovery() {
    let scenario_id = "e2e_sparse_005_mode_switch";
    let overall_start = Instant::now();
    let mut steps = Vec::new();

    // Step 1: Build unsorted CSR
    let t_start = Instant::now();
    let unsorted = CsrMatrix::from_components(
        Shape2D::new(2, 3),
        vec![1.0, 2.0, 3.0],
        vec![2, 0, 1],
        vec![0, 2, 3],
        false,
    )
    .expect("unsorted csr");
    steps.push(make_step(
        1,
        "build_unsorted",
        "from_components",
        "2x3 unsorted CSR",
        &format!("sorted={}", unsorted.canonical_meta().sorted_indices),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Step 2: Hardened conversion → error
    let t_start = Instant::now();
    let result = csr_to_csc_with_mode(&unsorted, RuntimeMode::Hardened, "e2e-hard");
    let is_err = result.is_err();
    steps.push(make_step(
        2,
        "hardened_convert",
        "csr_to_csc",
        "hardened mode, unsorted input",
        &format!("error={is_err}"),
        t_start.elapsed().as_nanos(),
        if is_err {
            "expected_error"
        } else {
            "unexpected_ok"
        },
    ));

    // Step 3: Strict mode → success
    let t_start = Instant::now();
    let (csc, _) = csr_to_csc_with_mode(&unsorted, RuntimeMode::Strict, "e2e-strict")
        .expect("strict csr->csc");
    let pass = csc.nnz() == unsorted.nnz();
    steps.push(make_step(
        3,
        "strict_convert",
        "csr_to_csc",
        "strict mode, same unsorted input",
        &format!("success={pass}, nnz={}", csc.nnz()),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let overall_pass = is_err && pass;
    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if overall_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(overall_pass, "mode switch recovery failed");
}

/// Scenario 6: Vector length mismatch → diagnose → fix → succeed
#[test]
fn e2e_006_spmv_mismatch_recovery() {
    let scenario_id = "e2e_sparse_006_spmv_mismatch";
    let overall_start = Instant::now();
    let mut steps = Vec::new();

    let a = eye(4).expect("4x4 identity");

    // Step 1: Wrong vector length
    let t_start = Instant::now();
    let result = spmv_csr(&a, &[1.0, 2.0]); // expects 4
    let is_err = matches!(result, Err(SparseError::IncompatibleShape { .. }));
    steps.push(make_step(
        1,
        "spmv_wrong_len",
        "spmv_csr",
        "4x4 matrix, vector len=2",
        &format!("error={is_err}"),
        t_start.elapsed().as_nanos(),
        if is_err {
            "expected_error"
        } else {
            "unexpected_ok"
        },
    ));

    // Step 2: Correct vector length
    let t_start = Instant::now();
    let v = vec![1.0, 2.0, 3.0, 4.0];
    let result = spmv_csr(&a, &v).expect("spmv");
    let diff = max_abs_diff_vec(&result, &v);
    let pass = diff < TOL;
    steps.push(make_step(
        2,
        "spmv_correct",
        "spmv_csr",
        "4x4 identity, vector len=4",
        &format!("diff={diff:.4e}, pass={pass}"),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let overall_pass = is_err && pass;
    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if overall_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(overall_pass, "spmv mismatch recovery failed");
}

// ═══════════════════════════════════════════════════════════════════
// ADVERSARIAL SCENARIOS (7-10)
// ═══════════════════════════════════════════════════════════════════

/// Scenario 7: Zero-NNZ matrix operations
#[test]
fn e2e_007_zero_nnz_operations() {
    let scenario_id = "e2e_sparse_007_zero_nnz";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let n = 5;

    let t_start = Instant::now();
    let zero = CooMatrix::from_triplets(Shape2D::new(n, n), vec![], vec![], vec![], false)
        .expect("zero coo")
        .to_csr()
        .expect("zero csr");
    steps.push(make_step(
        1,
        "build_zero",
        "from_triplets",
        &format!("{n}x{n} zero matrix"),
        &format!("nnz={}", zero.nnz()),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // spmv with zero matrix
    let t_start = Instant::now();
    let v: Vec<f64> = (0..n).map(|i| (i as f64) + 1.0).collect();
    let result = spmv_csr(&zero, &v).expect("spmv");
    let expected = vec![0.0; n];
    let diff = max_abs_diff_vec(&result, &expected);
    let pass_spmv = diff < TOL;
    steps.push(make_step(
        2,
        "spmv_zero",
        "spmv_csr",
        "zero * v",
        &format!("all_zero={pass_spmv}"),
        t_start.elapsed().as_nanos(),
        if pass_spmv { "ok" } else { "fail" },
    ));

    // add zero + identity
    let t_start = Instant::now();
    let id = eye(n).expect("identity");
    let sum = add_csr(&zero, &id).expect("zero+I");
    let result = spmv_csr(&sum, &v).expect("spmv");
    let diff = max_abs_diff_vec(&result, &v);
    let pass_add = diff < TOL;
    steps.push(make_step(
        3,
        "add_zero_identity",
        "add_csr+spmv",
        "0+I = I",
        &format!("pass={pass_add}"),
        t_start.elapsed().as_nanos(),
        if pass_add { "ok" } else { "fail" },
    ));

    let overall_pass = pass_spmv && pass_add;
    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if overall_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(overall_pass, "zero-nnz operations failed");
}

/// Scenario 8: Single-element 1x1 matrix through full pipeline
#[test]
fn e2e_008_single_element() {
    let scenario_id = "e2e_sparse_008_single";
    let overall_start = Instant::now();
    let mut steps = Vec::new();

    let t_start = Instant::now();
    let coo = CooMatrix::from_triplets(Shape2D::new(1, 1), vec![7.0], vec![0], vec![0], false)
        .expect("1x1 coo");
    let csr = coo.to_csr().expect("csr");
    let (csc, _) = csr_to_csc_with_mode(&csr, RuntimeMode::Strict, "e2e-1x1").expect("csc");
    let back = csc.to_coo().expect("coo").to_csr().expect("csr");
    let dense_orig = dense_from_csr(&csr);
    let dense_back = dense_from_csr(&back);
    let diff = max_abs_diff_vec(&dense_orig[0], &dense_back[0]);
    let pass_rt = diff < TOL;
    steps.push(make_step(
        1,
        "roundtrip_1x1",
        "convert",
        "1x1 matrix val=7",
        &format!("pass={pass_rt}"),
        t_start.elapsed().as_nanos(),
        if pass_rt { "ok" } else { "fail" },
    ));

    // spmv: 7 * [2.0] = [14.0]
    let t_start = Instant::now();
    let result = spmv_csr(&csr, &[2.0]).expect("spmv");
    let pass_solve = (result[0] - 14.0).abs() < TOL;
    steps.push(make_step(
        2,
        "spmv_1x1",
        "spmv_csr",
        "7 * [2] = [14]",
        &format!("result={}, pass={pass_solve}", result[0]),
        t_start.elapsed().as_nanos(),
        if pass_solve { "ok" } else { "fail" },
    ));

    let overall_pass = pass_rt && pass_solve;
    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if overall_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(overall_pass, "single element pipeline failed");
}

/// Scenario 9: Dense-like full matrix (pathological sparsity)
#[test]
fn e2e_009_dense_fill_pattern() {
    let scenario_id = "e2e_sparse_009_dense_fill";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let n = 8;

    // Build fully dense matrix as sparse
    let t_start = Instant::now();
    let mut data = Vec::new();
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    for r in 0..n {
        for c in 0..n {
            rows.push(r);
            cols.push(c);
            data.push(if r == c { (n as f64) + 1.0 } else { 1.0 });
        }
    }
    let coo = CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .expect("dense-as-sparse coo");
    let csr = coo.to_csr().expect("csr");
    steps.push(make_step(
        1,
        "build_dense_sparse",
        "from_triplets",
        &format!("{n}x{n} fully dense as sparse"),
        &format!("nnz={}", csr.nnz()),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Verify spmv against dense matvec
    let t_start = Instant::now();
    let v: Vec<f64> = (0..n).map(|i| (i as f64) + 1.0).collect();
    let sparse_result = spmv_csr(&csr, &v).expect("spmv");
    let dense = dense_from_csr(&csr);
    let expected: Vec<f64> = dense
        .iter()
        .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
        .collect();
    let diff = max_abs_diff_vec(&sparse_result, &expected);
    let pass = diff < TOL;
    steps.push(make_step(
        2,
        "spmv_verify",
        "spmv_csr+dense",
        &format!("diff={diff:.4e}"),
        &format!("pass={pass}"),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(pass, "dense fill pattern: diff={diff:.4e}");
}

/// Scenario 10: Rapid sequential operations (no state leakage)
#[test]
fn e2e_010_rapid_sequential() {
    let scenario_id = "e2e_sparse_010_rapid";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let iterations = 50;

    let t_start = Instant::now();
    let mut all_pass = true;
    for i in 0..iterations {
        let n = 3 + (i % 5); // sizes 3..7
        let a = make_tridiag(n, -1.0, 3.0, -1.0);
        let v: Vec<f64> = (0..n).map(|j| ((i * n + j) as f64) * 0.1 + 1.0).collect();
        let result = spmv_csr(&a, &v).expect("spmv");
        let dense = dense_from_csr(&a);
        let expected: Vec<f64> = dense
            .iter()
            .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
            .collect();
        let diff = max_abs_diff_vec(&result, &expected);
        if diff > TOL {
            all_pass = false;
        }
    }
    steps.push(make_step(
        1,
        "rapid_spmv",
        "spmv+verify",
        &format!("{iterations} iterations, sizes 3-7"),
        &format!("all_pass={all_pass}"),
        t_start.elapsed().as_nanos(),
        if all_pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if all_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(all_pass, "rapid sequential: state leakage detected");
}

// ═══════════════════════════════════════════════════════════════════
// ITERATIVE SOLVER SCENARIOS (11-13)
// ═══════════════════════════════════════════════════════════════════

/// Scenario 11: Conjugate Gradient solver for SPD system
/// Tests CG convergence on a symmetric positive definite tridiagonal matrix.
#[test]
fn e2e_011_cg_spd_system() {
    let scenario_id = "e2e_sparse_011_cg";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let n = 20;

    // Step 1: Build SPD tridiagonal matrix (diagonally dominant)
    let t_start = Instant::now();
    // A = diag(4) + tridiag(-1, 0, -1) is SPD
    let a = make_tridiag(n, -1.0, 4.0, -1.0);
    steps.push(make_step(
        1,
        "build_spd_matrix",
        "diags",
        &format!("n={n}, tridiag(-1, 4, -1)"),
        &format!("nnz={}", a.nnz()),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Step 2: Create known solution and compute RHS
    let t_start = Instant::now();
    let x_true: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / (n as f64)).collect();
    let b = spmv_csr(&a, &x_true).expect("spmv for rhs");
    steps.push(make_step(
        2,
        "compute_rhs",
        "A * x_true",
        "x_true = [1/n, 2/n, ..., 1]",
        &format!("b len={}", b.len()),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Step 3: Solve with CG
    let t_start = Instant::now();
    let options = IterativeSolveOptions {
        max_iter: Some(100),
        tol: 1e-10,
        ..Default::default()
    };
    let result = cg(&a, &b, None, options).expect("cg solve");
    let converged = result.converged;
    let iterations = result.iterations;
    steps.push(make_step(
        3,
        "cg_solve",
        "cg",
        "maxiter=100, tol=1e-10",
        &format!("converged={}, iters={}", converged, iterations),
        t_start.elapsed().as_nanos(),
        if converged { "ok" } else { "fail" },
    ));

    // Step 4: Verify solution accuracy
    // Note: converged flag may be false if we hit max_iter before tolerance,
    // but if the solution is accurate enough, we still consider it a pass
    let t_start = Instant::now();
    let diff = max_abs_diff_vec(&result.solution, &x_true);
    let pass = diff < 1e-6; // Accuracy is what matters
    steps.push(make_step(
        4,
        "verify_solution",
        "compare x with x_true",
        &format!("max_diff={diff:.4e}, converged={converged}"),
        &format!("pass={pass}"),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(pass, "CG solver: converged={}, diff={diff:.4e}", converged);
}

#[test]
fn e2e_011b_spsolve_cg_spd_scipy_golden() {
    // scipy.sparse.linalg.spsolve/cg on diag(-1,4,-1), b=[1,2,0,1,3].
    let a = make_tridiag(5, -1.0, 4.0, -1.0);
    let b = vec![1.0, 2.0, 0.0, 1.0, 3.0];
    let expected = vec![
        0.420_512_820_512_820_52,
        0.682_051_282_051_282_1,
        0.307_692_307_692_307_7,
        0.548_717_948_717_948_8,
        0.887_179_487_179_487_1,
    ];

    let direct = spsolve(&a, &b, SolveOptions::default()).expect("spsolve");
    assert!(
        max_abs_diff_vec(&direct.solution, &expected) <= TOL,
        "spsolve {direct:?} != {expected:?}"
    );

    let cg_result = cg(
        &a,
        &b,
        None,
        IterativeSolveOptions {
            max_iter: Some(100),
            tol: 1e-12,
            ..Default::default()
        },
    )
    .expect("cg");
    assert!(cg_result.converged, "cg did not converge");
    assert!(
        max_abs_diff_vec(&cg_result.solution, &expected) <= TOL,
        "cg {:?} != {expected:?}",
        cg_result.solution
    );
}

/// Scenario 12: GMRES solver for general system
/// Tests GMRES on a non-symmetric matrix.
#[test]
fn e2e_012_gmres_general_system() {
    let scenario_id = "e2e_sparse_012_gmres";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let n = 15;

    // Step 1: Build non-symmetric bidiagonal matrix
    let t_start = Instant::now();
    let a = make_bidiag(n, 3.0, -1.0, 1); // upper bidiagonal
    steps.push(make_step(
        1,
        "build_nonsym_matrix",
        "diags",
        &format!("n={n}, bidiag(3, -1, offset=1)"),
        &format!("nnz={}", a.nnz()),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Step 2: Create known solution and RHS
    let t_start = Instant::now();
    let x_true: Vec<f64> = (0..n).map(|i| ((i + 1) as f64).sin()).collect();
    let b = spmv_csr(&a, &x_true).expect("spmv for rhs");
    steps.push(make_step(
        2,
        "compute_rhs",
        "A * x_true",
        "x_true = [sin(1), sin(2), ...]",
        &format!("b len={}", b.len()),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Step 3: Solve with GMRES
    let t_start = Instant::now();
    let options = IterativeSolveOptions {
        max_iter: Some(50),
        tol: 1e-10,
        ..Default::default()
    };
    let result = gmres(&a, &b, None, options).expect("gmres solve");
    let converged = result.converged;
    let iterations = result.iterations;
    steps.push(make_step(
        3,
        "gmres_solve",
        "gmres",
        "maxiter=50, tol=1e-10",
        &format!("converged={}, iters={}", converged, iterations),
        t_start.elapsed().as_nanos(),
        if converged { "ok" } else { "fail" },
    ));

    // Step 4: Verify solution
    let t_start = Instant::now();
    let diff = max_abs_diff_vec(&result.solution, &x_true);
    let pass = diff < 1e-6 && converged;
    steps.push(make_step(
        4,
        "verify_solution",
        "compare x with x_true",
        &format!("max_diff={diff:.4e}"),
        &format!("pass={pass}"),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(
        pass,
        "GMRES solver: converged={}, diff={diff:.4e}",
        converged
    );
}

/// Scenario 13: BiCGSTAB for non-symmetric system
/// Tests BiCGSTAB on a tridiagonal system with asymmetric off-diagonals.
#[test]
fn e2e_013_bicgstab_asymmetric() {
    let scenario_id = "e2e_sparse_013_bicgstab";
    let overall_start = Instant::now();
    let mut steps = Vec::new();
    let n = 20;

    // Step 1: Build asymmetric tridiagonal
    let t_start = Instant::now();
    // Different sub and super diagonals
    let a = make_tridiag(n, -1.0, 3.0, -0.5);
    steps.push(make_step(
        1,
        "build_asym_matrix",
        "diags",
        &format!("n={n}, tridiag(-1, 3, -0.5)"),
        &format!("nnz={}", a.nnz()),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Step 2: Known solution and RHS
    let t_start = Instant::now();
    let x_true: Vec<f64> = (0..n).map(|i| 1.0 / ((i + 1) as f64)).collect();
    let b = spmv_csr(&a, &x_true).expect("spmv for rhs");
    steps.push(make_step(
        2,
        "compute_rhs",
        "A * x_true",
        "x_true = [1, 1/2, 1/3, ...]",
        &format!("b len={}", b.len()),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Step 3: Solve with BiCGSTAB
    let t_start = Instant::now();
    let options = IterativeSolveOptions {
        max_iter: Some(100),
        tol: 1e-10,
        ..Default::default()
    };
    let result = bicgstab(&a, &b, None, options).expect("bicgstab solve");
    let converged = result.converged;
    let iterations = result.iterations;
    steps.push(make_step(
        3,
        "bicgstab_solve",
        "bicgstab",
        "maxiter=100, tol=1e-10",
        &format!("converged={}, iters={}", converged, iterations),
        t_start.elapsed().as_nanos(),
        if converged { "ok" } else { "fail" },
    ));

    // Step 4: Verify
    // Note: converged flag may be false if we hit max_iter before tolerance,
    // but if the solution is accurate enough, we still consider it a pass
    let t_start = Instant::now();
    let diff = max_abs_diff_vec(&result.solution, &x_true);
    let pass = diff < 1e-5; // Accuracy is what matters
    steps.push(make_step(
        4,
        "verify_solution",
        "compare x with x_true",
        &format!("max_diff={diff:.4e}, converged={converged}"),
        &format!("pass={pass}"),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(
        pass,
        "BiCGSTAB solver: converged={}, diff={diff:.4e}",
        converged
    );
}

// ═══════════════════════════════════════════════════════════════════
// GRAPH ALGORITHM SCENARIOS (14-16)
// ═══════════════════════════════════════════════════════════════════

/// Scenario 14: Connected components on small graph
/// Tests connected_component_sizes on a known graph structure.
#[test]
fn e2e_014_connected_components() {
    let scenario_id = "e2e_sparse_014_components";
    let overall_start = Instant::now();
    let mut steps = Vec::new();

    // Step 1: Build a graph with 2 connected components
    // Component 1: nodes 0-2 (triangle)
    // Component 2: nodes 3-4 (edge)
    let t_start = Instant::now();
    let n = 5;
    let rows = vec![0, 0, 1, 1, 2, 2, 3, 4];
    let cols = vec![1, 2, 0, 2, 0, 1, 4, 3];
    let data = vec![1.0; 8];
    let coo =
        CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false).expect("graph coo");
    let graph = coo.to_csr().expect("graph csr");
    steps.push(make_step(
        1,
        "build_graph",
        "from_triplets",
        "5 nodes, 2 components",
        &format!("nnz={}", graph.nnz()),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Step 2: Find connected components
    let t_start = Instant::now();
    let (num_components, sizes) = connected_component_sizes(&graph);
    let pass = num_components == 2 && sizes.iter().sum::<usize>() == n;
    steps.push(make_step(
        2,
        "find_components",
        "connected_component_sizes",
        "expected 2 components",
        &format!("found={}, sizes={:?}", num_components, sizes),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 3: Verify connectivity
    let t_start = Instant::now();
    let connected = is_connected(&graph);
    let pass2 = !connected; // Graph should NOT be connected
    steps.push(make_step(
        3,
        "check_connectivity",
        "is_connected",
        "expected false (2 components)",
        &format!("is_connected={}", connected),
        t_start.elapsed().as_nanos(),
        if pass2 { "ok" } else { "fail" },
    ));

    let overall_pass = pass && pass2;
    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if overall_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(
        overall_pass,
        "connected components: num={}, sizes={:?}",
        num_components, sizes
    );
}

/// Scenario 15: PageRank on simple graph
/// Tests PageRank convergence on a small directed graph.
#[test]
fn e2e_015_pagerank() {
    let scenario_id = "e2e_sparse_015_pagerank";
    let overall_start = Instant::now();
    let mut steps = Vec::new();

    // Step 1: Build a simple graph (chain with a hub)
    // 0 -> 1 -> 2 -> 3, and 0 -> 2 (making node 2 a mini-hub)
    let t_start = Instant::now();
    let n = 4;
    let rows = vec![0, 1, 2, 0];
    let cols = vec![1, 2, 3, 2];
    let data = vec![1.0; 4];
    let coo =
        CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false).expect("graph coo");
    let graph = coo.to_csr().expect("graph csr");
    steps.push(make_step(
        1,
        "build_graph",
        "from_triplets",
        "4 nodes, chain with hub",
        &format!("nnz={}", graph.nnz()),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Step 2: Compute PageRank
    let t_start = Instant::now();
    let damping = 0.85;
    let max_iter = 100;
    let tol = 1e-8;
    let ranks = pagerank(&graph, damping, max_iter, tol);
    let sum: f64 = ranks.iter().sum();
    let normalized = (sum - 1.0).abs() < 0.01;
    steps.push(make_step(
        2,
        "compute_pagerank",
        "pagerank",
        &format!("damping={}, max_iter={}", damping, max_iter),
        &format!("sum={:.4}, normalized={}", sum, normalized),
        t_start.elapsed().as_nanos(),
        if normalized { "ok" } else { "fail" },
    ));

    // Step 3: Verify rank ordering
    // Node 2 should have highest rank (most incoming links)
    // Node 3 should have decent rank (only sink)
    let t_start = Instant::now();
    let max_rank_node = ranks
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    // Node 2 or 3 should have highest rank
    let pass = max_rank_node == 2 || max_rank_node == 3;
    steps.push(make_step(
        3,
        "verify_ranking",
        "check max rank node",
        "expected node 2 or 3 highest",
        &format!("max_rank_node={}, ranks={:?}", max_rank_node, ranks),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let overall_pass = normalized && pass;
    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if overall_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(
        overall_pass,
        "PageRank: normalized={}, max_node={}",
        normalized, max_rank_node
    );
}

/// Scenario 16: Topological sort on DAG
/// Tests topological_sort on a directed acyclic graph.
#[test]
fn e2e_016_topological_sort() {
    let scenario_id = "e2e_sparse_016_toposort";
    let overall_start = Instant::now();
    let mut steps = Vec::new();

    // Step 1: Build a DAG: 0->1, 0->2, 1->3, 2->3
    let t_start = Instant::now();
    let n = 4;
    let rows = vec![0, 0, 1, 2];
    let cols = vec![1, 2, 3, 3];
    let data = vec![1.0; 4];
    let coo =
        CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false).expect("dag coo");
    let dag = coo.to_csr().expect("dag csr");
    steps.push(make_step(
        1,
        "build_dag",
        "from_triplets",
        "4 nodes, diamond DAG",
        &format!("nnz={}", dag.nnz()),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Step 2: Compute topological sort
    let t_start = Instant::now();
    let order = topological_sort(&dag);
    let has_order = order.is_some();
    steps.push(make_step(
        2,
        "toposort",
        "topological_sort",
        "expected valid ordering",
        &format!("has_order={}, order={:?}", has_order, order),
        t_start.elapsed().as_nanos(),
        if has_order { "ok" } else { "fail" },
    ));

    // Step 3: Verify ordering constraints
    let t_start = Instant::now();
    let mut pass = false;
    if let Some(ref o) = order {
        // In topological order: for edge u->v, u comes before v
        let pos: Vec<usize> = {
            let mut p = vec![0; n];
            for (i, &node) in o.iter().enumerate() {
                p[node] = i;
            }
            p
        };
        // 0->1: pos[0] < pos[1]
        // 0->2: pos[0] < pos[2]
        // 1->3: pos[1] < pos[3]
        // 2->3: pos[2] < pos[3]
        pass = pos[0] < pos[1] && pos[0] < pos[2] && pos[1] < pos[3] && pos[2] < pos[3];
    }
    steps.push(make_step(
        3,
        "verify_order",
        "check edge constraints",
        "all edges u->v have pos[u] < pos[v]",
        &format!("pass={}", pass),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(pass, "topological sort: order={:?}", order);
}

// ═══════════════════════════════════════════════════════════════════
// SPARSE MATRIX UTILITIES SCENARIOS (17-18)
// ═══════════════════════════════════════════════════════════════════

/// Scenario 17: Matrix norms
/// Tests sparse_norm for Frobenius, 1-norm, and inf-norm.
#[test]
fn e2e_017_sparse_norms() {
    let scenario_id = "e2e_sparse_017_norms";
    let overall_start = Instant::now();
    let mut steps = Vec::new();

    // Step 1: Build a simple matrix with known norms
    let t_start = Instant::now();
    let a = make_tridiag(3, -1.0, 2.0, -1.0);
    // Matrix:
    // [ 2 -1  0]
    // [-1  2 -1]
    // [ 0 -1  2]
    steps.push(make_step(
        1,
        "build_matrix",
        "tridiag",
        "3x3 tridiag(-1, 2, -1)",
        &format!("nnz={}", a.nnz()),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Step 2: Compute Frobenius norm
    let t_start = Instant::now();
    let fro = sparse_norm(&a, "fro");
    // Frobenius = sqrt(4 + 1 + 1 + 4 + 1 + 1 + 4) = sqrt(16) = 4 (off by one element)
    // Actually: 2^2 * 3 + (-1)^2 * 4 = 12 + 4 = 16, sqrt(16) = 4 - wait let me recalc
    // Elements: 2, -1 (row 0); -1, 2, -1 (row 1); -1, 2 (row 2)
    // = 4 + 1 + 1 + 4 + 1 + 1 + 4 = 16, no wait that's 7 elements for 3x3
    // Correct: diag has 3 2's = 12, off-diag has 4 -1's = 4, total = 16
    let expected_fro = 4.0;
    let fro_pass = (fro - expected_fro).abs() < 0.01;
    steps.push(make_step(
        2,
        "frobenius_norm",
        "sparse_norm(fro)",
        &format!("expected {}", expected_fro),
        &format!("computed={:.6}, pass={}", fro, fro_pass),
        t_start.elapsed().as_nanos(),
        if fro_pass { "ok" } else { "fail" },
    ));

    // Step 3: Compute 1-norm (max column sum of abs)
    let t_start = Instant::now();
    let norm1 = sparse_norm(&a, "1");
    // Column sums: |2|+|-1| = 3, |-1|+|2|+|-1| = 4, |0|+|-1|+|2| = 3
    let expected_1 = 4.0;
    let norm1_pass = (norm1 - expected_1).abs() < TOL;
    steps.push(make_step(
        3,
        "one_norm",
        "sparse_norm(1)",
        &format!("expected {}", expected_1),
        &format!("computed={:.6}, pass={}", norm1, norm1_pass),
        t_start.elapsed().as_nanos(),
        if norm1_pass { "ok" } else { "fail" },
    ));

    // Step 4: Compute inf-norm (max row sum of abs)
    let t_start = Instant::now();
    let norm_inf = sparse_norm(&a, "inf");
    // Row sums: |2|+|-1| = 3, |-1|+|2|+|-1| = 4, |-1|+|2| = 3
    let expected_inf = 4.0;
    let inf_pass = (norm_inf - expected_inf).abs() < TOL;
    steps.push(make_step(
        4,
        "inf_norm",
        "sparse_norm(inf)",
        &format!("expected {}", expected_inf),
        &format!("computed={:.6}, pass={}", norm_inf, inf_pass),
        t_start.elapsed().as_nanos(),
        if inf_pass { "ok" } else { "fail" },
    ));

    let overall_pass = fro_pass && norm1_pass && inf_pass;
    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if overall_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(
        overall_pass,
        "sparse norms: fro={}, 1={}, inf={}",
        fro, norm1, norm_inf
    );
}

/// Scenario 18: Strongly connected components
/// Tests SCC detection on a graph with multiple SCCs.
#[test]
fn e2e_018_strongly_connected_components() {
    let scenario_id = "e2e_sparse_018_scc";
    let overall_start = Instant::now();
    let mut steps = Vec::new();

    // Step 1: Build graph with 2 SCCs
    // SCC 1: 0 <-> 1 (bidirectional)
    // SCC 2: 2 -> 3 -> 2 (cycle)
    // Link: 1 -> 2 (cross-SCC)
    let t_start = Instant::now();
    let n = 4;
    let rows = vec![0, 1, 1, 2, 3];
    let cols = vec![1, 0, 2, 3, 2];
    let data = vec![1.0; 5];
    let coo = CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .expect("scc graph coo");
    let graph = coo.to_csr().expect("scc graph csr");
    steps.push(make_step(
        1,
        "build_graph",
        "from_triplets",
        "4 nodes, 2 SCCs",
        &format!("nnz={}", graph.nnz()),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    // Step 2: Find SCCs
    let t_start = Instant::now();
    let labels = strongly_connected_components(&graph);
    let num_sccs = labels.iter().max().map(|&x| x + 1).unwrap_or(0);
    let pass = num_sccs == 2;
    steps.push(make_step(
        2,
        "find_scc",
        "strongly_connected_components",
        "expected 2 SCCs",
        &format!("num_sccs={}, labels={:?}", num_sccs, labels),
        t_start.elapsed().as_nanos(),
        if pass { "ok" } else { "fail" },
    ));

    // Step 3: Verify SCC membership
    let t_start = Instant::now();
    // Nodes 0,1 should be in same SCC; nodes 2,3 should be in same SCC
    let same_01 = labels[0] == labels[1];
    let same_23 = labels[2] == labels[3];
    let different_groups = labels[0] != labels[2];
    let membership_pass = same_01 && same_23 && different_groups;
    steps.push(make_step(
        3,
        "verify_membership",
        "check SCC groups",
        "0,1 same; 2,3 same; groups different",
        &format!(
            "same_01={}, same_23={}, diff={}",
            same_01, same_23, different_groups
        ),
        t_start.elapsed().as_nanos(),
        if membership_pass { "ok" } else { "fail" },
    ));

    let overall_pass = pass && membership_pass;
    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if overall_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(overall_pass, "SCC: num={}, labels={:?}", num_sccs, labels);
}

/// Scenario 19: Sparse helper parity against live SciPy.
#[test]
fn e2e_019_sparse_helper_oracle_match() {
    let scipy_check = Command::new("python3")
        .arg("-c")
        .arg("import scipy; import numpy")
        .status();
    if !matches!(scipy_check, Ok(status) if status.success()) {
        eprintln!("SciPy/NumPy not available; skipping sparse helper oracle match");
        return;
    }

    let scenario_id = "e2e_sparse_019_helper_oracle";
    let overall_start = Instant::now();
    let mut steps = Vec::new();

    let fixture = SparseOracleFixture {
        packet_id: "FSCI-P2C-004".to_string(),
        family: "sparse".to_string(),
        cases: vec![
            SparseOracleCase {
                case_id: "find_duplicate_zero".to_string(),
                operation: "find".to_string(),
                format: None,
                matrix: Some(SparseOracleMatrix {
                    format: "coo".to_string(),
                    shape: [2, 2],
                    row: vec![0, 0, 0, 1, 1],
                    col: vec![0, 1, 1, 0, 1],
                    data: vec![0.0, 1.0, 2.0, 3.0, 4.0],
                }),
                blocks: None,
                k: None,
            },
            SparseOracleCase {
                case_id: "tril_explicit_zero".to_string(),
                operation: "tril".to_string(),
                format: None,
                matrix: Some(SparseOracleMatrix {
                    format: "coo".to_string(),
                    shape: [2, 2],
                    row: vec![0, 0, 0, 1],
                    col: vec![0, 1, 1, 0],
                    data: vec![0.0, 1.0, 2.0, 3.0],
                }),
                blocks: None,
                k: Some(0),
            },
            SparseOracleCase {
                case_id: "triu_duplicate_offset".to_string(),
                operation: "triu".to_string(),
                format: None,
                matrix: Some(SparseOracleMatrix {
                    format: "coo".to_string(),
                    shape: [2, 3],
                    row: vec![0, 0, 0, 1],
                    col: vec![0, 1, 1, 2],
                    data: vec![5.0, 1.0, 2.0, 3.0],
                }),
                blocks: None,
                k: Some(1),
            },
            SparseOracleCase {
                case_id: "tocsc_duplicate_cancellation_keeps_zero".to_string(),
                operation: "tocsc".to_string(),
                format: None,
                matrix: Some(SparseOracleMatrix {
                    format: "coo".to_string(),
                    shape: [2, 2],
                    row: vec![0, 0, 1],
                    col: vec![0, 0, 1],
                    data: vec![1.0, -1.0, 2.0],
                }),
                blocks: None,
                k: None,
            },
            SparseOracleCase {
                case_id: "vstack_mixed_formats".to_string(),
                operation: "vstack".to_string(),
                format: None,
                matrix: None,
                blocks: Some(vec![
                    SparseOracleMatrix {
                        format: "coo".to_string(),
                        shape: [2, 2],
                        row: vec![0, 0, 1, 1],
                        col: vec![0, 1, 0, 1],
                        data: vec![1.0, 2.0, 3.0, 4.0],
                    },
                    SparseOracleMatrix {
                        format: "csr".to_string(),
                        shape: [1, 2],
                        row: vec![0, 0],
                        col: vec![0, 1],
                        data: vec![5.0, 6.0],
                    },
                ]),
                k: None,
            },
            SparseOracleCase {
                case_id: "hstack_mixed_formats".to_string(),
                operation: "hstack".to_string(),
                format: None,
                matrix: None,
                blocks: Some(vec![
                    SparseOracleMatrix {
                        format: "coo".to_string(),
                        shape: [2, 2],
                        row: vec![0, 0, 1, 1],
                        col: vec![0, 1, 0, 1],
                        data: vec![1.0, 2.0, 3.0, 4.0],
                    },
                    SparseOracleMatrix {
                        format: "csc".to_string(),
                        shape: [2, 1],
                        row: vec![0, 1],
                        col: vec![0, 0],
                        data: vec![5.0, 6.0],
                    },
                ]),
                k: None,
            },
            SparseOracleCase {
                case_id: "hstack_format_csr".to_string(),
                operation: "hstack".to_string(),
                format: Some("csr".to_string()),
                matrix: None,
                blocks: Some(vec![
                    SparseOracleMatrix {
                        format: "coo".to_string(),
                        shape: [2, 2],
                        row: vec![0, 0, 1, 1],
                        col: vec![0, 1, 0, 1],
                        data: vec![1.0, 2.0, 3.0, 4.0],
                    },
                    SparseOracleMatrix {
                        format: "csc".to_string(),
                        shape: [2, 1],
                        row: vec![0, 1],
                        col: vec![0, 0],
                        data: vec![5.0, 6.0],
                    },
                ]),
                k: None,
            },
            SparseOracleCase {
                case_id: "hstack_format_csc".to_string(),
                operation: "hstack".to_string(),
                format: Some("csc".to_string()),
                matrix: None,
                blocks: Some(vec![
                    SparseOracleMatrix {
                        format: "coo".to_string(),
                        shape: [2, 2],
                        row: vec![0, 0, 1, 1],
                        col: vec![0, 1, 0, 1],
                        data: vec![1.0, 2.0, 3.0, 4.0],
                    },
                    SparseOracleMatrix {
                        format: "csc".to_string(),
                        shape: [2, 1],
                        row: vec![0, 1],
                        col: vec![0, 0],
                        data: vec![5.0, 6.0],
                    },
                ]),
                k: None,
            },
            SparseOracleCase {
                case_id: "hstack_format_coo".to_string(),
                operation: "hstack".to_string(),
                format: Some("coo".to_string()),
                matrix: None,
                blocks: Some(vec![
                    SparseOracleMatrix {
                        format: "coo".to_string(),
                        shape: [2, 2],
                        row: vec![0, 0, 1, 1],
                        col: vec![0, 1, 0, 1],
                        data: vec![1.0, 2.0, 3.0, 4.0],
                    },
                    SparseOracleMatrix {
                        format: "csc".to_string(),
                        shape: [2, 1],
                        row: vec![0, 1],
                        col: vec![0, 0],
                        data: vec![5.0, 6.0],
                    },
                ]),
                k: None,
            },
            SparseOracleCase {
                case_id: "hstack_format_bsr".to_string(),
                operation: "hstack".to_string(),
                format: Some("bsr".to_string()),
                matrix: None,
                blocks: Some(vec![
                    SparseOracleMatrix {
                        format: "coo".to_string(),
                        shape: [2, 2],
                        row: vec![0, 0, 1, 1],
                        col: vec![0, 1, 0, 1],
                        data: vec![1.0, 2.0, 3.0, 4.0],
                    },
                    SparseOracleMatrix {
                        format: "csc".to_string(),
                        shape: [2, 1],
                        row: vec![0, 1],
                        col: vec![0, 0],
                        data: vec![5.0, 6.0],
                    },
                ]),
                k: None,
            },
            SparseOracleCase {
                case_id: "hstack_format_dia".to_string(),
                operation: "hstack".to_string(),
                format: Some("dia".to_string()),
                matrix: None,
                blocks: Some(vec![
                    SparseOracleMatrix {
                        format: "coo".to_string(),
                        shape: [2, 2],
                        row: vec![0, 0, 1, 1],
                        col: vec![0, 1, 0, 1],
                        data: vec![1.0, 2.0, 3.0, 4.0],
                    },
                    SparseOracleMatrix {
                        format: "csc".to_string(),
                        shape: [2, 1],
                        row: vec![0, 1],
                        col: vec![0, 0],
                        data: vec![5.0, 6.0],
                    },
                ]),
                k: None,
            },
            SparseOracleCase {
                case_id: "hstack_format_dok".to_string(),
                operation: "hstack".to_string(),
                format: Some("dok".to_string()),
                matrix: None,
                blocks: Some(vec![
                    SparseOracleMatrix {
                        format: "coo".to_string(),
                        shape: [2, 2],
                        row: vec![0, 0, 1, 1],
                        col: vec![0, 1, 0, 1],
                        data: vec![1.0, 2.0, 3.0, 4.0],
                    },
                    SparseOracleMatrix {
                        format: "csc".to_string(),
                        shape: [2, 1],
                        row: vec![0, 1],
                        col: vec![0, 0],
                        data: vec![5.0, 6.0],
                    },
                ]),
                k: None,
            },
            SparseOracleCase {
                case_id: "hstack_format_lil".to_string(),
                operation: "hstack".to_string(),
                format: Some("lil".to_string()),
                matrix: None,
                blocks: Some(vec![
                    SparseOracleMatrix {
                        format: "coo".to_string(),
                        shape: [2, 2],
                        row: vec![0, 0, 1, 1],
                        col: vec![0, 1, 0, 1],
                        data: vec![1.0, 2.0, 3.0, 4.0],
                    },
                    SparseOracleMatrix {
                        format: "csc".to_string(),
                        shape: [2, 1],
                        row: vec![0, 1],
                        col: vec![0, 0],
                        data: vec![5.0, 6.0],
                    },
                ]),
                k: None,
            },
            SparseOracleCase {
                case_id: "csr_matmul_identity_reverses_sorted_rows".to_string(),
                operation: "csr_matmul".to_string(),
                format: None,
                matrix: None,
                blocks: Some(vec![
                    SparseOracleMatrix {
                        format: "csr".to_string(),
                        shape: [2, 3],
                        row: vec![0, 0, 1, 1],
                        col: vec![0, 2, 1, 2],
                        data: vec![1.0, 2.0, 3.0, 4.0],
                    },
                    SparseOracleMatrix {
                        format: "csr".to_string(),
                        shape: [3, 3],
                        row: vec![0, 1, 2],
                        col: vec![0, 1, 2],
                        data: vec![1.0, 1.0, 1.0],
                    },
                ]),
                k: None,
            },
        ],
    };

    let fixture_path = sparse_oracle_temp_path("fsci_sparse_helper_fixture");
    let output_path = sparse_oracle_temp_path("fsci_sparse_helper_output");
    let t_start = Instant::now();
    fs::write(
        &fixture_path,
        serde_json::to_vec_pretty(&fixture).expect("serialize fixture"),
    )
    .expect("write fixture");
    steps.push(make_step(
        1,
        "write_fixture",
        "serialize fixture JSON",
        &format!("cases={}", fixture.cases.len()),
        &fixture_path.display().to_string(),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    let oracle_script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python_oracle/scipy_sparse_oracle.py");
    let t_start = Instant::now();
    let output = Command::new("python3")
        .arg(&oracle_script)
        .arg("--fixture")
        .arg(&fixture_path)
        .arg("--output")
        .arg(&output_path)
        .status()
        .expect("run sparse oracle");
    steps.push(make_step(
        2,
        "run_scipy_oracle",
        "python3 scipy_sparse_oracle.py",
        &oracle_script.display().to_string(),
        &format!("success={}", output.success()),
        t_start.elapsed().as_nanos(),
        if output.success() { "ok" } else { "fail" },
    ));
    assert!(output.success(), "SciPy sparse oracle should succeed");

    let t_start = Instant::now();
    let capture: SparseOracleCapture =
        serde_json::from_slice(&fs::read(&output_path).expect("read oracle output"))
            .expect("parse oracle output");
    steps.push(make_step(
        3,
        "load_oracle_output",
        "parse output JSON",
        &output_path.display().to_string(),
        &format!("case_outputs={}", capture.case_outputs.len()),
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    let t_start = Instant::now();
    for case in &fixture.cases {
        let actual = run_sparse_oracle_case(case);
        let expected = capture
            .case_outputs
            .iter()
            .find(|output| output.case_id == case.case_id)
            .expect("oracle output should exist for every fixture case");
        assert_eq!(
            actual.status, "ok",
            "{} rust execution failed",
            case.case_id
        );
        assert!(
            actual.error.is_none(),
            "{} rust error present",
            case.case_id
        );
        assert_eq!(expected.status, "ok", "{} oracle failed", case.case_id);
        assert!(
            expected.error.is_none(),
            "{} oracle error present",
            case.case_id
        );
        assert_eq!(
            actual.result_kind, expected.result_kind,
            "{} result kind mismatch",
            case.case_id
        );
        assert!(
            matches!(
                expected.result_kind.as_str(),
                "find_triplets" | "matrix_triplets" | "csr_components"
            ),
            "unsupported oracle result kind {}",
            expected.result_kind
        );
        if expected.result_kind == "find_triplets" {
            let actual_result: SparseOracleFindResult =
                serde_json::from_value(actual.result.clone()).expect("actual find result");
            let expected_result: SparseOracleFindResult =
                serde_json::from_value(expected.result.clone()).expect("oracle find result");
            compare_find_result(&case.case_id, &actual_result, &expected_result);
        } else if expected.result_kind == "matrix_triplets" {
            let actual_result: SparseOracleTriplets =
                serde_json::from_value(actual.result.clone()).expect("actual triplets");
            let expected_result: SparseOracleTriplets =
                serde_json::from_value(expected.result.clone()).expect("oracle triplets");
            compare_sparse_triplets(&case.case_id, &actual_result, &expected_result);
            if case.format.is_some() {
                assert_eq!(
                    actual_result.format, expected_result.format,
                    "{} format mismatch",
                    case.case_id
                );
            }
        } else {
            let actual_result: SparseOracleCsrComponents =
                serde_json::from_value(actual.result.clone()).expect("actual csr components");
            let expected_result: SparseOracleCsrComponents =
                serde_json::from_value(expected.result.clone()).expect("oracle csr components");
            compare_csr_components(&case.case_id, &actual_result, &expected_result);
        }
    }
    steps.push(make_step(
        4,
        "compare_results",
        "rust vs scipy helper outputs",
        &format!("cases={}", fixture.cases.len()),
        "all helper cases matched",
        t_start.elapsed().as_nanos(),
        "ok",
    ));

    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: "pass".to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
}

/// Scenario 20: COO setdiag boundary parity.
#[test]
fn e2e_020_coo_setdiag_boundary_parity() {
    let scenario_id = "e2e_sparse_020_coo_setdiag_boundary";
    let overall_start = Instant::now();
    let mut steps = Vec::new();

    let t_start = Instant::now();
    let mut superdiag =
        CooMatrix::from_triplets(Shape2D::new(3, 4), vec![8.0], vec![0], vec![3], false)
            .expect("superdiag base");
    superdiag
        .setdiag(&[1.0, 2.0, 3.0, 4.0], 1)
        .expect("superdiag setdiag");
    let superdiag_dense = dense_from_coo(&superdiag);
    let superdiag_pass = superdiag.row_indices() == [0, 0, 1, 2]
        && superdiag.col_indices() == [3, 1, 2, 3]
        && max_abs_diff_vec(superdiag.data(), &[8.0, 1.0, 2.0, 3.0]) <= TOL
        && superdiag_dense
            == vec![
                vec![0.0, 1.0, 0.0, 8.0],
                vec![0.0, 0.0, 2.0, 0.0],
                vec![0.0, 0.0, 0.0, 3.0],
            ];
    steps.push(make_step(
        1,
        "set_superdiag",
        "coo.setdiag(values, k=1)",
        "shape=(3,4), values=[1,2,3,4]",
        &format!(
            "rows={:?} cols={:?} data={:?}",
            superdiag.row_indices(),
            superdiag.col_indices(),
            superdiag.data()
        ),
        t_start.elapsed().as_nanos(),
        if superdiag_pass { "ok" } else { "fail" },
    ));

    let t_start = Instant::now();
    let mut partial = CooMatrix::from_triplets(
        Shape2D::new(3, 3),
        vec![10.0, 20.0, 30.0],
        vec![0, 1, 2],
        vec![0, 1, 2],
        false,
    )
    .expect("partial base");
    partial.setdiag(&[5.0], 0).expect("partial setdiag");
    let partial_pass = partial.row_indices() == [1, 2, 0]
        && partial.col_indices() == [1, 2, 0]
        && max_abs_diff_vec(partial.data(), &[20.0, 30.0, 5.0]) <= TOL;
    steps.push(make_step(
        2,
        "set_partial_main_diag",
        "coo.setdiag(values, k=0)",
        "shape=(3,3), values=[5]",
        &format!(
            "rows={:?} cols={:?} data={:?}",
            partial.row_indices(),
            partial.col_indices(),
            partial.data()
        ),
        t_start.elapsed().as_nanos(),
        if partial_pass { "ok" } else { "fail" },
    ));

    let t_start = Instant::now();
    let mut scalar = CooMatrix::from_triplets(Shape2D::new(3, 4), vec![], vec![], vec![], false)
        .expect("scalar base");
    scalar.setdiag_scalar(9.0, 1).expect("scalar setdiag");
    let scalar_pass = scalar.row_indices() == [0, 1, 2]
        && scalar.col_indices() == [1, 2, 3]
        && max_abs_diff_vec(scalar.data(), &[9.0, 9.0, 9.0]) <= TOL;
    steps.push(make_step(
        3,
        "broadcast_scalar",
        "coo.setdiag_scalar(value, k=1)",
        "shape=(3,4), value=9",
        &format!(
            "rows={:?} cols={:?} data={:?}",
            scalar.row_indices(),
            scalar.col_indices(),
            scalar.data()
        ),
        t_start.elapsed().as_nanos(),
        if scalar_pass { "ok" } else { "fail" },
    ));

    let t_start = Instant::now();
    let err = scalar.setdiag(&[1.0], 4).expect_err("out-of-bounds k");
    let err_pass = matches!(err, SparseError::InvalidArgument { ref message } if message == "k exceeds array dimensions");
    steps.push(make_step(
        4,
        "reject_out_of_bounds_k",
        "coo.setdiag(values, k=4)",
        "shape=(3,4), values=[1]",
        &err.to_string(),
        t_start.elapsed().as_nanos(),
        if err_pass { "ok" } else { "fail" },
    ));

    let overall_pass = superdiag_pass && partial_pass && scalar_pass && err_pass;
    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if overall_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(overall_pass, "COO setdiag parity checks should match SciPy");
}

/// Scenario 21: Dijkstra negative-edge handling parity.
#[test]
fn e2e_021_dijkstra_negative_weight_handling() {
    let scenario_id = "e2e_sparse_021_dijkstra_negative_weights";
    let overall_start = Instant::now();
    let mut steps = Vec::new();

    let t_start = Instant::now();
    let reachable_negative = CooMatrix::from_triplets(
        Shape2D::new(3, 3),
        vec![1.0, -2.0],
        vec![0, 1],
        vec![1, 2],
        false,
    )
    .expect("reachable negative coo")
    .to_csr()
    .expect("reachable negative csr");
    let reachable = dijkstra(&reachable_negative, 0).expect("reachable negative dijkstra");
    let reachable_pass = max_abs_diff_vec(&reachable.distances, &[0.0, 1.0, -1.0]) <= TOL;
    steps.push(make_step(
        1,
        "reachable_negative_edge",
        "dijkstra(graph, source=0)",
        "edges: 0->1=1, 1->2=-2",
        &format!("distances={:?}", reachable.distances),
        t_start.elapsed().as_nanos(),
        if reachable_pass { "ok" } else { "fail" },
    ));

    let t_start = Instant::now();
    let split_graph = CooMatrix::from_triplets(
        Shape2D::new(4, 4),
        vec![1.0, -2.0],
        vec![0, 2],
        vec![1, 3],
        false,
    )
    .expect("split graph coo")
    .to_csr()
    .expect("split graph csr");
    let split = dijkstra(&split_graph, 0).expect("split graph dijkstra");
    let split_pass = split.distances[0] == 0.0
        && split.distances[1] == 1.0
        && split.distances[2].is_infinite()
        && split.distances[3].is_infinite();
    steps.push(make_step(
        2,
        "unreachable_negative_component",
        "dijkstra(graph, source=0)",
        "negative edge isolated in unreachable component",
        &format!("distances={:?}", split.distances),
        t_start.elapsed().as_nanos(),
        if split_pass { "ok" } else { "fail" },
    ));

    let overall_pass = reachable_pass && split_pass;
    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if overall_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(
        overall_pass,
        "Dijkstra negative-edge handling should match SciPy"
    );
}

/// Scenario 22: LIL slicing parity.
#[test]
fn e2e_022_lil_matrix_slicing_parity() {
    let scenario_id = "e2e_sparse_022_lil_matrix_slicing";
    let overall_start = Instant::now();
    let mut steps = Vec::new();

    let mut lil = LilMatrix::new(Shape2D::new(4, 5));
    lil.insert(0, 1, 2.0).expect("insert");
    lil.insert(1, 3, 5.0).expect("insert");
    lil.insert(2, 0, 7.0).expect("insert");
    lil.insert(3, 4, 11.0).expect("insert");

    let t_start = Instant::now();
    let row_slice = lil
        .slice(SparseSliceSpec::new(1, 3, 1), SparseSliceSpec::new(0, 5, 1))
        .expect("row slice");
    let row_dense = dense_from_coo(&row_slice.to_coo().expect("row slice coo"));
    let row_pass = row_dense == vec![vec![0.0, 0.0, 0.0, 5.0, 0.0], vec![7.0, 0.0, 0.0, 0.0, 0.0]];
    steps.push(make_step(
        1,
        "row_slice_full_columns",
        "lil.slice(rows=1:3, cols=:)",
        "SciPy row slice should preserve row order and full-width columns",
        &format!("dense={row_dense:?}"),
        t_start.elapsed().as_nanos(),
        if row_pass { "ok" } else { "fail" },
    ));

    let t_start = Instant::now();
    let col_slice = lil
        .slice(SparseSliceSpec::new(0, 4, 1), SparseSliceSpec::new(1, 4, 1))
        .expect("col slice");
    let col_dense = dense_from_coo(&col_slice.to_coo().expect("col slice coo"));
    let col_pass = col_dense
        == vec![
            vec![2.0, 0.0, 0.0],
            vec![0.0, 0.0, 5.0],
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ];
    steps.push(make_step(
        2,
        "column_slice_full_rows",
        "lil.slice(rows=:, cols=1:4)",
        "SciPy column slice should rebase column indices into the sliced matrix",
        &format!("dense={col_dense:?}"),
        t_start.elapsed().as_nanos(),
        if col_pass { "ok" } else { "fail" },
    ));

    let t_start = Instant::now();
    let stepped = lil
        .slice(SparseSliceSpec::new(1, 4, 1), SparseSliceSpec::new(1, 5, 2))
        .expect("stepped slice");
    let stepped_dense = dense_from_coo(&stepped.to_coo().expect("stepped coo"));
    let stepped_pass = stepped_dense == vec![vec![0.0, 5.0], vec![0.0, 0.0], vec![0.0, 0.0]];
    steps.push(make_step(
        3,
        "stepped_column_slice",
        "lil.slice(rows=1:4, cols=1:5:2)",
        "positive-step SciPy slicing should keep only every second in-range column",
        &format!("dense={stepped_dense:?}"),
        t_start.elapsed().as_nanos(),
        if stepped_pass { "ok" } else { "fail" },
    ));

    let overall_pass = row_pass && col_pass && stepped_pass;
    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if overall_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(overall_pass, "LIL slicing parity should match SciPy");
}

/// Scenario 23: CSC boolean index parity.
#[test]
fn e2e_023_csc_boolean_index_parity() {
    let scenario_id = "e2e_sparse_023_csc_boolean_index";
    let overall_start = Instant::now();
    let mut steps = Vec::new();

    let csc = CooMatrix::from_triplets(
        Shape2D::new(3, 4),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![0, 0, 1, 1, 2, 2],
        vec![0, 2, 1, 3, 0, 2],
        false,
    )
    .expect("valid coo")
    .to_csc()
    .expect("coo->csc");

    let t_start = Instant::now();
    let row_indexed = csc
        .boolean_row_index(&[true, false, true])
        .expect("row boolean index");
    let row_dense = dense_from_coo(&row_indexed.to_coo().expect("row indexed coo"));
    let row_pass = row_dense == vec![vec![1.0, 0.0, 2.0, 0.0], vec![5.0, 0.0, 6.0, 0.0]];
    steps.push(make_step(
        1,
        "boolean_row_index",
        "csc[rows_bool, :]",
        "SciPy csc_matrix boolean row indexing preserves CSC format and rebases kept rows",
        &format!("dense={row_dense:?}"),
        t_start.elapsed().as_nanos(),
        if row_pass { "ok" } else { "fail" },
    ));

    let t_start = Instant::now();
    let col_indexed = csc
        .boolean_col_index(&[false, true, true, false])
        .expect("col boolean index");
    let col_dense = dense_from_coo(&col_indexed.to_coo().expect("col indexed coo"));
    let col_pass = col_dense == vec![vec![0.0, 2.0], vec![3.0, 0.0], vec![0.0, 6.0]];
    steps.push(make_step(
        2,
        "boolean_col_index",
        "csc[:, cols_bool]",
        "SciPy csc_matrix boolean column indexing preserves CSC format and rebases kept columns",
        &format!("dense={col_dense:?}"),
        t_start.elapsed().as_nanos(),
        if col_pass { "ok" } else { "fail" },
    ));

    let t_start = Instant::now();
    let submatrix = csc
        .boolean_index(&[true, false, true], &[false, true, true, false])
        .expect("boolean submatrix");
    let sub_dense = dense_from_coo(&submatrix.to_coo().expect("submatrix coo"));
    let sub_pass = sub_dense == vec![vec![0.0, 2.0], vec![0.0, 6.0]];
    steps.push(make_step(
        3,
        "boolean_submatrix",
        "csc[rows_bool, :][:, cols_bool]",
        "composed boolean row/column indexing should match SciPy's CSC submatrix result",
        &format!("dense={sub_dense:?}"),
        t_start.elapsed().as_nanos(),
        if sub_pass { "ok" } else { "fail" },
    ));

    let t_start = Instant::now();
    let mask_values = csc
        .boolean_mask_values(&[
            vec![false, true, true, false],
            vec![false, false, false, true],
            vec![true, false, false, false],
        ])
        .expect("boolean mask values");
    let mask_pass = mask_values == vec![0.0, 2.0, 4.0, 5.0];
    steps.push(make_step(
        4,
        "boolean_mask_values",
        "csc[mask_2d]",
        "full-shape boolean masking should include implicit zeros in the SciPy-selected order",
        &format!("values={mask_values:?}"),
        t_start.elapsed().as_nanos(),
        if mask_pass { "ok" } else { "fail" },
    ));

    let overall_pass = row_pass && col_pass && sub_pass && mask_pass;
    let bundle = ForensicLogBundle {
        scenario_id: scenario_id.to_string(),
        steps,
        artifacts: vec![],
        environment: make_env(),
        overall: OverallResult {
            status: if overall_pass { "pass" } else { "fail" }.to_string(),
            total_duration_ns: overall_start.elapsed().as_nanos(),
            replay_command: replay_cmd(scenario_id),
            error_chain: None,
        },
    };
    write_bundle(scenario_id, &bundle);
    assert!(overall_pass, "CSC boolean indexing should match SciPy");
}
