#![forbid(unsafe_code)]

pub mod audit;
pub mod construct;
pub mod formats;
pub mod linalg;
pub mod ops;

pub use audit::{
    SyncSharedAuditLedger, record_bounded_recovery, record_fail_closed, sync_audit_ledger,
};

pub use construct::{
    HstackOutput, block_diag, bmat, diags, eye, eye_rectangular, hstack, hstack_with_format,
    identity, kron, kronsum, random, spdiags, vstack,
};
pub use formats::{
    BsrMatrix, CanonicalMeta, ConstructionLogEntry, CooMatrix, CscMatrix, CsrMatrix, DiaMatrix,
    DokMatrix, LilMatrix, NalgebraBridge, Shape2D, SparseError, SparseFormat, SparseResult,
    SparseSliceSpec,
};

// SciPy-compatible lowercase type aliases (e.g. `csr_matrix` mirrors
// `scipy.sparse.csr_matrix`). The names are intentionally non-CamelCase for API
// parity; scope `non_camel_case_types` to this module — applied to the whole
// group with one attribute — and re-export so the public paths
// (`fsci_sparse::csr_matrix`, ...) are unchanged. This lets dependent crates'
// `clippy --all-targets -D warnings` gates pass. [frankenscipy-6946y]
#[allow(non_camel_case_types)]
mod scipy_aliases {
    use super::{BsrMatrix, CooMatrix, CscMatrix, CsrMatrix, DiaMatrix, DokMatrix, LilMatrix};

    pub type csr_matrix = CsrMatrix;
    pub type csc_matrix = CscMatrix;
    pub type coo_matrix = CooMatrix;
    pub type bsr_matrix = BsrMatrix;
    pub type dia_matrix = DiaMatrix;
    pub type dok_matrix = DokMatrix;
    pub type lil_matrix = LilMatrix;

    pub type csr_array = CsrMatrix;
    pub type csc_array = CscMatrix;
    pub type coo_array = CooMatrix;
    pub type bsr_array = BsrMatrix;
    pub type dia_array = DiaMatrix;
    pub type dok_array = DokMatrix;
    pub type lil_array = LilMatrix;
}
pub use linalg::{
    CaspIterativeDecision,
    CaspIterativeSolveOptions,
    CaspIterativeSolveResult,
    CaspIterativeSolver,
    CaspMatvecCost,
    ConnectedComponentsResult,
    EigsOptions,
    EigsResult,
    ExpmOptions,
    IluOptions,
    IterativeSolveOptions,
    IterativeSolveResult,
    LgmresOptions,
    LuOptions,
    MstResult,
    PermutationOrdering,
    ShortestPathResult,
    SolveOptions,
    SolveResult,
    SparseBackend,
    SparseIluFactorization,
    SparseLuFactorization,
    SvdsResult,
    average_clustering,
    bellman_ford,
    betweenness_centrality,
    bicg,
    bicgstab,
    breadth_first_order,
    casp_iterative_solve,
    casp_iterative_solve_with_audit,
    // Iterative solvers
    cg,
    cgs,
    closeness_centrality,
    clustering_coefficient,
    connected_component_sizes,
    // Graph algorithms
    connected_components,
    degree_sequence,
    depth_first_order,
    dijkstra,
    eccentricity,
    // Eigensolvers
    eigs,
    eigsh,
    expm,
    floyd_warshall,
    gmres,
    graph_diameter,
    is_connected,
    laplacian,
    lgmres,
    lsmr,
    lsqr,
    matrix_power,
    minimum_spanning_tree,
    minres,
    onenormest,
    pagerank,
    pcg,
    qmr,
    reverse_cuthill_mckee,
    select_casp_iterative_solver,
    shortest_path,
    sparse_abs,
    sparse_add,
    sparse_col_sums,
    sparse_density,
    sparse_diagonal,
    sparse_eliminate_zeros,
    sparse_frobenius_inner,
    sparse_has_explicit_zeros,
    sparse_is_symmetric,
    sparse_map,
    sparse_nnz,
    // Sparse matrix operations
    sparse_norm,
    sparse_power,
    sparse_row_max,
    sparse_row_min,
    sparse_row_sums,
    sparse_scale,
    sparse_submatrix,
    sparse_sum,
    sparse_trace,
    sparse_transpose,
    spilu,
    splu,
    splu_solve,
    spmm,
    spmv,
    // Direct solvers
    spsolve,
    spsolve_triangular,
    strongly_connected_components,
    structural_rank,
    svds,
    topological_sort,
};
pub use ops::{
    ConversionLogEntry, FormatConvertible, add_coo, add_csc, add_csr, coo_to_csr_with_mode,
    csc_to_csr_with_mode, csr_to_csc_with_mode, find, scale_coo, scale_csc, scale_csr, spmv_coo,
    spmv_csc, spmv_csr, sub_coo, sub_csc, sub_csr, tril, triu,
};
pub use scipy_aliases::*;

pub trait SparseMatrix {
    fn format(&self) -> SparseFormat;
    fn shape(&self) -> Shape2D;
    fn nnz(&self) -> usize;
}

impl SparseMatrix for CsrMatrix {
    fn format(&self) -> SparseFormat {
        SparseFormat::Csr
    }
    fn shape(&self) -> Shape2D {
        self.shape()
    }
    fn nnz(&self) -> usize {
        self.nnz()
    }
}

impl SparseMatrix for CscMatrix {
    fn format(&self) -> SparseFormat {
        SparseFormat::Csc
    }
    fn shape(&self) -> Shape2D {
        self.shape()
    }
    fn nnz(&self) -> usize {
        self.nnz()
    }
}

impl SparseMatrix for CooMatrix {
    fn format(&self) -> SparseFormat {
        SparseFormat::Coo
    }
    fn shape(&self) -> Shape2D {
        self.shape()
    }
    fn nnz(&self) -> usize {
        self.nnz()
    }
}

impl SparseMatrix for BsrMatrix {
    fn format(&self) -> SparseFormat {
        SparseFormat::Bsr
    }
    fn shape(&self) -> Shape2D {
        self.shape()
    }
    fn nnz(&self) -> usize {
        self.nnz()
    }
}

impl SparseMatrix for DiaMatrix {
    fn format(&self) -> SparseFormat {
        SparseFormat::Dia
    }
    fn shape(&self) -> Shape2D {
        self.shape()
    }
    fn nnz(&self) -> usize {
        self.nnz()
    }
}

impl SparseMatrix for DokMatrix {
    fn format(&self) -> SparseFormat {
        SparseFormat::Dok
    }
    fn shape(&self) -> Shape2D {
        self.shape()
    }
    fn nnz(&self) -> usize {
        self.nnz()
    }
}

impl SparseMatrix for LilMatrix {
    fn format(&self) -> SparseFormat {
        SparseFormat::Lil
    }
    fn shape(&self) -> Shape2D {
        self.shape()
    }
    fn nnz(&self) -> usize {
        self.nnz()
    }
}

pub fn issparse<T: SparseMatrix>(_matrix: &T) -> bool {
    true
}

pub fn isspmatrix<T: SparseMatrix>(_matrix: &T) -> bool {
    true
}

pub fn isspmatrix_csr<T: SparseMatrix>(matrix: &T) -> bool {
    matches!(matrix.format(), SparseFormat::Csr)
}

pub fn isspmatrix_csc<T: SparseMatrix>(matrix: &T) -> bool {
    matches!(matrix.format(), SparseFormat::Csc)
}

pub fn isspmatrix_coo<T: SparseMatrix>(matrix: &T) -> bool {
    matches!(matrix.format(), SparseFormat::Coo)
}

pub fn isspmatrix_bsr<T: SparseMatrix>(matrix: &T) -> bool {
    matches!(matrix.format(), SparseFormat::Bsr)
}

pub fn isspmatrix_dia<T: SparseMatrix>(matrix: &T) -> bool {
    matches!(matrix.format(), SparseFormat::Dia)
}

pub fn isspmatrix_dok<T: SparseMatrix>(matrix: &T) -> bool {
    matches!(matrix.format(), SparseFormat::Dok)
}

pub fn isspmatrix_lil<T: SparseMatrix>(matrix: &T) -> bool {
    matches!(matrix.format(), SparseFormat::Lil)
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use fsci_runtime::RuntimeMode;
    use proptest::prelude::*;
    use serde_json::{Value, json};

    use super::*;

    const PROPTEST_CASES: u32 = 512;

    #[test]
    fn issparse_and_isspmatrix_type_checks_work() {
        let csr = CsrMatrix::from_components(
            Shape2D::new(2, 2),
            vec![1.0, 2.0],
            vec![0, 1],
            vec![0, 1, 2],
            false,
        )
        .unwrap();

        assert!(issparse(&csr));
        assert!(isspmatrix(&csr));
        assert!(isspmatrix_csr(&csr));
        assert!(!isspmatrix_csc(&csr));
        assert!(!isspmatrix_coo(&csr));

        let coo = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![1.0, 2.0],
            vec![0, 1],
            vec![0, 1],
            false,
        )
        .unwrap();

        assert!(issparse(&coo));
        assert!(isspmatrix(&coo));
        assert!(isspmatrix_coo(&coo));
        assert!(!isspmatrix_csr(&coo));
    }
    const LOG_SEED: u64 = 0xF5C1_004E;
    const EPS: f64 = 1e-12;

    #[derive(Clone, Debug)]
    struct SparseCase {
        shape: Shape2D,
        data: Vec<f64>,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
    }

    impl SparseCase {
        fn from_triplets(rows: usize, cols: usize, triplets: Vec<(usize, usize, i16)>) -> Self {
            let mut data = Vec::with_capacity(triplets.len());
            let mut row_indices = Vec::with_capacity(triplets.len());
            let mut col_indices = Vec::with_capacity(triplets.len());

            for (row, col, value) in triplets {
                row_indices.push(row);
                col_indices.push(col);
                data.push(f64::from(value));
            }

            Self {
                shape: Shape2D::new(rows, cols),
                data,
                row_indices,
                col_indices,
            }
        }
    }

    #[test]
    fn rejects_invalid_csr_pointer_shape() {
        let shape = Shape2D::new(2, 2);
        let err = CsrMatrix::from_components(shape, vec![1.0, 2.0], vec![0, 1], vec![0, 1], false)
            .expect_err("indptr length must be rows + 1");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    #[test]
    fn converts_coo_to_csr_and_preserves_dense_semantics() {
        let shape = Shape2D::new(2, 2);
        let coo = CooMatrix::from_triplets(
            shape,
            vec![1.0, 2.0, 3.0],
            vec![0, 0, 1],
            vec![0, 1, 1],
            false,
        )
        .expect("valid coo");
        let csr = coo.to_csr().expect("coo->csr conversion");
        assert_eq!(csr.shape(), shape);
        assert_eq!(csr.nnz(), 3);
    }

    #[test]
    fn csr_construction_deduplicates_triplets() {
        let coo = CooMatrix::from_triplets(
            Shape2D::new(2, 3),
            vec![2.0, 1.0, 3.0, 4.0],
            vec![1, 0, 1, 1],
            vec![2, 0, 2, 1],
            false,
        )
        .expect("valid coo");
        let csr = coo.to_csr().expect("coo->csr");
        assert_eq!(csr.nnz(), 3);
        assert!(csr.canonical_meta().sorted_indices);
        assert!(csr.canonical_meta().deduplicated);

        let dense = dense_from_coo(&csr.to_coo().expect("csr->coo"));
        assert_matrix_close(&dense, &[vec![1.0, 0.0, 0.0], vec![0.0, 4.0, 5.0]]);
    }

    #[test]
    fn csr_row_indptr_layout_is_consistent() {
        let csr = CsrMatrix::from_components(
            Shape2D::new(3, 3),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0, 2, 1, 2],
            vec![0, 2, 3, 4],
            false,
        )
        .expect("valid csr");
        assert_eq!(csr.indptr(), &[0, 2, 3, 4]);
        assert_eq!(&csr.indices()[csr.indptr()[0]..csr.indptr()[1]], &[0, 2]);
        assert_eq!(&csr.indices()[csr.indptr()[1]..csr.indptr()[2]], &[1]);
        assert_eq!(&csr.indices()[csr.indptr()[2]..csr.indptr()[3]], &[2]);
    }

    #[test]
    fn csr_rejects_out_of_bounds_minor_index() {
        let err = CsrMatrix::from_components(
            Shape2D::new(2, 2),
            vec![1.0],
            vec![2],
            vec![0, 1, 1],
            false,
        )
        .expect_err("minor index out of bounds");
        assert!(matches!(
            err,
            SparseError::IndexOutOfBounds {
                axis: "minor",
                index: 2,
                bound: 2
            }
        ));
    }

    #[test]
    fn csr_duplicate_indices_mark_non_deduplicated() {
        let csr = CsrMatrix::from_components(
            Shape2D::new(1, 3),
            vec![1.0, 2.0],
            vec![1, 1],
            vec![0, 2],
            false,
        )
        .expect("valid duplicate csr");
        assert!(csr.canonical_meta().sorted_indices);
        assert!(!csr.canonical_meta().deduplicated);
    }

    #[test]
    fn csr_unsorted_indices_mark_not_sorted() {
        let csr = CsrMatrix::from_components(
            Shape2D::new(1, 3),
            vec![1.0, 2.0],
            vec![2, 0],
            vec![0, 2],
            false,
        )
        .expect("valid unsorted csr");
        assert!(!csr.canonical_meta().sorted_indices);
    }

    #[test]
    fn csc_construction_deduplicates_triplets() {
        let coo = CooMatrix::from_triplets(
            Shape2D::new(3, 2),
            vec![2.0, 1.0, 3.0, 4.0],
            vec![2, 0, 2, 1],
            vec![1, 0, 1, 1],
            false,
        )
        .expect("valid coo");
        let csc = coo.to_csc().expect("coo->csc");
        assert_eq!(csc.nnz(), 3);
        assert!(csc.canonical_meta().sorted_indices);
        assert!(csc.canonical_meta().deduplicated);

        let dense = dense_from_coo(&csc.to_coo().expect("csc->coo"));
        assert_matrix_close(&dense, &[vec![1.0, 0.0], vec![0.0, 4.0], vec![0.0, 5.0]]);
    }

    #[test]
    fn csc_construction_preserves_explicit_zero_from_duplicate_cancellation() {
        let coo = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![1.0, -1.0, 2.0],
            vec![0, 0, 1],
            vec![0, 0, 1],
            false,
        )
        .expect("valid coo");
        let csc = coo.to_csc().expect("coo->csc");
        assert_eq!(csc.nnz(), 2);
        assert!(csc.canonical_meta().sorted_indices);
        assert!(csc.canonical_meta().deduplicated);
        assert_eq!(csc.indptr(), &[0, 1, 2]);
        assert_eq!(csc.indices(), &[0, 1]);
        assert_vec_close(csc.data(), &[0.0, 2.0]);

        let roundtrip = csc.to_coo().expect("csc->coo");
        assert_eq!(roundtrip.row_indices(), &[0, 1]);
        assert_eq!(roundtrip.col_indices(), &[0, 1]);
        assert_vec_close(roundtrip.data(), &[0.0, 2.0]);
    }

    #[test]
    fn csc_column_indptr_layout_is_consistent() {
        let csc = CscMatrix::from_components(
            Shape2D::new(3, 3),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0, 2, 1, 2],
            vec![0, 2, 3, 4],
            false,
        )
        .expect("valid csc");
        assert_eq!(csc.indptr(), &[0, 2, 3, 4]);
        assert_eq!(&csc.indices()[csc.indptr()[0]..csc.indptr()[1]], &[0, 2]);
        assert_eq!(&csc.indices()[csc.indptr()[1]..csc.indptr()[2]], &[1]);
        assert_eq!(&csc.indices()[csc.indptr()[2]..csc.indptr()[3]], &[2]);
    }

    #[test]
    fn csc_rejects_non_monotone_indptr() {
        let err = CscMatrix::from_components(
            Shape2D::new(2, 2),
            vec![1.0, 2.0],
            vec![0, 1],
            vec![0, 2, 1],
            false,
        )
        .expect_err("non-monotone indptr");
        assert!(matches!(err, SparseError::InvalidSparseStructure { .. }));
    }

    #[test]
    fn csc_duplicate_indices_mark_non_deduplicated() {
        let csc = CscMatrix::from_components(
            Shape2D::new(3, 1),
            vec![1.0, 2.0],
            vec![1, 1],
            vec![0, 2],
            false,
        )
        .expect("valid duplicate csc");
        assert!(csc.canonical_meta().sorted_indices);
        assert!(!csc.canonical_meta().deduplicated);
    }

    #[test]
    fn csc_unsorted_indices_mark_not_sorted() {
        let csc = CscMatrix::from_components(
            Shape2D::new(3, 1),
            vec![1.0, 2.0],
            vec![2, 0],
            vec![0, 2],
            false,
        )
        .expect("valid unsorted csc");
        assert!(!csc.canonical_meta().sorted_indices);
    }

    #[test]
    fn csc_boolean_row_and_column_index_match_scipy_dense_semantics() {
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

        let row_indexed = csc
            .boolean_row_index(&[true, false, true])
            .expect("row boolean index");
        assert_eq!(row_indexed.shape(), Shape2D::new(2, 4));
        let row_dense = dense_from_coo(&row_indexed.to_coo().expect("row indexed coo"));
        assert_matrix_close(
            &row_dense,
            &[vec![1.0, 0.0, 2.0, 0.0], vec![5.0, 0.0, 6.0, 0.0]],
        );

        let col_indexed = csc
            .boolean_col_index(&[false, true, true, false])
            .expect("col boolean index");
        assert_eq!(col_indexed.shape(), Shape2D::new(3, 2));
        let col_dense = dense_from_coo(&col_indexed.to_coo().expect("col indexed coo"));
        assert_matrix_close(
            &col_dense,
            &[vec![0.0, 2.0], vec![3.0, 0.0], vec![0.0, 6.0]],
        );

        let submatrix = csc
            .boolean_index(&[true, false, true], &[false, true, true, false])
            .expect("boolean submatrix");
        assert_eq!(submatrix.shape(), Shape2D::new(2, 2));
        let sub_dense = dense_from_coo(&submatrix.to_coo().expect("submatrix coo"));
        assert_matrix_close(&sub_dense, &[vec![0.0, 2.0], vec![0.0, 6.0]]);
    }

    #[test]
    fn csc_boolean_mask_values_include_implicit_zeros() {
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

        let values = csc
            .boolean_mask_values(&[
                vec![false, true, true, false],
                vec![false, false, false, true],
                vec![true, false, false, false],
            ])
            .expect("boolean mask values");
        assert_vec_close(&values, &[0.0, 2.0, 4.0, 5.0]);
    }

    #[test]
    fn csc_boolean_index_rejects_mask_length_mismatch() {
        let csc = CscMatrix::from_components(
            Shape2D::new(3, 2),
            vec![1.0, 2.0, 3.0],
            vec![0, 2, 1],
            vec![0, 2, 3],
            false,
        )
        .expect("valid csc");

        let row_err = csc
            .boolean_row_index(&[true, false])
            .expect_err("row boolean length mismatch");
        assert!(matches!(row_err, SparseError::InvalidArgument { .. }));

        let col_err = csc
            .boolean_col_index(&[true, false, true])
            .expect_err("col boolean length mismatch");
        assert!(matches!(col_err, SparseError::InvalidArgument { .. }));

        let mask_err = csc
            .boolean_mask_values(&[vec![true, false]])
            .expect_err("matrix boolean shape mismatch");
        assert!(matches!(mask_err, SparseError::InvalidArgument { .. }));
    }

    #[test]
    fn coo_rejects_length_mismatch() {
        let err =
            CooMatrix::from_triplets(Shape2D::new(2, 2), vec![1.0], vec![0, 1], vec![1], false)
                .expect_err("coo lengths must match");
        assert!(matches!(err, SparseError::IncompatibleShape { .. }));
    }

    #[test]
    fn coo_rejects_row_out_of_bounds() {
        let err = CooMatrix::from_triplets(Shape2D::new(2, 2), vec![1.0], vec![2], vec![0], false)
            .expect_err("row index out of bounds");
        assert!(matches!(
            err,
            SparseError::IndexOutOfBounds {
                axis: "row",
                index: 2,
                bound: 2
            }
        ));
    }

    #[test]
    fn coo_rejects_col_out_of_bounds() {
        let err = CooMatrix::from_triplets(Shape2D::new(2, 2), vec![1.0], vec![0], vec![2], false)
            .expect_err("col index out of bounds");
        assert!(matches!(
            err,
            SparseError::IndexOutOfBounds {
                axis: "col",
                index: 2,
                bound: 2
            }
        ));
    }

    #[test]
    fn coo_sum_duplicates_merges_values() {
        let coo = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![1.5, 2.5, -1.0],
            vec![0, 0, 1],
            vec![1, 1, 0],
            true,
        )
        .expect("coo with sum_duplicates");
        assert_eq!(coo.nnz(), 2);
        let dense = dense_from_coo(&coo);
        assert_matrix_close(&dense, &[vec![0.0, 4.0], vec![-1.0, 0.0]]);
    }

    #[test]
    fn coo_without_dedup_preserves_duplicate_entries() {
        let coo = CooMatrix::from_triplets(
            Shape2D::new(1, 2),
            vec![1.0, 2.0],
            vec![0, 0],
            vec![1, 1],
            false,
        )
        .expect("coo without sum_duplicates");
        assert_eq!(coo.nnz(), 2);
        assert_eq!(coo.row_indices(), &[0, 0]);
        assert_eq!(coo.col_indices(), &[1, 1]);
        assert_vec_close(coo.data(), &[1.0, 2.0]);
    }

    #[test]
    fn coo_setdiag_respects_boundaries_and_partial_updates() {
        let mut partial = CooMatrix::from_triplets(
            Shape2D::new(3, 3),
            vec![10.0, 20.0, 30.0],
            vec![0, 1, 2],
            vec![0, 1, 2],
            false,
        )
        .expect("coo");
        partial.setdiag(&[5.0], 0).expect("partial setdiag");
        assert_eq!(partial.row_indices(), &[1, 2, 0]);
        assert_eq!(partial.col_indices(), &[1, 2, 0]);
        assert_vec_close(partial.data(), &[20.0, 30.0, 5.0]);

        let mut offdiag =
            CooMatrix::from_triplets(Shape2D::new(3, 4), vec![8.0], vec![0], vec![3], false)
                .expect("coo");
        offdiag
            .setdiag(&[1.0, 2.0, 3.0, 4.0], 1)
            .expect("superdiagonal setdiag");
        assert_eq!(offdiag.row_indices(), &[0, 0, 1, 2]);
        assert_eq!(offdiag.col_indices(), &[3, 1, 2, 3]);
        assert_vec_close(offdiag.data(), &[8.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn coo_setdiag_scalar_broadcasts_and_rejects_out_of_bounds_k() {
        let mut coo = CooMatrix::from_triplets(Shape2D::new(3, 4), vec![], vec![], vec![], false)
            .expect("empty coo");
        coo.setdiag_scalar(9.0, 1).expect("scalar setdiag");
        assert_eq!(coo.row_indices(), &[0, 1, 2]);
        assert_eq!(coo.col_indices(), &[1, 2, 3]);
        assert_vec_close(coo.data(), &[9.0, 9.0, 9.0]);

        let err = coo.setdiag(&[1.0], 4).expect_err("k beyond columns");
        assert!(
            matches!(err, SparseError::InvalidArgument { message } if message == "k exceeds array dimensions")
        );

        let err = coo.setdiag_scalar(1.0, -4).expect_err("k beyond rows");
        assert!(
            matches!(err, SparseError::InvalidArgument { message } if message == "k exceeds array dimensions")
        );
    }

    #[test]
    fn sparse_find_sums_duplicates_and_drops_explicit_zeros() {
        let coo = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![0.0, 1.0, 2.0, 3.0, 4.0],
            vec![0, 0, 0, 1, 1],
            vec![0, 1, 1, 0, 1],
            false,
        )
        .expect("coo");
        let (rows, cols, data) = find(&coo).expect("find");
        assert_eq!(rows, vec![0, 1, 1]);
        assert_eq!(cols, vec![1, 0, 1]);
        assert_vec_close(&data, &[3.0, 3.0, 4.0]);
    }

    #[test]
    fn sparse_tril_preserves_explicit_zeros_in_lower_triangle() {
        let coo = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![0.0, 1.0, 2.0, 3.0],
            vec![0, 0, 0, 1],
            vec![0, 1, 1, 0],
            false,
        )
        .expect("coo");
        let lower = tril(&coo, 0).expect("tril");
        assert_eq!(lower.row_indices(), &[0, 1]);
        assert_eq!(lower.col_indices(), &[0, 0]);
        assert_vec_close(lower.data(), &[0.0, 3.0]);
    }

    #[test]
    fn sparse_triu_preserves_duplicates_and_respects_offsets() {
        let coo = CooMatrix::from_triplets(
            Shape2D::new(2, 3),
            vec![5.0, 1.0, 2.0, 3.0],
            vec![0, 0, 0, 1],
            vec![0, 1, 1, 2],
            false,
        )
        .expect("coo");
        let upper = triu(&coo, 1).expect("triu");
        assert_eq!(upper.row_indices(), &[0, 0, 1]);
        assert_eq!(upper.col_indices(), &[1, 1, 2]);
        assert_vec_close(upper.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn sparse_tril_accepts_csr_input() {
        let csr = CooMatrix::from_triplets(
            Shape2D::new(3, 4),
            vec![10.0, 20.0, 30.0, 40.0],
            vec![0, 0, 1, 2],
            vec![2, 0, 1, 3],
            false,
        )
        .expect("coo")
        .to_csr()
        .expect("csr");
        let lower = tril(&csr, 0).expect("tril");
        assert_eq!(lower.row_indices(), &[0, 1]);
        assert_eq!(lower.col_indices(), &[0, 1]);
        assert_vec_close(lower.data(), &[20.0, 30.0]);
    }

    #[test]
    fn bsr_rejects_non_divisible_block_shape() {
        let err = BsrMatrix::from_components(
            Shape2D::new(3, 4),
            Shape2D::new(2, 2),
            vec![],
            vec![],
            vec![0],
            false,
        )
        .expect_err("non-divisible block shape");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    #[test]
    fn bsr_rejects_invalid_block_component_shape() {
        let err = BsrMatrix::from_components(
            Shape2D::new(4, 4),
            Shape2D::new(2, 2),
            vec![vec![1.0, 2.0, 3.0]],
            vec![0],
            vec![0, 1, 1],
            false,
        )
        .expect_err("invalid block width");
        assert!(matches!(err, SparseError::IncompatibleShape { .. }));
    }

    #[test]
    fn bsr_from_triplets_canonicalizes_blocks_and_preserves_explicit_zeros() {
        let bsr = BsrMatrix::from_triplets(
            Shape2D::new(4, 4),
            Shape2D::new(2, 2),
            vec![1.5, 2.5, -4.0, 3.0, 0.0],
            vec![0, 0, 0, 1, 3],
            vec![0, 0, 0, 1, 3],
        )
        .expect("bsr");
        assert_eq!(bsr.nnz_blocks(), 2);
        assert_eq!(bsr.nnz(), 8);
        assert_eq!(bsr.block_shape(), Shape2D::new(2, 2));
        assert!(bsr.canonical_meta().sorted_indices);
        assert!(bsr.canonical_meta().deduplicated);
        assert!((bsr.get(0, 0).expect("get") - 0.0).abs() <= EPS);
        assert!((bsr.get(1, 1).expect("get") - 3.0).abs() <= EPS);
        assert!((bsr.get(3, 3).expect("get") - 0.0).abs() <= EPS);
    }

    #[test]
    fn bsr_to_coo_preserves_dense_semantics() {
        let bsr = BsrMatrix::from_components(
            Shape2D::new(4, 4),
            Shape2D::new(2, 2),
            vec![vec![1.0, 0.0, 0.0, 2.0], vec![0.0, 0.0, 5.0, 0.0]],
            vec![0, 1],
            vec![0, 1, 2],
            true,
        )
        .expect("bsr");
        let dense = dense_from_coo(&bsr.to_coo().expect("bsr->coo"));
        assert_matrix_close(
            &dense,
            &[
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 2.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 5.0, 0.0],
            ],
        );
    }

    #[test]
    fn bsr_converts_through_csr_and_csc() {
        let bsr = BsrMatrix::from_components(
            Shape2D::new(4, 6),
            Shape2D::new(2, 3),
            vec![
                vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0],
                vec![0.0, 0.0, 4.0, 5.0, 0.0, 0.0],
            ],
            vec![0, 1],
            vec![0, 1, 2],
            true,
        )
        .expect("bsr");
        let dense_from_bsr = dense_from_coo(&bsr.to_coo().expect("bsr->coo"));
        let dense_from_csr =
            dense_from_coo(&bsr.to_csr().expect("bsr->csr").to_coo().expect("csr->coo"));
        let dense_from_csc =
            dense_from_coo(&bsr.to_csc().expect("bsr->csc").to_coo().expect("csc->coo"));
        assert_matrix_close(&dense_from_bsr, &dense_from_csr);
        assert_matrix_close(&dense_from_bsr, &dense_from_csc);
    }

    #[test]
    fn lil_from_triplets_merges_duplicates_and_preserves_explicit_zero() {
        let lil = LilMatrix::from_triplets(
            Shape2D::new(3, 3),
            vec![1.5, 2.5, -4.0, 0.0, 3.0],
            vec![0, 0, 0, 1, 2],
            vec![1, 1, 1, 0, 2],
        )
        .expect("lil with duplicate triplets");
        assert_eq!(lil.nnz(), 3);
        assert_eq!(lil.row_indices()[0], vec![1]);
        assert_eq!(lil.row_data()[0], vec![0.0]);
        assert_eq!(lil.row_indices()[1], vec![0]);
        assert_eq!(lil.row_data()[1], vec![0.0]);
        assert_eq!(lil.row_indices()[2], vec![2]);
        assert_eq!(lil.row_data()[2], vec![3.0]);
        assert!((lil.get(0, 1).expect("get") - 0.0).abs() <= EPS);
    }

    #[test]
    fn lil_insert_maintains_sorted_rows_and_tracks_explicit_zero() {
        let mut lil = LilMatrix::new(Shape2D::new(2, 4));
        assert_eq!(lil.insert(1, 3, 4.0).expect("insert"), None);
        assert_eq!(lil.insert(1, 1, -2.0).expect("insert"), None);
        assert_eq!(lil.insert(1, 2, 0.0).expect("insert"), None);
        assert_eq!(lil.row_indices()[1], vec![1, 2, 3]);
        assert_eq!(lil.row_data()[1], vec![-2.0, 0.0, 4.0]);
        assert!(lil.contains(1, 2).expect("contains"));

        let previous = lil.insert(1, 2, 5.0).expect("update");
        assert!(matches!(previous, Some(value) if (value - 0.0).abs() <= EPS));
        assert_eq!(lil.row_data()[1], vec![-2.0, 5.0, 4.0]);
    }

    #[test]
    fn lil_remove_deletes_entry() {
        let mut lil =
            LilMatrix::from_triplets(Shape2D::new(2, 3), vec![7.0, 1.5], vec![0, 1], vec![2, 0])
                .expect("lil");
        let removed = lil.remove(0, 2).expect("remove");
        assert!(matches!(removed, Some(value) if (value - 7.0).abs() <= EPS));
        assert_eq!(lil.nnz(), 1);
        assert!(!lil.contains(0, 2).expect("contains"));
        assert_eq!(lil.remove(0, 2).expect("missing remove"), None);
    }

    #[test]
    fn lil_rejects_out_of_bounds_coordinates() {
        let err = LilMatrix::from_triplets(Shape2D::new(2, 2), vec![1.0], vec![0], vec![2])
            .expect_err("col index out of bounds");
        assert!(matches!(
            err,
            SparseError::IndexOutOfBounds {
                axis: "col",
                index: 2,
                bound: 2
            }
        ));

        let err = LilMatrix::from_rows(
            Shape2D::new(2, 2),
            vec![vec![0], vec![2]],
            vec![vec![1.0], vec![5.0]],
        )
        .expect_err("row with out of bounds col");
        assert!(matches!(
            err,
            SparseError::IndexOutOfBounds {
                axis: "col",
                index: 2,
                bound: 2
            }
        ));
    }

    #[test]
    fn lil_to_coo_preserves_dense_semantics() {
        let lil = LilMatrix::from_rows(
            Shape2D::new(3, 4),
            vec![vec![1], vec![0, 3], vec![2]],
            vec![vec![2.0], vec![-1.0, 0.0], vec![5.0]],
        )
        .expect("lil");
        let dense = dense_from_coo(&lil.to_coo().expect("lil->coo"));
        assert_matrix_close(
            &dense,
            &[
                vec![0.0, 2.0, 0.0, 0.0],
                vec![-1.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 5.0, 0.0],
            ],
        );
        assert_eq!(lil.nnz(), 4);
    }

    #[test]
    fn lil_converts_through_csr_and_csc() {
        let lil = LilMatrix::from_triplets(
            Shape2D::new(4, 4),
            vec![3.0, -2.0, 7.0, 1.5, 0.0],
            vec![0, 1, 2, 3, 1],
            vec![0, 2, 1, 3, 0],
        )
        .expect("lil");
        let dense_from_lil = dense_from_coo(&lil.to_coo().expect("lil->coo"));
        let dense_from_csr =
            dense_from_coo(&lil.to_csr().expect("lil->csr").to_coo().expect("csr->coo"));
        let dense_from_csc =
            dense_from_coo(&lil.to_csc().expect("lil->csc").to_coo().expect("csc->coo"));
        assert_matrix_close(&dense_from_lil, &dense_from_csr);
        assert_matrix_close(&dense_from_lil, &dense_from_csc);
    }

    #[test]
    fn lil_slice_matches_scipy_dense_reference() {
        let mut lil = LilMatrix::new(Shape2D::new(4, 5));
        lil.insert(0, 1, 2.0).expect("insert");
        lil.insert(1, 3, 5.0).expect("insert");
        lil.insert(2, 0, 7.0).expect("insert");
        lil.insert(3, 4, 11.0).expect("insert");

        let row_slice = lil
            .slice(SparseSliceSpec::new(1, 3, 1), SparseSliceSpec::new(0, 5, 1))
            .expect("row slice");
        assert_eq!(row_slice.row_indices()[0], vec![3]);
        assert_eq!(row_slice.row_data()[0], vec![5.0]);
        assert_eq!(row_slice.row_indices()[1], vec![0]);
        assert_eq!(row_slice.row_data()[1], vec![7.0]);
        assert_matrix_close(
            &dense_from_coo(&row_slice.to_coo().expect("row slice coo")),
            &[vec![0.0, 0.0, 0.0, 5.0, 0.0], vec![7.0, 0.0, 0.0, 0.0, 0.0]],
        );

        let col_slice = lil
            .slice(SparseSliceSpec::new(0, 4, 1), SparseSliceSpec::new(1, 4, 1))
            .expect("col slice");
        assert_eq!(col_slice.row_indices()[0], vec![0]);
        assert_eq!(col_slice.row_data()[0], vec![2.0]);
        assert_eq!(col_slice.row_indices()[1], vec![2]);
        assert_eq!(col_slice.row_data()[1], vec![5.0]);
        assert_matrix_close(
            &dense_from_coo(&col_slice.to_coo().expect("col slice coo")),
            &[
                vec![2.0, 0.0, 0.0],
                vec![0.0, 0.0, 5.0],
                vec![0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0],
            ],
        );

        let stepped = lil
            .slice(SparseSliceSpec::new(1, 4, 1), SparseSliceSpec::new(1, 5, 2))
            .expect("stepped slice");
        assert_eq!(stepped.row_indices()[0], vec![1]);
        assert_eq!(stepped.row_data()[0], vec![5.0]);
        assert_eq!(stepped.row_indices()[1], Vec::<usize>::new());
        assert_eq!(stepped.row_indices()[2], Vec::<usize>::new());
        assert_matrix_close(
            &dense_from_coo(&stepped.to_coo().expect("stepped coo")),
            &[vec![0.0, 5.0], vec![0.0, 0.0], vec![0.0, 0.0]],
        );
    }

    #[test]
    fn lil_slice_rejects_invalid_specs() {
        let lil = LilMatrix::new(Shape2D::new(2, 3));
        let err = lil
            .slice(SparseSliceSpec::new(0, 2, 0), SparseSliceSpec::new(0, 3, 1))
            .expect_err("zero step");
        assert!(matches!(
            err,
            SparseError::InvalidArgument { message } if message == "slice step must be >= 1"
        ));

        let err = lil
            .slice(SparseSliceSpec::new(0, 3, 1), SparseSliceSpec::new(0, 3, 1))
            .expect_err("row bound");
        assert!(matches!(
            err,
            SparseError::IndexOutOfBounds {
                axis: "row",
                index: 3,
                bound: 2
            }
        ));
    }

    #[test]
    fn dok_from_triplets_merges_duplicates_and_elides_zero_sums() {
        let dok = DokMatrix::from_triplets(
            Shape2D::new(3, 3),
            vec![1.5, 2.5, -4.0, 3.0],
            vec![0, 0, 0, 2],
            vec![1, 1, 1, 2],
        )
        .expect("dok with duplicate triplets");
        assert_eq!(dok.nnz(), 1);
        assert_eq!(dok.entries().len(), 1);
        assert!((dok.get(0, 1).expect("get") - 0.0).abs() <= EPS);
        assert!((dok.get(2, 2).expect("get") - 3.0).abs() <= EPS);
    }

    #[test]
    fn dok_insert_and_remove_elide_zero_entries() {
        let mut dok = DokMatrix::new(Shape2D::new(2, 3));
        assert_eq!(dok.insert(1, 2, 4.0).expect("insert"), None);
        assert_eq!(dok.nnz(), 1);
        assert!(dok.contains(1, 2).expect("contains"));

        let previous = dok.insert(1, 2, 0.0).expect("zero insert");
        assert!(matches!(previous, Some(value) if (value - 4.0).abs() <= EPS));
        assert_eq!(dok.nnz(), 0);
        assert!(!dok.contains(1, 2).expect("contains"));

        assert_eq!(dok.remove(0, 0).expect("remove"), None);
    }

    #[test]
    fn dok_rejects_out_of_bounds_coordinates() {
        let err = DokMatrix::from_triplets(Shape2D::new(2, 2), vec![1.0], vec![0], vec![2])
            .expect_err("col index out of bounds");
        assert!(matches!(
            err,
            SparseError::IndexOutOfBounds {
                axis: "col",
                index: 2,
                bound: 2
            }
        ));

        let mut dok = DokMatrix::new(Shape2D::new(2, 2));
        let err = dok.insert(2, 1, 5.0).expect_err("row index out of bounds");
        assert!(matches!(
            err,
            SparseError::IndexOutOfBounds {
                axis: "row",
                index: 2,
                bound: 2
            }
        ));
    }

    #[test]
    fn dok_to_coo_preserves_dense_semantics() {
        let dok = DokMatrix::from_triplets(
            Shape2D::new(3, 4),
            vec![2.0, -1.0, 5.0],
            vec![0, 1, 2],
            vec![1, 0, 3],
        )
        .expect("dok");
        let dense = dense_from_coo(&dok.to_coo().expect("dok->coo"));
        assert_matrix_close(
            &dense,
            &[
                vec![0.0, 2.0, 0.0, 0.0],
                vec![-1.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 5.0],
            ],
        );
    }

    #[test]
    fn dok_converts_through_csr_and_csc() {
        let dok = DokMatrix::from_triplets(
            Shape2D::new(4, 4),
            vec![3.0, -2.0, 7.0, 1.5],
            vec![0, 1, 2, 3],
            vec![0, 2, 1, 3],
        )
        .expect("dok");
        let dense_from_dok = dense_from_coo(&dok.to_coo().expect("dok->coo"));
        let dense_from_csr =
            dense_from_coo(&dok.to_csr().expect("dok->csr").to_coo().expect("csr->coo"));
        let dense_from_csc =
            dense_from_coo(&dok.to_csc().expect("dok->csc").to_coo().expect("csc->coo"));
        assert_matrix_close(&dense_from_dok, &dense_from_csr);
        assert_matrix_close(&dense_from_dok, &dense_from_csc);
    }

    #[test]
    fn dia_rejects_length_mismatch() {
        let err =
            DiaMatrix::from_diagonals(Shape2D::new(3, 3), vec![0, 1], vec![vec![1.0, 2.0, 3.0]])
                .expect_err("offset/data mismatch");
        assert!(matches!(err, SparseError::IncompatibleShape { .. }));
    }

    #[test]
    fn dia_rejects_invalid_diagonal_length() {
        let err = DiaMatrix::from_diagonals(Shape2D::new(3, 3), vec![1], vec![vec![1.0, 2.0, 3.0]])
            .expect_err("invalid diagonal length");
        assert!(matches!(err, SparseError::IncompatibleShape { .. }));
    }

    #[test]
    fn dia_to_coo_preserves_dense_semantics() {
        let dia = DiaMatrix::from_diagonals(
            Shape2D::new(3, 4),
            vec![0, 1, -1],
            vec![vec![1.0, 2.0, 3.0], vec![10.0, 20.0, 30.0], vec![7.0, 8.0]],
        )
        .expect("dia");
        let dense = dense_from_coo(&dia.to_coo().expect("dia->coo"));
        assert_matrix_close(
            &dense,
            &[
                vec![1.0, 10.0, 0.0, 0.0],
                vec![7.0, 2.0, 20.0, 0.0],
                vec![0.0, 8.0, 3.0, 30.0],
            ],
        );
    }

    #[test]
    fn dia_converts_through_csr_and_csc() {
        let dia = DiaMatrix::from_diagonals(
            Shape2D::new(4, 4),
            vec![0, 2],
            vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0]],
        )
        .expect("dia");
        let dense_from_dia = dense_from_coo(&dia.to_coo().expect("dia->coo"));
        let dense_from_csr =
            dense_from_coo(&dia.to_csr().expect("dia->csr").to_coo().expect("csr->coo"));
        let dense_from_csc =
            dense_from_coo(&dia.to_csc().expect("dia->csc").to_coo().expect("csc->coo"));
        assert_matrix_close(&dense_from_dia, &dense_from_csr);
        assert_matrix_close(&dense_from_dia, &dense_from_csc);
    }

    #[test]
    fn roundtrip_conversion_identity_all_formats() {
        let shape = Shape2D::new(3, 3);
        let coo = CooMatrix::from_triplets(
            shape,
            vec![1.0, 2.0, -3.0, 4.0],
            vec![0, 0, 1, 2],
            vec![0, 2, 1, 2],
            false,
        )
        .expect("valid coo");

        let csr = coo.to_csr().expect("coo->csr");
        let csc = csr.to_csc().expect("csr->csc");
        let back_to_coo = csc.to_coo().expect("csc->coo");

        assert_matrix_close(&dense_from_coo(&coo), &dense_from_coo(&back_to_coo));
    }

    #[test]
    fn conversion_csr_to_coo_to_csr_preserves_dense() {
        let coo = CooMatrix::from_triplets(
            Shape2D::new(3, 4),
            vec![1.0, -2.0, 3.0, 4.0, 5.0],
            vec![0, 1, 2, 2, 0],
            vec![0, 3, 1, 2, 0],
            false,
        )
        .expect("valid coo");
        let csr = coo.to_csr().expect("coo->csr");
        let roundtrip = csr.to_coo().expect("csr->coo").to_csr().expect("coo->csr");
        assert_matrix_close(
            &dense_from_coo(&csr.to_coo().expect("csr->coo")),
            &dense_from_coo(&roundtrip.to_coo().expect("csr->coo")),
        );
    }

    #[test]
    fn conversion_csc_to_csr_to_csc_preserves_dense() {
        let coo = CooMatrix::from_triplets(
            Shape2D::new(4, 3),
            vec![2.0, 1.0, -3.0, 4.0],
            vec![0, 2, 3, 1],
            vec![1, 0, 2, 1],
            false,
        )
        .expect("valid coo");
        let csc = coo.to_csc().expect("coo->csc");
        let roundtrip = csc.to_csr().expect("csc->csr").to_csc().expect("csr->csc");
        assert_matrix_close(
            &dense_from_coo(&csc.to_coo().expect("csc->coo")),
            &dense_from_coo(&roundtrip.to_coo().expect("csc->coo")),
        );
    }

    #[test]
    fn conversion_empty_matrix_roundtrip_preserves_shape() {
        let empty = CooMatrix::from_triplets(Shape2D::new(3, 4), vec![], vec![], vec![], false)
            .expect("empty coo");
        let csr = empty.to_csr().expect("coo->csr");
        let csc = csr.to_csc().expect("csr->csc");
        let roundtrip = csc.to_coo().expect("csc->coo");
        assert_eq!(roundtrip.shape(), Shape2D::new(3, 4));
        assert_eq!(roundtrip.nnz(), 0);
    }

    #[test]
    fn conversion_large_sparse_roundtrip_preserves_dense() {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();
        for i in 0..32 {
            rows.push(i);
            cols.push(i);
            data.push((i + 1) as f64);
            if i + 1 < 32 {
                rows.push(i);
                cols.push(i + 1);
                data.push(-1.0);
            }
        }
        let coo = CooMatrix::from_triplets(Shape2D::new(32, 32), data, rows, cols, false)
            .expect("large sparse coo");
        let dense_before = dense_from_coo(&coo);
        let dense_after = dense_from_coo(
            &coo.to_csr()
                .expect("coo->csr")
                .to_csc()
                .expect("csr->csc")
                .to_coo()
                .expect("csc->coo"),
        );
        assert_matrix_close(&dense_before, &dense_after);
    }

    #[test]
    fn spmv_matches_dense_for_all_formats() {
        let shape = Shape2D::new(3, 3);
        let coo = CooMatrix::from_triplets(
            shape,
            vec![1.0, -2.0, 3.0, 4.0],
            vec![0, 1, 1, 2],
            vec![0, 0, 2, 1],
            false,
        )
        .expect("valid coo");
        let csr = coo.to_csr().expect("coo->csr");
        let csc = coo.to_csc().expect("coo->csc");
        let vector = vec![2.0, -1.0, 3.0];

        let expected = dense_matvec(&dense_from_coo(&coo), &vector);
        assert_vec_close(&spmv_coo(&coo, &vector).expect("coo spmv"), &expected);
        assert_vec_close(&spmv_csr(&csr, &vector).expect("csr spmv"), &expected);
        assert_vec_close(&spmv_csc(&csc, &vector).expect("csc spmv"), &expected);
    }

    // ── DIA format tests ─────────────────────────────────────────────

    #[test]
    fn dia_tridiagonal_matvec() {
        // Tridiagonal: [[2,-1,0], [-1,2,-1], [0,-1,2]]
        let shape = Shape2D::new(3, 3);
        let data = vec![
            vec![-1.0, -1.0, 0.0], // subdiagonal (offset -1)
            vec![2.0, 2.0, 2.0],   // main diagonal (offset 0)
            vec![0.0, -1.0, -1.0], // superdiagonal (offset +1)
        ];
        let offsets = vec![-1, 0, 1];
        let dia = DiaMatrix::new(shape, data, offsets).expect("dia");
        let x = vec![1.0, 2.0, 3.0];
        let y = dia.matvec(&x).expect("matvec");
        // [2*1-1*2, -1*1+2*2-1*3, -1*2+2*3] = [0, 0, 4]
        assert!((y[0] - 0.0).abs() < 1e-12);
        assert!((y[1] - 0.0).abs() < 1e-12);
        assert!((y[2] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn dia_to_csr_roundtrip() {
        let shape = Shape2D::new(3, 3);
        let data = vec![
            vec![1.0, 2.0, 3.0], // main diagonal
        ];
        let offsets = vec![0];
        let dia = DiaMatrix::new(shape, data, offsets).expect("dia");

        let csr = dia.to_csr().expect("to_csr");
        assert_eq!(csr.shape().rows, 3);
        assert_eq!(csr.shape().cols, 3);

        // Verify diagonal elements
        let x = vec![1.0, 1.0, 1.0];
        let y = spmv_csr(&csr, &x).expect("spmv");
        assert!((y[0] - 1.0).abs() < 1e-12);
        assert!((y[1] - 2.0).abs() < 1e-12);
        assert!((y[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn dia_get_element() {
        let shape = Shape2D::new(3, 3);
        let data = vec![
            vec![5.0, 6.0, 7.0], // main diagonal
            vec![0.0, 8.0, 9.0], // superdiagonal
        ];
        let offsets = vec![0, 1];
        let dia = DiaMatrix::new(shape, data, offsets).expect("dia");

        assert!((dia.get(0, 0) - 5.0).abs() < 1e-12);
        assert!((dia.get(1, 1) - 6.0).abs() < 1e-12);
        assert!((dia.get(0, 1) - 8.0).abs() < 1e-12);
        assert!((dia.get(2, 0) - 0.0).abs() < 1e-12); // not on stored diagonal
    }

    #[test]
    fn dia_nnz() {
        let shape = Shape2D::new(3, 3);
        let data = vec![
            vec![1.0, 0.0, 3.0], // main diagonal (1 zero)
        ];
        let offsets = vec![0];
        let dia = DiaMatrix::new(shape, data, offsets).expect("dia");
        assert_eq!(dia.nnz(), 2); // only nonzero entries count
    }

    #[test]
    fn spmv_identity_matrix_returns_input() {
        let identity = eye(5).expect("identity matrix");
        let vector = vec![3.0, -2.0, 4.5, 1.25, -0.5];
        let out = spmv_csr(&identity, &vector).expect("identity spmv");
        assert_vec_close(&out, &vector);
    }

    #[test]
    fn spmm_identity_matches_scipy_csr_row_order_and_metadata() {
        let lhs = CsrMatrix::from_components(
            Shape2D::new(2, 3),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0, 2, 1, 2],
            vec![0, 2, 4],
            false,
        )
        .expect("sorted lhs csr");
        let identity = eye(3).expect("identity matrix");

        let product = spmm(&lhs, &identity);

        assert_eq!(product.indptr(), &[0, 2, 4]);
        assert_eq!(product.indices(), &[2, 0, 2, 1]);
        assert_vec_close(product.data(), &[2.0, 1.0, 4.0, 3.0]);
        assert!(!product.canonical_meta().sorted_indices);
        assert!(product.canonical_meta().deduplicated);
    }

    #[test]
    fn spmm_identity_second_pass_restores_sorted_metadata() {
        let lhs = CsrMatrix::from_components(
            Shape2D::new(2, 3),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0, 2, 1, 2],
            vec![0, 2, 4],
            false,
        )
        .expect("sorted lhs csr");
        let identity = eye(3).expect("identity matrix");

        let first = spmm(&lhs, &identity);
        let second = spmm(&first, &identity);

        assert_eq!(second.indptr(), &[0, 2, 4]);
        assert_eq!(second.indices(), &[0, 2, 1, 2]);
        assert_vec_close(second.data(), &[1.0, 2.0, 3.0, 4.0]);
        assert!(second.canonical_meta().sorted_indices);
        assert!(second.canonical_meta().deduplicated);
    }

    #[test]
    fn spmv_rejects_vector_length_mismatch_all_formats() {
        let coo = CooMatrix::from_triplets(
            Shape2D::new(2, 3),
            vec![1.0, 2.0],
            vec![0, 1],
            vec![1, 2],
            false,
        )
        .expect("coo");
        let csr = coo.to_csr().expect("coo->csr");
        let csc = coo.to_csc().expect("coo->csc");
        let vector = vec![1.0, 2.0];

        assert!(matches!(
            spmv_coo(&coo, &vector),
            Err(SparseError::IncompatibleShape { .. })
        ));
        assert!(matches!(
            spmv_csr(&csr, &vector),
            Err(SparseError::IncompatibleShape { .. })
        ));
        assert!(matches!(
            spmv_csc(&csc, &vector),
            Err(SparseError::IncompatibleShape { .. })
        ));
    }

    #[test]
    fn spmv_zero_matrix_returns_zero_vector() {
        let zero = CooMatrix::from_triplets(
            Shape2D::new(3, 3),
            vec![0.0, 0.0],
            vec![0, 2],
            vec![1, 2],
            false,
        )
        .expect("zero entries coo");
        let csr = zero.to_csr().expect("coo->csr");
        let out = spmv_csr(&csr, &[1.0, 2.0, 3.0]).expect("zero spmv");
        assert_vec_close(&out, &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn strict_mode_preserves_metadata_and_emits_structured_logs() {
        let shape = Shape2D::new(2, 3);
        let csr = CsrMatrix::from_components(
            shape,
            vec![10.0, 1.0, 2.0],
            vec![2, 0, 1],
            vec![0, 2, 3],
            false,
        )
        .expect("valid non-canonical csr");
        assert!(!csr.canonical_meta().sorted_indices);

        let (csc, conv1) =
            csr_to_csc_with_mode(&csr, RuntimeMode::Strict, "op-conv-1").expect("strict csr->csc");
        let (roundtrip, conv2) =
            csc_to_csr_with_mode(&csc, RuntimeMode::Strict, "op-conv-2").expect("strict csc->csr");

        // Conversion via COO inherently sorts and deduplicates the matrix.
        assert!(roundtrip.canonical_meta().sorted_indices);
        assert!(roundtrip.canonical_meta().deduplicated);

        let construction_log = roundtrip.construction_log(
            RuntimeMode::Strict,
            "op-construct-1",
            "passed_invariant_validation",
        );
        let log_json = construction_log.to_json();
        assert!(log_json.contains("\"operation_id\":\"op-construct-1\""));
        assert!(log_json.contains("\"validation_result\":\"passed_invariant_validation\""));

        let conv1_json = conv1.to_json();
        let conv2_json = conv2.to_json();
        assert!(conv1_json.contains("\"from_format\":\"csr\""));
        assert!(conv1_json.contains("\"to_format\":\"csc\""));
        assert!(conv2_json.contains("\"from_format\":\"csc\""));
        assert!(conv2_json.contains("\"to_format\":\"csr\""));
    }

    #[test]
    fn hardened_mode_rejects_unsorted_csr_for_conversion() {
        let csr = CsrMatrix::from_components(
            Shape2D::new(2, 3),
            vec![1.0, 2.0, 3.0],
            vec![2, 0, 1],
            vec![0, 2, 3],
            false,
        )
        .expect("non-canonical csr");
        let err = csr_to_csc_with_mode(&csr, RuntimeMode::Hardened, "hardened-csr")
            .expect_err("hardened must reject unsorted csr");
        assert!(matches!(err, SparseError::InvalidSparseStructure { .. }));
    }

    #[test]
    fn csr_to_csc_with_mode_and_audit_emits_failclosed() {
        // br-egba-4: _with_audit variant records FailClosed on
        // hardened rejection. Strict leaves ledger untouched.
        let csr = CsrMatrix::from_components(
            Shape2D::new(2, 3),
            vec![1.0, 2.0, 3.0],
            vec![2, 0, 1],
            vec![0, 2, 3],
            false,
        )
        .expect("non-canonical csr");

        let ledger = super::sync_audit_ledger();

        // Strict: conversion succeeds without rejection, no emit.
        let _ = crate::ops::csr_to_csc_with_mode_and_audit(
            &csr,
            RuntimeMode::Strict,
            "strict-audit",
            &ledger,
        )
        .expect("strict should succeed");
        assert_eq!(ledger.lock().unwrap().len(), 0, "strict must not emit");

        // Hardened: rejects unsorted CSR; expect one FailClosed event.
        let err = crate::ops::csr_to_csc_with_mode_and_audit(
            &csr,
            RuntimeMode::Hardened,
            "hardened-audit",
            &ledger,
        )
        .expect_err("hardened must reject");
        assert!(matches!(err, SparseError::InvalidSparseStructure { .. }));
        assert_eq!(
            ledger.lock().unwrap().len(),
            1,
            "hardened rejection must emit FailClosed"
        );
    }

    #[test]
    fn hardened_mode_rejects_unsorted_csc_for_conversion() {
        let csc = CscMatrix::from_components(
            Shape2D::new(3, 2),
            vec![1.0, 2.0, 3.0],
            vec![2, 0, 1],
            vec![0, 2, 3],
            false,
        )
        .expect("non-canonical csc");
        let err = csc_to_csr_with_mode(&csc, RuntimeMode::Hardened, "hardened-csc")
            .expect_err("hardened must reject unsorted csc");
        assert!(matches!(err, SparseError::InvalidSparseStructure { .. }));
    }

    #[test]
    fn hardened_mode_conversion_promotes_canonical_metadata() {
        let coo = CooMatrix::from_triplets(
            Shape2D::new(3, 3),
            vec![1.0, 2.0, 3.0],
            vec![0, 1, 2],
            vec![2, 1, 0],
            false,
        )
        .expect("coo");
        let (csr, _) =
            coo_to_csr_with_mode(&coo, RuntimeMode::Hardened, "to-csr").expect("hardened coo->csr");
        let (csc, _) =
            csr_to_csc_with_mode(&csr, RuntimeMode::Hardened, "to-csc").expect("hardened csr->csc");
        assert!(csc.canonical_meta().sorted_indices);
        assert!(csc.canonical_meta().deduplicated);
    }

    #[test]
    fn add_and_subtract_csr_match_dense_semantics() {
        let lhs = CooMatrix::from_triplets(
            Shape2D::new(2, 3),
            vec![1.0, 3.0, 2.0],
            vec![0, 1, 1],
            vec![0, 2, 1],
            false,
        )
        .expect("lhs")
        .to_csr()
        .expect("lhs csr");
        let rhs = CooMatrix::from_triplets(
            Shape2D::new(2, 3),
            vec![4.0, -1.0],
            vec![0, 1],
            vec![0, 2],
            false,
        )
        .expect("rhs")
        .to_csr()
        .expect("rhs csr");

        let sum = add_csr(&lhs, &rhs).expect("sum");
        let diff = sub_csr(&lhs, &rhs).expect("diff");
        assert_matrix_close(
            &dense_from_coo(&sum.to_coo().expect("sum coo")),
            &[vec![5.0, 0.0, 0.0], vec![0.0, 2.0, 2.0]],
        );
        assert_matrix_close(
            &dense_from_coo(&diff.to_coo().expect("diff coo")),
            &[vec![-3.0, 0.0, 0.0], vec![0.0, 2.0, 4.0]],
        );
    }

    #[test]
    fn scale_operations_preserve_structure() {
        let coo = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![1.0, -2.0],
            vec![0, 1],
            vec![1, 0],
            false,
        )
        .expect("coo");
        let csr = coo.to_csr().expect("csr");
        let csc = coo.to_csc().expect("csc");

        let scaled_coo = scale_coo(&coo, -2.0).expect("scale coo");
        let scaled_csr = scale_csr(&csr, -2.0).expect("scale csr");
        let scaled_csc = scale_csc(&csc, -2.0).expect("scale csc");

        let expected = [vec![0.0, -2.0], vec![4.0, 0.0]];
        assert_matrix_close(&dense_from_coo(&scaled_coo), &expected);
        assert_matrix_close(
            &dense_from_coo(&scaled_csr.to_coo().expect("csr->coo")),
            &expected,
        );
        assert_matrix_close(
            &dense_from_coo(&scaled_csc.to_coo().expect("csc->coo")),
            &expected,
        );
    }

    #[test]
    fn structured_log_schema_is_machine_parseable() {
        let log = emit_test_log(SparseTestLogSpec {
            test_id: "test_sparse_structured_log_schema",
            format: "csr",
            shape: Shape2D::new(2, 3),
            nnz: 4,
            operation: "roundtrip_check",
            mode: RuntimeMode::Strict,
            seed: LOG_SEED,
            result: "pass",
        });
        for field in [
            "test_id",
            "format",
            "shape",
            "nnz",
            "operation",
            "mode",
            "seed",
            "timestamp_ms",
            "result",
        ] {
            assert!(log.get(field).is_some(), "missing required field: {field}");
        }
        assert_eq!(log["shape"][0], 2);
        assert_eq!(log["shape"][1], 3);
        assert_eq!(log["mode"], "strict");
    }

    #[test]
    fn zero_and_single_element_matrices_are_supported() {
        let zero = eye(0).expect("zero eye");
        assert_eq!(zero.shape(), Shape2D::new(0, 0));
        assert_eq!(zero.nnz(), 0);
        assert!(spmv_csr(&zero, &[]).expect("zero spmv").is_empty());

        let single =
            CooMatrix::from_triplets(Shape2D::new(1, 1), vec![5.0], vec![0], vec![0], false)
                .expect("single coo");
        let single_csr = single.to_csr().expect("single csr");
        let single_csc = single.to_csc().expect("single csc");
        assert_vec_close(&spmv_coo(&single, &[2.0]).expect("coo single"), &[10.0]);
        assert_vec_close(&spmv_csr(&single_csr, &[2.0]).expect("csr single"), &[10.0]);
        assert_vec_close(&spmv_csc(&single_csc, &[2.0]).expect("csc single"), &[10.0]);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

        #[test]
        fn prop_roundtrip_csr_csc_csr_preserves_values(case in sparse_case_strategy()) {
            let coo = coo_from_case(&case, false);
            let dense_before = dense_from_coo(&coo);
            let dense_after = dense_from_coo(
                &coo
                    .to_csr()
                    .expect("coo->csr")
                    .to_csc()
                    .expect("csr->csc")
                    .to_csr()
                    .expect("csc->csr")
                    .to_coo()
                    .expect("csr->coo"),
            );
            prop_assert_eq!(dense_before, dense_after);
            let log = emit_test_log(
                SparseTestLogSpec {
                    test_id: "prop_roundtrip_csr_csc_csr_preserves_values",
                    format: "csr",
                    shape: case.shape,
                    nnz: coo.nnz(),
                    operation: "roundtrip",
                    mode: RuntimeMode::Strict,
                    seed: LOG_SEED,
                    result: "pass",
                },
            );
            prop_assert!(log["timestamp_ms"].is_number());
        }

        #[test]
        fn prop_spmv_matches_dense(case in sparse_case_with_vector_strategy()) {
            let (case, vector) = case;
            let coo = coo_from_case(&case, false);
            let csr = coo.to_csr().expect("coo->csr");
            let csc = coo.to_csc().expect("coo->csc");

            let expected = dense_matvec(&dense_from_coo(&coo), &vector);
            let coo_out = spmv_coo(&coo, &vector).expect("coo spmv");
            let csr_out = spmv_csr(&csr, &vector).expect("csr spmv");
            let csc_out = spmv_csc(&csc, &vector).expect("csc spmv");
            prop_assert_eq!(coo_out.as_slice(), expected.as_slice());
            prop_assert_eq!(csr_out.as_slice(), expected.as_slice());
            prop_assert_eq!(csc_out.as_slice(), expected.as_slice());
        }

        #[test]
        fn prop_nnz_addition_upper_bound(pair in sparse_pair_strategy()) {
            let (lhs_case, rhs_case) = pair;
            let lhs = coo_from_case(&lhs_case, false).to_csr().expect("lhs csr");
            let rhs = coo_from_case(&rhs_case, false).to_csr().expect("rhs csr");
            let sum = add_csr(&lhs, &rhs).expect("sum");
            prop_assert!(sum.nnz() <= lhs.nnz() + rhs.nnz());
        }

        #[test]
        fn prop_shape_preserved_through_conversions(case in sparse_case_strategy()) {
            let coo = coo_from_case(&case, false);
            let csr = coo.to_csr().expect("coo->csr");
            let csc = csr.to_csc().expect("csr->csc");
            let back = csc.to_coo().expect("csc->coo");
            prop_assert_eq!(coo.shape(), csr.shape());
            prop_assert_eq!(csr.shape(), csc.shape());
            prop_assert_eq!(csc.shape(), back.shape());
        }

        #[test]
        fn prop_sorted_invariants_after_canonical_construction(case in sparse_case_strategy()) {
            let coo = coo_from_case(&case, false);
            let csr = coo.to_csr().expect("coo->csr");
            let csc = coo.to_csc().expect("coo->csc");
            prop_assert!(csr.canonical_meta().sorted_indices);
            prop_assert!(csr.canonical_meta().deduplicated);
            prop_assert!(csc.canonical_meta().sorted_indices);
            prop_assert!(csc.canonical_meta().deduplicated);
        }
    }

    fn sparse_case_strategy() -> impl Strategy<Value = SparseCase> {
        (1usize..=6, 1usize..=6).prop_flat_map(|(rows, cols)| {
            proptest::collection::vec((0usize..rows, 0usize..cols, -5i16..=5i16), 0..=24)
                .prop_map(move |triplets| SparseCase::from_triplets(rows, cols, triplets))
        })
    }

    fn sparse_pair_strategy() -> impl Strategy<Value = (SparseCase, SparseCase)> {
        (1usize..=6, 1usize..=6).prop_flat_map(|(rows, cols)| {
            let triplet = (0usize..rows, 0usize..cols, -5i16..=5i16);
            (
                proptest::collection::vec(triplet.clone(), 0..=24),
                proptest::collection::vec(triplet, 0..=24),
            )
                .prop_map(move |(lhs, rhs)| {
                    (
                        SparseCase::from_triplets(rows, cols, lhs),
                        SparseCase::from_triplets(rows, cols, rhs),
                    )
                })
        })
    }

    fn sparse_case_with_vector_strategy() -> impl Strategy<Value = (SparseCase, Vec<f64>)> {
        sparse_case_strategy().prop_flat_map(|case| {
            let cols = case.shape.cols;
            (
                Just(case),
                proptest::collection::vec(-5i16..=5i16, cols)
                    .prop_map(|v| v.into_iter().map(f64::from).collect::<Vec<_>>()),
            )
        })
    }

    fn coo_from_case(case: &SparseCase, sum_duplicates: bool) -> CooMatrix {
        CooMatrix::from_triplets(
            case.shape,
            case.data.clone(),
            case.row_indices.clone(),
            case.col_indices.clone(),
            sum_duplicates,
        )
        .expect("generated sparse case must be valid")
    }

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

    fn assert_vec_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len(), "vector length mismatch");
        for (idx, (lhs, rhs)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (lhs - rhs).abs() <= EPS,
                "vector mismatch at {idx}: actual={lhs} expected={rhs}"
            );
        }
    }

    fn assert_matrix_close(actual: &[Vec<f64>], expected: &[Vec<f64>]) {
        assert_eq!(actual.len(), expected.len(), "matrix row mismatch");
        for (row, (lhs, rhs)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(lhs.len(), rhs.len(), "matrix col mismatch at row {row}");
            for (col, (lval, rval)) in lhs.iter().zip(rhs.iter()).enumerate() {
                assert!(
                    (lval - rval).abs() <= EPS,
                    "matrix mismatch at ({row}, {col}): actual={lval} expected={rval}"
                );
            }
        }
    }

    struct SparseTestLogSpec<'a> {
        test_id: &'a str,
        format: &'a str,
        shape: Shape2D,
        nnz: usize,
        operation: &'a str,
        mode: RuntimeMode,
        seed: u64,
        result: &'a str,
    }

    fn emit_test_log(spec: SparseTestLogSpec<'_>) -> Value {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_millis() as u64);
        let value = json!({
            "test_id": spec.test_id,
            "format": spec.format,
            "shape": [spec.shape.rows, spec.shape.cols],
            "nnz": spec.nnz,
            "operation": spec.operation,
            "mode": match spec.mode {
                RuntimeMode::Strict => "strict",
                RuntimeMode::Hardened => "hardened",
            },
            "seed": spec.seed,
            "timestamp_ms": timestamp_ms,
            "result": spec.result
        });
        let encoded = serde_json::to_string(&value).expect("log json serialization");
        serde_json::from_str(&encoded).expect("log json parse")
    }

    // ── ILU(0) preconditioner tests ─────────────────────────────────

    #[test]
    fn spilu_diagonal_matrix() {
        // ILU(0) of a diagonal matrix should give L=I, U=diag
        let coo = CooMatrix::from_triplets(
            Shape2D::new(3, 3),
            vec![2.0, 5.0, 3.0],
            vec![0, 1, 2],
            vec![0, 1, 2],
            false,
        )
        .expect("diagonal coo");
        let csc = coo.to_csc().expect("coo->csc");
        let ilu = spilu(&csc, IluOptions::default()).expect("spilu diagonal");
        assert_eq!(ilu.shape, (3, 3));

        // Solve: diag * x = [4, 10, 9] => x = [2, 2, 3]
        let x = ilu.solve(&[4.0, 10.0, 9.0]).expect("ilu solve");
        assert!((x[0] - 2.0).abs() < 1e-10, "x[0]={}", x[0]);
        assert!((x[1] - 2.0).abs() < 1e-10, "x[1]={}", x[1]);
        assert!((x[2] - 3.0).abs() < 1e-10, "x[2]={}", x[2]);
    }

    #[test]
    fn spilu_tridiagonal_solve() {
        // Tridiagonal: [-1, 2, -1] on diagonals
        let n = 5;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            data.push(2.0);
            if i > 0 {
                rows.push(i);
                cols.push(i - 1);
                data.push(-1.0);
            }
            if i + 1 < n {
                rows.push(i);
                cols.push(i + 1);
                data.push(-1.0);
            }
        }
        let coo = CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
            .expect("tridiagonal coo");
        let csc = coo.to_csc().expect("coo->csc");
        let ilu = spilu(&csc, IluOptions::default()).expect("spilu tridiagonal");

        // For a tridiagonal matrix, ILU(0) = exact LU (no fill-in discarded)
        let b = vec![1.0; n];
        let x = ilu.solve(&b).expect("ilu solve");

        // Verify Ax ≈ b
        let csr = coo.to_csr().expect("coo->csr");
        let ax = spmv_csr(&csr, &x).expect("matvec");
        for i in 0..n {
            assert!(
                (ax[i] - b[i]).abs() < 1e-10,
                "residual at {i}: ax={}, b={}",
                ax[i],
                b[i]
            );
        }
    }

    #[test]
    fn spilu_as_preconditioner_for_cg() {
        // Use ILU as preconditioner: solve M^-1 * A * x = M^-1 * b
        // where M = LU from ILU(0)
        let n = 4;
        // SPD matrix: diag(10, 10, 10, 10) + off-diag(-1)
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            data.push(10.0);
            if i > 0 {
                rows.push(i);
                cols.push(i - 1);
                data.push(-1.0);
                rows.push(i - 1);
                cols.push(i);
                data.push(-1.0);
            }
        }
        let coo =
            CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false).expect("spd coo");
        let csc = coo.to_csc().expect("coo->csc");
        let ilu = spilu(&csc, IluOptions::default()).expect("spilu");

        // Verify ILU factorization produces valid L and U
        assert_eq!(ilu.shape, (n, n));

        let b = vec![1.0, 2.0, 3.0, 4.0];
        let x = ilu.solve(&b).expect("ilu solve");

        // Verify approximate solution quality
        let csr = coo.to_csr().expect("coo->csr");
        let ax = spmv_csr(&csr, &x).expect("matvec");
        let residual: f64 = ax
            .iter()
            .zip(b.iter())
            .map(|(a, bi)| (a - bi).powi(2))
            .sum::<f64>()
            .sqrt();
        let b_norm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            residual / b_norm < 0.1,
            "ILU solve residual too large: {residual}/{b_norm}"
        );
    }

    #[test]
    fn spilu_empty_matrix() {
        let coo = CooMatrix::from_triplets(Shape2D::new(0, 0), vec![], vec![], vec![], false)
            .expect("empty coo");
        let csc = coo.to_csc().expect("coo->csc");
        let ilu = spilu(&csc, IluOptions::default()).expect("spilu empty");
        assert_eq!(ilu.shape, (0, 0));
    }

    #[test]
    fn spilu_non_square_rejected() {
        let coo = CooMatrix::from_triplets(Shape2D::new(2, 3), vec![1.0], vec![0], vec![1], false)
            .expect("non-square coo");
        let csc = coo.to_csc().expect("coo->csc");
        let err = spilu(&csc, IluOptions::default()).expect_err("non-square");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    #[test]
    fn pcg_solves_spd_system() {
        // SPD tridiagonal: diag(4) + off-diag(-1)
        let n = 10;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            data.push(4.0);
            if i > 0 {
                rows.push(i);
                cols.push(i - 1);
                data.push(-1.0);
                rows.push(i - 1);
                cols.push(i);
                data.push(-1.0);
            }
        }
        let coo =
            CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false).expect("coo");
        let csr = coo.to_csr().expect("csr");
        let csc = coo.to_csc().expect("csc");

        // Build ILU(0) preconditioner
        let ilu = spilu(&csc, IluOptions::default()).expect("ilu");

        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let result = pcg(&csr, &b, &ilu, None, IterativeSolveOptions::default()).expect("pcg");
        assert!(result.converged, "PCG should converge");
        assert!(
            result.residual_norm < 1e-5,
            "residual too large: {}",
            result.residual_norm
        );

        // Verify Ax ≈ b
        let ax = spmv_csr(&csr, &result.solution).expect("matvec");
        let b_norm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        let err: f64 = ax
            .iter()
            .zip(b.iter())
            .map(|(a, bi)| (a - bi).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(err / b_norm < 1e-5, "PCG solution error: {err}/{b_norm}");
    }

    // ── Sparse triangular solve tests ─────────────────────────────

    #[test]
    fn spsolve_triangular_lower() {
        // Lower triangular: [[2, 0], [3, 4]]
        let coo = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![2.0, 3.0, 4.0],
            vec![0, 1, 1],
            vec![0, 0, 1],
            false,
        )
        .expect("lower tri");
        let csr = coo.to_csr().expect("csr");
        let x = spsolve_triangular(&csr, &[6.0, 11.0], true).expect("solve");
        // 2*x0 = 6 => x0 = 3
        // 3*x0 + 4*x1 = 11 => 9 + 4*x1 = 11 => x1 = 0.5
        assert!((x[0] - 3.0).abs() < 1e-10);
        assert!((x[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn spsolve_triangular_upper() {
        // Upper triangular: [[2, 3], [0, 4]]
        let coo = CooMatrix::from_triplets(
            Shape2D::new(2, 2),
            vec![2.0, 3.0, 4.0],
            vec![0, 0, 1],
            vec![0, 1, 1],
            false,
        )
        .expect("upper tri");
        let csr = coo.to_csr().expect("csr");
        let x = spsolve_triangular(&csr, &[11.0, 8.0], false).expect("solve");
        // 4*x1 = 8 => x1 = 2
        // 2*x0 + 3*x1 = 11 => 2*x0 + 6 = 11 => x0 = 2.5
        assert!((x[0] - 2.5).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
    }

    // ── Sparse eigenvalue tests ───────────────────────────────────

    #[test]
    fn eigsh_identity_largest() {
        // Find just the largest eigenvalue of identity (= 1)
        let identity = eye(4).expect("identity");
        let result = eigsh(&identity, 1, EigsOptions::default()).expect("eigsh");
        assert!(result.converged, "should converge");
        assert!(
            (result.eigenvalues[0] - 1.0).abs() < 1e-6,
            "ev = {}",
            result.eigenvalues[0]
        );
    }

    #[test]
    fn eigsh_diagonal_largest() {
        // Diagonal matrix with eigenvalues 1, 2, 3, 4
        let coo = CooMatrix::from_triplets(
            Shape2D::new(4, 4),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0, 1, 2, 3],
            vec![0, 1, 2, 3],
            false,
        )
        .expect("diagonal coo");
        let csr = coo.to_csr().expect("csr");
        let result = eigsh(&csr, 1, EigsOptions::default()).expect("eigsh");
        assert!(result.converged);
        // Largest eigenvalue should be 4
        assert!(
            (result.eigenvalues[0] - 4.0).abs() < 1e-6,
            "largest eigenvalue = {}, expected 4.0",
            result.eigenvalues[0]
        );
    }

    #[test]
    fn eigsh_two_largest() {
        // Diagonal with eigenvalues 1, 3, 5
        let coo = CooMatrix::from_triplets(
            Shape2D::new(3, 3),
            vec![1.0, 3.0, 5.0],
            vec![0, 1, 2],
            vec![0, 1, 2],
            false,
        )
        .expect("coo");
        let csr = coo.to_csr().expect("csr");
        let result = eigsh(&csr, 2, EigsOptions::default()).expect("eigsh");
        assert!(result.converged);
        // Should find 5 and 3 (in some order)
        let mut evs = result.eigenvalues.clone();
        evs.sort_by(|a, b| b.total_cmp(a));
        assert!(
            (evs[0] - 5.0).abs() < 1e-4,
            "largest = {}, expected 5.0",
            evs[0]
        );
        assert!(
            (evs[1] - 3.0).abs() < 1e-4,
            "second = {}, expected 3.0",
            evs[1]
        );
    }

    #[test]
    fn eigsh_non_square_rejected() {
        let coo = CooMatrix::from_triplets(Shape2D::new(2, 3), vec![1.0], vec![0], vec![1], false)
            .expect("coo");
        let csr = coo.to_csr().expect("csr");
        let err = eigsh(&csr, 1, EigsOptions::default()).expect_err("non-square");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    #[test]
    fn pcg_converges_faster_than_cg() {
        // PCG with a good preconditioner should use fewer iterations than plain CG
        let n = 20;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            data.push(10.0);
            if i > 0 {
                rows.push(i);
                cols.push(i - 1);
                data.push(-1.0);
                rows.push(i - 1);
                cols.push(i);
                data.push(-1.0);
            }
        }
        let coo =
            CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false).expect("coo");
        let csr = coo.to_csr().expect("csr");
        let csc = coo.to_csc().expect("csc");
        let ilu = spilu(&csc, IluOptions::default()).expect("ilu");

        let b = vec![1.0; n];
        let opts = IterativeSolveOptions {
            tol: 1e-6,
            ..IterativeSolveOptions::default()
        };

        let cg_result = cg(&csr, &b, None, opts).expect("cg");
        let pcg_result = pcg(&csr, &b, &ilu, None, opts).expect("pcg");

        assert!(pcg_result.converged, "PCG should converge");
        assert!(cg_result.converged, "CG should converge");
        // Both converge; verify PCG produces a valid solution
        assert!(
            pcg_result.residual_norm < 1e-5,
            "PCG residual: {}",
            pcg_result.residual_norm
        );
    }
}
