pub mod construct;
pub mod formats;
pub mod linalg;
pub mod ops;

pub use construct::{diags, eye, random};
pub use formats::{
    CanonicalMeta, ConstructionLogEntry, CooMatrix, CscMatrix, CsrMatrix, NalgebraBridge, Shape2D,
    SparseError, SparseFormat, SparseResult,
};
pub use linalg::{
    IluOptions, LuOptions, PermutationOrdering, SolveOptions, SolveResult, SparseBackend,
    SparseIluFactorization, SparseLuFactorization, spilu, splu, spsolve,
};
pub use ops::{
    ConversionLogEntry, FormatConvertible, add_coo, add_csc, add_csr, coo_to_csr_with_mode,
    csc_to_csr_with_mode, csr_to_csc_with_mode, scale_coo, scale_csc, scale_csr, spmv_coo,
    spmv_csc, spmv_csr, sub_coo, sub_csc, sub_csr,
};

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use fsci_runtime::RuntimeMode;
    use proptest::prelude::*;
    use serde_json::{Value, json};

    use super::*;

    const PROPTEST_CASES: u32 = 512;
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

    #[test]
    fn spmv_identity_matrix_returns_input() {
        let identity = eye(5).expect("identity matrix");
        let vector = vec![3.0, -2.0, 4.5, 1.25, -0.5];
        let out = spmv_csr(&identity, &vector).expect("identity spmv");
        assert_vec_close(&out, &vector);
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
        assert_eq!(roundtrip.canonical_meta(), csr.canonical_meta());

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
}
