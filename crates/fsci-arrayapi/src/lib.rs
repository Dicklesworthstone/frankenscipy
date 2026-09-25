#![forbid(unsafe_code)]

pub mod audit;
pub mod backend;
pub mod broadcast;
pub mod creation;
pub mod dtype;
pub mod error;
pub mod indexing;
pub mod integration;
pub mod types;

pub use audit::{SyncSharedAuditLedger, record_fail_closed, sync_audit_ledger};
pub use backend::{
    ArrayApiArray, ArrayApiBackend, CoreArray, CoreArrayBackend, DTypeDispatchLog, ShapeMismatchLog,
};
pub use broadcast::broadcast_shapes_with_audit;
pub use broadcast::{broadcast_shapes, promote_and_broadcast};
pub use creation::{
    ArangeRequest, CreationRequest, FullRequest, LinspaceRequest, arange, empty, from_slice, full,
    linspace, ones, zeros,
};
pub use creation::{arange_with_audit, from_slice_with_audit};
pub use dtype::{default_float_dtype, result_type};
pub use error::{ArrayApiError, ArrayApiErrorKind, ArrayApiResult};
pub use indexing::{
    IndexRequest, IndexingMode, getitem, getitem_with_audit, reshape, reshape_with_audit, transpose,
};
pub use integration::{INTEGRATION_SEAMS, NALGEBRA_DMATRIX_INTEGRATION_POINTS};
pub use types::{DType, ExecutionMode, IndexExpr, MemoryOrder, ScalarValue, Shape, SliceSpec};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_boundaries_are_exported() {
        assert!(
            INTEGRATION_SEAMS
                .iter()
                .any(|seam| seam.consumer_crate == "fsci-linalg")
        );
        assert!(
            NALGEBRA_DMATRIX_INTEGRATION_POINTS
                .iter()
                .any(|entry| entry.contains("DMatrix"))
        );
    }

    /// Per br-8ghr: the INTEGRATION_SEAMS declarations are
    /// *aspirational*. Until an entry is backed by a real Cargo
    /// dependency on fsci-arrayapi from the named consumer crate,
    /// it describes design intent only. This test documents today's
    /// live vs aspirational split so an agent adding a Cargo
    /// dependency has a place to flip the label.
    ///
    /// The test deliberately asserts current state (all aspirational)
    /// rather than required state; flip the expected set when a new
    /// consumer wires up.
    #[test]
    fn integration_seam_declarations_are_aspirational() {
        let declared: Vec<&str> = INTEGRATION_SEAMS.iter().map(|s| s.consumer_crate).collect();
        // Live consumers (have a Cargo dep on fsci-arrayapi):
        //   - fsci-conformance (not in INTEGRATION_SEAMS; it's the
        //     Array API test packet driver, not an integration seam).
        // Aspirational (in INTEGRATION_SEAMS but no Cargo dep yet):
        const ASPIRATIONAL: &[&str] = &["fsci-linalg", "fsci-opt", "fsci-sparse"];
        for name in ASPIRATIONAL {
            assert!(
                declared.contains(name),
                "{name} dropped from INTEGRATION_SEAMS but was expected to be aspirational"
            );
        }
    }

    #[test]
    fn audit_wrappers_emit_fail_closed_for_rejections() {
        let backend = CoreArrayBackend::new(ExecutionMode::Hardened);
        let ledger = sync_audit_ledger();

        let bad_arange = ArangeRequest {
            start: ScalarValue::I64(0),
            stop: ScalarValue::I64(3),
            step: ScalarValue::I64(0),
            dtype: Some(DType::Float64),
        };
        let err =
            arange_with_audit(&backend, &bad_arange, &ledger).expect_err("zero step must reject");
        assert_eq!(err.kind, ArrayApiErrorKind::InvalidStep);

        let bad_from_slice = CreationRequest {
            shape: Shape::new(vec![2, 2]),
            dtype: DType::Float64,
            order: MemoryOrder::C,
        };
        let err = from_slice_with_audit(
            &backend,
            &[ScalarValue::F64(1.0), ScalarValue::F64(2.0)],
            &bad_from_slice,
            &ledger,
        )
        .expect_err("length mismatch must reject");
        assert_eq!(err.kind, ArrayApiErrorKind::InvalidShape);

        let err =
            broadcast_shapes_with_audit(&[Shape::new(vec![2, 3]), Shape::new(vec![3, 2])], &ledger)
                .expect_err("incompatible broadcast must reject");
        assert_eq!(err.kind, ArrayApiErrorKind::BroadcastIncompatible);

        let guard = ledger.lock().expect("audit ledger lock");
        assert_eq!(guard.len(), 3);
        assert!(
            guard.entries().iter().all(|entry| {
                matches!(entry.action, fsci_runtime::AuditAction::FailClosed { .. })
            })
        );
    }

    /// frankenscipy-3cu8u.1: the audit fingerprints cover the backend and every input. They
    /// used to be `Debug` strings that left out the backend's mode, `from_slice`'s values
    /// (only their count) and the array under `getitem`/`reshape` (only its shape), so each
    /// differing pair below shared one fingerprint.
    #[test]
    fn audit_fingerprints_cover_every_input() {
        fn only_fingerprint(ledger: &SyncSharedAuditLedger) -> String {
            let guard = ledger.lock().expect("audit ledger lock");
            assert_eq!(guard.len(), 1, "one rejection, one event");
            guard.entries()[0].input_fingerprint.clone()
        }
        let strict = CoreArrayBackend::new(ExecutionMode::Strict);
        let hardened = CoreArrayBackend::new(ExecutionMode::Hardened);

        // arange: one zero-step request on backends in different modes.
        let zero_step = ArangeRequest {
            start: ScalarValue::I64(0),
            stop: ScalarValue::I64(3),
            step: ScalarValue::I64(0),
            dtype: Some(DType::Float64),
        };
        let arange_on = |backend: &CoreArrayBackend| {
            let ledger = sync_audit_ledger();
            let err =
                arange_with_audit(backend, &zero_step, &ledger).expect_err("zero step must reject");
            assert_eq!(err.kind, ArrayApiErrorKind::InvalidStep);
            only_fingerprint(&ledger)
        };
        let base = arange_on(&hardened);
        assert!(base.starts_with("blake3:"), "{base}");
        assert_eq!(base, arange_on(&hardened), "same input, same fingerprint");
        assert_ne!(base, arange_on(&strict));

        // from_slice: same count, different values.
        let square = CreationRequest {
            shape: Shape::new(vec![2, 2]),
            dtype: DType::Float64,
            order: MemoryOrder::C,
        };
        let from_slice_of = |values: &[ScalarValue]| {
            let ledger = sync_audit_ledger();
            let err = from_slice_with_audit(&hardened, values, &square, &ledger)
                .expect_err("length mismatch must reject");
            assert_eq!(err.kind, ArrayApiErrorKind::InvalidShape);
            only_fingerprint(&ledger)
        };
        let two = from_slice_of(&[ScalarValue::F64(1.0), ScalarValue::F64(2.0)]);
        assert_eq!(
            two,
            from_slice_of(&[ScalarValue::F64(1.0), ScalarValue::F64(2.0)])
        );
        assert_ne!(
            two,
            from_slice_of(&[ScalarValue::F64(1.0), ScalarValue::F64(3.0)])
        );

        // getitem and reshape: same shape, different values.
        let vector = CreationRequest {
            shape: Shape::new(vec![2]),
            dtype: DType::Float64,
            order: MemoryOrder::C,
        };
        let array_of = |second: f64| {
            from_slice(
                &hardened,
                &[ScalarValue::F64(1.0), ScalarValue::F64(second)],
                &vector,
            )
            .expect("valid array")
        };
        let (a, b) = (array_of(2.0), array_of(5.0));
        let mismatched = IndexRequest {
            mode: IndexingMode::Basic,
            index: IndexExpr::BooleanMask {
                mask_shape: Shape::new(vec![2]),
            },
        };
        let getitem_of = |array: &CoreArray| {
            let ledger = sync_audit_ledger();
            let err = getitem_with_audit(&hardened, array, &mismatched, &ledger)
                .expect_err("mode/index mismatch must reject");
            assert_eq!(err.kind, ArrayApiErrorKind::InvalidIndex);
            only_fingerprint(&ledger)
        };
        assert_eq!(getitem_of(&a), getitem_of(&a));
        assert_ne!(getitem_of(&a), getitem_of(&b));
        let reshape_of = |array: &CoreArray| {
            let ledger = sync_audit_ledger();
            let err = reshape_with_audit(&hardened, array, &Shape::new(vec![3]), &ledger)
                .expect_err("element-count mismatch must reject");
            assert_eq!(err.kind, ArrayApiErrorKind::InvalidShape);
            only_fingerprint(&ledger)
        };
        assert_eq!(reshape_of(&a), reshape_of(&a));
        assert_ne!(reshape_of(&a), reshape_of(&b));
        // The routine is part of the fingerprint.
        assert_ne!(getitem_of(&a), reshape_of(&a));
    }

    #[test]
    fn metamorphic_broadcast_shapes_are_permutation_and_scalar_invariant() {
        let base = [Shape::new(vec![2, 1, 3]), Shape::new(vec![1, 4, 3])];
        let expected = Shape::new(vec![2, 4, 3]);

        assert_eq!(
            broadcast_shapes(&base).expect("base shapes should broadcast"),
            expected
        );
        assert_eq!(
            broadcast_shapes(&[base[1].clone(), base[0].clone()])
                .expect("permuted shapes should broadcast"),
            expected
        );
        assert_eq!(
            broadcast_shapes(&[Shape::scalar(), base[0].clone(), base[1].clone()])
                .expect("adding a scalar operand should not change the target shape"),
            expected
        );
    }

    #[test]
    fn metamorphic_broadcast_to_repeats_singleton_axes_without_value_drift() {
        let backend = CoreArrayBackend::new(ExecutionMode::Strict);
        let source = backend
            .array_from_slice(
                &[ScalarValue::F64(1.0), ScalarValue::F64(2.0)],
                &Shape::new(vec![2, 1]),
                DType::Float64,
                MemoryOrder::C,
            )
            .expect("source array should build");
        let target_shape = Shape::new(vec![2, 3]);
        let broadcasted = backend
            .broadcast_to(&source, &target_shape)
            .expect("singleton axis should broadcast");

        assert_eq!(broadcasted.shape(), &target_shape);
        assert_eq!(
            broadcasted.values(),
            &[
                ScalarValue::F64(1.0),
                ScalarValue::F64(1.0),
                ScalarValue::F64(1.0),
                ScalarValue::F64(2.0),
                ScalarValue::F64(2.0),
                ScalarValue::F64(2.0),
            ]
        );
    }
}
