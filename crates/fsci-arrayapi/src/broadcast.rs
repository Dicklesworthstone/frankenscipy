use crate::backend::ArrayApiBackend;
use crate::error::{ArrayApiError, ArrayApiErrorKind, ArrayApiResult};
use crate::types::Shape;

pub fn broadcast_shapes(shapes: &[Shape]) -> ArrayApiResult<Shape> {
    if shapes.is_empty() {
        return Ok(Shape::scalar());
    }

    let rank = shapes.iter().map(Shape::rank).max().unwrap_or(0);
    let mut out = vec![1usize; rank];

    for shape in shapes {
        let len = shape.dims.len();
        for (offset, dim) in shape.dims.iter().enumerate() {
            let idx = rank - len + offset;
            let current = out[idx];
            if current == 1 {
                out[idx] = *dim;
                continue;
            }
            if *dim == 1 || *dim == current {
                continue;
            }
            return Err(ArrayApiError::new(
                ArrayApiErrorKind::BroadcastIncompatible,
                "Array shapes are incompatible for broadcasting.",
            ));
        }
    }

    Ok(Shape::new(out))
}

pub fn broadcast_shapes_with_audit(
    shapes: &[Shape],
    ledger: &crate::audit::SyncSharedAuditLedger,
) -> ArrayApiResult<Shape> {
    let result = broadcast_shapes(shapes);
    if let Err(err) = &result {
        crate::audit::record_array_api_error(
            ledger,
            "broadcast_shapes",
            format!("{shapes:?}").as_bytes(),
            err.kind,
        );
    }
    result
}

pub fn promote_and_broadcast<B: ArrayApiBackend>(
    backend: &B,
    arrays: &[&B::Array],
    force_floating: bool,
) -> ArrayApiResult<Vec<B::Array>> {
    if arrays.is_empty() {
        return Ok(Vec::new());
    }

    let dtypes = arrays
        .iter()
        .map(|array| backend.dtype_of(array))
        .collect::<Vec<_>>();
    let target_dtype = backend.result_type(&dtypes, force_floating)?;
    let target_shape = broadcast_shapes(
        &arrays
            .iter()
            .map(|array| backend.shape_of(array))
            .collect::<Vec<_>>(),
    )?;

    arrays
        .iter()
        .map(|array| {
            let cast = backend.astype(array, target_dtype)?;
            backend.broadcast_to(&cast, &target_shape)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{ArrayApiArray, ArrayApiBackend, CoreArrayBackend};
    use crate::types::{DType, ExecutionMode, MemoryOrder, ScalarValue};

    #[test]
    fn broadcast_shapes_covers_scalar_compatible_and_incompatible_inputs() {
        assert_eq!(
            broadcast_shapes(&[]).expect("empty shape list should broadcast to scalar"),
            Shape::scalar()
        );
        assert_eq!(
            broadcast_shapes(&[Shape::new(vec![2, 1]), Shape::new(vec![1, 3])])
                .expect("compatible shapes should broadcast"),
            Shape::new(vec![2, 3])
        );

        let err = broadcast_shapes(&[Shape::new(vec![2, 3]), Shape::new(vec![3, 2])])
            .expect_err("incompatible shapes should fail");
        assert_eq!(err.kind, ArrayApiErrorKind::BroadcastIncompatible);
    }

    #[test]
    fn broadcast_shapes_metamorphic_scalar_identity_and_idempotence() {
        let shapes = [
            Shape::scalar(),
            Shape::new(vec![1]),
            Shape::new(vec![2, 3]),
            Shape::new(vec![1, 4, 1]),
            Shape::new(vec![2, 1, 3]),
        ];

        for shape in shapes {
            assert_eq!(
                broadcast_shapes(&[shape.clone(), Shape::scalar()])
                    .expect("scalar should be the broadcast identity"),
                shape
            );
            assert_eq!(
                broadcast_shapes(&[shape.clone(), shape.clone()])
                    .expect("broadcasting a shape with itself should be idempotent"),
                shape
            );
        }
    }

    #[test]
    fn broadcast_shapes_metamorphic_associative_for_compatible_triples() {
        let triples = [
            (
                Shape::new(vec![2, 1, 3]),
                Shape::new(vec![1, 4, 3]),
                Shape::new(vec![2, 4, 1]),
            ),
            (
                Shape::new(vec![1, 5]),
                Shape::new(vec![3, 1, 5]),
                Shape::scalar(),
            ),
            (
                Shape::new(vec![0, 1]),
                Shape::new(vec![1, 7]),
                Shape::new(vec![0, 7]),
            ),
        ];

        for (left, middle, right) in triples {
            let all_at_once = broadcast_shapes(&[left.clone(), middle.clone(), right.clone()])
                .expect("triple should broadcast");
            let left_fold = broadcast_shapes(&[
                broadcast_shapes(&[left, middle]).expect("pair should broadcast"),
                right,
            ])
            .expect("folded broadcast should succeed");

            assert_eq!(all_at_once, left_fold);
        }
    }

    #[test]
    fn broadcast_shapes_metamorphic_scalar_does_not_repair_incompatibility() {
        let without_scalar = broadcast_shapes(&[Shape::new(vec![2, 3]), Shape::new(vec![3, 2])])
            .expect_err("base shapes are incompatible");
        let with_scalar = broadcast_shapes(&[
            Shape::new(vec![2, 3]),
            Shape::scalar(),
            Shape::new(vec![3, 2]),
        ])
        .expect_err("adding scalar should not hide incompatibility");

        assert_eq!(without_scalar.kind, with_scalar.kind);
    }

    #[test]
    fn promote_and_broadcast_promotes_dtype_and_targets_common_shape() {
        let backend = CoreArrayBackend::new(ExecutionMode::Strict);
        let left = backend
            .array_from_slice(
                &[ScalarValue::F64(1.0), ScalarValue::F64(2.0)],
                &Shape::new(vec![2, 1]),
                DType::Float32,
                MemoryOrder::C,
            )
            .expect("left array should build");
        let right = backend
            .array_from_slice(
                &[
                    ScalarValue::ComplexF64 { re: 3.0, im: 1.0 },
                    ScalarValue::ComplexF64 { re: 4.0, im: -2.0 },
                ],
                &Shape::new(vec![1, 2]),
                DType::Complex128,
                MemoryOrder::C,
            )
            .expect("right array should build");

        let result =
            promote_and_broadcast(&backend, &[&left, &right], false).expect("promotion succeeds");
        assert_eq!(result.len(), 2);
        for array in result {
            assert_eq!(array.shape(), &Shape::new(vec![2, 2]));
            assert_eq!(array.dtype(), DType::Complex128);
            assert_eq!(array.size(), 4);
        }
    }

    fn profile_sequence_values(len: usize) -> Vec<ScalarValue> {
        (0..len)
            .map(|idx| ScalarValue::F64((idx as f64) * 0.5 + 1.0))
            .collect()
    }

    fn update_fingerprint(hash: &mut u64, value: u64) {
        for byte in value.to_le_bytes() {
            *hash ^= u64::from(byte);
            *hash = hash.wrapping_mul(1_099_511_628_211);
        }
    }

    fn scalar_fingerprint(hash: &mut u64, value: ScalarValue) {
        match value {
            ScalarValue::Bool(value) => update_fingerprint(hash, u64::from(value)),
            ScalarValue::I64(value) => update_fingerprint(hash, value as u64),
            ScalarValue::U64(value) => update_fingerprint(hash, value),
            ScalarValue::F64(value) => update_fingerprint(hash, value.to_bits()),
            ScalarValue::ComplexF64 { re, im } => {
                update_fingerprint(hash, re.to_bits());
                update_fingerprint(hash, im.to_bits());
            }
        }
    }

    fn array_value_fingerprint(values: &[ScalarValue]) -> u64 {
        let mut hash = 14_695_981_039_346_656_037u64;
        update_fingerprint(&mut hash, values.len() as u64);
        for value in values {
            scalar_fingerprint(&mut hash, *value);
        }
        hash
    }

    #[test]
    fn promote_and_broadcast_profile_golden_snapshot() {
        let backend = CoreArrayBackend::new(ExecutionMode::Strict);
        let left_values = profile_sequence_values(10_000);
        let left = backend
            .array_from_slice(
                &left_values,
                &Shape::new(vec![10_000, 1]),
                DType::Float32,
                MemoryOrder::C,
            )
            .expect("left array should build");
        let right = backend
            .array_from_slice(
                &[
                    ScalarValue::ComplexF64 { re: 3.0, im: 1.0 },
                    ScalarValue::ComplexF64 { re: 4.0, im: -2.0 },
                ],
                &Shape::new(vec![1, 2]),
                DType::Complex128,
                MemoryOrder::C,
            )
            .expect("right array should build");

        let result =
            promote_and_broadcast(&backend, &[&left, &right], false).expect("promotion succeeds");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].shape(), &Shape::new(vec![10_000, 2]));
        assert_eq!(result[1].shape(), &Shape::new(vec![10_000, 2]));
        assert_eq!(result[0].dtype(), DType::Complex128);
        assert_eq!(result[1].dtype(), DType::Complex128);
        assert_eq!(result[0].order(), MemoryOrder::C);
        assert_eq!(result[1].order(), MemoryOrder::C);

        println!(
            "ARRAYAPI_BROADCAST_GOLDEN left_len={} left_hash={:016x} right_len={} right_hash={:016x}",
            result[0].values().len(),
            array_value_fingerprint(result[0].values()),
            result[1].values().len(),
            array_value_fingerprint(result[1].values())
        );
    }

    #[test]
    fn promote_and_broadcast_propagates_shape_mismatch_errors() {
        let backend = CoreArrayBackend::new(ExecutionMode::Strict);
        let left = backend
            .array_from_slice(
                &[
                    ScalarValue::F64(1.0),
                    ScalarValue::F64(2.0),
                    ScalarValue::F64(3.0),
                    ScalarValue::F64(4.0),
                ],
                &Shape::new(vec![2, 2]),
                DType::Float64,
                MemoryOrder::C,
            )
            .expect("left array should build");
        let right = backend
            .array_from_slice(
                &[
                    ScalarValue::F64(5.0),
                    ScalarValue::F64(6.0),
                    ScalarValue::F64(7.0),
                    ScalarValue::F64(8.0),
                    ScalarValue::F64(9.0),
                    ScalarValue::F64(10.0),
                ],
                &Shape::new(vec![2, 3]),
                DType::Float64,
                MemoryOrder::C,
            )
            .expect("right array should build");

        let err = promote_and_broadcast(&backend, &[&left, &right], false)
            .expect_err("shape mismatch should bubble up");
        assert_eq!(err.kind, ArrayApiErrorKind::BroadcastIncompatible);
    }
}
