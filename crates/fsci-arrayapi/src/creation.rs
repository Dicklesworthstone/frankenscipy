use crate::backend::ArrayApiBackend;
use crate::error::{ArrayApiError, ArrayApiErrorKind, ArrayApiResult};
use crate::types::{DType, MemoryOrder, ScalarValue, Shape};
use fsci_runtime::Fingerprinter;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreationRequest {
    pub shape: Shape,
    pub dtype: DType,
    pub order: MemoryOrder,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FullRequest {
    pub fill_value: ScalarValue,
    pub dtype: DType,
    pub order: MemoryOrder,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArangeRequest {
    pub start: ScalarValue,
    pub stop: ScalarValue,
    pub step: ScalarValue,
    pub dtype: Option<DType>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinspaceRequest {
    pub start: ScalarValue,
    pub stop: ScalarValue,
    pub num: usize,
    pub endpoint: bool,
    pub dtype: Option<DType>,
}

fn validate_shape(shape: &Shape) -> ArrayApiResult<()> {
    if shape.element_count().is_none() {
        return Err(ArrayApiError::new(
            ArrayApiErrorKind::Overflow,
            "shape element count overflow",
        ));
    }
    Ok(())
}

pub fn zeros<B: ArrayApiBackend>(
    backend: &B,
    request: &CreationRequest,
) -> ArrayApiResult<B::Array> {
    validate_shape(&request.shape)?;
    backend.zeros(&request.shape, request.dtype, request.order)
}

pub fn ones<B: ArrayApiBackend>(
    backend: &B,
    request: &CreationRequest,
) -> ArrayApiResult<B::Array> {
    validate_shape(&request.shape)?;
    backend.ones(&request.shape, request.dtype, request.order)
}

pub fn empty<B: ArrayApiBackend>(
    backend: &B,
    request: &CreationRequest,
) -> ArrayApiResult<B::Array> {
    validate_shape(&request.shape)?;
    backend.empty(&request.shape, request.dtype, request.order)
}

pub fn full<B: ArrayApiBackend>(
    backend: &B,
    shape: &Shape,
    request: &FullRequest,
) -> ArrayApiResult<B::Array> {
    validate_shape(shape)?;
    backend.full(shape, request.fill_value, request.dtype, request.order)
}

pub fn arange<B: ArrayApiBackend>(
    backend: &B,
    request: &ArangeRequest,
) -> ArrayApiResult<B::Array> {
    if scalar_is_zero(request.step) {
        return Err(ArrayApiError::new(
            ArrayApiErrorKind::InvalidStep,
            "step must be nonzero",
        ));
    }
    if !scalar_is_finite(request.step) {
        return Err(ArrayApiError::new(
            ArrayApiErrorKind::InvalidStep,
            "step must be finite",
        ));
    }
    backend.arange(request.start, request.stop, request.step, request.dtype)
}

pub fn arange_with_audit<B: ArrayApiBackend>(
    backend: &B,
    request: &ArangeRequest,
    ledger: &crate::audit::SyncSharedAuditLedger,
) -> ArrayApiResult<B::Array> {
    let result = arange(backend, request);
    if let Err(err) = &result {
        // frankenscipy-3cu8u.1: the backend configuration, then `start`, `stop`, `step` and
        // `dtype`. It used to be the request's `Debug` string, which omits the backend's mode
        // and renders every NaN alike.
        let mut fingerprinter = Fingerprinter::new("fsci_arrayapi::arange");
        backend.fingerprint_config(&mut fingerprinter);
        for value in [request.start, request.stop, request.step] {
            crate::audit::fingerprint_scalar(&mut fingerprinter, value);
        }
        fingerprinter.str(&format!("{:?}", request.dtype));
        crate::audit::record_array_api_error(ledger, "arange", &fingerprinter.finish(), err.kind);
    }
    result
}

pub fn linspace<B: ArrayApiBackend>(
    backend: &B,
    request: &LinspaceRequest,
) -> ArrayApiResult<B::Array> {
    backend.linspace(
        request.start,
        request.stop,
        request.num,
        request.endpoint,
        request.dtype,
    )
}

pub fn from_slice<B: ArrayApiBackend>(
    backend: &B,
    values: &[ScalarValue],
    request: &CreationRequest,
) -> ArrayApiResult<B::Array> {
    validate_shape(&request.shape)?;
    backend.array_from_slice(values, &request.shape, request.dtype, request.order)
}

pub fn from_slice_with_audit<B: ArrayApiBackend>(
    backend: &B,
    values: &[ScalarValue],
    request: &CreationRequest,
    ledger: &crate::audit::SyncSharedAuditLedger,
) -> ArrayApiResult<B::Array> {
    let result = from_slice(backend, values, request);
    if let Err(err) = &result {
        // frankenscipy-3cu8u.1: the backend configuration, every value, then the request's
        // shape, dtype and order. It used to carry only `values.len()`, not the values.
        let mut fingerprinter = Fingerprinter::new("fsci_arrayapi::from_slice");
        backend.fingerprint_config(&mut fingerprinter);
        crate::audit::fingerprint_scalars(&mut fingerprinter, values);
        fingerprinter
            .shape(&request.shape.dims)
            .str(&format!("{:?}", request.dtype))
            .str(&format!("{:?}", request.order));
        crate::audit::record_array_api_error(
            ledger,
            "from_slice",
            &fingerprinter.finish(),
            err.kind,
        );
    }
    result
}

fn scalar_is_zero(value: ScalarValue) -> bool {
    match value {
        ScalarValue::Bool(v) => !v,
        ScalarValue::I64(v) => v == 0,
        ScalarValue::U64(v) => v == 0,
        ScalarValue::F64(v) => v == 0.0,
        ScalarValue::ComplexF64 { re, im } => re == 0.0 && im == 0.0,
    }
}

fn scalar_is_finite(value: ScalarValue) -> bool {
    match value {
        ScalarValue::Bool(_) | ScalarValue::I64(_) | ScalarValue::U64(_) => true,
        ScalarValue::F64(v) => v.is_finite(),
        ScalarValue::ComplexF64 { re, im } => re.is_finite() && im.is_finite(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{ArrayApiArray, CoreArrayBackend};
    use crate::types::ExecutionMode;

    fn strict_backend() -> CoreArrayBackend {
        CoreArrayBackend::new(ExecutionMode::Strict)
    }

    #[test]
    fn constructors_reject_overflowing_shape() {
        let backend = strict_backend();
        let request = CreationRequest {
            shape: Shape::new(vec![usize::MAX, 2]),
            dtype: DType::Float64,
            order: MemoryOrder::C,
        };
        let full_request = FullRequest {
            fill_value: ScalarValue::F64(1.0),
            dtype: DType::Float64,
            order: MemoryOrder::C,
        };

        for result in [
            zeros(&backend, &request),
            ones(&backend, &request),
            empty(&backend, &request),
            from_slice(&backend, &[ScalarValue::F64(1.0)], &request),
            full(&backend, &request.shape, &full_request),
        ] {
            let err = result.expect_err("overflowing shape must fail");
            assert_eq!(err.kind, ArrayApiErrorKind::Overflow);
        }
    }

    #[test]
    fn arange_rejects_zero_step_for_all_supported_scalar_variants() {
        let backend = strict_backend();
        for step in [
            ScalarValue::I64(0),
            ScalarValue::U64(0),
            ScalarValue::F64(0.0),
            ScalarValue::ComplexF64 { re: 0.0, im: 0.0 },
        ] {
            let request = ArangeRequest {
                start: ScalarValue::F64(0.0),
                stop: ScalarValue::F64(3.0),
                step,
                dtype: Some(DType::Float64),
            };
            let err = arange(&backend, &request).expect_err("zero step must fail");
            assert_eq!(err.kind, ArrayApiErrorKind::InvalidStep);
        }
    }

    #[test]
    fn arange_rejects_non_finite_step_values() {
        let backend = strict_backend();
        for step in [
            ScalarValue::F64(f64::NAN),
            ScalarValue::F64(f64::INFINITY),
            ScalarValue::F64(f64::NEG_INFINITY),
            ScalarValue::ComplexF64 {
                re: f64::NAN,
                im: 0.0,
            },
            ScalarValue::ComplexF64 {
                re: 1.0,
                im: f64::INFINITY,
            },
        ] {
            let request = ArangeRequest {
                start: ScalarValue::F64(0.0),
                stop: ScalarValue::F64(3.0),
                step,
                dtype: Some(DType::Float64),
            };
            let err = arange(&backend, &request).expect_err("non-finite step must fail");
            assert_eq!(err.kind, ArrayApiErrorKind::InvalidStep);
        }
    }

    #[test]
    fn arange_non_integer_step_matches_numpy_length() {
        // numpy uses an analytic length n = ceil((stop-start)/step); accumulating
        // `current += step` lets FP drift emit a spurious extra element.
        let backend = strict_backend();
        // (start, stop, step, expected_len) — verified against numpy.arange.
        let cases = [
            (0.0, 1.0, 0.1, 10),
            (0.0, 1.0, 0.2, 5),
            (1.0, 2.0, 0.3, 4),
            (-2.0, 2.0, 0.5, 8),
            (0.0, 1.0, 0.3, 4),
            (5.0, 0.0, -1.0, 5),
            (0.0, 0.0, 1.0, 0),
            (0.0, -1.0, 1.0, 0),
        ];
        for (start, stop, step, expected_len) in cases {
            let request = ArangeRequest {
                start: ScalarValue::F64(start),
                stop: ScalarValue::F64(stop),
                step: ScalarValue::F64(step),
                dtype: Some(DType::Float64),
            };
            let arr = arange(&backend, &request).expect("arange should succeed");
            assert_eq!(
                arr.values().len(),
                expected_len,
                "arange({start}, {stop}, {step}) length"
            );
            for (i, value) in arr.values().iter().enumerate() {
                let expected = start + (i as f64) * step;
                assert_eq!(
                    *value,
                    ScalarValue::F64(expected),
                    "arange({start}, {stop}, {step})[{i}]"
                );
            }
        }
    }

    #[test]
    fn linspace_and_full_generate_expected_shapes() {
        let backend = strict_backend();
        let linspace_request = LinspaceRequest {
            start: ScalarValue::F64(-1.0),
            stop: ScalarValue::F64(1.0),
            num: 5,
            endpoint: true,
            dtype: Some(DType::Float32),
        };
        let lin = linspace(&backend, &linspace_request).expect("linspace should succeed");
        assert_eq!(lin.shape(), &Shape::new(vec![5]));
        assert_eq!(lin.dtype(), DType::Float32);

        let full_request = FullRequest {
            fill_value: ScalarValue::ComplexF64 { re: 2.0, im: -3.0 },
            dtype: DType::Complex64,
            order: MemoryOrder::F,
        };
        let full_array =
            full(&backend, &Shape::new(vec![2, 2]), &full_request).expect("full should succeed");
        assert_eq!(full_array.shape(), &Shape::new(vec![2, 2]));
        assert_eq!(full_array.dtype(), DType::Complex64);
        assert_eq!(full_array.size(), 4);
    }

    #[test]
    fn linspace_single_sample_is_start() {
        // numpy/scipy: linspace(start, stop, num=1) == [start], for either
        // endpoint value — the lone sample sits at `start`, not `stop`.
        let backend = strict_backend();
        for endpoint in [true, false] {
            let request = LinspaceRequest {
                start: ScalarValue::F64(2.0),
                stop: ScalarValue::F64(10.0),
                num: 1,
                endpoint,
                dtype: Some(DType::Float64),
            };
            let arr = linspace(&backend, &request).expect("linspace num=1");
            assert_eq!(
                arr.values(),
                &[ScalarValue::F64(2.0)],
                "endpoint={endpoint}"
            );
        }
        // num=2 with endpoint still spans the full [start, stop] interval.
        let request = LinspaceRequest {
            start: ScalarValue::F64(2.0),
            stop: ScalarValue::F64(10.0),
            num: 2,
            endpoint: true,
            dtype: Some(DType::Float64),
        };
        let arr = linspace(&backend, &request).expect("linspace num=2");
        assert_eq!(
            arr.values(),
            &[ScalarValue::F64(2.0), ScalarValue::F64(10.0)]
        );
    }
}
