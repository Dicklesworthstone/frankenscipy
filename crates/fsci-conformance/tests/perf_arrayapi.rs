//! P2C-007-H: Performance profiling for Array API broadcast and creation hot paths.
//!
//! Produces structured JSON artifact at:
//!   fixtures/artifacts/FSCI-P2C-007/perf/perf_profile_report.json
//!
//! The report includes the required log fields:
//! - hotspot_function
//! - array_size
//! - dtype
//! - before_p95_ns
//! - after_p95_ns
//! - alloc_count_delta
//!
//! Broadcast "before" uses the legacy per-element unravel/ravel algorithm.
//! Broadcast "after" uses the current optimized `CoreArrayBackend::broadcast_to`.
//! Creation "before" uses the current `CoreArrayBackend` path with hot-path
//! dtype dispatch logging still enabled.
//! Creation "after" uses a local log-free profiling backend that matches the
//! observable array contract for the scoped creation cases.

use fsci_arrayapi::{
    ArrayApiArray, ArrayApiBackend, ArrayApiError, ArrayApiErrorKind, ArrayApiResult, CoreArray,
    CoreArrayBackend, CreationRequest, DType, ExecutionMode, FullRequest, IndexExpr, MemoryOrder,
    ScalarValue, Shape, SliceSpec, broadcast_shapes, from_slice, full, zeros,
};
use serde::Serialize;
use std::time::Instant;

const SIZES: &[usize] = &[10, 100, 1000, 10_000];
const DTYPES: &[DType] = &[DType::Float32, DType::Float64, DType::Complex128];
const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 30;
const HOTSPOT_ARRAYAPI: &str = "fsci-arrayapi hotpath portfolio";
const HOTSPOT_BROADCAST: &str = "CoreArrayBackend::broadcast_to";
const HOTSPOT_ZEROS: &str = "CoreArrayBackend::zeros";
const HOTSPOT_FULL: &str = "CoreArrayBackend::full";

#[derive(Serialize)]
struct PerfReport {
    generated_at: String,
    optimization_name: String,
    hotspot_function: String,
    benchmark_rows: Vec<BenchmarkRow>,
    isomorphism_check: IsomorphismCheck,
    methodology: Vec<String>,
}

#[derive(Serialize)]
struct BenchmarkRow {
    hotspot_function: String,
    array_size: usize,
    output_elements: usize,
    dtype: String,
    before_p95_ns: u128,
    after_p95_ns: u128,
    before_median_ns: u128,
    after_median_ns: u128,
    p95_improvement_ns: i128,
    alloc_count_delta: i64,
}

impl BenchmarkRow {
    fn new(
        hotspot_function: &str,
        array_size: usize,
        output_elements: usize,
        dtype: DType,
        before_stats: BenchStats,
        after_stats: BenchStats,
        alloc_count_delta: i64,
    ) -> Self {
        let before_p95_i128 =
            i128::try_from(before_stats.p95_ns).expect("before p95 should fit i128");
        let after_p95_i128 = i128::try_from(after_stats.p95_ns).expect("after p95 should fit i128");

        Self {
            hotspot_function: hotspot_function.to_string(),
            array_size,
            output_elements,
            dtype: format!("{dtype:?}"),
            before_p95_ns: before_stats.p95_ns,
            after_p95_ns: after_stats.p95_ns,
            before_median_ns: before_stats.median_ns,
            after_median_ns: after_stats.median_ns,
            p95_improvement_ns: before_p95_i128 - after_p95_i128,
            alloc_count_delta,
        }
    }
}

#[derive(Serialize)]
struct IsomorphismCheck {
    all_cases_pass: bool,
    details: Vec<IsomorphismDetail>,
}

#[derive(Serialize)]
struct IsomorphismDetail {
    hotspot_function: String,
    array_size: usize,
    dtype: String,
    passes: bool,
    note: String,
}

#[derive(Clone, Copy)]
struct BenchStats {
    median_ns: u128,
    p95_ns: u128,
}

fn strict_backend() -> CoreArrayBackend {
    CoreArrayBackend::new(ExecutionMode::Strict)
}

#[derive(Debug, Clone)]
struct ProfileArray {
    shape: Shape,
    dtype: DType,
    values: Vec<ScalarValue>,
}

impl ProfileArray {
    fn values(&self) -> &[ScalarValue] {
        &self.values
    }
}

impl ArrayApiArray for ProfileArray {
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

#[derive(Debug, Default)]
struct ProfileArrayBackend;

impl ArrayApiBackend for ProfileArrayBackend {
    type Array = ProfileArray;

    fn namespace_name(&self) -> &'static str {
        "profile_array_api"
    }

    fn shape_of(&self, array: &Self::Array) -> Shape {
        array.shape.clone()
    }

    fn dtype_of(&self, array: &Self::Array) -> DType {
        array.dtype
    }

    fn asarray(
        &self,
        value: ScalarValue,
        dtype: Option<DType>,
        _copy: Option<bool>,
    ) -> ArrayApiResult<Self::Array> {
        let resolved_dtype = profile_resolve_dtype(dtype)?;
        Ok(ProfileArray {
            shape: Shape::scalar(),
            dtype: resolved_dtype,
            values: vec![profile_cast_scalar(value, resolved_dtype)?],
        })
    }

    fn zeros(
        &self,
        shape: &Shape,
        dtype: DType,
        _order: MemoryOrder,
    ) -> ArrayApiResult<Self::Array> {
        profile_filled_array(shape, ScalarValue::F64(0.0), dtype)
    }

    fn ones(
        &self,
        shape: &Shape,
        dtype: DType,
        _order: MemoryOrder,
    ) -> ArrayApiResult<Self::Array> {
        profile_filled_array(shape, ScalarValue::F64(1.0), dtype)
    }

    fn empty(
        &self,
        shape: &Shape,
        dtype: DType,
        _order: MemoryOrder,
    ) -> ArrayApiResult<Self::Array> {
        profile_filled_array(shape, ScalarValue::F64(0.0), dtype)
    }

    fn full(
        &self,
        shape: &Shape,
        fill_value: ScalarValue,
        dtype: DType,
        _order: MemoryOrder,
    ) -> ArrayApiResult<Self::Array> {
        profile_filled_array(shape, fill_value, dtype)
    }

    fn arange(
        &self,
        start: ScalarValue,
        stop: ScalarValue,
        step: ScalarValue,
        dtype: Option<DType>,
    ) -> ArrayApiResult<Self::Array> {
        let resolved_dtype = profile_resolve_dtype(dtype)?;
        let start_v = profile_finite_scalar_to_f64(
            start,
            ArrayApiErrorKind::NonFiniteInput,
            "arange start must be finite",
        )?;
        let stop_v = profile_finite_scalar_to_f64(
            stop,
            ArrayApiErrorKind::NonFiniteInput,
            "arange stop must be finite",
        )?;
        let step_v = profile_finite_scalar_to_f64(
            step,
            ArrayApiErrorKind::InvalidStep,
            "step must be finite",
        )?;
        if step_v == 0.0 {
            return Err(ArrayApiError::new(
                ArrayApiErrorKind::InvalidStep,
                "step must be nonzero",
            ));
        }

        let mut values = Vec::new();
        let mut current = start_v;
        if step_v > 0.0 {
            while current < stop_v {
                values.push(profile_cast_scalar(
                    ScalarValue::F64(current),
                    resolved_dtype,
                )?);
                current += step_v;
            }
        } else {
            while current > stop_v {
                values.push(profile_cast_scalar(
                    ScalarValue::F64(current),
                    resolved_dtype,
                )?);
                current += step_v;
            }
        }

        Ok(ProfileArray {
            shape: Shape::new(vec![values.len()]),
            dtype: resolved_dtype,
            values,
        })
    }

    fn linspace(
        &self,
        start: ScalarValue,
        stop: ScalarValue,
        num: usize,
        endpoint: bool,
        dtype: Option<DType>,
    ) -> ArrayApiResult<Self::Array> {
        let resolved_dtype = profile_resolve_dtype(dtype)?;
        let start_v = profile_scalar_to_f64(start)?;
        let stop_v = profile_scalar_to_f64(stop)?;
        if num == 0 {
            return Ok(ProfileArray {
                shape: Shape::new(vec![0]),
                dtype: resolved_dtype,
                values: Vec::new(),
            });
        }

        let step = if endpoint {
            if num == 1 {
                0.0
            } else {
                (stop_v - start_v) / (num.saturating_sub(1) as f64)
            }
        } else {
            (stop_v - start_v) / (num as f64)
        };

        let mut values = Vec::with_capacity(num);
        for idx in 0..num {
            let value = if endpoint && idx == num - 1 {
                stop_v
            } else {
                start_v + (idx as f64) * step
            };
            values.push(profile_cast_scalar(
                ScalarValue::F64(value),
                resolved_dtype,
            )?);
        }

        Ok(ProfileArray {
            shape: Shape::new(vec![values.len()]),
            dtype: resolved_dtype,
            values,
        })
    }

    fn getitem(&self, array: &Self::Array, index: &IndexExpr) -> ArrayApiResult<Self::Array> {
        match index {
            IndexExpr::Basic { slices } => profile_basic_getitem(array, slices),
            IndexExpr::Advanced { indices } => profile_advanced_getitem(array, indices),
            IndexExpr::BooleanMask { mask_shape } => {
                if *mask_shape != array.shape {
                    return Err(ArrayApiError::new(
                        ArrayApiErrorKind::InvalidShape,
                        "boolean index did not match indexed array shape",
                    ));
                }
                Ok(array.clone())
            }
        }
    }

    fn broadcast_to(&self, array: &Self::Array, shape: &Shape) -> ArrayApiResult<Self::Array> {
        let target_shape = broadcast_shapes(&[array.shape.clone(), shape.clone()])?;
        if target_shape != *shape {
            return Err(ArrayApiError::new(
                ArrayApiErrorKind::BroadcastIncompatible,
                "operands could not be broadcast together with the requested shape",
            ));
        }

        let output_size = profile_checked_size(shape)?;
        if output_size == 0 {
            return Ok(ProfileArray {
                shape: shape.clone(),
                dtype: array.dtype,
                values: Vec::new(),
            });
        }

        let out_rank = shape.rank();
        let in_rank = array.shape.rank();
        let in_strides = profile_row_major_strides(&array.shape.dims);
        let mut out_coords = vec![0usize; out_rank];
        let mut values = Vec::with_capacity(output_size);
        for linear in 0..output_size {
            let mut in_linear = 0usize;
            for (in_dim_idx, in_dim) in array.shape.dims.iter().enumerate() {
                let out_dim_idx = out_rank - in_rank + in_dim_idx;
                let coord = if *in_dim == 1 {
                    0
                } else {
                    out_coords[out_dim_idx]
                };
                in_linear += coord * in_strides[in_dim_idx];
            }
            values.push(array.values[in_linear]);
            if linear + 1 < output_size {
                profile_advance_row_major_coords(&mut out_coords, &shape.dims);
            }
        }

        Ok(ProfileArray {
            shape: shape.clone(),
            dtype: array.dtype,
            values,
        })
    }

    fn astype(&self, array: &Self::Array, dtype: DType) -> ArrayApiResult<Self::Array> {
        let resolved_dtype = profile_resolve_dtype(Some(dtype))?;
        let mut values = Vec::with_capacity(array.values.len());
        for value in &array.values {
            values.push(profile_cast_scalar(*value, resolved_dtype)?);
        }
        Ok(ProfileArray {
            shape: array.shape.clone(),
            dtype: resolved_dtype,
            values,
        })
    }

    fn result_type(&self, dtypes: &[DType], force_floating: bool) -> ArrayApiResult<DType> {
        if dtypes.is_empty() {
            return profile_resolve_dtype(None);
        }
        let mut resolved = if force_floating {
            profile_force_floating_dtype(dtypes[0])
        } else {
            dtypes[0]
        };
        for dtype in dtypes {
            let candidate = if force_floating {
                profile_force_floating_dtype(*dtype)
            } else {
                *dtype
            };
            resolved = profile_promote_dtype(resolved, candidate);
        }
        profile_resolve_dtype(Some(resolved))
    }

    fn array_from_slice(
        &self,
        values: &[ScalarValue],
        shape: &Shape,
        dtype: DType,
        _order: MemoryOrder,
    ) -> ArrayApiResult<Self::Array> {
        let resolved_dtype = profile_resolve_dtype(Some(dtype))?;
        let expected = profile_checked_size(shape)?;
        if expected != values.len() {
            return Err(ArrayApiError::new(
                ArrayApiErrorKind::InvalidShape,
                "input slice length does not match target shape",
            ));
        }

        let mut cast_values = Vec::with_capacity(values.len());
        for value in values {
            cast_values.push(profile_cast_scalar(*value, resolved_dtype)?);
        }

        Ok(ProfileArray {
            shape: shape.clone(),
            dtype: resolved_dtype,
            values: cast_values,
        })
    }

    fn reshape(&self, array: &Self::Array, new_shape: &Shape) -> ArrayApiResult<Self::Array> {
        let expected = profile_checked_size(new_shape)?;
        let actual = profile_checked_size(&array.shape)?;
        if expected != actual {
            return Err(ArrayApiError::new(
                ArrayApiErrorKind::InvalidShape,
                "cannot reshape array because element counts differ",
            ));
        }

        Ok(ProfileArray {
            shape: new_shape.clone(),
            dtype: array.dtype,
            values: array.values.clone(),
        })
    }

    fn transpose(&self, array: &Self::Array) -> ArrayApiResult<Self::Array> {
        if array.shape.rank() != 2 {
            return Err(ArrayApiError::new(
                ArrayApiErrorKind::InvalidShape,
                "transpose requires a rank-2 array",
            ));
        }

        let rows = array.shape.dims[0];
        let cols = array.shape.dims[1];
        let mut values = Vec::with_capacity(array.values.len());
        for col in 0..cols {
            for row in 0..rows {
                values.push(array.values[row * cols + col]);
            }
        }

        Ok(ProfileArray {
            shape: Shape::new(vec![cols, rows]),
            dtype: array.dtype,
            values,
        })
    }
}

fn make_sequence_values(len: usize) -> Vec<ScalarValue> {
    (0..len)
        .map(|idx| ScalarValue::F64((idx as f64) * 0.25 + 1.0))
        .collect()
}

fn make_array(backend: &CoreArrayBackend, shape: Shape, dtype: DType) -> CoreArray {
    let values = make_sequence_values(shape.element_count().expect("shape must not overflow"));
    let request = CreationRequest {
        shape,
        dtype,
        order: MemoryOrder::C,
    };
    from_slice(backend, &values, &request).expect("array construction should succeed")
}

fn profile_resolve_dtype(dtype: Option<DType>) -> ArrayApiResult<DType> {
    Ok(dtype.unwrap_or(DType::Float64))
}

fn profile_scalar_to_bool(value: ScalarValue) -> bool {
    match value {
        ScalarValue::Bool(v) => v,
        ScalarValue::I64(v) => v != 0,
        ScalarValue::U64(v) => v != 0,
        ScalarValue::F64(v) => v != 0.0,
        ScalarValue::ComplexF64 { re, im } => re != 0.0 || im != 0.0,
    }
}

fn profile_scalar_to_f64(value: ScalarValue) -> ArrayApiResult<f64> {
    match value {
        ScalarValue::Bool(v) => Ok(if v { 1.0 } else { 0.0 }),
        ScalarValue::I64(v) => Ok(v as f64),
        ScalarValue::U64(v) => Ok(v as f64),
        ScalarValue::F64(v) => Ok(v),
        ScalarValue::ComplexF64 { re, im } => {
            if im == 0.0 {
                Ok(re)
            } else {
                Err(ArrayApiError::new(
                    ArrayApiErrorKind::UnsupportedDtype,
                    "profile backend does not coerce complex values with nonzero imaginary part",
                ))
            }
        }
    }
}

fn profile_finite_scalar_to_f64(
    value: ScalarValue,
    kind: ArrayApiErrorKind,
    message: &'static str,
) -> ArrayApiResult<f64> {
    let converted = profile_scalar_to_f64(value)?;
    if converted.is_finite() {
        Ok(converted)
    } else {
        Err(ArrayApiError::new(kind, message))
    }
}

fn profile_scalar_to_complex_components(value: ScalarValue) -> ArrayApiResult<(f64, f64)> {
    match value {
        ScalarValue::Bool(v) => Ok((if v { 1.0 } else { 0.0 }, 0.0)),
        ScalarValue::I64(v) => Ok((v as f64, 0.0)),
        ScalarValue::U64(v) => Ok((v as f64, 0.0)),
        ScalarValue::F64(v) => Ok((v, 0.0)),
        ScalarValue::ComplexF64 { re, im } => Ok((re, im)),
    }
}

fn profile_cast_scalar(value: ScalarValue, dtype: DType) -> ArrayApiResult<ScalarValue> {
    match dtype {
        DType::Bool => Ok(ScalarValue::Bool(profile_scalar_to_bool(value))),
        DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64 => {
            Ok(ScalarValue::I64(profile_cast_signed_integer(value, dtype)?))
        }
        DType::UInt8 | DType::UInt16 | DType::UInt32 | DType::UInt64 => Ok(ScalarValue::U64(
            profile_cast_unsigned_integer(value, dtype)?,
        )),
        DType::Float32 => Ok(ScalarValue::F64(
            (profile_scalar_to_f64(value)? as f32) as f64,
        )),
        DType::Float64 => Ok(ScalarValue::F64(profile_scalar_to_f64(value)?)),
        DType::Complex64 => {
            let (re, im) = profile_scalar_to_complex_components(value)?;
            Ok(ScalarValue::ComplexF64 {
                re: (re as f32) as f64,
                im: (im as f32) as f64,
            })
        }
        DType::Complex128 => {
            let (re, im) = profile_scalar_to_complex_components(value)?;
            Ok(ScalarValue::ComplexF64 { re, im })
        }
    }
}

fn profile_cast_signed_integer(value: ScalarValue, dtype: DType) -> ArrayApiResult<i64> {
    let Some((min, max)) = profile_signed_integer_bounds(dtype) else {
        return Err(ArrayApiError::new(
            ArrayApiErrorKind::UnsupportedDtype,
            "dtype is not a signed integer",
        ));
    };
    let candidate = match value {
        ScalarValue::Bool(v) => i128::from(if v { 1_i64 } else { 0_i64 }),
        ScalarValue::I64(v) => i128::from(v),
        ScalarValue::U64(v) => i128::from(v),
        ScalarValue::F64(v) => profile_truncated_f64_to_i128(v)?,
        ScalarValue::ComplexF64 { .. } => {
            profile_truncated_f64_to_i128(profile_scalar_to_f64(value)?)?
        }
    };
    let lower = i128::from(min);
    let upper = i128::from(max);
    if candidate < lower || candidate > upper {
        return Err(ArrayApiError::new(
            ArrayApiErrorKind::Overflow,
            "scalar value is outside the requested signed integer dtype range",
        ));
    }
    Ok(candidate as i64)
}

fn profile_cast_unsigned_integer(value: ScalarValue, dtype: DType) -> ArrayApiResult<u64> {
    let Some(max) = profile_unsigned_integer_max(dtype) else {
        return Err(ArrayApiError::new(
            ArrayApiErrorKind::UnsupportedDtype,
            "dtype is not an unsigned integer",
        ));
    };
    let candidate = match value {
        ScalarValue::Bool(v) => u128::from(if v { 1_u64 } else { 0_u64 }),
        ScalarValue::I64(v) => {
            if v < 0 {
                return Err(ArrayApiError::new(
                    ArrayApiErrorKind::Overflow,
                    "negative scalar value cannot be cast to unsigned integer dtype",
                ));
            }
            u128::from(v as u64)
        }
        ScalarValue::U64(v) => u128::from(v),
        ScalarValue::F64(v) => profile_truncated_f64_to_u128(v)?,
        ScalarValue::ComplexF64 { .. } => {
            profile_truncated_f64_to_u128(profile_scalar_to_f64(value)?)?
        }
    };
    if candidate > u128::from(max) {
        return Err(ArrayApiError::new(
            ArrayApiErrorKind::Overflow,
            "scalar value is outside the requested unsigned integer dtype range",
        ));
    }
    Ok(candidate as u64)
}

fn profile_truncated_f64_to_i128(value: f64) -> ArrayApiResult<i128> {
    if !value.is_finite() {
        return Err(ArrayApiError::new(
            ArrayApiErrorKind::NonFiniteInput,
            "non-finite scalar cannot be cast to integer dtype",
        ));
    }
    Ok(value.trunc() as i128)
}

fn profile_truncated_f64_to_u128(value: f64) -> ArrayApiResult<u128> {
    if !value.is_finite() {
        return Err(ArrayApiError::new(
            ArrayApiErrorKind::NonFiniteInput,
            "non-finite scalar cannot be cast to integer dtype",
        ));
    }
    let truncated = value.trunc();
    if truncated < 0.0 {
        return Err(ArrayApiError::new(
            ArrayApiErrorKind::Overflow,
            "negative scalar value cannot be cast to unsigned integer dtype",
        ));
    }
    Ok(truncated as u128)
}

fn profile_signed_integer_bounds(dtype: DType) -> Option<(i64, i64)> {
    match dtype {
        DType::Int8 => Some((i64::from(i8::MIN), i64::from(i8::MAX))),
        DType::Int16 => Some((i64::from(i16::MIN), i64::from(i16::MAX))),
        DType::Int32 => Some((i64::from(i32::MIN), i64::from(i32::MAX))),
        DType::Int64 => Some((i64::MIN, i64::MAX)),
        _ => None,
    }
}

fn profile_unsigned_integer_max(dtype: DType) -> Option<u64> {
    match dtype {
        DType::UInt8 => Some(u64::from(u8::MAX)),
        DType::UInt16 => Some(u64::from(u16::MAX)),
        DType::UInt32 => Some(u64::from(u32::MAX)),
        DType::UInt64 => Some(u64::MAX),
        _ => None,
    }
}

fn profile_filled_array(
    shape: &Shape,
    fill_value: ScalarValue,
    dtype: DType,
) -> ArrayApiResult<ProfileArray> {
    let resolved_dtype = profile_resolve_dtype(Some(dtype))?;
    let size = shape.element_count().ok_or_else(|| {
        ArrayApiError::new(ArrayApiErrorKind::Overflow, "shape element count overflow")
    })?;
    let fill = profile_cast_scalar(fill_value, resolved_dtype)?;
    Ok(ProfileArray {
        shape: shape.clone(),
        dtype: resolved_dtype,
        values: vec![fill; size],
    })
}

fn profile_checked_size(shape: &Shape) -> ArrayApiResult<usize> {
    shape.element_count().ok_or_else(|| {
        ArrayApiError::new(ArrayApiErrorKind::Overflow, "shape element count overflow")
    })
}

fn profile_force_floating_dtype(dtype: DType) -> DType {
    match dtype {
        DType::Bool
        | DType::Int8
        | DType::Int16
        | DType::Int32
        | DType::Int64
        | DType::UInt8
        | DType::UInt16
        | DType::UInt32
        | DType::UInt64 => DType::Float64,
        _ => dtype,
    }
}

fn profile_promote_dtype(left: DType, right: DType) -> DType {
    if left == right {
        return left;
    }
    if matches!(left, DType::Complex128) || matches!(right, DType::Complex128) {
        return DType::Complex128;
    }
    if matches!(left, DType::Complex64) || matches!(right, DType::Complex64) {
        return DType::Complex64;
    }
    if matches!(left, DType::Float64 | DType::Float32)
        || matches!(right, DType::Float64 | DType::Float32)
    {
        return profile_promote_float_dtype(left, right);
    }
    profile_promote_integer_dtype(left, right)
}

fn profile_promote_float_dtype(left: DType, right: DType) -> DType {
    if matches!(left, DType::Float64)
        || matches!(right, DType::Float64)
        || profile_integer_kind(left).is_some()
        || profile_integer_kind(right).is_some()
    {
        DType::Float64
    } else {
        DType::Float32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProfileIntegerKind {
    Bool,
    Signed(u8),
    Unsigned(u8),
}

fn profile_integer_kind(dtype: DType) -> Option<ProfileIntegerKind> {
    match dtype {
        DType::Bool => Some(ProfileIntegerKind::Bool),
        DType::Int8 => Some(ProfileIntegerKind::Signed(8)),
        DType::Int16 => Some(ProfileIntegerKind::Signed(16)),
        DType::Int32 => Some(ProfileIntegerKind::Signed(32)),
        DType::Int64 => Some(ProfileIntegerKind::Signed(64)),
        DType::UInt8 => Some(ProfileIntegerKind::Unsigned(8)),
        DType::UInt16 => Some(ProfileIntegerKind::Unsigned(16)),
        DType::UInt32 => Some(ProfileIntegerKind::Unsigned(32)),
        DType::UInt64 => Some(ProfileIntegerKind::Unsigned(64)),
        _ => None,
    }
}

fn profile_promote_integer_dtype(left: DType, right: DType) -> DType {
    match (profile_integer_kind(left), profile_integer_kind(right)) {
        (Some(ProfileIntegerKind::Bool), Some(ProfileIntegerKind::Bool)) => DType::Bool,
        (Some(ProfileIntegerKind::Bool), Some(_)) => right,
        (Some(_), Some(ProfileIntegerKind::Bool)) => left,
        (
            Some(ProfileIntegerKind::Signed(left_bits)),
            Some(ProfileIntegerKind::Signed(right_bits)),
        ) => profile_signed_dtype_for_bits(left_bits.max(right_bits)),
        (
            Some(ProfileIntegerKind::Unsigned(left_bits)),
            Some(ProfileIntegerKind::Unsigned(right_bits)),
        ) => profile_unsigned_dtype_for_bits(left_bits.max(right_bits)),
        (
            Some(ProfileIntegerKind::Signed(signed_bits)),
            Some(ProfileIntegerKind::Unsigned(unsigned_bits)),
        )
        | (
            Some(ProfileIntegerKind::Unsigned(unsigned_bits)),
            Some(ProfileIntegerKind::Signed(signed_bits)),
        ) => profile_promote_mixed_integer_dtype(signed_bits, unsigned_bits),
        _ => DType::Float64,
    }
}

fn profile_signed_dtype_for_bits(bits: u8) -> DType {
    match bits {
        0..=8 => DType::Int8,
        9..=16 => DType::Int16,
        17..=32 => DType::Int32,
        _ => DType::Int64,
    }
}

fn profile_unsigned_dtype_for_bits(bits: u8) -> DType {
    match bits {
        0..=8 => DType::UInt8,
        9..=16 => DType::UInt16,
        17..=32 => DType::UInt32,
        _ => DType::UInt64,
    }
}

fn profile_promote_mixed_integer_dtype(signed_bits: u8, unsigned_bits: u8) -> DType {
    if signed_bits > unsigned_bits {
        return profile_signed_dtype_for_bits(signed_bits);
    }
    match unsigned_bits {
        0..=7 => DType::Int8,
        8..=15 => DType::Int16,
        16..=31 => DType::Int32,
        32..=63 => DType::Int64,
        _ => DType::Float64,
    }
}

fn profile_basic_slice_indices(spec: &SliceSpec, len: usize) -> ArrayApiResult<Vec<usize>> {
    if spec.step == 0 {
        return Err(ArrayApiError::new(
            ArrayApiErrorKind::InvalidStep,
            "slice step cannot be zero",
        ));
    }

    let len_i = len as isize;
    let step = spec.step;
    let mut start = spec.start.unwrap_or(if step > 0 { 0 } else { len_i - 1 });
    let mut stop = spec
        .stop
        .unwrap_or(if step > 0 { len_i } else { -len_i - 1 });

    if start < 0 {
        start += len_i;
    }
    if stop < 0 && spec.stop.is_some() {
        stop += len_i;
    }

    let mut out = Vec::new();
    if step > 0 {
        start = start.clamp(0, len_i);
        stop = stop.clamp(0, len_i);
        let mut cur = start;
        while cur < stop {
            out.push(cur as usize);
            cur += step;
        }
    } else {
        start = start.clamp(0, (len_i - 1).max(0));
        let mut cur = start;
        while cur > stop && cur >= 0 {
            if cur < len_i {
                out.push(cur as usize);
            }
            cur += step;
        }
    }
    Ok(out)
}

fn profile_basic_getitem(
    array: &ProfileArray,
    slices: &[SliceSpec],
) -> ArrayApiResult<ProfileArray> {
    match array.shape.rank() {
        1 => {
            if slices.len() != 1 {
                return Err(ArrayApiError::new(
                    ArrayApiErrorKind::InvalidIndex,
                    "1D indexing requires exactly one slice",
                ));
            }
            let indices = profile_basic_slice_indices(&slices[0], array.shape.dims[0])?;
            let mut values = Vec::with_capacity(indices.len());
            for idx in &indices {
                values.push(array.values[*idx]);
            }
            Ok(ProfileArray {
                shape: Shape::new(vec![indices.len()]),
                dtype: array.dtype,
                values,
            })
        }
        2 => {
            if slices.len() != 2 {
                return Err(ArrayApiError::new(
                    ArrayApiErrorKind::InvalidIndex,
                    "2D indexing requires two slices",
                ));
            }
            let rows = profile_basic_slice_indices(&slices[0], array.shape.dims[0])?;
            let cols = profile_basic_slice_indices(&slices[1], array.shape.dims[1])?;
            let mut values = Vec::with_capacity(rows.len().saturating_mul(cols.len()));
            let ncols = array.shape.dims[1];
            for row in &rows {
                for col in &cols {
                    values.push(array.values[row * ncols + col]);
                }
            }
            Ok(ProfileArray {
                shape: Shape::new(vec![rows.len(), cols.len()]),
                dtype: array.dtype,
                values,
            })
        }
        _ => Err(ArrayApiError::new(
            ArrayApiErrorKind::UnsupportedShape,
            "basic slicing is currently scoped to rank-1 and rank-2 arrays",
        )),
    }
}

fn profile_normalize_single_index(index: isize, len: usize) -> ArrayApiResult<usize> {
    let len_i = len as isize;
    let mut normalized = index;
    if normalized < 0 {
        normalized += len_i;
    }
    if normalized < 0 || normalized >= len_i {
        return Err(ArrayApiError::new(
            ArrayApiErrorKind::InvalidIndex,
            "index out of bounds",
        ));
    }
    Ok(normalized as usize)
}

fn profile_advanced_getitem(
    array: &ProfileArray,
    indices: &[Vec<isize>],
) -> ArrayApiResult<ProfileArray> {
    if array.shape.rank() != 1 || indices.len() != 1 {
        return Err(ArrayApiError::new(
            ArrayApiErrorKind::UnsupportedShape,
            "advanced indexing is currently scoped to 1D arrays",
        ));
    }
    let len = array.shape.dims[0];
    let mut values = Vec::with_capacity(indices[0].len());
    for index in &indices[0] {
        let normalized = profile_normalize_single_index(*index, len)?;
        values.push(array.values[normalized]);
    }
    Ok(ProfileArray {
        shape: Shape::new(vec![values.len()]),
        dtype: array.dtype,
        values,
    })
}

fn profile_row_major_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; dims.len()];
    let mut stride = 1usize;
    for pos in (0..dims.len()).rev() {
        strides[pos] = stride;
        stride = stride.saturating_mul(dims[pos].max(1));
    }
    strides
}

fn profile_advance_row_major_coords(coords: &mut [usize], dims: &[usize]) {
    debug_assert_eq!(coords.len(), dims.len());
    for axis in (0..coords.len()).rev() {
        coords[axis] += 1;
        if coords[axis] < dims[axis] {
            break;
        }
        coords[axis] = 0;
    }
}

fn legacy_unravel_index(mut index: usize, dims: &[usize]) -> Vec<usize> {
    if dims.is_empty() {
        return Vec::new();
    }
    let mut coords = vec![0usize; dims.len()];
    for pos in (0..dims.len()).rev() {
        let dim = dims[pos];
        if dim == 0 {
            coords[pos] = 0;
        } else {
            coords[pos] = index % dim;
            index /= dim;
        }
    }
    coords
}

fn legacy_ravel_index(coords: &[usize], dims: &[usize]) -> usize {
    let mut stride = 1usize;
    let mut index = 0usize;
    for pos in (0..dims.len()).rev() {
        index += coords[pos] * stride;
        stride = stride.saturating_mul(dims[pos].max(1));
    }
    index
}

fn legacy_broadcast_values(array: &CoreArray, out_shape: &Shape) -> Vec<ScalarValue> {
    let output_size = out_shape
        .element_count()
        .expect("broadcast output shape must not overflow");
    let out_rank = out_shape.rank();
    let in_shape = array.shape();
    let in_rank = in_shape.rank();
    let mut values = Vec::with_capacity(output_size);

    for linear in 0..output_size {
        let out_coords = legacy_unravel_index(linear, &out_shape.dims);
        let mut in_coords = vec![0usize; in_rank];
        for (in_dim_idx, in_dim) in in_shape.dims.iter().enumerate() {
            let out_dim_idx = out_rank - in_rank + in_dim_idx;
            in_coords[in_dim_idx] = if *in_dim == 1 {
                0
            } else {
                out_coords[out_dim_idx]
            };
        }
        let in_linear = legacy_ravel_index(&in_coords, &in_shape.dims);
        values.push(array.values()[in_linear]);
    }

    values
}

fn time_operation<F: FnMut()>(mut f: F) -> BenchStats {
    for _ in 0..WARMUP_ITERS {
        f();
    }

    let mut timings = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        f();
        timings.push(start.elapsed().as_nanos());
    }
    timings.sort_unstable();

    let median_idx = timings.len() / 2;
    let p95_idx = (timings.len() * 95).saturating_sub(1) / 100;

    BenchStats {
        median_ns: timings[median_idx],
        p95_ns: timings[p95_idx],
    }
}

fn estimated_alloc_count_delta(output_elements: usize) -> i64 {
    // Legacy loop: 1 values Vec + 2 temporary Vec allocations per output element.
    // Optimized loop: values Vec + in_strides Vec + out_coords Vec.
    // Delta is (after - before) in allocation events.
    let before = 1usize.saturating_add(output_elements.saturating_mul(2));
    let after = 3usize;
    let before_i64 = i64::try_from(before).expect("before allocation estimate fits i64");
    let after_i64 = i64::try_from(after).expect("after allocation estimate fits i64");
    after_i64 - before_i64
}

fn bench_broadcast_profile(rows: &mut Vec<BenchmarkRow>, iso_details: &mut Vec<IsomorphismDetail>) {
    let backend = strict_backend();

    for &dtype in DTYPES {
        for &size in SIZES {
            let input = make_array(&backend, Shape::new(vec![size, 1]), dtype);
            let out_shape = Shape::new(vec![size, 2]);
            let output_elements = out_shape.element_count().expect("shape must not overflow");

            let before_stats = time_operation(|| {
                let _ = legacy_broadcast_values(&input, &out_shape);
            });
            let after_stats = time_operation(|| {
                let _ = backend
                    .broadcast_to(&input, &out_shape)
                    .expect("optimized broadcast should succeed");
            });

            let legacy_values = legacy_broadcast_values(&input, &out_shape);
            let optimized = backend
                .broadcast_to(&input, &out_shape)
                .expect("optimized broadcast should succeed");
            let isomorphic = legacy_values == optimized.values();
            let note = if isomorphic {
                "legacy and optimized outputs are byte-identical".to_string()
            } else {
                format!(
                    "value mismatch: legacy_len={}, optimized_len={}",
                    legacy_values.len(),
                    optimized.values().len()
                )
            };
            iso_details.push(IsomorphismDetail {
                hotspot_function: HOTSPOT_BROADCAST.to_string(),
                array_size: size,
                dtype: format!("{dtype:?}"),
                passes: isomorphic,
                note,
            });

            rows.push(BenchmarkRow::new(
                HOTSPOT_BROADCAST,
                size,
                output_elements,
                dtype,
                before_stats,
                after_stats,
                estimated_alloc_count_delta(output_elements),
            ));
        }
    }
}

fn bench_creation_profile(rows: &mut Vec<BenchmarkRow>, iso_details: &mut Vec<IsomorphismDetail>) {
    let backend = strict_backend();
    let profile_backend = ProfileArrayBackend;

    for &dtype in DTYPES {
        for &size in SIZES {
            let request = CreationRequest {
                shape: Shape::new(vec![size]),
                dtype,
                order: MemoryOrder::C,
            };
            let full_request = FullRequest {
                fill_value: ScalarValue::F64(3.25),
                dtype,
                order: MemoryOrder::C,
            };

            let before_zero = time_operation(|| {
                let _ = zeros(&backend, &request).expect("zeros should succeed");
            });
            let after_zero = time_operation(|| {
                let _ = zeros(&profile_backend, &request).expect("profile zeros should succeed");
            });

            let current_zero = zeros(&backend, &request).expect("zeros should succeed");
            let profile_zero =
                zeros(&profile_backend, &request).expect("profile zeros should succeed");
            let zero_isomorphic = current_zero.shape() == profile_zero.shape()
                && current_zero.dtype() == profile_zero.dtype()
                && current_zero.values() == profile_zero.values();
            let zero_note = if zero_isomorphic {
                "current and log-free zeros paths match exactly".to_string()
            } else {
                format!(
                    "zeros mismatch: current_len={}, profile_len={}",
                    current_zero.values().len(),
                    profile_zero.values().len()
                )
            };
            iso_details.push(IsomorphismDetail {
                hotspot_function: HOTSPOT_ZEROS.to_string(),
                array_size: size,
                dtype: format!("{dtype:?}"),
                passes: zero_isomorphic,
                note: zero_note,
            });
            rows.push(BenchmarkRow::new(
                HOTSPOT_ZEROS,
                size,
                size,
                dtype,
                before_zero,
                after_zero,
                0,
            ));

            let before_full = time_operation(|| {
                let _ = full(&backend, &request.shape, &full_request).expect("full should succeed");
            });
            let after_full = time_operation(|| {
                let _ = full(&profile_backend, &request.shape, &full_request)
                    .expect("profile full should succeed");
            });

            let current_full =
                full(&backend, &request.shape, &full_request).expect("full should succeed");
            let profile_full = full(&profile_backend, &request.shape, &full_request)
                .expect("profile full should succeed");
            let full_isomorphic = current_full.shape() == profile_full.shape()
                && current_full.dtype() == profile_full.dtype()
                && current_full.values() == profile_full.values();
            let full_note = if full_isomorphic {
                "current and log-free full paths match exactly".to_string()
            } else {
                format!(
                    "full mismatch: current_len={}, profile_len={}",
                    current_full.values().len(),
                    profile_full.values().len()
                )
            };
            iso_details.push(IsomorphismDetail {
                hotspot_function: HOTSPOT_FULL.to_string(),
                array_size: size,
                dtype: format!("{dtype:?}"),
                passes: full_isomorphic,
                note: full_note,
            });
            rows.push(BenchmarkRow::new(
                HOTSPOT_FULL,
                size,
                size,
                dtype,
                before_full,
                after_full,
                0,
            ));
        }
    }
}

#[test]
fn profile_backend_scoped_operations_are_real() {
    let backend = ProfileArrayBackend;

    let aranged = backend
        .arange(
            ScalarValue::F64(0.0),
            ScalarValue::F64(4.0),
            ScalarValue::F64(1.0),
            Some(DType::Int32),
        )
        .expect("profile arange should construct a real array");
    assert_eq!(aranged.shape(), &Shape::new(vec![4]));
    assert_eq!(aranged.dtype(), DType::Int32);
    assert_eq!(
        aranged.values(),
        &[
            ScalarValue::I64(0),
            ScalarValue::I64(1),
            ScalarValue::I64(2),
            ScalarValue::I64(3),
        ]
    );

    let cast = backend
        .astype(&aranged, DType::Float64)
        .expect("profile astype should cast values");
    assert_eq!(
        cast.values(),
        &[
            ScalarValue::F64(0.0),
            ScalarValue::F64(1.0),
            ScalarValue::F64(2.0),
            ScalarValue::F64(3.0),
        ]
    );
    assert_eq!(
        backend
            .result_type(&[DType::Bool, DType::UInt16, DType::Float32], false)
            .expect("profile result_type should promote dtypes"),
        DType::Float64
    );
    assert_eq!(
        backend
            .result_type(&[DType::Int32], true)
            .expect("force_floating should promote integers"),
        DType::Float64
    );

    let linspace = backend
        .linspace(
            ScalarValue::F64(0.0),
            ScalarValue::F64(1.0),
            3,
            true,
            Some(DType::Complex128),
        )
        .expect("profile linspace should construct complex values");
    assert_eq!(
        linspace.values(),
        &[
            ScalarValue::ComplexF64 { re: 0.0, im: 0.0 },
            ScalarValue::ComplexF64 { re: 0.5, im: 0.0 },
            ScalarValue::ComplexF64 { re: 1.0, im: 0.0 },
        ]
    );

    let matrix = backend
        .array_from_slice(
            &[ScalarValue::F64(1.0), ScalarValue::F64(2.0)],
            &Shape::new(vec![2, 1]),
            DType::Float64,
            MemoryOrder::C,
        )
        .expect("profile array_from_slice should build matrix");
    let broadcast = backend
        .broadcast_to(&matrix, &Shape::new(vec![2, 3]))
        .expect("profile broadcast_to should repeat singleton axes");
    assert_eq!(broadcast.shape(), &Shape::new(vec![2, 3]));
    assert_eq!(
        broadcast.values(),
        &[
            ScalarValue::F64(1.0),
            ScalarValue::F64(1.0),
            ScalarValue::F64(1.0),
            ScalarValue::F64(2.0),
            ScalarValue::F64(2.0),
            ScalarValue::F64(2.0),
        ]
    );

    let sliced = backend
        .getitem(
            &broadcast,
            &IndexExpr::Basic {
                slices: vec![
                    SliceSpec {
                        start: Some(0),
                        stop: Some(2),
                        step: 1,
                    },
                    SliceSpec {
                        start: Some(1),
                        stop: Some(3),
                        step: 1,
                    },
                ],
            },
        )
        .expect("profile getitem should slice rank-2 arrays");
    assert_eq!(sliced.shape(), &Shape::new(vec![2, 2]));
    assert_eq!(
        sliced.values(),
        &[
            ScalarValue::F64(1.0),
            ScalarValue::F64(1.0),
            ScalarValue::F64(2.0),
            ScalarValue::F64(2.0),
        ]
    );

    let advanced = backend
        .getitem(
            &cast,
            &IndexExpr::Advanced {
                indices: vec![vec![3, 1, -1]],
            },
        )
        .expect("profile getitem should handle 1D advanced indexing");
    assert_eq!(
        advanced.values(),
        &[
            ScalarValue::F64(3.0),
            ScalarValue::F64(1.0),
            ScalarValue::F64(3.0),
        ]
    );

    let masked = backend
        .getitem(
            &cast,
            &IndexExpr::BooleanMask {
                mask_shape: cast.shape.clone(),
            },
        )
        .expect("matching boolean mask should pass through profile array");
    assert_eq!(masked.values(), cast.values());
}

#[test]
fn perf_p2c007_arrayapi_hotpath_profile() {
    let mut rows = Vec::new();
    let mut iso_details = Vec::new();

    bench_broadcast_profile(&mut rows, &mut iso_details);
    bench_creation_profile(&mut rows, &mut iso_details);

    let all_pass = iso_details.iter().all(|entry| entry.passes);
    assert!(
        all_pass,
        "arrayapi perf characterization changed observable values"
    );

    let report = PerfReport {
        generated_at: chrono_lite_now(),
        optimization_name: "broadcast optimization plus creation-path characterization"
            .to_string(),
        hotspot_function: HOTSPOT_ARRAYAPI.to_string(),
        benchmark_rows: rows,
        isomorphism_check: IsomorphismCheck {
            all_cases_pass: all_pass,
            details: iso_details,
        },
        methodology: vec![
            format!("warmup_iters={WARMUP_ITERS}"),
            format!("bench_iters={BENCH_ITERS}"),
            "broadcast before=legacy per-element unravel/ravel + per-iteration coordinate Vec allocations"
                .to_string(),
            "broadcast after=CoreArrayBackend::broadcast_to with precomputed strides and in-place coordinate advancement"
                .to_string(),
            "creation before=current CoreArrayBackend creation path with dtype dispatch logging enabled"
                .to_string(),
            "creation after=local log-free profiling backend matching scoped observable creation semantics"
                .to_string(),
            "broadcast alloc_count_delta is an algorithmic estimate (after - before), not allocator-instrumented"
                .to_string(),
            "creation alloc_count_delta is set to 0 because the characterization isolates hot-path logging overhead rather than allocator-instrumented buffer counts"
                .to_string(),
        ],
    };

    let artifact_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/artifacts/FSCI-P2C-007/perf");
    std::fs::create_dir_all(&artifact_dir).expect("perf artifact directory should be creatable");

    let pretty_json = serde_json::to_string_pretty(&report).expect("perf report should serialize");
    std::fs::write(artifact_dir.join("perf_profile_report.json"), pretty_json)
        .expect("perf profile report should be written");

    // Emit a machine-readable line so remote test runs can reconstruct the report locally.
    let compact_json =
        serde_json::to_string(&report).expect("compact perf report should serialize");
    println!("P2C007_PERF_REPORT_JSON={compact_json}");
}

/// Minimal timestamp without pulling in chrono.
fn chrono_lite_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_secs();
    format!("unix:{secs}")
}
