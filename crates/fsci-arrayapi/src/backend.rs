use crate::broadcast::broadcast_shapes;
use crate::dtype::default_float_dtype;
use crate::error::{ArrayApiError, ArrayApiErrorKind, ArrayApiResult};
use crate::types::{DType, ExecutionMode, IndexExpr, MemoryOrder, ScalarValue, Shape, SliceSpec};
use nalgebra::{DMatrix, DVector};
use std::collections::VecDeque;
use std::sync::Mutex;

const MAX_DIAGNOSTIC_LOGS: usize = 256;

pub trait ArrayApiArray {
    fn shape(&self) -> &Shape;

    fn dtype(&self) -> DType;

    fn ndim(&self) -> usize {
        self.shape().rank()
    }

    fn size(&self) -> usize {
        self.shape().element_count().unwrap_or(0)
    }
}

pub trait ArrayApiBackend {
    type Array: ArrayApiArray;

    fn namespace_name(&self) -> &'static str;

    fn shape_of(&self, array: &Self::Array) -> Shape;

    fn dtype_of(&self, array: &Self::Array) -> DType;

    fn asarray(
        &self,
        value: ScalarValue,
        dtype: Option<DType>,
        copy: Option<bool>,
    ) -> ArrayApiResult<Self::Array>;

    fn zeros(&self, shape: &Shape, dtype: DType, order: MemoryOrder)
    -> ArrayApiResult<Self::Array>;

    fn ones(&self, shape: &Shape, dtype: DType, order: MemoryOrder) -> ArrayApiResult<Self::Array>;

    fn empty(&self, shape: &Shape, dtype: DType, order: MemoryOrder)
    -> ArrayApiResult<Self::Array>;

    fn full(
        &self,
        shape: &Shape,
        fill_value: ScalarValue,
        dtype: DType,
        order: MemoryOrder,
    ) -> ArrayApiResult<Self::Array>;

    fn arange(
        &self,
        start: ScalarValue,
        stop: ScalarValue,
        step: ScalarValue,
        dtype: Option<DType>,
    ) -> ArrayApiResult<Self::Array>;

    fn linspace(
        &self,
        start: ScalarValue,
        stop: ScalarValue,
        num: usize,
        endpoint: bool,
        dtype: Option<DType>,
    ) -> ArrayApiResult<Self::Array>;

    fn getitem(&self, array: &Self::Array, index: &IndexExpr) -> ArrayApiResult<Self::Array>;

    fn broadcast_to(&self, array: &Self::Array, shape: &Shape) -> ArrayApiResult<Self::Array>;

    fn astype(&self, array: &Self::Array, dtype: DType) -> ArrayApiResult<Self::Array>;

    fn result_type(&self, dtypes: &[DType], force_floating: bool) -> ArrayApiResult<DType>;

    fn array_from_slice(
        &self,
        values: &[ScalarValue],
        shape: &Shape,
        dtype: DType,
        order: MemoryOrder,
    ) -> ArrayApiResult<Self::Array>;

    fn reshape(&self, array: &Self::Array, new_shape: &Shape) -> ArrayApiResult<Self::Array>;

    fn transpose(&self, array: &Self::Array) -> ArrayApiResult<Self::Array>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DTypeDispatchLog {
    pub requested_dtype: Option<DType>,
    pub resolved_dtype: DType,
    pub mode: ExecutionMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeMismatchLog {
    pub operation: &'static str,
    pub expected_shape: Shape,
    pub actual_shape: Shape,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CoreArray {
    shape: Shape,
    dtype: DType,
    values: Vec<ScalarValue>,
    order: MemoryOrder,
}

impl CoreArray {
    #[must_use]
    pub fn from_dmatrix(matrix: &DMatrix<f64>) -> Self {
        let shape = Shape::new(vec![matrix.nrows(), matrix.ncols()]);
        let mut values = Vec::with_capacity(matrix.nrows().saturating_mul(matrix.ncols()));
        for row in 0..matrix.nrows() {
            for col in 0..matrix.ncols() {
                values.push(ScalarValue::F64(matrix[(row, col)]));
            }
        }
        Self {
            shape,
            dtype: DType::Float64,
            values,
            order: MemoryOrder::C,
        }
    }

    pub fn to_dmatrix(&self) -> ArrayApiResult<DMatrix<f64>> {
        if self.shape.rank() != 2 {
            return Err(ArrayApiError::new(
                ArrayApiErrorKind::InvalidShape,
                "DMatrix conversion requires a 2D array",
            ));
        }

        let rows = self.shape.dims[0];
        let cols = self.shape.dims[1];
        let mut data = Vec::with_capacity(rows.saturating_mul(cols));
        for value in &self.values {
            data.push(scalar_to_f64(*value)?);
        }

        match self.order {
            MemoryOrder::F => Ok(DMatrix::from_column_slice(rows, cols, &data)),
            _ => Ok(DMatrix::from_row_slice(rows, cols, &data)),
        }
    }

    #[must_use]
    pub fn from_dvector(vector: &DVector<f64>) -> Self {
        let shape = Shape::new(vec![vector.len()]);
        let values = vector
            .iter()
            .map(|value| ScalarValue::F64(*value))
            .collect();
        Self {
            shape,
            dtype: DType::Float64,
            values,
            order: MemoryOrder::C,
        }
    }

    pub fn to_dvector(&self) -> ArrayApiResult<DVector<f64>> {
        if self.shape.rank() != 1 {
            return Err(ArrayApiError::new(
                ArrayApiErrorKind::InvalidShape,
                "DVector conversion requires a 1D array",
            ));
        }
        let mut data = Vec::with_capacity(self.values.len());
        for value in &self.values {
            data.push(scalar_to_f64(*value)?);
        }
        Ok(DVector::from_vec(data))
    }

    #[must_use]
    pub fn values(&self) -> &[ScalarValue] {
        &self.values
    }

    #[must_use]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    #[must_use]
    pub fn order(&self) -> MemoryOrder {
        self.order
    }

    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

impl ArrayApiArray for CoreArray {
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

#[derive(Debug)]
pub struct CoreArrayBackend {
    mode: ExecutionMode,
    dtype_dispatch_logs: Mutex<VecDeque<DTypeDispatchLog>>,
    shape_mismatch_logs: Mutex<VecDeque<ShapeMismatchLog>>,
}

impl CoreArrayBackend {
    #[must_use]
    pub fn new(mode: ExecutionMode) -> Self {
        Self {
            mode,
            dtype_dispatch_logs: Mutex::new(VecDeque::new()),
            shape_mismatch_logs: Mutex::new(VecDeque::new()),
        }
    }

    #[must_use]
    pub fn mode(&self) -> ExecutionMode {
        self.mode
    }

    #[must_use]
    pub fn dtype_dispatch_logs(&self) -> Vec<DTypeDispatchLog> {
        Self::lock_or_recover(&self.dtype_dispatch_logs)
            .iter()
            .cloned()
            .collect()
    }

    #[must_use]
    pub fn shape_mismatch_logs(&self) -> Vec<ShapeMismatchLog> {
        Self::lock_or_recover(&self.shape_mismatch_logs)
            .iter()
            .cloned()
            .collect()
    }

    /// Acquire a log mutex, restoring it to a healthy unpoisoned state
    /// before handing back the guard. Resolves [frankenscipy-74g9v]:
    /// the previous `.lock().unwrap_or_else(|e| e.into_inner())` pattern
    /// recovered the data but left the mutex permanently poisoned, so
    /// is_poisoned() stayed true forever and every subsequent call had
    /// to walk the Err branch. Mirrors the workspace audit-ledger
    /// pattern (clear_poison + into_inner).
    fn lock_or_recover<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
        match m.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                m.clear_poison();
                poisoned.into_inner()
            }
        }
    }

    fn record_dtype_dispatch(&self, requested_dtype: Option<DType>, resolved_dtype: DType) {
        let mut logs = Self::lock_or_recover(&self.dtype_dispatch_logs);
        push_bounded_log(
            &mut logs,
            DTypeDispatchLog {
                requested_dtype,
                resolved_dtype,
                mode: self.mode,
            },
        );
    }

    fn record_shape_mismatch(
        &self,
        operation: &'static str,
        expected_shape: Shape,
        actual_shape: Shape,
    ) {
        let mut logs = Self::lock_or_recover(&self.shape_mismatch_logs);
        push_bounded_log(
            &mut logs,
            ShapeMismatchLog {
                operation,
                expected_shape,
                actual_shape,
            },
        );
    }

    fn resolve_supported_dtype(&self, requested: Option<DType>) -> ArrayApiResult<DType> {
        let resolved = requested.unwrap_or(default_float_dtype(self.mode));
        if !is_scoped_dtype(resolved) {
            return Err(ArrayApiError::new(
                ArrayApiErrorKind::UnsupportedDtype,
                "only f32/f64/complex64/complex128 are supported",
            ));
        }
        self.record_dtype_dispatch(requested, resolved);
        Ok(resolved)
    }
}

impl Default for CoreArrayBackend {
    fn default() -> Self {
        Self::new(ExecutionMode::Strict)
    }
}

impl ArrayApiBackend for CoreArrayBackend {
    type Array = CoreArray;

    fn namespace_name(&self) -> &'static str {
        "array_api"
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
        let resolved_dtype = self.resolve_supported_dtype(dtype)?;
        let cast = cast_scalar_to_dtype(value, resolved_dtype)?;
        Ok(CoreArray {
            shape: Shape::scalar(),
            dtype: resolved_dtype,
            values: vec![cast],
            order: MemoryOrder::C,
        })
    }

    fn zeros(
        &self,
        shape: &Shape,
        dtype: DType,
        order: MemoryOrder,
    ) -> ArrayApiResult<Self::Array> {
        let resolved_dtype = self.resolve_supported_dtype(Some(dtype))?;
        let size = checked_size(shape)?;
        let values = filled_values(size, ScalarValue::F64(0.0), resolved_dtype)?;
        Ok(CoreArray {
            shape: shape.clone(),
            dtype: resolved_dtype,
            values,
            order,
        })
    }

    fn ones(&self, shape: &Shape, dtype: DType, order: MemoryOrder) -> ArrayApiResult<Self::Array> {
        let resolved_dtype = self.resolve_supported_dtype(Some(dtype))?;
        let size = checked_size(shape)?;
        let values = filled_values(size, ScalarValue::F64(1.0), resolved_dtype)?;
        Ok(CoreArray {
            shape: shape.clone(),
            dtype: resolved_dtype,
            values,
            order,
        })
    }

    fn empty(
        &self,
        shape: &Shape,
        dtype: DType,
        order: MemoryOrder,
    ) -> ArrayApiResult<Self::Array> {
        let resolved_dtype = self.resolve_supported_dtype(Some(dtype))?;
        let size = checked_size(shape)?;
        let values = filled_values(size, ScalarValue::F64(0.0), resolved_dtype)?;
        Ok(CoreArray {
            shape: shape.clone(),
            dtype: resolved_dtype,
            values,
            order,
        })
    }

    fn full(
        &self,
        shape: &Shape,
        fill_value: ScalarValue,
        dtype: DType,
        order: MemoryOrder,
    ) -> ArrayApiResult<Self::Array> {
        let resolved_dtype = self.resolve_supported_dtype(Some(dtype))?;
        let size = checked_size(shape)?;
        let values = filled_values(size, fill_value, resolved_dtype)?;
        Ok(CoreArray {
            shape: shape.clone(),
            dtype: resolved_dtype,
            values,
            order,
        })
    }

    fn arange(
        &self,
        start: ScalarValue,
        stop: ScalarValue,
        step: ScalarValue,
        dtype: Option<DType>,
    ) -> ArrayApiResult<Self::Array> {
        let resolved_dtype = self.resolve_supported_dtype(dtype)?;
        let start_v = finite_scalar_to_f64(
            start,
            ArrayApiErrorKind::NonFiniteInput,
            "arange start must be finite",
        )?;
        let stop_v = finite_scalar_to_f64(
            stop,
            ArrayApiErrorKind::NonFiniteInput,
            "arange stop must be finite",
        )?;
        let step_v =
            finite_scalar_to_f64(step, ArrayApiErrorKind::InvalidStep, "step must be finite")?;
        if step_v == 0.0 {
            return Err(ArrayApiError::new(
                ArrayApiErrorKind::InvalidStep,
                "step must be nonzero",
            ));
        }

        // numpy computes the length analytically as ceil((stop-start)/step),
        // then fills value[i] = start + i*step. Accumulating `current += step`
        // lets floating-point drift emit a spurious extra element (e.g.
        // arange(0, 1, 0.1) would yield 11 values instead of 10).
        let length = {
            let raw = (stop_v - start_v) / step_v;
            let n = raw.ceil();
            if n.is_finite() && n > 0.0 {
                n as usize
            } else {
                0
            }
        };

        let mut values = Vec::with_capacity(length);
        for i in 0..length {
            let current = start_v + (i as f64) * step_v;
            values.push(cast_scalar_to_dtype(
                ScalarValue::F64(current),
                resolved_dtype,
            )?);
        }

        Ok(CoreArray {
            shape: Shape::new(vec![values.len()]),
            dtype: resolved_dtype,
            values,
            order: MemoryOrder::C,
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
        let resolved_dtype = self.resolve_supported_dtype(dtype)?;
        let start_v = scalar_to_f64(start)?;
        let stop_v = scalar_to_f64(stop)?;
        if num == 0 {
            return Ok(CoreArray {
                shape: Shape::new(vec![0]),
                dtype: resolved_dtype,
                values: Vec::new(),
                order: MemoryOrder::C,
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
            // numpy pins the final sample to `stop` only when there is more
            // than one sample; linspace(start, stop, num=1) is [start].
            let value = if endpoint && num > 1 && idx == num - 1 {
                stop_v
            } else {
                start_v + (idx as f64) * step
            };
            values.push(cast_scalar_to_dtype(
                ScalarValue::F64(value),
                resolved_dtype,
            )?);
        }

        Ok(CoreArray {
            shape: Shape::new(vec![values.len()]),
            dtype: resolved_dtype,
            values,
            order: MemoryOrder::C,
        })
    }

    fn getitem(&self, array: &Self::Array, index: &IndexExpr) -> ArrayApiResult<Self::Array> {
        match index {
            IndexExpr::Basic { slices } => basic_getitem(array, slices, self.mode),
            IndexExpr::Advanced { indices } => advanced_getitem(array, indices, self.mode),
            IndexExpr::BooleanMask { mask_shape } => {
                if *mask_shape != array.shape {
                    self.record_shape_mismatch(
                        "boolean_mask",
                        array.shape.clone(),
                        mask_shape.clone(),
                    );
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
            self.record_shape_mismatch("broadcast_to", shape.clone(), array.shape.clone());
            return Err(ArrayApiError::new(
                ArrayApiErrorKind::BroadcastIncompatible,
                "operands could not be broadcast together with the requested shape",
            ));
        }

        let output_size = checked_size(shape)?;
        if output_size == 0 {
            return Ok(CoreArray {
                shape: shape.clone(),
                dtype: array.dtype,
                values: Vec::new(),
                order: array.order,
            });
        }
        if let Some(values) = rank2_broadcast_values(array, shape, output_size) {
            return Ok(CoreArray {
                shape: shape.clone(),
                dtype: array.dtype,
                values,
                order: array.order,
            });
        }

        let out_rank = shape.rank();
        let in_rank = array.shape.rank();
        let in_strides = row_major_strides(&array.shape.dims);
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
                advance_row_major_coords(&mut out_coords, &shape.dims);
            }
        }

        Ok(CoreArray {
            shape: shape.clone(),
            dtype: array.dtype,
            values,
            order: array.order,
        })
    }

    fn astype(&self, array: &Self::Array, dtype: DType) -> ArrayApiResult<Self::Array> {
        let resolved_dtype = self.resolve_supported_dtype(Some(dtype))?;
        let values = cast_values_to_dtype(&array.values, array.dtype, resolved_dtype)?;
        Ok(CoreArray {
            shape: array.shape.clone(),
            dtype: resolved_dtype,
            values,
            order: array.order,
        })
    }

    fn result_type(&self, dtypes: &[DType], force_floating: bool) -> ArrayApiResult<DType> {
        if dtypes.is_empty() {
            return self.resolve_supported_dtype(None);
        }
        let mut resolved = if force_floating {
            force_floating_dtype(dtypes[0])
        } else {
            dtypes[0]
        };
        for dtype in dtypes {
            let candidate = if force_floating {
                force_floating_dtype(*dtype)
            } else {
                *dtype
            };
            resolved = promote_dtype(resolved, candidate);
        }
        self.resolve_supported_dtype(Some(resolved))
    }

    fn array_from_slice(
        &self,
        values: &[ScalarValue],
        shape: &Shape,
        dtype: DType,
        order: MemoryOrder,
    ) -> ArrayApiResult<Self::Array> {
        let resolved_dtype = self.resolve_supported_dtype(Some(dtype))?;
        let expected = checked_size(shape)?;
        if expected != values.len() {
            self.record_shape_mismatch("from_slice", shape.clone(), Shape::new(vec![values.len()]));
            return Err(ArrayApiError::new(
                ArrayApiErrorKind::InvalidShape,
                "input slice length does not match target shape",
            ));
        }
        let mut cast_values = Vec::with_capacity(values.len());
        for value in values {
            cast_values.push(cast_scalar_to_dtype(*value, resolved_dtype)?);
        }
        Ok(CoreArray {
            shape: shape.clone(),
            dtype: resolved_dtype,
            values: cast_values,
            order,
        })
    }

    fn reshape(&self, array: &Self::Array, new_shape: &Shape) -> ArrayApiResult<Self::Array> {
        let expected = checked_size(new_shape)?;
        let actual = checked_size(&array.shape)?;
        if expected != actual {
            self.record_shape_mismatch("reshape", new_shape.clone(), array.shape.clone());
            return Err(ArrayApiError::new(
                ArrayApiErrorKind::InvalidShape,
                "cannot reshape array because element counts differ",
            ));
        }
        Ok(CoreArray {
            shape: new_shape.clone(),
            dtype: array.dtype,
            values: array.values.clone(),
            order: array.order,
        })
    }

    fn transpose(&self, array: &Self::Array) -> ArrayApiResult<Self::Array> {
        if array.shape.rank() != 2 {
            self.record_shape_mismatch("transpose", Shape::new(vec![2]), array.shape.clone());
            return Err(ArrayApiError::new(
                ArrayApiErrorKind::InvalidShape,
                "transpose requires a rank-2 array",
            ));
        }

        let rows = array.shape.dims[0];
        let cols = array.shape.dims[1];

        // Zero-copy transpose: swap dimensions and flip memory order
        let new_order = match array.order {
            MemoryOrder::C => MemoryOrder::F,
            MemoryOrder::F => MemoryOrder::C,
            _ => array.order, // Fallback for other orders
        };

        Ok(CoreArray {
            shape: Shape::new(vec![cols, rows]),
            dtype: array.dtype,
            values: array.values.clone(),
            order: new_order,
        })
    }
}

fn rank2_broadcast_values(
    array: &CoreArray,
    shape: &Shape,
    output_size: usize,
) -> Option<Vec<ScalarValue>> {
    if array.shape.rank() != 2 || shape.rank() != 2 {
        return None;
    }

    let in_rows = array.shape.dims[0];
    let in_cols = array.shape.dims[1];
    let rows = shape.dims[0];
    let cols = shape.dims[1];

    if in_rows == rows && in_cols == cols {
        return Some(array.values.clone());
    }

    if in_rows == 1 && in_cols == 1 {
        return Some(vec![array.values[0]; output_size]);
    }

    if in_rows == 1 && in_cols == cols {
        let mut values = Vec::with_capacity(output_size);
        for _ in 0..rows {
            values.extend_from_slice(&array.values);
        }
        return Some(values);
    }

    if in_rows == rows && in_cols == 1 {
        let mut values = Vec::with_capacity(output_size);
        for row in 0..rows {
            let value = array.values[row];
            for _ in 0..cols {
                values.push(value);
            }
        }
        return Some(values);
    }

    None
}

fn is_scoped_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::Bool
            | DType::Int8
            | DType::Int16
            | DType::Int32
            | DType::Int64
            | DType::UInt8
            | DType::UInt16
            | DType::UInt32
            | DType::UInt64
            | DType::Float32
            | DType::Float64
            | DType::Complex64
            | DType::Complex128
    )
}

fn checked_size(shape: &Shape) -> ArrayApiResult<usize> {
    shape.element_count().ok_or_else(|| {
        ArrayApiError::new(ArrayApiErrorKind::Overflow, "shape element count overflow")
    })
}

fn push_bounded_log<T>(logs: &mut VecDeque<T>, entry: T) {
    if logs.len() == MAX_DIAGNOSTIC_LOGS {
        logs.pop_front();
    }
    logs.push_back(entry);
}

fn filled_values(
    size: usize,
    fill_value: ScalarValue,
    dtype: DType,
) -> ArrayApiResult<Vec<ScalarValue>> {
    let fill = cast_scalar_to_dtype(fill_value, dtype)?;
    Ok(vec![fill; size])
}

fn cast_values_to_dtype(
    values: &[ScalarValue],
    source_dtype: DType,
    target_dtype: DType,
) -> ArrayApiResult<Vec<ScalarValue>> {
    match (source_dtype, target_dtype) {
        (DType::Float32 | DType::Float64, DType::Complex128) => {
            let mut out = Vec::with_capacity(values.len());
            for value in values {
                match *value {
                    ScalarValue::F64(re) => {
                        out.push(ScalarValue::ComplexF64 { re, im: 0.0 });
                    }
                    other => out.push(cast_scalar_to_dtype(other, target_dtype)?),
                }
            }
            Ok(out)
        }
        _ => {
            let mut out = Vec::with_capacity(values.len());
            for value in values {
                out.push(cast_scalar_to_dtype(*value, target_dtype)?);
            }
            Ok(out)
        }
    }
}

fn scalar_to_f64(value: ScalarValue) -> ArrayApiResult<f64> {
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
                    "cannot coerce complex value with nonzero imaginary part to real dtype",
                ))
            }
        }
    }
}

fn finite_scalar_to_f64(
    value: ScalarValue,
    kind: ArrayApiErrorKind,
    message: &'static str,
) -> ArrayApiResult<f64> {
    let converted = scalar_to_f64(value)?;
    if converted.is_finite() {
        Ok(converted)
    } else {
        Err(ArrayApiError::new(kind, message))
    }
}

fn cast_scalar_to_dtype(value: ScalarValue, dtype: DType) -> ArrayApiResult<ScalarValue> {
    match dtype {
        DType::Bool => Ok(ScalarValue::Bool(scalar_to_bool(value))),
        DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64 => Ok(ScalarValue::I64(
            cast_scalar_to_signed_integer(value, dtype)?,
        )),
        DType::UInt8 | DType::UInt16 | DType::UInt32 | DType::UInt64 => Ok(ScalarValue::U64(
            cast_scalar_to_unsigned_integer(value, dtype)?,
        )),
        DType::Float32 => Ok(ScalarValue::F64((scalar_to_f64(value)? as f32) as f64)),
        DType::Float64 => Ok(ScalarValue::F64(scalar_to_f64(value)?)),
        DType::Complex64 => {
            let (re, im) = scalar_to_complex_components(value)?;
            Ok(ScalarValue::ComplexF64 {
                re: (re as f32) as f64,
                im: (im as f32) as f64,
            })
        }
        DType::Complex128 => {
            let (re, im) = scalar_to_complex_components(value)?;
            Ok(ScalarValue::ComplexF64 { re, im })
        }
    }
}

fn scalar_to_bool(value: ScalarValue) -> bool {
    match value {
        ScalarValue::Bool(v) => v,
        ScalarValue::I64(v) => v != 0,
        ScalarValue::U64(v) => v != 0,
        ScalarValue::F64(v) => v != 0.0,
        ScalarValue::ComplexF64 { re, im } => re != 0.0 || im != 0.0,
    }
}

fn cast_scalar_to_signed_integer(value: ScalarValue, dtype: DType) -> ArrayApiResult<i64> {
    let Some((min, max)) = signed_integer_bounds(dtype) else {
        return Err(ArrayApiError::new(
            ArrayApiErrorKind::UnsupportedDtype,
            "dtype is not a signed integer",
        ));
    };
    let candidate = match value {
        ScalarValue::Bool(v) => i128::from(if v { 1_i64 } else { 0_i64 }),
        ScalarValue::I64(v) => i128::from(v),
        ScalarValue::U64(v) => i128::from(v),
        ScalarValue::F64(v) => truncated_f64_to_i128(v)?,
        ScalarValue::ComplexF64 { .. } => truncated_f64_to_i128(scalar_to_f64(value)?)?,
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

fn cast_scalar_to_unsigned_integer(value: ScalarValue, dtype: DType) -> ArrayApiResult<u64> {
    let Some(max) = unsigned_integer_max(dtype) else {
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
        ScalarValue::F64(v) => truncated_f64_to_u128(v)?,
        ScalarValue::ComplexF64 { .. } => truncated_f64_to_u128(scalar_to_f64(value)?)?,
    };
    if candidate > u128::from(max) {
        return Err(ArrayApiError::new(
            ArrayApiErrorKind::Overflow,
            "scalar value is outside the requested unsigned integer dtype range",
        ));
    }
    Ok(candidate as u64)
}

fn truncated_f64_to_i128(value: f64) -> ArrayApiResult<i128> {
    if !value.is_finite() {
        return Err(ArrayApiError::new(
            ArrayApiErrorKind::NonFiniteInput,
            "non-finite scalar cannot be cast to integer dtype",
        ));
    }
    Ok(value.trunc() as i128)
}

fn truncated_f64_to_u128(value: f64) -> ArrayApiResult<u128> {
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

fn signed_integer_bounds(dtype: DType) -> Option<(i64, i64)> {
    match dtype {
        DType::Int8 => Some((i64::from(i8::MIN), i64::from(i8::MAX))),
        DType::Int16 => Some((i64::from(i16::MIN), i64::from(i16::MAX))),
        DType::Int32 => Some((i64::from(i32::MIN), i64::from(i32::MAX))),
        DType::Int64 => Some((i64::MIN, i64::MAX)),
        _ => None,
    }
}

fn unsigned_integer_max(dtype: DType) -> Option<u64> {
    match dtype {
        DType::UInt8 => Some(u64::from(u8::MAX)),
        DType::UInt16 => Some(u64::from(u16::MAX)),
        DType::UInt32 => Some(u64::from(u32::MAX)),
        DType::UInt64 => Some(u64::MAX),
        _ => None,
    }
}

fn scalar_to_complex_components(value: ScalarValue) -> ArrayApiResult<(f64, f64)> {
    match value {
        ScalarValue::Bool(v) => Ok((if v { 1.0 } else { 0.0 }, 0.0)),
        ScalarValue::I64(v) => Ok((v as f64, 0.0)),
        ScalarValue::U64(v) => Ok((v as f64, 0.0)),
        ScalarValue::F64(v) => Ok((v, 0.0)),
        ScalarValue::ComplexF64 { re, im } => Ok((re, im)),
    }
}

fn force_floating_dtype(dtype: DType) -> DType {
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

fn promote_dtype(left: DType, right: DType) -> DType {
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
        return promote_float_dtype(left, right);
    }
    promote_integer_dtype(left, right)
}

fn promote_float_dtype(left: DType, right: DType) -> DType {
    if matches!(left, DType::Float64)
        || matches!(right, DType::Float64)
        || integer_kind(left).is_some()
        || integer_kind(right).is_some()
    {
        DType::Float64
    } else {
        DType::Float32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IntegerKind {
    Bool,
    Signed(u8),
    Unsigned(u8),
}

fn integer_kind(dtype: DType) -> Option<IntegerKind> {
    match dtype {
        DType::Bool => Some(IntegerKind::Bool),
        DType::Int8 => Some(IntegerKind::Signed(8)),
        DType::Int16 => Some(IntegerKind::Signed(16)),
        DType::Int32 => Some(IntegerKind::Signed(32)),
        DType::Int64 => Some(IntegerKind::Signed(64)),
        DType::UInt8 => Some(IntegerKind::Unsigned(8)),
        DType::UInt16 => Some(IntegerKind::Unsigned(16)),
        DType::UInt32 => Some(IntegerKind::Unsigned(32)),
        DType::UInt64 => Some(IntegerKind::Unsigned(64)),
        _ => None,
    }
}

fn promote_integer_dtype(left: DType, right: DType) -> DType {
    match (integer_kind(left), integer_kind(right)) {
        (Some(IntegerKind::Bool), Some(IntegerKind::Bool)) => DType::Bool,
        (Some(IntegerKind::Bool), Some(_)) => right,
        (Some(_), Some(IntegerKind::Bool)) => left,
        (Some(IntegerKind::Signed(left_bits)), Some(IntegerKind::Signed(right_bits))) => {
            signed_dtype_for_bits(left_bits.max(right_bits))
        }
        (Some(IntegerKind::Unsigned(left_bits)), Some(IntegerKind::Unsigned(right_bits))) => {
            unsigned_dtype_for_bits(left_bits.max(right_bits))
        }
        (Some(IntegerKind::Signed(signed_bits)), Some(IntegerKind::Unsigned(unsigned_bits)))
        | (Some(IntegerKind::Unsigned(unsigned_bits)), Some(IntegerKind::Signed(signed_bits))) => {
            promote_mixed_integer_dtype(signed_bits, unsigned_bits)
        }
        _ => DType::Float64,
    }
}

fn signed_dtype_for_bits(bits: u8) -> DType {
    match bits {
        0..=8 => DType::Int8,
        9..=16 => DType::Int16,
        17..=32 => DType::Int32,
        _ => DType::Int64,
    }
}

fn unsigned_dtype_for_bits(bits: u8) -> DType {
    match bits {
        0..=8 => DType::UInt8,
        9..=16 => DType::UInt16,
        17..=32 => DType::UInt32,
        _ => DType::UInt64,
    }
}

fn promote_mixed_integer_dtype(signed_bits: u8, unsigned_bits: u8) -> DType {
    if signed_bits > unsigned_bits {
        return signed_dtype_for_bits(signed_bits);
    }
    match unsigned_bits {
        0..=7 => DType::Int8,
        8..=15 => DType::Int16,
        16..=31 => DType::Int32,
        32..=63 => DType::Int64,
        _ => DType::Float64,
    }
}

fn normalize_single_index(index: isize, len: usize, mode: ExecutionMode) -> ArrayApiResult<usize> {
    let len_i = len as isize;
    if mode == ExecutionMode::Hardened && (index < -len_i || index >= len_i) {
        return Err(ArrayApiError::new(
            ArrayApiErrorKind::InvalidIndex,
            "index out of bounds in hardened mode",
        ));
    }
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

fn basic_slice_indices(
    spec: &SliceSpec,
    len: usize,
    mode: ExecutionMode,
) -> ArrayApiResult<Vec<usize>> {
    if spec.step == 0 {
        return Err(ArrayApiError::new(
            ArrayApiErrorKind::InvalidStep,
            "slice step cannot be zero",
        ));
    }

    let len_i = len as isize;
    if mode == ExecutionMode::Hardened {
        for endpoint in [spec.start, spec.stop] {
            if let Some(value) = endpoint
                && (value < -len_i || value > len_i)
            {
                return Err(ArrayApiError::new(
                    ArrayApiErrorKind::InvalidIndex,
                    "slice bounds out of range in hardened mode",
                ));
            }
        }
    }

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
        // stop for negative step can be -1 to include index 0
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

fn basic_getitem(
    array: &CoreArray,
    slices: &[SliceSpec],
    mode: ExecutionMode,
) -> ArrayApiResult<CoreArray> {
    match array.shape.rank() {
        1 => {
            if slices.len() != 1 {
                return Err(ArrayApiError::new(
                    ArrayApiErrorKind::InvalidIndex,
                    "1D indexing requires exactly one slice",
                ));
            }
            let indices = basic_slice_indices(&slices[0], array.shape.dims[0], mode)?;
            let mut values = Vec::with_capacity(indices.len());
            for idx in &indices {
                values.push(array.values[*idx]);
            }
            Ok(CoreArray {
                shape: Shape::new(vec![indices.len()]),
                dtype: array.dtype,
                values,
                order: array.order,
            })
        }
        2 => {
            if slices.len() != 2 {
                return Err(ArrayApiError::new(
                    ArrayApiErrorKind::InvalidIndex,
                    "2D indexing requires two slices",
                ));
            }
            let rows = basic_slice_indices(&slices[0], array.shape.dims[0], mode)?;
            let cols = basic_slice_indices(&slices[1], array.shape.dims[1], mode)?;
            let mut values = Vec::with_capacity(rows.len().saturating_mul(cols.len()));
            let nrows = array.shape.dims[0];
            let ncols = array.shape.dims[1];
            let is_f_order = array.order == MemoryOrder::F;
            for row in &rows {
                for col in &cols {
                    let idx = if is_f_order {
                        col * nrows + row
                    } else {
                        row * ncols + col
                    };
                    values.push(array.values[idx]);
                }
            }
            Ok(CoreArray {
                shape: Shape::new(vec![rows.len(), cols.len()]),
                dtype: array.dtype,
                values,
                order: MemoryOrder::C, // Note: slicing conceptually returns C-order in this layout
            })
        }
        _ => Err(ArrayApiError::new(
            ArrayApiErrorKind::UnsupportedShape,
            "basic slicing is currently scoped to rank-1 and rank-2 arrays",
        )),
    }
}

fn advanced_getitem(
    array: &CoreArray,
    indices: &[Vec<isize>],
    mode: ExecutionMode,
) -> ArrayApiResult<CoreArray> {
    if array.shape.rank() != 1 || indices.len() != 1 {
        return Err(ArrayApiError::new(
            ArrayApiErrorKind::UnsupportedShape,
            "advanced indexing is currently scoped to 1D arrays",
        ));
    }
    let len = array.shape.dims[0];
    let mut values = Vec::with_capacity(indices[0].len());
    for index in &indices[0] {
        let normalized = normalize_single_index(*index, len, mode)?;
        values.push(array.values[normalized]);
    }
    Ok(CoreArray {
        shape: Shape::new(vec![values.len()]),
        dtype: array.dtype,
        values,
        order: array.order,
    })
}

fn row_major_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; dims.len()];
    let mut stride = 1usize;
    for pos in (0..dims.len()).rev() {
        strides[pos] = stride;
        stride = stride.saturating_mul(dims[pos].max(1));
    }
    strides
}

fn advance_row_major_coords(coords: &mut [usize], dims: &[usize]) {
    debug_assert_eq!(coords.len(), dims.len());
    for axis in (0..coords.len()).rev() {
        coords[axis] += 1;
        if coords[axis] < dims[axis] {
            break;
        }
        coords[axis] = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::broadcast::broadcast_shapes;
    use crate::creation::{
        ArangeRequest, CreationRequest, FullRequest, LinspaceRequest, arange, empty, from_slice,
        full, linspace, ones, zeros,
    };
    use crate::indexing::{IndexRequest, IndexingMode, getitem, reshape, transpose};
    use proptest::prelude::*;
    use serde::Serialize;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn strict_backend() -> CoreArrayBackend {
        CoreArrayBackend::new(ExecutionMode::Strict)
    }

    fn hardened_backend() -> CoreArrayBackend {
        CoreArrayBackend::new(ExecutionMode::Hardened)
    }

    #[test]
    fn array_accessors_and_creation_flow() {
        let backend = strict_backend();
        let request = CreationRequest {
            shape: Shape::new(vec![2, 2]),
            dtype: DType::Float64,
            order: MemoryOrder::C,
        };

        let array = zeros(&backend, &request).expect("zeros should succeed");
        assert_eq!(array.shape(), &Shape::new(vec![2, 2]));
        assert_eq!(array.dtype(), DType::Float64);
        assert_eq!(array.ndim(), 2);
        assert_eq!(array.size(), 4);

        let log = backend.dtype_dispatch_logs();
        assert!(!log.is_empty());
        assert_eq!(log[0].requested_dtype, Some(DType::Float64));
        assert_eq!(log[0].resolved_dtype, DType::Float64);
        assert_eq!(log[0].mode, ExecutionMode::Strict);
    }

    #[test]
    fn from_slice_reshape_and_transpose_round_trip() {
        let backend = strict_backend();
        let request = CreationRequest {
            shape: Shape::new(vec![2, 3]),
            dtype: DType::Float64,
            order: MemoryOrder::C,
        };
        let values = [
            ScalarValue::F64(1.0),
            ScalarValue::F64(2.0),
            ScalarValue::F64(3.0),
            ScalarValue::F64(4.0),
            ScalarValue::F64(5.0),
            ScalarValue::F64(6.0),
        ];

        let array = from_slice(&backend, &values, &request).expect("from_slice should succeed");
        let transposed = transpose(&backend, &array).expect("transpose should succeed");
        assert_eq!(transposed.shape(), &Shape::new(vec![3, 2]));

        let reshaped =
            reshape(&backend, &array, &Shape::new(vec![3, 2])).expect("reshape should succeed");
        assert_eq!(reshaped.shape(), &Shape::new(vec![3, 2]));
    }

    #[test]
    fn broadcast_to_rank2_singletons_preserves_row_major_values() {
        let backend = strict_backend();

        let left = from_slice(
            &backend,
            &[
                ScalarValue::F64(1.0),
                ScalarValue::F64(2.0),
                ScalarValue::F64(3.0),
            ],
            &CreationRequest {
                shape: Shape::new(vec![3, 1]),
                dtype: DType::Float64,
                order: MemoryOrder::C,
            },
        )
        .expect("left column should build");
        let left_broadcast = backend
            .broadcast_to(&left, &Shape::new(vec![3, 2]))
            .expect("left column should broadcast");
        assert_eq!(
            left_broadcast.values(),
            &[
                ScalarValue::F64(1.0),
                ScalarValue::F64(1.0),
                ScalarValue::F64(2.0),
                ScalarValue::F64(2.0),
                ScalarValue::F64(3.0),
                ScalarValue::F64(3.0),
            ]
        );

        let right = from_slice(
            &backend,
            &[ScalarValue::F64(10.0), ScalarValue::F64(20.0)],
            &CreationRequest {
                shape: Shape::new(vec![1, 2]),
                dtype: DType::Float64,
                order: MemoryOrder::C,
            },
        )
        .expect("right row should build");
        let right_broadcast = backend
            .broadcast_to(&right, &Shape::new(vec![3, 2]))
            .expect("right row should broadcast");
        assert_eq!(
            right_broadcast.values(),
            &[
                ScalarValue::F64(10.0),
                ScalarValue::F64(20.0),
                ScalarValue::F64(10.0),
                ScalarValue::F64(20.0),
                ScalarValue::F64(10.0),
                ScalarValue::F64(20.0),
            ]
        );

        let scalar_2d = from_slice(
            &backend,
            &[ScalarValue::F64(7.0)],
            &CreationRequest {
                shape: Shape::new(vec![1, 1]),
                dtype: DType::Float64,
                order: MemoryOrder::C,
            },
        )
        .expect("rank-2 scalar should build");
        let scalar_broadcast = backend
            .broadcast_to(&scalar_2d, &Shape::new(vec![2, 3]))
            .expect("rank-2 scalar should broadcast");
        assert_eq!(scalar_broadcast.values(), &[ScalarValue::F64(7.0); 6]);
    }

    #[test]
    fn strict_vs_hardened_slice_bounds_behavior() {
        let strict = strict_backend();
        let hardened = hardened_backend();

        let request = CreationRequest {
            shape: Shape::new(vec![5]),
            dtype: DType::Float64,
            order: MemoryOrder::C,
        };
        let values = [
            ScalarValue::F64(1.0),
            ScalarValue::F64(2.0),
            ScalarValue::F64(3.0),
            ScalarValue::F64(4.0),
            ScalarValue::F64(5.0),
        ];
        let strict_array =
            from_slice(&strict, &values, &request).expect("strict from_slice should succeed");
        let hardened_array =
            from_slice(&hardened, &values, &request).expect("hardened from_slice should succeed");

        let request = IndexRequest {
            mode: IndexingMode::Basic,
            index: IndexExpr::Basic {
                slices: vec![SliceSpec {
                    start: Some(0),
                    stop: Some(20),
                    step: 1,
                }],
            },
        };

        let strict_out = getitem(&strict, &strict_array, &request).expect("strict slice clamps");
        assert_eq!(strict_out.shape(), &Shape::new(vec![5]));

        let hardened_error =
            getitem(&hardened, &hardened_array, &request).expect_err("hardened should reject");
        assert_eq!(hardened_error.kind, ArrayApiErrorKind::InvalidIndex);
    }

    #[test]
    fn basic_getitem_rejects_rank_three_arrays_with_unsupported_shape() {
        let backend = strict_backend();
        let request = CreationRequest {
            shape: Shape::new(vec![1, 1, 1]),
            dtype: DType::Float64,
            order: MemoryOrder::C,
        };
        let values = [ScalarValue::F64(42.0)];
        let array = from_slice(&backend, &values, &request).expect("from_slice should succeed");

        let index = IndexRequest {
            mode: IndexingMode::Basic,
            index: IndexExpr::Basic {
                slices: vec![
                    SliceSpec {
                        start: None,
                        stop: None,
                        step: 1,
                    },
                    SliceSpec {
                        start: None,
                        stop: None,
                        step: 1,
                    },
                    SliceSpec {
                        start: None,
                        stop: None,
                        step: 1,
                    },
                ],
            },
        };

        let err = getitem(&backend, &array, &index).expect_err("rank-3 slicing should fail");
        assert_eq!(err.kind, ArrayApiErrorKind::UnsupportedShape);
    }

    #[test]
    fn advanced_getitem_rejects_non_vector_arrays_with_unsupported_shape() {
        let backend = strict_backend();
        let request = CreationRequest {
            shape: Shape::new(vec![1, 2]),
            dtype: DType::Float64,
            order: MemoryOrder::C,
        };
        let values = [ScalarValue::F64(1.0), ScalarValue::F64(2.0)];
        let array = from_slice(&backend, &values, &request).expect("from_slice should succeed");

        let index = IndexRequest {
            mode: IndexingMode::Advanced,
            index: IndexExpr::Advanced {
                indices: vec![vec![0]],
            },
        };

        let err = getitem(&backend, &array, &index).expect_err("2D advanced indexing should fail");
        assert_eq!(err.kind, ArrayApiErrorKind::UnsupportedShape);
    }

    #[test]
    fn nalgebra_dmatrix_and_dvector_round_trip() {
        let matrix = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let array = CoreArray::from_dmatrix(&matrix);
        let matrix_back = array.to_dmatrix().expect("dmatrix conversion");
        assert_eq!(matrix_back, matrix);

        let vector = DVector::from_vec(vec![10.0, 20.0, 30.0]);
        let vector_array = CoreArray::from_dvector(&vector);
        let vector_back = vector_array.to_dvector().expect("dvector conversion");
        assert_eq!(vector_back, vector);
    }

    #[test]
    fn shape_mismatch_logs_are_emitted() {
        let backend = strict_backend();
        let request = CreationRequest {
            shape: Shape::new(vec![2, 2]),
            dtype: DType::Float64,
            order: MemoryOrder::C,
        };
        let values = [
            ScalarValue::F64(1.0),
            ScalarValue::F64(2.0),
            ScalarValue::F64(3.0),
            ScalarValue::F64(4.0),
        ];
        let array = from_slice(&backend, &values, &request).expect("from_slice should succeed");
        let err = reshape(&backend, &array, &Shape::new(vec![3, 2])).expect_err("reshape mismatch");
        assert_eq!(err.kind, ArrayApiErrorKind::InvalidShape);

        let logs = backend.shape_mismatch_logs();
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].operation, "reshape");
        assert_eq!(logs[0].expected_shape, Shape::new(vec![3, 2]));
        assert_eq!(logs[0].actual_shape, Shape::new(vec![2, 2]));
    }

    #[test]
    fn dtype_dispatch_logs_are_bounded_to_recent_entries() {
        let backend = strict_backend();
        let dtypes = [
            DType::Float32,
            DType::Float64,
            DType::Complex64,
            DType::Complex128,
        ];
        let total = MAX_DIAGNOSTIC_LOGS + 5;

        for idx in 0..total {
            let request = CreationRequest {
                shape: Shape::new(vec![1]),
                dtype: dtypes[idx % dtypes.len()],
                order: MemoryOrder::C,
            };
            let _array = zeros(&backend, &request).expect("zeros should succeed");
        }

        let logs = backend.dtype_dispatch_logs();
        assert_eq!(logs.len(), MAX_DIAGNOSTIC_LOGS);
        assert_eq!(
            logs[0].requested_dtype,
            Some(dtypes[(total - MAX_DIAGNOSTIC_LOGS) % dtypes.len()])
        );
        assert_eq!(
            logs[MAX_DIAGNOSTIC_LOGS - 1].requested_dtype,
            Some(dtypes[(total - 1) % dtypes.len()])
        );
    }

    #[test]
    fn shape_mismatch_logs_are_bounded_to_recent_entries() {
        let backend = strict_backend();
        let request = CreationRequest {
            shape: Shape::new(vec![2, 2]),
            dtype: DType::Float64,
            order: MemoryOrder::C,
        };
        let values = [
            ScalarValue::F64(1.0),
            ScalarValue::F64(2.0),
            ScalarValue::F64(3.0),
            ScalarValue::F64(4.0),
        ];
        let array = from_slice(&backend, &values, &request).expect("from_slice should succeed");
        let total = MAX_DIAGNOSTIC_LOGS + 3;

        for idx in 0..total {
            let target_rows = idx + 3;
            let err = reshape(&backend, &array, &Shape::new(vec![target_rows, 2]))
                .expect_err("reshape mismatch should fail");
            assert_eq!(err.kind, ArrayApiErrorKind::InvalidShape);
        }

        let logs = backend.shape_mismatch_logs();
        assert_eq!(logs.len(), MAX_DIAGNOSTIC_LOGS);
        assert_eq!(logs[0].expected_shape, Shape::new(vec![6, 2]));
        assert_eq!(
            logs[MAX_DIAGNOSTIC_LOGS - 1].expected_shape,
            Shape::new(vec![total + 2, 2])
        );
    }

    #[test]
    fn integer_and_bool_dtypes_create_cast_and_promote() {
        let backend = strict_backend();

        let int_request = CreationRequest {
            shape: Shape::new(vec![2]),
            dtype: DType::Int64,
            order: MemoryOrder::C,
        };
        let int_array = zeros(&backend, &int_request).expect("int zeros should be supported");
        assert_eq!(int_array.dtype(), DType::Int64);
        assert_eq!(
            int_array.values(),
            &[ScalarValue::I64(0), ScalarValue::I64(0)]
        );

        let uint_request = CreationRequest {
            shape: Shape::new(vec![3]),
            dtype: DType::UInt8,
            order: MemoryOrder::C,
        };
        let uint_array = ones(&backend, &uint_request).expect("uint ones should be supported");
        assert_eq!(uint_array.dtype(), DType::UInt8);
        assert_eq!(
            uint_array.values(),
            &[
                ScalarValue::U64(1),
                ScalarValue::U64(1),
                ScalarValue::U64(1),
            ]
        );

        let bool_request = FullRequest {
            fill_value: ScalarValue::I64(2),
            dtype: DType::Bool,
            order: MemoryOrder::C,
        };
        let bool_array = full(&backend, &Shape::new(vec![1]), &bool_request)
            .expect("bool full should be supported");
        assert_eq!(bool_array.dtype(), DType::Bool);
        assert_eq!(bool_array.values(), &[ScalarValue::Bool(true)]);

        let float_request = CreationRequest {
            shape: Shape::new(vec![2]),
            dtype: DType::Float64,
            order: MemoryOrder::C,
        };
        let float_values = [ScalarValue::F64(1.9), ScalarValue::F64(-2.1)];
        let float_array =
            from_slice(&backend, &float_values, &float_request).expect("float source");
        let cast_array = backend
            .astype(&float_array, DType::Int32)
            .expect("integer astype should truncate toward zero");
        assert_eq!(cast_array.dtype(), DType::Int32);
        assert_eq!(
            cast_array.values(),
            &[ScalarValue::I64(1), ScalarValue::I64(-2)]
        );

        let narrow_overflow = FullRequest {
            fill_value: ScalarValue::I64(128),
            dtype: DType::Int8,
            order: MemoryOrder::C,
        };
        let err = full(&backend, &Shape::new(vec![1]), &narrow_overflow)
            .expect_err("out-of-range int8 cast should fail");
        assert_eq!(err.kind, ArrayApiErrorKind::Overflow);

        let unsigned_overflow = FullRequest {
            fill_value: ScalarValue::I64(-1),
            dtype: DType::UInt8,
            order: MemoryOrder::C,
        };
        let err = full(&backend, &Shape::new(vec![1]), &unsigned_overflow)
            .expect_err("negative unsigned cast should fail");
        assert_eq!(err.kind, ArrayApiErrorKind::Overflow);

        assert_eq!(
            backend
                .result_type(&[DType::Bool, DType::Int32], false)
                .expect("bool should promote to int"),
            DType::Int32
        );
        assert_eq!(
            backend
                .result_type(&[DType::Int8, DType::UInt8], false)
                .expect("mixed int should widen"),
            DType::Int16
        );
        assert_eq!(
            backend
                .result_type(&[DType::Int64, DType::UInt64], false)
                .expect("unrepresentable mixed int should widen to float"),
            DType::Float64
        );
        assert_eq!(
            backend
                .result_type(&[DType::Int32], true)
                .expect("force floating should still coerce ints"),
            DType::Float64
        );

        let full_request = crate::creation::FullRequest {
            fill_value: ScalarValue::Bool(true),
            dtype: DType::Complex128,
            order: MemoryOrder::C,
        };
        let ok = full(&backend, &Shape::new(vec![1]), &full_request);
        assert!(ok.is_ok());
    }

    #[test]
    fn backend_arange_rejects_non_finite_bounds_and_step() {
        let backend = strict_backend();

        let start_err = backend
            .arange(
                ScalarValue::F64(f64::NAN),
                ScalarValue::F64(3.0),
                ScalarValue::F64(1.0),
                Some(DType::Float64),
            )
            .expect_err("non-finite start must fail");
        assert_eq!(start_err.kind, ArrayApiErrorKind::NonFiniteInput);

        let stop_err = backend
            .arange(
                ScalarValue::F64(0.0),
                ScalarValue::F64(f64::INFINITY),
                ScalarValue::F64(1.0),
                Some(DType::Float64),
            )
            .expect_err("non-finite stop must fail");
        assert_eq!(stop_err.kind, ArrayApiErrorKind::NonFiniteInput);

        let step_err = backend
            .arange(
                ScalarValue::F64(0.0),
                ScalarValue::F64(3.0),
                ScalarValue::F64(f64::NAN),
                Some(DType::Float64),
            )
            .expect_err("non-finite step must fail");
        assert_eq!(step_err.kind, ArrayApiErrorKind::InvalidStep);
    }

    #[derive(Debug, Serialize)]
    struct StructuredTestLog {
        test_id: String,
        operation: String,
        shape: Vec<usize>,
        dtype: String,
        pass: bool,
        seed: u64,
        timestamp_ms: u128,
    }

    fn now_unix_ms() -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic from unix epoch")
            .as_millis()
    }

    fn emit_structured_log(log: &StructuredTestLog) {
        println!(
            "{}",
            serde_json::to_string(log).expect("structured log should serialize")
        );
    }

    fn scoped_dtypes() -> [DType; 4] {
        [
            DType::Float32,
            DType::Float64,
            DType::Complex64,
            DType::Complex128,
        ]
    }

    fn scoped_orders() -> [MemoryOrder; 4] {
        [
            MemoryOrder::C,
            MemoryOrder::F,
            MemoryOrder::A,
            MemoryOrder::K,
        ]
    }

    fn scalar_is_zero(value: ScalarValue) -> bool {
        match value {
            ScalarValue::Bool(flag) => !flag,
            ScalarValue::I64(v) => v == 0,
            ScalarValue::U64(v) => v == 0,
            ScalarValue::F64(v) => v == 0.0,
            ScalarValue::ComplexF64 { re, im } => re == 0.0 && im == 0.0,
        }
    }

    #[test]
    fn generated_unit_matrix_with_structured_logs() {
        let strict = strict_backend();
        let hardened = hardened_backend();
        let shapes = vec![
            Shape::scalar(),
            Shape::new(vec![0]),
            Shape::new(vec![1]),
            Shape::new(vec![2]),
            Shape::new(vec![3]),
            Shape::new(vec![2, 2]),
            Shape::new(vec![2, 3]),
            Shape::new(vec![1, 4]),
        ];
        let seed = 18_500_u64;
        let mut total = 0usize;
        let mut passed = 0usize;
        let mut case_counter = 0usize;

        let mut record = |operation: &str, shape: &[usize], dtype: DType, pass: bool| {
            case_counter += 1;
            total += 1;
            if pass {
                passed += 1;
            }
            emit_structured_log(&StructuredTestLog {
                test_id: format!("matrix-{case_counter:04}"),
                operation: operation.to_owned(),
                shape: shape.to_vec(),
                dtype: format!("{dtype:?}"),
                pass,
                seed,
                timestamp_ms: now_unix_ms(),
            });
        };

        for shape in &shapes {
            for dtype in scoped_dtypes() {
                for order in scoped_orders() {
                    let request = CreationRequest {
                        shape: shape.clone(),
                        dtype,
                        order,
                    };

                    let zeros_result = zeros(&strict, &request);
                    let zeros_pass = zeros_result
                        .as_ref()
                        .is_ok_and(|array| array.shape() == shape && array.dtype() == dtype);
                    record("zeros", &shape.dims, dtype, zeros_pass);

                    let ones_result = ones(&strict, &request);
                    let ones_pass = ones_result
                        .as_ref()
                        .is_ok_and(|array| array.shape() == shape && array.dtype() == dtype);
                    record("ones", &shape.dims, dtype, ones_pass);

                    let empty_result = empty(&strict, &request);
                    let empty_pass = empty_result
                        .as_ref()
                        .is_ok_and(|array| array.shape() == shape && array.dtype() == dtype);
                    record("empty", &shape.dims, dtype, empty_pass);

                    let full_request = FullRequest {
                        fill_value: ScalarValue::F64(2.5),
                        dtype,
                        order,
                    };
                    let full_result = full(&strict, shape, &full_request);
                    let full_pass = full_result
                        .as_ref()
                        .is_ok_and(|array| array.shape() == shape && array.dtype() == dtype);
                    record("full", &shape.dims, dtype, full_pass);
                }
            }
        }

        for stop in [0_i64, 1, 2, 3, 7, 15] {
            for (start, step) in [(0_i64, 1_i64), (0, 2), (15, -1), (15, -2)] {
                for dtype in scoped_dtypes() {
                    let request = ArangeRequest {
                        start: ScalarValue::I64(start),
                        stop: ScalarValue::I64(stop),
                        step: ScalarValue::I64(step),
                        dtype: Some(dtype),
                    };
                    let result = arange(&strict, &request);
                    let expected_len = if step > 0 && start < stop {
                        ((stop - start) as usize).div_ceil(step as usize)
                    } else if step < 0 && start > stop {
                        ((start - stop) as usize).div_ceil(step.unsigned_abs() as usize)
                    } else {
                        0
                    };
                    let pass = result
                        .as_ref()
                        .is_ok_and(|array| array.size() == expected_len && array.dtype() == dtype);
                    record("arange", &[expected_len], dtype, pass);
                }
            }
        }

        for num in [0usize, 1, 2, 5, 10, 17] {
            for endpoint in [false, true] {
                for dtype in scoped_dtypes() {
                    let request = LinspaceRequest {
                        start: ScalarValue::F64(-3.5),
                        stop: ScalarValue::F64(7.25),
                        num,
                        endpoint,
                        dtype: Some(dtype),
                    };
                    let result = linspace(&strict, &request);
                    let pass = result
                        .as_ref()
                        .is_ok_and(|array| array.size() == num && array.dtype() == dtype);
                    record("linspace", &[num], dtype, pass);
                }
            }
        }

        let index_values: Vec<ScalarValue> = (0..10).map(|v| ScalarValue::F64(v as f64)).collect();
        let index_request = CreationRequest {
            shape: Shape::new(vec![10]),
            dtype: DType::Float64,
            order: MemoryOrder::C,
        };
        let strict_index_array = from_slice(&strict, &index_values, &index_request)
            .expect("strict index array should build");
        let hardened_index_array = from_slice(&hardened, &index_values, &index_request)
            .expect("hardened index array should build");
        for start in [-12_isize, -1, 0, 3, 10, 12] {
            for stop in [-12_isize, -1, 0, 5, 10, 12] {
                let request = IndexRequest {
                    mode: IndexingMode::Basic,
                    index: IndexExpr::Basic {
                        slices: vec![SliceSpec {
                            start: Some(start),
                            stop: Some(stop),
                            step: 1,
                        }],
                    },
                };

                let strict_pass = getitem(&strict, &strict_index_array, &request).is_ok();
                record("index_basic_strict", &[10], DType::Float64, strict_pass);

                let hardened_expected = (-10..=10).contains(&start) && (-10..=10).contains(&stop);
                let hardened_pass = getitem(&hardened, &hardened_index_array, &request).is_ok()
                    == hardened_expected;
                record("index_basic_hardened", &[10], DType::Float64, hardened_pass);
            }
        }

        let broadcast_cases = [
            (vec![2, 3], vec![1, 3], true),
            (vec![4, 1], vec![1, 5], true),
            (vec![0], vec![0], true),
            (vec![], vec![3, 2], true),
            (vec![2, 3], vec![3, 2], false),
            (vec![2, 1, 4], vec![3, 4], true),
            (vec![2, 1, 4], vec![2, 2], false),
            (vec![1], vec![1, 1, 1], true),
            (vec![5], vec![2, 5], true),
            (vec![2, 0], vec![1, 0], true),
        ];
        for (left, right, expected_ok) in broadcast_cases {
            let result = broadcast_shapes(&[Shape::new(left.clone()), Shape::new(right.clone())]);
            let pass = result.is_ok() == expected_ok;
            record("broadcast_shapes", &left, DType::Float64, pass);
        }

        for left in scoped_dtypes() {
            for right in scoped_dtypes() {
                let forward = strict.result_type(&[left, right], false);
                let reverse = strict.result_type(&[right, left], false);
                let pass = match (forward, reverse) {
                    (Ok(a), Ok(b)) => a == b,
                    _ => false,
                };
                record("dtype_promotion_commutative", &[2], left, pass);
            }
        }

        assert!(
            total >= 500,
            "expected at least 500 generated unit cases, got {total}"
        );
        let pass_ratio = (passed as f64) / (total as f64);
        assert!(
            pass_ratio >= 0.95,
            "expected >=95% matrix pass ratio, got {:.2}%",
            pass_ratio * 100.0
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 1000, .. ProptestConfig::default() })]

        #[test]
        fn prop_zeros_are_zero_float32(dims in proptest::collection::vec(0usize..4, 0..3)) {
            let backend = strict_backend();
            let request = CreationRequest {
                shape: Shape::new(dims),
                dtype: DType::Float32,
                order: MemoryOrder::C,
            };
            let array = zeros(&backend, &request).expect("zeros should succeed");
            prop_assert!(array.values().iter().all(|value| scalar_is_zero(*value)));
        }

        #[test]
        fn prop_zeros_are_zero_float64(dims in proptest::collection::vec(0usize..4, 0..3)) {
            let backend = strict_backend();
            let request = CreationRequest {
                shape: Shape::new(dims),
                dtype: DType::Float64,
                order: MemoryOrder::C,
            };
            let array = zeros(&backend, &request).expect("zeros should succeed");
            prop_assert!(array.values().iter().all(|value| scalar_is_zero(*value)));
        }

        #[test]
        fn prop_zeros_are_zero_complex64(dims in proptest::collection::vec(0usize..4, 0..3)) {
            let backend = strict_backend();
            let request = CreationRequest {
                shape: Shape::new(dims),
                dtype: DType::Complex64,
                order: MemoryOrder::C,
            };
            let array = zeros(&backend, &request).expect("zeros should succeed");
            prop_assert!(array.values().iter().all(|value| scalar_is_zero(*value)));
        }

        #[test]
        fn prop_zeros_are_zero_complex128(dims in proptest::collection::vec(0usize..4, 0..3)) {
            let backend = strict_backend();
            let request = CreationRequest {
                shape: Shape::new(dims),
                dtype: DType::Complex128,
                order: MemoryOrder::C,
            };
            let array = zeros(&backend, &request).expect("zeros should succeed");
            prop_assert!(array.values().iter().all(|value| scalar_is_zero(*value)));
        }

        #[test]
        fn prop_arange_len_matches_n(n in 0usize..256) {
            let backend = strict_backend();
            let request = ArangeRequest {
                start: ScalarValue::I64(0),
                stop: ScalarValue::I64(n as i64),
                step: ScalarValue::I64(1),
                dtype: Some(DType::Float64),
            };
            let array = arange(&backend, &request).expect("arange should succeed");
            prop_assert_eq!(array.size(), n);
        }

        #[test]
        fn prop_arange_len_matches_descending_n(n in 0usize..256) {
            let backend = strict_backend();
            let request = ArangeRequest {
                start: ScalarValue::I64(n as i64),
                stop: ScalarValue::I64(0),
                step: ScalarValue::I64(-1),
                dtype: Some(DType::Float64),
            };
            let array = arange(&backend, &request).expect("descending arange should succeed");
            prop_assert_eq!(array.size(), n);
        }

        #[test]
        fn prop_broadcast_shape_is_commutative(
            left_dims in proptest::collection::vec(0usize..4, 0..3),
            right_dims in proptest::collection::vec(0usize..4, 0..3)
        ) {
            let left = Shape::new(left_dims);
            let right = Shape::new(right_dims);
            let lr = broadcast_shapes(&[left.clone(), right.clone()]);
            let rl = broadcast_shapes(&[right, left]);
            prop_assert_eq!(lr.is_ok(), rl.is_ok());
            if let (Ok(lhs), Ok(rhs)) = (lr, rl) {
                prop_assert_eq!(lhs, rhs);
            }
        }

        #[test]
        fn prop_dtype_promotion_is_commutative(
            left_idx in 0usize..4,
            right_idx in 0usize..4
        ) {
            let backend = strict_backend();
            let dtypes = scoped_dtypes();
            let left = dtypes[left_idx];
            let right = dtypes[right_idx];
            let forward = backend.result_type(&[left, right], false).expect("forward promotion");
            let reverse = backend.result_type(&[right, left], false).expect("reverse promotion");
            prop_assert_eq!(forward, reverse);
        }

        #[test]
        fn prop_reshape_preserves_size(n in 0usize..128) {
            let backend = strict_backend();
            let values: Vec<ScalarValue> = (0..n).map(|value| ScalarValue::F64(value as f64)).collect();
            let source = CreationRequest {
                shape: Shape::new(vec![n]),
                dtype: DType::Float64,
                order: MemoryOrder::C,
            };
            let array = from_slice(&backend, &values, &source).expect("from_slice should succeed");
            let reshaped = reshape(&backend, &array, &Shape::new(vec![1, n])).expect("reshape should succeed");
            prop_assert_eq!(reshaped.size(), array.size());
        }

        #[test]
        fn prop_transpose_roundtrip_preserves_values(rows in 0usize..6, cols in 0usize..6) {
            let backend = strict_backend();
            let count = rows.saturating_mul(cols);
            let values: Vec<ScalarValue> = (0..count).map(|value| ScalarValue::F64(value as f64)).collect();
            let request = CreationRequest {
                shape: Shape::new(vec![rows, cols]),
                dtype: DType::Float64,
                order: MemoryOrder::C,
            };
            let array = from_slice(&backend, &values, &request).expect("from_slice should succeed");
            let transposed = transpose(&backend, &array).expect("transpose should succeed");
            let restored = transpose(&backend, &transposed).expect("transpose should succeed");
            prop_assert_eq!(restored.shape(), array.shape());
            prop_assert_eq!(restored.values(), array.values());
        }

        #[test]
        fn prop_hardened_slice_rejects_far_bounds(len in 1usize..32, offset in 1isize..16) {
            let backend = hardened_backend();
            let values: Vec<ScalarValue> = (0..len).map(|value| ScalarValue::F64(value as f64)).collect();
            let request = CreationRequest {
                shape: Shape::new(vec![len]),
                dtype: DType::Float64,
                order: MemoryOrder::C,
            };
            let array = from_slice(&backend, &values, &request).expect("from_slice should succeed");
            let index = IndexRequest {
                mode: IndexingMode::Basic,
                index: IndexExpr::Basic {
                    slices: vec![SliceSpec {
                        start: Some((len as isize) + offset),
                        stop: Some((len as isize) + offset + 1),
                        step: 1,
                    }],
                },
            };
            let err = getitem(&backend, &array, &index).expect_err("hardened should reject far bounds");
            prop_assert_eq!(err.kind, ArrayApiErrorKind::InvalidIndex);
        }

        #[test]
        fn prop_advanced_index_returns_selected_value(len in 1usize..64, idx_seed in 0usize..64) {
            let backend = strict_backend();
            let values: Vec<ScalarValue> = (0..len).map(|value| ScalarValue::F64(value as f64)).collect();
            let request = CreationRequest {
                shape: Shape::new(vec![len]),
                dtype: DType::Float64,
                order: MemoryOrder::C,
            };
            let array = from_slice(&backend, &values, &request).expect("from_slice should succeed");
            let idx = idx_seed % len;
            let index = IndexRequest {
                mode: IndexingMode::Advanced,
                index: IndexExpr::Advanced {
                    indices: vec![vec![idx as isize]],
                },
            };
            let selected = getitem(&backend, &array, &index).expect("advanced index should succeed");
            prop_assert_eq!(selected.size(), 1);
            prop_assert_eq!(selected.values()[0], ScalarValue::F64(idx as f64));
        }
    }

    #[test]
    fn test_negative_step_slicing() {
        let backend = strict_backend();
        let values: Vec<ScalarValue> = (0..5).map(|v| ScalarValue::F64(v as f64)).collect();
        let request = CreationRequest {
            shape: Shape::new(vec![5]),
            dtype: DType::Float64,
            order: MemoryOrder::C,
        };
        let array = from_slice(&backend, &values, &request).expect("from_slice");

        // arr[::-1]
        let index = IndexRequest {
            mode: IndexingMode::Basic,
            index: IndexExpr::Basic {
                slices: vec![SliceSpec {
                    start: None,
                    stop: None,
                    step: -1,
                }],
            },
        };
        let sliced = getitem(&backend, &array, &index).expect("slice ::-1");
        assert_eq!(sliced.size(), 5);
        let expected: Vec<ScalarValue> = vec![4.0, 3.0, 2.0, 1.0, 0.0]
            .into_iter()
            .map(ScalarValue::F64)
            .collect();
        assert_eq!(sliced.values(), &expected);
    }

    #[test]
    fn test_complex_slicing() {
        let backend = strict_backend();
        let values: Vec<ScalarValue> = (0..10).map(|v| ScalarValue::F64(v as f64)).collect();
        let request = CreationRequest {
            shape: Shape::new(vec![10]),
            dtype: DType::Float64,
            order: MemoryOrder::C,
        };
        let array = from_slice(&backend, &values, &request).expect("from_slice");

        // arr[1:8:2] -> [1, 3, 5, 7]
        let idx1 = IndexRequest {
            mode: IndexingMode::Basic,
            index: IndexExpr::Basic {
                slices: vec![SliceSpec {
                    start: Some(1),
                    stop: Some(8),
                    step: 2,
                }],
            },
        };
        let res1 = getitem(&backend, &array, &idx1).unwrap();
        assert_eq!(res1.values(), &[1.0, 3.0, 5.0, 7.0].map(ScalarValue::F64));

        // arr[8:1:-2] -> [8, 6, 4, 2]
        let idx2 = IndexRequest {
            mode: IndexingMode::Basic,
            index: IndexExpr::Basic {
                slices: vec![SliceSpec {
                    start: Some(8),
                    stop: Some(1),
                    step: -2,
                }],
            },
        };
        let res2 = getitem(&backend, &array, &idx2).unwrap();
        assert_eq!(res2.values(), &[8.0, 6.0, 4.0, 2.0].map(ScalarValue::F64));

        // arr[5:-1] -> [5, 6, 7, 8]
        let idx3 = IndexRequest {
            mode: IndexingMode::Basic,
            index: IndexExpr::Basic {
                slices: vec![SliceSpec {
                    start: Some(5),
                    stop: Some(-1),
                    step: 1,
                }],
            },
        };
        let res3 = getitem(&backend, &array, &idx3).unwrap();
        assert_eq!(res3.values(), &[5.0, 6.0, 7.0, 8.0].map(ScalarValue::F64));

        // arr[:-5:-1] -> [9, 8, 7, 6]
        let idx4 = IndexRequest {
            mode: IndexingMode::Basic,
            index: IndexExpr::Basic {
                slices: vec![SliceSpec {
                    start: None,
                    stop: Some(-5),
                    step: -1,
                }],
            },
        };
        let res4 = getitem(&backend, &array, &idx4).unwrap();
        assert_eq!(res4.values(), &[9.0, 8.0, 7.0, 6.0].map(ScalarValue::F64));
    }

    #[test]
    fn dispatch_log_reads_clear_poison_after_recovery() {
        // Regression for [frankenscipy-74g9v]: prior `.lock().unwrap_or_else(|e|
        // e.into_inner())` recovered the data but did not clear_poison.
        // After the lock_or_recover canonicalization, is_poisoned() must
        // be false on the fresh recovery branch and on subsequent calls.
        use std::sync::Arc;
        let backend = Arc::new(CoreArrayBackend::new(ExecutionMode::Strict));

        let bp = Arc::clone(&backend);
        let _ = std::thread::spawn(move || {
            let _g = bp
                .dtype_dispatch_logs
                .lock()
                .expect("acquire dtype_dispatch_logs");
            std::panic::resume_unwind(Box::new("intentional poison for 74g9v regression"));
        })
        .join();
        assert!(
            backend.dtype_dispatch_logs.is_poisoned(),
            "setup: dtype_dispatch_logs must be poisoned"
        );

        // First call after poison should clear it.
        let _ = backend.dtype_dispatch_logs();
        assert!(
            !backend.dtype_dispatch_logs.is_poisoned(),
            "lock_or_recover must clear poison on the read path"
        );
    }
}
