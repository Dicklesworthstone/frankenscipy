use std::collections::HashSet;

use crate::formats::{CooMatrix, CsrMatrix, Shape2D, SparseError, SparseResult};
use crate::ops::FormatConvertible;

pub fn eye(size: usize) -> SparseResult<CsrMatrix> {
    let shape = Shape2D::new(size, size);
    let data = vec![1.0; size];
    let rows: Vec<usize> = (0..size).collect();
    let cols = rows.clone();
    let coo = CooMatrix::from_triplets(shape, data, rows, cols, false)?;
    coo.to_csr()
}

pub fn diags(
    diagonals: &[Vec<f64>],
    offsets: &[isize],
    shape: Option<Shape2D>,
) -> SparseResult<CsrMatrix> {
    if diagonals.len() != offsets.len() {
        return Err(SparseError::InvalidArgument {
            message: "diagonals and offsets lengths must match".to_string(),
        });
    }

    let mut seen = HashSet::new();
    for &offset in offsets {
        if !seen.insert(offset) {
            return Err(SparseError::InvalidArgument {
                message: "repeated diagonal offsets are not allowed".to_string(),
            });
        }
    }

    let inferred = infer_shape(diagonals, offsets)?;
    let shape = shape.unwrap_or(inferred);

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    for (diag, &offset) in diagonals.iter().zip(offsets.iter()) {
        let start_row = if offset < 0 { (-offset) as usize } else { 0 };
        let start_col = if offset > 0 { offset as usize } else { 0 };
        for (k, &value) in diag.iter().enumerate() {
            let row = start_row + k;
            let col = start_col + k;
            if row >= shape.rows || col >= shape.cols {
                return Err(SparseError::InvalidShape {
                    message: "diagonal length exceeds matrix shape bounds".to_string(),
                });
            }
            rows.push(row);
            cols.push(col);
            data.push(value);
        }
    }

    let coo = CooMatrix::from_triplets(shape, data, rows, cols, true)?;
    coo.to_csr()
}

pub fn random(shape: Shape2D, density: f64, seed: u64) -> SparseResult<CooMatrix> {
    if !(0.0..=1.0).contains(&density) {
        return Err(SparseError::InvalidArgument {
            message: "density must be in [0.0, 1.0]".to_string(),
        });
    }
    let total = shape
        .rows
        .checked_mul(shape.cols)
        .ok_or_else(|| SparseError::IndexOverflow {
            message: "rows * cols overflows usize".to_string(),
        })?;

    let mut state = seed.max(1);
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    for index in 0..total {
        state = xorshift64(state);
        let sample = (state as f64) / (u64::MAX as f64);
        if sample <= density {
            let row = index / shape.cols.max(1);
            let col = index % shape.cols.max(1);
            state = xorshift64(state);
            let value = ((state as f64) / (u64::MAX as f64)) * 2.0 - 1.0;
            rows.push(row.min(shape.rows.saturating_sub(1)));
            cols.push(col.min(shape.cols.saturating_sub(1)));
            data.push(value);
        }
    }

    CooMatrix::from_triplets(shape, data, rows, cols, true)
}

fn infer_shape(diagonals: &[Vec<f64>], offsets: &[isize]) -> SparseResult<Shape2D> {
    let mut rows = 0usize;
    let mut cols = 0usize;
    for (diag, &offset) in diagonals.iter().zip(offsets.iter()) {
        let len = diag.len();
        if offset >= 0 {
            rows = rows.max(len);
            cols = cols.max(len + offset as usize);
        } else {
            let abs = (-offset) as usize;
            rows = rows.max(len + abs);
            cols = cols.max(len);
        }
    }
    if rows == 0 && cols == 0 {
        return Err(SparseError::InvalidShape {
            message: "cannot infer shape from empty diagonals".to_string(),
        });
    }
    Ok(Shape2D::new(rows, cols))
}

fn xorshift64(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::FormatConvertible;

    #[test]
    fn eye_constructs_expected_identity() {
        let id = eye(4).expect("identity");
        assert_eq!(id.shape(), Shape2D::new(4, 4));
        assert_eq!(id.nnz(), 4);

        let coo = id.to_coo().expect("csr->coo");
        for idx in 0..coo.nnz() {
            assert_eq!(coo.row_indices()[idx], coo.col_indices()[idx]);
            assert!((coo.data()[idx] - 1.0).abs() <= f64::EPSILON);
        }
    }

    #[test]
    fn eye_zero_size_is_empty() {
        let id = eye(0).expect("identity");
        assert_eq!(id.shape(), Shape2D::new(0, 0));
        assert_eq!(id.nnz(), 0);
    }

    #[test]
    fn diags_rejects_length_mismatch() {
        let err = diags(&[vec![1.0]], &[0, 1], None).expect_err("length mismatch");
        assert!(matches!(err, SparseError::InvalidArgument { .. }));
    }

    #[test]
    fn diags_rejects_repeated_offsets() {
        let err = diags(&[vec![1.0], vec![2.0]], &[0, 0], None).expect_err("repeated offsets");
        assert!(matches!(err, SparseError::InvalidArgument { .. }));
    }

    #[test]
    fn diags_rejects_empty_shape_inference() {
        let err = diags(&[], &[], None).expect_err("empty inference");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    #[test]
    fn diags_infers_shape_with_positive_and_negative_offsets() {
        let csr = diags(&[vec![1.0, 2.0, 3.0], vec![4.0, 5.0]], &[1, -2], None).expect("diags");
        assert_eq!(csr.shape(), Shape2D::new(4, 4));

        let dense = dense_from_csr(&csr);
        let expected = vec![
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 2.0, 0.0],
            vec![4.0, 0.0, 0.0, 3.0],
            vec![0.0, 5.0, 0.0, 0.0],
        ];
        assert_eq!(dense, expected);
    }

    #[test]
    fn diags_honors_explicit_shape() {
        let csr = diags(
            &[vec![1.0, 2.0], vec![3.0]],
            &[0, 2],
            Some(Shape2D::new(5, 5)),
        )
        .expect("diags");
        assert_eq!(csr.shape(), Shape2D::new(5, 5));
        assert_eq!(csr.nnz(), 3);
    }

    #[test]
    fn diags_rejects_out_of_bounds_for_explicit_shape() {
        let err = diags(&[vec![1.0, 2.0, 3.0]], &[1], Some(Shape2D::new(3, 3)))
            .expect_err("bounds violation");
        assert!(matches!(err, SparseError::InvalidShape { .. }));
    }

    #[test]
    fn random_rejects_density_out_of_range() {
        let shape = Shape2D::new(2, 2);
        assert!(matches!(
            random(shape, -0.1, 7),
            Err(SparseError::InvalidArgument { .. })
        ));
        assert!(matches!(
            random(shape, 1.1, 7),
            Err(SparseError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn random_rejects_shape_overflow() {
        let err = random(Shape2D::new(usize::MAX, 2), 0.5, 9).expect_err("overflow");
        assert!(matches!(err, SparseError::IndexOverflow { .. }));
    }

    #[test]
    fn random_density_zero_returns_empty_matrix() {
        let coo = random(Shape2D::new(4, 5), 0.0, 11).expect("random");
        assert_eq!(coo.shape(), Shape2D::new(4, 5));
        assert_eq!(coo.nnz(), 0);
    }

    #[test]
    fn random_density_one_fills_every_position() {
        let coo = random(Shape2D::new(3, 2), 1.0, 11).expect("random");
        assert_eq!(coo.nnz(), 6);
        for idx in 0..coo.nnz() {
            assert!(coo.row_indices()[idx] < 3);
            assert!(coo.col_indices()[idx] < 2);
            assert!((-1.0..=1.0).contains(&coo.data()[idx]));
        }
    }

    #[test]
    fn random_is_deterministic_for_same_seed() {
        let first = random(Shape2D::new(5, 4), 0.35, 12345).expect("random first");
        let second = random(Shape2D::new(5, 4), 0.35, 12345).expect("random second");
        assert_eq!(first.row_indices(), second.row_indices());
        assert_eq!(first.col_indices(), second.col_indices());
        assert_eq!(first.data(), second.data());
    }

    #[test]
    fn random_zero_dimension_returns_empty() {
        let coo = random(Shape2D::new(0, 7), 1.0, 99).expect("random");
        assert_eq!(coo.nnz(), 0);
    }

    #[test]
    fn xorshift_changes_nonzero_values() {
        assert_eq!(xorshift64(0), 0);
        assert_ne!(xorshift64(1), 1);
    }

    fn dense_from_csr(csr: &CsrMatrix) -> Vec<Vec<f64>> {
        let shape = csr.shape();
        let mut dense = vec![vec![0.0; shape.cols]; shape.rows];
        for (row, row_dense) in dense.iter_mut().enumerate().take(shape.rows) {
            for idx in csr.indptr()[row]..csr.indptr()[row + 1] {
                row_dense[csr.indices()[idx]] += csr.data()[idx];
            }
        }
        dense
    }
}
