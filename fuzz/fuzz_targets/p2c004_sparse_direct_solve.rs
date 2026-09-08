#![no_main]

use fsci_sparse::{
    CooMatrix, CsrMatrix, FormatConvertible, LuOptions, Shape2D, SolveOptions, SparseBackend, splu,
    splu_solve, spsolve,
};
use libfuzzer_sys::fuzz_target;

const MAX_SMALL_DIM: usize = 16;
const MAX_EXTRA_ENTRIES: usize = 96;
const NATIVE_DIM: usize = 32_769;
const EPS: f64 = 1e-7;

#[derive(Clone, Copy, Debug)]
struct Entry {
    row: usize,
    col: usize,
    value: f64,
}

fn normalized_dim(byte: u8) -> usize {
    usize::from(byte) % MAX_SMALL_DIM + 1
}

fn normalized_value(byte: u8) -> f64 {
    let signed = i16::from(byte) - 128;
    let value = f64::from(signed.rem_euclid(17) - 8);
    if value == 0.0 { 1.0 } else { value }
}

fn build_small_case(data: &[u8]) -> Option<(CsrMatrix, Vec<f64>)> {
    let (&dim_byte, rest) = data.split_first()?;
    let n = normalized_dim(dim_byte);
    let mut entries = Vec::with_capacity(n + rest.len().min(MAX_EXTRA_ENTRIES / 3));

    for row in 0..n {
        let diag = 8.0 + f64::from((row % 7) as u8);
        entries.push(Entry {
            row,
            col: row,
            value: diag,
        });
    }

    for chunk in rest.as_chunks::<3>().0.iter().take(MAX_EXTRA_ENTRIES / 3) {
        let row = usize::from(chunk[0]) % n;
        let col = usize::from(chunk[1]) % n;
        let value = normalized_value(chunk[2]) / 8.0;
        entries.push(Entry { row, col, value });
    }

    let rhs = (0..n)
        .map(|idx| {
            let byte = rest.get(idx).copied().unwrap_or(idx as u8);
            normalized_value(byte)
        })
        .collect::<Vec<_>>();
    let csr = csr_from_entries(n, &entries)?;
    Some((csr, rhs))
}

fn csr_from_entries(n: usize, entries: &[Entry]) -> Option<CsrMatrix> {
    let data = entries.iter().map(|entry| entry.value).collect::<Vec<_>>();
    let row_indices = entries.iter().map(|entry| entry.row).collect::<Vec<_>>();
    let col_indices = entries.iter().map(|entry| entry.col).collect::<Vec<_>>();
    let coo =
        CooMatrix::from_triplets(Shape2D::new(n, n), data, row_indices, col_indices, true).ok()?;
    coo.to_csr().ok()
}

fn build_large_native_case(data: &[u8]) -> Option<(CsrMatrix, Vec<f64>, Vec<usize>)> {
    let flavor = data.get(1).copied().unwrap_or(0) % 4;
    let rhs_scale = f64::from(data.get(2).copied().unwrap_or(1) % 9 + 1);
    let n = NATIVE_DIM;
    let mut values = Vec::with_capacity(n + 2);
    let mut cols = Vec::with_capacity(n + 2);
    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0);

    for row in 0..n {
        match (flavor, row) {
            (1, 0) => {
                values.push(2.0);
                cols.push(1);
            }
            (1, 1) => {
                values.push(1.0);
                cols.push(0);
                values.push(3.0);
                cols.push(1);
            }
            (2, _) if row + 1 < n => {
                values.push(1.0);
                cols.push(row);
                values.push(0.125);
                cols.push(row + 1);
            }
            (3, _) if row > 0 => {
                values.push(-0.125);
                cols.push(row - 1);
                values.push(1.0);
                cols.push(row);
            }
            _ => {
                values.push(1.0 + f64::from((row % 5) as u8));
                cols.push(row);
            }
        }
        indptr.push(values.len());
    }

    let csr = CsrMatrix::from_components(Shape2D::new(n, n), values, cols, indptr, false).ok()?;
    let rhs = (0..n)
        .map(|idx| rhs_scale + f64::from((idx % 11) as u8) / 4.0)
        .collect::<Vec<_>>();
    let sample_rows = vec![0, 1, 2, n / 2, n - 3, n - 2, n - 1];
    Some((csr, rhs, sample_rows))
}

fn sampled_residuals_are_small(matrix: &CsrMatrix, x: &[f64], b: &[f64], rows: &[usize]) -> bool {
    rows.iter().all(|&row| {
        let start = matrix.indptr()[row];
        let end = matrix.indptr()[row + 1];
        let ax = (start..end)
            .map(|idx| matrix.data()[idx] * x[matrix.indices()[idx]])
            .sum::<f64>();
        (ax - b[row]).abs() <= EPS * (1.0 + b[row].abs())
    })
}

fn all_residuals_are_small(matrix: &CsrMatrix, x: &[f64], b: &[f64]) -> bool {
    let rows = (0..matrix.shape().rows).collect::<Vec<_>>();
    sampled_residuals_are_small(matrix, x, b, &rows)
}

fn exercise_small_path(data: &[u8]) {
    let Some((matrix, rhs)) = build_small_case(data) else {
        return;
    };
    let Ok(solve_result) = spsolve(&matrix, &rhs, SolveOptions::default()) else {
        return;
    };
    assert_eq!(solve_result.solution.len(), rhs.len());
    assert!(
        all_residuals_are_small(&matrix, &solve_result.solution, &rhs),
        "small sparse-direct solve residual exceeded tolerance"
    );

    let Ok(csc) = matrix.to_csc() else {
        return;
    };
    let Ok(factorization) = splu(&csc, LuOptions::default()) else {
        return;
    };
    let Ok(factored_solution) = splu_solve(&factorization, &rhs) else {
        return;
    };
    assert_eq!(factored_solution.len(), solve_result.solution.len());
    for (left, right) in factored_solution.iter().zip(solve_result.solution.iter()) {
        assert!(
            (left - right).abs() <= EPS * (1.0 + right.abs()),
            "spsolve and splu_solve disagree"
        );
    }
}

fn exercise_native_path(data: &[u8]) {
    let Some((matrix, rhs, sample_rows)) = build_large_native_case(data) else {
        return;
    };
    let Ok(solve_result) = spsolve(&matrix, &rhs, SolveOptions::default()) else {
        return;
    };
    assert_eq!(solve_result.backend_used, SparseBackend::NativeSparseLu);
    assert!(
        sampled_residuals_are_small(&matrix, &solve_result.solution, &rhs, &sample_rows),
        "native sparse direct solve sampled residual exceeded tolerance"
    );

    let Ok(csc) = matrix.to_csc() else {
        return;
    };
    let Ok(factorization) = splu(&csc, LuOptions::default()) else {
        return;
    };
    assert_eq!(factorization.backend_used, SparseBackend::NativeSparseLu);
    let Ok(factored_solution) = splu_solve(&factorization, &rhs) else {
        return;
    };
    for &row in &sample_rows {
        assert!(
            (factored_solution[row] - solve_result.solution[row]).abs()
                <= EPS * (1.0 + solve_result.solution[row].abs()),
            "native spsolve and splu_solve disagree on sampled row {row}"
        );
    }
}

fuzz_target!(|data: &[u8]| {
    if data.first().copied() == Some(0x4e) {
        exercise_native_path(data);
    } else {
        exercise_small_path(data);
    }
});
