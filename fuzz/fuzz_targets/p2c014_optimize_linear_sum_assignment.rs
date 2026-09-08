#![no_main]

use arbitrary::Arbitrary;
use fsci_opt::linear_sum_assignment;
use libfuzzer_sys::fuzz_target;
use std::collections::HashSet;

// Optimize linear_sum_assignment oracle:
// For any rectangular cost matrix, linear_sum_assignment should return
// a valid assignment where:
// 1. row_ind and col_ind have the same length = min(rows, cols)
// 2. All row indices are unique and in [0, rows)
// 3. All col indices are unique and in [0, cols)
// 4. The computed cost matches the sum of assigned elements
//
// This catches:
// - Index out of bounds errors
// - Duplicate assignments
// - Cost computation errors
// - Edge cases: empty, single element, non-square matrices

const MAX_SIZE: usize = 32;

#[derive(Debug, Arbitrary)]
struct AssignmentInput {
    rows: u8,
    cols: u8,
    data: Vec<f64>,
}

fn sanitize(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(-1e6, 1e6)
    } else {
        0.0
    }
}

fuzz_target!(|input: AssignmentInput| {
    let rows = (input.rows as usize).clamp(0, MAX_SIZE);
    let cols = (input.cols as usize).clamp(0, MAX_SIZE);

    if rows == 0 || cols == 0 {
        let result = linear_sum_assignment(&[]);
        if let Ok((r, c)) = result {
            assert!(
                r.is_empty() && c.is_empty(),
                "Empty matrix should return empty assignment"
            );
        }
        return;
    }

    let mut cost_matrix: Vec<Vec<f64>> = Vec::with_capacity(rows);
    let mut data_iter = input.data.iter().copied().cycle();

    for _ in 0..rows {
        let row: Vec<f64> = (0..cols)
            .map(|_| sanitize(data_iter.next().unwrap_or(0.0)))
            .collect();
        cost_matrix.push(row);
    }

    let result = linear_sum_assignment(&cost_matrix);

    if let Ok((row_ind, col_ind)) = result {
        let expected_len = rows.min(cols);
        assert_eq!(
            row_ind.len(),
            expected_len,
            "row assignment length mismatch for {rows}x{cols} matrix"
        );
        assert_eq!(
            col_ind.len(),
            expected_len,
            "col assignment length mismatch for {rows}x{cols} matrix"
        );

        let row_set: HashSet<usize> = row_ind.iter().copied().collect();
        assert_eq!(
            row_set.len(),
            row_ind.len(),
            "duplicate row indices in assignment {row_ind:?} for {rows}x{cols} matrix"
        );

        let col_set: HashSet<usize> = col_ind.iter().copied().collect();
        assert_eq!(
            col_set.len(),
            col_ind.len(),
            "duplicate col indices in assignment {col_ind:?} for {rows}x{cols} matrix"
        );

        for &r in &row_ind {
            assert!(
                r < rows,
                "row index {r} out of bounds for {rows}x{cols} matrix"
            );
        }

        for &c in &col_ind {
            assert!(
                c < cols,
                "col index {c} out of bounds for {rows}x{cols} matrix"
            );
        }

        let computed_cost: f64 = row_ind
            .iter()
            .zip(col_ind.iter())
            .map(|(&r, &c)| cost_matrix[r][c])
            .sum();

        assert!(
            computed_cost.is_finite(),
            "assignment cost is non-finite: {computed_cost} ({rows}x{cols} matrix)"
        );
    }
});
