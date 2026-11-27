use fsci_sparse::{
    CsrMatrix, Shape2D, SparseError, SparseTriangularSolveOptions, sparse_transpose,
    spsolve_triangular, spsolve_triangular_many, spsolve_triangular_with_options,
};

fn check(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1e-13 * expected.abs().max(1.0),
            "expected {expected}, got {actual}"
        );
    }
}

#[test]
fn duplicate_diagonals_are_summed_in_both_triangles() {
    // [[5, 0], [3, 5]]; diagonals and the lower off-diagonal are duplicated.
    let lower = CsrMatrix::from_components(
        Shape2D::new(2, 2),
        vec![2., 3., 4., 2., -1., 3.],
        vec![0, 0, 0, 1, 0, 1],
        vec![0, 2, 6],
        false,
    )
    .unwrap();
    let rhs = [vec![10., 16.], vec![-5., 7.]];
    for (a, use_lower, expected) in [
        (lower.clone(), true, [[2., 2.], [-1., 2.]]),
        (sparse_transpose(&lower), false, [[0.08, 3.2], [-1.84, 1.4]]),
    ] {
        let options = SparseTriangularSolveOptions {
            lower: use_lower,
            ..Default::default()
        };
        let solutions = spsolve_triangular_many(&a, &rhs, options).unwrap();
        assert_eq!(solutions.len(), rhs.len());
        for ((b, solution), expected) in rhs.iter().zip(&solutions).zip(&expected) {
            check(solution, expected);
            check(&spsolve_triangular(&a, b, use_lower).unwrap(), expected);
            check(&a.matvec(solution).unwrap(), b);
        }
    }
}

#[test]
fn unit_diagonal_ignores_stored_pivots_and_supplies_absent_ones() {
    // The stored diagonal is zero, missing, and NaN, respectively.
    let lower = CsrMatrix::from_components(
        Shape2D::new(3, 3),
        vec![0., 2., -1., 3., f64::NAN],
        vec![0, 0, 0, 1, 2],
        vec![0, 1, 2, 5],
        true,
    )
    .unwrap();
    let rhs = [vec![1., 4., 9.], vec![-2., 0., 3.]];
    for (a, use_lower, expected) in [
        (lower.clone(), true, [[1., 2., 4.], [-2., 4., -11.]]),
        (
            sparse_transpose(&lower),
            false,
            [[56., -23., 9.], [19., -9., 3.]],
        ),
    ] {
        let options = SparseTriangularSolveOptions {
            lower: use_lower,
            unit_diagonal: true,
        };
        let solutions = spsolve_triangular_many(&a, &rhs, options).unwrap();
        for ((b, x), expected) in rhs.iter().zip(&solutions).zip(&expected) {
            check(x, expected);
            check(
                &spsolve_triangular_with_options(&a, b, options).unwrap(),
                expected,
            );
        }
    }
}

#[test]
fn duplicate_cancellation_precedes_substitution_and_singularity_checks() {
    let matrix = CsrMatrix::from_components(
        Shape2D::new(2, 2),
        vec![1., 1e290, 1., -1e290],
        vec![0, 0, 1, 0],
        vec![0, 1, 4],
        false,
    )
    .unwrap();
    let rhs = [1e35, 7.];
    assert_eq!(spsolve_triangular(&matrix, &rhs, true).unwrap(), rhs);

    let singular = CsrMatrix::from_components(
        Shape2D::new(1, 1),
        vec![4., -4.],
        vec![0, 0],
        vec![0, 2],
        false,
    )
    .unwrap();
    for lower in [true, false] {
        let options = SparseTriangularSolveOptions {
            lower,
            ..Default::default()
        };
        assert!(matches!(
            spsolve_triangular(&singular, &[1.], lower),
            Err(SparseError::SingularMatrix { .. })
        ));
        assert!(matches!(
            spsolve_triangular_many(&singular, &[], options),
            Err(SparseError::SingularMatrix { .. })
        ));
    }
}

#[test]
fn rhs_panel_checks_all_dimensions_and_supports_empty_axes() {
    let a = CsrMatrix::from_components(
        Shape2D::new(2, 2),
        vec![1., 1.],
        vec![0, 1],
        vec![0, 1, 2],
        true,
    )
    .unwrap();
    let options = SparseTriangularSolveOptions::default();
    assert!(matches!(
        spsolve_triangular_many(&a, &[vec![1., 2.], vec![1.]], options),
        Err(SparseError::IncompatibleShape { .. })
    ));
    assert!(
        spsolve_triangular_many(&a, &[], options)
            .unwrap()
            .is_empty()
    );

    let empty =
        CsrMatrix::from_components(Shape2D::new(0, 0), vec![], vec![], vec![0], true).unwrap();
    let columns: [Vec<f64>; 2] = [vec![], vec![]];
    assert_eq!(
        spsolve_triangular_many(&empty, &columns, options).unwrap(),
        columns
    );

    let rectangle =
        CsrMatrix::from_components(Shape2D::new(1, 2), vec![], vec![], vec![0, 0], true).unwrap();
    assert!(matches!(
        spsolve_triangular_many(&rectangle, &[], options),
        Err(SparseError::InvalidShape { .. })
    ));
}
