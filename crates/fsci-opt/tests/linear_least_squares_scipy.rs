#![forbid(unsafe_code)]

//! Public bounded linear least-squares regressions and live SciPy comparisons.
//! Set FSCI_REQUIRE_SCIPY_ORACLE=1 to require pinned SciPy 1.17.1 / NumPy 2.4.3.

use fsci_opt::{
    LeastSquaresMatrix, LinearLeastSquaresOptions, OptError, lsq_linear, lsq_linear_with_options,
};
use fsci_sparse::{CsrMatrix, DenseLinearOperator, LinearOperator, Shape2D, SparseResult};

fn sparse_matrix(a: &[Vec<f64>]) -> LeastSquaresMatrix {
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = vec![0];
    for row in a {
        for (column, &value) in row.iter().enumerate() {
            if value != 0.0 {
                data.push(value);
                indices.push(column);
            }
        }
        indptr.push(data.len());
    }
    LeastSquaresMatrix::Sparse(
        CsrMatrix::from_components(
            Shape2D::new(a.len(), a[0].len()),
            data,
            indices,
            indptr,
            true,
        )
        .unwrap(),
    )
}

fn representations(a: &[Vec<f64>]) -> [LeastSquaresMatrix; 3] {
    [
        LeastSquaresMatrix::Dense(a.to_vec()),
        sparse_matrix(a),
        LeastSquaresMatrix::from_operator(DenseLinearOperator::from_rows(a).unwrap()),
    ]
}

#[test]
fn linear_trf_solves_active_bounds_and_inconsistent_stationarity() {
    let a = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0]];
    for matrix in representations(&a) {
        let fit = lsq_linear_with_options(
            &matrix,
            &[2.0, -3.0, 4.0],
            &[0.0, -1.0],
            &[0.5, 1.0],
            LinearLeastSquaresOptions::default(),
        )
        .unwrap();
        assert!(fit.success, "{fit:?}");
        assert!((fit.x[0] - 0.5).abs() < 1e-10, "{fit:?}");
        assert!((fit.x[1] + 1.0).abs() < 1e-10, "{fit:?}");
        assert_eq!(fit.active_mask, vec![1, -1]);
        assert!((fit.cost - 11.125).abs() < 1e-9, "{fit:?}");
        assert!(fit.optimality < 1e-9, "{fit:?}");

        let stationary = lsq_linear_with_options(
            &matrix,
            &[0.0, 0.0, 4.0],
            &[-1.0; 2],
            &[1.0; 2],
            LinearLeastSquaresOptions::default(),
        )
        .unwrap();
        assert!(stationary.success);
        assert_eq!(stationary.status, 3);
        assert_eq!(stationary.nit, 0);
        assert_eq!(stationary.x, vec![0.0; 2]);
        assert_eq!(stationary.fun, vec![0.0, 0.0, -4.0]);
        assert_eq!(stationary.cost, 8.0);
        assert_eq!(stationary.optimality, 0.0);
    }
    let no_variables = lsq_linear_with_options(
        &LeastSquaresMatrix::Dense(vec![vec![], vec![]]),
        &[3.0, -4.0],
        &[],
        &[],
        Default::default(),
    )
    .unwrap();
    assert!(no_variables.success);
    assert!(no_variables.x.is_empty());
    assert_eq!(no_variables.fun, vec![-3.0, 4.0]);
    assert_eq!(no_variables.cost, 12.5);
}

#[test]
fn linear_trf_matrix_free_large_problem_never_probes_entries() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct ProductsOnly {
        n: usize,
        calls: Arc<AtomicUsize>,
    }

    impl LinearOperator for ProductsOnly {
        fn shape(&self) -> Shape2D {
            Shape2D::new(self.n, self.n)
        }

        fn matvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            y.copy_from_slice(x);
            Ok(())
        }

        fn rmatvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
            self.matvec_into(x, y)
        }

        fn to_dense(&self) -> SparseResult<Vec<Vec<f64>>> {
            panic!("bounded linear fitting must not materialize operator entries")
        }
    }

    // A dense representation would require 3.2 GB; the actual problem needs
    // only vectors. Fewer than n products also rules out manual basis probing.
    let n = 20_000;
    let calls = Arc::new(AtomicUsize::new(0));
    let matrix = LeastSquaresMatrix::from_operator(ProductsOnly {
        n,
        calls: calls.clone(),
    });
    let fit = lsq_linear_with_options(
        &matrix,
        &vec![2.0; n],
        &vec![0.0; n],
        &vec![1.0; n],
        LinearLeastSquaresOptions::default(),
    )
    .unwrap();
    assert!(fit.success, "{}: {}", fit.status, fit.message);
    assert!(fit.x.iter().all(|value| (value - 1.0).abs() < 1e-9));
    assert!(fit.active_mask.iter().all(|&side| side == 1));
    assert!((fit.cost - n as f64 / 2.0).abs() < 1e-5);
    assert!(calls.load(Ordering::Relaxed) < 1_000);
}

#[test]
fn linear_trf_reports_limits_and_rejects_invalid_or_partial_products() {
    let matrix = LeastSquaresMatrix::Dense(vec![vec![1.0, 0.0], vec![0.0, 0.01]]);
    let fit = lsq_linear_with_options(
        &matrix,
        &[2.0, 3.0],
        &[0.0; 2],
        &[1.0; 2],
        LinearLeastSquaresOptions {
            max_iter: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!fit.success, "{fit:?}");
    assert_eq!(fit.status, 0);
    assert_eq!(fit.nit, 1);
    assert!(fit.optimality > 1e-10);

    let truncated = LeastSquaresMatrix::Dense(vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 0.1, 0.0],
        vec![0.0, 0.0, 0.01],
    ]);
    let incomplete = lsq_linear_with_options(
        &truncated,
        &[1.0; 3],
        &[f64::NEG_INFINITY; 3],
        &[f64::INFINITY; 3],
        LinearLeastSquaresOptions {
            max_iter: 1,
            lsmr_max_iter: Some(1),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(incomplete.unbounded_sol.istop, 7);
    assert!(
        !incomplete.success,
        "a feasible truncated solve is not an optimum: {incomplete:?}"
    );
    assert_eq!(incomplete.status, 0);

    for options in [
        LinearLeastSquaresOptions {
            tol: f64::NAN,
            ..Default::default()
        },
        LinearLeastSquaresOptions {
            tol: 0.0,
            ..Default::default()
        },
        LinearLeastSquaresOptions {
            max_iter: 0,
            ..Default::default()
        },
        LinearLeastSquaresOptions {
            lsmr_tol: Some(f64::INFINITY),
            ..Default::default()
        },
        LinearLeastSquaresOptions {
            lsmr_max_iter: Some(0),
            ..Default::default()
        },
    ] {
        assert!(matches!(
            lsq_linear_with_options(&matrix, &[1.0; 2], &[0.0; 2], &[1.0; 2], options),
            Err(OptError::InvalidArgument { .. })
        ));
    }
    for (lower, upper) in [
        (vec![0.0; 2], vec![0.0; 2]),
        (vec![f64::NAN; 2], vec![1.0; 2]),
    ] {
        assert!(matches!(
            lsq_linear_with_options(&matrix, &[1.0; 2], &lower, &upper, Default::default()),
            Err(OptError::InvalidBounds { .. })
        ));
    }
    let ragged = LeastSquaresMatrix::Dense(vec![vec![1.0], vec![2.0, 3.0]]);
    assert!(
        lsq_linear_with_options(&ragged, &[1.0; 2], &[0.0], &[1.0], Default::default()).is_err()
    );

    struct PartialProduct;

    impl LinearOperator for PartialProduct {
        fn shape(&self) -> Shape2D {
            Shape2D::new(2, 2)
        }

        fn matvec_into(&self, _: &[f64], y: &mut [f64]) -> SparseResult<()> {
            y[0] = 1.0;
            Ok(())
        }

        fn rmatvec_into(&self, x: &[f64], y: &mut [f64]) -> SparseResult<()> {
            self.matvec_into(x, y)
        }
    }

    let partial = LeastSquaresMatrix::from_operator(PartialProduct);
    assert!(matches!(
        lsq_linear_with_options(
            &partial,
            &[1.0; 2],
            &[0.0; 2],
            &[1.0; 2],
            Default::default()
        ),
        Err(OptError::NonFiniteInput { .. })
    ));
}

#[test]
fn linear_trf_sparse_and_operator_against_live_scipy() {
    #[derive(serde::Serialize)]
    struct Case {
        name: String,
        a: Vec<Vec<f64>>,
        b: Vec<f64>,
        lower: Vec<Option<f64>>,
        upper: Vec<Option<f64>>,
        unique: bool,
    }

    #[derive(serde::Deserialize)]
    struct Reference {
        x: Vec<f64>,
        cost: f64,
        success: bool,
    }

    #[derive(serde::Deserialize)]
    struct Oracle {
        scipy: String,
        numpy: String,
        rows: Vec<Vec<Reference>>,
    }

    let mut cases = vec![
        Case {
            name: "opposing_active_bounds".into(),
            a: vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0]],
            b: vec![2.0, -3.0, 4.0],
            lower: vec![Some(0.0), Some(-1.0)],
            upper: vec![Some(0.5), Some(1.0)],
            unique: true,
        },
        Case {
            name: "rank_deficient".into(),
            a: vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![0.0, 0.0]],
            b: vec![2.0, 4.0, 3.0],
            lower: vec![Some(0.0); 2],
            upper: vec![Some(0.7); 2],
            unique: false,
        },
        Case {
            name: "wide_active_bounds".into(),
            a: vec![vec![1.0, 0.0, 1.0], vec![0.0, 1.0, 1.0]],
            b: vec![1.0, 2.0],
            lower: vec![Some(0.0); 3],
            upper: vec![Some(0.4), Some(0.6), Some(0.7)],
            unique: false,
        },
        Case {
            name: "wide_unconstrained".into(),
            a: vec![vec![1.0, 0.0, 1.0], vec![0.0, 1.0, 1.0]],
            b: vec![1.0, 2.0],
            lower: vec![None; 3],
            upper: vec![None; 3],
            unique: true, // Both LSMR solves select the minimum-norm solution.
        },
        Case {
            name: "zero_operator_nonzero_rhs".into(),
            a: vec![vec![0.0; 2]; 3],
            b: vec![1.0, -2.0, 3.0],
            lower: vec![Some(1.0), Some(-2.0)],
            upper: vec![Some(2.0), Some(-1.0)],
            unique: false,
        },
    ];
    for seed in 0..18 {
        let m = 3 + seed % 6;
        let n = 2 + seed % 4;
        let a: Vec<Vec<f64>> = (0..m)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        (((i + 1) * (j + 2)) as f64 + 0.31 * seed as f64).sin()
                            + if i == j { 2.0 } else { 0.0 }
                    })
                    .collect()
            })
            .collect();
        cases.push(Case {
            name: format!("mixed_rectangular_{seed}"),
            a,
            b: (0..m)
                .map(|i| 3.0 * ((i + seed + 1) as f64 * 0.71).cos())
                .collect(),
            lower: (0..n)
                .map(|j| {
                    if (j + seed) % 3 == 0 {
                        None
                    } else {
                        Some(-0.2)
                    }
                })
                .collect(),
            upper: (0..n)
                .map(|j| if (j + seed) % 4 == 0 { None } else { Some(0.6) })
                .collect(),
            unique: m >= n,
        });
    }
    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let python = std::env::var("FSCI_PYTHON").unwrap_or_else(|_| "python3".into());
    let available = std::process::Command::new(&python)
        .args(["-c", "import scipy; import numpy"])
        .output()
        .is_ok_and(|output| output.status.success());
    if !available {
        assert!(!required, "required SciPy oracle unavailable via {python}");
        eprintln!("SKIP linear TRF: missing SciPy; compared cases = 0");
        return;
    }
    let script = r#"
import json, sys
import numpy as np
import scipy
from scipy.optimize import lsq_linear
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import aslinearoperator
rows = []
for case in json.loads(sys.argv[1]):
    a, b = np.array(case['a']), np.array(case['b'])
    lower = [-np.inf if x is None else x for x in case['lower']]
    upper = [np.inf if x is None else x for x in case['upper']]
    comparisons = []
    for matrix in (a, csr_matrix(a), aslinearoperator(a)):
        fit = lsq_linear(matrix, b, bounds=(lower, upper), method='trf',
                         lsq_solver='lsmr', tol=1e-12, lsmr_tol=1e-14,
                         max_iter=500, lsmr_maxiter=4*a.shape[1])
        comparisons.append(dict(x=fit.x.tolist(), cost=float(fit.cost), success=bool(fit.success)))
    rows.append(comparisons)
print(json.dumps(dict(scipy=scipy.__version__, numpy=np.__version__, rows=rows), allow_nan=False))
"#;
    let output = std::process::Command::new(&python)
        .args(["-c", script, &serde_json::to_string(&cases).unwrap()])
        .output()
        .expect("launch SciPy linear TRF oracle");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let oracle: Oracle = serde_json::from_slice(&output.stdout).expect("linear TRF reference JSON");
    if required {
        assert_eq!((&*oracle.scipy, &*oracle.numpy), ("1.17.1", "2.4.3"));
    }
    assert_eq!(oracle.rows.len(), cases.len());
    let mut compared = 0;
    for (case, references) in cases.iter().zip(&oracle.rows) {
        let lower: Vec<f64> = case
            .lower
            .iter()
            .map(|x| x.unwrap_or(f64::NEG_INFINITY))
            .collect();
        let upper: Vec<f64> = case
            .upper
            .iter()
            .map(|x| x.unwrap_or(f64::INFINITY))
            .collect();
        assert_eq!(references.len(), 3);
        for (representation, (matrix, reference)) in representations(&case.a)
            .into_iter()
            .zip(references)
            .enumerate()
        {
            let fit = lsq_linear_with_options(
                &matrix,
                &case.b,
                &lower,
                &upper,
                LinearLeastSquaresOptions {
                    tol: 1e-12,
                    max_iter: 500,
                    lsmr_tol: Some(1e-14),
                    lsmr_max_iter: Some(4 * lower.len()),
                    ..Default::default()
                },
            )
            .unwrap_or_else(|error| panic!("{}[{representation}]: {error}", case.name));
            assert!(
                reference.success,
                "{}[{representation}]: SciPy did not converge",
                case.name
            );
            assert!(fit.success, "{}[{representation}]: {fit:?}", case.name);
            assert!(
                (fit.cost - reference.cost).abs() < 2e-10 * (1.0 + reference.cost),
                "{}[{representation}]: cost {} != {}",
                case.name,
                fit.cost,
                reference.cost
            );
            for (j, &value) in fit.x.iter().enumerate() {
                assert!(
                    value >= lower[j] && value <= upper[j],
                    "{}: infeasible {fit:?}",
                    case.name
                );
                if case.unique {
                    assert!(
                        (value - reference.x[j]).abs() < 2e-6 * (1.0 + reference.x[j].abs()),
                        "{}[{representation}][{j}]: {value:e} != {:e}",
                        case.name,
                        reference.x[j]
                    );
                }
                let gradient: f64 = case.a.iter().zip(&fit.fun).map(|(row, r)| row[j] * r).sum();
                let violation = if value <= lower[j] + 1e-7 {
                    -gradient
                } else if value >= upper[j] - 1e-7 {
                    gradient
                } else {
                    gradient.abs()
                };
                assert!(
                    violation <= 2e-7 * (1.0 + reference.cost.sqrt()),
                    "{}[{representation}][{j}]: KKT violation {violation:e}, {fit:?}",
                    case.name
                );
            }
            eprintln!(
                "{}[{representation}]: status={} iterations={} cost={:e} optimality={:e}",
                case.name, fit.status, fit.nit, fit.cost, fit.optimality
            );
            compared += 1;
        }
    }
    assert_eq!(compared, 69);
    eprintln!(
        "linear TRF: compared {compared} cases against SciPy {} / NumPy {}",
        oracle.scipy, oracle.numpy
    );
}

#[test]
fn lsq_linear_rank_deficient_and_underdetermined() {
    let inf = f64::INFINITY;
    let close = |actual: &[f64], expected: &[f64]| {
        for (&got, &want) in actual.iter().zip(expected) {
            assert!((got - want).abs() < 2e-13, "{actual:?} != {expected:?}");
        }
    };

    // Singular normal equations previously returned the initial [0, 0]
    // without checking stationarity of either free variable.
    let duplicate = [vec![1.0, 1.0], vec![2.0, 2.0]];
    let x = lsq_linear(&duplicate, &[2.0, 4.0], &[-inf; 2], &[inf; 2]).unwrap();
    close(&x, &[1.0, 1.0]);

    let wide = [vec![1.0, 0.0, 1.0], vec![0.0, 1.0, 1.0]];
    let x = lsq_linear(&wide, &[1.0, 2.0], &[-inf; 3], &[inf; 3]).unwrap();
    close(&x, &[0.0, 1.0, 1.0]);
    let x = lsq_linear(&wide, &[1.0, 2.0], &[0.0; 3], &[0.4, 0.6, 0.7]).unwrap();
    close(&x, &[0.3, 0.6, 0.7]);

    // The free submatrix remains singular after the third variable binds.
    let dependent_free = [
        vec![1.0, 1.0, 0.0],
        vec![2.0, 2.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let x = lsq_linear(
        &dependent_free,
        &[2.0, 4.0, 3.0],
        &[-inf, -inf, 0.0],
        &[inf, inf, 1.0],
    )
    .unwrap();
    close(&x, &[1.0, 1.0, 1.0]);
    let x = lsq_linear(
        &dependent_free,
        &[3.0, 6.0, 2.0],
        &[-inf, -inf, 0.0],
        &[1.0, inf, 1.0],
    )
    .unwrap();
    close(&x, &[1.0, 2.0, 1.0]);

    let x = lsq_linear(&duplicate, &[2.0, 4.0], &[0.5, -inf], &[0.5, inf]).unwrap();
    close(&x, &[0.5, 1.5]);
    let x = lsq_linear(
        &[vec![0.0; 3], vec![0.0; 3]],
        &[1.0, -1.0],
        &[1.0, -2.0, -inf],
        &[2.0, -1.0, inf],
    )
    .unwrap();
    close(&x, &[1.0, -1.0, 0.0]);
}

#[test]
fn lsq_linear_retains_small_singular_directions() {
    let a = [vec![1.0, 1.0], vec![1.0, 1.0 + 1e-9], vec![1.0, 1.0 - 1e-9]];
    let b: Vec<f64> = a.iter().map(|row| row[0] - row[1]).collect();
    for scale in [1e-170, 1e-12, 1.0, 1e12, 1e160] {
        let scaled_a: Vec<Vec<f64>> = a
            .iter()
            .map(|row| row.iter().map(|value| value * scale).collect())
            .collect();
        let scaled_b: Vec<f64> = b.iter().map(|value| value * scale).collect();
        let x = lsq_linear(
            &scaled_a,
            &scaled_b,
            &[f64::NEG_INFINITY; 2],
            &[f64::INFINITY; 2],
        )
        .unwrap();
        assert!((x[0] - 1.0).abs() < 5e-7, "scale {scale:e}: {x:?}");
        assert!((x[1] + 1.0).abs() < 5e-7, "scale {scale:e}: {x:?}");
        for (row, &bi) in a.iter().zip(&b) {
            let residual = row[0] * x[0] + row[1] * x[1] - bi;
            assert!(
                residual.abs() < 2e-15,
                "scale {scale:e}: residual {residual:e}"
            );
        }
    }
}

#[test]
fn lsq_linear_reports_unrepresentable_problems() {
    let error = lsq_linear(
        &[vec![1.0, 1.0]],
        &[0.0],
        &[f64::MAX; 2],
        &[f64::INFINITY; 2],
    )
    .expect_err("overflowed fitted values cannot certify optimality");
    assert!(matches!(error, OptError::NotConverged { .. }));
    let error = lsq_linear(
        &[vec![1e-300]],
        &[1e300],
        &[f64::NEG_INFINITY],
        &[f64::INFINITY],
    )
    .expect_err("underflowing A during scaling must not silently erase the problem");
    assert!(matches!(error, OptError::NotConverged { .. }));
}

/// Live coverage for rank loss, free-set changes and data scaling. Singular
/// constrained problems can have different minimizing x, so compare their
/// objective and KKT conditions; unconstrained minimum-norm x is canonical.
#[test]
fn lsq_linear_rank_and_bounds_against_live_scipy() {
    #[derive(serde::Serialize)]
    struct Case {
        name: String,
        a: Vec<Vec<f64>>,
        b: Vec<f64>,
        lower: Vec<Option<f64>>,
        upper: Vec<Option<f64>>,
        compare_x: bool,
    }
    #[derive(serde::Deserialize)]
    struct Reference {
        x: Vec<f64>,
        cost: f64,
        status: i32,
        raw_status: i32,
    }
    #[derive(serde::Deserialize)]
    struct Oracle {
        scipy: String,
        numpy: String,
        rows: Vec<Reference>,
    }
    let mut cases = vec![
        Case {
            name: "duplicate_columns".into(),
            a: vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![0.0, 0.0]],
            b: vec![2.0, 4.0, 3.0],
            lower: vec![None; 2],
            upper: vec![None; 2],
            compare_x: true,
        },
        Case {
            name: "singular_free_set_after_bound".into(),
            a: vec![
                vec![1.0, 1.0, 0.0],
                vec![2.0, 2.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
            b: vec![2.0, 4.0, 3.0],
            lower: vec![None, None, Some(0.0)],
            upper: vec![None, None, Some(1.0)],
            compare_x: false,
        },
        Case {
            name: "underdetermined_minimum_norm".into(),
            a: vec![vec![1.0, 0.0, 1.0], vec![0.0, 1.0, 1.0]],
            b: vec![1.0, 2.0],
            lower: vec![None; 3],
            upper: vec![None; 3],
            compare_x: true,
        },
        Case {
            name: "underdetermined_active_bounds".into(),
            a: vec![vec![1.0, 0.0, 1.0], vec![0.0, 1.0, 1.0]],
            b: vec![1.0, 2.0],
            lower: vec![Some(0.0); 3],
            upper: vec![Some(0.4), Some(0.6), Some(0.7)],
            compare_x: true,
        },
    ];
    let nearly_dependent = vec![vec![1.0, 1.0], vec![1.0, 1.0 + 1e-9], vec![1.0, 1.0 - 1e-9]];
    cases.push(Case {
        name: "condition_number_beyond_normal_equations".into(),
        b: nearly_dependent.iter().map(|row| row[0] - row[1]).collect(),
        a: nearly_dependent,
        lower: vec![None; 2],
        upper: vec![None; 2],
        compare_x: true,
    });
    for scale in [1e-170, 1e-12, 1.0, 1e12, 1e160] {
        for bounded in [false, true] {
            cases.push(Case {
                name: format!("scaled_{scale:e}_bounded_{bounded}"),
                a: vec![
                    vec![scale, 0.0],
                    vec![-scale, scale],
                    vec![0.0, 2.0 * scale],
                ],
                b: vec![scale, 0.0, 3.0 * scale],
                lower: vec![bounded.then_some(0.0); 2],
                upper: if bounded {
                    vec![Some(0.3), Some(1.0)]
                } else {
                    vec![None; 2]
                },
                compare_x: true,
            });
        }
    }
    for seed in 0..20 {
        let m = 3 + seed % 5;
        let n = 2 + seed % 6;
        let mut a: Vec<Vec<f64>> = (0..m)
            .map(|i| {
                (0..n)
                    .map(|j| (((i + 2) * (j + 3)) as f64 + 0.37 * seed as f64).sin())
                    .collect()
            })
            .collect();
        if seed % 3 == 0 {
            for row in &mut a {
                row[n - 1] = row[0];
            }
        }
        let b: Vec<f64> = (0..m)
            .map(|i| ((i + 1) as f64 * 0.61 + seed as f64).cos())
            .collect();
        cases.push(Case {
            name: format!("mixed_active_set_{seed}"),
            a,
            b,
            lower: vec![Some(-0.2); n],
            upper: vec![Some(0.6); n],
            compare_x: m >= n && seed % 3 != 0,
        });
    }
    let required = std::env::var("FSCI_REQUIRE_SCIPY_ORACLE").as_deref() == Ok("1");
    let python = std::env::var("FSCI_PYTHON").unwrap_or_else(|_| "python3".into());
    let available = std::process::Command::new(&python)
        .args(["-c", "import scipy; import numpy"])
        .output()
        .is_ok_and(|output| output.status.success());
    if !available {
        assert!(!required, "required SciPy oracle unavailable via {python}");
        eprintln!("SKIP lsq_linear: missing SciPy; compared cases = 0");
        return;
    }
    let script = r#"
import json, sys
import numpy as np
import scipy
from scipy.optimize import lsq_linear
rows = []
for case in json.loads(sys.argv[1]):
    a, b = np.array(case['a']), np.array(case['b'])
    lower = [-np.inf if value is None else value for value in case['lower']]
    upper = [np.inf if value is None else value for value in case['upper']]
    raw = lsq_linear(a, b, bounds=(lower, upper), method='bvls', max_iter=1000)
    scale = max(np.max(np.abs(a)), np.max(np.abs(b)), np.finfo(float).tiny)
    # A common positive scale preserves the minimizer. SciPy's raw BVLS cost
    # can overflow at 1e160, so record its status and use the equivalent scaled
    # problem as the finite reference for every case, without skipping a row.
    normalized_a, normalized_b = a / scale, b / scale
    result = lsq_linear(normalized_a, normalized_b, bounds=(lower, upper), method='bvls', max_iter=1000)
    residual = normalized_a @ result.x - normalized_b
    rows.append(dict(x=result.x.tolist(), cost=float(0.5 * residual @ residual),
                     status=int(result.status), raw_status=int(raw.status)))
print(json.dumps(dict(scipy=scipy.__version__, numpy=np.__version__, rows=rows), allow_nan=False))
"#;
    let output = std::process::Command::new(&python)
        .args(["-c", script, &serde_json::to_string(&cases).unwrap()])
        .output()
        .expect("launch SciPy BVLS oracle");
    assert!(
        output.status.success(),
        "SciPy BVLS failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let oracle: Oracle = serde_json::from_slice(&output.stdout).expect("SciPy BVLS result JSON");
    if required {
        assert_eq!((&*oracle.scipy, &*oracle.numpy), ("1.17.1", "2.4.3"));
    }
    assert_eq!(oracle.rows.len(), cases.len());
    let mut compared = 0;
    let mut raw_failures = 0;
    for (case, reference) in cases.iter().zip(&oracle.rows) {
        let lower: Vec<f64> = case
            .lower
            .iter()
            .map(|value| value.unwrap_or(f64::NEG_INFINITY))
            .collect();
        let upper: Vec<f64> = case
            .upper
            .iter()
            .map(|value| value.unwrap_or(f64::INFINITY))
            .collect();
        let x = lsq_linear(&case.a, &case.b, &lower, &upper)
            .unwrap_or_else(|error| panic!("{}: {error}", case.name));
        assert!(
            reference.status > 0,
            "{}: scale-normalized SciPy did not converge",
            case.name
        );
        if reference.raw_status <= 0 {
            raw_failures += 1;
            eprintln!(
                "{}: raw SciPy status {}; comparing equivalent scale-normalized SciPy status {}",
                case.name, reference.raw_status, reference.status
            );
        }
        for (j, &value) in x.iter().enumerate() {
            assert!(
                value.is_finite() && value >= lower[j] && value <= upper[j],
                "{}: infeasible x {x:?}",
                case.name
            );
            if case.compare_x {
                assert!(
                    (value - reference.x[j]).abs() <= 5e-7 * (1.0 + reference.x[j].abs()),
                    "{}[{j}]: {value:e} != {:e}",
                    case.name,
                    reference.x[j]
                );
            }
        }
        let scale = case
            .a
            .iter()
            .flatten()
            .chain(&case.b)
            .fold(0.0_f64, |scale, value| scale.max(value.abs()));
        let residual: Vec<f64> = case
            .a
            .iter()
            .zip(&case.b)
            .map(|(row, &bi)| {
                row.iter()
                    .zip(&x)
                    .map(|(&aij, &xj)| (aij / scale) * xj)
                    .sum::<f64>()
                    - bi / scale
            })
            .collect();
        let cost = 0.5 * residual.iter().map(|value| value * value).sum::<f64>();
        assert!(
            (cost - reference.cost).abs() <= 2e-12 * (1.0 + reference.cost),
            "{}: cost {cost:e} != {:e}",
            case.name,
            reference.cost
        );
        for j in 0..x.len() {
            let gradient: f64 = case
                .a
                .iter()
                .zip(&residual)
                .map(|(row, &ri)| (row[j] / scale) * ri)
                .sum();
            let violation = if x[j] == lower[j] {
                -gradient
            } else if x[j] == upper[j] {
                gradient
            } else {
                gradient.abs()
            };
            assert!(
                violation <= 5e-10,
                "{}[{j}]: projected gradient {violation:e}, x {x:?}",
                case.name
            );
        }
        compared += 1;
    }
    assert_eq!(compared, 35);
    eprintln!(
        "lsq_linear: compared {compared} cases against scale-normalized SciPy {} / NumPy {}; raw oracle nonconvergences = {raw_failures}",
        oracle.scipy, oracle.numpy
    );
}
