#![forbid(unsafe_code)]

use fsci_interpolate::{GriddataMethod, InterpError, LinearNDInterpolator, griddata};

fn affine_value(point: &[f64]) -> f64 {
    2.5 + point
        .iter()
        .enumerate()
        .map(|(axis, coordinate)| (axis as f64 + 1.0) * coordinate)
        .sum::<f64>()
}

fn skew_simplex(ndim: usize) -> Vec<Vec<f64>> {
    let origin: Vec<f64> = (0..ndim).map(|axis| -0.25 * axis as f64).collect();
    let mut points = vec![origin.clone()];
    for vertex in 0..ndim {
        let mut point = origin.clone();
        for (axis, coordinate) in point.iter_mut().enumerate().skip(vertex) {
            *coordinate += if axis == vertex {
                1.0 + 0.25 * axis as f64
            } else {
                0.125 * (axis - vertex) as f64
            };
        }
        points.push(point);
    }
    points
}

fn assert_close(actual: f64, expected: f64, context: &str) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= 2e-12 * expected.abs().max(1.0),
        "{context}: got {actual:.17e}, expected {expected:.17e}"
    );
}

#[test]
fn linear_nd_and_griddata_reproduce_affine_fields_through_seven_dimensions() {
    for ndim in 3..=7 {
        let points = skew_simplex(ndim);
        let values: Vec<f64> = points.iter().map(|point| affine_value(point)).collect();
        let interpolator = LinearNDInterpolator::new(&points, &values)
            .unwrap_or_else(|error| panic!("{ndim}-D interpolation failed: {error}"));
        let mut queries = points.clone();
        for case in 0..24 {
            let weights: Vec<f64> = (0..=ndim)
                .map(|vertex| 1.0 + ((case * 13 + vertex * 7) % 19) as f64)
                .collect();
            let total: f64 = weights.iter().sum();
            queries.push(
                (0..ndim)
                    .map(|axis| {
                        points
                            .iter()
                            .zip(&weights)
                            .map(|(point, weight)| point[axis] * weight / total)
                            .sum()
                    })
                    .collect(),
            );
        }
        // A face centroid must be accepted, as well as strictly interior points.
        queries.push(
            (0..ndim)
                .map(|axis| {
                    points[..ndim]
                        .iter()
                        .map(|point| point[axis] / ndim as f64)
                        .sum()
                })
                .collect(),
        );
        let batch = interpolator.eval_many(&queries).unwrap();
        let grid = griddata(&points, &values, &queries, GriddataMethod::Linear).unwrap();
        for (index, query) in queries.iter().enumerate() {
            let expected = affine_value(query);
            let context = format!("{ndim}-D query {index}");
            assert_close(batch[index], expected, &context);
            assert_close(grid[index], expected, &context);
            assert_close(interpolator.eval(query).unwrap(), expected, &context);
        }
        let outside: Vec<f64> = points[0]
            .iter()
            .zip(&points[1])
            .map(|(origin, vertex)| origin - 0.5 * (vertex - origin))
            .collect();
        assert!(interpolator.eval(&outside).unwrap().is_nan());
    }
}

#[test]
fn linear_nd_cube_interpolates_sites_and_preserves_outside_hull_fill() {
    let mut points: Vec<Vec<f64>> = (0..8)
        .map(|bits| {
            (0..3)
                .map(|axis| f64::from((bits >> axis) & 1))
                .collect()
        })
        .collect();
    points.push(vec![0.31, 0.27, 0.43]);
    // Duplicate data with the same value must not prevent N-D triangulation.
    points.push(points[2].clone());
    let values: Vec<f64> = points
        .iter()
        .map(|point| point[0] * point[0] + 2.0 * point[1] * point[2])
        .collect();
    let interpolator = LinearNDInterpolator::new(&points, &values).unwrap();
    let actual = interpolator.eval_many(&points).unwrap();
    for (index, (&got, &expected)) in actual.iter().zip(&values).enumerate() {
        assert_close(got, expected, &format!("cube data site {index}"));
    }

    let queries = vec![
        vec![-0.1, 0.5, 0.5],
        vec![0.5, 1.1, 0.5],
        vec![0.5, 0.5, -0.1],
        vec![f64::NAN, 0.5, 0.5],
        vec![0.5, f64::INFINITY, 0.5],
    ];
    assert!(
        interpolator
            .eval_many(&queries)
            .unwrap()
            .iter()
            .all(|value| value.is_nan())
    );
    for query in &queries {
        assert!(interpolator.eval(query).unwrap().is_nan());
    }
    assert!(interpolator.eval_many(&[]).unwrap().is_empty());
    assert!(matches!(
        interpolator.eval(&[0.5, 0.5]),
        Err(InterpError::InvalidArgument { .. })
    ));
    assert!(matches!(
        interpolator.eval_many(&[vec![0.5, 0.5, 0.5], vec![0.5, 0.5]]),
        Err(InterpError::InvalidArgument { .. })
    ));
}

#[test]
fn linear_nd_rejects_invalid_geometry_and_value_lengths() {
    let points = skew_simplex(3);
    let values = vec![1.0; points.len()];
    assert!(matches!(
        LinearNDInterpolator::new(&points, &values[..3]),
        Err(InterpError::LengthMismatch { .. })
    ));
    assert!(matches!(
        LinearNDInterpolator::new(&points[..3], &values[..3]),
        Err(InterpError::TooFewPoints {
            minimum: 4,
            actual: 3
        })
    ));
    let mut malformed = points.clone();
    malformed[2].pop();
    assert!(matches!(
        LinearNDInterpolator::new(&malformed, &values),
        Err(InterpError::InvalidArgument { .. })
    ));
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut nonfinite = points.clone();
        nonfinite[1][2] = invalid;
        assert!(matches!(
            LinearNDInterpolator::new(&nonfinite, &values),
            Err(InterpError::NonFiniteX)
        ));
    }
    let flat = vec![
        vec![0.0, 0.0, 2.0],
        vec![1.0, 0.0, 2.0],
        vec![0.0, 1.0, 2.0],
        vec![1.0, 1.0, 2.0],
        vec![0.2, 0.3, 2.0],
    ];
    assert!(matches!(
        LinearNDInterpolator::new(&flat, &[1.0; 5]),
        Err(InterpError::InvalidArgument { .. })
    ));
    for ndim in [0, 1, 8] {
        assert!(matches!(
            LinearNDInterpolator::new(&vec![vec![0.0; ndim]; ndim + 2], &vec![0.0; ndim + 2]),
            Err(InterpError::InvalidArgument { .. })
        ));
    }
}

#[test]
fn griddata_one_dimensional_linear_and_cubic_sort_sites_and_fill_outside() {
    let points = vec![vec![2.0], vec![-1.0], vec![0.0], vec![3.0], vec![1.0]];
    let queries = vec![
        vec![-1.5],
        vec![-1.0],
        vec![-0.25],
        vec![0.0],
        vec![0.5],
        vec![1.75],
        vec![3.0],
        vec![3.5],
    ];
    for method in [GriddataMethod::Linear, GriddataMethod::Cubic] {
        let field = |x: f64| match method {
            GriddataMethod::Linear => 2.0 * x - 3.0,
            GriddataMethod::Cubic => x * x * x - 2.0 * x + 1.0,
            GriddataMethod::Nearest => unreachable!(),
        };
        let values: Vec<f64> = points.iter().map(|point| field(point[0])).collect();
        let actual = griddata(&points, &values, &queries, method).unwrap();
        assert!(actual[0].is_nan() && actual[7].is_nan());
        for index in 1..7 {
            assert_close(actual[index], field(queries[index][0]), "one-dimensional griddata");
        }
        assert!(griddata(&points, &values, &[vec![0.0, 1.0]], method).is_err());
    }
    let singleton = griddata(
        &[vec![2.0]],
        &[7.0],
        &[vec![1.0], vec![2.0], vec![3.0]],
        GriddataMethod::Linear,
    )
    .unwrap();
    assert!(singleton[0].is_nan() && singleton[2].is_nan());
    assert_eq!(singleton[1], 7.0);
}
