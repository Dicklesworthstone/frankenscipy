//! Lebedev quadrature on the unit sphere.
//!
//! Bit-faithful port of `scipy.integrate.lebedev_rule` (the Parrish/Laikov
//! `getLebedevSphere` translation). The per-order coefficient table in
//! [`table`] was extracted verbatim from scipy's `_lebedev.py`, and the
//! octahedral-orbit expansion below mirrors `get_lebedev_recurrence_points`
//! point-for-point, so the points and weights match scipy bit-for-bit.

use std::f64::consts::PI;

use crate::validation::IntegrateValidationError;

mod table;

/// One octahedral-orbit generator: an integer `code` (1..=6) selecting the
/// orbit type, plus its `a`/`b` parameters and the shared weight value `v`.
#[derive(Clone, Copy)]
struct LebGen {
    code: u8,
    a: f64,
    b: f64,
    v: f64,
}

/// Sample points and weights for a Lebedev quadrature rule on the unit sphere.
///
/// Returned by [`lebedev_rule`]. `points[i] == [x, y, z]` is the i-th sample
/// point (on the unit sphere) and `weights[i]` its weight. This corresponds to
/// scipy's `(x, w)` return where scipy's `x` has shape `(3, m)`: scipy's
/// `x[:, i]` equals `points[i]`. The weights are normalized to sum to 4π.
#[derive(Clone, Debug, PartialEq)]
pub struct LebedevRule {
    /// Cartesian sample points on the unit sphere, `[x, y, z]` each.
    pub points: Vec<[f64; 3]>,
    /// Quadrature weights (sum to 4π).
    pub weights: Vec<f64>,
}

/// Expand one octahedral-orbit generator into its full set of points/weights,
/// appending them in scipy's fill order.
fn expand(g: &LebGen, points: &mut Vec<[f64; 3]>, weights: &mut Vec<f64>) {
    let w = 4.0 * PI * g.v;
    match g.code {
        1 => {
            let a = 1.0_f64;
            for p in [
                [a, 0.0, 0.0],
                [-a, 0.0, 0.0],
                [0.0, a, 0.0],
                [0.0, -a, 0.0],
                [0.0, 0.0, a],
                [0.0, 0.0, -a],
            ] {
                points.push(p);
                weights.push(w);
            }
        }
        2 => {
            let a = 0.5_f64.sqrt();
            for p in [
                [0.0, a, a],
                [0.0, -a, a],
                [0.0, a, -a],
                [0.0, -a, -a],
                [a, 0.0, a],
                [a, 0.0, -a],
                [-a, 0.0, a],
                [-a, 0.0, -a],
                [a, a, 0.0],
                [-a, a, 0.0],
                [a, -a, 0.0],
                [-a, -a, 0.0],
            ] {
                points.push(p);
                weights.push(w);
            }
        }
        3 => {
            let a = (1.0_f64 / 3.0).sqrt();
            for p in [
                [a, a, a],
                [-a, a, a],
                [a, -a, a],
                [a, a, -a],
                [-a, -a, a],
                [a, -a, -a],
                [-a, a, -a],
                [-a, -a, -a],
            ] {
                points.push(p);
                weights.push(w);
            }
        }
        4 => {
            let a = g.a;
            let b = (1.0 - 2.0 * a * a).sqrt();
            for p in [
                [a, a, b],
                [-a, a, b],
                [a, -a, b],
                [a, a, -b],
                [-a, -a, b],
                [-a, a, -b],
                [a, -a, -b],
                [-a, -a, -b],
                [-a, b, a],
                [a, -b, a],
                [a, b, -a],
                [-a, -b, a],
                [-a, b, -a],
                [a, -b, -a],
                [-a, -b, -a],
                [a, b, a],
                [b, a, a],
                [-b, a, a],
                [b, -a, a],
                [b, a, -a],
                [-b, -a, a],
                [-b, a, -a],
                [b, -a, -a],
                [-b, -a, -a],
            ] {
                points.push(p);
                weights.push(w);
            }
        }
        5 => {
            let a = g.a;
            let b = (1.0 - a * a).sqrt();
            for p in [
                [a, b, 0.0],
                [-a, b, 0.0],
                [a, -b, 0.0],
                [-a, -b, 0.0],
                [b, a, 0.0],
                [-b, a, 0.0],
                [b, -a, 0.0],
                [-b, -a, 0.0],
                [a, 0.0, b],
                [-a, 0.0, b],
                [a, 0.0, -b],
                [-a, 0.0, -b],
                [b, 0.0, a],
                [-b, 0.0, a],
                [b, 0.0, -a],
                [-b, 0.0, -a],
                [0.0, a, b],
                [0.0, -a, b],
                [0.0, a, -b],
                [0.0, -a, -b],
                [0.0, b, a],
                [0.0, -b, a],
                [0.0, b, -a],
                [0.0, -b, -a],
            ] {
                points.push(p);
                weights.push(w);
            }
        }
        6 => {
            let a = g.a;
            let b = g.b;
            let c = (1.0 - a * a - b * b).sqrt();
            for p in [
                [a, b, c],
                [-a, b, c],
                [a, -b, c],
                [a, b, -c],
                [-a, -b, c],
                [a, -b, -c],
                [-a, b, -c],
                [-a, -b, -c],
                [b, a, c],
                [-b, a, c],
                [b, -a, c],
                [b, a, -c],
                [-b, -a, c],
                [b, -a, -c],
                [-b, a, -c],
                [-b, -a, -c],
                [c, a, b],
                [-c, a, b],
                [c, -a, b],
                [c, a, -b],
                [-c, -a, b],
                [c, -a, -b],
                [-c, a, -b],
                [-c, -a, -b],
                [c, b, a],
                [-c, b, a],
                [c, -b, a],
                [c, b, -a],
                [-c, -b, a],
                [c, -b, -a],
                [-c, b, -a],
                [-c, -b, -a],
                [a, c, b],
                [-a, c, b],
                [a, -c, b],
                [a, c, -b],
                [-a, -c, b],
                [a, -c, -b],
                [-a, c, -b],
                [-a, -c, -b],
                [b, c, a],
                [-b, c, a],
                [b, -c, a],
                [b, c, -a],
                [-b, -c, a],
                [b, -c, -a],
                [-b, c, -a],
                [-b, -c, -a],
            ] {
                points.push(p);
                weights.push(w);
            }
        }
        // The table only ever emits codes 1..=6.
        _ => unreachable!("invalid Lebedev orbit code: {}", g.code),
    }
}

/// Compute the sample points and weights for Lebedev quadrature of the given
/// `order` `n` over the surface of the unit sphere.
///
/// Mirrors `scipy.integrate.lebedev_rule(n)`. The available orders are
/// `3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41, 47, 53,
/// 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131`.
///
/// # Errors
/// Returns [`IntegrateValidationError::LebedevOrderUnavailable`] if `n` is not
/// one of the supported orders (scipy raises `NotImplementedError`).
pub fn lebedev_rule(n: i64) -> Result<LebedevRule, IntegrateValidationError> {
    let (_, degree, gens) = table::LEBEDEV_TABLE
        .iter()
        .find(|(order, _, _)| *order == n)
        .ok_or(IntegrateValidationError::LebedevOrderUnavailable { order: n })?;

    let mut points: Vec<[f64; 3]> = Vec::with_capacity(*degree);
    let mut weights: Vec<f64> = Vec::with_capacity(*degree);
    for g in gens.iter() {
        expand(g, &mut points, &mut weights);
    }
    debug_assert_eq!(points.len(), *degree);
    Ok(LebedevRule { points, weights })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lebedev_n3_matches_scipy() {
        // scipy.integrate.lebedev_rule(3): 6 points on the coordinate axes,
        // each with weight 4π/6.
        let rule = lebedev_rule(3).unwrap();
        assert_eq!(rule.points.len(), 6);
        assert_eq!(rule.weights.len(), 6);
        let w0 = 4.0 * PI / 6.0;
        for &w in &rule.weights {
            assert!((w - w0).abs() < 1e-15, "weight = {w}");
        }
        // Points are ±e_x, ±e_y, ±e_z in scipy's fill order.
        let expected = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        assert_eq!(rule.points, expected);
    }

    #[test]
    fn lebedev_weights_sum_to_four_pi_and_points_on_sphere() {
        for n in [5, 17, 47, 131] {
            let rule = lebedev_rule(n).unwrap();
            let sum: f64 = rule.weights.iter().sum();
            assert!((sum - 4.0 * PI).abs() < 1e-12, "n={n} sum={sum}");
            for p in &rule.points {
                let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                assert!((r - 1.0).abs() < 1e-13, "n={n} radius={r}");
            }
        }
    }

    #[test]
    fn lebedev_integrates_low_degree_polynomial_exactly() {
        // ∫_S (x² + y² − z²) dΩ = 4π/3 for the unit sphere; a degree-5 rule
        // (n>=3) is exact. Reference value 4.188790204786399.
        let rule = lebedev_rule(5).unwrap();
        let integral: f64 = rule
            .points
            .iter()
            .zip(rule.weights.iter())
            .map(|(p, &w)| w * (p[0] * p[0] + p[1] * p[1] - p[2] * p[2]))
            .sum();
        assert!(
            (integral - 4.188_790_204_786_399).abs() < 1e-13,
            "integral = {integral}"
        );
    }

    #[test]
    fn lebedev_unsupported_order_errors() {
        assert!(matches!(
            lebedev_rule(4),
            Err(IntegrateValidationError::LebedevOrderUnavailable { order: 4 })
        ));
        assert!(lebedev_rule(133).is_err());
    }
}
