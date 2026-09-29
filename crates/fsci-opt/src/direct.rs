#![forbid(unsafe_code)]

//! The DIRECT (DIviding RECTangles) and DIRECT-L global optimization algorithms.
//!
//! Matches `scipy.optimize.direct`.
//!
//! # References
//! - Jones, D. R., Perttunen, C. D., & Stuckman, B. E. (1993).
//!   "Lipschitzian optimization without the Lipschitz constant."
//!   Journal of Optimization Theory and Applications, 79(1), 157-181.
//! - Gablonsky, J. M., & Kelley, C. T. (2001).
//!   "A Locally-Biased form of the DIRECT Algorithm."
//!   Journal of Global Optimization, 21(1), 27-37.

use crate::types::{Bounds, OptError};
use fsci_runtime::{HARDENED_MAX_DIM, RuntimeMode};

/// Options for the DIRECT global optimization algorithm.
///
/// Matches arguments to `scipy.optimize.direct`.
#[derive(Debug, Clone, PartialEq)]
pub struct DirectOptions {
    /// Tradeoff between local and global search. Default is `1e-4`.
    pub eps: f64,
    /// Maximum objective function evaluations. Default is `1000 * n`.
    pub maxfun: Option<usize>,
    /// Maximum iterations. Default is `1000`.
    pub maxiter: usize,
    /// If `true` (default), use DIRECT-L (locally biased).
    /// If `false`, use original unbiased DIRECT.
    pub locally_biased: bool,
    /// Function value of global minimum if known. Default is `-inf`.
    pub f_min: f64,
    /// Relative tolerance for `f_min` termination. Default is `1e-4`.
    pub f_min_rtol: f64,
    /// Volume tolerance for termination. Default is `1e-16`.
    pub vol_tol: f64,
    /// Side length measure tolerance for termination. Default is `1e-6`.
    pub len_tol: f64,
    /// Strict or Hardened mode.
    pub mode: RuntimeMode,
}

impl Default for DirectOptions {
    fn default() -> Self {
        Self {
            eps: 1e-4,
            maxfun: None,
            maxiter: 1000,
            locally_biased: true,
            f_min: f64::NEG_INFINITY,
            f_min_rtol: 1e-4,
            vol_tol: 1e-16,
            len_tol: 1e-6,
            mode: RuntimeMode::Strict,
        }
    }
}

/// Result of DIRECT global optimization.
///
/// Matches `scipy.optimize.OptimizeResult` returned by `scipy.optimize.direct`.
#[derive(Debug, Clone, PartialEq)]
pub struct DirectResult {
    /// Solution vector at the best point found.
    pub x: Vec<f64>,
    /// Objective function value at minimum.
    pub fun: f64,
    /// Integer status code matching SciPy:
    /// - 1: `maxfun` exceeded (failure/stop)
    /// - 2: `maxiter` exceeded (failure/stop)
    /// - 3: `f_min` target reached within `f_min_rtol` (success)
    /// - 4: `vol_tol` reached (success)
    /// - 5: `len_tol` reached (success)
    pub status: i32,
    /// True if status > 2.
    pub success: bool,
    /// Descriptive termination message matching SciPy.
    pub message: String,
    /// Number of objective function evaluations.
    pub nfev: usize,
    /// Number of iterations.
    pub nit: usize,
}

#[derive(Clone, Debug)]
struct HyperRectangle {
    center: Vec<f64>,
    levels: Vec<u8>,
    sum_levels: u32,
    min_level: u8,
    d: f64,
    f: f64,
}

impl HyperRectangle {
    #[inline]
    fn volume(&self, pow3_inv: &[f64; 64]) -> f64 {
        if self.sum_levels > 600 {
            0.0
        } else {
            let mut v = 1.0;
            for &k in &self.levels {
                v *= pow3_inv[k.min(63) as usize];
            }
            v
        }
    }

    #[inline]
    fn side_length_measure(&self, locally_biased: bool, pow3_inv: &[f64; 64]) -> f64 {
        if locally_biased {
            0.5 * pow3_inv[self.min_level.min(63) as usize]
        } else {
            let sum_sq: f64 = self
                .levels
                .iter()
                .map(|&k| {
                    let l = pow3_inv[k.min(63) as usize];
                    l * l
                })
                .sum();
            0.5 * sum_sq.sqrt()
        }
    }
}

/// Finds the global minimum of a function using the DIRECT or DIRECT-L algorithm.
///
/// Matches `scipy.optimize.direct`.
pub fn direct<F>(func: F, bounds: &Bounds, options: DirectOptions) -> Result<DirectResult, OptError>
where
    F: FnMut(&[f64]) -> f64,
{
    direct_with_callback::<F, fn(&[f64])>(func, bounds, options, None)
}

/// Finds the global minimum with an optional per-iteration callback.
pub fn direct_with_callback<F, C>(
    mut func: F,
    bounds: &Bounds,
    options: DirectOptions,
    mut callback: Option<C>,
) -> Result<DirectResult, OptError>
where
    F: FnMut(&[f64]) -> f64,
    C: FnMut(&[f64]),
{
    let n = bounds.len();
    if n == 0 {
        return Err(OptError::InvalidArgument {
            detail: "bounds must not be empty".to_string(),
        });
    }

    if options.mode == RuntimeMode::Hardened && n > HARDENED_MAX_DIM {
        return Err(OptError::InvalidArgument {
            detail: format!("dimension {n} exceeds HARDENED_MAX_DIM ({HARDENED_MAX_DIM})"),
        });
    }

    // Validate bounds: lb < ub and no infs
    for i in 0..n {
        let lo = bounds.lb[i];
        let hi = bounds.ub[i];
        if lo.is_infinite() || hi.is_infinite() {
            return Err(OptError::InvalidBounds {
                detail: "Bounds must not be inf.".to_string(),
            });
        }
        if lo >= hi {
            return Err(OptError::InvalidBounds {
                detail: "Bounds are not consistent min < max".to_string(),
            });
        }
    }

    // Validate tolerances
    if !(0.0..=1.0).contains(&options.vol_tol) {
        return Err(OptError::InvalidArgument {
            detail: "vol_tol must be between 0 and 1.".to_string(),
        });
    }
    if !(0.0..=1.0).contains(&options.len_tol) {
        return Err(OptError::InvalidArgument {
            detail: "len_tol must be between 0 and 1.".to_string(),
        });
    }
    if !(0.0..=1.0).contains(&options.f_min_rtol) {
        return Err(OptError::InvalidArgument {
            detail: "f_min_rtol must be between 0 and 1.".to_string(),
        });
    }
    if options.maxiter == 0 {
        return Err(OptError::InvalidArgument {
            detail: "maxiter must be > 0.".to_string(),
        });
    }

    let maxfun = options.maxfun.unwrap_or(1000 * n);
    if maxfun == 0 {
        return Err(OptError::InvalidArgument {
            detail: "maxfun must be > 0.".to_string(),
        });
    }

    // Precompute powers of 1/3: 3^(-k)
    let mut pow3_inv = [0.0; 64];
    pow3_inv[0] = 1.0;
    for i in 1..64 {
        pow3_inv[i] = pow3_inv[i - 1] / 3.0;
    }

    let lb = &bounds.lb;
    let ub = &bounds.ub;
    let scales: Vec<f64> = (0..n).map(|i| ub[i] - lb[i]).collect();

    // Helper to evaluate in original space
    let mut nfev = 0;
    let mut eval_point = |norm_point: &[f64]| -> Result<f64, OptError> {
        let mut x = vec![0.0; n];
        for i in 0..n {
            x[i] = lb[i] + norm_point[i] * scales[i];
        }
        let val = func(&x);
        if options.mode == RuntimeMode::Hardened && !val.is_finite() {
            return Err(OptError::NonFiniteInput {
                detail: format!("Objective evaluated to non-finite value {val} at {x:?}"),
            });
        }
        Ok(val)
    };

    let denormalize = |norm_point: &[f64]| -> Vec<f64> {
        (0..n).map(|i| lb[i] + norm_point[i] * scales[i]).collect()
    };

    // Initialization: center of unit hypercube
    let c0 = vec![0.5; n];
    nfev += 1;
    let f0 = eval_point(&c0)?;

    let mut f_best = f0;
    let mut x_best = denormalize(&c0);

    let mut rectangles: Vec<HyperRectangle> = Vec::with_capacity(maxfun.min(100_000));

    // Sample along each coordinate axis: c0 +/- 1/3 e_i
    let mut axis_evals = Vec::with_capacity(n);
    for i in 0..n {
        let mut c_plus = c0.clone();
        c_plus[i] += 1.0 / 3.0;
        nfev += 1;
        let f_plus = eval_point(&c_plus)?;

        let mut c_minus = c0.clone();
        c_minus[i] -= 1.0 / 3.0;
        nfev += 1;
        let f_minus = eval_point(&c_minus)?;

        if f_plus < f_best {
            f_best = f_plus;
            x_best = denormalize(&c_plus);
        }
        if f_minus < f_best {
            f_best = f_minus;
            x_best = denormalize(&c_minus);
        }

        let w = f_plus.min(f_minus);
        axis_evals.push((i, w, f_plus, f_minus));
    }

    // Sort dimensions by w_i in ascending order
    axis_evals.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    // Create the initial 2n + 1 hyperrectangles
    let mut current_levels = vec![0u8; n];
    let mut current_sum_levels = 0u32;

    for &(dim, _w, f_plus, f_minus) in &axis_evals {
        current_levels[dim] += 1;
        current_sum_levels += 1;

        let mut c_plus = c0.clone();
        c_plus[dim] += pow3_inv[current_levels[dim] as usize];

        let mut c_minus = c0.clone();
        c_minus[dim] -= pow3_inv[current_levels[dim] as usize];

        let min_lvl = *current_levels.iter().min().unwrap();

        let mut rect_plus = HyperRectangle {
            center: c_plus,
            levels: current_levels.clone(),
            sum_levels: current_sum_levels,
            min_level: min_lvl,
            d: 0.0,
            f: f_plus,
        };
        rect_plus.d = rect_plus.side_length_measure(options.locally_biased, &pow3_inv);
        rectangles.push(rect_plus);

        let mut rect_minus = HyperRectangle {
            center: c_minus,
            levels: current_levels.clone(),
            sum_levels: current_sum_levels,
            min_level: min_lvl,
            d: 0.0,
            f: f_minus,
        };
        rect_minus.d = rect_minus.side_length_measure(options.locally_biased, &pow3_inv);
        rectangles.push(rect_minus);
    }

    let min_lvl0 = *current_levels.iter().min().unwrap();
    let mut rect0 = HyperRectangle {
        center: c0,
        levels: current_levels,
        sum_levels: current_sum_levels,
        min_level: min_lvl0,
        d: 0.0,
        f: f0,
    };
    rect0.d = rect0.side_length_measure(options.locally_biased, &pow3_inv);
    rectangles.push(rect0);
    // The centre holds `f0`, so it is the best rectangle until a sample beats it, as SciPy's
    // `minpos` is. The scan below re-points this at the best rectangle whenever `f_best` is
    // comparable; a NaN `f0` is never beaten (`f < NaN` is false), and the termination tests must
    // then measure the NaN centre, not whichever rectangle was pushed first (frankenscipy-vfs3g).
    let mut best_rect_idx = rectangles.len() - 1;

    // Update best rectangle index
    for (idx, r) in rectangles.iter().enumerate() {
        if r.f <= f_best {
            f_best = r.f;
            x_best = denormalize(&r.center);
            best_rect_idx = idx;
        }
    }

    let mut nit = 0;

    let format_message = |code: i32| -> String {
        match code {
            1 => format!("Number of function evaluations done is larger than maxfun={maxfun}"),
            2 => format!(
                "Number of iterations is larger than maxiter={}",
                options.maxiter
            ),
            3 => format!(
                "The best function value found is within a relative error={} of the (known) global optimum f_min",
                options.f_min_rtol
            ),
            4 => format!(
                "The volume of the hyperrectangle containing the lowest function value found is below vol_tol={}",
                options.vol_tol
            ),
            5 => format!(
                "The side length measure of the hyperrectangle containing the lowest function value found is below len_tol={}",
                options.len_tol
            ),
            _ => "Optimization terminated".to_string(),
        }
    };

    // Check termination after initialization
    let check_termination =
        |best_rect: &HyperRectangle, nit: usize, nfev: usize| -> Option<(i32, bool, String)> {
            // 1. Known f_min condition
            if options.f_min > f64::NEG_INFINITY {
                let denom = options.f_min.abs().max(1.0);
                if (best_rect.f - options.f_min).abs() / denom <= options.f_min_rtol {
                    return Some((3, true, format_message(3)));
                }
            }

            // 2. Volume tolerance
            if best_rect.volume(&pow3_inv) <= options.vol_tol {
                return Some((4, true, format_message(4)));
            }

            // 3. Side length measure tolerance
            if best_rect.side_length_measure(options.locally_biased, &pow3_inv) <= options.len_tol {
                return Some((5, true, format_message(5)));
            }

            // 4. Budgets
            if nfev >= maxfun {
                return Some((1, false, format_message(1)));
            }
            if nit >= options.maxiter {
                return Some((2, false, format_message(2)));
            }

            None
        };

    if let Some((status, success, message)) =
        check_termination(&rectangles[best_rect_idx], nit, nfev)
    {
        return Ok(DirectResult {
            x: x_best,
            fun: f_best,
            status,
            success,
            message,
            nfev,
            nit,
        });
    }

    // Main DIRECT iteration loop
    while nit < options.maxiter && nfev < maxfun {
        nit += 1;

        // Step 1: Identify potentially optimal hyperrectangles (POH)
        let mut best_by_size: Vec<(f64, f64, usize)> = Vec::new();

        // Sort all rectangles by size measure d
        let mut sorted_indices: Vec<usize> = (0..rectangles.len()).collect();
        sorted_indices.sort_by(|&a, &b| {
            rectangles[a]
                .d
                .partial_cmp(&rectangles[b].d)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Group by distinct sizes (within 1e-12 relative difference)
        for &idx in &sorted_indices {
            let r = &rectangles[idx];
            if let Some(last) = best_by_size.last_mut() {
                let rel_diff = (r.d - last.0).abs() / last.0.max(1e-15);
                if rel_diff < 1e-12 {
                    if r.f < last.1 {
                        last.1 = r.f;
                        last.2 = idx;
                    }
                    continue;
                }
            }
            best_by_size.push((r.d, r.f, idx));
        }

        // Lower convex hull test among candidates
        let m = best_by_size.len();
        let mut poh = Vec::new();
        let f_min_denom = f_best.abs().max(1.0);

        for k in 0..m {
            let (dk, fk, idx_k) = best_by_size[k];

            // K_L: lower bound on slope
            let mut k_l = 0.0f64;
            for j in 0..k {
                let (dj, fj, _) = best_by_size[j];
                let slope = (fk - fj) / (dk - dj);
                if slope > k_l {
                    k_l = slope;
                }
            }

            // K_U: upper bound on slope
            let mut k_u = f64::INFINITY;
            for j in (k + 1)..m {
                let (dj, fj, _) = best_by_size[j];
                let slope = (fj - fk) / (dj - dk);
                if slope < k_u {
                    k_u = slope;
                }
            }

            // Epsilon tradeoff condition: fk - K * dk <= f_best - eps * max(|f_best|, 1.0)
            let k_eps = (fk - (f_best - options.eps * f_min_denom)) / dk;

            let lower_bound = k_l.max(k_eps).max(0.0);
            if lower_bound <= k_u {
                poh.push(idx_k);
            }
        }

        // If locally biased, always ensure the overall best rectangle is included -- unless its
        // value is NaN: forcing a NaN centre into every division shrank it to `len_tol` and
        // reported success, where SciPy runs to `maxfun` (status 1).
        if options.locally_biased
            && !rectangles[best_rect_idx].f.is_nan()
            && !poh.contains(&best_rect_idx)
        {
            poh.push(best_rect_idx);
        }

        // Step 2: Divide each potentially optimal hyperrectangle
        let mut new_rectangles = Vec::new();

        for &rect_idx in &poh {
            if nfev >= maxfun {
                break;
            }

            // Find dimensions with the maximal side length (min level)
            let min_lvl = rectangles[rect_idx].min_level;
            let max_dims: Vec<usize> = (0..n)
                .filter(|&dim| rectangles[rect_idx].levels[dim] == min_lvl)
                .collect();

            // Evaluate objective along each maximal dimension
            let mut dim_evals = Vec::with_capacity(max_dims.len());
            let delta = pow3_inv[(min_lvl + 1).min(63) as usize];

            for &dim in &max_dims {
                if nfev >= maxfun {
                    break;
                }
                let mut c_plus = rectangles[rect_idx].center.clone();
                c_plus[dim] += delta;
                nfev += 1;
                let f_plus = eval_point(&c_plus)?;

                if nfev >= maxfun {
                    break;
                }
                let mut c_minus = rectangles[rect_idx].center.clone();
                c_minus[dim] -= delta;
                nfev += 1;
                let f_minus = eval_point(&c_minus)?;

                if f_plus < f_best {
                    f_best = f_plus;
                    x_best = denormalize(&c_plus);
                }
                if f_minus < f_best {
                    f_best = f_minus;
                    x_best = denormalize(&c_minus);
                }

                let w = f_plus.min(f_minus);
                dim_evals.push((dim, w, f_plus, f_minus));
            }

            // Sort by w_i in ascending order
            dim_evals.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });

            // Divide sequentially along the sorted dimensions
            for &(dim, _w, f_plus, f_minus) in &dim_evals {
                rectangles[rect_idx].levels[dim] += 1;
                rectangles[rect_idx].sum_levels += 1;

                let mut c_plus = rectangles[rect_idx].center.clone();
                c_plus[dim] += delta;

                let mut c_minus = rectangles[rect_idx].center.clone();
                c_minus[dim] -= delta;

                let new_min_lvl = *rectangles[rect_idx].levels.iter().min().unwrap();

                let mut r_plus = HyperRectangle {
                    center: c_plus,
                    levels: rectangles[rect_idx].levels.clone(),
                    sum_levels: rectangles[rect_idx].sum_levels,
                    min_level: new_min_lvl,
                    d: 0.0,
                    f: f_plus,
                };
                r_plus.d = r_plus.side_length_measure(options.locally_biased, &pow3_inv);
                new_rectangles.push(r_plus);

                let mut r_minus = HyperRectangle {
                    center: c_minus,
                    levels: rectangles[rect_idx].levels.clone(),
                    sum_levels: rectangles[rect_idx].sum_levels,
                    min_level: new_min_lvl,
                    d: 0.0,
                    f: f_minus,
                };
                r_minus.d = r_minus.side_length_measure(options.locally_biased, &pow3_inv);
                new_rectangles.push(r_minus);
            }

            // Update the center rectangle's min_level and d
            rectangles[rect_idx].min_level = *rectangles[rect_idx].levels.iter().min().unwrap();
            rectangles[rect_idx].d =
                rectangles[rect_idx].side_length_measure(options.locally_biased, &pow3_inv);
        }

        rectangles.extend(new_rectangles);

        // Update overall best rectangle index
        for (idx, r) in rectangles.iter().enumerate() {
            if r.f <= f_best {
                f_best = r.f;
                x_best = denormalize(&r.center);
                best_rect_idx = idx;
            }
        }

        if let Some(cb) = callback.as_mut() {
            cb(&x_best);
        }

        // Check termination conditions
        if let Some((status, success, message)) =
            check_termination(&rectangles[best_rect_idx], nit, nfev)
        {
            return Ok(DirectResult {
                x: x_best,
                fun: f_best,
                status,
                success,
                message,
                nfev,
                nit,
            });
        }
    }

    // Default exit if loop finishes without trigger
    let status = if nfev >= maxfun { 1 } else { 2 };
    Ok(DirectResult {
        x: x_best,
        fun: f_best,
        status,
        success: false,
        message: format_message(status),
        nfev,
        nit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direct_sphere_1d() {
        let bounds = Bounds::new(vec![-3.0], vec![3.0]).unwrap();
        let res = direct(|x| x[0] * x[0], &bounds, DirectOptions::default()).unwrap();
        assert!(res.success);
        assert!(res.fun.abs() < 1e-6);
        assert!(res.x[0].abs() < 1e-3);
    }

    #[test]
    fn test_direct_sphere_2d() {
        let bounds = Bounds::new(vec![-5.0, -5.0], vec![5.0, 5.0]).unwrap();
        let res = direct(
            |x| x[0] * x[0] + x[1] * x[1],
            &bounds,
            DirectOptions::default(),
        )
        .unwrap();
        assert!(res.success);
        assert!(res.fun.abs() < 1e-6);
        assert!(res.x[0].abs() < 1e-3);
        assert!(res.x[1].abs() < 1e-3);
    }

    #[test]
    fn test_direct_rosenbrock_2d() {
        let bounds = Bounds::new(vec![-2.0, -1.0], vec![2.0, 3.0]).unwrap();
        let opts = DirectOptions {
            maxiter: 50,
            ..Default::default()
        };
        let res = direct(
            |x| 100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2),
            &bounds,
            opts,
        )
        .unwrap();
        assert!(res.fun < 0.1);
        assert!((res.x[0] - 1.0).abs() < 0.1);
        assert!((res.x[1] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_direct_styblinski_tang() {
        let bounds = Bounds::new(vec![-4.0, -4.0], vec![4.0, 4.0]).unwrap();
        let opts = DirectOptions {
            len_tol: 1e-3,
            ..Default::default()
        };
        let res = direct(
            |x| {
                0.5 * (x[0].powi(4) - 16.0 * x[0].powi(2) + 5.0 * x[0] + x[1].powi(4)
                    - 16.0 * x[1].powi(2)
                    + 5.0 * x[1])
            },
            &bounds,
            opts,
        )
        .unwrap();
        assert!(res.success);
        assert!((res.fun - -78.3323).abs() < 1e-2);
        assert!((res.x[0] - -2.9044).abs() < 1e-2);
        assert!((res.x[1] - -2.9044).abs() < 1e-2);
    }

    #[test]
    fn test_direct_invalid_bounds() {
        let bounds = Bounds {
            lb: vec![5.0],
            ub: vec![2.0],
        };
        let err = direct(|x| x[0], &bounds, DirectOptions::default()).unwrap_err();
        assert!(matches!(err, OptError::InvalidBounds { .. }));
    }

    /// frankenscipy-vfs3g: a NaN at the centre is never beaten (`f < NaN` is false), so `f_best`
    /// stays NaN. The termination tests then measured rectangle 0 instead of the centre, and
    /// DIRECT-L forced that rectangle into every division, so it shrank to `len_tol` and the run
    /// reported success (status 5). SciPy 1.17.1, default options:
    ///
    /// * NaN at x = 0.5, (x - 0.2)² elsewhere, bounds [(0, 1)]: x [0.5], fun nan, status 1,
    ///   success False, nfev 1011, nit 34, "Number of function evaluations done is larger than
    ///   maxfun=1000".
    /// * NaN at (0, 0), (x0 - 0.3)² + (x1 + 0.4)² elsewhere, bounds [(-1, 1)]²: x [0, 0], fun nan,
    ///   status 1, success False, nfev 2017, nit 56.
    #[test]
    fn test_direct_nan_centre_runs_to_maxfun_like_scipy() -> Result<(), OptError> {
        let unit = Bounds::new(vec![0.0], vec![1.0])?;
        let nan_centre = |x: &[f64]| {
            if x[0] == 0.5 {
                f64::NAN
            } else {
                (x[0] - 0.2).powi(2)
            }
        };
        let res = direct(nan_centre, &unit, DirectOptions::default())?;
        assert_eq!((res.status, res.success), (1, false), "{res:?}");
        assert!(
            res.fun.is_nan() && res.x == [0.5] && res.nfev >= 1000,
            "{res:?}"
        );
        assert_eq!(
            res.message,
            "Number of function evaluations done is larger than maxfun=1000"
        );

        let square = Bounds::new(vec![-1.0, -1.0], vec![1.0, 1.0])?;
        let nan_centre_2d = |x: &[f64]| {
            if x == [0.0, 0.0] {
                f64::NAN
            } else {
                (x[0] - 0.3).powi(2) + (x[1] + 0.4).powi(2)
            }
        };
        let res = direct(nan_centre_2d, &square, DirectOptions::default())?;
        assert_eq!((res.status, res.success), (1, false), "{res:?}");
        assert!(res.fun.is_nan() && res.x == [0.0, 0.0], "{res:?}");

        // Must not change: a comparable best value -- finite, or -inf at the centre -- still ends
        // on len_tol (SciPy: status 5 at x ≈ 0.2, and at x = 0.5 with fun -inf).
        let res = direct(|x| (x[0] - 0.2).powi(2), &unit, DirectOptions::default())?;
        assert_eq!(
            (res.status, res.success, res.nfev, res.nit),
            (5, true, 87, 11),
            "{res:?}"
        );
        assert!((res.x[0] - 0.2).abs() < 1e-5, "{res:?}");
        let neg_inf_centre = |x: &[f64]| {
            if x[0] == 0.5 {
                f64::NEG_INFINITY
            } else {
                (x[0] - 0.2).powi(2)
            }
        };
        let res = direct(neg_inf_centre, &unit, DirectOptions::default())?;
        assert_eq!((res.status, res.success), (5, true), "{res:?}");
        assert!(res.fun == f64::NEG_INFINITY && res.x == [0.5], "{res:?}");
        Ok(())
    }

    #[test]
    fn test_direct_infinite_bounds() {
        let bounds = Bounds {
            lb: vec![f64::NEG_INFINITY],
            ub: vec![2.0],
        };
        let err = direct(|x| x[0], &bounds, DirectOptions::default()).unwrap_err();
        assert!(matches!(err, OptError::InvalidBounds { .. }));
    }
}
