//! Chandrupatla's bracketing root finder — `scipy.optimize.elementwise.find_root`.
//!
//! WHY A SEPARATE MODULE. `root.rs` holds this crate's root finders, but it is 4,700 lines and
//! under concurrent edit by another pane, and Agent Mail — the only way to take a reservation
//! on it — is unreachable. SciPy keeps this algorithm in its own `_chandrupatla.py` for the
//! same reason it stands alone here: it is a distinct method with its own state machine, not a
//! variation on `brentq`. Nothing in `root.rs` is duplicated.
//!
//! ## The algorithm
//!
//! Chandrupatla's method interpolates where interpolation is trustworthy and bisects where it
//! is not. Each iteration keeps three points — the current pair `(x1, x2)` that brackets the
//! root plus the point `x3` displaced by the last step — and picks the next abscissa as
//! `x = x1 + t·(x2 - x1)`, where `t` is:
//!
//!   * the **inverse quadratic interpolation** weight, when the three points pass Chandrupatla's
//!     admissibility test `1 - √(1 - ξ) < φ < √ξ` (his Equation 1), which is what makes the
//!     method converge faster than bisection on smooth functions; or
//!   * **½**, a plain bisection, whenever that test fails — which is what stops it degenerating
//!     on awkward ones.
//!
//! `t` is then clipped away from the interval boundary by `0.5·tol/dx`, so a step can never
//! land on an endpoint and stall.
//!
//! Verified by hand against the incumbent before a line was written: for `x³ - 2` on `(1, 2)`
//! the first two abscissae are `1.5` (bisection, since `t` starts at ½) and then
//! `1.230644178…` from the interpolation weight `t = 0.538712…`. Both arms produce exactly
//! that, and `diff_optimize_find_root` compares the whole schedule.
//!
//! ## The sign convention is NOT Rust's
//!
//! The method compares `sign(f)` in two places. NumPy's `sign` maps zero to **zero**, while
//! Rust's [`f64::signum`] maps `0.0` to `1.0` and `-0.0` to `-1.0`. Using `signum` here would
//! silently take the wrong branch whenever an evaluation lands exactly on the root — the case
//! that matters most — so [`numpy_sign`] is used throughout and is pinned by its own test.

use crate::types::OptError;

/// SciPy's default iteration cap, `log2(f64::MAX) - log2(f64::MIN_POSITIVE)` = `1024 - (-1022)`.
/// It is a bisection bound: that many halvings exhausts the exponent range.
const DEFAULT_MAXITER: usize = 2046;

/// Why a [`find_root`] search stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FindRootStatus {
    /// A root was found to within the tolerances.
    Converged,
    /// The initial interval does not bracket a sign change.
    SignError,
    /// `maxiter` was exhausted.
    MaxIterations,
    /// A non-finite abscissa, or both function values NaN.
    NonFinite,
}

/// Outcome of a [`find_root`] search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FindRootResult {
    /// The best estimate of the root — whichever endpoint has the smaller `|f|`.
    ///
    /// NaN for [`FindRootStatus::SignError`] and [`FindRootStatus::NonFinite`], where no
    /// iterate means anything. NOT NaN for [`FindRootStatus::MaxIterations`]: the search was
    /// converging when the budget ran out, and its best estimate is usable.
    pub x: f64,
    /// `f` at [`Self::x`], NaN under the same two conditions.
    pub f_x: f64,
    /// The final bracketing interval, ordered ascending.
    pub bracket: (f64, f64),
    /// `f` at each end of [`Self::bracket`], in the same order.
    pub f_bracket: (f64, f64),
    /// Iterations performed. Zero when the initial interval already resolves the search.
    pub nit: usize,
    /// Function evaluations performed, including the two initial ones.
    pub nfev: usize,
    /// Why the search stopped.
    pub status: FindRootStatus,
    /// Whether a root was found.
    pub success: bool,
}

/// Options for [`find_root`], defaulting to SciPy's.
///
/// The convergence test is `|x2 - x1| < xatol + |x| · xrtol` OR `|f(x)| <= fatol + frtol · m`,
/// where `m` is `min(|f(xl0)|, |f(xr0)|)` at the ORIGINAL interval ends — `frtol` is relative
/// to how large the function was to begin with, not to the current iterate.
/// Every field defaults to `None`, meaning "use SciPy's value", so `Default` is derived rather
/// than written out — unlike `BracketOptions`, whose `maxiter` carries a concrete default.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct FindRootOptions {
    /// Absolute tolerance on the bracket width. `None` means `4 · f64::MIN_POSITIVE`.
    pub xatol: Option<f64>,
    /// Relative tolerance on the bracket width. `None` means `4 · f64::EPSILON`.
    pub xrtol: Option<f64>,
    /// Absolute tolerance on the function value. `None` means `f64::MIN_POSITIVE`.
    pub fatol: Option<f64>,
    /// Relative tolerance on the function value. `None` means `0.0`.
    pub frtol: Option<f64>,
    /// Maximum iterations. `None` means 2046.
    pub maxiter: Option<usize>,
}

/// NumPy's `sign`, which maps ZERO TO ZERO — unlike [`f64::signum`], which maps `0.0` to `1.0`
/// and `-0.0` to `-1.0`.
///
/// The distinction is load-bearing twice over. `sign(f1) == sign(f2)` decides whether the
/// interval still brackets a root: with `signum`, an exact zero at one end would read as a
/// definite `+1` or `-1` and could compare equal to the other end's sign, reporting "no
/// bracket" for an interval whose endpoint IS the root. And `sign(f) == sign(f1)` decides which
/// point the method discards each iteration, so the same confusion would silently reshape the
/// search.
fn numpy_sign(v: f64) -> f64 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else if v == 0.0 {
        0.0
    } else {
        f64::NAN
    }
}

/// The three-point state Chandrupatla's method carries between iterations.
#[derive(Debug, Clone, Copy)]
struct Work {
    x1: f64,
    f1: f64,
    x2: f64,
    f2: f64,
    x3: f64,
    f3: f64,
}

/// What a termination check concluded.
struct Verdict {
    stop: Option<FindRootStatus>,
    /// The better of the two current endpoints, by `|f|`.
    xmin: f64,
    fmin: f64,
    /// Current interval width, and the width tolerance it is measured against. Both feed the
    /// next step's boundary clip, so they are computed even when the search continues.
    dx: f64,
    tol: f64,
}

/// Find a root of `f` inside a bracketing interval, by Chandrupatla's method.
///
/// Returns `success: false` with a [`FindRootStatus`] rather than an error for the outcomes
/// that are answers about the problem — the interval not bracketing a sign change, or the
/// budget running out — and reserves `Err` for arguments the search cannot start from.
///
/// # Errors
///
/// Returns [`OptError::InvalidArgument`] if the interval ends are not finite and distinct, or
/// if any supplied tolerance is negative or not finite.
pub fn find_root<F>(
    f: F,
    init: (f64, f64),
    options: FindRootOptions,
) -> Result<FindRootResult, OptError>
where
    F: Fn(f64) -> f64,
{
    let (xl0, xr0) = init;
    if !xl0.is_finite() || !xr0.is_finite() {
        return Err(OptError::InvalidArgument {
            detail: format!("find_root requires a finite interval, got ({xl0}, {xr0})"),
        });
    }
    if xl0 == xr0 {
        return Err(OptError::InvalidArgument {
            detail: format!("find_root requires distinct interval ends, both are {xl0}"),
        });
    }

    let xatol = options.xatol.unwrap_or(4.0 * f64::MIN_POSITIVE);
    let xrtol = options.xrtol.unwrap_or(4.0 * f64::EPSILON);
    let fatol = options.fatol.unwrap_or(f64::MIN_POSITIVE);
    let frtol = options.frtol.unwrap_or(0.0);
    for (name, value) in [
        ("xatol", xatol),
        ("xrtol", xrtol),
        ("fatol", fatol),
        ("frtol", frtol),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(OptError::InvalidArgument {
                detail: format!("find_root requires a finite, non-negative {name}, got {value}"),
            });
        }
    }
    let maxiter = options.maxiter.unwrap_or(DEFAULT_MAXITER);

    let mut work = Work {
        x1: xl0,
        f1: f(xl0),
        x2: xr0,
        f2: f(xr0),
        // Set by the first `advance`; never read before then, because the interpolation weight
        // is only computed after an iteration has produced a third point.
        x3: f64::NAN,
        f3: f64::NAN,
    };
    let mut nfev = 2usize;

    // `frtol` is relative to the function's size at the ORIGINAL interval ends, fixed once here
    // rather than re-derived from the shrinking bracket.
    let frtol_scaled = frtol * work.f1.abs().min(work.f2.abs());

    let check = |w: &Work| -> Verdict {
        // Section 4 of the paper: track whichever end is closer to a root.
        let better_is_first = w.f1.abs() < w.f2.abs();
        let (xmin, fmin) = if better_is_first {
            (w.x1, w.f1)
        } else {
            (w.x2, w.f2)
        };
        let dx = (w.x2 - w.x1).abs();
        let tol = xmin.abs() * xrtol + xatol;

        // A satisfied function tolerance wins over every other condition, including the
        // sign check below — an endpoint sitting exactly on the root is a success, not a
        // degenerate bracket.
        let stop = if fmin.abs() <= fatol + frtol_scaled {
            Some(FindRootStatus::Converged)
        } else if numpy_sign(w.f1) == numpy_sign(w.f2) {
            Some(FindRootStatus::SignError)
        } else if !(w.x1.is_finite() && w.x2.is_finite()) || (w.f1.is_nan() && w.f2.is_nan()) {
            Some(FindRootStatus::NonFinite)
        } else if dx < tol {
            Some(FindRootStatus::Converged)
        } else {
            None
        };
        Verdict {
            stop,
            xmin,
            fmin,
            dx,
            tol,
        }
    };

    let finish = |w: &Work, verdict: &Verdict, status: FindRootStatus, nit, nfev| {
        // Only the two conditions that invalidate the ITERATE erase it. A spent budget does
        // not: the search was converging, and its best estimate so far is real information a
        // caller can restart from. SciPy makes the same distinction — it assigns NaN inside
        // the sign-error and value-error branches specifically, not on max-iterations, which
        // returns 1.2306441780125992 for `x**3 - 2` on (1, 2) at maxiter=2.
        let answer_is_meaningless = matches!(
            status,
            FindRootStatus::SignError | FindRootStatus::NonFinite
        );
        let failed = !matches!(status, FindRootStatus::Converged);
        // Report the interval ascending regardless of which end the search was working from.
        let (bracket, f_bracket) = if w.x1 < w.x2 {
            ((w.x1, w.x2), (w.f1, w.f2))
        } else {
            ((w.x2, w.x1), (w.f2, w.f1))
        };
        FindRootResult {
            x: if answer_is_meaningless {
                f64::NAN
            } else {
                verdict.xmin
            },
            f_x: if answer_is_meaningless {
                f64::NAN
            } else {
                verdict.fmin
            },
            bracket,
            f_bracket,
            nit,
            nfev,
            status,
            success: !failed,
        }
    };

    // The initial interval may already settle the question — a root at an endpoint, or no sign
    // change at all — in which case no iteration happens and `nit` is 0.
    let mut verdict = check(&work);
    if let Some(status) = verdict.stop {
        return Ok(finish(&work, &verdict, status, 0, nfev));
    }

    // The first step is a plain bisection: there is no third point to interpolate through yet.
    let mut t = 0.5;

    for nit in 1..=maxiter {
        let x = work.x1 + t * (work.x2 - work.x1);
        let fx = f(x);
        nfev += 1;

        // Retire whichever point leaves a bracketing pair behind. `x3` keeps the discarded
        // point, which the interpolation needs.
        work.x3 = work.x2;
        work.f3 = work.f2;
        if numpy_sign(fx) == numpy_sign(work.f1) {
            work.x3 = work.x1;
            work.f3 = work.f1;
        } else {
            work.x2 = work.x1;
            work.f2 = work.f1;
        }
        work.x1 = x;
        work.f1 = fx;

        verdict = check(&work);
        if let Some(status) = verdict.stop {
            return Ok(finish(&work, &verdict, status, nit, nfev));
        }
        if nit == maxiter {
            break;
        }

        // Chandrupatla's Equation 1: interpolate only where the three points make it safe.
        let xi1 = (work.x1 - work.x2) / (work.x3 - work.x2);
        let phi1 = (work.f1 - work.f2) / (work.f3 - work.f2);
        let admissible = (1.0 - (1.0 - xi1).sqrt()) < phi1 && phi1 < xi1.sqrt();
        t = if admissible {
            let alpha = (work.x3 - work.x1) / (work.x2 - work.x1);
            work.f1 / (work.f1 - work.f2) * work.f3 / (work.f3 - work.f2)
                - alpha * work.f1 / (work.f3 - work.f1) * work.f2 / (work.f2 - work.f3)
        } else {
            0.5
        };

        // "Adjust T away from the interval boundary" — without this a step can land on an
        // endpoint and the method stalls. Written as explicit comparisons rather than
        // `f64::clamp`, which panics when the bounds cross and which would need the invariant
        // `dx >= tol` argued rather than simply not relied upon. NaN falls through unchanged,
        // matching `numpy.clip`.
        let tl = 0.5 * verdict.tol / verdict.dx;
        if t < tl {
            t = tl;
        } else if t > 1.0 - tl {
            t = 1.0 - tl;
        }
    }

    Ok(finish(
        &work,
        &verdict,
        FindRootStatus::MaxIterations,
        maxiter,
        nfev,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn traced<F: Fn(f64) -> f64>(
        f: F,
        visited: &std::cell::RefCell<Vec<f64>>,
    ) -> impl Fn(f64) -> f64 {
        move |x| {
            visited.borrow_mut().push(x);
            f(x)
        }
    }

    #[test]
    fn numpy_sign_maps_zero_to_zero_unlike_rust_signum() {
        // The whole reason this helper exists. If these two agreed, `signum` would do.
        assert_eq!(numpy_sign(0.0), 0.0);
        assert_eq!(numpy_sign(-0.0), 0.0);
        assert_eq!(0.0f64.signum(), 1.0);
        assert_eq!((-0.0f64).signum(), -1.0);

        assert_eq!(numpy_sign(3.5), 1.0);
        assert_eq!(numpy_sign(-3.5), -1.0);
        assert!(numpy_sign(f64::NAN).is_nan());

        // The consequence, stated directly: a root sitting exactly at an endpoint must not
        // read as "both ends have the same sign".
        assert_ne!(numpy_sign(0.0), numpy_sign(6.0));
        assert_eq!(0.0f64.signum(), 6.0f64.signum());
    }

    #[test]
    fn find_root_matches_live_scipy_on_a_cubic() {
        // Live `elementwise.find_root(lambda x: x**3 - 2, (1.0, 2.0))` on scipy 1.17.1 returns
        // x = 1.2599210498948732 (the cube root of 2), nit=6, nfev=8, sampling
        // 1.0, 2.0, 1.5, 1.230644178..., 1.2619221954..., 1.2599084888..., 1.2599210507...,
        // 1.2599210499...
        let visited = std::cell::RefCell::new(Vec::new());
        let result = find_root(
            traced(|x| x * x * x - 2.0, &visited),
            (1.0, 2.0),
            FindRootOptions::default(),
        )
        .expect("root search succeeds");

        assert!(result.success);
        assert_eq!(result.status, FindRootStatus::Converged);
        assert_eq!(result.x, 1.259_921_049_894_873_2);
        assert_eq!(result.f_x, 0.0);
        assert_eq!(result.nit, 6);
        assert_eq!(result.nfev, 8);

        let seen = visited.borrow();
        assert_eq!(seen.len(), 8);
        assert_eq!(
            &seen[..3],
            &[1.0, 2.0, 1.5],
            "the first step is a bisection"
        );
        // The second step is the interpolation, and it is the one that distinguishes this
        // method from bisection: bisecting again would have given 1.25.
        assert_eq!(seen[3], 1.230_644_178_012_599_2);
        assert!(
            (seen[3] - 1.25).abs() > 1e-3,
            "step 2 must be interpolated, not bisected"
        );
    }

    #[test]
    fn find_root_reports_a_root_sitting_on_an_endpoint_without_iterating() {
        // Live scipy: find_root(lambda x: x - 1, (1.0, 2.0)) -> x=1.0, nit=0, nfev=2. This is
        // the case Rust's `signum` would break: with it, sign(0.0) == sign(1.0) and the search
        // would report SignError on an interval whose left end IS the root.
        let result = find_root(|x| x - 1.0, (1.0, 2.0), FindRootOptions::default())
            .expect("root search succeeds");
        assert!(result.success);
        assert_eq!(result.x, 1.0);
        assert_eq!(result.f_x, 0.0);
        assert_eq!(result.nit, 0);
        assert_eq!(result.nfev, 2);
    }

    #[test]
    fn find_root_reports_a_sign_error_rather_than_inventing_a_root() {
        // Live scipy: x=nan, nit=0, nfev=2, status=-1, success=False.
        let result = find_root(|x| x * x + 1.0, (1.0, 2.0), FindRootOptions::default())
            .expect("a non-bracketing interval is an answer, not an error");
        assert!(!result.success);
        assert_eq!(result.status, FindRootStatus::SignError);
        assert!(result.x.is_nan(), "got {}", result.x);
        assert!(result.f_x.is_nan());
        assert_eq!(result.nit, 0);
        assert_eq!(result.nfev, 2);
        // The interval it could not use is still reported, so a caller can see where it looked.
        assert_eq!(result.bracket, (1.0, 2.0));
    }

    #[test]
    fn find_root_stops_at_its_iteration_budget() {
        // Live scipy with maxiter=2: x=1.2306441780125992, nit=2, nfev=4, status=-2.
        let result = find_root(
            |x| x * x * x - 2.0,
            (1.0, 2.0),
            FindRootOptions {
                maxiter: Some(2),
                ..FindRootOptions::default()
            },
        )
        .expect("root search succeeds");
        assert!(!result.success);
        assert_eq!(result.status, FindRootStatus::MaxIterations);
        assert_eq!(result.nit, 2);
        assert_eq!(result.nfev, 4);
        // A SPENT BUDGET DOES NOT ERASE THE ITERATE. The search was converging; its best
        // estimate is real information to restart from, and scipy reports it too. Only a sign
        // error or a non-finite value NaNs the answer, and conflating the three is exactly the
        // bug the differential test caught here.
        assert_eq!(
            result.x, 1.230_644_178_012_599_2,
            "max-iterations must keep the best estimate, not report NaN"
        );
        assert!(result.f_x.is_finite());
    }

    #[test]
    fn find_root_honours_loosened_tolerances() {
        // Live scipy with frtol=1e-3 stops early: x=1.2599084888427166, nit=4, nfev=6.
        let loose = find_root(
            |x| x * x * x - 2.0,
            (1.0, 2.0),
            FindRootOptions {
                frtol: Some(1e-3),
                ..FindRootOptions::default()
            },
        )
        .expect("root search succeeds");
        assert!(loose.success);
        assert_eq!(loose.x, 1.259_908_488_842_716_6);
        assert_eq!(loose.nit, 4);
        assert_eq!(loose.nfev, 6);

        // And a loosened xatol stops on the width test instead: nit=6, nfev=8, and scipy
        // reports the OTHER endpoint here, 1.2599210506845442.
        let wide = find_root(
            |x| x * x * x - 2.0,
            (1.0, 2.0),
            FindRootOptions {
                xatol: Some(1e-6),
                xrtol: Some(0.0),
                fatol: Some(0.0),
                frtol: Some(0.0),
                ..FindRootOptions::default()
            },
        )
        .expect("root search succeeds");
        assert!(wide.success);
        assert_eq!(wide.x, 1.259_921_050_684_544_2);
        assert_eq!(wide.nit, 6);
        assert_eq!(wide.nfev, 8);
    }

    #[test]
    fn find_root_reports_its_bracket_ascending_whichever_way_it_was_given() {
        let forward = find_root(|x| x * x - 2.0, (1.0, 2.0), FindRootOptions::default())
            .expect("root search succeeds");
        let reversed = find_root(|x| x * x - 2.0, (2.0, 1.0), FindRootOptions::default())
            .expect("root search succeeds");
        assert!(forward.bracket.0 <= forward.bracket.1);
        assert!(reversed.bracket.0 <= reversed.bracket.1);
        assert!((forward.x - 2.0f64.sqrt()).abs() < 1e-12);
        assert!((reversed.x - 2.0f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn find_root_rejects_intervals_it_cannot_start_from() {
        let opts = FindRootOptions::default();
        assert!(find_root(|x| x, (1.0, 1.0), opts).is_err(), "degenerate");
        assert!(
            find_root(|x| x, (f64::NAN, 1.0), opts).is_err(),
            "non-finite"
        );
        assert!(
            find_root(|x| x, (0.0, f64::INFINITY), opts).is_err(),
            "non-finite"
        );
        assert!(
            find_root(
                |x| x,
                (-1.0, 1.0),
                FindRootOptions {
                    xatol: Some(-1.0),
                    ..opts
                }
            )
            .is_err(),
            "a negative tolerance is not a loose tolerance"
        );
        assert!(
            find_root(
                |x| x,
                (-1.0, 1.0),
                FindRootOptions {
                    frtol: Some(f64::NAN),
                    ..opts
                }
            )
            .is_err()
        );
    }

    #[test]
    fn find_root_solves_functions_bisection_would_be_slow_on() {
        // A flat-then-steep function: interpolation should beat the ~50 iterations plain
        // bisection needs for this interval and tolerance.
        let result = find_root(
            |x| x * x * x * x * x - 1e-6,
            (0.0, 10.0),
            FindRootOptions::default(),
        )
        .expect("root search succeeds");
        assert!(result.success, "status {:?}", result.status);
        assert!(
            (result.x - 1e-6f64.powf(0.2)).abs() < 1e-9,
            "got {}",
            result.x
        );
        assert!(result.nit < 60, "took {} iterations", result.nit);
    }
}
