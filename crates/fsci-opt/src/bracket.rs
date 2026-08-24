//! Automatic bracket search for a scalar root — `scipy.optimize.elementwise.bracket_root`.
//!
//! WHY THIS EXISTS. `root.rs` already implements every bracketing solver SciPy has
//! (`brentq`, `brenth`, `bisect`, `ridder`, `toms748`), and every one of them REQUIRES a
//! bracket the caller has already found. SciPy ships the search that produces one; a surface
//! diff of `scipy.optimize.elementwise` against this crate found all four of its callables
//! absent (`bracket_root`, `bracket_minimum`, `find_root`, `find_minimum`), and this is the
//! first of them — the one the existing solvers immediately consume.
//!
//! ## The algorithm, matched to SciPy's `_bracket_root` rather than reinvented
//!
//! Two searches run simultaneously and interleaved, one walking left from `xl0` and one
//! walking right from `xr0`, each with its own "fixed" end (the other side's start) and its
//! own distance `d`:
//!
//!   * an UNLIMITED side grows away from its fixed end: `d *= factor`, `x = x0 + d`;
//!   * a LIMITED side (one with `xmin`/`xmax`) shrinks toward its limit instead:
//!     `d /= factor`, `x = limit - d`, so it approaches the limit geometrically and never
//!     steps past it.
//!
//! The search stops as soon as EITHER side sees a sign change between its previous and
//! current point — including an exact zero at either — and returns that side's
//! `(x_last, x)`. Verified against the live incumbent: for `f(x) = x - 100` from `xl0 = 1`
//! SciPy samples 1, 2, 0, 3, -2, 5, -6, 9, -14, 17, -30, 33, -62, 65 and returns
//! `(65, 129)`; with `xmin = 0` and a root at 0.25 it samples 1, 2, 0.5, 3, 0.25, 5 and
//! returns `(0.25, 0.5)`. Both are reproduced exactly by `bracket_root_matches_live_scipy_*`.
//!
//! The interleaving order is part of the contract, not an implementation detail: it decides
//! which side wins when both would bracket on the same iteration, and therefore which
//! bracket is returned.
//!
//! ## Two DELIBERATE divergences from the incumbent, both measured
//!
//! 1. **`nfev` when the leftward search finishes first.** SciPy vectorizes the two searches
//!    into one array and evaluates both halves in a single call, so it pays for the rightward
//!    point even on the iteration where the leftward one has already bracketed. For
//!    `f(x) = x + 50` from `xl0 = 0` it reports `nfev = 14`; we report 13, having skipped an
//!    evaluation whose result cannot change the answer. The returned bracket is identical.
//!    Not calling a user's function an extra time is worth one integer of divergence.
//!
//! 2. **Both searches bracketing on the SAME iteration.** Here the incumbent is simply wrong.
//!    `scipy.optimize.elementwise.bracket_root(lambda x: x**2 - 100, -1.0, xr0=1.0)` on
//!    scipy 1.17.1 returns `bracket = (7.0, -7.0)`, `f_bracket = (-51.0, -51.0)`,
//!    `success = True` — it takes `xl` from the RIGHTWARD search and `xr` from the LEFTWARD
//!    one, producing two endpoints of the SAME sign, which is not a bracket and which any
//!    downstream `brentq` will reject. We return the leftward search's `(-15, -7)`, a real
//!    bracket. `bracket_root_returns_a_true_bracket_where_the_incumbent_does_not` pins this,
//!    and `success` here means an actual sign change, so it can be trusted by callers.

use crate::types::OptError;

/// Outcome of a bracket search.
///
/// Mirrors the fields SciPy returns (`bracket`, `f_bracket`, `nit`, `nfev`, `status`,
/// `success`) so a caller porting from SciPy finds what it expects.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BracketResult {
    /// The bracketing endpoints, ordered `(xl, xr)` with `xl <= xr`.
    pub bracket: (f64, f64),
    /// `f` at each endpoint, in the same order.
    pub f_bracket: (f64, f64),
    /// Iterations performed. Zero when the initial pair already brackets.
    pub nit: usize,
    /// Function evaluations performed, including the two initial ones.
    pub nfev: usize,
    /// Whether a bracket was found.
    pub success: bool,
}

/// Options for [`bracket_root`], defaulting to SciPy's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BracketOptions {
    /// Right endpoint of the initial interval. `None` means `xl0 + 1.0`, as SciPy does.
    pub xr0: Option<f64>,
    /// Lower limit the leftward search may approach but never cross.
    pub xmin: Option<f64>,
    /// Upper limit the rightward search may approach but never cross.
    pub xmax: Option<f64>,
    /// Geometric factor. `None` means 2.0.
    pub factor: Option<f64>,
    /// Maximum iterations.
    pub maxiter: usize,
}

impl Default for BracketOptions {
    fn default() -> Self {
        Self {
            xr0: None,
            xmin: None,
            xmax: None,
            factor: None,
            maxiter: 1000,
        }
    }
}

/// One side of the two-sided search.
#[derive(Debug, Clone, Copy)]
struct Side {
    /// Moving end.
    x: f64,
    /// Value at the moving end.
    f: f64,
    /// Previous moving end — the other half of a bracket once the sign flips.
    x_last: f64,
    /// Value at `x_last`.
    f_last: f64,
    /// Fixed end: the OTHER side's starting point, used to grow an unlimited search.
    x0: f64,
    /// Distance, grown or shrunk each iteration depending on whether a limit is set.
    d: f64,
    /// The limit this side approaches, or infinite when unbounded.
    limit: f64,
}

impl Side {
    /// Advance the moving end one step, exactly as SciPy's `pre_func_eval` does.
    fn step(&mut self, factor: f64) -> f64 {
        if self.limit.is_infinite() {
            // Unlimited: walk away from the fixed end, geometrically faster each time.
            self.d *= factor;
            self.x0 + self.d
        } else {
            // Limited: close the remaining gap to the limit by `factor` each time, so the
            // search densifies toward the limit and never crosses it.
            self.d /= factor;
            self.limit - self.d
        }
    }

    /// Does the last step straddle a root? An exact zero at either end counts, matching
    /// SciPy's `(sf_last == -sf) | (sf_last == 0) | (sf == 0)`.
    fn brackets(&self) -> bool {
        let sf = self.f.signum();
        let sf_last = self.f_last.signum();
        (self.f == 0.0) || (self.f_last == 0.0) || (sf_last == -sf)
    }

    /// The bracket this side found, ordered ascending.
    fn as_bracket(&self) -> ((f64, f64), (f64, f64)) {
        if self.x_last <= self.x {
            ((self.x_last, self.x), (self.f_last, self.f))
        } else {
            ((self.x, self.x_last), (self.f, self.f_last))
        }
    }
}

/// Search outward from an initial interval until `f` changes sign.
///
/// Guaranteed to find a bracket when `f` is monotonic; it often finds one otherwise. Returns
/// `success: false` (rather than an error) when `maxiter` is exhausted without a sign change,
/// because "no bracket here" is an ordinary answer for a function that never crosses zero.
///
/// # Errors
///
/// Returns [`OptError::InvalidArgument`] if the starting points or limits are not finite and
/// ordered, if `factor <= 1`, or if `f` returns a NaN — a NaN cannot be given a sign, so the
/// sign-change test that drives the search would silently never fire.
pub fn bracket_root<F>(f: F, xl0: f64, options: BracketOptions) -> Result<BracketResult, OptError>
where
    F: Fn(f64) -> f64,
{
    let xr0 = options.xr0.unwrap_or(xl0 + 1.0);
    let factor = options.factor.unwrap_or(2.0);
    let xmin = options.xmin.unwrap_or(f64::NEG_INFINITY);
    let xmax = options.xmax.unwrap_or(f64::INFINITY);

    if !xl0.is_finite() || !xr0.is_finite() {
        return Err(OptError::InvalidArgument {
            detail: "bracket_root requires finite xl0 and xr0".to_string(),
        });
    }
    if xl0 >= xr0 {
        return Err(OptError::InvalidArgument {
            detail: format!("bracket_root requires xl0 < xr0, got {xl0} and {xr0}"),
        });
    }
    // `is_finite` FIRST, so a NaN factor is rejected there rather than reaching `<=`, which a
    // NaN would pass silently.
    if !factor.is_finite() || factor <= 1.0 {
        return Err(OptError::InvalidArgument {
            detail: format!("bracket_root requires a finite factor > 1, got {factor}"),
        });
    }
    if xmin > xl0 || xmax < xr0 {
        return Err(OptError::InvalidArgument {
            detail: "bracket_root requires xmin <= xl0 < xr0 <= xmax".to_string(),
        });
    }

    let evaluate = |x: f64| -> Result<f64, OptError> {
        let value = f(x);
        if value.is_nan() {
            return Err(OptError::InvalidArgument {
                detail: format!("bracket_root objective returned NaN at x = {x}"),
            });
        }
        Ok(value)
    };

    let fl0 = evaluate(xl0)?;
    let fr0 = evaluate(xr0)?;
    let mut nfev = 2usize;

    // The initial pair may already bracket, in which case SciPy reports nit = 0.
    let initial = Side {
        x: xr0,
        f: fr0,
        x_last: xl0,
        f_last: fl0,
        x0: xl0,
        d: xr0 - xl0,
        limit: xmax,
    };
    if initial.brackets() {
        let (bracket, f_bracket) = initial.as_bracket();
        return Ok(BracketResult {
            bracket,
            f_bracket,
            nit: 0,
            nfev,
            success: true,
        });
    }

    // Each side's FIXED end is the other side's start, and its initial distance is measured
    // from whichever anchor that side grows or shrinks against.
    let mut left = Side {
        x: xl0,
        f: fl0,
        x_last: xr0,
        f_last: fr0,
        x0: xr0,
        d: if xmin.is_infinite() {
            xl0 - xr0
        } else {
            xmin - xl0
        },
        limit: xmin,
    };
    let mut right = Side {
        x: xr0,
        f: fr0,
        x_last: xl0,
        f_last: fl0,
        x0: xl0,
        d: if xmax.is_infinite() {
            xr0 - xl0
        } else {
            xmax - xr0
        },
        limit: xmax,
    };

    for nit in 1..=options.maxiter {
        // LEFT FIRST, then right. The order is observable: when both sides would bracket on
        // the same iteration it decides which bracket is returned, so it matches SciPy's
        // interleave rather than being chosen for convenience.
        for side in [&mut left, &mut right] {
            let x = side.step(factor);
            let value = evaluate(x)?;
            nfev += 1;
            side.x_last = side.x;
            side.f_last = side.f;
            side.x = x;
            side.f = value;
            if side.brackets() {
                let (bracket, f_bracket) = side.as_bracket();
                return Ok(BracketResult {
                    bracket,
                    f_bracket,
                    nit,
                    nfev,
                    success: true,
                });
            }
        }
    }

    // No sign change within the budget. Report the widest interval actually explored so the
    // caller can see where the search looked rather than only that it failed.
    Ok(BracketResult {
        bracket: (left.x, right.x),
        f_bracket: (left.f, right.f),
        nit: options.maxiter,
        nfev,
        success: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Record every abscissa the search visits, so the SAMPLING can be compared with SciPy's
    /// and not just the answer. Two implementations can agree on the returned bracket while
    /// walking completely different points; the schedule is the part that has to match.
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
    fn bracket_root_matches_live_scipy_unlimited() {
        // Live `scipy.optimize.elementwise.bracket_root(lambda x: x - 100.0, 1.0)` on
        // scipy 1.17.1 returns bracket (65, 129), nit=7, nfev=16, and samples
        // 1, 2, 0, 3, -2, 5, -6, 9, -14, 17, -30, 33, -62, 65.
        let visited = std::cell::RefCell::new(Vec::new());
        let result = bracket_root(
            traced(|x| x - 100.0, &visited),
            1.0,
            BracketOptions::default(),
        )
        .expect("bracket search succeeds");

        assert!(result.success);
        assert_eq!(result.bracket, (65.0, 129.0));
        assert_eq!(result.nit, 7);
        assert_eq!(result.nfev, 16);
        assert_eq!(
            &visited.borrow()[..14],
            &[
                1.0, 2.0, 0.0, 3.0, -2.0, 5.0, -6.0, 9.0, -14.0, 17.0, -30.0, 33.0, -62.0, 65.0
            ],
            "the expansion schedule must match SciPy's, not merely the returned bracket"
        );
    }

    #[test]
    fn bracket_root_matches_live_scipy_from_zero() {
        // Live scipy: bracket_root(lambda x: x + 50.0, 0.0) -> (-63, -31), nit=6, nfev=14,
        // sampling 0, 1, -1, 2, -3, 4, -7, 8, -15, 16, -31, 32, -63, 64.
        let visited = std::cell::RefCell::new(Vec::new());
        let result = bracket_root(
            traced(|x| x + 50.0, &visited),
            0.0,
            BracketOptions::default(),
        )
        .expect("bracket search succeeds");

        assert_eq!(result.bracket, (-63.0, -31.0));
        assert_eq!(result.nit, 6);
        // DIVERGENCE 1, deliberate: scipy reports nfev = 14 because its vectorized step
        // evaluates the rightward point (64.0) in the same call even though the leftward
        // search bracketed at -63.0 first. We skip that evaluation, so we report 13 and never
        // sample 64.0. The bracket is identical.
        assert_eq!(result.nfev, 13);
        assert_eq!(
            &visited.borrow()[..],
            &[
                0.0, 1.0, -1.0, 2.0, -3.0, 4.0, -7.0, 8.0, -15.0, 16.0, -31.0, 32.0, -63.0
            ],
            "every point up to the one that brackets must match scipy's schedule exactly"
        );
    }

    #[test]
    fn bracket_root_matches_live_scipy_with_a_lower_limit() {
        // THE LIMITED BRANCH, which is a different schedule entirely: the side with a limit
        // CONTRACTS toward it instead of expanding away. Live scipy with xmin=0 and a root at
        // 0.25 returns (0.25, 0.5), nit=2, nfev=6, sampling 1, 2, 0.5, 3, 0.25, 5.
        let visited = std::cell::RefCell::new(Vec::new());
        let result = bracket_root(
            traced(|x| x - 0.25, &visited),
            1.0,
            BracketOptions {
                xmin: Some(0.0),
                ..BracketOptions::default()
            },
        )
        .expect("bracket search succeeds");

        assert_eq!(result.bracket, (0.25, 0.5));
        assert_eq!(result.nit, 2);
        // Same deliberate divergence as above: scipy's lockstep also samples 5.0 and reports
        // nfev = 6; the leftward search had already bracketed at 0.25.
        assert_eq!(result.nfev, 5);
        assert_eq!(
            &visited.borrow()[..],
            &[1.0, 2.0, 0.5, 3.0, 0.25],
            "a limited side must contract toward its limit, not expand away from it"
        );
    }

    #[test]
    fn bracket_root_reports_an_immediate_bracket_without_iterating() {
        // scipy: bracket_root(lambda x: x**2 - 4, 1.0) -> bracket (1, 2), nit=0, nfev=2,
        // because f(2) is exactly zero and an exact zero counts as a bracket.
        let result = bracket_root(|x| x * x - 4.0, 1.0, BracketOptions::default())
            .expect("bracket search succeeds");
        assert_eq!(result.bracket, (1.0, 2.0));
        assert_eq!(result.f_bracket, (-3.0, 0.0));
        assert_eq!(result.nit, 0);
        assert_eq!(result.nfev, 2);
    }

    #[test]
    fn bracket_root_returns_a_true_bracket_where_the_incumbent_does_not() {
        // DIVERGENCE 2, deliberate, and the incumbent is the one that is wrong. Live scipy
        // 1.17.1 on this exact call returns bracket (7.0, -7.0) with f_bracket (-51.0, -51.0)
        // and success = True: when both searches bracket on the same iteration it takes xl
        // from the RIGHTWARD search and xr from the LEFTWARD one, yielding two same-signed
        // endpoints that are not a bracket. Reproducing that would hand brentq an input it
        // must reject.
        let result = bracket_root(
            |x| x * x - 100.0,
            -1.0,
            BracketOptions {
                xr0: Some(1.0),
                ..BracketOptions::default()
            },
        )
        .expect("bracket search succeeds");

        assert!(result.success);
        assert_eq!(result.nit, 3);
        assert_eq!(
            result.bracket,
            (-15.0, -7.0),
            "the leftward search is checked first, so its bracket is the one reported"
        );

        // The invariant that makes `success` worth trusting, stated directly rather than
        // implied by the endpoints: a successful search really did straddle a sign change.
        let (fl, fr) = result.f_bracket;
        assert!(
            fl == 0.0 || fr == 0.0 || fl.signum() != fr.signum(),
            "success must mean an actual sign change, got f_bracket = ({fl}, {fr})"
        );

        // And it is directly usable, which is the whole point of not copying the defect.
        use crate::root::brentq;
        use crate::types::RootOptions;
        let root = brentq(|x| x * x - 100.0, result.bracket, RootOptions::default())
            .expect("the returned bracket is accepted by a bracketing solver");
        assert!((root.root + 10.0).abs() < 1e-12, "got {}", root.root);
    }

    #[test]
    fn bracket_root_feeds_the_existing_solvers() {
        // The point of the function: its output is directly consumable by the bracketing
        // solvers this crate already has, which is what made its absence a real gap.
        use crate::root::brentq;
        use crate::types::RootOptions;

        let found =
            bracket_root(|x| x * x * x - 2.0, 0.0, BracketOptions::default()).expect("bracket");
        assert!(found.success);
        let root = brentq(|x| x * x * x - 2.0, found.bracket, RootOptions::default())
            .expect("brentq converges on the discovered bracket");
        assert!(
            (root.root - 2.0_f64.cbrt()).abs() < 1e-12,
            "got {}",
            root.root
        );
    }

    #[test]
    fn bracket_root_reports_failure_rather_than_erroring_when_no_root_exists() {
        // A function that never crosses zero is not a caller mistake; it is an answer.
        let result = bracket_root(
            |x| x * x + 1.0,
            0.0,
            BracketOptions {
                maxiter: 12,
                ..BracketOptions::default()
            },
        )
        .expect("exhausting maxiter is not an error");
        assert!(!result.success);
        assert_eq!(result.nit, 12);
    }

    #[test]
    fn bracket_root_rejects_inputs_it_cannot_search() {
        assert!(
            bracket_root(
                |x| x,
                1.0,
                BracketOptions {
                    xr0: Some(0.5),
                    ..BracketOptions::default()
                }
            )
            .is_err()
        );
        assert!(
            bracket_root(
                |x| x,
                1.0,
                BracketOptions {
                    factor: Some(1.0),
                    ..BracketOptions::default()
                }
            )
            .is_err()
        );
        assert!(
            bracket_root(
                |x| x,
                1.0,
                BracketOptions {
                    xmin: Some(2.0),
                    ..BracketOptions::default()
                }
            )
            .is_err()
        );
        // A NaN objective cannot be given a sign, so the sign-change test would never fire
        // and the search would run to maxiter reporting nothing useful.
        assert!(bracket_root(|_| f64::NAN, 1.0, BracketOptions::default()).is_err());
    }
}
