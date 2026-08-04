#![forbid(unsafe_code)]

use crate::types::{ConvergenceStatus, OptError, RootMethod, RootOptions};

#[derive(Debug, Clone, PartialEq)]
pub struct RootResult {
    pub root: f64,
    pub converged: bool,
    pub flag: String,
    pub status: ConvergenceStatus,
    pub iterations: usize,
    pub function_calls: usize,
    pub method: RootMethod,
    pub message: String,
}

/// SciPy-compatible public spelling for scalar root-finding results.
///
/// FrankenSciPy historically exposed the singular [`RootResult`]. SciPy calls
/// the same result record `RootResults`; keeping this alias makes both spellings
/// name the result returned by every scalar root finder.
pub type RootResults = RootResult;

impl RootResult {
    #[must_use]
    pub fn terminal(
        method: RootMethod,
        root: f64,
        converged: bool,
        status: ConvergenceStatus,
        iterations: usize,
        function_calls: usize,
        message: impl Into<String>,
    ) -> Self {
        Self {
            root,
            converged,
            flag: scipy_root_flag(status).to_string(),
            status,
            iterations,
            function_calls,
            method,
            message: message.into(),
        }
    }
}

const fn scipy_root_flag(status: ConvergenceStatus) -> &'static str {
    match status {
        ConvergenceStatus::Success => "converged",
        ConvergenceStatus::MaxIterations
        | ConvergenceStatus::MaxEvaluations
        | ConvergenceStatus::PrecisionLoss
        | ConvergenceStatus::CallbackStop => "convergence error",
        ConvergenceStatus::NanEncountered
        | ConvergenceStatus::OutOfBounds
        | ConvergenceStatus::NotImplemented
        | ConvergenceStatus::InvalidInput => "value error",
    }
}

pub fn root_scalar<F>(
    f: F,
    bracket: Option<(f64, f64)>,
    x0: Option<f64>,
    x1: Option<f64>,
    options: RootOptions,
) -> Result<RootResult, OptError>
where
    F: Fn(f64) -> f64,
{
    validate_root_options(options)?;

    let method = if let Some(method) = options.method {
        method
    } else if bracket.is_some() {
        RootMethod::Brentq
    } else if x0.is_some() {
        RootMethod::Secant
    } else {
        return Err(OptError::InvalidArgument {
            detail: String::from("unable to select a root solver: provide bracket or x0"),
        });
    };

    match method {
        RootMethod::Brentq => {
            let interval = require_bracket(bracket)?;
            brentq(&f, interval, options)
        }
        RootMethod::Brenth => {
            let interval = require_bracket(bracket)?;
            brenth(&f, interval, options)
        }
        RootMethod::Bisect => {
            let interval = require_bracket(bracket)?;
            bisect(&f, interval, options)
        }
        RootMethod::Ridder => {
            let interval = require_bracket(bracket)?;
            ridder(&f, interval, options)
        }
        RootMethod::Toms748 => {
            let interval = require_bracket(bracket)?;
            toms748(&f, interval, options)
        }
        RootMethod::Secant => {
            let start = x0.ok_or_else(|| OptError::InvalidArgument {
                detail: "secant method requires x0".to_string(),
            })?;
            secant(&f, start, x1, options)
        }
        RootMethod::Newton | RootMethod::Halley => Err(OptError::InvalidArgument {
            detail: format!("{method:?} requires fprime; use newton_scalar() or halley() directly"),
        }),
    }
}

pub fn brentq<F>(f: F, bracket: (f64, f64), options: RootOptions) -> Result<RootResult, OptError>
where
    F: Fn(f64) -> f64,
{
    validate_root_options(options)?;
    validate_bracket_finite(bracket)?;
    let mut eval = RootEvaluator::new(&f, options);
    let mut state = prepare_bracket(&mut eval, bracket, RootMethod::Brentq)?;
    if state.fa == 0.0 {
        return Ok(RootResult::terminal(
            RootMethod::Brentq,
            state.a,
            true,
            ConvergenceStatus::Success,
            0,
            eval.function_calls,
            "brentq accepted bracket start as root",
        ));
    }
    if state.fb == 0.0 {
        return Ok(RootResult::terminal(
            RootMethod::Brentq,
            state.b,
            true,
            ConvergenceStatus::Success,
            0,
            eval.function_calls,
            "brentq accepted bracket end as root",
        ));
    }

    let mut c = state.a;
    let mut fc = state.fa;
    let mut d = state.b - state.a;
    let mut e = d;

    for iter in 1..=options.maxiter {
        if state.fb == 0.0 {
            return Ok(RootResult::terminal(
                RootMethod::Brentq,
                state.b,
                true,
                ConvergenceStatus::Success,
                iter,
                eval.function_calls,
                "brentq converged (f(x) == 0)",
            ));
        }

        if same_sign(state.fa, state.fb) {
            state.a = c;
            state.fa = fc;
            d = state.b - state.a;
            e = d;
        }

        if state.fa.abs() < state.fb.abs() {
            c = state.b;
            fc = state.fb;
            state.b = state.a;
            state.fb = state.fa;
            state.a = c;
            state.fa = fc;
        }

        let tol = options.xtol + options.rtol * state.b.abs();
        let m = 0.5 * (state.a - state.b);
        if m.abs() <= tol {
            return Ok(RootResult::terminal(
                RootMethod::Brentq,
                state.b,
                true,
                ConvergenceStatus::Success,
                iter,
                eval.function_calls,
                "brentq converged (interval width <= tol)",
            ));
        }

        if e.abs() >= tol && fc.abs() > state.fb.abs() {
            let s = state.fb / fc;
            let (mut p, mut q) = if state.a == c {
                (2.0 * m * s, 1.0 - s)
            } else {
                let q_val = fc / state.fa;
                let r_val = state.fb / state.fa;
                (
                    s * (2.0 * m * q_val * (q_val - r_val) - (state.b - c) * (r_val - 1.0)),
                    (q_val - 1.0) * (r_val - 1.0) * (s - 1.0),
                )
            };
            if p > 0.0 {
                q = -q;
            }
            p = p.abs();
            let min1 = 3.0 * m * q.abs() - (tol * q).abs();
            let min2 = (e * q).abs();
            if 2.0 * p < min1.min(min2) {
                e = d;
                d = p / q;
            } else {
                d = m;
                e = d;
            }
        } else {
            d = m;
            e = d;
        }

        c = state.b;
        fc = state.fb;
        if d.abs() > tol {
            state.b += d;
        } else if m > 0.0 {
            state.b += tol;
        } else {
            state.b -= tol;
        }
        state.fb = eval.evaluate(state.b)?;
    }

    Ok(RootResult::terminal(
        RootMethod::Brentq,
        state.b,
        false,
        ConvergenceStatus::MaxIterations,
        options.maxiter,
        eval.function_calls,
        "brentq failed to converge within maxiter",
    ))
}

/// Batched 1-D root finding: solve `f(x, params) = 0` for MANY parameter sets over a shared
/// `bracket`, one [`RootResult`] per set. This is the vmap-over-solver primitive for scalar
/// roots — a 1-D root SWEEP (implied volatility per option, percentile/quantile inversion per
/// channel, threshold calibration per series) loops `brentq` in Python, N solves SERIALLY. fsci
/// `brentq_many` (param-sweep `F: Fn(f64 x, &[f64] params)->f64`) fans the N independent Brent
/// solves across cores and inlines the function. Result `i` is byte-identical to
/// `brentq(|x| f(x, &param_rows[i]), bracket, options)`. Like `root_many`, the per-solve win is
/// parallelism-dominated (Brent is cheap), so expect a modest but real speedup over looped scipy.
pub fn brentq_many<F>(
    f: F,
    bracket: (f64, f64),
    param_rows: &[Vec<f64>],
    options: RootOptions,
) -> Vec<Result<RootResult, OptError>>
where
    F: Fn(f64, &[f64]) -> f64 + Sync,
{
    let nrows = param_rows.len();
    if nrows == 0 {
        return Vec::new();
    }
    let f_ref = &f;
    let solve_one = move |params: &[f64]| brentq(|x| f_ref(x, params), bracket, options);

    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(nrows);
    if nthreads <= 1 || nrows < 4 {
        return param_rows.iter().map(|p| solve_one(p)).collect();
    }

    let chunk = nrows.div_ceil(nthreads);
    let solve_one = &solve_one;
    let chunk_results: Vec<Vec<Result<RootResult, OptError>>> = std::thread::scope(|scope| {
        (0..nthreads)
            .filter_map(|t| {
                let lo = t * chunk;
                if lo >= nrows {
                    return None;
                }
                let hi = (lo + chunk).min(nrows);
                Some(scope.spawn(move || {
                    (lo..hi)
                        .map(|i| solve_one(&param_rows[i]))
                        .collect::<Vec<_>>()
                }))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("brentq_many worker panicked"))
            .collect()
    });

    let mut out = Vec::with_capacity(nrows);
    for cr in chunk_results {
        out.extend(cr);
    }
    out
}

/// Vectorised Newton's-method root find over `n` independent parameter rows —
/// the vmap-over-solver sibling of [`brentq_many`] for [`newton`] with an
/// analytic derivative. `func(x, params)` and `fprime(x, params)` are inlined
/// Rust closures (no per-iteration Python callback, unlike a looped
/// `scipy.optimize.newton`), and the independent solves fan across threads.
/// Entry `k` equals `newton(|x| func(x, &param_rows[k]), x0, Some(fprime …), …)`.
pub fn newton_many<F, G>(
    func: F,
    fprime: G,
    x0: f64,
    param_rows: &[Vec<f64>],
    tol: f64,
    rtol: f64,
    maxiter: usize,
) -> Vec<Result<RootResult, OptError>>
where
    F: Fn(f64, &[f64]) -> f64 + Sync,
    G: Fn(f64, &[f64]) -> f64 + Sync,
{
    let nrows = param_rows.len();
    if nrows == 0 {
        return Vec::new();
    }
    let f_ref = &func;
    let fp_ref = &fprime;
    let solve_one = move |params: &[f64]| -> Result<RootResult, OptError> {
        let fp = |x: f64| fp_ref(x, params);
        newton(
            |x| f_ref(x, params),
            x0,
            Some(&fp),
            None,
            None,
            tol,
            rtol,
            maxiter,
        )
    };

    // A scalar Newton solve with inlined Rust closures is very cheap (~0.1µs for a
    // simple `func`), so the WIN here is eliminating SciPy's per-solve Python
    // overhead (~69µs/solve) — a serial Rust loop already beats the scipy loop
    // ~700x. Threading such tiny work over-subscribes (64 threads' spawn dwarfs the
    // solves), so cap workers by WORK: each thread takes >=2048 solves, keeping
    // typical batches serial and only very large batches (or an expensive `func`)
    // fanning out. Order-preserving ⇒ bit-identical to the serial map.
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(nrows / 2048).max(1);
    if nthreads <= 1 {
        return param_rows.iter().map(|p| solve_one(p)).collect();
    }

    let chunk = nrows.div_ceil(nthreads);
    let solve_one = &solve_one;
    let chunk_results: Vec<Vec<Result<RootResult, OptError>>> = std::thread::scope(|scope| {
        (0..nthreads)
            .filter_map(|t| {
                let lo = t * chunk;
                if lo >= nrows {
                    return None;
                }
                let hi = (lo + chunk).min(nrows);
                Some(scope.spawn(move || {
                    (lo..hi)
                        .map(|i| solve_one(&param_rows[i]))
                        .collect::<Vec<_>>()
                }))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("newton_many worker panicked"))
            .collect()
    });

    let mut out = Vec::with_capacity(nrows);
    for cr in chunk_results {
        out.extend(cr);
    }
    out
}

/// Vectorised derivative-free [`secant`] root find over `n` parameter rows — the
/// sibling of [`newton_many`] for `scipy.optimize.newton` WITHOUT `fprime` (secant
/// method). `func(x, params)` is an inlined Rust closure (no per-iteration Python
/// callback). Work-capped like [`newton_many`] (cheap scalar solve ⇒ the win is
/// Python-overhead elimination, so typical batches stay serial). Entry `k` equals
/// `secant(|x| func(x, &param_rows[k]), x0, x1, options)`.
pub fn secant_many<F>(
    func: F,
    x0: f64,
    x1: Option<f64>,
    param_rows: &[Vec<f64>],
    options: RootOptions,
) -> Vec<Result<RootResult, OptError>>
where
    F: Fn(f64, &[f64]) -> f64 + Sync,
{
    let nrows = param_rows.len();
    if nrows == 0 {
        return Vec::new();
    }
    let f_ref = &func;
    let solve_one = move |params: &[f64]| secant(|x| f_ref(x, params), x0, x1, options);

    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(nrows / 2048).max(1);
    if nthreads <= 1 {
        return param_rows.iter().map(|p| solve_one(p)).collect();
    }
    let chunk = nrows.div_ceil(nthreads);
    let solve_one = &solve_one;
    let chunk_results: Vec<Vec<Result<RootResult, OptError>>> = std::thread::scope(|scope| {
        (0..nthreads)
            .filter_map(|t| {
                let lo = t * chunk;
                if lo >= nrows {
                    return None;
                }
                let hi = (lo + chunk).min(nrows);
                Some(scope.spawn(move || {
                    (lo..hi)
                        .map(|i| solve_one(&param_rows[i]))
                        .collect::<Vec<_>>()
                }))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("secant_many worker panicked"))
            .collect()
    });
    let mut out = Vec::with_capacity(nrows);
    for cr in chunk_results {
        out.extend(cr);
    }
    out
}

/// Root finder dispatched as `brenth`.
///
/// # Implementation note
/// This is currently an ALIAS for [`brentq`]; it runs Brent's method with
/// inverse-quadratic interpolation and relabels the output. SciPy's
/// `brenth` uses hyperbolic interpolation (Harris 1970) which has
/// different convergence dynamics on functions with near-horizontal
/// tangents. A caller selecting `brenth` for its documented Brent-Harris
/// convergence behavior will observe `brentq`-equivalent iteration counts
/// and numerical traces here. Tracked by frankenscipy-88gz; the algorithm
/// swap-in will change this docstring.
pub fn brenth<F>(f: F, bracket: (f64, f64), options: RootOptions) -> Result<RootResult, OptError>
where
    F: Fn(f64) -> f64,
{
    let mut result = brentq(f, bracket, options)?;
    result.method = RootMethod::Brenth;
    result.message = result.message.replace("brentq", "brenth");
    Ok(result)
}

pub fn bisect<F>(f: F, bracket: (f64, f64), options: RootOptions) -> Result<RootResult, OptError>
where
    F: Fn(f64) -> f64,
{
    validate_root_options(options)?;
    validate_bracket_finite(bracket)?;
    let mut eval = RootEvaluator::new(&f, options);
    let mut state = prepare_bracket(&mut eval, bracket, RootMethod::Bisect)?;
    if state.fa == 0.0 {
        return Ok(RootResult::terminal(
            RootMethod::Bisect,
            state.a,
            true,
            ConvergenceStatus::Success,
            0,
            eval.function_calls,
            "bisection accepted bracket start as root",
        ));
    }
    if state.fb == 0.0 {
        return Ok(RootResult::terminal(
            RootMethod::Bisect,
            state.b,
            true,
            ConvergenceStatus::Success,
            0,
            eval.function_calls,
            "bisection accepted bracket end as root",
        ));
    }

    for iter in 1..=options.maxiter {
        let mid = 0.5 * (state.a + state.b);
        let f_mid = eval.evaluate(mid)?;
        let interval_tol = options.xtol + options.rtol * mid.abs();
        if f_mid == 0.0 || (state.b - state.a).abs() <= interval_tol {
            return Ok(RootResult::terminal(
                RootMethod::Bisect,
                mid,
                true,
                ConvergenceStatus::Success,
                iter,
                eval.function_calls,
                "bisection converged",
            ));
        }
        if same_sign(state.fa, f_mid) {
            state.a = mid;
            state.fa = f_mid;
        } else {
            state.b = mid;
            state.fb = f_mid;
        }
    }

    let root = 0.5 * (state.a + state.b);
    Ok(RootResult::terminal(
        RootMethod::Bisect,
        root,
        false,
        ConvergenceStatus::MaxIterations,
        options.maxiter,
        eval.function_calls,
        "bisection failed to converge within maxiter",
    ))
}

pub fn ridder<F>(f: F, bracket: (f64, f64), options: RootOptions) -> Result<RootResult, OptError>
where
    F: Fn(f64) -> f64,
{
    validate_root_options(options)?;
    validate_bracket_finite(bracket)?;
    let mut eval = RootEvaluator::new(&f, options);
    let mut state = prepare_bracket(&mut eval, bracket, RootMethod::Ridder)?;
    if state.fa == 0.0 {
        return Ok(RootResult::terminal(
            RootMethod::Ridder,
            state.a,
            true,
            ConvergenceStatus::Success,
            0,
            eval.function_calls,
            "ridder accepted bracket start as root",
        ));
    }
    if state.fb == 0.0 {
        return Ok(RootResult::terminal(
            RootMethod::Ridder,
            state.b,
            true,
            ConvergenceStatus::Success,
            0,
            eval.function_calls,
            "ridder accepted bracket end as root",
        ));
    }

    for iter in 1..=options.maxiter {
        let mid = 0.5 * (state.a + state.b);
        let f_mid = eval.evaluate(mid)?;
        if f_mid == 0.0 {
            return Ok(RootResult::terminal(
                RootMethod::Ridder,
                mid,
                true,
                ConvergenceStatus::Success,
                iter,
                eval.function_calls,
                "ridder converged (f(mid) == 0)",
            ));
        }
        let disc = f_mid * f_mid - state.fa * state.fb;
        if disc <= 0.0 {
            return Ok(RootResult::terminal(
                RootMethod::Ridder,
                mid,
                false,
                ConvergenceStatus::PrecisionLoss,
                iter,
                eval.function_calls,
                "ridder encountered non-positive discriminant",
            ));
        }
        let s = disc.sqrt();
        let sign = if (state.fa - state.fb) < 0.0 {
            -1.0
        } else {
            1.0
        };
        let x_new = mid + (mid - state.a) * sign * f_mid / s;
        let f_new = eval.evaluate(x_new)?;

        let interval_tol = options.xtol + options.rtol * x_new.abs();
        if f_new == 0.0 || (state.b - state.a).abs() <= interval_tol {
            return Ok(RootResult::terminal(
                RootMethod::Ridder,
                x_new,
                true,
                ConvergenceStatus::Success,
                iter,
                eval.function_calls,
                "ridder converged",
            ));
        }

        if !same_sign(f_mid, f_new) {
            state.a = mid;
            state.fa = f_mid;
            state.b = x_new;
            state.fb = f_new;
        } else if !same_sign(state.fa, f_new) {
            state.b = x_new;
            state.fb = f_new;
        } else {
            state.a = x_new;
            state.fa = f_new;
        }
    }

    let root = 0.5 * (state.a + state.b);
    Ok(RootResult::terminal(
        RootMethod::Ridder,
        root,
        false,
        ConvergenceStatus::MaxIterations,
        options.maxiter,
        eval.function_calls,
        "ridder failed to converge within maxiter",
    ))
}

/// Root finder dispatched as `toms748`.
///
/// # Implementation note
/// This currently implements an **Illinois-modified Regula Falsi**
/// (secant-interpolant with a mid-50% fallback to bisection), NOT the
/// TOMS Algorithm 748 of Alefeld-Potra-Shi 1995 whose cubic inverse
/// interpolation yields an asymptotic convergence rate of
/// `4^(1/3) ≈ 1.587`. Regula-Falsi convergence here is golden-ratio
/// (`≈ 1.618`). A scipy-migrating caller selecting `toms748` for the
/// AP-Shi convergence proof or the double-length interval taking
/// technique will observe different iteration dynamics here. Tracked
/// by frankenscipy-88gz; the algorithm swap-in will change this
/// docstring.
pub fn toms748<F>(f: F, bracket: (f64, f64), options: RootOptions) -> Result<RootResult, OptError>
where
    F: Fn(f64) -> f64,
{
    validate_root_options(options)?;
    validate_bracket_finite(bracket)?;
    let (mut a, mut b) = bracket;
    let mut eval = RootEvaluator::new(&f, options);
    let mut fa = eval.evaluate(a)?;
    let mut fb = eval.evaluate(b)?;

    if fa * fb > 0.0 {
        return Err(OptError::SignChangeRequired {
            detail: format!("f(a)={fa} and f(b)={fb} must have opposite signs"),
        });
    }

    if fa == 0.0 {
        return Ok(RootResult::terminal(
            RootMethod::Toms748,
            a,
            true,
            ConvergenceStatus::Success,
            0,
            eval.function_calls,
            "exact root at a",
        ));
    }
    if fb == 0.0 {
        return Ok(RootResult::terminal(
            RootMethod::Toms748,
            b,
            true,
            ConvergenceStatus::Success,
            0,
            eval.function_calls,
            "exact root at b",
        ));
    }

    if fa > 0.0 {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    for iter in 0..options.maxiter {
        let tol = options.xtol + options.rtol * b.abs();

        if (b - a).abs() < tol || fa == 0.0 || fb == 0.0 {
            let root = if fa.abs() < fb.abs() { a } else { b };
            return Ok(RootResult::terminal(
                RootMethod::Toms748,
                root,
                true,
                ConvergenceStatus::Success,
                iter,
                eval.function_calls,
                "converged",
            ));
        }

        let mid = 0.5 * (a + b);
        let c = if fa != fb {
            let s = a - fa * (b - a) / (fb - fa);
            if s > a + 0.25 * (b - a) && s < b - 0.25 * (b - a) {
                s
            } else {
                mid
            }
        } else {
            mid
        };

        let fc = eval.evaluate(c)?;
        if fc == 0.0 {
            return Ok(RootResult::terminal(
                RootMethod::Toms748,
                c,
                true,
                ConvergenceStatus::Success,
                iter,
                eval.function_calls,
                "exact root found",
            ));
        }

        if fc < 0.0 {
            a = c;
            fa = fc;
        } else {
            b = c;
            fb = fc;
        }
    }

    let root = 0.5 * (a + b);
    Ok(RootResult::terminal(
        RootMethod::Toms748,
        root,
        false,
        ConvergenceStatus::MaxIterations,
        options.maxiter,
        eval.function_calls,
        "toms748 failed to converge within maxiter",
    ))
}

/// Newton's method for scalar root finding (requires derivative).
///
/// Matches `scipy.optimize.newton` with fprime provided.
pub fn newton_scalar<F, G>(
    f: F,
    fprime: G,
    x0: f64,
    options: RootOptions,
) -> Result<RootResult, OptError>
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    validate_root_options(options)?;
    if !x0.is_finite() {
        return Err(OptError::NonFiniteInput {
            detail: String::from("newton_scalar: x0 must be finite"),
        });
    }
    let mut x = x0;
    let mut nfev = 0usize;

    for iter in 0..options.maxiter {
        nfev += 1;
        let fx = f(x);
        if fx.abs() < options.xtol {
            return Ok(RootResult::terminal(
                RootMethod::Newton,
                x,
                true,
                ConvergenceStatus::Success,
                iter,
                nfev,
                "converged",
            ));
        }

        let dfx = fprime(x);
        if dfx == 0.0 {
            return Ok(RootResult::terminal(
                RootMethod::Newton,
                x,
                false,
                ConvergenceStatus::PrecisionLoss,
                iter,
                nfev,
                "zero derivative encountered",
            ));
        }

        let x_new = x - fx / dfx;
        if (x_new - x).abs() < options.xtol + options.rtol * x.abs() {
            return Ok(RootResult::terminal(
                RootMethod::Newton,
                x_new,
                true,
                ConvergenceStatus::Success,
                iter + 1,
                nfev,
                "converged",
            ));
        }
        x = x_new;
    }

    Ok(RootResult::terminal(
        RootMethod::Newton,
        x,
        false,
        ConvergenceStatus::MaxIterations,
        options.maxiter,
        nfev,
        "newton failed to converge",
    ))
}

/// Secant method for scalar root finding (derivative-free).
///
/// Matches `scipy.optimize.newton` without fprime.
pub fn secant<F>(
    f: F,
    x0: f64,
    x1: Option<f64>,
    options: RootOptions,
) -> Result<RootResult, OptError>
where
    F: Fn(f64) -> f64,
{
    validate_root_options(options)?;
    if !x0.is_finite() {
        return Err(OptError::NonFiniteInput {
            detail: String::from("secant: x0 must be finite"),
        });
    }
    if let Some(x1_val) = x1
        && !x1_val.is_finite()
    {
        return Err(OptError::NonFiniteInput {
            detail: String::from("secant: x1 must be finite"),
        });
    }
    let mut nfev = 0usize;
    let mut xprev = x0;
    nfev += 1;
    let mut fprev = f(xprev);
    let mut xcurr = x1.unwrap_or(x0 * (1.0 + 1e-4) + 1e-4);
    nfev += 1;
    let mut fcurr = f(xcurr);

    for iter in 0..options.maxiter {
        if fcurr.abs() < options.xtol {
            return Ok(RootResult::terminal(
                RootMethod::Secant,
                xcurr,
                true,
                ConvergenceStatus::Success,
                iter,
                nfev,
                "converged",
            ));
        }

        let denom = fcurr - fprev;
        if denom == 0.0 {
            return Ok(RootResult::terminal(
                RootMethod::Secant,
                xcurr,
                false,
                ConvergenceStatus::PrecisionLoss,
                iter,
                nfev,
                "identical function values",
            ));
        }

        let x_new = xcurr - fcurr * (xcurr - xprev) / denom;
        let step = (x_new - xcurr).abs();
        xprev = xcurr;
        fprev = fcurr;
        xcurr = x_new;
        nfev += 1;
        fcurr = f(xcurr);

        // Step-size convergence check
        if step < options.xtol + options.rtol * xcurr.abs() {
            return Ok(RootResult::terminal(
                RootMethod::Secant,
                xcurr,
                true,
                ConvergenceStatus::Success,
                iter + 1,
                nfev,
                "converged",
            ));
        }
    }

    Ok(RootResult::terminal(
        RootMethod::Secant,
        xcurr,
        false,
        ConvergenceStatus::MaxIterations,
        options.maxiter,
        nfev,
        "secant failed to converge",
    ))
}

/// Halley's method (requires f, f', f'').
///
/// Cubic convergence rate. Matches `scipy.optimize.newton` with fprime2.
pub fn halley<F, G, H>(
    f: F,
    fprime: G,
    fprime2: H,
    x0: f64,
    options: RootOptions,
) -> Result<RootResult, OptError>
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
    H: Fn(f64) -> f64,
{
    validate_root_options(options)?;
    if !x0.is_finite() {
        return Err(OptError::NonFiniteInput {
            detail: String::from("halley: x0 must be finite"),
        });
    }
    let mut x = x0;
    let mut nfev = 0usize;

    for iter in 0..options.maxiter {
        nfev += 1;
        let fx = f(x);
        if fx.abs() < options.xtol {
            return Ok(RootResult::terminal(
                RootMethod::Halley,
                x,
                true,
                ConvergenceStatus::Success,
                iter,
                nfev,
                "converged",
            ));
        }

        let dfx = fprime(x);
        let d2fx = fprime2(x);

        if dfx == 0.0 {
            return Ok(RootResult::terminal(
                RootMethod::Halley,
                x,
                false,
                ConvergenceStatus::PrecisionLoss,
                iter,
                nfev,
                "zero derivative encountered",
            ));
        }

        let denom = 2.0 * dfx * dfx - fx * d2fx;
        if denom.abs() < 1e-30 {
            // Halley denominator degenerate; fall back to Newton step
            let step = fx / dfx;
            x -= step;
            if step.abs() < options.xtol + options.rtol * x.abs() {
                return Ok(RootResult::terminal(
                    RootMethod::Halley,
                    x,
                    true,
                    ConvergenceStatus::Success,
                    iter + 1,
                    nfev,
                    "converged (Newton fallback)",
                ));
            }
        } else {
            let x_new = x - 2.0 * fx * dfx / denom;
            if (x_new - x).abs() < options.xtol + options.rtol * x.abs() {
                return Ok(RootResult::terminal(
                    RootMethod::Halley,
                    x_new,
                    true,
                    ConvergenceStatus::Success,
                    iter + 1,
                    nfev,
                    "converged",
                ));
            }
            x = x_new;
        }
    }

    Ok(RootResult::terminal(
        RootMethod::Halley,
        x,
        false,
        ConvergenceStatus::MaxIterations,
        options.maxiter,
        nfev,
        "halley failed to converge",
    ))
}

/// Unified scalar root finder, matching `scipy.optimize.newton(func, x0, ...)`.
///
/// Dispatches exactly as scipy does: with `fprime` it runs Newton-Raphson (and
/// Halley's method when `fprime2` is also given), otherwise the secant method
/// (starting from `x1` if provided, else `x0·(1+1e-4)` nudged away from zero).
/// Convergence uses scipy's `np.isclose(p, p_prev, rtol, atol=tol)` test, i.e.
/// `|p − p_prev| ≤ tol + rtol·|p_prev|`. Defaults: `tol = 1.48e-8`,
/// `rtol = 0.0`, `maxiter = 50`.
#[expect(clippy::too_many_arguments)]
pub fn newton<F>(
    func: F,
    x0: f64,
    fprime: Option<&dyn Fn(f64) -> f64>,
    fprime2: Option<&dyn Fn(f64) -> f64>,
    x1: Option<f64>,
    tol: f64,
    rtol: f64,
    maxiter: usize,
) -> Result<RootResult, OptError>
where
    F: Fn(f64) -> f64,
{
    if !tol.is_finite() || tol <= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("tol must be finite and > 0"),
        });
    }
    if !rtol.is_finite() || rtol < 0.0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("rtol must be finite and >= 0"),
        });
    }
    if maxiter < 1 {
        return Err(OptError::InvalidArgument {
            detail: String::from("maxiter must be greater than 0"),
        });
    }
    if !x0.is_finite() {
        return Err(OptError::NonFiniteInput {
            detail: String::from("newton: x0 must be finite"),
        });
    }
    if fprime.is_none() && fprime2.is_some() {
        return Err(OptError::InvalidArgument {
            detail: String::from("fprime2 requires fprime"),
        });
    }
    let close = |p: f64, prev: f64| (p - prev).abs() <= tol + rtol * prev.abs();
    let mut funcalls = 0usize;

    if let Some(fp) = fprime {
        // Newton-Raphson (and Halley when fprime2 is supplied).
        let mut p0 = x0;
        let mut method = RootMethod::Newton;
        for itr in 0..maxiter {
            let fval = func(p0);
            funcalls += 1;
            if fval == 0.0 {
                return Ok(RootResult::terminal(
                    method,
                    p0,
                    true,
                    ConvergenceStatus::Success,
                    itr,
                    funcalls,
                    "converged",
                ));
            }
            let fder = fp(p0);
            funcalls += 1;
            if fder == 0.0 {
                return Ok(RootResult::terminal(
                    method,
                    p0,
                    false,
                    ConvergenceStatus::PrecisionLoss,
                    itr + 1,
                    funcalls,
                    "Derivative was zero.",
                ));
            }
            let mut newton_step = fval / fder;
            if let Some(fp2) = fprime2 {
                let fder2 = fp2(p0);
                funcalls += 1;
                method = RootMethod::Halley;
                let adj = newton_step * fder2 / fder / 2.0;
                if adj.abs() < 1.0 {
                    newton_step /= 1.0 - adj;
                }
            }
            let p = p0 - newton_step;
            if close(p, p0) {
                return Ok(RootResult::terminal(
                    method,
                    p,
                    true,
                    ConvergenceStatus::Success,
                    itr + 1,
                    funcalls,
                    "converged",
                ));
            }
            p0 = p;
        }
        return Ok(RootResult::terminal(
            method,
            p0,
            false,
            ConvergenceStatus::MaxIterations,
            maxiter,
            funcalls,
            "Failed to converge",
        ));
    }

    // Secant method.
    let method = RootMethod::Secant;
    let mut p0 = x0;
    let mut p1 = match x1 {
        Some(x1v) => {
            if !x1v.is_finite() {
                return Err(OptError::NonFiniteInput {
                    detail: String::from("newton: x1 must be finite"),
                });
            }
            if x1v == x0 {
                return Err(OptError::InvalidArgument {
                    detail: String::from("x1 and x0 must be different"),
                });
            }
            x1v
        }
        None => {
            let eps = 1e-4;
            let mut p1 = x0 * (1.0 + eps);
            p1 += if p1 >= 0.0 { eps } else { -eps };
            p1
        }
    };
    let mut q0 = func(p0);
    funcalls += 1;
    let mut q1 = func(p1);
    funcalls += 1;
    if q1.abs() < q0.abs() {
        std::mem::swap(&mut p0, &mut p1);
        std::mem::swap(&mut q0, &mut q1);
    }
    let mut p = p1;
    for itr in 0..maxiter {
        if q1 == q0 {
            p = (p1 + p0) / 2.0;
            return Ok(RootResult::terminal(
                method,
                p,
                false,
                ConvergenceStatus::PrecisionLoss,
                itr + 1,
                funcalls,
                "Tolerance reached.",
            ));
        }
        if q1.abs() > q0.abs() {
            p = (-q0 / q1 * p1 + p0) / (1.0 - q0 / q1);
        } else {
            p = (-q1 / q0 * p0 + p1) / (1.0 - q1 / q0);
        }
        if close(p, p1) {
            return Ok(RootResult::terminal(
                method,
                p,
                true,
                ConvergenceStatus::Success,
                itr + 1,
                funcalls,
                "converged",
            ));
        }
        p0 = p1;
        q0 = q1;
        p1 = p;
        q1 = func(p1);
        funcalls += 1;
    }
    Ok(RootResult::terminal(
        method,
        p,
        false,
        ConvergenceStatus::MaxIterations,
        maxiter,
        funcalls,
        "Failed to converge",
    ))
}

fn require_bracket(bracket: Option<(f64, f64)>) -> Result<(f64, f64), OptError> {
    bracket.ok_or_else(|| OptError::InvalidArgument {
        detail: String::from("bracket is required for bracketing root methods"),
    })
}

fn validate_root_options(options: RootOptions) -> Result<(), OptError> {
    if options.xtol <= 0.0 || !options.xtol.is_finite() {
        return Err(OptError::InvalidArgument {
            detail: String::from("xtol must be finite and > 0"),
        });
    }
    if options.rtol < 0.0 || !options.rtol.is_finite() {
        return Err(OptError::InvalidArgument {
            detail: String::from("rtol must be finite and >= 0"),
        });
    }
    if options.maxiter == 0 {
        return Err(OptError::InvalidArgument {
            detail: String::from("maxiter must be >= 1"),
        });
    }
    Ok(())
}

fn validate_bracket_finite(bracket: (f64, f64)) -> Result<(), OptError> {
    if !bracket.0.is_finite() || !bracket.1.is_finite() {
        return Err(OptError::NonFiniteInput {
            detail: String::from("bracket endpoints must be finite"),
        });
    }
    if bracket.0 >= bracket.1 {
        return Err(OptError::InvalidBounds {
            detail: String::from("bracket must satisfy a < b"),
        });
    }
    Ok(())
}

fn validate_multivariate_root_params(x0: &[f64], tol: f64, maxiter: usize) -> Result<(), OptError> {
    if x0.is_empty() {
        return Err(OptError::InvalidArgument {
            detail: "x0 must not be empty".to_string(),
        });
    }
    if x0.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: "x0 must contain only finite values".to_string(),
        });
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err(OptError::InvalidArgument {
            detail: "tol must be finite and > 0".to_string(),
        });
    }
    if maxiter == 0 {
        return Err(OptError::InvalidArgument {
            detail: "maxiter must be >= 1".to_string(),
        });
    }
    Ok(())
}

fn evaluate_multivariate_root<F>(
    func: &F,
    x: &[f64],
    expected_len: usize,
    method: &str,
) -> Result<Vec<f64>, OptError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let values = func(x);
    if values.len() != expected_len {
        return Err(OptError::InvalidArgument {
            detail: format!(
                "{method}: function output length must match x0 length (got {}, expected {expected_len})",
                values.len()
            ),
        });
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(OptError::NonFiniteInput {
            detail: format!("{method}: function output must contain only finite values"),
        });
    }
    Ok(values)
}

fn same_sign(lhs: f64, rhs: f64) -> bool {
    lhs.is_sign_positive() == rhs.is_sign_positive()
}

#[derive(Debug, Clone, Copy)]
struct BracketState {
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
}

struct RootEvaluator<'a, F>
where
    F: Fn(f64) -> f64,
{
    f: &'a F,
    options: RootOptions,
    function_calls: usize,
}

impl<'a, F> RootEvaluator<'a, F>
where
    F: Fn(f64) -> f64,
{
    fn new(f: &'a F, options: RootOptions) -> Self {
        Self {
            f,
            options,
            function_calls: 0,
        }
    }

    fn evaluate(&mut self, x: f64) -> Result<f64, OptError> {
        if !x.is_finite() {
            return Err(OptError::NonFiniteInput {
                detail: String::from("root solver evaluated a non-finite point"),
            });
        }
        let value = (self.f)(x);
        self.function_calls += 1;
        if !value.is_finite() {
            return match self.options.mode {
                fsci_runtime::RuntimeMode::Strict => Err(OptError::InvalidArgument {
                    detail: String::from("root function returned non-finite value"),
                }),
                fsci_runtime::RuntimeMode::Hardened => Err(OptError::NonFiniteInput {
                    detail: String::from("hardened mode rejects non-finite function values"),
                }),
            };
        }
        Ok(value)
    }
}

fn prepare_bracket<F>(
    eval: &mut RootEvaluator<'_, F>,
    bracket: (f64, f64),
    _method: RootMethod,
) -> Result<BracketState, OptError>
where
    F: Fn(f64) -> f64,
{
    let fa = eval.evaluate(bracket.0)?;
    let fb = eval.evaluate(bracket.1)?;
    if same_sign(fa, fb) && fa != 0.0 && fb != 0.0 {
        return Err(OptError::SignChangeRequired {
            detail: String::from("bracketing methods require f(a) and f(b) to have opposite signs"),
        });
    }

    Ok(BracketState {
        a: bracket.0,
        b: bracket.1,
        fa,
        fb,
    })
}

// ══════════════════════════════════════════════════════════════════════
// Multivariate Root Finding: root, fsolve
// ══════════════════════════════════════════════════════════════════════

/// Result of multivariate root finding.
#[derive(Debug, Clone)]
pub struct MultivariateRootResult {
    /// Solution vector.
    pub x: Vec<f64>,
    /// Residuals F(x) at the solution.
    pub fun: Vec<f64>,
    /// Whether the solver converged.
    pub converged: bool,
    /// Human-readable message.
    pub message: String,
    /// Number of iterations.
    pub iterations: usize,
    /// Number of function evaluations.
    pub function_calls: usize,
}

/// Find roots of a system of nonlinear equations F(x) = 0.
///
/// Matches `scipy.optimize.fsolve(func, x0)`.
///
/// Uses Newton's method with finite-difference Jacobian and
/// simple step-size damping for robustness.
///
/// # Arguments
/// * `func` — Vector function F: R^n → R^n.
/// * `x0` — Initial guess.
pub fn fsolve<F>(func: F, x0: &[f64]) -> Result<MultivariateRootResult, OptError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    fsolve_with_options(func, x0, 1e-10, 200)
}

fn fsolve_with_options<F>(
    func: F,
    x0: &[f64],
    tol: f64,
    maxiter: usize,
) -> Result<MultivariateRootResult, OptError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    validate_multivariate_root_params(x0, tol, maxiter)?;
    let n = x0.len();

    let eps = 1e-8; // finite difference step

    let mut x: Vec<f64> = x0.to_vec();
    let mut nfev = 0usize;
    let mut jac = vec![vec![0.0; n]; n];

    // F(x) is carried across iterations (refreshed only when we take a step), so it
    // is never re-evaluated at the top of the loop.
    let mut fx = evaluate_multivariate_root(&func, &x, n, "fsolve")?;
    nfev += 1;

    // Broyden ("good") quasi-Newton. scipy/MINPACK's hybr does not rebuild the
    // finite-difference Jacobian every step — it computes it once and refreshes it
    // with an O(n^2) rank-1 update per accepted step, recomputing the full FD
    // Jacobian only when an update goes stale. fsci previously recomputed the full
    // FD Jacobian (n residual evaluations) every iteration; for expensive residuals
    // that dominates the cost. Here `jac_valid` is cleared whenever a step fails to
    // reduce ‖F‖, forcing a fresh FD Jacobian and keeping the method as robust as
    // plain Newton (restart-on-failure), while accepted steps cost only one residual
    // evaluation plus the O(n^2) update.
    let mut jac_valid = false;

    for iteration in 0..maxiter {
        let norm_fx: f64 = fx.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm_fx < tol {
            return Ok(MultivariateRootResult {
                x,
                fun: fx,
                converged: true,
                message: "fsolve converged".to_string(),
                iterations: iteration,
                function_calls: nfev,
            });
        }

        // (Re)build the finite-difference Jacobian only when we lack a usable one.
        if !jac_valid {
            for j in 0..n {
                let h = eps * (1.0 + x[j].abs());
                let original = x[j];
                x[j] += h;
                let fx_plus = evaluate_multivariate_root(&func, &x, n, "fsolve")?;
                x[j] = original; // restore
                nfev += 1;
                for i in 0..n {
                    jac[i][j] = (fx_plus[i] - fx[i]) / h;
                }
            }
            jac_valid = true;
        }

        // Solve J * dx = -F(x) using Gaussian elimination with partial pivoting.
        let mut neg_fx = fx.clone();
        for val in &mut neg_fx {
            *val = -*val;
        }
        let mut singular = false;
        let dx = match solve_dense(&jac, &neg_fx) {
            Some(d) => d,
            None => {
                singular = true;
                // Rank-deficient/singular Jacobian (e.g. a start where two columns
                // coincide). Fall back to a Levenberg-Marquardt damped Gauss-Newton
                // step (JᵀJ + λI) dx = -Jᵀ F: well-posed for λ>0 and a descent
                // direction for ½‖F‖², so the line search below still makes progress.
                // scipy's hybr (MINPACK dogleg) is likewise robust to rank deficiency.
                let mut jtj = vec![vec![0.0; n]; n];
                let mut jtf = vec![0.0; n];
                for i in 0..n {
                    for j in 0..n {
                        jtj[i][j] = (0..n).map(|k| jac[k][i] * jac[k][j]).sum();
                    }
                    jtf[i] = (0..n).map(|k| jac[k][i] * fx[k]).sum();
                }
                let trace: f64 = (0..n).map(|i| jtj[i][i]).sum();
                let lambda = (1.0e-3 * trace / n as f64).max(1.0e-12);
                for (i, row) in jtj.iter_mut().enumerate() {
                    row[i] += lambda;
                }
                let rhs: Vec<f64> = jtf.iter().map(|v| -v).collect();
                match solve_dense(&jtj, &rhs) {
                    Some(d) => d,
                    None => {
                        return Ok(MultivariateRootResult {
                            x,
                            fun: fx,
                            converged: false,
                            message: "fsolve: singular Jacobian".to_string(),
                            iterations: iteration,
                            function_calls: nfev,
                        });
                    }
                }
            }
        };

        // Line search: try full step, then halve until ‖F‖ improves. Keep the
        // residual at the accepted point so the Broyden update needs no extra eval.
        let mut alpha = 1.0;
        let mut improved = false;
        let mut new_x = x.clone();
        let mut new_fx = fx.clone();
        for _ in 0..10 {
            let trial: Vec<f64> = x
                .iter()
                .zip(&dx)
                .map(|(&xi, &di)| xi + alpha * di)
                .collect();
            let ftrial = evaluate_multivariate_root(&func, &trial, n, "fsolve")?;
            nfev += 1;
            let trial_norm: f64 = ftrial.iter().map(|v| v * v).sum::<f64>().sqrt();
            if trial_norm < norm_fx {
                new_x = trial;
                new_fx = ftrial;
                improved = true;
                break;
            }
            alpha *= 0.5;
        }

        if improved {
            if singular {
                // The step came from the LM-damped fallback, not the true Jacobian —
                // a Broyden update off it is meaningless. Rebuild a fresh FD Jacobian
                // next iteration (as plain Newton did) until the Jacobian is
                // well-conditioned, then let Broyden take over.
                jac_valid = false;
            } else {
                // Broyden "good" update: J ← J + ((y − J s) / (sᵀs)) sᵀ, with
                // s = new_x − x and y = new_fx − fx. Costs O(n^2) and no residual evals.
                let s: Vec<f64> = new_x.iter().zip(&x).map(|(a, b)| a - b).collect();
                let sts: f64 = s.iter().map(|v| v * v).sum();
                if sts > 0.0 {
                    for i in 0..n {
                        let js_i: f64 = (0..n).map(|k| jac[i][k] * s[k]).sum();
                        let coeff = (new_fx[i] - fx[i] - js_i) / sts;
                        if coeff != 0.0 {
                            for k in 0..n {
                                jac[i][k] += coeff * s[k];
                            }
                        }
                    }
                } else {
                    jac_valid = false; // degenerate step; rebuild the FD Jacobian next iter
                }
            }
            x = new_x;
            fx = new_fx;
        } else {
            // No tried step reduced ‖F‖. Take the smallest step to avoid stalling and
            // force a fresh FD Jacobian next iteration (whether the current Jacobian
            // was a stale Broyden approximation or a fresh finite-difference one).
            let trial: Vec<f64> = x
                .iter()
                .zip(&dx)
                .map(|(&xi, &di)| xi + alpha * di)
                .collect();
            let ftrial = evaluate_multivariate_root(&func, &trial, n, "fsolve")?;
            nfev += 1;
            x = trial;
            fx = ftrial;
            jac_valid = false;
        }
    }

    Ok(MultivariateRootResult {
        x,
        fun: fx,
        converged: false,
        message: format!("fsolve failed to converge within {maxiter} iterations"),
        iterations: maxiter,
        function_calls: nfev,
    })
}

/// Solve a dense linear system Ax = b using Gaussian elimination with partial pivoting.
/// Returns None if the matrix is singular.
fn solve_dense(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    // Augmented matrix.
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();

    for col in 0..n {
        // Partial pivoting.
        let max_row = (col..n)
            .max_by(|&i, &j| aug[i][col].abs().total_cmp(&aug[j][col].abs()))
            .unwrap_or(col);
        aug.swap(col, max_row);

        if aug[col][col].abs() < 1e-15 {
            return None; // Singular
        }

        let pivot = aug[col][col];
        for row in (col + 1)..n {
            let (aug_col, aug_row) = {
                let (left, right) = aug.split_at_mut(row);
                (&left[col], &mut right[0])
            };
            let factor = aug_row[col] / pivot;
            for (aug_row_j, &aug_col_j) in
                aug_row.iter_mut().zip(aug_col.iter()).take(n + 1).skip(col)
            {
                *aug_row_j -= factor * aug_col_j;
            }
        }
    }

    // Back substitution.
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    Some(x)
}

/// Method for multivariate root finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MultivariateRootMethod {
    /// Powell's hybrid method (Newton + finite-difference Jacobian + line search).
    #[default]
    Hybr,
    /// Broyden's first method (quasi-Newton with approximate Jacobian updates).
    Broyden1,
    /// Broyden's second ("bad") method with approximate inverse Jacobian.
    Broyden2,
    /// Anderson acceleration (DIIS mixing).
    Anderson,
    /// Levenberg-Marquardt algorithm (damped least squares).
    Lm,
    /// Jacobian-free Newton-Krylov method (Newton + finite-difference GMRES).
    NewtonKrylov,
    /// Derivative-free spectral residual method (DF-SANE).
    DfSane,
}

/// Options for multivariate root finding.
#[derive(Debug, Clone)]
pub struct MultivariateRootOptions {
    pub method: MultivariateRootMethod,
    pub tol: f64,
    pub max_iter: usize,
}

impl Default for MultivariateRootOptions {
    fn default() -> Self {
        Self {
            method: MultivariateRootMethod::Hybr,
            tol: 1e-10,
            max_iter: 200,
        }
    }
}

/// Solve a system of nonlinear equations F(x) = 0.
///
/// Matches `scipy.optimize.root(fun, x0, method)`.
///
/// Supports:
/// - method="hybr" (Powell's hybrid: Newton with finite-difference Jacobian)
/// - method="broyden1" (Broyden's first quasi-Newton method)
/// - method="broyden2" (Broyden's second "bad" method)
/// - method="anderson" (Anderson acceleration/DIIS mixing)
/// - method="lm" (Levenberg-Marquardt damped least squares)
pub fn root<F>(
    func: F,
    x0: &[f64],
    options: MultivariateRootOptions,
) -> Result<MultivariateRootResult, OptError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    match options.method {
        MultivariateRootMethod::Hybr => {
            fsolve_with_options(func, x0, options.tol, options.max_iter)
        }
        MultivariateRootMethod::Broyden1 => broyden1(func, x0, options.tol, options.max_iter),
        MultivariateRootMethod::Broyden2 => broyden2(func, x0, options.tol, options.max_iter),
        MultivariateRootMethod::Anderson => {
            anderson(func, x0, options.tol, options.max_iter, 5, 1.0)
        }
        MultivariateRootMethod::Lm => lm_root(func, x0, options.tol, options.max_iter),
        MultivariateRootMethod::NewtonKrylov => {
            newton_krylov(func, x0, options.tol, options.max_iter)
        }
        MultivariateRootMethod::DfSane => df_sane(func, x0, options.tol, options.max_iter),
    }
}

/// Batched nonlinear-system solving: solve `func(x, params) = 0` for MANY parameter sets
/// (`param_rows`), one [`MultivariateRootResult`] per set, all from the shared start `x0`.
/// This is the vmap-over-solver primitive SciPy lacks — a parameter sweep / equilibrium scan
/// loops `root` in Python, calling the Python residual (and Jacobian) per iteration, N solves
/// SERIALLY; here the N independent solves are fanned across cores and the residual is an
/// inlined Rust closure (callback lever × N-way parallel). Result `i` is byte-identical to
/// `root(|x| func(x, &param_rows[i]), x0, options)`.
///
/// Common for steady-state / equilibrium analysis across a parameter grid.
pub fn root_many<F>(
    func: F,
    x0: &[f64],
    param_rows: &[Vec<f64>],
    options: MultivariateRootOptions,
) -> Vec<Result<MultivariateRootResult, OptError>>
where
    F: Fn(&[f64], &[f64]) -> Vec<f64> + Sync,
{
    let nrows = param_rows.len();
    if nrows == 0 {
        return Vec::new();
    }
    let func_ref = &func;
    let opts_ref = &options;
    let solve_one = move |params: &[f64]| {
        let residual = |x: &[f64]| func_ref(x, params);
        root(residual, x0, opts_ref.clone())
    };

    // Each solve is an independent nonlinear-system solve → fan whole parameter sets across
    // cores, capped by the row count; a tiny sweep stays serial.
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    let nthreads = cores.min(nrows);
    if nthreads <= 1 || nrows < 4 {
        return param_rows.iter().map(|p| solve_one(p)).collect();
    }

    let chunk = nrows.div_ceil(nthreads);
    let solve_one = &solve_one;
    let chunk_results: Vec<Vec<Result<MultivariateRootResult, OptError>>> =
        std::thread::scope(|scope| {
            (0..nthreads)
                .filter_map(|t| {
                    let lo = t * chunk;
                    if lo >= nrows {
                        return None;
                    }
                    let hi = (lo + chunk).min(nrows);
                    Some(scope.spawn(move || {
                        (lo..hi)
                            .map(|i| solve_one(&param_rows[i]))
                            .collect::<Vec<_>>()
                    }))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().expect("root_many worker panicked"))
                .collect()
        });

    let mut out = Vec::with_capacity(nrows);
    for cr in chunk_results {
        out.extend(cr);
    }
    out
}

/// Broyden's first method for solving F(x) = 0.
///
/// Quasi-Newton method that maintains an approximate Jacobian B and updates it
/// using the rank-1 formula: B_{k+1} = B_k + (Δf - B_k Δx) Δx^T / ||Δx||²
///
/// More efficient than Newton when the Jacobian is expensive to compute.
pub fn broyden1<F>(
    func: F,
    x0: &[f64],
    tol: f64,
    maxiter: usize,
) -> Result<MultivariateRootResult, OptError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    validate_multivariate_root_params(x0, tol, maxiter)?;
    let n = x0.len();

    let mut x = x0.to_vec();
    let mut fx = evaluate_multivariate_root(&func, &x, n, "broyden1")?;
    let mut nfev = 1usize;

    // Initialize approximate Jacobian as identity
    let mut b_inv: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            row
        })
        .collect();

    for iteration in 0..maxiter {
        let norm_fx: f64 = fx.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm_fx < tol {
            return Ok(MultivariateRootResult {
                x,
                fun: fx,
                converged: true,
                message: "broyden1 converged".to_string(),
                iterations: iteration,
                function_calls: nfev,
            });
        }

        // Compute step: dx = -B^{-1} F(x)
        let dx: Vec<f64> = (0..n)
            .map(|i| {
                -b_inv[i]
                    .iter()
                    .zip(fx.iter())
                    .map(|(b, f)| b * f)
                    .sum::<f64>()
            })
            .collect();

        // Update x
        let x_new: Vec<f64> = x.iter().zip(dx.iter()).map(|(xi, di)| xi + di).collect();
        let fx_new = evaluate_multivariate_root(&func, &x_new, n, "broyden1")?;
        nfev += 1;

        // Broyden update: B^{-1}_{k+1} = B^{-1}_k + (Δx - B^{-1}_k Δf) Δx^T B^{-1}_k / (Δx^T B^{-1}_k Δf)
        let df: Vec<f64> = fx_new
            .iter()
            .zip(fx.iter())
            .map(|(fn_, fo)| fn_ - fo)
            .collect();

        // b_inv_df = B^{-1} * Δf
        let b_inv_df: Vec<f64> = (0..n)
            .map(|i| {
                b_inv[i]
                    .iter()
                    .zip(df.iter())
                    .map(|(b, d)| b * d)
                    .sum::<f64>()
            })
            .collect();

        // numerator = Δx - B^{-1} Δf
        let numer: Vec<f64> = dx
            .iter()
            .zip(b_inv_df.iter())
            .map(|(d, bd)| d - bd)
            .collect();

        // dx^T B^{-1} = row vector
        let dx_b_inv: Vec<f64> = (0..n)
            .map(|j| {
                dx.iter()
                    .zip(b_inv.iter())
                    .map(|(d, row)| d * row[j])
                    .sum::<f64>()
            })
            .collect();

        // denominator = Δx^T B^{-1} Δf
        let denom: f64 = dx.iter().zip(b_inv_df.iter()).map(|(d, bd)| d * bd).sum();

        if denom.abs() > 1e-30 {
            for i in 0..n {
                for j in 0..n {
                    b_inv[i][j] += numer[i] * dx_b_inv[j] / denom;
                }
            }
        }

        x = x_new;
        fx = fx_new;
    }

    Ok(MultivariateRootResult {
        x,
        fun: fx,
        converged: false,
        message: format!("broyden1 failed to converge within {maxiter} iterations"),
        iterations: maxiter,
        function_calls: nfev,
    })
}

/// Broyden's second ("bad") method for root finding.
///
/// Matches `scipy.optimize.broyden2`.
///
/// Quasi-Newton method that maintains an approximate inverse Jacobian H and
/// updates it using:
///   H_{k+1} = H_k + (Δx - H_k Δf) Δf^T / ||Δf||²
///
/// This is the "bad" Broyden method (as opposed to broyden1 which is "good").
/// It tends to be slightly less robust but can work well on some problems.
pub fn broyden2<F>(
    func: F,
    x0: &[f64],
    tol: f64,
    maxiter: usize,
) -> Result<MultivariateRootResult, OptError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    validate_multivariate_root_params(x0, tol, maxiter)?;
    let n = x0.len();

    let mut x = x0.to_vec();
    let mut fx = evaluate_multivariate_root(&func, &x, n, "broyden2")?;
    let mut nfev = 1usize;

    // Initialize approximate inverse Jacobian as identity
    let mut h: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            row
        })
        .collect();

    for iteration in 0..maxiter {
        let norm_fx: f64 = fx.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm_fx < tol {
            return Ok(MultivariateRootResult {
                x,
                fun: fx,
                converged: true,
                message: "broyden2 converged".to_string(),
                iterations: iteration,
                function_calls: nfev,
            });
        }

        // Compute step: dx = -H * F(x)
        let dx: Vec<f64> = (0..n)
            .map(|i| {
                -h[i]
                    .iter()
                    .zip(fx.iter())
                    .map(|(hi, fi)| hi * fi)
                    .sum::<f64>()
            })
            .collect();

        // Update x
        let x_new: Vec<f64> = x.iter().zip(dx.iter()).map(|(xi, di)| xi + di).collect();
        let fx_new = evaluate_multivariate_root(&func, &x_new, n, "broyden2")?;
        nfev += 1;

        // Broyden's "bad" update: H_{k+1} = H_k + (Δx - H_k Δf) Δf^T / ||Δf||²
        let df: Vec<f64> = fx_new
            .iter()
            .zip(fx.iter())
            .map(|(fn_, fo)| fn_ - fo)
            .collect();

        // ||Δf||²
        let df_norm_sq: f64 = df.iter().map(|d| d * d).sum();

        if df_norm_sq > 1e-30 {
            // H_k Δf
            let h_df: Vec<f64> = (0..n)
                .map(|i| {
                    h[i].iter()
                        .zip(df.iter())
                        .map(|(hi, dfi)| hi * dfi)
                        .sum::<f64>()
                })
                .collect();

            // numerator = Δx - H_k Δf
            let numer: Vec<f64> = dx
                .iter()
                .zip(h_df.iter())
                .map(|(dxi, hdf)| dxi - hdf)
                .collect();

            // Update H: H_{k+1}[i][j] = H_k[i][j] + numer[i] * df[j] / ||Δf||²
            for i in 0..n {
                for j in 0..n {
                    h[i][j] += numer[i] * df[j] / df_norm_sq;
                }
            }
        }

        x = x_new;
        fx = fx_new;
    }

    Ok(MultivariateRootResult {
        x,
        fun: fx,
        converged: false,
        message: format!("broyden2 failed to converge within {maxiter} iterations"),
        iterations: maxiter,
        function_calls: nfev,
    })
}

/// Anderson acceleration for root finding.
///
/// Matches `scipy.optimize.anderson`.
///
/// Anderson acceleration (also known as Anderson mixing or DIIS) accelerates
/// the convergence of fixed-point iterations by using a history of previous
/// iterates to compute an optimal linear combination.
///
/// For solving f(x) = 0, it reformulates as a fixed-point problem x = x + f(x)
/// and applies mixing to accelerate convergence.
///
/// # Arguments
/// * `func` - Function f(x) whose root is sought
/// * `x0` - Initial guess
/// * `tol` - Convergence tolerance on ||f(x)||
/// * `maxiter` - Maximum iterations
/// * `m` - Number of previous iterates to store (mixing memory)
/// * `beta` - Mixing parameter (step size, typically 1.0)
pub fn anderson<F>(
    func: F,
    x0: &[f64],
    tol: f64,
    maxiter: usize,
    m: usize,
    beta: f64,
) -> Result<MultivariateRootResult, OptError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    validate_multivariate_root_params(x0, tol, maxiter)?;
    let n = x0.len();
    if m == 0 {
        return Err(OptError::InvalidArgument {
            detail: "anderson memory m must be greater than zero".to_string(),
        });
    }
    if !beta.is_finite() {
        return Err(OptError::InvalidArgument {
            detail: "anderson beta must be finite".to_string(),
        });
    }

    let mut x = x0.to_vec();
    let mut fx = evaluate_multivariate_root(&func, &x, n, "anderson")?;
    let mut nfev = 1usize;

    // History storage: stores (x_k, f_k) pairs
    let mut x_hist: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut f_hist: Vec<Vec<f64>> = Vec::with_capacity(m);

    for iteration in 0..maxiter {
        let norm_fx: f64 = fx.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm_fx < tol {
            return Ok(MultivariateRootResult {
                x,
                fun: fx,
                converged: true,
                message: "anderson converged".to_string(),
                iterations: iteration,
                function_calls: nfev,
            });
        }

        // Store current iterate in history
        if x_hist.len() >= m {
            x_hist.remove(0);
            f_hist.remove(0);
        }
        x_hist.push(x.clone());
        f_hist.push(fx.clone());

        // Compute the Anderson mixing step (displacement from the current x).
        let mk = x_hist.len();
        let step: Vec<f64> = if mk == 1 {
            // Simple fixed-point step: x + beta * f(x).
            (0..n).map(|i| beta * fx[i]).collect()
        } else {
            // Build the residual difference matrix ΔF_j = f_{j+1} - f_j.
            let num_cols = mk - 1;
            let mut delta_f: Vec<Vec<f64>> = Vec::with_capacity(num_cols);
            for j in 0..num_cols {
                let df: Vec<f64> = f_hist[j + 1]
                    .iter()
                    .zip(f_hist[j].iter())
                    .map(|(a, b)| a - b)
                    .collect();
                delta_f.push(df);
            }

            // Normal equations (ΔFᵀΔF) γ = ΔFᵀ f_k with Tikhonov (w0) regularization,
            // so an ill-conditioned ΔFᵀΔF cannot produce huge, divergent γ.
            let mut ata = vec![vec![0.0; num_cols]; num_cols];
            let mut atb = vec![0.0; num_cols];
            for i in 0..num_cols {
                for j in 0..num_cols {
                    ata[i][j] = delta_f[i]
                        .iter()
                        .zip(delta_f[j].iter())
                        .map(|(a, b)| a * b)
                        .sum();
                }
                atb[i] = delta_f[i].iter().zip(fx.iter()).map(|(a, b)| a * b).sum();
            }
            const W0: f64 = 0.01;
            let max_diag = (0..num_cols).map(|i| ata[i][i]).fold(0.0_f64, f64::max);
            let reg = W0 * W0 * max_diag.max(1.0e-30);
            for (i, row) in ata.iter_mut().enumerate() {
                row[i] += reg;
            }
            let gamma = solve_small_system(&ata, &atb);

            // step = beta*f_k - Σ_j γ_j (Δx_j + beta·Δf_j).
            let mut step = vec![0.0; n];
            for (i, s) in step.iter_mut().enumerate() {
                *s = beta * fx[i];
            }
            for j in 0..num_cols {
                for (i, s) in step.iter_mut().enumerate() {
                    let dx_ji = x_hist[j + 1][i] - x_hist[j][i];
                    let df_ji = delta_f[j][i];
                    *s -= gamma[j] * (dx_ji + beta * df_ji);
                }
            }
            step
        };

        // Apply the (w0-regularized) Anderson step. A backtracking line search is
        // deliberately NOT used here: Anderson must take its steps to accumulate the
        // residual history that the least-squares mixing relies on to correct the
        // search direction, so the w0 regularization (not a line search) is what
        // tames the previously-divergent, ill-conditioned coefficients.
        for (xi, si) in x.iter_mut().zip(step.iter()) {
            *xi += si;
        }
        fx = evaluate_multivariate_root(&func, &x, n, "anderson")?;
        nfev += 1;
    }

    Ok(MultivariateRootResult {
        x,
        fun: fx,
        converged: false,
        message: format!("anderson failed to converge within {maxiter} iterations"),
        iterations: maxiter,
        function_calls: nfev,
    })
}

/// Levenberg-Marquardt algorithm for root finding.
///
/// Matches `scipy.optimize.root(method='lm')`.
///
/// The Levenberg-Marquardt algorithm is a damped least-squares method that
/// combines Gauss-Newton and gradient descent. It solves:
///
/// (J^T J + λI) dx = -J^T F(x)
///
/// where λ is an adaptive damping parameter. When λ is small, this approaches
// L2 norm.
fn nk_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn nk_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// Jacobian-free GMRES (non-restarted, up to `max_dim` Arnoldi vectors): solve
// `A·x = b` where `matvec(v) ≈ A·v`. Returns the approximate solution; for
// `max_dim >= n` this is the exact solve (the Krylov space spans Rⁿ).
fn nk_gmres<MV>(matvec: &mut MV, b: &[f64], tol: f64, max_dim: usize) -> Vec<f64>
where
    MV: FnMut(&[f64]) -> Vec<f64>,
{
    let n = b.len();
    let bnorm = nk_norm(b);
    if bnorm == 0.0 {
        return vec![0.0; n];
    }
    let m = max_dim.clamp(1, n);
    let mut q: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
    let mut h = vec![vec![0.0_f64; m]; m + 1];
    let mut cs = vec![0.0_f64; m];
    let mut sn = vec![0.0_f64; m];
    let mut g = vec![0.0_f64; m + 1];
    g[0] = bnorm;
    q.push(b.iter().map(|v| v / bnorm).collect());

    let mut k_used = 0;
    for k in 0..m {
        let mut w = matvec(&q[k]);
        // Modified Gram-Schmidt against q[0..=k].
        for i in 0..=k {
            h[i][k] = nk_dot(&w, &q[i]);
            for j in 0..n {
                w[j] -= h[i][k] * q[i][j];
            }
        }
        h[k + 1][k] = nk_norm(&w);
        let breakdown = h[k + 1][k] <= 1e-14 * bnorm;
        if !breakdown {
            let hk = h[k + 1][k];
            q.push(w.iter().map(|v| v / hk).collect());
        }
        // Apply previous Givens rotations to column k.
        for i in 0..k {
            let temp = cs[i] * h[i][k] + sn[i] * h[i + 1][k];
            h[i + 1][k] = -sn[i] * h[i][k] + cs[i] * h[i + 1][k];
            h[i][k] = temp;
        }
        // New Givens rotation to zero h[k+1][k].
        let r = (h[k][k] * h[k][k] + h[k + 1][k] * h[k + 1][k]).sqrt();
        if r == 0.0 {
            k_used = k + 1;
            break;
        }
        cs[k] = h[k][k] / r;
        sn[k] = h[k + 1][k] / r;
        h[k][k] = r;
        h[k + 1][k] = 0.0;
        g[k + 1] = -sn[k] * g[k];
        g[k] *= cs[k];
        k_used = k + 1;
        if g[k + 1].abs() <= tol * bnorm || breakdown {
            break;
        }
    }

    // Back-substitution: solve upper-triangular H[0..k_used] y = g[0..k_used].
    let kk = k_used;
    let mut y = vec![0.0_f64; kk];
    for i in (0..kk).rev() {
        let mut s = g[i];
        for j in (i + 1)..kk {
            s -= h[i][j] * y[j];
        }
        y[i] = if h[i][i] != 0.0 { s / h[i][i] } else { 0.0 };
    }
    let mut x = vec![0.0_f64; n];
    for i in 0..kk {
        for j in 0..n {
            x[j] += y[i] * q[i][j];
        }
    }
    x
}

/// Solve `F(x) = 0` with a Jacobian-free Newton-Krylov method, matching
/// `scipy.optimize.newton_krylov(func, x0, f_tol=...)`.
///
/// Each Newton step solves `J·dx = −F(x)` with an inner GMRES that approximates
/// `J·v` by the forward finite difference `(F(x + εv) − F(x))/ε`
/// (`ε = √ε_mach·(1+‖x‖)/‖v‖`, as in scipy's `KrylovJacobian`), globalized by an
/// Armijo backtracking line search on `‖F‖`. Terminates when `‖F(x)‖ < f_tol`.
/// Returns the same [`MultivariateRootResult`] as the other root methods.
pub fn newton_krylov<F>(
    func: F,
    x0: &[f64],
    f_tol: f64,
    maxiter: usize,
) -> Result<MultivariateRootResult, OptError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    validate_multivariate_root_params(x0, f_tol, maxiter)?;
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut fx = evaluate_multivariate_root(&func, &x, n, "newton_krylov")?;
    let mut nfev = 1usize;

    for iteration in 0..maxiter {
        let norm_fx = nk_norm(&fx);
        if norm_fx < f_tol {
            return Ok(MultivariateRootResult {
                x,
                fun: fx,
                converged: true,
                message: "newton_krylov converged".to_string(),
                iterations: iteration,
                function_calls: nfev,
            });
        }

        // Inexact-Newton inner tolerance (Eisenstat-Walker style, capped).
        let eta = (0.1 * norm_fx).clamp(1e-10, 0.1);
        let rhs: Vec<f64> = fx.iter().map(|v| -v).collect();
        let normx = nk_norm(&x);
        let rdiff = f64::EPSILON.sqrt();
        let mut matvec = |v: &[f64]| -> Vec<f64> {
            let normv = nk_norm(v);
            if normv == 0.0 {
                return vec![0.0; n];
            }
            let eps = rdiff * (1.0 + normx) / normv;
            let xp: Vec<f64> = x.iter().zip(v).map(|(xi, vi)| xi + eps * vi).collect();
            let fxp = func(&xp);
            nfev += 1;
            fxp.iter().zip(&fx).map(|(a, b)| (a - b) / eps).collect()
        };
        let dx = nk_gmres(&mut matvec, &rhs, eta, n);

        // Armijo backtracking line search on ‖F‖.
        let mut lam = 1.0_f64;
        let (x_next, fx_next) = loop {
            let x_new: Vec<f64> = x.iter().zip(&dx).map(|(xi, di)| xi + lam * di).collect();
            let fx_new = func(&x_new);
            nfev += 1;
            let norm_new = nk_norm(&fx_new);
            if norm_new < (1.0 - 1e-4 * lam) * norm_fx || lam < 1e-10 {
                break (x_new, fx_new);
            }
            lam *= 0.5;
        };
        x = x_next;
        fx = fx_next;
    }

    let converged = nk_norm(&fx) < f_tol;
    Ok(MultivariateRootResult {
        x,
        fun: fx,
        converged,
        message: if converged {
            "newton_krylov converged".to_string()
        } else {
            "newton_krylov reached maxiter without converging".to_string()
        },
        iterations: maxiter,
        function_calls: nfev,
    })
}

fn df_sane_line_search_budget(maxiter: usize, n: usize) -> usize {
    let per_iteration = n.saturating_mul(2).saturating_add(4);
    maxiter.saturating_mul(per_iteration).saturating_add(1000)
}

/// Solve `F(x) = 0` with the derivative-free spectral residual method (DF-SANE),
/// matching `scipy.optimize.root(method='df-sane')`.
///
/// Uses spectral (Barzilai-Borwein) steps `d = -σ_k F(x_k)` with the Cruz-
/// Martinez-Raydan nonmonotone backtracking line search (window `M = 10`,
/// `η_k = ‖F₀‖²/(1+k)²`). Converges when `‖F(x_k)‖ < ftol·‖F(x₀)‖ + 1e-300`.
pub fn df_sane<F>(
    func: F,
    x0: &[f64],
    ftol: f64,
    maxiter: usize,
) -> Result<MultivariateRootResult, OptError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    validate_multivariate_root_params(x0, ftol, maxiter)?;
    let n = x0.len();
    const M: usize = 10;
    const SIGMA_EPS: f64 = 1e-10;
    const GAMMA: f64 = 1e-4;
    const TAU_MIN: f64 = 0.1;
    const TAU_MAX: f64 = 0.5;
    let fatol = 1e-300;

    let merit = |fv: &[f64]| -> f64 { fv.iter().map(|v| v * v).sum::<f64>() };

    let mut x_k = x0.to_vec();
    let mut f_k_res = evaluate_multivariate_root(&func, &x_k, n, "df_sane")?;
    let mut nfev = 1usize;
    let mut f_k = merit(&f_k_res); // ‖F‖²
    let f_0 = f_k;
    let f0_norm = f_0.sqrt();
    let mut sigma_k = 1.0_f64;
    let mut prev_fs: std::collections::VecDeque<f64> = std::collections::VecDeque::new();
    prev_fs.push_back(f_k);

    for k in 0..maxiter {
        let fk_norm = f_k.sqrt();
        if fk_norm < ftol * f0_norm + fatol {
            return Ok(MultivariateRootResult {
                x: x_k,
                fun: f_k_res,
                converged: true,
                message: "df_sane converged".to_string(),
                iterations: k,
                function_calls: nfev,
            });
        }
        // Clamp the spectral parameter magnitude.
        if sigma_k.abs() > 1.0 / SIGMA_EPS {
            sigma_k = sigma_k.signum() / SIGMA_EPS;
        } else if sigma_k.abs() < SIGMA_EPS {
            sigma_k = SIGMA_EPS;
        }
        let d: Vec<f64> = f_k_res.iter().map(|v| -sigma_k * v).collect();
        let eta = f_0 / ((1 + k) as f64).powi(2);
        let f_bar = prev_fs.iter().cloned().fold(f64::MIN, f64::max);
        let fk_cur = f_k; // prev_fs[-1]

        // Cruz nonmonotone line search.
        let mut alpha_p = 1.0_f64;
        let mut alpha_m = 1.0_f64;
        let (xp, fp, fp_res) = loop {
            let xp: Vec<f64> = x_k
                .iter()
                .zip(&d)
                .map(|(xi, di)| xi + alpha_p * di)
                .collect();
            let fp_res = func(&xp);
            nfev += 1;
            let fp = merit(&fp_res);
            if fp <= f_bar + eta - GAMMA * alpha_p * alpha_p * fk_cur {
                break (xp, fp, fp_res);
            }
            let alpha_tp = alpha_p * alpha_p * fk_cur / (fp + (2.0 * alpha_p - 1.0) * fk_cur);

            let xm: Vec<f64> = x_k
                .iter()
                .zip(&d)
                .map(|(xi, di)| xi - alpha_m * di)
                .collect();
            let fm_res = func(&xm);
            nfev += 1;
            let fm = merit(&fm_res);
            if fm <= f_bar + eta - GAMMA * alpha_m * alpha_m * fk_cur {
                break (xm, fm, fm_res);
            }
            let alpha_tm = alpha_m * alpha_m * fk_cur / (fm + (2.0 * alpha_m - 1.0) * fk_cur);

            alpha_p = alpha_tp.clamp(TAU_MIN * alpha_p, TAU_MAX * alpha_p);
            alpha_m = alpha_tm.clamp(TAU_MIN * alpha_m, TAU_MAX * alpha_m);
            if nfev > df_sane_line_search_budget(maxiter, n) {
                // Safety: abandon a stalled line search.
                break (xp, fp, fp_res);
            }
        };

        // Spectral parameter update: σ = <s,s>/<s,y>.
        let s_k: Vec<f64> = xp.iter().zip(&x_k).map(|(a, b)| a - b).collect();
        let y_k: Vec<f64> = fp_res.iter().zip(&f_k_res).map(|(a, b)| a - b).collect();
        let ss = nk_dot(&s_k, &s_k);
        let sy = nk_dot(&s_k, &y_k);
        sigma_k = if sy != 0.0 { ss / sy } else { 1.0 };

        x_k = xp;
        f_k_res = fp_res;
        f_k = fp;
        prev_fs.push_back(fp);
        if prev_fs.len() > M {
            prev_fs.pop_front();
        }
    }

    let converged = f_k.sqrt() < ftol * f0_norm + fatol;
    Ok(MultivariateRootResult {
        x: x_k,
        fun: f_k_res,
        converged,
        message: if converged {
            "df_sane converged".to_string()
        } else {
            "df_sane reached maxiter without converging".to_string()
        },
        iterations: maxiter,
        function_calls: nfev,
    })
}

/// the Gauss-Newton method; when λ is large, it approaches gradient descent.
///
/// # Arguments
/// * `func` - Function F(x) whose root is sought
/// * `x0` - Initial guess
/// * `tol` - Convergence tolerance on ||F(x)||
/// * `maxiter` - Maximum iterations
pub fn lm_root<F>(
    func: F,
    x0: &[f64],
    tol: f64,
    maxiter: usize,
) -> Result<MultivariateRootResult, OptError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    validate_multivariate_root_params(x0, tol, maxiter)?;
    let n = x0.len();

    let eps = 1e-8; // Finite difference step
    let mut lambda = 1e-3; // Initial damping parameter
    let lambda_up = 10.0; // Factor to increase lambda on bad step
    let lambda_down = 0.1; // Factor to decrease lambda on good step

    let mut x = x0.to_vec();
    let mut fx = evaluate_multivariate_root(&func, &x, n, "lm_root")?;
    let mut nfev = 1usize;

    for iteration in 0..maxiter {
        let norm_fx: f64 = fx.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm_fx < tol {
            return Ok(MultivariateRootResult {
                x,
                fun: fx,
                converged: true,
                message: "lm converged".to_string(),
                iterations: iteration,
                function_calls: nfev,
            });
        }

        // Compute Jacobian via finite differences
        let mut jac = vec![vec![0.0; n]; n];
        for j in 0..n {
            let h = eps * (1.0 + x[j].abs());
            let original = x[j];
            x[j] += h;
            let fx_plus = evaluate_multivariate_root(&func, &x, n, "lm_root")?;
            x[j] = original;
            nfev += 1;
            for i in 0..n {
                jac[i][j] = (fx_plus[i] - fx[i]) / h;
            }
        }

        // Compute J^T J and J^T F. Single row-outer pass (jac read contiguously,
        // jtj streamed row-major) mirroring `curvefit::jtj_matrix_into`, exploiting
        // J^T J symmetry (upper triangle + mirror) to halve the products. BYTE-IDENTICAL
        // to the naive `for i { for j { for row } }`: `row[i]*row[j] == row[j]*row[i]`
        // (exact IEEE754 commutativity) and each cell still accumulates rows in ascending
        // order. Replaces the strided column re-scan (n² cache-missing passes over jac).
        let mut jtj = vec![vec![0.0; n]; n];
        let mut jtf = vec![0.0; n];
        for (row, &fx_value) in jac.iter().zip(fx.iter()) {
            for i in 0..n {
                let ri = row[i];
                for j in i..n {
                    let v = ri * row[j];
                    jtj[i][j] += v;
                    if i != j {
                        jtj[j][i] += v;
                    }
                }
                jtf[i] += ri * fx_value;
            }
        }

        // Try steps with adaptive damping
        let mut step_accepted = false;
        for _ in 0..10 {
            // Build (J^T J + λI)
            let mut lhs = jtj.clone();
            for (i, row) in lhs.iter_mut().enumerate().take(n) {
                row[i] += lambda;
            }

            // Solve (J^T J + λI) dx = -J^T F
            let neg_jtf: Vec<f64> = jtf.iter().map(|v| -v).collect();
            let dx = match solve_lm_system(&lhs, &neg_jtf) {
                Some(d) => d,
                None => {
                    lambda *= lambda_up;
                    continue;
                }
            };

            // Trial step
            let x_trial: Vec<f64> = x.iter().zip(&dx).map(|(&xi, &di)| xi + di).collect();
            let fx_trial = evaluate_multivariate_root(&func, &x_trial, n, "lm_root")?;
            nfev += 1;

            let trial_norm: f64 = fx_trial.iter().map(|v| v * v).sum::<f64>().sqrt();

            if trial_norm < norm_fx {
                // Good step: accept and decrease damping
                x = x_trial;
                fx = fx_trial;
                lambda *= lambda_down;
                lambda = lambda.max(1e-10); // Prevent underflow
                step_accepted = true;
                break;
            } else {
                // Bad step: increase damping and retry
                lambda *= lambda_up;
                lambda = lambda.min(1e10); // Prevent overflow
            }
        }

        if !step_accepted {
            // Failed to find an improving step; continue with current x
            // but give it one more iteration chance
        }
    }

    Ok(MultivariateRootResult {
        x,
        fun: fx,
        converged: false,
        message: format!("lm failed to converge within {maxiter} iterations"),
        iterations: maxiter,
        function_calls: nfev,
    })
}

/// Solve a linear system for LM using Cholesky-like decomposition.
fn solve_lm_system(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    if n == 0 {
        return Some(vec![]);
    }

    // Use Gaussian elimination with partial pivoting (same as solve_dense)
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();

    for col in 0..n {
        // Partial pivoting
        let max_row = (col..n)
            .max_by(|&i, &j| aug[i][col].abs().total_cmp(&aug[j][col].abs()))
            .unwrap_or(col);
        aug.swap(col, max_row);

        if aug[col][col].abs() < 1e-15 {
            return None;
        }

        let pivot = aug[col][col];
        let pivot_row = aug[col].clone();
        for row in aug.iter_mut().take(n).skip(col + 1) {
            let factor = row[col] / pivot;
            for (entry, pivot_entry) in row[col..=n].iter_mut().zip(pivot_row[col..=n].iter()) {
                *entry -= factor * *pivot_entry;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for (entry, x_value) in aug[i][(i + 1)..n].iter().zip(x[(i + 1)..n].iter()) {
            sum -= *entry * *x_value;
        }
        x[i] = sum / aug[i][i];
    }

    Some(x)
}

/// Solve a small linear system Ax = b using Gaussian elimination with partial pivoting.
fn solve_small_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    if n == 0 {
        return vec![];
    }

    // Create augmented matrix
    let mut aug: Vec<Vec<f64>> = a.to_vec();
    let mut rhs = b.to_vec();

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for (row, row_values) in aug.iter().enumerate().take(n).skip(col + 1) {
            if row_values[col].abs() > max_val {
                max_val = row_values[col].abs();
                max_row = row;
            }
        }

        // Swap rows
        if max_row != col {
            aug.swap(col, max_row);
            rhs.swap(col, max_row);
        }

        // Check for singularity
        if aug[col][col].abs() < 1e-14 {
            // Return zero coefficients for singular matrix
            return vec![0.0; n];
        }

        // Eliminate
        let pivot = aug[col][col];
        let pivot_row = aug[col].clone();
        let rhs_col = rhs[col];
        for (row, row_values) in aug.iter_mut().enumerate().take(n).skip(col + 1) {
            let factor = row_values[col] / pivot;
            for (entry, pivot_entry) in row_values[col..n].iter_mut().zip(pivot_row[col..n].iter())
            {
                *entry -= factor * *pivot_entry;
            }
            rhs[row] -= factor * rhs_col;
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    x
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, OnceLock};
    use std::time::{SystemTime, UNIX_EPOCH};

    use fsci_runtime::RuntimeMode;
    use proptest::prelude::*;
    use serde::Serialize;

    use super::{
        MultivariateRootMethod, MultivariateRootOptions, anderson, broyden1, broyden2, fsolve,
        lm_root, root, root_many,
    };
    use crate::{
        ConvergenceStatus, RootMethod, RootOptions, RootResults, bisect, brenth, brentq,
        brentq_many, df_sane, halley, newton, newton_krylov, newton_many, newton_scalar, ridder,
        root_scalar, secant, secant_many, toms748,
    };

    #[derive(Debug, Serialize)]
    struct TestLogEntry<'a> {
        test_id: &'a str,
        optimizer: &'a str,
        problem: &'a str,
        n_dim: usize,
        mode: &'a str,
        converged: bool,
        nfev: usize,
        final_f: Option<f64>,
        seed: u64,
        timestamp_ms: u64,
    }

    fn test_log_sink() -> &'static Mutex<Vec<String>> {
        static TEST_LOGS: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
        TEST_LOGS.get_or_init(|| Mutex::new(Vec::new()))
    }

    fn now_unix_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_millis() as u64)
    }

    fn mode_name(mode: RuntimeMode) -> &'static str {
        match mode {
            RuntimeMode::Strict => "strict",
            RuntimeMode::Hardened => "hardened",
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn push_test_log(
        test_id: &str,
        optimizer: &str,
        problem: &str,
        mode: RuntimeMode,
        converged: bool,
        nfev: usize,
        final_f: Option<f64>,
        seed: u64,
    ) {
        let entry = TestLogEntry {
            test_id,
            optimizer,
            problem,
            n_dim: 1,
            mode: mode_name(mode),
            converged,
            nfev,
            final_f,
            seed,
            timestamp_ms: now_unix_ms(),
        };
        let payload = serde_json::to_string(&entry).expect("serialize test log");
        let parsed: serde_json::Value =
            serde_json::from_str(&payload).expect("re-parse serialized log payload");
        assert!(parsed.get("test_id").is_some());
        assert!(parsed.get("optimizer").is_some());
        assert!(parsed.get("timestamp_ms").is_some());
        test_log_sink()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(payload);
    }

    fn cubic(x: f64) -> f64 {
        x * x * x - 2.0
    }

    fn monotone_linear(x: f64) -> f64 {
        x - 3.0
    }

    fn discontinuous_step(x: f64) -> f64 {
        if x < 0.5 { -1.0 } else { 1.0 }
    }

    #[test]
    fn bisect_finds_root() {
        let options = RootOptions {
            method: Some(RootMethod::Bisect),
            xtol: 1.0e-10,
            rtol: 1.0e-10,
            maxiter: 200,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let result: RootResults = bisect(cubic, (0.0, 2.0), options).expect("bisect executes");
        assert!(result.converged, "{}", result.message);
        assert_eq!(result.flag, "converged");
        assert_eq!(result.status, ConvergenceStatus::Success);
        assert!((result.root - 2f64.cbrt()).abs() < 1.0e-8);
        push_test_log(
            "bisect-monotone-root",
            "bisect",
            "cubic",
            RuntimeMode::Strict,
            result.converged,
            result.function_calls,
            Some(cubic(result.root)),
            301,
        );
    }

    #[test]
    fn root_results_flags_match_scipy_status_strings() {
        let cases = [
            (ConvergenceStatus::Success, "converged", true),
            (ConvergenceStatus::MaxIterations, "convergence error", false),
            (
                ConvergenceStatus::MaxEvaluations,
                "convergence error",
                false,
            ),
            (ConvergenceStatus::PrecisionLoss, "convergence error", false),
            (ConvergenceStatus::CallbackStop, "convergence error", false),
            (ConvergenceStatus::NanEncountered, "value error", false),
            (ConvergenceStatus::OutOfBounds, "value error", false),
            (ConvergenceStatus::NotImplemented, "value error", false),
            (ConvergenceStatus::InvalidInput, "value error", false),
        ];

        for (status, expected_flag, expected_converged) in cases {
            let result = RootResults::terminal(
                RootMethod::Bisect,
                0.0,
                expected_converged,
                status,
                0,
                0,
                "test status",
            );
            assert_eq!(result.flag, expected_flag);
            assert_eq!(result.converged, expected_converged);
        }
    }

    #[test]
    fn df_sane_finds_roots() {
        // F(x) = [x0+x1-3, x0^2+x1^2-5]; root (2,1).
        let f1 = |x: &[f64]| vec![x[0] + x[1] - 3.0, x[0] * x[0] + x[1] * x[1] - 5.0];
        let r = df_sane(f1, &[2.0, 0.5], 1e-8, 1000).unwrap();
        assert!(r.converged, "{}", r.message);
        let fr: f64 = r.fun.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(fr < 1e-6, "||F|| = {fr}");

        // Powell singular-ish smooth system with root at origin-shifted point.
        let f2 = |x: &[f64]| vec![(x[0] - 1.0).powi(3) - x[1] + 1.0, x[1] - 2.0];
        let r2 = df_sane(f2, &[2.0, 3.0], 1e-8, 2000).unwrap();
        assert!(r2.converged, "{}", r2.message);
        // root: x1=2, (x0-1)^3 = x1-1 = 1 -> x0 = 2.
        assert!(
            (r2.x[0] - 2.0).abs() < 1e-4 && (r2.x[1] - 2.0).abs() < 1e-4,
            "x = {:?}",
            r2.x
        );
    }

    #[test]
    fn df_sane_line_search_budget_saturates_on_large_dimensions() {
        assert_eq!(super::df_sane_line_search_budget(10, 3), 1100);
        assert_eq!(super::df_sane_line_search_budget(usize::MAX, 1), usize::MAX);
        assert_eq!(super::df_sane_line_search_budget(1, usize::MAX), usize::MAX);
    }

    #[test]
    fn newton_krylov_finds_roots() {
        // F(x) = [x0+x1-3, x0^2+x1^2-5]; scipy newton_krylov from [2,0.5] -> (2,1).
        let f1 = |x: &[f64]| vec![x[0] + x[1] - 3.0, x[0] * x[0] + x[1] * x[1] - 5.0];
        let r = newton_krylov(f1, &[2.0, 0.5], 1e-10, 100).unwrap();
        assert!(r.converged, "{}", r.message);
        assert!(
            (r.x[0] - 2.0).abs() < 1e-7 && (r.x[1] - 1.0).abs() < 1e-7,
            "x = {:?}",
            r.x
        );

        // Classic 3-equation system; scipy root (0.5, 0, -0.52359878).
        let f2 = |x: &[f64]| {
            vec![
                3.0 * x[0] - (x[1] * x[2]).cos() - 0.5,
                x[0] * x[0] - 81.0 * (x[1] + 0.1).powi(2) + x[2].sin() + 1.06,
                (-x[0] * x[1]).exp() + 20.0 * x[2] + (10.0 * std::f64::consts::PI - 3.0) / 3.0,
            ]
        };
        let r2 = newton_krylov(f2, &[0.1, 0.1, -0.1], 1e-10, 200).unwrap();
        assert!(r2.converged, "{}", r2.message);
        assert!(
            (r2.x[0] - 0.5).abs() < 1e-6
                && r2.x[1].abs() < 1e-6
                && (r2.x[2] + std::f64::consts::FRAC_PI_6).abs() < 1e-6,
            "x2 = {:?}",
            r2.x
        );
    }

    #[test]
    fn newton_unified_matches_scipy() {
        // f(x) = x^3 - 2x - 5, root ~2.0945514815423265.
        let f = |x: f64| x * x * x - 2.0 * x - 5.0;
        let fp = |x: f64| 3.0 * x * x - 2.0;
        let fp2 = |x: f64| 6.0 * x;
        let root = 2.0945514815423265_f64;

        // Newton-Raphson: scipy 4 iters, 8 funcalls.
        let r = newton(f, 2.0, Some(&fp), None, None, 1.48e-8, 0.0, 50).unwrap();
        assert!(r.converged && r.method == RootMethod::Newton);
        assert!((r.root - root).abs() < 1e-12);
        assert_eq!((r.iterations, r.function_calls), (4, 8));

        // Halley: scipy 3 iters, 9 funcalls.
        let r = newton(f, 2.0, Some(&fp), Some(&fp2), None, 1.48e-8, 0.0, 50).unwrap();
        assert!(r.converged && r.method == RootMethod::Halley);
        assert!((r.root - root).abs() < 1e-12);
        assert_eq!((r.iterations, r.function_calls), (3, 9));

        // Secant: scipy 5 iters, 6 funcalls.
        let r = newton(f, 2.0, None, None, None, 1.48e-8, 0.0, 50).unwrap();
        assert!(r.converged && r.method == RootMethod::Secant);
        assert!((r.root - root).abs() < 1e-12);
        assert_eq!((r.iterations, r.function_calls), (5, 6));

        // Secant on cos(x) - x: scipy root 0.7390851332151601, 5 iters, 6 funcalls.
        let g = |x: f64| x.cos() - x;
        let r = newton(g, 0.5, None, None, None, 1.48e-8, 0.0, 50).unwrap();
        assert!((r.root - 0.7390851332151601).abs() < 1e-12);
        assert_eq!((r.iterations, r.function_calls), (5, 6));
    }

    #[test]
    fn newton_rejects_invalid_tolerances() {
        let f = |x: f64| x * x - 2.0;
        let fp = |x: f64| 2.0 * x;

        let cases = [
            (f64::NAN, 0.0),
            (f64::INFINITY, 0.0),
            (1.0e-8, -1.0e-3),
            (1.0e-8, f64::NAN),
            (1.0e-8, f64::INFINITY),
        ];

        for (tol, rtol) in cases {
            let err = newton(f, 1.0, Some(&fp), None, None, tol, rtol, 20)
                .expect_err("invalid scalar Newton tolerance should fail");
            assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
        }
    }

    #[test]
    fn newton_rejects_second_derivative_without_first_derivative() {
        let f = |x: f64| x * x - 2.0;
        let fp2 = |_x: f64| 2.0;

        let err = newton(f, 1.0, None, Some(&fp2), None, 1.0e-8, 0.0, 20)
            .expect_err("fprime2 without fprime must be rejected");
        assert_eq!(
            err,
            crate::OptError::InvalidArgument {
                detail: String::from("fprime2 requires fprime"),
            }
        );
    }

    #[test]
    fn newton_rejects_nonfinite_secant_seed() {
        let f = |x: f64| x * x - 2.0;

        for x1 in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = newton(f, 1.0, None, None, Some(x1), 1.0e-8, 0.0, 20)
                .expect_err("non-finite secant seed should fail");
            assert!(matches!(err, crate::OptError::NonFiniteInput { .. }));
        }
    }

    #[test]
    fn brentq_finds_root() {
        let options = RootOptions {
            method: Some(RootMethod::Brentq),
            xtol: 1.0e-12,
            rtol: 1.0e-12,
            maxiter: 100,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let result = brentq(cubic, (0.0, 2.0), options).expect("brentq executes");
        assert!(result.converged, "{}", result.message);
        assert!((result.root - 2f64.cbrt()).abs() < 1.0e-10);
        push_test_log(
            "brentq-polynomial-root",
            "brentq",
            "cubic",
            RuntimeMode::Strict,
            result.converged,
            result.function_calls,
            Some(cubic(result.root)),
            302,
        );
    }

    #[test]
    fn bracketing_root_finders_match_scipy() {
        // All bracketing methods must converge to the root of cos(x)-x on [0,1]
        // — the Dottie number 0.7390851332151607, to which scipy.optimize's
        // brentq/ridder/toms748/bisect all converge.
        let f = |x: f64| x.cos() - x;
        let true_root = 0.739_085_133_215_160_6_f64;
        let opts = || RootOptions {
            xtol: 1.0e-12,
            rtol: 1.0e-12,
            maxiter: 100,
            ..RootOptions::default()
        };
        let runs = [
            ("brentq", brentq(f, (0.0, 1.0), opts())),
            ("ridder", ridder(f, (0.0, 1.0), opts())),
            ("toms748", toms748(f, (0.0, 1.0), opts())),
            ("bisect", bisect(f, (0.0, 1.0), opts())),
        ];
        for (name, r) in runs {
            let res = r.unwrap_or_else(|e| panic!("{name} errored: {e:?}"));
            assert!(res.converged, "{name} did not converge: {}", res.message);
            assert!(
                (res.root - true_root).abs() < 1e-10,
                "{name}: {} != {true_root}",
                res.root
            );
        }
    }

    #[test]
    fn brentq_accepts_root_at_bracket_endpoint() {
        let options = RootOptions {
            method: Some(RootMethod::Brentq),
            xtol: 1.0e-12,
            rtol: 1.0e-12,
            maxiter: 100,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let result =
            brentq(|x| x - 1.0, (1.0, 2.0), options).expect("brentq should accept endpoint root");
        assert!(result.converged);
        assert!((result.root - 1.0).abs() <= f64::EPSILON);
        push_test_log(
            "brentq-endpoint-root",
            "brentq",
            "linear",
            RuntimeMode::Strict,
            result.converged,
            result.function_calls,
            Some(result.root - 1.0),
            303,
        );
    }

    #[test]
    fn brentq_rejects_missing_sign_change() {
        let options = RootOptions {
            method: Some(RootMethod::Brentq),
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let err = brentq(cubic, (2.0, 3.0), options).expect_err("must fail without sign change");
        assert!(matches!(err, crate::OptError::SignChangeRequired { .. }));
        push_test_log(
            "brentq-sign-change-error",
            "brentq",
            "cubic",
            RuntimeMode::Strict,
            false,
            0,
            None,
            304,
        );
    }

    #[test]
    fn brenth_matches_brentq_on_same_problem() {
        let options = RootOptions {
            method: Some(RootMethod::Brenth),
            xtol: 1.0e-12,
            rtol: 1.0e-12,
            maxiter: 150,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let brentq_result = brentq(cubic, (0.0, 2.0), options).expect("brentq executes");
        let brenth_result = brenth(cubic, (0.0, 2.0), options).expect("brenth executes");
        assert!((brenth_result.root - brentq_result.root).abs() < 1.0e-10);
        push_test_log(
            "brenth-matches-brentq",
            "brenth",
            "cubic",
            RuntimeMode::Strict,
            brenth_result.converged,
            brenth_result.function_calls,
            Some(cubic(brenth_result.root)),
            305,
        );
    }

    #[test]
    fn root_scalar_uses_default_method_for_bracket() {
        let options = RootOptions {
            method: None,
            xtol: 1.0e-12,
            rtol: 1.0e-12,
            maxiter: 120,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let result = root_scalar(cubic, Some((0.0, 2.0)), None, None, options)
            .expect("root_scalar executes");
        assert!(result.converged);
        assert_eq!(result.method, RootMethod::Brentq);
        push_test_log(
            "root-scalar-default-method",
            "root_scalar",
            "cubic",
            RuntimeMode::Strict,
            result.converged,
            result.function_calls,
            Some(cubic(result.root)),
            306,
        );
    }

    #[test]
    fn sign_change_validation_is_enforced() {
        let options = RootOptions {
            method: Some(RootMethod::Bisect),
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let err = bisect(cubic, (2.0, 3.0), options).expect_err("must fail without sign change");
        assert!(matches!(err, crate::OptError::SignChangeRequired { .. }));
        push_test_log(
            "bisect-sign-change-error",
            "bisect",
            "cubic",
            RuntimeMode::Strict,
            false,
            0,
            None,
            307,
        );
    }

    #[test]
    fn bisect_respects_tolerance_convergence() {
        let options = RootOptions {
            method: Some(RootMethod::Bisect),
            xtol: 1.0e-6,
            rtol: 1.0e-6,
            maxiter: 80,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let result = bisect(monotone_linear, (0.0, 5.0), options).expect("bisect executes");
        assert!(result.converged);
        assert!((result.root - 3.0).abs() < 5.0e-6);
        push_test_log(
            "bisect-tolerance-convergence",
            "bisect",
            "linear",
            RuntimeMode::Strict,
            result.converged,
            result.function_calls,
            Some(monotone_linear(result.root)),
            308,
        );
    }

    #[test]
    fn brentq_handles_discontinuous_function_with_defined_output() {
        let options = RootOptions {
            method: Some(RootMethod::Brentq),
            xtol: 1.0e-8,
            rtol: 1.0e-8,
            maxiter: 120,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let result = brentq(discontinuous_step, (0.0, 1.0), options).expect("brentq executes");
        assert!(result.root.is_finite());
        assert!((0.0..=1.0).contains(&result.root));
        push_test_log(
            "brentq-discontinuous-defined",
            "brentq",
            "step-discontinuity",
            RuntimeMode::Strict,
            result.converged,
            result.function_calls,
            Some(discontinuous_step(result.root)),
            309,
        );
    }

    #[test]
    fn bracketing_methods_agree_on_root_within_tolerance() {
        let options = RootOptions {
            xtol: 1.0e-10,
            rtol: 1.0e-10,
            maxiter: 200,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let bis = bisect(
            cubic,
            (0.0, 2.0),
            RootOptions {
                method: Some(RootMethod::Bisect),
                ..options
            },
        )
        .expect("bisect executes");
        let brq = brentq(
            cubic,
            (0.0, 2.0),
            RootOptions {
                method: Some(RootMethod::Brentq),
                ..options
            },
        )
        .expect("brentq executes");
        let rid = ridder(
            cubic,
            (0.0, 2.0),
            RootOptions {
                method: Some(RootMethod::Ridder),
                ..options
            },
        )
        .expect("ridder executes");
        assert!((bis.root - brq.root).abs() < 1.0e-7);
        assert!((rid.root - brq.root).abs() < 1.0e-7);
        push_test_log(
            "root-method-agreement",
            "all",
            "cubic",
            RuntimeMode::Strict,
            true,
            bis.function_calls + brq.function_calls + rid.function_calls,
            Some(cubic(brq.root)),
            310,
        );
    }

    #[test]
    fn tight_bracket_converges_quickly() {
        let root = 2f64.cbrt();
        let options = RootOptions {
            method: Some(RootMethod::Brentq),
            xtol: 1.0e-14,
            rtol: 1.0e-14,
            maxiter: 100,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let result =
            brentq(cubic, (root - 1.0e-4, root + 1.0e-4), options).expect("brentq executes");
        assert!(result.converged);
        assert!(result.iterations <= 10);
        push_test_log(
            "tight-bracket-fast",
            "brentq",
            "cubic",
            RuntimeMode::Strict,
            result.converged,
            result.function_calls,
            Some(cubic(result.root)),
            311,
        );
    }

    #[test]
    fn wide_bracket_still_converges() {
        let options = RootOptions {
            method: Some(RootMethod::Brentq),
            xtol: 1.0e-10,
            rtol: 1.0e-10,
            maxiter: 250,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let result = brentq(cubic, (-1000.0, 1000.0), options).expect("brentq executes");
        assert!(result.converged);
        assert!((result.root - 2f64.cbrt()).abs() < 1.0e-7);
        push_test_log(
            "wide-bracket-converges",
            "brentq",
            "cubic",
            RuntimeMode::Strict,
            result.converged,
            result.function_calls,
            Some(cubic(result.root)),
            312,
        );
    }

    #[test]
    fn ridder_finds_root_for_cubic() {
        let options = RootOptions {
            method: Some(RootMethod::Ridder),
            xtol: 1.0e-10,
            rtol: 1.0e-10,
            maxiter: 120,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let result = ridder(cubic, (0.0, 2.0), options).expect("ridder executes");
        assert!(result.converged);
        assert!((result.root - 2f64.cbrt()).abs() < 1.0e-7);
        push_test_log(
            "ridder-cubic-root",
            "ridder",
            "cubic",
            RuntimeMode::Strict,
            result.converged,
            result.function_calls,
            Some(cubic(result.root)),
            313,
        );
    }

    #[test]
    fn root_scalar_with_x0_uses_secant() {
        let options = RootOptions {
            method: None,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let result =
            root_scalar(cubic, None, Some(0.5), Some(2.0), options).expect("secant via x0");
        assert!(result.converged, "secant failed: {}", result.message);
        assert!(
            (cubic(result.root)).abs() < 1e-8,
            "residual too large: {}",
            cubic(result.root)
        );
        push_test_log(
            "root-scalar-missing-bracket",
            "root_scalar",
            "cubic",
            RuntimeMode::Strict,
            false,
            0,
            None,
            314,
        );
    }

    #[test]
    fn root_scalar_without_inputs_reports_selection_error() {
        let options = RootOptions {
            method: None,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let err =
            root_scalar(cubic, None, None, None, options).expect_err("must reject missing inputs");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
        push_test_log(
            "root-scalar-selection-error",
            "root_scalar",
            "cubic",
            RuntimeMode::Strict,
            false,
            0,
            None,
            315,
        );
    }

    #[test]
    fn hardened_mode_rejects_non_finite_function_values() {
        let options = RootOptions {
            method: Some(RootMethod::Bisect),
            mode: RuntimeMode::Hardened,
            ..RootOptions::default()
        };
        let err = bisect(
            |x| if x > 0.8 { f64::NAN } else { x - 0.2 },
            (0.1, 0.9),
            options,
        )
        .expect_err("hardened mode should reject non-finite values");
        assert!(matches!(err, crate::OptError::NonFiniteInput { .. }));
        push_test_log(
            "root-hardened-non-finite",
            "bisect",
            "nan-branch",
            RuntimeMode::Hardened,
            false,
            0,
            None,
            316,
        );
    }

    #[test]
    fn strict_mode_non_finite_function_values_map_to_invalid_argument() {
        let options = RootOptions {
            method: Some(RootMethod::Bisect),
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let err = bisect(
            |x| if x > 0.8 { f64::NAN } else { x - 0.2 },
            (0.1, 0.9),
            options,
        )
        .expect_err("strict mode should map non-finite values to invalid argument");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
        push_test_log(
            "root-strict-non-finite",
            "bisect",
            "nan-branch",
            RuntimeMode::Strict,
            false,
            0,
            None,
            317,
        );
    }

    #[test]
    fn bisect_reports_max_iterations_when_budget_is_too_low() {
        let options = RootOptions {
            method: Some(RootMethod::Bisect),
            xtol: 1.0e-18,
            rtol: 1.0e-18,
            maxiter: 1,
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let result = bisect(|x| x - 0.3, (0.0, 1.0), options).expect("bisect executes");
        assert!(!result.converged);
        assert_eq!(result.flag, "convergence error");
        assert_eq!(result.status, ConvergenceStatus::MaxIterations);
        push_test_log(
            "bisect-maxiter-budget",
            "bisect",
            "linear",
            RuntimeMode::Strict,
            result.converged,
            result.function_calls,
            Some(result.root - 0.3),
            318,
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 500,
            failure_persistence: None,
            .. ProptestConfig::default()
        })]

        #[test]
        fn property_brentq_returns_small_residual_for_valid_bracket(
            target in -50.0f64..50.0f64,
        ) {
            let f = |x: f64| x - target;
            let options = RootOptions {
                method: Some(RootMethod::Brentq),
                xtol: 1.0e-10,
                rtol: 1.0e-10,
                maxiter: 80,
                mode: RuntimeMode::Strict,
                ..RootOptions::default()
            };
            let bracket = (target - 1.0, target + 1.0);
            let result = brentq(f, bracket, options).expect("brentq executes");
            prop_assert!(result.converged);
            prop_assert!(f(result.root).abs() <= 2.0 * options.xtol);
            push_test_log(
                "property-brentq-small-residual",
                "brentq",
                "linear-shift",
                RuntimeMode::Strict,
                result.converged,
                result.function_calls,
                Some(f(result.root)),
                401,
            );
        }
    }

    // ── Multivariate root tests ────────────────────────────────────

    #[test]
    fn fsolve_2x2_circle_intersection() {
        // x^2 + y^2 = 1, x - y = 0 → (1/√2, 1/√2)
        let f = |x: &[f64]| vec![x[0] * x[0] + x[1] * x[1] - 1.0, x[0] - x[1]];
        let result = fsolve(f, &[0.5, 0.5]).expect("fsolve should converge");
        assert!(result.converged, "fsolve failed: {}", result.message);
        let s2 = std::f64::consts::FRAC_1_SQRT_2;
        assert!(
            (result.x[0] - s2).abs() < 0.01,
            "x = {}, expected {}",
            result.x[0],
            s2
        );
        assert!(
            (result.x[1] - s2).abs() < 0.01,
            "y = {}, expected {}",
            result.x[1],
            s2
        );
    }

    #[test]
    fn fsolve_linear_system() {
        // 2x + y = 5, x - y = 1 → x=2, y=1
        let f = |x: &[f64]| vec![2.0 * x[0] + x[1] - 5.0, x[0] - x[1] - 1.0];
        let result = fsolve(f, &[0.0, 0.0]).expect("fsolve should converge");
        assert!(result.converged, "fsolve failed: {}", result.message);
        assert!(
            (result.x[0] - 2.0).abs() < 0.01,
            "x = {}, expected 2.0",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 0.01,
            "y = {}, expected 1.0",
            result.x[1]
        );
    }

    #[test]
    fn fsolve_3x3_system() {
        // x + y + z = 6, x - y + 2z = 5, 2x + y - z = 1 → x=1, y=2, z=3
        let f = |x: &[f64]| {
            vec![
                x[0] + x[1] + x[2] - 6.0,
                x[0] - x[1] + 2.0 * x[2] - 5.0,
                2.0 * x[0] + x[1] - x[2] - 1.0,
            ]
        };
        let result = fsolve(f, &[0.0, 0.0, 0.0]).expect("fsolve should converge");
        assert!(result.converged, "fsolve failed: {}", result.message);
        assert!((result.x[0] - 1.0).abs() < 0.1, "x = {}", result.x[0]);
        assert!((result.x[1] - 2.0).abs() < 0.1, "y = {}", result.x[1]);
        assert!((result.x[2] - 3.0).abs() < 0.1, "z = {}", result.x[2]);
    }

    #[test]
    fn fsolve_broyden_cuts_function_calls() {
        // A larger nonlinear system: F_i(x) = x_i^2 - (i+1), root x_i = sqrt(i+1).
        // Broyden reuses the Jacobian via rank-1 updates instead of rebuilding the
        // full finite-difference Jacobian (n residual evals) every iteration, so the
        // total residual evaluations stay well under `iterations * n` (which a
        // full-FD-every-iteration solver would exceed).
        let n = 12usize;
        let f = |x: &[f64]| {
            (0..x.len())
                .map(|i| x[i] * x[i] - (i as f64 + 1.0))
                .collect()
        };
        let result = fsolve(f, &vec![0.5; n]).expect("fsolve should converge");
        assert!(result.converged, "fsolve failed: {}", result.message);
        for i in 0..n {
            let want = ((i + 1) as f64).sqrt();
            assert!(
                (result.x[i] - want).abs() < 1e-4,
                "x[{i}]={}, want {want}",
                result.x[i]
            );
        }
        // Full-FD-every-iteration would need >= iterations*n Jacobian evals alone;
        // Broyden's bound is one FD Jacobian plus O(1) evals per iteration.
        assert!(
            result.function_calls <= 2 * n + 4 * result.iterations.max(1),
            "function_calls={} too high for {} iterations (n={n})",
            result.function_calls,
            result.iterations
        );
    }

    #[test]
    fn fsolve_near_solution_fast() {
        let f = |x: &[f64]| vec![x[0] * x[0] - 4.0, x[1] * x[1] - 9.0];
        let result = fsolve(f, &[1.9, 2.9]).expect("fsolve should converge");
        assert!(result.converged);
        assert!(
            result.iterations < 20,
            "too many iterations: {}",
            result.iterations
        );
    }

    #[test]
    fn fsolve_residual_is_small() {
        let f = |x: &[f64]| vec![x[0].sin() + x[1] - 1.0, x[0] + x[1].cos() - 1.0];
        let result = fsolve(f, &[0.5, 0.5]).expect("fsolve should converge");
        if result.converged {
            let residual = f(&result.x);
            let max_residual = residual
                .iter()
                .map(|r| r.abs())
                .fold(0.0_f64, |a: f64, b: f64| {
                    if a.is_nan() || b.is_nan() {
                        f64::NAN
                    } else {
                        a.max(b)
                    }
                });
            assert!(max_residual < 1e-6, "residual too large: {max_residual}");
        }
    }

    #[test]
    fn fsolve_rejects_nonfinite_x0() {
        let f = |x: &[f64]| vec![x[0]];
        let err = fsolve(f, &[f64::NAN]).expect_err("non-finite x0 must fail closed");
        assert!(matches!(err, crate::OptError::NonFiniteInput { .. }));
    }

    #[test]
    fn multivariate_roots_reject_wrong_residual_length() {
        let wrong_len = |_x: &[f64]| vec![1.0];

        let err = fsolve(wrong_len, &[0.0, 0.0]).expect_err("fsolve shape mismatch");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));

        let err = broyden1(wrong_len, &[0.0, 0.0], 1e-10, 10).expect_err("broyden1 shape mismatch");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));

        let err = broyden2(wrong_len, &[0.0, 0.0], 1e-10, 10).expect_err("broyden2 shape mismatch");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));

        let err = anderson(wrong_len, &[0.0, 0.0], 1e-10, 10, 2, 1.0)
            .expect_err("anderson shape mismatch");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));

        let err = lm_root(wrong_len, &[0.0, 0.0], 1e-10, 10).expect_err("lm shape mismatch");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    // ── root dispatcher tests ──────────────────────────────────────

    #[test]
    fn root_hybr_matches_fsolve() {
        let f = |x: &[f64]| vec![2.0 * x[0] + x[1] - 5.0, x[0] - x[1] - 1.0];
        let direct = fsolve(f, &[0.0, 0.0]).expect("fsolve");
        let dispatched = root(f, &[0.0, 0.0], MultivariateRootOptions::default()).expect("root");
        assert!(dispatched.converged);
        assert!((direct.x[0] - dispatched.x[0]).abs() < 1e-10);
        assert!((direct.x[1] - dispatched.x[1]).abs() < 1e-10);
    }

    #[test]
    fn root_hybr_handles_singular_start_jacobian() {
        // x0+x1+x2=3, x0²+x1²-x2=1, x0·x1·x2=1; root at [1,1,1]. At the start
        // [1.5,1.5,0.5] the Jacobian has two identical columns (exactly singular).
        // Plain Newton gave up here ("singular Jacobian"); the LM-damped fallback
        // escapes, matching scipy's hybr.
        let f = |x: &[f64]| {
            vec![
                x[0] + x[1] + x[2] - 3.0,
                x[0] * x[0] + x[1] * x[1] - x[2] - 1.0,
                x[0] * x[1] * x[2] - 1.0,
            ]
        };
        let r = root(f, &[1.5, 1.5, 0.5], MultivariateRootOptions::default()).expect("root");
        assert!(
            r.converged,
            "hybr should converge from a singular start: {}",
            r.message
        );
        for (i, &v) in r.x.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-6, "x[{i}]={v}, expected 1");
        }
    }

    #[test]
    fn brentq_many_byte_identical_to_per_param() {
        // 1-D root sweep: solve x^3 - p = 0 (cube root of p) over a shared bracket for many p.
        // The batched solve must equal looping brentq per parameter, bit-for-bit.
        let f = |x: f64, p: &[f64]| x * x * x - p[0];
        let bracket = (0.0, 10.0);
        let opts = RootOptions::default();
        let mut s = 9u64;
        let mut rng = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            1.0 + 900.0 * ((s >> 11) as f64 / (1u64 << 53) as f64)
        };
        let nrows = 16usize; // crosses the serial->parallel gate
        let params: Vec<Vec<f64>> = (0..nrows).map(|_| vec![rng()]).collect();

        let batched = brentq_many(f, bracket, &params, opts);
        assert_eq!(batched.len(), nrows);
        for (i, p) in params.iter().enumerate() {
            let single = brentq(|x| f(x, p), bracket, opts).expect("single");
            let many = batched[i].as_ref().expect("batched member");
            assert_eq!(
                many.root.to_bits(),
                single.root.to_bits(),
                "root mismatch param {i}"
            );
            assert_eq!(
                many.converged, single.converged,
                "converged mismatch param {i}"
            );
        }
        assert!(
            batched
                .iter()
                .filter(|r| r.as_ref().map(|x| x.converged).unwrap_or(false))
                .count()
                == nrows
        );
        assert!(brentq_many(f, bracket, &[], opts).is_empty());
    }

    #[test]
    fn newton_many_byte_identical_to_per_param() {
        // Newton sweep: solve x^3 - p = 0 with fprime = 3x^2 over many p. The batched
        // solve must equal looping newton per parameter, bit-for-bit. nrows crosses the
        // work-capped serial->parallel gate (>= 2*2048) so the parallel path is exercised.
        let f = |x: f64, p: &[f64]| x * x * x - p[0];
        let fp = |x: f64, _p: &[f64]| 3.0 * x * x;
        let mut s = 11u64;
        let mut rng = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            1.0 + 900.0 * ((s >> 11) as f64 / (1u64 << 53) as f64)
        };
        let nrows = 5000usize;
        let params: Vec<Vec<f64>> = (0..nrows).map(|_| vec![rng()]).collect();

        let batched = newton_many(f, fp, 1.5, &params, 1e-12, 0.0, 50);
        assert_eq!(batched.len(), nrows);
        for (i, p) in params.iter().enumerate() {
            let single = newton(
                |x| f(x, p),
                1.5,
                Some(&|x| fp(x, p)),
                None,
                None,
                1e-12,
                0.0,
                50,
            )
            .expect("single");
            let many = batched[i].as_ref().expect("batched member");
            assert_eq!(
                many.root.to_bits(),
                single.root.to_bits(),
                "root mismatch {i}"
            );
            // root should be the real cube root of p.
            assert!(
                (many.root - p[0].cbrt()).abs() < 1e-9,
                "root {} vs {}",
                many.root,
                p[0].cbrt()
            );
        }
        assert!(newton_many(f, fp, 1.5, &[], 1e-12, 0.0, 50).is_empty());
    }

    #[test]
    fn secant_many_byte_identical_to_per_param() {
        // Derivative-free secant sweep: x^3 - p = 0 over many p, nrows crossing the gate.
        let f = |x: f64, p: &[f64]| x * x * x - p[0];
        let opts = RootOptions::default();
        let mut s = 13u64;
        let mut rng = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            1.0 + 900.0 * ((s >> 11) as f64 / (1u64 << 53) as f64)
        };
        let nrows = 5000usize;
        let params: Vec<Vec<f64>> = (0..nrows).map(|_| vec![rng()]).collect();
        let batched = secant_many(f, 1.5, None, &params, opts);
        assert_eq!(batched.len(), nrows);
        for (i, p) in params.iter().enumerate() {
            let single = secant(|x| f(x, p), 1.5, None, opts).expect("single");
            let many = batched[i].as_ref().expect("batched member");
            assert_eq!(
                many.root.to_bits(),
                single.root.to_bits(),
                "root mismatch {i}"
            );
        }
        assert!(secant_many(f, 1.5, None, &[], opts).is_empty());
    }

    #[test]
    fn root_many_byte_identical_to_per_param() {
        // Parameter sweep over a well-conditioned 3-equation system (Jacobian at [1,1,1] is
        // the SPD [[2,1,1],[1,2,1],[1,1,2]]). The batched solve must equal looping root per
        // parameter, bit-for-bit, and these parameters converge to a genuine root.
        let sys = |x: &[f64], p: &[f64]| {
            vec![
                x[0] * x[0] + x[1] + x[2] - p[0],
                x[0] + x[1] * x[1] + x[2] - p[1],
                x[0] + x[1] + x[2] * x[2] - p[2],
            ]
        };
        let mut s = 5u64;
        let mut rng = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            2.5 + 2.0 * ((s >> 11) as f64 / (1u64 << 53) as f64)
        };
        let nrows = 12usize; // crosses the serial->parallel gate
        let params: Vec<Vec<f64>> = (0..nrows).map(|_| vec![rng(), rng(), rng()]).collect();
        let x0 = [1.0, 1.0, 1.0];
        let opts = MultivariateRootOptions::default();

        let batched = root_many(&sys, &x0, &params, opts.clone());
        assert_eq!(batched.len(), nrows);
        for (i, p) in params.iter().enumerate() {
            let single = root(|x: &[f64]| sys(x, p), &x0, opts.clone()).expect("single");
            let many = batched[i].as_ref().expect("batched member");
            assert_eq!(many.x.len(), single.x.len());
            for (a, b) in many.x.iter().zip(single.x.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "x mismatch param {i}");
            }
            for (a, b) in many.fun.iter().zip(single.fun.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "fun mismatch param {i}");
            }
        }
        // The sweep exercises genuine converging solves, not just identical failures.
        let n_conv = batched
            .iter()
            .filter(|r| r.as_ref().map(|x| x.converged).unwrap_or(false))
            .count();
        assert!(
            n_conv >= nrows / 2,
            "expected most to converge, got {n_conv}/{nrows}"
        );
        assert!(root_many(&sys, &x0, &[], opts).is_empty());
    }

    #[test]
    fn root_hybr_rejects_invalid_options() {
        let f = |x: &[f64]| vec![x[0]];
        let options = MultivariateRootOptions {
            method: MultivariateRootMethod::Hybr,
            tol: 0.0,
            max_iter: 200,
        };
        let err = root(f, &[1.0], options).expect_err("invalid hybr tolerance");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn root_hybr_respects_iteration_budget() {
        let f = |x: &[f64]| vec![x[0] * x[0] - 2.0];
        let options = MultivariateRootOptions {
            method: MultivariateRootMethod::Hybr,
            tol: 1e-30,
            max_iter: 1,
        };
        let result = root(f, &[1.0], options).expect("hybr executes");
        assert!(!result.converged);
        assert_eq!(result.iterations, 1);
    }

    #[test]
    fn root_broyden1_linear_system() {
        let f = |x: &[f64]| vec![2.0 * x[0] + x[1] - 5.0, x[0] - x[1] - 1.0];
        let options = MultivariateRootOptions {
            method: MultivariateRootMethod::Broyden1,
            ..MultivariateRootOptions::default()
        };
        let result = root(f, &[0.0, 0.0], options).expect("root broyden1");
        assert!(result.converged, "broyden1 failed: {}", result.message);
        assert!(
            (result.x[0] - 2.0).abs() < 0.05,
            "x = {}, expected 2.0",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 0.05,
            "y = {}, expected 1.0",
            result.x[1]
        );
    }

    #[test]
    fn broyden1_nonlinear_circle() {
        // x² + y² = 1, x - y = 0 → x=y=1/√2
        let f = |x: &[f64]| vec![x[0] * x[0] + x[1] * x[1] - 1.0, x[0] - x[1]];
        let result = broyden1(f, &[0.5, 0.5], 1e-10, 200).expect("broyden1");
        assert!(result.converged, "broyden1 failed: {}", result.message);
        let s2 = std::f64::consts::FRAC_1_SQRT_2;
        assert!(
            (result.x[0] - s2).abs() < 0.01,
            "x = {}, expected {}",
            result.x[0],
            s2
        );
    }

    #[test]
    fn broyden1_3d_system() {
        let f = |x: &[f64]| {
            vec![
                x[0] + x[1] + x[2] - 6.0,
                x[0] - x[1] + 2.0 * x[2] - 5.0,
                2.0 * x[0] + x[1] - x[2] - 1.0,
            ]
        };
        let result = broyden1(f, &[0.0, 0.0, 0.0], 1e-10, 200).expect("broyden1 3d");
        assert!(result.converged, "broyden1 3d failed: {}", result.message);
        assert!((result.x[0] - 1.0).abs() < 0.1, "x = {}", result.x[0]);
        assert!((result.x[1] - 2.0).abs() < 0.1, "y = {}", result.x[1]);
        assert!((result.x[2] - 3.0).abs() < 0.1, "z = {}", result.x[2]);
    }

    #[test]
    fn broyden1_empty_x0_rejected() {
        let f = |_x: &[f64]| vec![];
        let err = broyden1(f, &[], 1e-10, 200).expect_err("empty");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn broyden2_nonlinear_circle() {
        // x² + y² = 1, x - y = 0 → x=y=1/√2
        let f = |x: &[f64]| vec![x[0] * x[0] + x[1] * x[1] - 1.0, x[0] - x[1]];
        let result = broyden2(f, &[0.5, 0.5], 1e-10, 200).expect("broyden2");
        assert!(result.converged, "broyden2 failed: {}", result.message);
        let s2 = std::f64::consts::FRAC_1_SQRT_2;
        assert!(
            (result.x[0] - s2).abs() < 0.01,
            "x = {}, expected {}",
            result.x[0],
            s2
        );
    }

    #[test]
    fn broyden2_3d_system() {
        let f = |x: &[f64]| {
            vec![
                x[0] + x[1] + x[2] - 6.0,
                x[0] - x[1] + 2.0 * x[2] - 5.0,
                2.0 * x[0] + x[1] - x[2] - 1.0,
            ]
        };
        let result = broyden2(f, &[0.0, 0.0, 0.0], 1e-10, 200).expect("broyden2 3d");
        assert!(result.converged, "broyden2 3d failed: {}", result.message);
        assert!((result.x[0] - 1.0).abs() < 0.1, "x = {}", result.x[0]);
        assert!((result.x[1] - 2.0).abs() < 0.1, "y = {}", result.x[1]);
        assert!((result.x[2] - 3.0).abs() < 0.1, "z = {}", result.x[2]);
    }

    #[test]
    fn broyden2_empty_x0_rejected() {
        let f = |_x: &[f64]| vec![];
        let err = broyden2(f, &[], 1e-10, 200).expect_err("empty");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn anderson_linear_system() {
        // Simple linear system: x + y = 3, x - y = 1 → x=2, y=1
        let f = |x: &[f64]| vec![x[0] + x[1] - 3.0, x[0] - x[1] - 1.0];
        let result = anderson(f, &[0.0, 0.0], 1e-8, 100, 5, 0.5).expect("anderson");
        assert!(result.converged, "anderson failed: {}", result.message);
        assert!(
            (result.x[0] - 2.0).abs() < 0.1,
            "x = {}, expected 2.0",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 0.1,
            "y = {}, expected 1.0",
            result.x[1]
        );
    }

    #[test]
    fn anderson_nonlinear() {
        // x² + y² = 1, x - y = 0 → x=y=±1/√2
        let f = |x: &[f64]| vec![x[0] * x[0] + x[1] * x[1] - 1.0, x[0] - x[1]];
        let result = anderson(f, &[0.5, 0.5], 1e-8, 200, 5, 0.3).expect("anderson nonlinear");
        assert!(result.converged, "anderson failed: {}", result.message);
        let s2 = std::f64::consts::FRAC_1_SQRT_2;
        // Either root is valid
        assert!(
            (result.x[0].abs() - s2).abs() < 0.05,
            "|x| = {}, expected {}",
            result.x[0].abs(),
            s2
        );
        // x should equal y (up to sign)
        assert!(
            (result.x[0] - result.x[1]).abs() < 0.01,
            "x={}, y={} should be equal",
            result.x[0],
            result.x[1]
        );
    }

    #[test]
    fn anderson_does_not_diverge_on_ill_conditioned_system() {
        // x²+y²=4, x·y=1 → root [1.93185, 0.51764]. Without Tikhonov (w0)
        // regularization of the least-squares mixing, the coefficients blew up and
        // the iteration diverged to ~[-2.7e5, -5.9e4]. The w0 term keeps it finite
        // and convergent.
        let f = |x: &[f64]| vec![x[0] * x[0] + x[1] * x[1] - 4.0, x[0] * x[1] - 1.0];
        let result = anderson(f, &[2.0, 0.5], 1e-6, 500, 5, 1.0).expect("anderson");
        assert!(
            result.converged,
            "anderson should converge: {}",
            result.message
        );
        assert!(
            (result.x[0] - 1.931_851_652_578).abs() < 1e-4
                && (result.x[1] - 0.517_638_090_205).abs() < 1e-4,
            "anderson x={:?}, expected ~[1.93185, 0.51764]",
            result.x
        );
    }

    #[test]
    fn anderson_empty_x0_rejected() {
        let f = |_x: &[f64]| vec![];
        let err = anderson(f, &[], 1e-10, 200, 5, 1.0).expect_err("empty");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn anderson_rejects_invalid_mixing_controls() {
        let f = |x: &[f64]| vec![x[0]];

        let err = anderson(f, &[1.0], 1e-10, 200, 0, 1.0).expect_err("zero memory");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));

        for beta in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = anderson(f, &[1.0], 1e-10, 200, 5, beta).expect_err("non-finite beta");
            assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
        }
    }

    #[test]
    fn toms748_finds_sqrt2() {
        let f = |x: f64| x * x - 2.0;
        let result = toms748(f, (0.0, 2.0), RootOptions::default()).expect("toms748");
        assert!(result.converged, "toms748 failed: {}", result.message);
        assert!(
            (result.root - std::f64::consts::SQRT_2).abs() < 1e-10,
            "root = {}, expected sqrt(2)",
            result.root
        );
    }

    #[test]
    fn toms748_rejects_non_finite_function_values() {
        let f = |x: f64| if x > 0.8 { f64::NAN } else { x - 0.2 };

        let hardened = RootOptions {
            mode: RuntimeMode::Hardened,
            ..RootOptions::default()
        };
        let err = toms748(f, (0.1, 0.9), hardened)
            .expect_err("hardened mode rejects non-finite callback values");
        assert!(matches!(err, crate::OptError::NonFiniteInput { .. }));

        let strict = RootOptions {
            mode: RuntimeMode::Strict,
            ..RootOptions::default()
        };
        let err = toms748(f, (0.1, 0.9), strict)
            .expect_err("strict mode maps non-finite callback values");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }

    #[test]
    fn newton_scalar_finds_sqrt2() {
        let f = |x: f64| x * x - 2.0;
        let df = |x: f64| 2.0 * x;
        let result = newton_scalar(f, df, 1.0, RootOptions::default()).expect("newton");
        assert!(result.converged, "newton failed: {}", result.message);
        assert!(
            (result.root - std::f64::consts::SQRT_2).abs() < 1e-10,
            "root = {}",
            result.root
        );
    }

    #[test]
    fn secant_finds_sqrt2() {
        let f = |x: f64| x * x - 2.0;
        let result = secant(f, 1.0, Some(2.0), RootOptions::default()).expect("secant");
        assert!(result.converged, "secant failed: {}", result.message);
        assert!(
            (result.root - std::f64::consts::SQRT_2).abs() < 1e-10,
            "root = {}",
            result.root
        );
    }

    #[test]
    fn halley_finds_cbrt2() {
        let f = |x: f64| x * x * x - 2.0;
        let df = |x: f64| 3.0 * x * x;
        let d2f = |x: f64| 6.0 * x;
        let result = halley(f, df, d2f, 1.0, RootOptions::default()).expect("halley");
        assert!(result.converged, "halley failed: {}", result.message);
        assert!(
            (result.root - 2.0f64.cbrt()).abs() < 1e-10,
            "root = {}",
            result.root
        );
    }

    #[test]
    fn root_scalar_secant_auto() {
        let f = |x: f64| x * x - 2.0;
        let result = root_scalar(f, None, Some(1.0), Some(2.0), RootOptions::default())
            .expect("root_scalar secant");
        assert!(result.converged, "failed: {}", result.message);
        assert!(
            (result.root - std::f64::consts::SQRT_2).abs() < 1e-10,
            "root = {}",
            result.root
        );
    }

    // --- Levenberg-Marquardt tests ---

    #[test]
    fn lm_root_linear_system() {
        // x + y = 3, x - y = 1 → x=2, y=1
        let f = |x: &[f64]| vec![x[0] + x[1] - 3.0, x[0] - x[1] - 1.0];
        let result = lm_root(f, &[0.0, 0.0], 1e-8, 200).expect("lm_root linear");
        assert!(result.converged, "lm_root failed: {}", result.message);
        assert!(
            (result.x[0] - 2.0).abs() < 1e-6,
            "x = {}, expected 2.0",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 1e-6,
            "y = {}, expected 1.0",
            result.x[1]
        );
    }

    #[test]
    fn lm_root_rosenbrock_2d() {
        // Rosenbrock residuals: f1 = 1-x, f2 = 10(y-x²) → root at (1,1)
        let f = |x: &[f64]| vec![1.0 - x[0], 10.0 * (x[1] - x[0] * x[0])];
        let result = lm_root(f, &[0.0, 0.0], 1e-8, 500).expect("lm_root rosenbrock");
        assert!(result.converged, "lm_root failed: {}", result.message);
        assert!(
            (result.x[0] - 1.0).abs() < 1e-5,
            "x = {}, expected 1.0",
            result.x[0]
        );
        assert!(
            (result.x[1] - 1.0).abs() < 1e-5,
            "y = {}, expected 1.0",
            result.x[1]
        );
    }

    #[test]
    fn lm_root_via_root_interface() {
        // Test that lm is accessible via the root() dispatcher
        let f = |x: &[f64]| vec![x[0] * x[0] - 2.0];
        let opts = MultivariateRootOptions {
            method: MultivariateRootMethod::Lm,
            tol: 1e-10,
            max_iter: 100,
        };
        let result = root(f, &[1.0], opts).expect("root with lm");
        assert!(result.converged, "root lm failed: {}", result.message);
        assert!(
            (result.x[0].abs() - std::f64::consts::SQRT_2).abs() < 1e-6,
            "x = {}, expected ±sqrt(2)",
            result.x[0]
        );
    }

    #[test]
    fn lm_root_empty_x0_rejected() {
        let f = |_x: &[f64]| vec![];
        let err = lm_root(f, &[], 1e-10, 200).expect_err("empty");
        assert!(matches!(err, crate::OptError::InvalidArgument { .. }));
    }
}
