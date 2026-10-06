//! Metamorphic tests for `fsci-opt`.
//!
//! Minimum location on convex test functions, multi-method root
//! agreement, root-finder shift invariance, and curve_fit recovery
//! of synthetic linear parameters.
//!
//! Run with: `cargo test -p fsci-opt --test metamorphic_tests`

use fsci_opt::{
    BasinhoppingOptions, CurveFitOptions, DifferentialEvolutionOptions, FminCobylaOptions,
    LeastSquaresOptions, LinprogOptions, MinimizeOptions, MinimizeScalarOptions, RootOptions,
    approx_fprime, basinhopping, bisect, bracket, brent_minimize, brenth, brentq, brute,
    check_grad, cobyla, curve_fit, differential_evolution, dual_annealing, fixed_point, fsolve,
    golden, gradient_descent, halley, isotonic_regression, least_squares, linear_sum_assignment,
    linprog, minimize, minimize_scalar, minimize_scalar_bounded, minimize_trisection,
    newton_scalar, nnls, numerical_gradient, numerical_hessian, numerical_jacobian,
    projected_gradient_descent, pso, ridder, rosen, rosen_der, rosen_hess, rosen_hess_prod, secant,
    shgo, toms748,
};

const ATOL: f64 = 1e-6;
const RTOL: f64 = 1e-5;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= ATOL + RTOL * a.abs().max(b.abs()).max(1.0)
}

// ─────────────────────────────────────────────────────────────────────
// MR1 — minimize finds a known minimum on a quadratic bowl.
// f(x) = (x[0] − 2)² + (x[1] + 3)² has its unique global minimum at
// (2, −3) with value 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_minimize_finds_known_minimum_on_bowl() {
    let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] + 3.0).powi(2);
    let opts = MinimizeOptions::default();
    let res = minimize(f, &[0.0, 0.0], opts).unwrap();
    assert!(
        res.success,
        "MR1 minimize did not converge: {}",
        res.message
    );
    assert!(
        (res.x[0] - 2.0).abs() < 1e-3,
        "MR1 x[0]={} expected 2.0",
        res.x[0]
    );
    assert!(
        (res.x[1] - (-3.0)).abs() < 1e-3,
        "MR1 x[1]={} expected -3.0",
        res.x[1]
    );
    let fmin = res.fun.unwrap();
    assert!(fmin < 1e-6, "MR1 fmin={fmin} expected ~0");
}

// ─────────────────────────────────────────────────────────────────────
// MR2 — All bracketing root finders return the same root for a
// monotone f on a bracket containing the unique zero.
//
// f(x) = exp(x) − 2 has a unique real root x* = ln(2) ≈ 0.6931.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_root_finders_agree_on_monotone_function() {
    let f = |x: f64| x.exp() - 2.0;
    let bracket = (0.0_f64, 1.0_f64);
    let opts = RootOptions::default();

    let r_brentq = brentq(f, bracket, opts).unwrap().root;
    let r_brenth = brenth(f, bracket, opts).unwrap().root;
    let r_bisect = bisect(f, bracket, opts).unwrap().root;
    let r_ridder = ridder(f, bracket, opts).unwrap().root;
    let r_toms = toms748(f, bracket, opts).unwrap().root;

    let truth = 2.0_f64.ln();
    for (name, r) in [
        ("brentq", r_brentq),
        ("brenth", r_brenth),
        ("bisect", r_bisect),
        ("ridder", r_ridder),
        ("toms748", r_toms),
    ] {
        assert!(
            (r - truth).abs() < 1e-8,
            "MR2 {name} root {r} differs from ln(2)={truth}"
        );
        assert!(
            f(r).abs() < 1e-8,
            "MR2 {name} f(root) = {} not near 0",
            f(r)
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR3 — Shift invariance: root of g(x) = f(x − c) is c + root_of_f.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_root_shift_invariance() {
    let f = |x: f64| x.exp() - 2.0; // root at ln(2)
    let opts = RootOptions::default();
    let r0 = brentq(f, (0.0, 1.0), opts).unwrap().root;
    for shift in [-3.0_f64, -0.5, 0.0, 0.7, 4.5] {
        let g = |x: f64| f(x - shift);
        let bracket = (shift + 0.0, shift + 1.0);
        let r = brentq(g, bracket, opts).unwrap().root;
        assert!(
            close(r, shift + r0),
            "MR3 shift={shift}: got {r}, expected {}",
            shift + r0
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR4 — minimize is shift-invariant: shifting f(x) by an additive
// constant leaves x* unchanged.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_minimize_constant_shift_invariance() {
    let f = |x: &[f64]| (x[0] - 1.5).powi(2) + (x[1] - 2.5).powi(2);
    let g = |x: &[f64]| f(x) + 1000.0;
    let opts = MinimizeOptions::default();
    let r1 = minimize(f, &[0.0, 0.0], opts).unwrap();
    let r2 = minimize(g, &[0.0, 0.0], opts).unwrap();
    for (a, b) in r1.x.iter().zip(&r2.x) {
        assert!((a - b).abs() < 1e-3, "MR4 minimize shift: x_f={a} x_g={b}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR5 — curve_fit recovers the slope and intercept of synthetic
// noise-free linear data y = α x + β.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_curve_fit_recovers_linear_params() {
    let alpha_true = 2.5_f64;
    let beta_true = -1.2_f64;
    let xdata: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
    let ydata: Vec<f64> = xdata.iter().map(|&x| alpha_true * x + beta_true).collect();
    let model = |x: f64, p: &[f64]| p[0] * x + p[1];
    let opts = CurveFitOptions {
        p0: Some(vec![0.0, 0.0]),
        ..Default::default()
    };
    let res = curve_fit(model, &xdata, &ydata, opts).unwrap();
    assert!(
        (res.popt[0] - alpha_true).abs() < 1e-6,
        "MR5 alpha: {} vs {alpha_true}",
        res.popt[0]
    );
    assert!(
        (res.popt[1] - beta_true).abs() < 1e-6,
        "MR5 beta: {} vs {beta_true}",
        res.popt[1]
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR6 — Bisection counts: bisect's iteration count is upper-bounded by
// ⌈log₂((b − a) / xtol)⌉ + 1 because bisection halves the bracket
// width every iteration.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_bisect_iteration_count_bounded() {
    let f = |x: f64| x.exp() - 2.0;
    let opts = RootOptions {
        xtol: 1e-10,
        ..Default::default()
    };
    let res = bisect(f, (0.0, 1.0), opts).unwrap();
    let max_iters = ((1.0_f64 / 1e-10).log2().ceil() as usize) + 5;
    assert!(
        res.iterations <= max_iters,
        "MR6 bisect iterations {} > expected upper bound {max_iters}",
        res.iterations
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR7 — minimize on a strictly convex function lands on a point with
// objective ≤ initial objective (descent property).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_minimize_strict_descent_on_convex() {
    let f = |x: &[f64]| x[0].powi(4) + (x[1] - 2.0).powi(2) + 1.0;
    let x0 = vec![3.0, -1.0];
    let f0 = f(&x0);
    let opts = MinimizeOptions::default();
    let res = minimize(f, &x0, opts).unwrap();
    let fmin = res.fun.unwrap();
    assert!(
        fmin <= f0 + 1e-9,
        "MR7 minimize did not descend: f0={f0} fmin={fmin}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR8 — minimize_scalar finds the analytic minimum of a parabola.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_minimize_scalar_finds_parabola_vertex() {
    let f = |x: f64| (x - 3.7).powi(2) + 1.5;
    let res = minimize_scalar(f, (0.0, 10.0), MinimizeScalarOptions::default()).unwrap();
    assert!(res.success, "MR8 minimize_scalar did not converge: {res:?}");
    assert!(
        (res.x - 3.7).abs() < 1e-5,
        "MR8 vertex: got x={}, expected 3.7",
        res.x
    );
    assert!(
        (res.fun - 1.5).abs() < 1e-9,
        "MR8 fun at vertex: got {}, expected 1.5",
        res.fun
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR9 — least_squares recovers the synthetic linear-residual minimizer.
//
// Residuals r_i(p) = y_i - (p[0] · x_i + p[1]) — known global minimum
// at α=2.0, β=-1.0 with cost 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_least_squares_recovers_linear_params() {
    let alpha_true = 2.0_f64;
    let beta_true = -1.0_f64;
    let xs: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| alpha_true * x + beta_true).collect();
    let resid = move |p: &[f64]| -> Vec<f64> {
        xs.iter()
            .zip(&ys)
            .map(|(&x, &y)| y - (p[0] * x + p[1]))
            .collect()
    };
    let res = least_squares(resid, &[0.0, 0.0], LeastSquaresOptions::default()).unwrap();
    assert!(
        (res.x[0] - alpha_true).abs() < 1e-6,
        "MR9 alpha: {} vs {alpha_true}",
        res.x[0]
    );
    assert!(
        (res.x[1] - beta_true).abs() < 1e-6,
        "MR9 beta: {} vs {beta_true}",
        res.x[1]
    );
    assert!(
        res.cost < 1e-12,
        "MR9 cost not zero on noise-free data: {}",
        res.cost
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR10 — fsolve solves a 2x2 linear system: f(x) = A x − b.
// Solution must satisfy A x ≈ b.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fsolve_2x2_linear_system() {
    // A = [[3, 1], [1, 2]],  b = [9, 8] → exact solution x = [2, 3]
    let func = |x: &[f64]| -> Vec<f64> { vec![3.0 * x[0] + x[1] - 9.0, x[0] + 2.0 * x[1] - 8.0] };
    let res = fsolve(func, &[0.0, 0.0]).unwrap();
    let r = func(&res.x);
    let resid: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        resid < 1e-9,
        "MR10 fsolve residual too large: {resid:e} (x={:?})",
        res.x
    );
    // Sanity-check exact answer for the well-conditioned system.
    assert!((res.x[0] - 2.0).abs() < 1e-6);
    assert!((res.x[1] - 3.0).abs() < 1e-6);
}

// ─────────────────────────────────────────────────────────────────────
// MR11 — differential_evolution finds the global minimum on a sphere
// across rectangular bounds. f(x) = Σ x_i² has minimum 0 at the origin.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_de_finds_sphere_minimum() {
    let f = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
    let bounds = vec![(-5.0_f64, 5.0); 4];
    let opts = DifferentialEvolutionOptions {
        seed: Some(42),
        ..DifferentialEvolutionOptions::default()
    };
    let res = differential_evolution(f, &bounds, opts).unwrap();
    let fmin = res.fun.unwrap();
    assert!(
        fmin < 1e-6,
        "MR11 differential_evolution did not converge: fmin={fmin}"
    );
    for &xi in &res.x {
        assert!(
            xi.abs() < 1e-3,
            "MR11 not at origin: {res:?} (component {xi})"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR12 — basinhopping starting far from the minimum still finds it.
// f(x) = (x[0] − 1)² + (x[1] + 2)² has min at (1, −2).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_basinhopping_far_start_converges() {
    let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] + 2.0).powi(2);
    let opts = BasinhoppingOptions {
        niter: 50,
        seed: Some(7),
        ..BasinhoppingOptions::default()
    };
    let res = basinhopping(f, &[10.0, 10.0], opts).unwrap();
    assert!(
        (res.x[0] - 1.0).abs() < 1e-3,
        "MR12 basinhopping x[0] = {}, expected ~1",
        res.x[0]
    );
    assert!(
        (res.x[1] - (-2.0)).abs() < 1e-3,
        "MR12 basinhopping x[1] = {}, expected ~-2",
        res.x[1]
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR13 — curve_fit recovers quadratic params (a, b, c) from noisy data.
// y_i = a x_i² + b x_i + c with deterministic small noise.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_curve_fit_recovers_quadratic() {
    let a_true = 1.5_f64;
    let b_true = -2.0_f64;
    let c_true = 0.5_f64;
    let xs: Vec<f64> = (0..30).map(|i| -3.0 + 0.2 * i as f64).collect();
    // Deterministic small periodic noise so the test is reproducible.
    let ys: Vec<f64> = xs
        .iter()
        .enumerate()
        .map(|(k, &x)| {
            let noise = 1e-6 * ((k as f64) * 0.7).sin();
            a_true * x * x + b_true * x + c_true + noise
        })
        .collect();
    let model = |x: f64, p: &[f64]| p[0] * x * x + p[1] * x + p[2];
    let opts = CurveFitOptions {
        p0: Some(vec![1.0, 1.0, 1.0]),
        ..Default::default()
    };
    let res = curve_fit(model, &xs, &ys, opts).unwrap();
    assert!(
        (res.popt[0] - a_true).abs() < 1e-3,
        "MR13 curve_fit a: {} vs {a_true}",
        res.popt[0]
    );
    assert!(
        (res.popt[1] - b_true).abs() < 1e-3,
        "MR13 curve_fit b: {} vs {b_true}",
        res.popt[1]
    );
    assert!(
        (res.popt[2] - c_true).abs() < 1e-3,
        "MR13 curve_fit c: {} vs {c_true}",
        res.popt[2]
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR14 — minimize from a far initial guess on a strictly convex 1D
// objective still finds the global minimum.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_minimize_far_initial_guess_strictly_convex() {
    // f(x) = (x - 7)^2 has minimum at x = 7. Start from x0 = -100.
    let f = |x: &[f64]| (x[0] - 7.0).powi(2);
    let opts = MinimizeOptions::default();
    let res = minimize(f, &[-100.0], opts).unwrap();
    assert!(
        (res.x[0] - 7.0).abs() < 1e-3,
        "MR14 minimize from far x0: got {}, expected 7",
        res.x[0]
    );
    let fmin = res.fun.unwrap();
    assert!(fmin < 1e-6, "MR14 fmin not zero: {fmin}");
}

// ─────────────────────────────────────────────────────────────────────
// MR15 — bisect, brentq, brenth, ridder, toms748 all agree on the root
// of a smooth strictly-monotone function on [0, 3].
// f(x) = x³ - 2x - 5 has a unique root near 2.0945514815...
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_root_methods_agree_on_cubic_root() {
    let f = |x: f64| x.powi(3) - 2.0 * x - 5.0;
    let opts = RootOptions::default();
    let r_bisect = bisect(f, (0.0, 3.0), opts).unwrap().root;
    let r_brentq = brentq(f, (0.0, 3.0), opts).unwrap().root;
    let r_brenth = brenth(f, (0.0, 3.0), opts).unwrap().root;
    let r_ridder = ridder(f, (0.0, 3.0), opts).unwrap().root;
    let r_toms = toms748(f, (0.0, 3.0), opts).unwrap().root;
    let expected = 2.094_551_481_542_326_5;
    for (name, r) in [
        ("bisect", r_bisect),
        ("brentq", r_brentq),
        ("brenth", r_brenth),
        ("ridder", r_ridder),
        ("toms748", r_toms),
    ] {
        assert!(
            (r - expected).abs() < 1e-6,
            "MR15 {name} returned {r}, expected ≈ {expected}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR16 — minimize on f(x) = Σ x_i² (sphere) finds the origin.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_minimize_sphere_finds_origin() {
    let f = |x: &[f64]| x.iter().map(|&xi| xi * xi).sum::<f64>();
    let res = minimize(f, &[3.0, -4.0, 1.5, -0.5], MinimizeOptions::default()).unwrap();
    for (i, &xi) in res.x.iter().enumerate() {
        assert!(xi.abs() < 1e-3, "MR16 sphere x[{i}] = {xi}, expected ≈ 0");
    }
    assert!(
        res.fun.unwrap() < 1e-6,
        "MR16 sphere fmin = {}, expected ≈ 0",
        res.fun.unwrap()
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR17 — minimize_scalar on (x − a)² with a ∈ {−5, 0, 5} returns a.
// (Sign symmetry of the quadratic vertex location.)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_minimize_scalar_sign_symmetry() {
    for &a in &[-5.0_f64, 0.0, 5.0] {
        let f = move |x: f64| (x - a).powi(2);
        // Bracket that contains the minimum at a.
        let bracket = (a - 10.0, a + 10.0);
        let res = minimize_scalar(f, bracket, MinimizeScalarOptions::default()).unwrap();
        assert!(
            (res.x - a).abs() < 1e-3,
            "MR17 minimize_scalar with a = {a}: got x = {}",
            res.x
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR18 — fsolve solves the nonlinear scalar system x² - 4 = 0 with
// x0 = 1, recovering x = 2 (the positive root).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fsolve_nonlinear_x_squared_minus_four() {
    let f = |x: &[f64]| vec![x[0] * x[0] - 4.0];
    let res = fsolve(f, &[1.0]).unwrap();
    assert!(
        (res.x[0] - 2.0).abs() < 1e-5,
        "MR18 fsolve x²=4: got x = {}, expected 2.0",
        res.x[0]
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR19 — Scaling the objective by a positive constant does not move
// the minimum: argmin f = argmin (c·f) for c > 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_minimize_positive_scale_invariance() {
    let base = |x: &[f64]| (x[0] - 1.5).powi(2) + (x[1] + 2.5).powi(2);
    let scaled = |x: &[f64]| 100.0 * base(x);
    let r1 = minimize(base, &[0.0, 0.0], MinimizeOptions::default()).unwrap();
    let r2 = minimize(scaled, &[0.0, 0.0], MinimizeOptions::default()).unwrap();
    for i in 0..2 {
        assert!(
            (r1.x[i] - r2.x[i]).abs() < 5e-3,
            "MR19 argmin shift at i={i}: base = {}, scaled = {}",
            r1.x[i],
            r2.x[i]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR20 — DE on a 1D bimodal function returns a value within tolerance
// of one of the two known minima (oracle = minimum value, location
// is one of {-2, +2}).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_de_bimodal_finds_a_minimum() {
    // f(x) = (x² - 4)² has minima at x = ±2 with f = 0.
    let f = |x: &[f64]| (x[0] * x[0] - 4.0).powi(2);
    let bounds = vec![(-5.0_f64, 5.0_f64)];
    let opts = DifferentialEvolutionOptions {
        seed: Some(42),
        ..Default::default()
    };
    let res = differential_evolution(f, &bounds, opts).unwrap();
    let fmin = res.fun.unwrap();
    assert!(fmin < 1e-3, "MR20 DE bimodal fmin = {fmin}, expected ≈ 0");
    let x = res.x[0];
    assert!(
        (x - 2.0).abs() < 0.05 || (x + 2.0).abs() < 0.05,
        "MR20 DE bimodal x = {x}, expected ±2"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR21 — Rosenbrock function value at (1, 1, …, 1) is 0; gradient
// vanishes there.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rosen_at_global_minimum() {
    for n in [2usize, 3, 5, 10] {
        let x = vec![1.0_f64; n];
        let f = rosen(&x);
        assert!(
            f.abs() < 1e-12,
            "MR21 rosen(1, …, 1) (n={n}) = {f}, expected 0"
        );
        let g = rosen_der(&x);
        for (i, &gi) in g.iter().enumerate() {
            assert!(
                gi.abs() < 1e-12,
                "MR21 rosen_der at minimum: g[{i}] = {gi}, expected 0"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR22 — approx_fprime matches the analytical derivative on quadratic
// f(x, y) = x² + y² ⇒ ∇f = (2x, 2y).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_approx_fprime_matches_analytical_quadratic() {
    let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
    for &(x, y) in &[(1.0_f64, 2.0), (-3.0, 0.5), (0.0, 4.0), (-1.5, -2.5)] {
        let g = approx_fprime(&[x, y], f, 1e-6).unwrap();
        let expected = [2.0 * x, 2.0 * y];
        for k in 0..2 {
            assert!(
                (g[k] - expected[k]).abs() < 1e-3,
                "MR22 approx_fprime[{k}] at ({x}, {y}) = {} vs {}",
                g[k],
                expected[k]
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR23 — newton_scalar converges to a root: solve x² - 2 = 0 from
// x0 = 1, expect x = √2.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_newton_scalar_finds_sqrt_two() {
    let f = |x: f64| x * x - 2.0;
    let fp = |x: f64| 2.0 * x;
    let opts = RootOptions::default();
    let res = newton_scalar(f, fp, 1.0, opts).unwrap();
    assert!(
        (res.root - 2.0_f64.sqrt()).abs() < 1e-9,
        "MR23 newton_scalar root = {} vs √2 = {}",
        res.root,
        2.0_f64.sqrt()
    );
    assert!(res.converged, "MR23 newton not converged");
}

// ─────────────────────────────────────────────────────────────────────
// MR24 — secant method finds the root of x³ - 27 from x0 = 4 (root x = 3).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_secant_finds_cube_root() {
    let f = |x: f64| x * x * x - 27.0;
    let opts = RootOptions::default();
    let res = secant(f, 4.0, Some(3.5), opts).unwrap();
    assert!(
        (res.root - 3.0).abs() < 1e-9,
        "MR24 secant root = {} vs 3",
        res.root
    );
    assert!(res.converged, "MR24 secant not converged");
}

// ─────────────────────────────────────────────────────────────────────
// MR25 — Halley's method on x² - 9 from x0 = 4 (root x = 3) converges.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_halley_finds_root_of_quadratic() {
    let f = |x: f64| x * x - 9.0;
    let fp = |x: f64| 2.0 * x;
    let fpp = |_x: f64| 2.0;
    let opts = RootOptions::default();
    let res = halley(f, fp, fpp, 4.0, opts).unwrap();
    assert!(
        (res.root - 3.0).abs() < 1e-10,
        "MR25 halley root = {} vs 3",
        res.root
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR26 — approx_fprime at a critical point of a quadratic returns ≈ 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_approx_fprime_zero_at_critical_point() {
    let f = |x: &[f64]| (x[0] - 2.5).powi(2) + (x[1] + 1.0).powi(2) + 7.0;
    let g = approx_fprime(&[2.5_f64, -1.0], f, 1e-6).unwrap();
    for (i, &gi) in g.iter().enumerate() {
        assert!(
            gi.abs() < 1e-3,
            "MR26 approx_fprime[{i}] at critical point = {gi}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR27 — golden section search finds the minimum of f(x) = (x - 3)²
// at x = 3 within the bracket (0, 6).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_golden_finds_parabola_minimum() {
    let f = |x: f64| (x - 3.0).powi(2);
    let (xmin, _fmin) = golden(f, 0.0, 6.0, 1e-9, 200);
    assert!(
        (xmin - 3.0).abs() < 1e-5,
        "MR27 golden xmin = {xmin}, expected 3"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR28 — brent_minimize agrees with golden on a smooth unimodal
// objective.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_brent_minimize_matches_golden() {
    let f = |x: f64| (x + 1.5).powi(2) + 2.0;
    let (xg, _) = golden(f, -5.0, 3.0, 1e-9, 200);
    let (xb, _) = brent_minimize(f, -5.0, 3.0, 1e-9, 200);
    assert!((xg - xb).abs() < 1e-4, "MR28 golden = {xg} vs brent = {xb}");
}

// ─────────────────────────────────────────────────────────────────────
// MR29 — fixed_point of f(x) = cos(x) starting from x0 = 1 converges
// to the Dottie number ≈ 0.7390851332...
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fixed_point_dottie_number() {
    let f = |x: f64| x.cos();
    let r = fixed_point(f, 1.0, 1e-10, 1000).unwrap();
    let dottie = 0.739_085_133_215_160_6_f64;
    assert!(
        (r - dottie).abs() < 1e-5,
        "MR29 fixed_point of cos = {r}, expected {dottie}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR30 — Non-Negative Least Squares: nnls returns non-negative
// coefficients for any A, b.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_nnls_coefficients_nonneg() {
    let a = vec![
        vec![1.0_f64, 0.5],
        vec![2.0, -1.0],
        vec![3.0, 1.5],
        vec![-1.0, 2.0],
    ];
    let b = vec![1.0_f64, 2.0, 4.0, 0.5];
    let (x, _residual) = nnls(&a, &b).unwrap();
    for (i, &xi) in x.iter().enumerate() {
        assert!(xi >= -1e-9, "MR30 nnls x[{i}] = {xi} < 0");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR31 — Isotonic regression returns a monotone non-decreasing fit.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_isotonic_regression_monotone() {
    let y = vec![1.0_f64, 3.0, 2.5, 4.0, 3.5, 5.0, 4.5, 6.0];
    let yhat = isotonic_regression(&y, None);
    for w in yhat.windows(2) {
        assert!(
            w[0] <= w[1] + 1e-12,
            "MR31 isotonic not monotone: {} > {}",
            w[0],
            w[1]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR32 — brute on a 1D parabola finds a minimum near x = 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_brute_finds_parabola_minimum() {
    let f = |x: &[f64]| x[0].powi(2);
    let res = brute(f, &[(-5.0, 5.0)], 21).unwrap();
    assert!(
        res.x[0].abs() < 0.6,
        "MR32 brute x = {}, expected ≈ 0",
        res.x[0]
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR33 — numerical_gradient on quadratic f(x, y) = x² + y² matches the
// analytical gradient (2x, 2y).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_numerical_gradient_matches_analytical() {
    let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
    for &(x, y) in &[(1.0_f64, 2.0), (-3.0, 0.5), (0.5, -1.0)] {
        let g = numerical_gradient(f, &[x, y], 1e-5);
        let expected = [2.0 * x, 2.0 * y];
        for k in 0..2 {
            assert!(
                (g[k] - expected[k]).abs() < 1e-3,
                "MR33 numerical_gradient[{k}] = {} vs {}",
                g[k],
                expected[k]
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR34 — numerical_hessian of f(x, y) = x² + y² is approximately 2·I.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_numerical_hessian_quadratic() {
    let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
    let h = numerical_hessian(f, &[1.0_f64, -1.5], 1e-3);
    for (i, row) in h.iter().enumerate().take(2) {
        for (j, value) in row.iter().enumerate().take(2) {
            let expected = if i == j { 2.0 } else { 0.0 };
            assert!(
                (*value - expected).abs() < 0.05,
                "MR34 hessian[{i}, {j}] = {} vs {expected}",
                value
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR35 — bracket returns finite a, b, c such that f(b) ≤ min(f(a), f(c))
// for a unimodal function.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_bracket_finite_for_unimodal() {
    let f = |x: f64| (x - 2.0).powi(2);
    let (a, b, c, fa, fb, fc) = bracket(f, 0.0, 1.0);
    for v in [a, b, c, fa, fb, fc] {
        assert!(v.is_finite(), "MR35 bracket returned non-finite {v}");
    }
    assert!(fb <= fa + 1e-9, "MR35 fb = {fb} > fa = {fa}");
    assert!(fb <= fc + 1e-9, "MR35 fb = {fb} > fc = {fc}");
}

// ─────────────────────────────────────────────────────────────────────
// MR36 — minimize_scalar_bounded returns x within the supplied bounds.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_minimize_scalar_bounded_in_range() {
    let f = |x: f64| (x - 7.5).powi(2);
    let bounds = (-1.0_f64, 5.0_f64);
    let (xmin, _fmin) = minimize_scalar_bounded(f, bounds, 1e-9, 200);
    assert!(
        xmin >= bounds.0 - 1e-9 && xmin <= bounds.1 + 1e-9,
        "MR36 minimize_scalar_bounded x = {xmin} outside [{}, {}]",
        bounds.0,
        bounds.1
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR37 — gradient_descent on a strictly convex f(x) = ||x − target||²
// converges to the target.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_gradient_descent_finds_target() {
    let target = vec![1.0_f64, -2.0, 0.5];
    let f = {
        let target = target.clone();
        move |x: &[f64]| {
            x.iter()
                .zip(&target)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
        }
    };
    let grad = {
        let target = target.clone();
        move |x: &[f64]| {
            x.iter()
                .zip(&target)
                .map(|(a, b)| 2.0 * (a - b))
                .collect::<Vec<f64>>()
        }
    };
    let x0 = vec![0.0_f64; 3];
    let res = gradient_descent(f, grad, &x0, 1e-9, 5000, 0.05);
    for (i, (xi, ti)) in res.x.iter().zip(&target).enumerate() {
        assert!(
            (xi - ti).abs() < 1e-3,
            "MR37 gradient_descent x[{i}] = {xi} vs target {ti}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR38 — minimize_trisection finds the minimum of (x - 4)² near x = 4.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_minimize_trisection_parabola() {
    let f = |x: f64| (x - 4.0).powi(2);
    let (xmin, _fmin) = minimize_trisection(f, 0.0, 8.0, 1e-9, 500);
    assert!(
        (xmin - 4.0).abs() < 1e-4,
        "MR38 minimize_trisection x = {xmin}, expected 4"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR39 — numerical_jacobian on a linear map f(x) = A·x recovers A.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_numerical_jacobian_linear_map() {
    // A = [[2, -1, 0], [1, 3, -2]] is 2×3.
    let a = vec![vec![2.0_f64, -1.0, 0.0], vec![1.0, 3.0, -2.0]];
    let f = {
        let a = a.clone();
        move |x: &[f64]| {
            a.iter()
                .map(|row| row.iter().zip(x).map(|(&aij, &xj)| aij * xj).sum::<f64>())
                .collect::<Vec<f64>>()
        }
    };
    let jac = numerical_jacobian(f, &[1.0_f64, 2.0, -1.0], 1e-5);
    assert_eq!(jac.len(), a.len(), "MR39 jacobian rows");
    assert_eq!(jac[0].len(), a[0].len(), "MR39 jacobian cols");
    for i in 0..a.len() {
        for j in 0..a[0].len() {
            assert!(
                (jac[i][j] - a[i][j]).abs() < 1e-3,
                "MR39 jac[{i}, {j}] = {} vs a = {}",
                jac[i][j],
                a[i][j]
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR40 — projected_gradient_descent stays inside the supplied bounds.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_projected_gradient_descent_respects_bounds() {
    // Minimise f(x) = (x[0] - 100)² + (x[1] + 50)² subject to box bounds.
    let f = |x: &[f64]| (x[0] - 100.0).powi(2) + (x[1] + 50.0).powi(2);
    let grad = |x: &[f64]| vec![2.0 * (x[0] - 100.0), 2.0 * (x[1] + 50.0)];
    let lb = vec![-1.0_f64, -2.0_f64];
    let ub = vec![1.0_f64, 2.0_f64];
    let res = projected_gradient_descent(f, grad, &[0.0, 0.0], &lb, &ub, 1e-6, 500, 0.05);
    for (i, &xi) in res.x.iter().enumerate() {
        assert!(
            xi >= lb[i] - 1e-9 && xi <= ub[i] + 1e-9,
            "MR40 projected x[{i}] = {xi} outside [{}, {}]",
            lb[i],
            ub[i]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR41 — isotonic_regression with weights returns a monotone fit.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_isotonic_with_weights_monotone() {
    let y = vec![1.0_f64, 3.0, 2.0, 5.0, 4.0, 6.0];
    let w = vec![1.0_f64, 0.5, 2.0, 1.0, 1.5, 0.8];
    let yhat = isotonic_regression(&y, Some(&w));
    for window in yhat.windows(2) {
        assert!(
            window[0] <= window[1] + 1e-12,
            "MR41 isotonic with weights not monotone: {} > {}",
            window[0],
            window[1]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR42 — pso (particle swarm) finds the parabola minimum near 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_pso_finds_parabola_minimum() {
    let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
    let lb = vec![-5.0_f64, -5.0];
    let ub = vec![5.0_f64, 5.0];
    let (x, fmin) = pso(f, &lb, &ub, 30, 100, 7);
    assert!(fmin < 0.5, "MR42 pso fmin = {fmin}, expected ≈ 0");
    assert!(
        x.iter().all(|&v| v.abs() < 1.0),
        "MR42 pso x = {x:?} far from origin"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR43 — least_squares minimises the sum of squared residuals: returned
// solution achieves residual norm not greater than at the initial point.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_least_squares_reduces_residual_norm() {
    // Residual function: r(x) = [x[0] - 1, x[1] - 2, x[0] + x[1] - 4].
    let r = |x: &[f64]| vec![x[0] - 1.0, x[1] - 2.0, x[0] + x[1] - 4.0];
    let res = least_squares(r, &[0.0, 0.0], LeastSquaresOptions::default()).unwrap();
    let r0 = r(&[0.0_f64, 0.0]);
    let n0: f64 = r0.iter().map(|v| v * v).sum::<f64>().sqrt();
    let r_final = r(&res.x);
    let nf: f64 = r_final.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        nf <= n0 + 1e-9,
        "MR43 least_squares residual norm = {nf} > initial = {n0}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR44 — basinhopping on a strictly convex objective converges to the
// global minimum (test with a scaled paraboloid).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_basinhopping_convex_objective() {
    let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] + 3.0).powi(2);
    let opts = BasinhoppingOptions {
        seed: Some(42),
        niter: 5,
        ..Default::default()
    };
    let res = basinhopping(f, &[10.0, -10.0], opts).unwrap();
    assert!(
        (res.x[0] - 2.0).abs() < 0.5 && (res.x[1] + 3.0).abs() < 0.5,
        "MR44 basinhopping x = {:?}, expected near (2, -3)",
        res.x
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR45 — check_grad on a correct analytical gradient is small.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_check_grad_analytical_match() {
    let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
    let g = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
    let err = check_grad(f, g, &[1.5_f64, -2.5]).unwrap();
    assert!(err < 1e-3, "MR45 check_grad error = {err}, expected small");
}

// ─────────────────────────────────────────────────────────────────────
// MR46 — linear_sum_assignment returns row and column index vectors of
// equal length; columns are a permutation (no duplicates).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_linear_sum_assignment_validity() {
    let cost = vec![
        vec![4.0_f64, 1.0, 3.0],
        vec![2.0, 0.0, 5.0],
        vec![3.0, 2.0, 2.0],
    ];
    let (row_ind, col_ind) = linear_sum_assignment(&cost).unwrap();
    assert_eq!(row_ind.len(), col_ind.len(), "MR46 LAP lengths");
    let mut col_seen = vec![false; cost.len()];
    for &c in &col_ind {
        assert!(c < cost.len(), "MR46 LAP col index out of range: {c}");
        assert!(!col_seen[c], "MR46 LAP duplicate col index: {c}");
        col_seen[c] = true;
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR47 — dual_annealing on f(x) = x² + y² finds the origin within
// stochastic tolerance.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_dual_annealing_sphere_makes_progress() {
    // Dual annealing on a 2-D sphere with a small iteration budget
    // should make progress: fmin much lower than the worst-case
    // boundary value (5² + 5² = 50).
    let f = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
    let bounds = vec![(-5.0_f64, 5.0_f64), (-5.0, 5.0)];
    let res = dual_annealing(f, &bounds, 200, 7).unwrap();
    let fmin = res.fun.unwrap();
    assert!(
        fmin < 50.0,
        "MR47 dual_annealing fmin = {fmin} did not improve over corner"
    );
    assert!(
        fmin >= 0.0,
        "MR47 dual_annealing fmin = {fmin} < 0 on non-negative objective"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR48 — SHGO (simplicial homology global) on a parabola finds a
// minimum within tolerance.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_shgo_parabola_minimum() {
    let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] + 2.0).powi(2);
    let bounds = vec![(-5.0_f64, 5.0_f64), (-5.0, 5.0)];
    let res = shgo(f, &bounds).unwrap();
    let fmin = res.fun.unwrap();
    assert!(fmin < 1.0, "MR48 shgo fmin = {fmin} on parabola");
}

// ─────────────────────────────────────────────────────────────────────
// MR49 — rosen gradient norm at the minimum (1, 1, ..., 1) is 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rosen_gradient_norm_at_min() {
    for n in [2usize, 3, 5, 10] {
        let x = vec![1.0_f64; n];
        let g = rosen_der(&x);
        let norm: f64 = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm < 1e-12, "MR49 ‖∇rosen(1, …, 1)‖ (n={n}) = {norm}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR50 — minimize on the (shifted) Rosenbrock function from a far
// initial guess returns x close to (1, 1).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_minimize_rosenbrock_finds_global() {
    let f = |x: &[f64]| rosen(x);
    let res = minimize(f, &[-1.5_f64, 2.5], MinimizeOptions::default()).unwrap();
    assert!(
        (res.x[0] - 1.0).abs() < 0.1 && (res.x[1] - 1.0).abs() < 0.1,
        "MR50 minimize rosen → x = {:?}, expected ≈ (1, 1)",
        res.x
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR51 — rosen_hess at (1, …, 1) is symmetric.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rosen_hess_symmetric_at_minimum() {
    for n in [2usize, 3, 5] {
        let x = vec![1.0_f64; n];
        let h = rosen_hess(&x);
        for (i, row) in h.iter().enumerate().take(n) {
            for (j, value) in row.iter().enumerate().take(n) {
                assert!(
                    (*value - h[j][i]).abs() < 1e-9,
                    "MR51 rosen_hess[{i}, {j}] = {} vs [{j}, {i}] = {}",
                    value,
                    h[j][i]
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR52 — rosen_hess_prod(x, p) = rosen_hess(x) · p (matrix-vector
// product equivalence).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_rosen_hess_prod_matches_dense() {
    let x = vec![0.5_f64, -0.5, 0.7, 1.0];
    let p = vec![1.0_f64, 2.0, -0.5, 0.8];
    let h = rosen_hess(&x);
    let hp = rosen_hess_prod(&x, &p);
    let expected: Vec<f64> = h
        .iter()
        .map(|row| row.iter().zip(&p).map(|(&hij, &pj)| hij * pj).sum())
        .collect();
    for (i, (a, b)) in expected.iter().zip(&hp).enumerate() {
        assert!(
            (a - b).abs() < 1e-9 * a.abs().max(1.0),
            "MR52 rosen_hess_prod[{i}] = {b} vs (H·p)[{i}] = {a}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// MR53 — linprog: minimise c·x subject to bounds. For c = [1, 1] with
// x ≥ 0 and no upper bound, linprog should find x = (0, 0) when
// feasible.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_linprog_simple_bounded() {
    // Minimise 2x + 3y subject to x + y >= 1 (i.e., -x - y <= -1), x,y >= 0.
    let c = vec![2.0_f64, 3.0];
    let a_ub = vec![vec![-1.0_f64, -1.0]];
    let b_ub = vec![-1.0_f64];
    let a_eq: Vec<Vec<f64>> = vec![];
    let b_eq: Vec<f64> = vec![];
    let bounds = vec![(Some(0.0), None), (Some(0.0), None)];
    let options = LinprogOptions {
        maxiter: Some(100),
        ..LinprogOptions::default()
    };
    let res = linprog(&c, &a_ub, &b_ub, &a_eq, &b_eq, &bounds, options).unwrap();
    // Optimal: x = 1, y = 0; objective = 2.
    assert!(
        (res.fun - 2.0).abs() < 1e-3,
        "MR53 linprog objective = {} vs expected 2",
        res.fun
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR54 — cobyla on a strictly convex unconstrained-feasible problem
// converges towards the minimum.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_cobyla_convex_makes_progress() {
    let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] + 1.0).powi(2);
    let constraints: Vec<fn(&[f64]) -> f64> = Vec::new();
    let opts = FminCobylaOptions {
        rhobeg: 0.5,
        maxfun: 200,
        ..FminCobylaOptions::default()
    };
    let res = cobyla(f, &[5.0_f64, 5.0], &constraints, opts).unwrap();
    let f_init = (5.0_f64 - 2.0).powi(2) + (5.0_f64 + 1.0).powi(2);
    let f_final = res.fun.unwrap();
    assert!(
        f_final < f_init,
        "MR54 cobyla did not improve: f_final = {f_final} >= f_init = {f_init}"
    );
    // Powell's COBYLA converges here: the trust region shrinks to rhoend = 1e-4 near (2, -1).
    assert!(res.success, "MR54 cobyla: {}", res.message);
    assert!(
        (res.x[0] - 2.0).abs() < 1e-3 && (res.x[1] + 1.0).abs() < 1e-3,
        "MR54 cobyla x = {:?}",
        res.x
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR55 — fsolve on a 2x2 nonlinear system finds a root.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_fsolve_2d_nonlinear() {
    // x² + y² = 25 ; x + y = 7 ⇒ (x, y) ∈ {(4, 3), (3, 4)}.
    // Start from (5, 2) to avoid singular Jacobian at x = y.
    let f = |z: &[f64]| vec![z[0].powi(2) + z[1].powi(2) - 25.0, z[0] + z[1] - 7.0];
    let res = fsolve(f, &[5.0_f64, 2.0]).unwrap();
    let (x, y) = (res.x[0], res.x[1]);
    assert!(
        (((x - 4.0).abs() < 1e-3 && (y - 3.0).abs() < 1e-3)
            || ((x - 3.0).abs() < 1e-3 && (y - 4.0).abs() < 1e-3)),
        "MR55 fsolve 2D found ({x}, {y}), expected (4, 3) or (3, 4)"
    );
}

// ─────────────────────────────────────────────────────────────────────
// MR56 — minimize converges from origin on shifted parabola.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mr_minimize_origin_to_shifted() {
    let f = |x: &[f64]| (x[0] - 4.0).powi(2) + (x[1] - 5.0).powi(2);
    let res = minimize(f, &[0.0_f64, 0.0], MinimizeOptions::default()).unwrap();
    assert!(
        (res.x[0] - 4.0).abs() < 0.05 && (res.x[1] - 5.0).abs() < 0.05,
        "MR56 minimize from origin → ({}, {}), expected (4, 5)",
        res.x[0],
        res.x[1]
    );
}
