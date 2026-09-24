#![forbid(unsafe_code)]

use fsci_opt::{MinimizeOptions, OptimizeMethod, OptimizeResult, cg_pr_plus};
use fsci_runtime::RuntimeMode;

fn rosenbrock_nd(x: &[f64]) -> f64 {
    let mut acc = 0.0;
    for i in 0..x.len() - 1 {
        let a = 1.0 - x[i];
        let b = x[i + 1] - x[i] * x[i];
        acc += a * a + 100.0 * b * b;
    }
    acc
}

fn quadratic(x: &[f64]) -> f64 {
    x.iter()
        .enumerate()
        .map(|(idx, value)| (idx as f64 + 1.0) * value * value)
        .sum()
}

fn nonconvex_saddle(x: &[f64]) -> f64 {
    x[0] * x[0] - x[1] * x[1] + 0.1 * x[1].powi(4)
}

fn options(maxiter: usize, maxfev: usize, tol: f64) -> MinimizeOptions<'static> {
    MinimizeOptions {
        method: Some(OptimizeMethod::ConjugateGradient),
        tol: Some(tol),
        maxiter: Some(maxiter),
        maxfev: Some(maxfev),
        mode: RuntimeMode::Strict,
        ..MinimizeOptions::default()
    }
}

fn emit_result(name: &str, result: &OptimizeResult) {
    print!(
        "{name}|success={}|status={:?}|nit={}|nfev={}|njev={}|nhev={}|fun={:016x}",
        result.success,
        result.status,
        result.nit,
        result.nfev,
        result.njev,
        result.nhev,
        result.fun.unwrap_or(f64::NAN).to_bits()
    );
    print!("|x=");
    for value in &result.x {
        print!("{:016x},", value.to_bits());
    }
    print!("|jac=");
    if let Some(jac) = &result.jac {
        for value in jac {
            print!("{:016x},", value.to_bits());
        }
    }
    println!();
}

fn run(name: &str, f: fn(&[f64]) -> f64, x0: &[f64], opts: MinimizeOptions) {
    let result = cg_pr_plus(&f, x0, opts).expect("cg_pr_plus");
    emit_result(name, &result);
}

fn main() {
    let rb10 = vec![0.0; 10];
    let q10: Vec<f64> = (0..10)
        .map(|idx| if idx % 2 == 0 { 2.5 } else { -1.25 })
        .collect();
    run(
        "rosenbrock10",
        rosenbrock_nd,
        &rb10,
        options(2_500, 100_000, 1.0e-6),
    );
    run(
        "quadratic10",
        quadratic,
        &q10,
        options(600, 200_000, 1.0e-6),
    );
    run(
        "nonconvex2",
        nonconvex_saddle,
        &[1.0, 1.0],
        options(200, 100_000, 1.0e-6),
    );
}
