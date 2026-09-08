#![no_main]

use arbitrary::Arbitrary;
use fsci_opt::{bisect, brentq, brenth, ridder, toms748, RootOptions};
use libfuzzer_sys::fuzz_target;

const TOL: f64 = 1e-8;

#[derive(Debug, Arbitrary)]
struct RootInput {
    a: f64,
    b: f64,
    func_type: u8,
    poly_coeffs: [f64; 4],
}

fn sanitize(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(-100.0, 100.0)
    } else {
        0.0
    }
}

fn eval_func(func_type: u8, coeffs: &[f64; 4], x: f64) -> f64 {
    match func_type % 5 {
        0 => {
            let c = coeffs.map(sanitize);
            c[0] + c[1] * x + c[2] * x * x + c[3] * x * x * x
        }
        1 => x.sin(),
        2 => x.exp() - 2.0,
        3 => x.tanh(),
        4 => x.powi(3) - x,
        _ => x,
    }
}

fuzz_target!(|input: RootInput| {
    let mut a = sanitize(input.a);
    let mut b = sanitize(input.b);

    if (a - b).abs() < 1e-10 {
        return;
    }
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }

    let f = |x: f64| eval_func(input.func_type, &input.poly_coeffs, x);
    let fa = f(a);
    let fb = f(b);

    if !fa.is_finite() || !fb.is_finite() {
        return;
    }
    if fa.signum() == fb.signum() {
        return;
    }

    let bracket = (a, b);
    let opts = RootOptions::default();

    let methods: [(&str, Box<dyn Fn() -> Option<f64>>); 5] = [
        ("bisect", Box::new(|| bisect(f, bracket, opts).ok().map(|r| r.root))),
        ("brentq", Box::new(|| brentq(f, bracket, opts).ok().map(|r| r.root))),
        ("brenth", Box::new(|| brenth(f, bracket, opts).ok().map(|r| r.root))),
        ("ridder", Box::new(|| ridder(f, bracket, opts).ok().map(|r| r.root))),
        ("toms748", Box::new(|| toms748(f, bracket, opts).ok().map(|r| r.root))),
    ];

    let mut roots: Vec<(&str, f64)> = Vec::new();

    for (name, find_root) in &methods {
        if let Some(root) = find_root() {
            if root < a - TOL || root > b + TOL {
                panic!(
                    "{}: root {} outside bracket [{}, {}]",
                    name, root, a, b
                );
            }

            let froot = f(root);
            if froot.abs() > 1e-6 {
                panic!(
                    "{}: f(root) = {} is not close to zero (root={})",
                    name, froot, root
                );
            }

            roots.push((name, root));
        }
    }

    if roots.len() >= 2 {
        let first_root = roots[0].1;
        for (name, root) in &roots[1..] {
            let diff = (root - first_root).abs();
            if diff > 1e-4 {
                panic!(
                    "Root disagreement: {} found {} but {} found {} (diff={})",
                    roots[0].0, first_root, name, root, diff
                );
            }
        }
    }
});
