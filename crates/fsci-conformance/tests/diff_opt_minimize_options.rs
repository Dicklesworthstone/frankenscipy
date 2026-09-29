#![forbid(unsafe_code)]
//! frankenscipy-6ycp2: `minimize(..., options={...})`, live against SciPy. Every row runs
//! SciPy's `minimize(rosen, x0, jac=rosen_der, method=..., tol=..., options={...})` (Nelder-Mead
//! and Powell without `jac`) and fsci's `minimize` with the same `MinimizeMethodOptions`:
//! (nit, nfev) must be SciPy's exactly and x within 1e-7 (L-BFGS-B and BFGS differ from SciPy
//! only in the last digits SciPy itself moves between OpenBLAS kernels). Each option row must
//! differ from its method's default row, in SciPy and in fsci (must-differ). The compared-row
//! count is asserted under `FSCI_REQUIRE_SCIPY_ORACLE` (CI G3 optimize shard, glob diff_opt*).

use std::process::Stdio;

use fsci_opt::{GradientFunc, MinimizeMethodOptions, MinimizeOptions, OptimizeMethod, minimize};
use serde::Deserialize;

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

/// SciPy's `rosen` / `rosen_der`, in numpy's order.
fn rosen(x: &[f64]) -> f64 {
    (0..x.len() - 1)
        .map(|i| 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2))
        .sum()
}

fn rosen_der(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut d = vec![0.0; n];
    for i in 1..n - 1 {
        d[i] = 200.0 * (x[i] - x[i - 1] * x[i - 1])
            - 400.0 * (x[i + 1] - x[i] * x[i]) * x[i]
            - 2.0 * (1.0 - x[i]);
    }
    d[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
    d[n - 1] = 200.0 * (x[n - 1] - x[n - 2] * x[n - 2]);
    d
}

#[derive(Debug, Deserialize)]
struct ScipyRow {
    name: String,
    nit: usize,
    nfev: usize,
    x: Vec<f64>,
}

const X5: [f64; 5] = [-1.2, 1.0, -1.2, 1.0, -1.2];
const X4: [f64; 4] = [-1.2, 1.0, -1.2, 1.0];
const X3: [f64; 3] = [-1.2, 1.0, 0.5];

fn scipy_rows_or_skip() -> Option<Vec<ScipyRow>> {
    let script = r#"
import json
from scipy.optimize import minimize, rosen, rosen_der
x5 = [-1.2, 1.0, -1.2, 1.0, -1.2]
x4 = [-1.2, 1.0, -1.2, 1.0]
x3 = [-1.2, 1.0, 0.5]
simplex = [[-1.2, 1.0, 0.5], [-1.0, 1.0, 0.5], [-1.2, 1.2, 0.5], [-1.2, 1.0, 0.7]]
direc = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
rows = [
    ("lbfgsb/default", "L-BFGS-B", x5, None, {}),
    ("lbfgsb/maxcor=3", "L-BFGS-B", x5, None, {"maxcor": 3}),
    ("lbfgsb/maxls=2", "L-BFGS-B", x5, None, {"maxls": 2}),
    ("lbfgsb/gtol=0.1", "L-BFGS-B", x5, None, {"gtol": 0.1}),
    ("lbfgsb/ftol=1e-4", "L-BFGS-B", x5, None, {"ftol": 1e-4}),
    ("lbfgsb/tol=1e-4,gtol=1e-9", "L-BFGS-B", x5, 1e-4, {"gtol": 1e-9}),
    ("bfgs/default", "BFGS", x5, None, {}),
    ("bfgs/gtol=0.1", "BFGS", x5, None, {"gtol": 0.1}),
    ("bfgs/norm=2", "BFGS", x5, None, {"norm": 2}),
    ("bfgs/xrtol=1e-3", "BFGS", x5, None, {"xrtol": 1e-3}),
    ("bfgs4/default", "BFGS", x4, None, {}),
    ("bfgs4/c2=0.5", "BFGS", x4, None, {"c2": 0.5}),
    ("cg/default", "CG", x4, None, {}),
    ("cg/gtol=1e-3", "CG", x4, None, {"gtol": 1e-3}),
    ("nm/default", "Nelder-Mead", x3, None, {}),
    ("nm/adaptive", "Nelder-Mead", x3, None, {"adaptive": True}),
    ("nm/initial_simplex", "Nelder-Mead", x3, None, {"initial_simplex": simplex}),
    ("nm/xatol=fatol=1e-2", "Nelder-Mead", x3, None, {"xatol": 1e-2, "fatol": 1e-2}),
    ("powell/default", "Powell", x3, None, {}),
    ("powell/direc", "Powell", x3, None, {"direc": direc}),
    ("powell/xtol=1e-2", "Powell", x3, None, {"xtol": 1e-2}),
    ("powell/ftol=1e-3", "Powell", x3, None, {"ftol": 1e-3}),
]
out = []
for name, method, x0, tol, opts in rows:
    jac = rosen_der if method in ("L-BFGS-B", "BFGS", "CG") else None
    r = minimize(rosen, x0, jac=jac, method=method, tol=tol, options=opts)
    out.append({"name": name, "nit": int(r.nit), "nfev": int(r.nfev), "x": [float(v) for v in r.x]})
print(json.dumps(out))
"#;
    let output = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
    {
        Ok(output) => output,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "python spawn failed: {e}"
            );
            return None;
        }
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "minimize options oracle failed: {stderr}"
        );
        return None;
    }
    Some(serde_json::from_slice(&output.stdout).expect("parse the oracle's rows"))
}

/// The fsci side of each row: (method, x0, tol, options).
fn fsci_row<'a>(
    name: &str,
    simplex: &'a [Vec<f64>],
    direc: &'a [Vec<f64>],
) -> (
    OptimizeMethod,
    &'static [f64],
    Option<f64>,
    MinimizeMethodOptions<'a>,
) {
    let none = MinimizeMethodOptions::default();
    match name {
        "lbfgsb/default" => (OptimizeMethod::LBfgsB, &X5, None, none),
        "lbfgsb/maxcor=3" => (
            OptimizeMethod::LBfgsB,
            &X5,
            None,
            MinimizeMethodOptions {
                maxcor: Some(3),
                ..none
            },
        ),
        "lbfgsb/maxls=2" => (
            OptimizeMethod::LBfgsB,
            &X5,
            None,
            MinimizeMethodOptions {
                maxls: Some(2),
                ..none
            },
        ),
        "lbfgsb/gtol=0.1" => (
            OptimizeMethod::LBfgsB,
            &X5,
            None,
            MinimizeMethodOptions {
                gtol: Some(0.1),
                ..none
            },
        ),
        "lbfgsb/ftol=1e-4" => (
            OptimizeMethod::LBfgsB,
            &X5,
            None,
            MinimizeMethodOptions {
                ftol: Some(1e-4),
                ..none
            },
        ),
        "lbfgsb/tol=1e-4,gtol=1e-9" => (
            OptimizeMethod::LBfgsB,
            &X5,
            Some(1e-4),
            MinimizeMethodOptions {
                gtol: Some(1e-9),
                ..none
            },
        ),
        "bfgs/default" => (OptimizeMethod::Bfgs, &X5, None, none),
        "bfgs/gtol=0.1" => (
            OptimizeMethod::Bfgs,
            &X5,
            None,
            MinimizeMethodOptions {
                gtol: Some(0.1),
                ..none
            },
        ),
        "bfgs/norm=2" => (
            OptimizeMethod::Bfgs,
            &X5,
            None,
            MinimizeMethodOptions {
                norm: Some(2.0),
                ..none
            },
        ),
        "bfgs/xrtol=1e-3" => (
            OptimizeMethod::Bfgs,
            &X5,
            None,
            MinimizeMethodOptions {
                xrtol: Some(1e-3),
                ..none
            },
        ),
        "bfgs4/default" => (OptimizeMethod::Bfgs, &X4, None, none),
        "bfgs4/c2=0.5" => (
            OptimizeMethod::Bfgs,
            &X4,
            None,
            MinimizeMethodOptions {
                c2: Some(0.5),
                ..none
            },
        ),
        "cg/default" => (OptimizeMethod::ConjugateGradient, &X4, None, none),
        "cg/gtol=1e-3" => (
            OptimizeMethod::ConjugateGradient,
            &X4,
            None,
            MinimizeMethodOptions {
                gtol: Some(1e-3),
                ..none
            },
        ),
        "nm/default" => (OptimizeMethod::NelderMead, &X3, None, none),
        "nm/adaptive" => (
            OptimizeMethod::NelderMead,
            &X3,
            None,
            MinimizeMethodOptions {
                adaptive: Some(true),
                ..none
            },
        ),
        "nm/initial_simplex" => (
            OptimizeMethod::NelderMead,
            &X3,
            None,
            MinimizeMethodOptions {
                initial_simplex: Some(simplex),
                ..none
            },
        ),
        "nm/xatol=fatol=1e-2" => (
            OptimizeMethod::NelderMead,
            &X3,
            None,
            MinimizeMethodOptions {
                xatol: Some(1e-2),
                fatol: Some(1e-2),
                ..none
            },
        ),
        "powell/default" => (OptimizeMethod::Powell, &X3, None, none),
        "powell/direc" => (
            OptimizeMethod::Powell,
            &X3,
            None,
            MinimizeMethodOptions {
                direc: Some(direc),
                ..none
            },
        ),
        "powell/xtol=1e-2" => (
            OptimizeMethod::Powell,
            &X3,
            None,
            MinimizeMethodOptions {
                xtol: Some(1e-2),
                ..none
            },
        ),
        "powell/ftol=1e-3" => (
            OptimizeMethod::Powell,
            &X3,
            None,
            MinimizeMethodOptions {
                ftol: Some(1e-3),
                ..none
            },
        ),
        other => unreachable!("no fsci row for {other}"),
    }
}

#[test]
fn diff_opt_minimize_options() {
    let Some(rows) = scipy_rows_or_skip() else {
        println!("SciPy unavailable: no row compared");
        return;
    };
    let simplex = vec![
        vec![-1.2, 1.0, 0.5],
        vec![-1.0, 1.0, 0.5],
        vec![-1.2, 1.2, 0.5],
        vec![-1.2, 1.0, 0.7],
    ];
    let direc = vec![
        vec![0.0, 0.0, 1.0],
        vec![0.0, 1.0, 0.0],
        vec![1.0, 0.0, 0.0],
    ];
    let mut compared = 0;
    let mut failures = Vec::new();
    let mut defaults: Vec<(String, (usize, usize), (usize, usize))> = Vec::new();
    for row in &rows {
        let (method, x0, tol, method_options) = fsci_row(&row.name, &simplex, &direc);
        let gradient = matches!(
            method,
            OptimizeMethod::LBfgsB | OptimizeMethod::Bfgs | OptimizeMethod::ConjugateGradient
        )
        .then_some(rosen_der as GradientFunc);
        let options = MinimizeOptions {
            method: Some(method),
            tol,
            gradient,
            method_options,
            ..MinimizeOptions::default()
        };
        let result = minimize(rosen, x0, options).expect(&row.name);
        let x_error = result
            .x
            .iter()
            .zip(&row.x)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        println!(
            "{}: fsci ({}, {}), SciPy ({}, {}); max |x - x_scipy| = {x_error:e}",
            row.name, result.nit, result.nfev, row.nit, row.nfev
        );
        if (result.nit, result.nfev) != (row.nit, row.nfev) || x_error > 1e-7 {
            failures.push(row.name.clone());
        }
        let family = row.name.split('/').next().expect("family").to_string();
        let runs = ((row.nit, row.nfev), (result.nit, result.nfev));
        if row.name.ends_with("/default") {
            defaults.push((family, runs.0, runs.1));
        } else if let Some((_, scipy_default, fsci_default)) =
            defaults.iter().find(|(f, _, _)| *f == family)
        {
            assert_ne!(
                runs.0, *scipy_default,
                "{}: SciPy's row equals its default",
                row.name
            );
            assert_ne!(
                runs.1, *fsci_default,
                "{}: fsci's row equals its default",
                row.name
            );
        }
        compared += 1;
    }
    println!("{compared} rows compared");
    assert!(failures.is_empty(), "rows off SciPy's path: {failures:?}");
    assert_eq!(compared, 22, "every row was compared");
}
