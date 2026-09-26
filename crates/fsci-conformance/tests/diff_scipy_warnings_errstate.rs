#![forbid(unsafe_code)]
//! Live SciPy diff for SciPy's warnings and `scipy.special.errstate` (frankenscipy-8dndw.1).
//!
//! The warning classes and the special-function error state used to be names that did
//! nothing: twelve `pub struct XWarning(pub String)` that nothing constructed, and
//! `seterr`/`geterr` that ignored their argument. This file holds each implemented name to
//! the incumbent on the inputs where SciPy raises it, and on neighbours where it does not.
//!
//! Warnings: each scenario runs the same call on both sides under
//! `warnings.catch_warnings(record=True)` / `fsci_runtime::catch_warnings`. The lists of
//! SciPy's own warning classes raised must be equal, in order (numpy's `RuntimeWarning`s are
//! not SciPy's classes and are dropped), as must the scenario's values. `NoConvergence` is
//! an exception in SciPy and an `Err` here; its scenario compares the class.
//!
//! errstate: every wired special function over a grid, under `errstate(all='raise')`: the
//! raised code (or none) must be SciPy's for every point, and under `all='warn'` the number
//! of `SpecialFunctionWarning`s per call must be SciPy's. The grids leave out the inputs on
//! which SciPy's report comes from a floating-point exception flag set inside its C kernel
//! rather than from an `sf_error` call: NaN ("domain error"), subnormals ("underflow") and
//! arguments whose intermediates overflow. Those are not reproduced (see
//! `fsci_special::sf_error_unary`), and a grid point that trips one fails this test.
//!
//! The compared counts are asserted, and each side's must-raise and must-not-raise points
//! are both present, so an oracle or harness that answers "nothing" cannot pass.

use std::io::Write;
use std::process::Stdio;

use fsci_runtime::{RuntimeMode, catch_warnings};
use fsci_special::{
    SpecialErrConfig, SpecialErrMode, SpecialErrorKind, SpecialResult, SpecialTensor, errstate,
};
use serde::Deserialize;

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
/// Values carried by the warning scenarios: fsci within this relative distance of SciPy.
const VALUE_REL_TOL: f64 = 1e-9;
/// The curve_fit scenarios' popt are held to curve_fit's own parity contract
/// (diff_opt_curve_fit.rs ABS_TOL), not the tighter bound above: on the two-point exact
/// fit fsci's Levenberg-Marquardt stops 3.8e-9 from SciPy's exact [2, 1], inside both
/// solvers' xtol (1.49e-8). The subject of these rows is the warning, compared exactly.
const FIT_VALUE_ABS_TOL: f64 = 1e-6;
/// Grid points compared, and how many of them SciPy raises on: both lower bounds keep the
/// comparison from passing over an empty or all-quiet grid.
const MIN_ERRSTATE_POINTS: usize = 300;
const MIN_ERRSTATE_RAISES: usize = 100;
const WARNING_SCENARIOS: usize = 16;

#[derive(Debug, Clone)]
struct GridRow {
    func: &'static str,
    args: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct Oracle {
    /// Per grid row: SciPy's raised code ("singularity", "domain error", ...) or null.
    raised: Vec<Option<String>>,
    /// Per function: SpecialFunctionWarnings from one vectorised call over its grid.
    warn_counts: Vec<(String, usize)>,
    scenarios: Vec<Scenario>,
}

#[derive(Debug, Deserialize)]
struct Scenario {
    name: String,
    /// SciPy warning classes raised, in order; an exception is `"raise:<Class>"`.
    classes: Vec<String>,
    /// Values to compare (NaN encoded as null).
    values: Vec<Option<f64>>,
}

const SCIPY_CLASSES: &str = "{'DegenerateDataWarning','ConstantInputWarning','NearConstantInputWarning',\
'OptimizeWarning','IntegrationWarning','ODEintWarning','BadCoefficients','SpecialFunctionWarning'}";

fn grid() -> Vec<GridRow> {
    let unary: [(&'static str, &[f64]); 15] = [
        (
            "gamma",
            &[
                f64::NEG_INFINITY,
                -1e300,
                -3.0,
                -2.5,
                -2.0,
                -1.0,
                -0.5,
                -0.0,
                0.0,
                0.5,
                1.0,
                2.5,
                10.0,
                100.0,
                f64::INFINITY,
            ],
        ),
        (
            "gammaln",
            &[
                f64::NEG_INFINITY,
                -1e300,
                -3.0,
                -2.5,
                -2.0,
                -1.0,
                -0.5,
                -0.0,
                0.0,
                0.5,
                1.0,
                2.0,
                100.0,
                f64::INFINITY,
            ],
        ),
        (
            "digamma",
            &[
                f64::NEG_INFINITY,
                -1e300,
                -3.0,
                -2.5,
                -2.0,
                -1.0,
                -0.5,
                -0.0,
                0.0,
                0.5,
                1.0,
                2.0,
                100.0,
                f64::INFINITY,
            ],
        ),
        ("psi", &[-3.0, -2.5, -1.0, -0.0, 0.0, 0.5, 1.0, 2.0]),
        (
            "loggamma",
            &[
                f64::NEG_INFINITY,
                -2.0,
                -1.0,
                -0.5,
                -0.0,
                0.0,
                0.5,
                1.0,
                10.0,
                f64::INFINITY,
            ],
        ),
        (
            "ndtri",
            &[
                f64::NEG_INFINITY,
                -0.5,
                -1e-300,
                0.0,
                1e-300,
                0.3,
                0.5,
                1.0,
                1.000_000_000_000_001,
                1.5,
                2.0,
                f64::INFINITY,
            ],
        ),
        (
            "erfinv",
            &[
                f64::NEG_INFINITY,
                -2.0,
                -1.0,
                -1.0 + 1e-16,
                -0.5,
                0.0,
                0.5,
                1.0 - 1e-16,
                1.0,
                1.5,
                f64::INFINITY,
            ],
        ),
        (
            "erfcinv",
            &[
                -1.0,
                0.0,
                1e-300,
                0.5,
                1.0,
                2.0 - 1e-16,
                2.0,
                2.5,
                f64::INFINITY,
            ],
        ),
        (
            "y0",
            &[
                f64::NEG_INFINITY,
                -10.0,
                -1.0,
                -0.5,
                -0.0,
                0.0,
                0.5,
                1.0,
                2.0,
                10.0,
                100.0,
                700.0,
            ],
        ),
        (
            "y1",
            &[
                f64::NEG_INFINITY,
                -10.0,
                -1.0,
                -0.5,
                -0.0,
                0.0,
                0.5,
                1.0,
                2.0,
                10.0,
                100.0,
                700.0,
            ],
        ),
        (
            "k0",
            &[
                f64::NEG_INFINITY,
                -10.0,
                -1.0,
                -0.5,
                -0.0,
                0.0,
                0.5,
                1.0,
                2.0,
                10.0,
                100.0,
                700.0,
                f64::INFINITY,
            ],
        ),
        (
            "k1",
            &[
                f64::NEG_INFINITY,
                -10.0,
                -1.0,
                -0.5,
                -0.0,
                0.0,
                0.5,
                1.0,
                2.0,
                10.0,
                100.0,
                700.0,
                f64::INFINITY,
            ],
        ),
        (
            "ellipk",
            &[
                f64::NEG_INFINITY,
                -1e300,
                -10.0,
                -1.0,
                0.0,
                0.5,
                1.0 - 1e-16,
                1.0,
                1.000_000_000_000_001,
                2.0,
                f64::INFINITY,
            ],
        ),
        (
            "ellipkm1",
            &[-1.0, -0.0, 0.0, 1e-300, 0.5, 1.0, 2.0, f64::INFINITY],
        ),
        (
            "spence",
            &[f64::NEG_INFINITY, -1.0, -1e-300, -0.0, 0.0, 0.5, 1.0, 2.0],
        ),
    ];
    let mut rows = Vec::new();
    for (func, xs) in unary {
        for &x in xs {
            rows.push(GridRow {
                func,
                args: vec![x],
            });
        }
    }
    let a_grid = [
        f64::NEG_INFINITY,
        -2.0,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        2.0,
        f64::INFINITY,
    ];
    let x_grid = [
        f64::NEG_INFINITY,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        2.0,
        f64::INFINITY,
    ];
    for func in ["gammainc", "gammaincc"] {
        for &a in &a_grid {
            for &x in &x_grid {
                rows.push(GridRow {
                    func,
                    args: vec![a, x],
                });
            }
        }
    }
    for n in [-2.0, 0.0, 1.0, 2.0, 5.0] {
        for x in [-10.0, -1.0, -0.0, 0.0, 0.5, 1.0, 10.0] {
            rows.push(GridRow {
                func: "yn",
                args: vec![n, x],
            });
        }
    }
    rows
}

fn call(func: &str, args: &[SpecialTensor]) -> SpecialResult {
    let m = RuntimeMode::Strict;
    match func {
        "gamma" => fsci_special::gamma(&args[0], m),
        "gammaln" => fsci_special::gammaln(&args[0], m),
        "digamma" => fsci_special::digamma(&args[0], m),
        "psi" => fsci_special::psi(&args[0], m),
        "loggamma" => fsci_special::loggamma(&args[0], m),
        "ndtri" => fsci_special::ndtri(&args[0], m),
        "erfinv" => fsci_special::erfinv(&args[0], m),
        "erfcinv" => fsci_special::erfcinv(&args[0], m),
        "y0" => fsci_special::y0(&args[0], m),
        "y1" => fsci_special::y1(&args[0], m),
        "k0" => fsci_special::k0(&args[0], m),
        "k1" => fsci_special::k1(&args[0], m),
        "ellipk" => fsci_special::ellipk(&args[0], m),
        "ellipkm1" => fsci_special::ellipkm1(&args[0], m),
        "spence" => fsci_special::spence(&args[0], m),
        "gammainc" => fsci_special::gammainc(&args[0], &args[1], m),
        "gammaincc" => fsci_special::gammaincc(&args[0], &args[1], m),
        "yn" => fsci_special::yn(&args[0], &args[1], m),
        other => unreachable!("grid() only names bound functions, not {other}"),
    }
}

/// fsci's answer for one grid row under `errstate(all=raise)`: the raised code, or a
/// description of any other error (which never matches SciPy's answer).
fn fsci_raised(row: &GridRow) -> Option<String> {
    let _guard = errstate(&SpecialErrConfig::all(SpecialErrMode::Raise));
    let args: Vec<SpecialTensor> = row
        .args
        .iter()
        .map(|&v| SpecialTensor::RealScalar(v))
        .collect();
    match call(row.func, &args) {
        Ok(_) => None,
        Err(e) => Some(match e.kind {
            SpecialErrorKind::Errstate(code) => code.scipy_message().to_string(),
            other => format!("fsci error {other:?}: {}", e.detail),
        }),
    }
}

/// SpecialFunctionWarnings from one vectorised call over a function's grid under
/// `errstate(all=warn)`, one warning per failing element as in SciPy.
fn fsci_warn_count(func: &str, rows: &[&GridRow]) -> usize {
    let _guard = errstate(&SpecialErrConfig::all(SpecialErrMode::Warn));
    let columns = rows[0].args.len();
    let args: Vec<SpecialTensor> = (0..columns)
        .map(|c| SpecialTensor::RealVec(rows.iter().map(|r| r.args[c]).collect()))
        .collect();
    let (result, warnings) = catch_warnings(|| call(func, &args));
    assert!(
        result.is_ok(),
        "{func}: warn mode must not fail: {result:?}"
    );
    warnings
        .iter()
        .filter(|w| w.category == fsci_runtime::WarningCategory::SpecialFunctionWarning)
        .count()
}

fn scipy_oracle(rows: &[GridRow]) -> Option<Oracle> {
    let script = format!(
        r#"
import json, math, sys, warnings
import numpy as np
from scipy import special as sc, stats, optimize, integrate, signal

SCIPY = {SCIPY_CLASSES}
funcs = {{"gamma": sc.gamma, "gammaln": sc.gammaln, "digamma": sc.digamma, "psi": sc.psi,
          "loggamma": sc.loggamma, "ndtri": sc.ndtri, "erfinv": sc.erfinv,
          "erfcinv": sc.erfcinv, "y0": sc.y0, "y1": sc.y1, "k0": sc.k0, "k1": sc.k1,
          "ellipk": sc.ellipk, "ellipkm1": sc.ellipkm1, "spence": sc.spence,
          "gammainc": sc.gammainc, "gammaincc": sc.gammaincc, "yn": sc.yn}}

def raised(f, args):
    with sc.errstate(all="raise"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f(*args)
            return None
        except sc.SpecialFunctionError as e:
            return str(e).split(": ", 1)[1]

# JSON has no infinities, so the arguments arrive as Rust's round-trip float strings.
rows = [{{"func": r["func"], "args": [float(a) for a in r["args"]]}} for r in json.load(sys.stdin)]
out_raised = [raised(funcs[r["func"]], [int(a) if r["func"] == "yn" and i == 0 else a
                                        for i, a in enumerate(r["args"])]) for r in rows]
warn_counts = []
for name in dict.fromkeys(r["func"] for r in rows):
    mine = [r["args"] for r in rows if r["func"] == name]
    cols = [np.array([a[c] for a in mine]) for c in range(len(mine[0]))]
    if name == "yn":
        cols[0] = cols[0].astype(int)
    with sc.errstate(all="warn"):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            funcs[name](*cols)
    warn_counts.append([name, sum(1 for x in w if x.category.__name__ == "SpecialFunctionWarning")])

def scenario(name, f):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            values = f()
            classes = []
        except Exception as e:
            values, classes = [], ["raise:" + type(e).__name__]
    classes = [x.category.__name__ for x in w if x.category.__name__ in SCIPY] + classes
    vals = [None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
            for v in values]
    return {{"name": name, "classes": classes, "values": vals}}

mean = lambda x, axis=-1: np.mean(x, axis=axis)
scen = [
    ("pearsonr_constant", lambda: list(stats.pearsonr([0.1, 0.1, 0.1], [1.0, 2.0, 3.0]))),
    ("pearsonr_near_constant", lambda: [stats.pearsonr([1.0, 1.0 + 1e-15, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0])[0]]),
    ("pearsonr_ordinary", lambda: [stats.pearsonr([1.0, 3.0, 2.0], [1.0, 2.0, 3.0])[0]]),
    ("spearmanr_constant", lambda: list(stats.spearmanr([2.0, 2.0, 2.0, 2.0], [1.0, 2.0, 3.0, 4.0]))),
    ("pointbiserialr_constant", lambda: list(stats.pointbiserialr([0.0, 1.0, 0.0, 1.0], [2.0, 2.0, 2.0, 2.0]))),
    ("bootstrap_bca_degenerate", lambda: list(stats.bootstrap(([1.0] * 5,), mean, n_resamples=31, method="BCa", rng=0).confidence_interval)),
    ("bootstrap_percentile_degenerate", lambda: list(stats.bootstrap(([1.0] * 5,), mean, n_resamples=31, method="percentile", rng=0).confidence_interval)),
    ("curve_fit_singular", lambda: [optimize.curve_fit(lambda x, a, b: a * x + 0.0 * b, [1.0, 2.0, 3.0], [2.0, 4.0, 6.0])[0][0]]),
    ("curve_fit_zero_dof", lambda: list(optimize.curve_fit(lambda x, a, b: a * x + b, [0.0, 1.0], [1.0, 3.0], p0=[1.0, 1.0])[0])),
    ("quad_limit", lambda: [integrate.quad(lambda x: math.sin(1.0 / x) if x else 0.0, 0.0, 1.0, limit=5)[0]]),
    ("quad_ordinary", lambda: [integrate.quad(lambda x: x * x, 0.0, 1.0)[0]]),
    ("normalize_bad", lambda: list(signal.normalize([1e-16, 1.0, 2.0], [1.0, 2.0])[0])),
    ("normalize_ordinary", lambda: list(signal.normalize([1e-13, 1.0], [1.0, 2.0])[0])),
    ("tf2zpk_bad", lambda: list(np.real(signal.tf2zpk([1e-16, 1.0, 2.0], [1.0, 2.0, 3.0])[0]))),
    ("lp2lp_bad", lambda: list(signal.lp2lp([0.0, 1.0], [1.0, 1.0], 2.0)[0])),
    ("broyden1_no_root", lambda: list(optimize.broyden1(lambda x: x * x + 1.0, [1.0], maxiter=5))),
]
print(json.dumps({{"raised": out_raised, "warn_counts": warn_counts,
                  "scenarios": [scenario(n, f) for n, f in scen]}}))
"#
    );
    // `{:?}` is the shortest round-trip form and spells the infinities `inf`/`-inf`, which
    // JSON numbers cannot (serde_json would send them as null).
    let wire: Vec<serde_json::Value> = rows
        .iter()
        .map(|r| {
            let args: Vec<String> = r.args.iter().map(|v| format!("{v:?}")).collect();
            serde_json::json!({ "func": r.func, "args": args })
        })
        .collect();
    let query = serde_json::to_string(&wire).expect("serialize grid");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(&script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn the warnings oracle: {e}"
            );
            eprintln!("skipping warnings oracle: python not available ({e})");
            return None;
        }
    };
    child
        .stdin
        .as_mut()
        .expect("oracle stdin")
        .write_all(query.as_bytes())
        .expect("write oracle query");
    let output = child
        .wait_with_output()
        .expect("wait for the warnings oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "warnings oracle failed: {stderr}"
        );
        eprintln!("skipping warnings oracle: scipy not available\n{stderr}");
        return None;
    }
    Some(serde_json::from_slice(&output.stdout).expect("parse warnings oracle JSON"))
}

/// fsci's side of a warning scenario: the SciPy class names raised and the values.
fn fsci_scenario(name: &str) -> (Vec<String>, Vec<Option<f64>>) {
    use fsci_stats::{BootstrapIntervalMethod, BootstrapMethod};
    let finite = |v: f64| (!v.is_nan()).then_some(v);
    let (values, warnings) = catch_warnings(|| -> Result<Vec<Option<f64>>, String> {
        Ok(match name {
            "pearsonr_constant" => {
                let r = fsci_stats::pearsonr(&[0.1, 0.1, 0.1], &[1.0, 2.0, 3.0]);
                vec![finite(r.statistic), finite(r.pvalue)]
            }
            "pearsonr_near_constant" => {
                let r = fsci_stats::pearsonr(&[1.0, 1.0 + 1e-15, 1.0, 1.0], &[1.0, 2.0, 3.0, 4.0]);
                vec![finite(r.statistic)]
            }
            "pearsonr_ordinary" => {
                vec![finite(
                    fsci_stats::pearsonr(&[1.0, 3.0, 2.0], &[1.0, 2.0, 3.0]).statistic,
                )]
            }
            "spearmanr_constant" => {
                let r = fsci_stats::spearmanr(&[2.0; 4], &[1.0, 2.0, 3.0, 4.0]);
                vec![finite(r.statistic), finite(r.pvalue)]
            }
            "pointbiserialr_constant" => {
                let r = fsci_stats::pointbiserialr(&[0.0, 1.0, 0.0, 1.0], &[2.0; 4]);
                vec![finite(r.statistic), finite(r.pvalue)]
            }
            "bootstrap_bca_degenerate" | "bootstrap_percentile_degenerate" => {
                let interval = if name.contains("bca") {
                    BootstrapIntervalMethod::Bca
                } else {
                    BootstrapIntervalMethod::Percentile
                };
                let method =
                    BootstrapMethod::new(31, None, 0, interval).map_err(|e| e.to_string())?;
                let mean = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
                let r = fsci_stats::bootstrap(&[1.0; 5], mean, 0.95, &method)
                    .map_err(|e| e.to_string())?;
                vec![
                    finite(r.confidence_interval.0),
                    finite(r.confidence_interval.1),
                ]
            }
            "curve_fit_singular" => {
                let r = fsci_opt::curve_fit(
                    |x: f64, p: &[f64]| p[0] * x + 0.0 * p[1],
                    &[1.0, 2.0, 3.0],
                    &[2.0, 4.0, 6.0],
                    // SciPy's default p0 (ones); fsci cannot infer the arity of a closure.
                    fsci_opt::CurveFitOptions {
                        p0: Some(vec![1.0, 1.0]),
                        ..fsci_opt::CurveFitOptions::default()
                    },
                )
                .map_err(|e| e.to_string())?;
                vec![finite(r.popt[0])]
            }
            "curve_fit_zero_dof" => {
                let r = fsci_opt::curve_fit(
                    |x: f64, p: &[f64]| p[0] * x + p[1],
                    &[0.0, 1.0],
                    &[1.0, 3.0],
                    fsci_opt::CurveFitOptions {
                        p0: Some(vec![1.0, 1.0]),
                        ..fsci_opt::CurveFitOptions::default()
                    },
                )
                .map_err(|e| e.to_string())?;
                r.popt.iter().map(|&v| finite(v)).collect()
            }
            "quad_limit" => {
                let r = fsci_integrate::quad(
                    |x: f64| if x == 0.0 { 0.0 } else { (1.0 / x).sin() },
                    0.0,
                    1.0,
                    fsci_integrate::QuadOptions {
                        limit: 5,
                        ..fsci_integrate::QuadOptions::default()
                    },
                )
                .map_err(|e| e.to_string())?;
                vec![finite(r.integral)]
            }
            "quad_ordinary" => {
                let r = fsci_integrate::quad(|x: f64| x * x, 0.0, 1.0, Default::default())
                    .map_err(|e| e.to_string())?;
                vec![finite(r.integral)]
            }
            "normalize_bad" | "normalize_ordinary" => {
                let b: &[f64] = if name == "normalize_bad" {
                    &[1e-16, 1.0, 2.0]
                } else {
                    &[1e-13, 1.0]
                };
                let (num, _) = fsci_signal::normalize(b, &[1.0, 2.0]).map_err(|e| e.to_string())?;
                num.iter().map(|&v| finite(v)).collect()
            }
            "tf2zpk_bad" => {
                let zpk = fsci_signal::tf2zpk(&[1e-16, 1.0, 2.0], &[1.0, 2.0, 3.0])
                    .map_err(|e| e.to_string())?;
                zpk.zeros_re.iter().map(|&v| finite(v)).collect()
            }
            "lp2lp_bad" => {
                let (b, _) =
                    fsci_signal::lp2lp(&[0.0, 1.0], &[1.0, 1.0], 2.0).map_err(|e| e.to_string())?;
                b.iter().map(|&v| finite(v)).collect()
            }
            "broyden1_no_root" => {
                match fsci_opt::broyden1(|x: &[f64]| vec![x[0] * x[0] + 1.0], &[1.0], 1e-10, 5) {
                    Err(fsci_opt::OptError::NoConvergence { .. }) => {
                        return Err("raise:NoConvergence".into());
                    }
                    other => return Err(format!("unexpected: {other:?}")),
                }
            }
            other => unreachable!("the oracle only runs this file's scenarios, not {other}"),
        })
    });
    let mut classes: Vec<String> = warnings
        .iter()
        .map(|w| w.category.scipy_name().to_string())
        .collect();
    let values = match values {
        Ok(v) => v,
        Err(raised) => {
            classes.push(raised);
            Vec::new()
        }
    };
    (classes, values)
}

fn values_agree(scenario: &str, fsci: &[Option<f64>], scipy: &[Option<f64>]) -> bool {
    let close = |a: f64, b: f64| {
        if scenario.starts_with("curve_fit") {
            (a - b).abs() <= FIT_VALUE_ABS_TOL
        } else {
            (a - b).abs() <= VALUE_REL_TOL * a.abs().max(b.abs())
        }
    };
    fsci.len() == scipy.len()
        && fsci.iter().zip(scipy).all(|(a, b)| match (a, b) {
            (None, None) => true,
            (Some(a), Some(b)) => a == b || close(*a, *b),
            _ => false,
        })
}

#[test]
fn diff_scipy_warnings_errstate() {
    let rows = grid();
    let Some(oracle) = scipy_oracle(&rows) else {
        return;
    };
    assert_eq!(
        oracle.raised.len(),
        rows.len(),
        "oracle answered a different grid"
    );

    let mut failures = Vec::new();
    let mut scipy_raises = 0usize;
    let mut fsci_raises = 0usize;
    for (row, scipy) in rows.iter().zip(&oracle.raised) {
        let fsci = fsci_raised(row);
        scipy_raises += usize::from(scipy.is_some());
        fsci_raises += usize::from(fsci.is_some());
        if &fsci != scipy {
            failures.push(format!(
                "errstate {}{:?}: scipy raises {scipy:?}, fsci {fsci:?}",
                row.func, row.args
            ));
        }
    }

    let mut funcs: Vec<&str> = rows.iter().map(|r| r.func).collect();
    funcs.dedup();
    for (func, scipy_count) in &oracle.warn_counts {
        let mine: Vec<&GridRow> = rows.iter().filter(|r| r.func == func.as_str()).collect();
        let fsci_count = fsci_warn_count(func, &mine);
        if fsci_count != *scipy_count {
            failures.push(format!(
                "errstate warn {func}: scipy {scipy_count} SpecialFunctionWarnings, fsci {fsci_count}"
            ));
        }
    }
    assert_eq!(oracle.warn_counts.len(), funcs.len());

    assert_eq!(oracle.scenarios.len(), WARNING_SCENARIOS);
    let mut warned_scenarios = 0usize;
    for s in &oracle.scenarios {
        let (classes, values) = fsci_scenario(&s.name);
        warned_scenarios += usize::from(!s.classes.is_empty());
        eprintln!(
            "{}: scipy {:?} {:?} | fsci {classes:?} {values:?}",
            s.name, s.classes, s.values
        );
        if classes != s.classes {
            failures.push(format!(
                "{}: scipy raised {:?}, fsci {classes:?}",
                s.name, s.classes
            ));
        }
        if !values_agree(&s.name, &values, &s.values) {
            failures.push(format!(
                "{}: scipy values {:?}, fsci {values:?}",
                s.name, s.values
            ));
        }
    }

    eprintln!(
        "errstate: {} points, scipy raises on {scipy_raises}, fsci on {fsci_raises}; \
         {} functions warn-counted; {} scenarios, {warned_scenarios} of them warn or raise",
        rows.len(),
        funcs.len(),
        oracle.scenarios.len()
    );
    assert!(
        failures.is_empty(),
        "{} disagreements with SciPy:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(
        rows.len() >= MIN_ERRSTATE_POINTS,
        "grid shrank to {}",
        rows.len()
    );
    assert!(
        scipy_raises >= MIN_ERRSTATE_RAISES && rows.len() - scipy_raises >= MIN_ERRSTATE_RAISES,
        "grid lost its raising or its quiet points: {scipy_raises} of {}",
        rows.len()
    );
    // Four scenarios are the quiet neighbours; the rest must warn or raise on both sides.
    assert_eq!(warned_scenarios, WARNING_SCENARIOS - 4);
}
