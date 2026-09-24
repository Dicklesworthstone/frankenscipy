#![forbid(unsafe_code)]
//! Live SciPy guard for the SciPy-name distribution aliases in fsci-stats
//! (`pub type Norm = Normal;`, `pub type Chi2 = ChiSquared;`, ...).
//!
//! br-szq1n.2: four aliases named a DIFFERENT SciPy distribution
//! (`Landau = Moyal`, `Kstwo = KsTwoBign`, `LevyStable = Levy`,
//! `VonmisesLine = VonMises`), so code written against the SciPy name got another
//! law's numbers. Each row below constructs a distribution THROUGH the alias,
//! with the parameter mapping a SciPy user would write, and compares pdf/pmf and
//! cdf with the named `scipy.stats` distribution at several points. A wrong
//! alias, or a wrong parameter convention behind a right one, fails here.

use std::collections::HashMap;
use std::io::Write;
use std::process::Stdio;

use fsci_stats::{
    Beta, Betabinom, Betanbinom, Binom, Burr, Chi2, ContinuousDistribution, Cosine,
    CosineDistribution, Dgamma, DiscreteDistribution, Dlaplace, Dweibull, Expon, Exponweib, F,
    Foldcauchy, Foldnorm, Gamma, Genexpon, Geom, GumbelL, GumbelR, HalfNormal, Halfnorm, Hypergeom,
    Invgamma, Invgauss, LevyL, Lognorm, Logser, Nbinom, Ncf, Nct, Ncx2, Nhypergeom, Norm, T,
    Triang, Truncnorm, WeibullMin,
};
use serde::{Deserialize, Serialize};

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const REL_TOL: f64 = 1.0e-9;
const ABS_TOL: f64 = 1.0e-12;

type Eval = Box<dyn Fn(f64) -> (f64, f64)>;

struct AliasCase {
    alias: &'static str,
    /// Python expression for the frozen `scipy.stats` distribution.
    scipy: &'static str,
    discrete: bool,
    points: Vec<f64>,
    /// (pdf or pmf, cdf) through the Rust alias.
    eval: Eval,
}

fn cont<D: ContinuousDistribution + 'static>(d: D) -> Eval {
    Box::new(move |x| (d.pdf(x), d.cdf(x)))
}

fn disc<D: DiscreteDistribution + 'static>(d: D) -> Eval {
    Box::new(move |k| (d.pmf(k as i64), d.cdf(k as i64)))
}

fn cases() -> Vec<AliasCase> {
    let c = |alias, scipy, points: &[f64], eval| AliasCase {
        alias,
        scipy,
        discrete: false,
        points: points.to_vec(),
        eval,
    };
    let d = |alias, scipy, points: &[f64], eval| AliasCase {
        alias,
        scipy,
        discrete: true,
        points: points.to_vec(),
        eval,
    };
    let halfnorm: Halfnorm = HalfNormal;
    let cosine: Cosine = CosineDistribution;
    vec![
        c(
            "Beta",
            "stats.beta(2, 5)",
            &[0.1, 0.3, 0.8],
            cont(Beta::new(2.0, 5.0)),
        ),
        d(
            "Betabinom",
            "stats.betabinom(10, 2, 3)",
            &[0.0, 3.0, 9.0],
            disc(Betabinom::new(10, 2.0, 3.0)),
        ),
        d(
            "Betanbinom",
            "stats.betanbinom(5, 2, 3)",
            &[0.0, 2.0, 11.0],
            disc(Betanbinom::new(5, 2.0, 3.0)),
        ),
        d(
            "Binom",
            "stats.binom(10, 0.3)",
            &[0.0, 3.0, 8.0],
            disc(Binom::new(10, 0.3)),
        ),
        c(
            "Burr",
            "stats.burr(2, 3)",
            &[0.2, 1.0, 3.5],
            cont(Burr::new(2.0, 3.0)),
        ),
        c(
            "Chi2",
            "stats.chi2(4)",
            &[0.5, 3.0, 11.0],
            cont(Chi2::new(4.0)),
        ),
        c(
            "Dgamma",
            "stats.dgamma(1.5)",
            &[-2.0, 0.4, 3.0],
            cont(Dgamma::new(1.5)),
        ),
        d(
            "Dlaplace",
            "stats.dlaplace(0.8)",
            &[0.0, 1.0, 4.0],
            disc(Dlaplace::new(0.8)),
        ),
        c(
            "Dweibull",
            "stats.dweibull(2)",
            &[-1.3, 0.2, 1.1],
            cont(Dweibull::new(2.0)),
        ),
        // fsci Exponential is parameterised by its RATE; SciPy's expon by scale.
        c(
            "Expon",
            "stats.expon(scale=0.5)",
            &[0.1, 0.7, 2.5],
            cont(Expon::new(2.0)),
        ),
        c(
            "F",
            "stats.f(5, 7)",
            &[0.3, 1.0, 4.0],
            cont(F::new(5.0, 7.0)),
        ),
        c(
            "Foldcauchy",
            "stats.foldcauchy(1.5)",
            &[0.2, 1.5, 6.0],
            cont(Foldcauchy::new(1.5)),
        ),
        c(
            "Foldnorm",
            "stats.foldnorm(1.2)",
            &[0.1, 1.0, 3.0],
            cont(Foldnorm::new(1.2)),
        ),
        c(
            "Gamma",
            "stats.gamma(2.5, scale=1.5)",
            &[0.4, 3.0, 9.0],
            cont(Gamma::new(2.5, 1.5)),
        ),
        c(
            "Genexpon",
            "stats.genexpon(1, 2, 3)",
            &[0.1, 0.5, 2.0],
            cont(Genexpon::new(1.0, 2.0, 3.0)),
        ),
        d(
            "Geom",
            "stats.geom(0.3)",
            &[1.0, 2.0, 7.0],
            disc(Geom::new(0.3)),
        ),
        c(
            "GumbelL",
            "stats.gumbel_l(loc=0.5, scale=2)",
            &[-3.0, 0.0, 2.0],
            cont(GumbelL::new(0.5, 2.0)),
        ),
        c(
            "GumbelR",
            "stats.gumbel_r(loc=0.5, scale=2)",
            &[-1.0, 0.5, 4.0],
            cont(GumbelR::new(0.5, 2.0)),
        ),
        c(
            "Halfnorm",
            "stats.halfnorm()",
            &[0.1, 1.0, 2.5],
            cont(halfnorm),
        ),
        d(
            "Hypergeom",
            "stats.hypergeom(20, 7, 12)",
            &[1.0, 4.0, 7.0],
            disc(Hypergeom::new(20, 7, 12)),
        ),
        c(
            "Invgamma",
            "stats.invgamma(3)",
            &[0.2, 0.6, 2.0],
            cont(Invgamma::new(3.0)),
        ),
        c(
            "Invgauss",
            "stats.invgauss(0.7)",
            &[0.2, 0.7, 2.0],
            cont(Invgauss::new(0.7)),
        ),
        c(
            "LevyL",
            "stats.levy_l(loc=0, scale=1.5)",
            &[-6.0, -1.0, -0.2],
            cont(LevyL::new(0.0, 1.5)),
        ),
        c(
            "Lognorm",
            "stats.lognorm(0.8, scale=2)",
            &[0.5, 2.0, 7.0],
            cont(Lognorm::new(0.8, 2.0)),
        ),
        d(
            "Logser",
            "stats.logser(0.6)",
            &[1.0, 2.0, 6.0],
            disc(Logser::new(0.6)),
        ),
        d(
            "Nbinom",
            "stats.nbinom(5, 0.4)",
            &[0.0, 5.0, 14.0],
            disc(Nbinom::new(5.0, 0.4)),
        ),
        c(
            "Ncf",
            "stats.ncf(5, 8, 1.5)",
            &[0.4, 1.2, 3.5],
            cont(Ncf::new(5.0, 8.0, 1.5)),
        ),
        c(
            "Nct",
            "stats.nct(6, 1.2)",
            &[-1.0, 1.0, 3.0],
            cont(Nct::new(6.0, 1.2)),
        ),
        c(
            "Ncx2",
            "stats.ncx2(4, 2)",
            &[0.8, 4.0, 12.0],
            cont(Ncx2::new(4.0, 2.0)),
        ),
        d(
            "Nhypergeom",
            "stats.nhypergeom(20, 7, 12)",
            &[1.0, 3.0, 6.0],
            disc(Nhypergeom::new(20, 7, 12)),
        ),
        c(
            "Norm",
            "stats.norm(1, 2)",
            &[-2.0, 1.0, 4.5],
            cont(Norm::new(1.0, 2.0)),
        ),
        c("T", "stats.t(5)", &[-2.0, 0.3, 3.0], cont(T::new(5.0))),
        c(
            "Triang",
            "stats.triang(0.3, loc=0, scale=1)",
            &[0.1, 0.3, 0.8],
            cont(Triang::new(0.0, 0.3, 1.0)),
        ),
        c(
            "Truncnorm",
            "stats.truncnorm(-1, 2)",
            &[-0.8, 0.0, 1.7],
            cont(Truncnorm::new(-1.0, 2.0)),
        ),
        c(
            "WeibullMin",
            "stats.weibull_min(1.8, scale=1.5)",
            &[0.3, 1.5, 4.0],
            cont(WeibullMin::new(1.8, 1.5)),
        ),
        c("Cosine", "stats.cosine()", &[-2.0, 0.0, 1.0], cont(cosine)),
        c(
            "Exponweib",
            "stats.exponweib(2, 1.5)",
            &[0.2, 1.0, 2.5],
            cont(Exponweib::new(2.0, 1.5)),
        ),
    ]
}

#[derive(Serialize)]
struct QueryCase<'a> {
    alias: &'a str,
    scipy: &'a str,
    discrete: bool,
    points: &'a [f64],
}

#[derive(Deserialize)]
struct OracleRow {
    alias: String,
    /// (pdf or pmf, cdf) per point; None when SciPy raised.
    values: Option<Vec<(f64, f64)>>,
}

fn scipy_values(query: &[QueryCase<'_>]) -> Option<HashMap<String, Option<Vec<(f64, f64)>>>> {
    let script = r#"
import json, sys
from scipy import stats
rows = []
for case in json.load(sys.stdin):
    try:
        dist = eval(case["scipy"], {"stats": stats})
        dens = dist.pmf if case["discrete"] else dist.pdf
        vals = [[float(dens(x)), float(dist.cdf(x))] for x in case["points"]]
        rows.append({"alias": case["alias"], "values": vals})
    except Exception:
        rows.append({"alias": case["alias"], "values": None})
print(json.dumps(rows))
"#;
    let payload = serde_json::to_string(query).expect("serialize alias query");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn the SciPy oracle: {err}"
            );
            eprintln!("skipping alias oracle: python not available ({err})");
            return None;
        }
    };
    child
        .stdin
        .as_mut()
        .expect("oracle stdin")
        .write_all(payload.as_bytes())
        .expect("write alias query");
    let output = child.wait_with_output().expect("wait for alias oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "alias oracle failed: {stderr}"
        );
        eprintln!("skipping alias oracle: scipy not available\n{stderr}");
        return None;
    }
    let rows: Vec<OracleRow> =
        serde_json::from_slice(&output.stdout).expect("parse alias oracle JSON");
    Some(rows.into_iter().map(|r| (r.alias, r.values)).collect())
}

fn close(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() <= ABS_TOL + REL_TOL * expected.abs()
}

#[test]
fn diff_stats_scipy_name_aliases() {
    let cases = cases();
    let query: Vec<QueryCase<'_>> = cases
        .iter()
        .map(|c| QueryCase {
            alias: c.alias,
            scipy: c.scipy,
            discrete: c.discrete,
            points: &c.points,
        })
        .collect();
    let Some(oracle) = scipy_values(&query) else {
        return;
    };

    let mut compared = 0usize;
    let mut failures = Vec::new();
    for case in &cases {
        let Some(Some(expected)) = oracle.get(case.alias) else {
            failures.push(format!(
                "{}: SciPy did not evaluate `{}`",
                case.alias, case.scipy
            ));
            continue;
        };
        for (&x, &(e_dens, e_cdf)) in case.points.iter().zip(expected) {
            let (dens, cdf) = (case.eval)(x);
            compared += 1;
            if !close(dens, e_dens) || !close(cdf, e_cdf) {
                failures.push(format!(
                    "{} vs {} at x={x}: density {dens:e} vs {e_dens:e}, cdf {cdf:e} vs {e_cdf:e}",
                    case.alias, case.scipy
                ));
            }
        }
    }

    let expected_points: usize = cases.iter().map(|c| c.points.len()).sum();
    assert_eq!(
        compared, expected_points,
        "aliases: compared {compared} of {expected_points} points; failures: {failures:#?}"
    );
    assert!(
        failures.is_empty(),
        "alias divergences:\n{}",
        failures.join("\n")
    );
}
