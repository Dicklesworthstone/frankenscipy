#![forbid(unsafe_code)]
//! Live SciPy regression diff for the moments and modes frankenscipy-szq1n.14 fixed.
//!
//! WrapCauchy's var/skew/kurt returned NaN behind a comment claiming SciPy does too (SciPy gives
//! finite linear moments on [0, 2π)); GeneralizedExponential, BetaNegativeBinomial and
//! NegHypergeometric hard-coded `mode = 0`. This compares, against the pinned SciPy computed at run
//! time:
//! - WrapCauchy(c), c ∈ {0.1, 0.5, 0.9}: mean, var, skewness, kurtosis vs `wrapcauchy(c).stats('mvsk')`;
//! - modes at parameter sets other than the unit tests' (fresh subjects, AGENTS #12): the
//!   discrete families against SciPy's pmf over the support, exactly: fsci's mode must be one of
//!   the k whose pmf is the maximum (within 1e-12 relative, so an exact tie such as
//!   betanbinom(12, 1.5, 6) at k = 21 and 22 — pmf ratio exactly 1 — admits both, where
//!   `np.argmax` picks 21 only by rounding); genexpon against a grid argmax of SciPy's pdf, refined
//!   by bounded minimize_scalar and then by a root of the central-difference derivative of
//!   `logpdf` (a flat maximum limits minimize_scalar alone to ~1e-8, the tolerance itself).
//!
//! The compared count is asserted.

use std::io::Write;
use std::process::Stdio;

use fsci_conformance::CompareLedger;
use fsci_stats::{
    BetaNegativeBinomial, ContinuousDistribution, DiscreteDistribution, GeneralizedExponential,
    NegHypergeometric, WrapCauchy,
};
use serde::{Deserialize, Serialize};

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
/// WrapCauchy linear moments. The acceptance asked for rel ≤ 1e-9; measured against pinned SciPy
/// 1.17.1 the worst is 7.5e-15, so this holds 100x margin instead.
const MOMENT_REL_TOL: f64 = 1e-12;
/// Continuous mode location. The acceptance asked for 1e-8 abs; the refined reference agrees with
/// fsci's closed form to 3.9e-12 at worst, so this holds 25x margin.
const MODE_ABS_TOL: f64 = 1e-10;

const WRAPCAUCHY_C: [f64; 3] = [0.1, 0.5, 0.9];
const GENEXPON: [(f64, f64, f64); 4] = [
    (0.7, 3.0, 2.0),
    (1.5, 2.5, 4.0),
    (0.2, 6.0, 1.2),
    (3.0, 1.0, 2.0), // b·c <= a²: the density decreases from 0, mode 0
];
const BETANBINOM: [(u64, f64, f64); 4] =
    [(7, 3.0, 4.0), (12, 1.5, 6.0), (3, 0.8, 2.5), (20, 5.0, 1.5)];
const NHYPERGEOM: [(u64, u64, u64); 3] = [(25, 9, 6), (40, 12, 10), (18, 4, 3)];

#[derive(Debug, Serialize)]
struct Query {
    wrapcauchy: Vec<f64>,
    genexpon: Vec<(f64, f64, f64)>,
    betanbinom: Vec<(u64, f64, f64)>,
    nhypergeom: Vec<(u64, u64, u64)>,
}

#[derive(Debug, Deserialize)]
struct Answer {
    wrapcauchy: Vec<[f64; 4]>,
    genexpon: Vec<f64>,
    /// Every k at the pmf maximum (a tie has more than one).
    betanbinom: Vec<Vec<f64>>,
    nhypergeom: Vec<Vec<f64>>,
}

fn scipy_answer(query: &Query) -> Option<Answer> {
    let script = r#"
import json, sys
import numpy as np
from scipy import stats
from scipy.optimize import brentq, minimize_scalar

def modes(pmf, k):
    p = pmf(k)
    return [float(v) for v in k[p >= p.max() * (1 - 1e-12)]]

q = json.load(sys.stdin)
out = {"wrapcauchy": [], "genexpon": [], "betanbinom": [], "nhypergeom": []}
for c in q["wrapcauchy"]:
    out["wrapcauchy"].append([float(v) for v in stats.wrapcauchy(c).stats("mvsk")])
for a, b, c in q["genexpon"]:
    d = stats.genexpon(a, b, c)
    xs = np.linspace(0.0, float(d.ppf(0.999)), 20001)
    i = int(np.argmax(d.pdf(xs)))
    if i == 0:
        mode = 0.0
    else:
        lo, hi = xs[i - 1], xs[min(i + 1, len(xs) - 1)]
        mode = float(minimize_scalar(lambda x: -d.pdf(x), bounds=(lo, hi), method="bounded",
                                     options={"xatol": 1e-13}).x)
        dlog = lambda x, h=1e-6: (d.logpdf(x + h) - d.logpdf(x - h)) / (2 * h)
        lo2, hi2 = max(mode - 1e-4, 1e-9), mode + 1e-4
        if dlog(lo2) > 0 > dlog(hi2):
            mode = float(brentq(dlog, lo2, hi2, xtol=1e-15))
    out["genexpon"].append(mode)
for n, a, b in q["betanbinom"]:
    d = stats.betanbinom(n, a, b)
    out["betanbinom"].append(modes(d.pmf, np.arange(0, int(d.ppf(0.99999)) + 2)))
for M, n, r in q["nhypergeom"]:
    d = stats.nhypergeom(M, n, r)
    out["nhypergeom"].append(modes(d.pmf, np.arange(0, M - n + 1)))
print(json.dumps(out))
"#;
    let query_json = serde_json::to_string(query).expect("serialize query");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn the moments/modes oracle: {e}"
            );
            eprintln!("skipping moments/modes oracle: python not available ({e})");
            return None;
        }
    };
    child
        .stdin
        .as_mut()
        .expect("oracle stdin")
        .write_all(query_json.as_bytes())
        .expect("write oracle query");
    let output = child.wait_with_output().expect("wait for the oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "moments/modes oracle failed: {stderr}"
        );
        eprintln!("skipping moments/modes oracle: scipy not available\n{stderr}");
        return None;
    }
    Some(serde_json::from_slice(&output.stdout).expect("parse oracle JSON"))
}

fn rel(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs().max(1.0)
}

#[test]
fn diff_stats_moments_modes_regressions() {
    let query = Query {
        wrapcauchy: WRAPCAUCHY_C.to_vec(),
        genexpon: GENEXPON.to_vec(),
        betanbinom: BETANBINOM.to_vec(),
        nhypergeom: NHYPERGEOM.to_vec(),
    };
    let Some(answer) = scipy_answer(&query) else {
        return;
    };
    let mut compared = 0;
    let mut failures = Vec::new();
    let mut ledger = CompareLedger::new(
        "diff_stats_moments_modes_regressions",
        &[
            "wrapcauchy.mean",
            "wrapcauchy.var",
            "wrapcauchy.skew",
            "wrapcauchy.kurt",
            "genexpon.mode",
            "betanbinom.mode",
            "nhypergeom.mode",
        ],
    );

    for (&c, want) in WRAPCAUCHY_C.iter().zip(&answer.wrapcauchy) {
        let d = WrapCauchy::new(c);
        let got = [d.mean(), d.var(), d.skewness(), d.kurtosis()];
        let case_id = format!("c{c}");
        for (name, (g, w)) in ["mean", "var", "skew", "kurt"]
            .iter()
            .zip(got.iter().zip(want))
        {
            let r = rel(*g, *w);
            println!("wrapcauchy({c}) {name}: fsci {g:e} SciPy {w:e} rel {r:.1e}");
            compared += 1;
            let arm = format!("wrapcauchy.{name}");
            // A non-finite fsci moment is recorded by the ledger as an fsci failure.
            let Some((w, g)) = ledger.pair(&arm, &case_id, Some(*w), Some(*g)) else {
                continue;
            };
            let pass = !(r.is_nan() || r > MOMENT_REL_TOL);
            ledger.compared(&arm, &case_id, pass);
            if !pass {
                failures.push(format!("wrapcauchy({c}) {name}: {g:e} vs {w:e}"));
            }
        }
    }
    for (&(a, b, c), &want) in GENEXPON.iter().zip(&answer.genexpon) {
        let got = GeneralizedExponential::new(a, b, c).mode();
        println!("genexpon({a},{b},{c}) mode: fsci {got} SciPy argmax {want}");
        compared += 1;
        let case_id = format!("a{a}_b{b}_c{c}");
        let Some((want, got)) = ledger.pair("genexpon.mode", &case_id, Some(want), Some(got))
        else {
            continue;
        };
        let err = (got - want).abs();
        let pass = !(err.is_nan() || err > MODE_ABS_TOL);
        ledger.compared("genexpon.mode", &case_id, pass);
        if !pass {
            failures.push(format!("genexpon({a},{b},{c}) mode {got} vs {want}"));
        }
    }
    for (&(n, a, b), want) in BETANBINOM.iter().zip(&answer.betanbinom) {
        let got = BetaNegativeBinomial::new(n, a, b).mode();
        println!("betanbinom({n},{a},{b}) mode: fsci {got} SciPy pmf maximum at {want:?}");
        compared += 1;
        let case_id = format!("n{n}_a{a}_b{b}");
        // The SciPy side is the set of k at the pmf maximum; a NaN mode is in no set.
        let Some((want, got)) = ledger.both("betanbinom.mode", &case_id, Some(want), Some(got))
        else {
            continue;
        };
        let pass = want.contains(&got);
        ledger.compared("betanbinom.mode", &case_id, pass);
        if !pass {
            failures.push(format!("betanbinom({n},{a},{b}) mode {got} vs {want:?}"));
        }
    }
    for (&(m, n, r), want) in NHYPERGEOM.iter().zip(&answer.nhypergeom) {
        let got = NegHypergeometric::new(m, n, r).mode();
        println!("nhypergeom({m},{n},{r}) mode: fsci {got} SciPy pmf maximum at {want:?}");
        compared += 1;
        let case_id = format!("M{m}_n{n}_r{r}");
        let Some((want, got)) = ledger.both("nhypergeom.mode", &case_id, Some(want), Some(got))
        else {
            continue;
        };
        let pass = want.contains(&got);
        ledger.compared("nhypergeom.mode", &case_id, pass);
        if !pass {
            failures.push(format!("nhypergeom({m},{n},{r}) mode {got} vs {want:?}"));
        }
    }

    let expected = 4 * WRAPCAUCHY_C.len() + GENEXPON.len() + BETANBINOM.len() + NHYPERGEOM.len();
    assert_eq!(compared, expected, "every row must be compared");
    assert!(
        failures.is_empty(),
        "moments/modes disagree with SciPy: {failures:#?}"
    );
    // Each arm must compare every row of its family; the smallest family sets the floor.
    ledger.finish(
        query
            .wrapcauchy
            .len()
            .min(query.genexpon.len())
            .min(query.betanbinom.len())
            .min(query.nhypergeom.len()),
    );
}
