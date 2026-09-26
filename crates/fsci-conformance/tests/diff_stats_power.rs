#![forbid(unsafe_code)]
//! Live SciPy diff for `fsci_stats::power` / `power_simulate` (frankenscipy-szq1n.11).
//!
//! `power` used to take no alternative generator, always sampled N(0, 1) and used 0.05, so for a
//! location test it returned the test's SIZE (~0.05) whatever the effect. The target here is a
//! one-sample t-test against 0 with n = 20 draws from N(0.5, 1), 20 000 resamples:
//! - its exact power from the noncentral t (`nct.sf(t_crit) + nct.cdf(-t_crit)`, df 19,
//!   nc 0.5·√20), computed by the pinned SciPy — 0.5645 at significance 0.05;
//! - `scipy.stats.power`'s own simulation of the same thing.
//!
//! fsci's estimate must agree with the analytic value, and with SciPy's estimate, within
//! POWER_SE_TOL binomial standard errors (seeded, so a row passes or fails deterministically);
//! under N(0, 1) it must estimate the size, the significance level itself; `power`'s default
//! significance is SciPy's 0.01. A second test checks that the same seed gives bit-identical
//! p-values and a different seed does not.

use std::process::Stdio;

use fsci_conformance::CompareLedger;
use fsci_stats::{PowerResult, power, power_simulate, ttest_1samp};
use serde::Deserialize;

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
/// Agreement in binomial standard errors sqrt(p(1-p)/n_resamples).
const POWER_SE_TOL: f64 = 3.0;
const N_OBS: usize = 20;
const N_RESAMPLES: usize = 20_000;
const EFFECT: f64 = 0.5;

/// splitmix64 uniforms feeding a Box-Muller normal sampler.
struct Normal {
    state: u64,
    spare: Option<f64>,
}

impl Normal {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    fn uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // (0, 1]: never 0, so ln is finite.
        ((z >> 11) + 1) as f64 / (1_u64 << 53) as f64
    }

    fn next(&mut self) -> f64 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let (u1, u2) = (self.uniform(), self.uniform());
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }
}

fn sampler(seed: u64, mean: f64) -> impl FnMut(usize) -> Vec<f64> {
    let mut normal = Normal::new(seed);
    move |n| (0..n).map(|_| mean + normal.next()).collect()
}

fn t_test(sample: &[f64]) -> f64 {
    ttest_1samp(sample, 0.0).pvalue
}

fn se(p: f64) -> f64 {
    (p * (1.0 - p) / N_RESAMPLES as f64).sqrt()
}

#[derive(Debug, Deserialize)]
struct Oracle {
    analytic_05: f64,
    analytic_01: f64,
    scipy_power_05: f64,
    scipy_size_05: f64,
}

fn scipy_oracle() -> Option<Oracle> {
    let script = r#"
import json
import numpy as np
from scipy import stats

n, eff, R = 20, 0.5, 20000
def analytic(alpha):
    tc = stats.t.ppf(1 - alpha / 2, n - 1)
    nc = eff * np.sqrt(n)
    return float(stats.nct.sf(tc, n - 1, nc) + stats.nct.cdf(-tc, n - 1, nc))
test = lambda x, axis=-1: stats.ttest_1samp(x, 0.0, axis=axis).pvalue
rng = np.random.default_rng(12345)
alt = stats.power(test, lambda size: rng.normal(eff, 1.0, size=size), n,
                  significance=0.05, n_resamples=R, vectorized=True)
rng0 = np.random.default_rng(54321)
null = stats.power(test, lambda size: rng0.normal(0.0, 1.0, size=size), n,
                   significance=0.05, n_resamples=R, vectorized=True)
print(json.dumps({"analytic_05": analytic(0.05), "analytic_01": analytic(0.01),
                  "scipy_power_05": float(alt.power), "scipy_size_05": float(null.power)}))
"#;
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
                "failed to spawn the power oracle: {e}"
            );
            eprintln!("skipping power oracle: python not available ({e})");
            return None;
        }
    };
    drop(child.stdin.take()); // the script reads no input
    let output = child.wait_with_output().expect("wait for the power oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "power oracle failed: {stderr}"
        );
        eprintln!("skipping power oracle: scipy not available\n{stderr}");
        return None;
    }
    Some(serde_json::from_slice(&output.stdout).expect("parse power oracle JSON"))
}

#[test]
fn power_is_deterministic_for_a_seed() {
    let run = |seed| power_simulate(t_test, sampler(seed, EFFECT), N_OBS, 2_000, 0.05);
    let (a, b, c): (PowerResult, PowerResult, PowerResult) = (run(7), run(7), run(8));
    let bits = |r: &PowerResult| r.pvalues.iter().map(|p| p.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&a),
        bits(&b),
        "same seed must give bit-identical p-values"
    );
    assert_eq!(a.power.to_bits(), b.power.to_bits());
    // Must-miss: a different seed gives different p-values, so the check above can fail.
    assert_ne!(
        bits(&a),
        bits(&c),
        "a different seed gave the same p-values"
    );
}

#[test]
fn diff_stats_power() {
    let Some(oracle) = scipy_oracle() else {
        return;
    };
    let alt = power_simulate(t_test, sampler(2026, EFFECT), N_OBS, N_RESAMPLES, 0.05);
    let null = power_simulate(t_test, sampler(2027, 0.0), N_OBS, N_RESAMPLES, 0.05);
    let default_level = power(t_test, sampler(2028, EFFECT), N_OBS, N_RESAMPLES);

    // (label, estimate, target, standard error of the comparison)
    let rows = [
        (
            "fsci power @0.05 vs noncentral-t",
            alt.power,
            oracle.analytic_05,
            se(oracle.analytic_05),
        ),
        (
            "SciPy power @0.05 vs noncentral-t (oracle sanity)",
            oracle.scipy_power_05,
            oracle.analytic_05,
            se(oracle.analytic_05),
        ),
        (
            "fsci vs scipy.stats.power @0.05",
            alt.power,
            oracle.scipy_power_05,
            std::f64::consts::SQRT_2 * se(oracle.analytic_05),
        ),
        ("fsci size under N(0,1) vs 0.05", null.power, 0.05, se(0.05)),
        (
            "SciPy size under N(0,1) vs 0.05",
            oracle.scipy_size_05,
            0.05,
            se(0.05),
        ),
        (
            "fsci power() default significance 0.01 vs noncentral-t",
            default_level.power,
            oracle.analytic_01,
            se(oracle.analytic_01),
        ),
    ];
    // One arm; each row is a case. The target is the reference side (the analytic value, SciPy's
    // estimate, or the nominal size) and the estimate the side under test.
    let mut ledger = CompareLedger::new("diff_stats_power", &["power"]);
    let mut failures = Vec::new();
    for (label, estimate, target, sigma) in rows {
        let Some((target, estimate)) = ledger.pair("power", label, Some(target), Some(estimate))
        else {
            continue;
        };
        let z = (estimate - target).abs() / sigma;
        println!("{label}: {estimate:.5} vs {target:.5} ({z:.2} SE)");
        let fail = z.is_nan() || z > POWER_SE_TOL;
        ledger.compared("power", label, !fail);
        if fail {
            failures.push(format!("{label}: {estimate} vs {target} ({z:.2} SE)"));
        }
    }
    assert_eq!(alt.pvalues.len(), N_RESAMPLES);
    assert!(failures.is_empty(), "power disagrees: {failures:#?}");
    ledger.finish(rows.len());
}
