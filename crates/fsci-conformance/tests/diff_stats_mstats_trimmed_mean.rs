#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.stats.mstats.trimmed_mean(a, limits, inclusive,
//! relative)` against `fsci_stats::trimmed_mean`: relative (rank) and absolute (value) limits,
//! asymmetric and one-sided tails, both rounding flags at counts that land on `.5`, ties, NaN
//! (ranked last by SciPy's argsort), fully trimmed and empty input (SciPy's `masked`, fsci's
//! NaN), and out-of-range relative limits (ValueError in both).
//!
//! The kept values are the same set on both sides; only the order of the final summation can
//! differ (numpy's pairwise sum over the zero-filled array), so means are compared to
//! 1e-14 relative.

use std::io::Write;
use std::process::Stdio;

use fsci_conformance::CompareLedger;
use fsci_stats::{
    FastGeneratorInversion, Normal, NumericalInversePolynomial, SimpleRatioUniforms, StatsError,
    trimmed_mean,
};
use serde::{Deserialize, Serialize};

const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const ABS_TOL: f64 = 1.0e-14;
const TOL: f64 = ABS_TOL;

#[derive(Debug, Clone, Serialize)]
struct Query {
    id: String,
    a: Vec<Option<f64>>,
    lo: Option<f64>,
    up: Option<f64>,
    loin: bool,
    upin: bool,
    relative: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct Arm {
    id: String,
    /// "value" (with `v`, `None` = masked), or "error".
    kind: String,
    v: Option<f64>,
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn corpus() -> Vec<Query> {
    let mut rng = Lcg(424_242);
    let mut samples: Vec<(String, Vec<f64>)> = Vec::new();
    for &n in &[1usize, 2, 5, 10, 20, 37, 100] {
        samples.push((
            format!("u{n}"),
            (0..n).map(|_| rng.next() * 100.0 - 50.0).collect(),
        ));
    }
    // Ties and integers (rank trimming among equal values).
    samples.push((
        "ties".into(),
        vec![3.0, 1.0, 2.0, 2.0, 5.0, 5.0, 5.0, 1.0, 4.0, 2.0],
    ));
    // NaN ranks last in SciPy.
    samples.push((
        "nan".into(),
        vec![1.0, f64::NAN, 3.0, 7.0, 2.0, 9.0, 4.0, 8.0, 6.0, 5.0],
    ));
    samples.push(("empty".into(), Vec::new()));

    let relative_limits: &[(Option<f64>, Option<f64>)] = &[
        (Some(0.1), Some(0.1)),
        (Some(0.2), Some(0.05)),
        (Some(0.25), Some(0.25)),
        (Some(0.15), Some(0.35)),
        (None, Some(0.3)),
        (Some(0.3), None),
        (Some(0.0), Some(0.0)),
        (Some(0.5), Some(0.5)),
        (Some(0.6), Some(0.6)),
        (Some(1.0), None),
        (Some(-0.1), Some(0.1)),
        (Some(0.1), Some(1.2)),
    ];
    let mut out = Vec::new();
    for (name, a) in &samples {
        let a_opt: Vec<Option<f64>> = a.iter().map(|v| (!v.is_nan()).then_some(*v)).collect();
        for (li, &(lo, up)) in relative_limits.iter().enumerate() {
            for (loin, upin) in [(true, true), (false, false), (true, false), (false, true)] {
                out.push(Query {
                    id: format!("{name}_rel{li}_{loin}{upin}"),
                    a: a_opt.clone(),
                    lo,
                    up,
                    loin,
                    upin,
                    relative: true,
                });
            }
        }
        // Absolute limits at data values (so the inclusive flags matter) and between them.
        let mut sorted: Vec<f64> = a.iter().copied().filter(|v| !v.is_nan()).collect();
        sorted.sort_by(f64::total_cmp);
        if sorted.len() >= 3 {
            let q1 = sorted[sorted.len() / 4];
            let q3 = sorted[3 * sorted.len() / 4];
            let abs_limits = [
                (Some(q1), Some(q3)),
                (Some(q1 - 0.5), None),
                (None, Some(q3 + 0.5)),
                (Some(q3), Some(q1)),
            ];
            for (li, &(lo, up)) in abs_limits.iter().enumerate() {
                for (loin, upin) in [(true, true), (false, false), (true, false)] {
                    out.push(Query {
                        id: format!("{name}_abs{li}_{loin}{upin}"),
                        a: a_opt.clone(),
                        lo,
                        up,
                        loin,
                        upin,
                        relative: false,
                    });
                }
            }
        }
    }
    out
}

const ORACLE: &str = r#"
import json, sys, warnings
import numpy as np
from scipy.stats import mstats
warnings.simplefilter("ignore")
out = []
for q in json.load(sys.stdin):
    a = np.array([np.nan if v is None else v for v in q["a"]], dtype=float)
    try:
        r = mstats.trimmed_mean(a, limits=(q["lo"], q["up"]),
                                inclusive=(q["loin"], q["upin"]), relative=q["relative"])
        if np.ma.is_masked(r):
            out.append({"id": q["id"], "kind": "value", "v": None})
        else:
            v = float(r)
            out.append({"id": q["id"], "kind": "value", "v": None if np.isnan(v) else v,
                        "nan": bool(np.isnan(v))})
    except ValueError:
        out.append({"id": q["id"], "kind": "error", "v": None})
print(json.dumps(out))
"#;

#[test]
fn diff_stats_mstats_trimmed_mean() {
    let queries = corpus();
    let payload = serde_json::to_string(&queries).expect("serialize");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(ORACLE)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "spawn oracle: {e}"
            );
            eprintln!("skipping trimmed_mean oracle: {e}");
            return;
        }
    };
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(payload.as_bytes())
        .expect("write query");
    let output = child.wait_with_output().expect("oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "oracle failed: {stderr}"
        );
        eprintln!("skipping trimmed_mean oracle: {stderr}");
        return;
    }
    let arms: Vec<Arm> = serde_json::from_slice(&output.stdout).expect("oracle JSON");
    let mut ledger = CompareLedger::new("diff_stats_mstats_trimmed_mean", &["trimmed_mean"]);
    let (mut values, mut nans, mut errors) = (0usize, 0usize, 0usize);
    let mut worst = 0.0_f64;
    let mut failures = Vec::new();
    for (q, arm) in queries.iter().zip(&arms) {
        assert_eq!(q.id, arm.id);
        let a: Vec<f64> = q.a.iter().map(|v| v.unwrap_or(f64::NAN)).collect();
        let ours = trimmed_mean(&a, (q.lo, q.up), (q.loin, q.upin), q.relative);
        let pass = match (arm.kind.as_str(), &ours) {
            ("error", Err(_)) => {
                errors += 1;
                true
            }
            ("value", Ok(x)) => match arm.v {
                // SciPy's masked constant, or a NaN mean.
                None => {
                    nans += 1;
                    x.is_nan()
                }
                Some(want) => {
                    values += 1;
                    let err = (x - want).abs() / want.abs().max(1.0);
                    worst = worst.max(err);
                    err <= TOL
                }
            },
            _ => false,
        };
        if !pass {
            failures.push(format!(
                "{}: fsci {ours:?} vs SciPy {} {:?}",
                q.id, arm.kind, arm.v
            ));
        }
        ledger.compared("trimmed_mean", &q.id, pass);
    }
    for f in failures.iter().take(15) {
        println!("FAIL {f}");
    }
    println!(
        "{} cases: {values} means (worst rel err {worst:.2e}), {nans} masked/NaN, {errors} \
         ValueErrors; {} failed",
        queries.len(),
        failures.len()
    );
    assert!(failures.is_empty(), "trimmed_mean differs from SciPy");
    ledger.finish(queries.len());
}

const SAMPLING_ORACLE: &str = r#"
import json
import sys
import numpy as np
from scipy import stats
from scipy.stats import sampling

us = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
xs = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

fgi = sampling.FastGeneratorInversion(stats.norm(), domain=(-3.0, 3.0))
fgi_ppf = [float(v) for v in fgi.ppf(us)]

class StdNorm:
    def pdf(self, x):
        return float(np.exp(-0.5 * x * x))
    def dpdf(self, x):
        return float(-x * np.exp(-0.5 * x * x))

pinv = sampling.NumericalInversePolynomial(
    StdNorm(), center=0.0, domain=(-5.0, 5.0), order=5, u_resolution=1e-10
)
pinv_ppf = [float(v) for v in pinv.ppf(us)]
pinv_cdf = [float(v) for v in pinv.cdf(xs)]

sru = sampling.SimpleRatioUniforms(
    StdNorm(),
    mode=0.0,
    pdf_area=float(np.sqrt(2.0 * np.pi)),
    cdf_at_mode=0.5,
    random_state=np.random.default_rng(12345),
)
sru_samples = sru.rvs(4000)
sru_mean = float(np.mean(sru_samples))
sru_std = float(np.std(sru_samples, ddof=1))

class ZeroPdf:
    def pdf(self, x):
        return 0.0

unuran_err = False
try:
    sampling.NumericalInversePolynomial(ZeroPdf(), center=0.0, domain=(-1.0, 1.0))
except sampling.UNURANError:
    unuran_err = True

print(json.dumps({
    "us": us,
    "xs": xs,
    "fgi_ppf": fgi_ppf,
    "pinv_ppf": pinv_ppf,
    "pinv_cdf": pinv_cdf,
    "sru_mean": sru_mean,
    "sru_std": sru_std,
    "unuran_err": unuran_err,
}))
"#;

#[derive(Debug, Deserialize)]
struct SamplingOracleOut {
    us: Vec<f64>,
    xs: Vec<f64>,
    fgi_ppf: Vec<f64>,
    pinv_ppf: Vec<f64>,
    pinv_cdf: Vec<f64>,
    sru_mean: f64,
    sru_std: f64,
    unuran_err: bool,
}

#[test]
fn diff_stats_sampling_unuran() {
    let output = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(SAMPLING_ORACLE)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
    {
        Ok(o) => o,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "spawn sampling oracle: {e}"
            );
            eprintln!("skipping sampling oracle: {e}");
            return;
        }
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "sampling oracle failed: {stderr}"
        );
        eprintln!("skipping sampling oracle: {stderr}");
        return;
    }
    let oracle: SamplingOracleOut =
        serde_json::from_slice(&output.stdout).expect("parse sampling oracle JSON");

    let mut ledger = CompareLedger::new(
        "diff_stats_sampling_unuran",
        &[
            "FastGeneratorInversion",
            "NumericalInversePolynomial",
            "SimpleRatioUniforms",
            "UNURANError",
        ],
    );

    let fgi = FastGeneratorInversion::new(Normal::standard(), Some((-3.0, 3.0))).expect("fgi");
    for (u, want) in oracle.us.iter().zip(&oracle.fgi_ppf) {
        let got = fgi.ppf(*u).expect("fgi ppf");
        let pass = (got - want).abs() <= 1e-6;
        ledger.compared("FastGeneratorInversion", &format!("u={u}"), pass);
    }

    let pinv = NumericalInversePolynomial::new(
        |x: f64| (-0.5 * x * x).exp(),
        Some(0.0),
        Some((-5.0, 5.0)),
        5,
        1e-10,
    )
    .expect("pinv");
    for (u, want) in oracle.us.iter().zip(&oracle.pinv_ppf) {
        let got = pinv.ppf(*u).expect("pinv ppf");
        let pass = (got - want).abs() <= 1e-5;
        ledger.compared("NumericalInversePolynomial", &format!("ppf_u={u}"), pass);
    }
    for (x, want) in oracle.xs.iter().zip(&oracle.pinv_cdf) {
        let got = pinv.cdf(*x).expect("pinv cdf");
        let pass = (got - want).abs() <= 1e-5;
        ledger.compared("NumericalInversePolynomial", &format!("cdf_x={x}"), pass);
    }

    let sru = SimpleRatioUniforms::new(
        |x: f64| (-0.5 * x * x).exp(),
        0.0,
        (2.0 * std::f64::consts::PI).sqrt(),
        Some(0.5),
    )
    .expect("sru");
    let mut rng_state = 12345_u64;
    let samples = sru
        .sample(4000, &mut || {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng_state >> 11) as f64) / ((1u64 << 53) as f64)
        })
        .expect("sru sample");
    let mean = samples.iter().sum::<f64>() / (samples.len() as f64);
    let var = samples.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / ((samples.len() - 1) as f64);
    let std = var.sqrt();
    ledger.compared(
        "SimpleRatioUniforms",
        "mean",
        (mean - oracle.sru_mean).abs() <= 0.1,
    );
    ledger.compared(
        "SimpleRatioUniforms",
        "std",
        (std - oracle.sru_std).abs() <= 0.1,
    );

    let zero_res = NumericalInversePolynomial::new(|_: f64| 0.0, Some(0.0), Some((-1.0, 1.0)), 5, 1e-10);
    let err_matches = oracle.unuran_err && matches!(zero_res, Err(StatsError::InvalidArgument { .. }));
    ledger.compared("UNURANError", "zero_pdf", err_matches);

    ledger.finish(1);
}
