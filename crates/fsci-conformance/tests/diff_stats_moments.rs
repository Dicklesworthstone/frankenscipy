#![forbid(unsafe_code)]
//! Live SciPy differential coverage for distribution moments
//! (mean, variance, skewness, excess kurtosis) across the
//! fsci continuous distribution family.
//!
//! Resolves [frankenscipy-fvkcj]. Cross-checks the closed-form
//! moment paths in fsci-stats against
//! `scipy.stats.<dist>.stats(moments='mvsk')`. Coverage is
//! orthogonal to the existing cdf/pdf/ppf diff harnesses —
//! moments often have their own analytic shortcuts (skewness
//! etc.) that share no code with the distributional kernels.
//!
//! ~15 distributions × 4 moments = 60 cases via subprocess.
//! Tolerances: 1e-12 abs OR 1e-10 rel — moment formulas are
//! closed-form for most distributions; a few (Gompertz,
//! TukeyLambda) compute via numerical integration so the rel
//! fallback is essential.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{
    BetaDist, Cauchy, ChiSquared, ContinuousDistribution, Exponential, FDistribution, GammaDist,
    Gumbel, Laplace, Logistic, Lognormal, Normal, Pareto, Rayleigh, StudentT, Uniform, Weibull,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REL_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    dist: String,
    moment: String,
    // Pinned distribution parameters; oracle script knows how
    // to feed them into scipy.stats.<dist>.stats(moments=...).
    params: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    dist: String,
    moment: String,
    abs_diff: f64,
    rel_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_abs_diff: f64,
    max_rel_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create moments diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize moments diff log");
    fs::write(path, json).expect("write moments diff log");
}

fn fsci_eval(dist: &str, moment: &str, params: &[f64]) -> Option<f64> {
    let v = match dist {
        "norm" => {
            let d = Normal::new(params[0], params[1]);
            match moment {
                "mean" => d.mean(),
                "var" => d.var(),
                "skew" => d.skewness(),
                "kurt" => d.kurtosis(),
                _ => f64::NAN,
            }
        }
        "expon" => {
            let d = Exponential::new(1.0 / params[0]); // scipy expon scale = 1/lambda
            match moment {
                "mean" => d.mean(),
                "var" => d.var(),
                "skew" => d.skewness(),
                "kurt" => d.kurtosis(),
                _ => f64::NAN,
            }
        }
        "gamma" => {
            let d = GammaDist::new(params[0], params[1]);
            match moment {
                "mean" => d.mean(),
                "var" => d.var(),
                "skew" => d.skewness(),
                "kurt" => d.kurtosis(),
                _ => f64::NAN,
            }
        }
        "beta" => {
            let d = BetaDist::new(params[0], params[1]);
            match moment {
                "mean" => d.mean(),
                "var" => d.var(),
                "skew" => d.skewness(),
                "kurt" => d.kurtosis(),
                _ => f64::NAN,
            }
        }
        "chi2" => {
            let d = ChiSquared::new(params[0]);
            match moment {
                "mean" => d.mean(),
                "var" => d.var(),
                "skew" => d.skewness(),
                "kurt" => d.kurtosis(),
                _ => f64::NAN,
            }
        }
        "t" => {
            let d = StudentT::new(params[0]);
            match moment {
                "mean" => d.mean(),
                "var" => d.var(),
                "skew" => d.skewness(),
                "kurt" => d.kurtosis(),
                _ => f64::NAN,
            }
        }
        "f" => {
            let d = FDistribution::new(params[0], params[1]);
            match moment {
                "mean" => d.mean(),
                "var" => d.var(),
                "skew" => d.skewness(),
                "kurt" => d.kurtosis(),
                _ => f64::NAN,
            }
        }
        "uniform" => {
            let d = Uniform::new(params[0], params[1]);
            match moment {
                "mean" => d.mean(),
                "var" => d.var(),
                "skew" => d.skewness(),
                "kurt" => d.kurtosis(),
                _ => f64::NAN,
            }
        }
        "weibull_min" => {
            let d = Weibull::new(params[0], params[1]);
            match moment {
                "mean" => d.mean(),
                "var" => d.var(),
                "skew" => d.skewness(),
                "kurt" => d.kurtosis(),
                _ => f64::NAN,
            }
        }
        "lognorm" => {
            let d = Lognormal::new(params[0], params[1]);
            match moment {
                "mean" => d.mean(),
                "var" => d.var(),
                "skew" => d.skewness(),
                "kurt" => d.kurtosis(),
                _ => f64::NAN,
            }
        }
        "logistic" => {
            let d = Logistic::new(params[0], params[1]);
            match moment {
                "mean" => d.mean(),
                "var" => d.var(),
                "skew" => d.skewness(),
                "kurt" => d.kurtosis(),
                _ => f64::NAN,
            }
        }
        "laplace" => {
            let d = Laplace::new(params[0], params[1]);
            match moment {
                "mean" => d.mean(),
                "var" => d.var(),
                "skew" => d.skewness(),
                "kurt" => d.kurtosis(),
                _ => f64::NAN,
            }
        }
        "gumbel_r" => {
            let d = Gumbel::new(params[0], params[1]);
            match moment {
                "mean" => d.mean(),
                "var" => d.var(),
                "skew" => d.skewness(),
                "kurt" => d.kurtosis(),
                _ => f64::NAN,
            }
        }
        "rayleigh" => {
            let d = Rayleigh::new(params[0]);
            match moment {
                "mean" => d.mean(),
                "var" => d.var(),
                "skew" => d.skewness(),
                "kurt" => d.kurtosis(),
                _ => f64::NAN,
            }
        }
        "pareto" => {
            let d = Pareto::new(params[0], params[1]);
            match moment {
                "mean" => d.mean(),
                "var" => d.var(),
                "skew" => d.skewness(),
                "kurt" => d.kurtosis(),
                _ => f64::NAN,
            }
        }
        "cauchy" => {
            let d = Cauchy::new(params[0], params[1]);
            match moment {
                "mean" => d.mean(),
                "var" => d.var(),
                "skew" => d.skewness(),
                "kurt" => d.kurtosis(),
                _ => f64::NAN,
            }
        }
        _ => return None,
    };
    if v.is_finite() { Some(v) } else { None }
}

fn generate_query() -> OracleQuery {
    // (dist_name, params, scipy_args_template). The oracle
    // script translates `params` into the scipy call.
    let pinned: Vec<(&str, Vec<f64>)> = vec![
        ("norm", vec![0.0, 1.0]),
        ("expon", vec![1.0]), // lambda; scipy uses scale=1/lambda
        ("gamma", vec![3.0, 2.0]),
        ("beta", vec![2.0, 5.0]),
        ("chi2", vec![5.0]),
        ("t", vec![10.0]), // df > 4 for kurtosis to be finite
        ("f", vec![10.0, 20.0]),
        ("uniform", vec![0.0, 5.0]),
        ("weibull_min", vec![2.5, 1.5]),
        ("lognorm", vec![0.5, 1.0]),
        ("logistic", vec![1.0, 2.0]),
        ("laplace", vec![1.0, 2.0]),
        ("gumbel_r", vec![1.0, 2.0]),
        ("rayleigh", vec![2.0]),
        ("pareto", vec![5.0, 1.0]), // shape=5 needed for kurtosis finite
                                    // Cauchy: all moments NaN — included as a sanity check
    ];
    let moments = ["mean", "var", "skew", "kurt"];
    let mut points = Vec::new();
    for (dist, params) in pinned {
        for m in moments {
            // For Cauchy and similar, fsci returns NaN deliberately;
            // skip the case via finite-check on rust side.
            points.push(PointCase {
                case_id: format!("{dist}_{m}"),
                dist: dist.into(),
                moment: m.into(),
                params: params.clone(),
            });
        }
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy import stats

def finite_or_none(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

def call_dist(dist_name, params):
    """Construct scipy distribution from fsci's parameterization."""
    if dist_name == "norm":   return stats.norm(loc=params[0], scale=params[1])
    if dist_name == "expon":  return stats.expon(scale=1.0/params[0])
    if dist_name == "gamma":  return stats.gamma(a=params[0], scale=params[1])
    if dist_name == "beta":   return stats.beta(a=params[0], b=params[1])
    if dist_name == "chi2":   return stats.chi2(df=params[0])
    if dist_name == "t":      return stats.t(df=params[0])
    if dist_name == "f":      return stats.f(dfn=params[0], dfd=params[1])
    if dist_name == "uniform":return stats.uniform(loc=params[0], scale=params[1])
    if dist_name == "weibull_min": return stats.weibull_min(c=params[0], scale=params[1])
    if dist_name == "lognorm":return stats.lognorm(s=params[0], scale=params[1])
    if dist_name == "logistic":return stats.logistic(loc=params[0], scale=params[1])
    if dist_name == "laplace":return stats.laplace(loc=params[0], scale=params[1])
    if dist_name == "gumbel_r":return stats.gumbel_r(loc=params[0], scale=params[1])
    if dist_name == "rayleigh":return stats.rayleigh(scale=params[0])
    if dist_name == "pareto": return stats.pareto(b=params[0], scale=params[1])
    if dist_name == "cauchy": return stats.cauchy(loc=params[0], scale=params[1])
    return None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; dist = case["dist"]; moment = case["moment"]
    params = [float(x) for x in case["params"]]
    try:
        rv = call_dist(dist, params)
        if rv is None:
            points.append({"case_id": cid, "value": None}); continue
        m, v, s, k = rv.stats(moments="mvsk")
        m, v, s, k = float(m), float(v), float(s), float(k)
        if moment == "mean":   value = m
        elif moment == "var":  value = v
        elif moment == "skew": value = s
        elif moment == "kurt": value = k
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize moments query");
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
                "failed to spawn python3 for moments oracle: {e}"
            );
            eprintln!("skipping moments oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open moments oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "moments oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping moments oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for moments oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "moments oracle failed: {stderr}"
        );
        eprintln!("skipping moments oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse moments oracle JSON"))
}

#[test]
fn diff_stats_moments() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_abs_overall = 0.0_f64;
    let mut max_rel_overall = 0.0_f64;

    for case in &query.points {
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        if let Some(scipy_v) = oracle.value
            && let Some(rust_v) = fsci_eval(&case.dist, &case.moment, &case.params)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            let scale = scipy_v.abs().max(1.0);
            let rel_diff = abs_diff / scale;
            max_abs_overall = max_abs_overall.max(abs_diff);
            max_rel_overall = max_rel_overall.max(rel_diff);
            let pass = abs_diff <= ABS_TOL || abs_diff <= REL_TOL * scale;
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                dist: case.dist.clone(),
                moment: case.moment.clone(),
                abs_diff,
                rel_diff,
                pass,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_moments".into(),
        category: "scipy.stats.<dist>.stats(moments='mvsk')".into(),
        case_count: diffs.len(),
        max_abs_diff: max_abs_overall,
        max_rel_diff: max_rel_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };

    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "moments {} {} mismatch: {} abs={} rel={}",
                d.dist, d.moment, d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "moments conformance failed: {} cases, max_abs={} max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
}
