#![forbid(unsafe_code)]
//! Live SciPy differential coverage for distribution entropies.
//!
//! Resolves [frankenscipy-184sk]. Cross-checks the closed-form
//! entropy paths in fsci-stats against
//! `scipy.stats.<dist>.entropy()`. Coverage is orthogonal to
//! all the cdf/pdf/ppf and moments diff harnesses — entropy
//! often has its own analytic shortcut sharing no code with
//! the distribution kernels.
//!
//! ~12 distributions via subprocess. Tolerances: 1e-10 abs OR
//! 1e-9 rel — entropy formulas compose log + digamma helpers.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{
    BetaDist, ChiSquared, ContinuousDistribution, Exponential, GammaDist, Gumbel, Laplace,
    Logistic, Lognormal, Normal, Rayleigh, Uniform, Weibull,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-10;
const REL_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    dist: String,
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
    fs::create_dir_all(output_dir()).expect("create entropy diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize entropy diff log");
    fs::write(path, json).expect("write entropy diff log");
}

fn fsci_entropy(dist: &str, params: &[f64]) -> Option<f64> {
    let v = match dist {
        "norm" => Normal::new(params[0], params[1]).entropy(),
        "expon" => Exponential::new(1.0 / params[0]).entropy(),
        "gamma" => GammaDist::new(params[0], params[1]).entropy(),
        "beta" => BetaDist::new(params[0], params[1]).entropy(),
        "chi2" => ChiSquared::new(params[0]).entropy(),
        "uniform" => Uniform::new(params[0], params[1]).entropy(),
        "weibull_min" => Weibull::new(params[0], params[1]).entropy(),
        "lognorm" => Lognormal::new(params[0], params[1]).entropy(),
        "logistic" => Logistic::new(params[0], params[1]).entropy(),
        "laplace" => Laplace::new(params[0], params[1]).entropy(),
        "gumbel_r" => Gumbel::new(params[0], params[1]).entropy(),
        "rayleigh" => Rayleigh::new(params[0]).entropy(),
        _ => return None,
    };
    if v.is_finite() { Some(v) } else { None }
}

fn generate_query() -> OracleQuery {
    let pinned: Vec<(&str, Vec<f64>)> = vec![
        ("norm", vec![0.0, 1.0]),
        ("expon", vec![1.0]),
        ("gamma", vec![3.0, 2.0]),
        ("beta", vec![2.0, 5.0]),
        ("chi2", vec![5.0]),
        ("uniform", vec![0.0, 5.0]),
        ("weibull_min", vec![2.5, 1.5]),
        ("lognorm", vec![0.5, 1.0]),
        ("logistic", vec![1.0, 2.0]),
        ("laplace", vec![1.0, 2.0]),
        ("gumbel_r", vec![1.0, 2.0]),
        ("rayleigh", vec![2.0]),
    ];
    let mut points = Vec::new();
    for (dist, params) in pinned {
        points.push(PointCase {
            case_id: format!("{dist}_entropy"),
            dist: dist.into(),
            params,
        });
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
    if dist_name == "norm":   return stats.norm(loc=params[0], scale=params[1])
    if dist_name == "expon":  return stats.expon(scale=1.0/params[0])
    if dist_name == "gamma":  return stats.gamma(a=params[0], scale=params[1])
    if dist_name == "beta":   return stats.beta(a=params[0], b=params[1])
    if dist_name == "chi2":   return stats.chi2(df=params[0])
    if dist_name == "uniform":return stats.uniform(loc=params[0], scale=params[1])
    if dist_name == "weibull_min": return stats.weibull_min(c=params[0], scale=params[1])
    if dist_name == "lognorm":return stats.lognorm(s=params[0], scale=params[1])
    if dist_name == "logistic":return stats.logistic(loc=params[0], scale=params[1])
    if dist_name == "laplace":return stats.laplace(loc=params[0], scale=params[1])
    if dist_name == "gumbel_r":return stats.gumbel_r(loc=params[0], scale=params[1])
    if dist_name == "rayleigh":return stats.rayleigh(scale=params[0])
    return None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; dist = case["dist"]
    params = [float(x) for x in case["params"]]
    try:
        rv = call_dist(dist, params)
        if rv is None:
            points.append({"case_id": cid, "value": None}); continue
        value = float(rv.entropy())
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize entropy query");
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
                "failed to spawn python3 for entropy oracle: {e}"
            );
            eprintln!("skipping entropy oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open entropy oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "entropy oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping entropy oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for entropy oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "entropy oracle failed: {stderr}"
        );
        eprintln!("skipping entropy oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse entropy oracle JSON"))
}

#[test]
fn diff_stats_entropy() {
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
            && let Some(rust_v) = fsci_entropy(&case.dist, &case.params)
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
                abs_diff,
                rel_diff,
                pass,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_entropy".into(),
        category: "scipy.stats.<dist>.entropy()".into(),
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
                "entropy {} mismatch: {} abs={} rel={}",
                d.dist, d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "entropy conformance failed: {} cases, max_abs={} max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
}
