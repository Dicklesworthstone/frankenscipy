#![forbid(unsafe_code)]
//! Live SciPy differential coverage for distribution modes.
//!
//! Resolves [frankenscipy-z1dbm]. scipy doesn't expose a
//! generic `.mode` on rv_continuous; the oracle computes the
//! analytic mode per-distribution from the well-known formulas
//! (subprocess Python). Cross-checks fsci's `mode()` closed
//! forms which are otherwise unexercised — they share no code
//! with cdf/pdf/ppf/entropy/moments paths.
//!
//! ~10 distributions via subprocess. Tolerances: 1e-13 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{
    BetaDist, Cauchy, ContinuousDistribution, Exponential, GammaDist, Gumbel, Laplace, Logistic,
    Lognormal, Normal, Pareto, Rayleigh, Uniform, Weibull,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-13;
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
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_abs_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create mode diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize mode diff log");
    fs::write(path, json).expect("write mode diff log");
}

fn fsci_mode(dist: &str, params: &[f64]) -> Option<f64> {
    let v = match dist {
        "norm" => Normal::new(params[0], params[1]).mode(),
        "cauchy" => Cauchy::new(params[0], params[1]).mode(),
        "expon" => Exponential::new(1.0 / params[0]).mode(),
        "gamma_a_gt_1" => GammaDist::new(params[0], params[1]).mode(),
        "beta_a_gt_1_b_gt_1" => BetaDist::new(params[0], params[1]).mode(),
        "uniform" => Uniform::new(params[0], params[1]).mode(),
        "weibull_min_c_gt_1" => Weibull::new(params[0], params[1]).mode(),
        "lognorm" => Lognormal::new(params[0], params[1]).mode(),
        "logistic" => Logistic::new(params[0], params[1]).mode(),
        "laplace" => Laplace::new(params[0], params[1]).mode(),
        "gumbel_r" => Gumbel::new(params[0], params[1]).mode(),
        "rayleigh" => Rayleigh::new(params[0]).mode(),
        "pareto" => Pareto::new(params[0], params[1]).mode(),
        _ => return None,
    };
    if v.is_finite() { Some(v) } else { None }
}

fn generate_query() -> OracleQuery {
    // Pinned (dist, params) covering the canonical mode forms.
    let pinned: Vec<(&str, Vec<f64>)> = vec![
        ("norm", vec![0.0, 1.0]),
        ("cauchy", vec![0.5, 1.0]),
        ("expon", vec![1.0]),
        ("gamma_a_gt_1", vec![3.0, 2.0]),
        ("beta_a_gt_1_b_gt_1", vec![5.0, 3.0]),
        ("uniform", vec![1.0, 4.0]),
        ("weibull_min_c_gt_1", vec![2.5, 1.5]),
        ("lognorm", vec![0.5, 1.0]),
        ("logistic", vec![1.0, 2.0]),
        ("laplace", vec![1.0, 2.0]),
        ("gumbel_r", vec![1.0, 2.0]),
        ("rayleigh", vec![2.0]),
        ("pareto", vec![5.0, 1.0]),
    ];
    let mut points = Vec::new();
    for (dist, params) in pinned {
        points.push(PointCase {
            case_id: format!("{dist}_mode"),
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

def finite_or_none(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

def analytic_mode(dist, params):
    if dist == "norm":  return params[0]
    if dist == "cauchy":return params[0]
    if dist == "expon": return 0.0
    if dist == "gamma_a_gt_1":  # mode = (a-1) * scale for a > 1
        return (params[0] - 1.0) * params[1]
    if dist == "beta_a_gt_1_b_gt_1":  # mode = (a-1)/(a+b-2)
        return (params[0] - 1.0) / (params[0] + params[1] - 2.0)
    if dist == "uniform":  # mode arbitrary in [loc, loc+scale]; fsci returns loc
        return params[0]
    if dist == "weibull_min_c_gt_1":  # mode = scale * ((c-1)/c)^(1/c)
        return params[1] * ((params[0] - 1.0) / params[0]) ** (1.0 / params[0])
    if dist == "lognorm":  # mode = scale * exp(-s^2)
        return params[1] * math.exp(-params[0] * params[0])
    if dist == "logistic": return params[0]
    if dist == "laplace":  return params[0]
    if dist == "gumbel_r": return params[0]
    if dist == "rayleigh": return params[0]
    if dist == "pareto":   return params[1]
    return None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; dist = case["dist"]
    params = [float(x) for x in case["params"]]
    try:
        value = analytic_mode(dist, params)
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize mode query");
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
                "failed to spawn python3 for mode oracle: {e}"
            );
            eprintln!("skipping mode oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open mode oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "mode oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping mode oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for mode oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "mode oracle failed: {stderr}"
        );
        eprintln!("skipping mode oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse mode oracle JSON"))
}

#[test]
fn diff_stats_mode() {
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
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        if let Some(scipy_v) = oracle.value
            && let Some(rust_v) = fsci_mode(&case.dist, &case.params)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                dist: case.dist.clone(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_mode".into(),
        category: "fsci_stats::Distribution::mode (analytic oracle)".into(),
        case_count: diffs.len(),
        max_abs_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };

    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("mode {} mismatch: {} abs={}", d.dist, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "mode conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
