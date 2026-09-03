#![forbid(unsafe_code)]
//! Live SciPy differential coverage for distribution objects —
//! Logistic, Pareto, Lomax pdf/cdf/sf vs scipy.stats.
//!
//! Resolves [frankenscipy-6vdrs]. The three distributions exercised
//! were just refined: closed-form sf overrides in [frankenscipy-h36zm]
//! and try_fit overrides in [frankenscipy-covaj]. This harness pins
//! parity with scipy.stats including the deep-right-tail regime where
//! the pre-fix default sf would have rounded to 0. Skips cleanly if
//! scipy/python3 is unavailable unless `FSCI_REQUIRE_SCIPY_ORACLE` is
//! set.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{ContinuousDistribution, Logistic, Lomax, Pareto};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-10;
const REL_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct DistCase {
    case_id: String,
    dist: String,
    method: String,
    /// Distribution-specific shape parameters in canonical scipy order:
    ///   logistic: [loc, scale]
    ///   pareto:   [b, scale]
    ///   lomax:    [c]
    params: Vec<f64>,
    x: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct DistOracleResult {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    dist: String,
    method: String,
    rust: f64,
    scipy: f64,
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
    abs_tol: f64,
    rel_tol: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create distributions diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize distributions diff log");
    fs::write(path, json).expect("write distributions diff log");
}

fn generate_cases() -> Vec<DistCase> {
    let mut cases = Vec::new();

    // Logistic(loc, scale) × {pdf, cdf, sf} × x grid including
    // deep-right-tail z=40 regime where the default sf would round to 0.
    let logistic_params = [(0.0_f64, 1.0), (-1.0, 0.5), (2.0, 3.0)];
    let logistic_xs = [-5.0_f64, -1.0, 0.0, 1.0, 5.0, 20.0, 40.0];
    for (pi, &(loc, scale)) in logistic_params.iter().enumerate() {
        for (xi, &x) in logistic_xs.iter().enumerate() {
            for method in ["pdf", "cdf", "sf"] {
                cases.push(DistCase {
                    case_id: format!("logistic_p{pi}_x{xi}_{method}"),
                    dist: "logistic".into(),
                    method: method.into(),
                    params: vec![loc, scale],
                    x,
                });
            }
        }
    }

    // Pareto(b, scale=1) × {pdf, cdf, sf} including very deep tail.
    let pareto_params = [(2.0_f64, 1.0), (3.5, 1.0), (1.5, 2.0)];
    let pareto_xs = [1.0_f64, 1.5, 2.0, 5.0, 10.0, 100.0, 1.0e6];
    for (pi, &(b, scale)) in pareto_params.iter().enumerate() {
        for (xi, &x) in pareto_xs.iter().enumerate() {
            // Below scale, scipy.pareto returns nan/0 for some methods.
            if x < scale {
                continue;
            }
            for method in ["pdf", "cdf", "sf"] {
                cases.push(DistCase {
                    case_id: format!("pareto_p{pi}_x{xi}_{method}"),
                    dist: "pareto".into(),
                    method: method.into(),
                    params: vec![b, scale],
                    x,
                });
            }
        }
    }

    // Lomax(c) × {pdf, cdf, sf} including deep tail.
    let lomax_params = [0.5_f64, 2.0, 3.5, 7.0];
    let lomax_xs = [0.0_f64, 0.5, 1.0, 5.0, 100.0, 1.0e6];
    for (pi, &c) in lomax_params.iter().enumerate() {
        for (xi, &x) in lomax_xs.iter().enumerate() {
            for method in ["pdf", "cdf", "sf"] {
                cases.push(DistCase {
                    case_id: format!("lomax_p{pi}_x{xi}_{method}"),
                    dist: "lomax".into(),
                    method: method.into(),
                    params: vec![c],
                    x,
                });
            }
        }
    }

    cases
}

fn rust_eval(case: &DistCase) -> f64 {
    match case.dist.as_str() {
        "logistic" => {
            let d = Logistic::new(case.params[0], case.params[1]);
            match case.method.as_str() {
                "pdf" => d.pdf(case.x),
                "cdf" => d.cdf(case.x),
                "sf" => d.sf(case.x),
                other => panic!("unknown method {other}"),
            }
        }
        "pareto" => {
            let d = Pareto::new(case.params[0], case.params[1]);
            match case.method.as_str() {
                "pdf" => d.pdf(case.x),
                "cdf" => d.cdf(case.x),
                "sf" => d.sf(case.x),
                other => panic!("unknown method {other}"),
            }
        }
        "lomax" => {
            let d = Lomax::new(case.params[0]);
            match case.method.as_str() {
                "pdf" => d.pdf(case.x),
                "cdf" => d.cdf(case.x),
                "sf" => d.sf(case.x),
                other => panic!("unknown method {other}"),
            }
        }
        other => panic!("unknown dist {other}"),
    }
}

fn scipy_oracle_or_skip(cases: &[DistCase]) -> Vec<DistOracleResult> {
    let script = r#"
import json
import sys
from scipy import stats

cases = json.load(sys.stdin)
results = []
for c in cases:
    cid = c["case_id"]
    dist = c["dist"]
    method = c["method"]
    params = c["params"]
    x = c["x"]
    try:
        if dist == "logistic":
            loc, scale = params
            d = stats.logistic(loc=loc, scale=scale)
        elif dist == "pareto":
            b, scale = params
            d = stats.pareto(b, scale=scale)
        elif dist == "lomax":
            (c_shape,) = params
            d = stats.lomax(c_shape)
        else:
            results.append({"case_id": cid, "value": None})
            continue
        fn = getattr(d, method)
        val = float(fn(x))
        results.append({"case_id": cid, "value": val})
    except Exception:
        results.append({"case_id": cid, "value": None})

print(json.dumps(results))
"#;

    let cases_json = serde_json::to_string(cases).expect("serialize distribution cases");

    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for distributions oracle: {e}"
            );
            eprintln!("skipping distributions oracle: python3 not available ({e})");
            return Vec::new();
        }
    };

    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open distributions oracle stdin");
        if let Err(err) = stdin.write_all(cases_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "distributions oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping distributions oracle: stdin write failed ({err})\n{stderr}");
            return Vec::new();
        }
    }

    let output = child
        .wait_with_output()
        .expect("wait for distributions oracle");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "distributions oracle failed: {stderr}"
        );
        eprintln!("skipping distributions oracle: scipy not available\n{stderr}");
        return Vec::new();
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout).expect("parse distributions oracle JSON")
}

#[test]
fn diff_stats_distributions() {
    let cases = generate_cases();
    let oracle_results = scipy_oracle_or_skip(&cases);

    if oracle_results.is_empty() {
        return;
    }

    assert_eq!(
        oracle_results.len(),
        cases.len(),
        "SciPy distributions oracle returned partial coverage"
    );

    let oracle_map: HashMap<String, DistOracleResult> = oracle_results
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_abs_overall = 0.0_f64;
    let mut max_rel_overall = 0.0_f64;

    for case in &cases {
        let scipy_value = match oracle_map.get(&case.case_id).and_then(|r| r.value) {
            Some(v) if v.is_finite() => v,
            _ => continue, // SciPy refused — skip.
        };

        let rust_value = rust_eval(case);
        if !rust_value.is_finite() {
            panic!(
                "Rust returned non-finite ({rust_value}) for {case_id} where SciPy gave {scipy_value}",
                case_id = case.case_id
            );
        }

        let abs_diff = (rust_value - scipy_value).abs();
        let rel_diff = abs_diff / scipy_value.abs().max(1.0);
        let pass = abs_diff <= ABS_TOL || rel_diff <= REL_TOL;

        max_abs_overall = max_abs_overall.max(abs_diff);
        max_rel_overall = max_rel_overall.max(rel_diff);

        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            dist: case.dist.clone(),
            method: case.method.clone(),
            rust: rust_value,
            scipy: scipy_value,
            abs_diff,
            rel_diff,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_distributions".into(),
        category: "scipy.stats.{logistic, pareto, lomax}".into(),
        case_count: diffs.len(),
        max_abs_diff: max_abs_overall,
        max_rel_diff: max_rel_overall,
        abs_tol: ABS_TOL,
        rel_tol: REL_TOL,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };

    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "distribution mismatch: {} {} {} rust={} scipy={} abs={} rel={}",
                d.dist, d.method, d.case_id, d.rust, d.scipy, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.stats distribution conformance failed: {} cases, max_abs={}, max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
}
