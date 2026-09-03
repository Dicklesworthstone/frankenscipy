#![forbid(unsafe_code)]
//! Live SciPy differential coverage for distribution objects —
//! Laplace, Gumbel, GumbelLeft, Cauchy pdf/cdf/sf vs scipy.stats.
//!
//! Resolves [frankenscipy-wlp4b]. Continues the [frankenscipy-6vdrs]
//! pattern (Logistic/Pareto/Lomax) for the 4 distributions whose sf
//! got closed-form overrides in [frankenscipy-6uo0s] and
//! [frankenscipy-ltdct]. Skips cleanly if scipy/python3 is unavailable
//! unless `FSCI_REQUIRE_SCIPY_ORACLE` is set.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{Cauchy, ContinuousDistribution, Gumbel, GumbelLeft, Laplace};
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
    /// scipy parameter order:
    ///   laplace:    [loc, scale]
    ///   gumbel_r:   [loc, scale]
    ///   gumbel_l:   [loc, scale]
    ///   cauchy:     [loc, scale]
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
    fs::create_dir_all(output_dir()).expect("create distributions_more diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize distributions_more diff log");
    fs::write(path, json).expect("write distributions_more diff log");
}

fn generate_cases() -> Vec<DistCase> {
    let mut cases = Vec::new();

    // Laplace × {pdf, cdf, sf} including deep right tail (z=80) where the
    // 6uo0s sf override matters.
    let laplace_params = [(0.0_f64, 1.0), (2.0, 0.5), (-1.0, 3.0)];
    let laplace_xs = [-5.0_f64, -1.0, 0.0, 1.0, 5.0, 30.0, 80.0];
    for (pi, &(loc, scale)) in laplace_params.iter().enumerate() {
        for (xi, &x) in laplace_xs.iter().enumerate() {
            for method in ["pdf", "cdf", "sf"] {
                cases.push(DistCase {
                    case_id: format!("laplace_p{pi}_x{xi}_{method}"),
                    dist: "laplace".into(),
                    method: method.into(),
                    params: vec![loc, scale],
                    x,
                });
            }
        }
    }

    // Gumbel right (extreme value type I) × {pdf, cdf, sf} including z=60.
    let gumbel_params = [(0.0_f64, 1.0), (5.0, 2.0), (-3.0, 0.5)];
    let gumbel_xs = [-3.0_f64, -1.0, 0.0, 1.0, 5.0, 20.0, 60.0];
    for (pi, &(loc, scale)) in gumbel_params.iter().enumerate() {
        for (xi, &x) in gumbel_xs.iter().enumerate() {
            for method in ["pdf", "cdf", "sf"] {
                cases.push(DistCase {
                    case_id: format!("gumbel_r_p{pi}_x{xi}_{method}"),
                    dist: "gumbel_r".into(),
                    method: method.into(),
                    params: vec![loc, scale],
                    x,
                });
            }
        }
    }

    // GumbelLeft × {pdf, cdf, sf}.
    let gumbel_left_params = [(0.0_f64, 1.0), (-2.0, 1.5), (3.0, 0.75)];
    let gumbel_left_xs = [-30.0_f64, -3.0, 0.0, 1.0, 3.0, 10.0];
    for (pi, &(loc, scale)) in gumbel_left_params.iter().enumerate() {
        for (xi, &x) in gumbel_left_xs.iter().enumerate() {
            for method in ["pdf", "cdf", "sf"] {
                cases.push(DistCase {
                    case_id: format!("gumbel_l_p{pi}_x{xi}_{method}"),
                    dist: "gumbel_l".into(),
                    method: method.into(),
                    params: vec![loc, scale],
                    x,
                });
            }
        }
    }

    // Cauchy × {pdf, cdf, sf} including deep right tail x=1e20·scale where
    // the ltdct closed-form override matters.
    let cauchy_params = [(0.0_f64, 1.0), (-1.0, 0.5), (2.0, 3.0)];
    let cauchy_xs = [-1.0e6_f64, -1.0, 0.0, 1.0, 100.0, 1.0e10, 1.0e20];
    for (pi, &(loc, scale)) in cauchy_params.iter().enumerate() {
        for (xi, &x) in cauchy_xs.iter().enumerate() {
            for method in ["pdf", "cdf", "sf"] {
                cases.push(DistCase {
                    case_id: format!("cauchy_p{pi}_x{xi}_{method}"),
                    dist: "cauchy".into(),
                    method: method.into(),
                    params: vec![loc, scale],
                    x,
                });
            }
        }
    }

    cases
}

fn rust_eval(case: &DistCase) -> f64 {
    match case.dist.as_str() {
        "laplace" => {
            let d = Laplace::new(case.params[0], case.params[1]);
            match case.method.as_str() {
                "pdf" => d.pdf(case.x),
                "cdf" => d.cdf(case.x),
                "sf" => d.sf(case.x),
                other => panic!("unknown method {other}"),
            }
        }
        "gumbel_r" => {
            let d = Gumbel::new(case.params[0], case.params[1]);
            match case.method.as_str() {
                "pdf" => d.pdf(case.x),
                "cdf" => d.cdf(case.x),
                "sf" => d.sf(case.x),
                other => panic!("unknown method {other}"),
            }
        }
        "gumbel_l" => {
            let d = GumbelLeft::new(case.params[0], case.params[1]);
            match case.method.as_str() {
                "pdf" => d.pdf(case.x),
                "cdf" => d.cdf(case.x),
                "sf" => d.sf(case.x),
                other => panic!("unknown method {other}"),
            }
        }
        "cauchy" => {
            let d = Cauchy::new(case.params[0], case.params[1]);
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
    loc, scale = c["params"]
    x = c["x"]
    try:
        if dist == "laplace":
            d = stats.laplace(loc=loc, scale=scale)
        elif dist == "gumbel_r":
            d = stats.gumbel_r(loc=loc, scale=scale)
        elif dist == "gumbel_l":
            d = stats.gumbel_l(loc=loc, scale=scale)
        elif dist == "cauchy":
            d = stats.cauchy(loc=loc, scale=scale)
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

    let cases_json = serde_json::to_string(cases).expect("serialize distributions_more cases");

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
                "failed to spawn python3 for distributions_more oracle: {e}"
            );
            eprintln!("skipping distributions_more oracle: python3 not available ({e})");
            return Vec::new();
        }
    };

    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open distributions_more oracle stdin");
        if let Err(err) = stdin.write_all(cases_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "distributions_more oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping distributions_more oracle: stdin write failed ({err})\n{stderr}");
            return Vec::new();
        }
    }

    let output = child
        .wait_with_output()
        .expect("wait for distributions_more oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "distributions_more oracle failed: {stderr}"
        );
        eprintln!("skipping distributions_more oracle: scipy not available\n{stderr}");
        return Vec::new();
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout).expect("parse distributions_more oracle JSON")
}

#[test]
fn diff_stats_distributions_more() {
    let cases = generate_cases();
    let oracle_results = scipy_oracle_or_skip(&cases);

    if oracle_results.is_empty() {
        return;
    }

    assert_eq!(
        oracle_results.len(),
        cases.len(),
        "scipy distributions_more oracle returned partial coverage"
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
            _ => continue,
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
        test_id: "diff_stats_distributions_more".into(),
        category: "scipy.stats.{laplace, gumbel_r, gumbel_l, cauchy}".into(),
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
        "scipy.stats distributions_more conformance failed: {} cases, max_abs={}, max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
}
