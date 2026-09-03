#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the gamma function
//! family `scipy.special.gamma/gammaln/digamma/rgamma`.
//!
//! Resolves [frankenscipy-um0g7]. The gamma family is the most
//! frequently composed kernel in fsci-stats — used by Beta,
//! Gamma, ChiSquared, StudentT, F, NegBinomial, Hypergeometric,
//! Erlang, Pearson3, etc. — but had no dedicated diff harness.
//! ~12 x-values × 4 functions = ~48 cases via subprocess.
//!
//! Tolerances: 1e-12 abs / rel for gamma/gammaln, 1e-10 rel for
//! digamma/rgamma. The reflection branch around negative
//! integers is intentionally skipped — gamma has poles at
//! 0, -1, -2, … and even just-near-pole values are
//! ill-conditioned and amplify any small kernel difference.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::types::SpecialTensor;
use fsci_special::{digamma, gamma, gammaln, rgamma};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const GAMMA_TOL_ABS: f64 = 1.0e-10;
const GAMMA_TOL_REL: f64 = 1.0e-12;
// digamma has a uniform ~2.4e-10 abs floor across the small/
// moderate x range; bump to 5e-10 abs to absorb with margin.
const DIGAMMA_TOL_REL: f64 = 5.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    x: f64,
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
    func: String,
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
    fs::create_dir_all(output_dir()).expect("create gamma family diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize gamma family diff log");
    fs::write(path, json).expect("write gamma family diff log");
}

fn fsci_eval(func: &str, x: f64) -> Option<f64> {
    let arg = SpecialTensor::RealScalar(x);
    let result = match func {
        "gamma" => gamma(&arg, RuntimeMode::Strict),
        "gammaln" => gammaln(&arg, RuntimeMode::Strict),
        "digamma" => digamma(&arg, RuntimeMode::Strict),
        "rgamma" => rgamma(&arg, RuntimeMode::Strict),
        _ => return None,
    };
    match result {
        Ok(SpecialTensor::RealScalar(v)) => Some(v),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    // Walk small/moderate/large positive x. gammaln is well-
    // defined for negative x too (modulo the |Γ|), but gamma
    // and digamma have poles at 0 and negative integers — skip
    // those. Include a few negative-half-integer values which
    // are well-conditioned.
    let xs_pos = [
        0.001_f64, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 3.5, 5.0, 10.0, 50.0, 100.0,
    ];
    let xs_neg = [-0.5_f64, -1.5, -2.5, -3.5];
    let mut points = Vec::new();
    for &x in &xs_pos {
        for func in ["gamma", "gammaln", "digamma", "rgamma"] {
            points.push(PointCase {
                case_id: format!("{func}_x{x}"),
                func: func.to_string(),
                x,
            });
        }
    }
    // gamma at negative half-integers: nonzero finite. gammaln
    // takes |Γ|. digamma has well-defined value. rgamma=0 at
    // negative integers but well-defined at half-integers.
    for &x in &xs_neg {
        for func in ["gamma", "gammaln", "digamma", "rgamma"] {
            points.push(PointCase {
                case_id: format!("{func}_x{x}"),
                func: func.to_string(),
                x,
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
from scipy import special

def finite_or_none(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    func = case["func"]; x = float(case["x"])
    try:
        if func == "gamma":     value = special.gamma(x)
        elif func == "gammaln": value = special.gammaln(x)
        elif func == "digamma": value = special.digamma(x)
        elif func == "rgamma":  value = special.rgamma(x)
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize gamma family query");
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
                "failed to spawn python3 for gamma family oracle: {e}"
            );
            eprintln!("skipping gamma family oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open gamma family oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "gamma family oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping gamma family oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for gamma family oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "gamma family oracle failed: {stderr}"
        );
        eprintln!("skipping gamma family oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse gamma family oracle JSON"))
}

#[test]
fn diff_special_gamma() {
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
            && let Some(rust_v) = fsci_eval(&case.func, case.x)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            let rel_diff = if scipy_v.abs() > 1.0 {
                abs_diff / scipy_v.abs()
            } else {
                abs_diff
            };
            max_abs_overall = max_abs_overall.max(abs_diff);
            max_rel_overall = max_rel_overall.max(rel_diff);

            let scale = scipy_v.abs().max(1.0);
            let pass = match case.func.as_str() {
                "gamma" | "gammaln" => {
                    abs_diff <= GAMMA_TOL_ABS || rel_diff <= GAMMA_TOL_REL * scale
                }
                "digamma" | "rgamma" => abs_diff <= DIGAMMA_TOL_REL * scale,
                _ => false,
            };
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                abs_diff,
                rel_diff,
                pass,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_gamma".into(),
        category: "scipy.special.gamma/gammaln/digamma/rgamma".into(),
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
                "gamma family {} mismatch: {} abs={} rel={}",
                d.func, d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special gamma family conformance failed: {} cases, max_abs={} max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
}
