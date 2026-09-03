#![forbid(unsafe_code)]
//! Live SciPy differential coverage for binomial inverse special functions.
//!
//! Resolves [frankenscipy-0h0tz]. Covers `bdtrik`, `bdtrin`, `nbdtrik`,
//! and `nbdtrin`, the inverse-with-respect-to-shape variants adjacent to the
//! existing binomial and negative-binomial CDF helpers.

use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::io::{Error as IoError, Write};
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{bdtrik, bdtrin, nbdtrik, nbdtrin};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const BINOMIAL_INVERSE_TOL: f64 = 2.0e-6;

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    a: f64,
    b: f64,
    c: f64,
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

fn test_error(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(IoError::other(message.into()))
}

fn emit_log(log: &DiffLog) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(output_dir())?;
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log)?;
    fs::write(path, json)?;
    Ok(())
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn fsci_eval(func: &str, a: f64, b: f64, c: f64) -> Option<f64> {
    let value = match func {
        "bdtrik" => bdtrik(a, b, c),
        "bdtrin" => bdtrin(a, b, c),
        "nbdtrik" => nbdtrik(a, b, c),
        "nbdtrin" => nbdtrin(a, b, c),
        _ => return None,
    };
    value.is_finite().then_some(value)
}

fn generate_query() -> OracleQuery {
    let cases: &[(&str, f64, f64, f64)] = &[
        ("bdtrik", 0.5, 10.0, 0.5),
        ("bdtrik", 0.9, 10.0, 0.5),
        ("bdtrik", 0.5, 5.0, 0.2),
        ("bdtrin", 5.0, 0.5, 0.5),
        ("bdtrin", 5.0, 0.9, 0.5),
        ("bdtrin", 0.0, 0.000_976_562_5, 0.5),
        ("nbdtrik", 0.5, 10.0, 0.5),
        ("nbdtrik", 0.9, 10.0, 0.5),
        ("nbdtrik", 0.5, 5.0, 0.2),
        ("nbdtrin", 5.0, 0.5, 0.5),
        ("nbdtrin", 5.0, 0.9, 0.5),
        ("nbdtrin", 0.0, 0.5, 0.5),
    ];
    OracleQuery {
        points: cases
            .iter()
            .enumerate()
            .map(|(idx, (func, a, b, c))| PointCase {
                case_id: format!("{func}_{idx}"),
                func: (*func).into(),
                a: *a,
                b: *b,
                c: *c,
            })
            .collect(),
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Result<Option<OracleResult>, Box<dyn Error>> {
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

q = json.loads(sys.argv[1])
points = []
for case in q["points"]:
    cid = case["case_id"]
    func = case["func"]
    a = float(case["a"]); b = float(case["b"]); c = float(case["c"])
    try:
        if func == "bdtrik": value = special.bdtrik(a, b, c)
        elif func == "bdtrin": value = special.bdtrin(a, b, c)
        elif func == "nbdtrik": value = special.nbdtrik(a, b, c)
        elif func == "nbdtrin": value = special.nbdtrin(a, b, c)
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query)?;
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-")
        .arg(query_json)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            if std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
                return Err(test_error(format!(
                    "failed to spawn python3 for binomial inverse oracle: {e}"
                )));
            }
            eprintln!("skipping binomial inverse oracle: python3 not available ({e})");
            return Ok(None);
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| test_error("open binomial inverse oracle stdin"))?;
        if let Err(err) = stdin.write_all(script.as_bytes()) {
            let output = child.wait_with_output()?;
            let stderr = String::from_utf8_lossy(&output.stderr);
            if std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
                return Err(test_error(format!(
                    "binomial inverse oracle stdin write failed: {err}; stderr: {stderr}"
                )));
            }
            eprintln!("skipping binomial inverse oracle: stdin write failed ({err})\n{stderr}");
            return Ok(None);
        }
    }
    let output = child.wait_with_output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if std::env::var(REQUIRE_SCIPY_ENV).is_ok() {
            return Err(test_error(format!(
                "binomial inverse oracle failed: {stderr}"
            )));
        }
        eprintln!("skipping binomial inverse oracle: scipy not available\n{stderr}");
        return Ok(None);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(Some(serde_json::from_str(&stdout)?))
}

#[test]
fn diff_special_binomial_inverses() -> Result<(), Box<dyn Error>> {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query)? else {
        return Ok(());
    };
    if oracle.points.len() != query.points.len() {
        return Err(test_error(format!(
            "binomial inverse oracle returned {} points for {} queries",
            oracle.points.len(),
            query.points.len()
        )));
    }

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
        let oracle = pmap
            .get(&case.case_id)
            .ok_or_else(|| test_error(format!("missing oracle point {}", case.case_id)))?;
        if let Some(scipy_v) = oracle.value
            && let Some(rust_v) = fsci_eval(&case.func, case.a, case.b, case.c)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            let scale = scipy_v.abs().max(1.0);
            let rel_diff = abs_diff / scale;
            max_abs_overall = max_abs_overall.max(abs_diff);
            max_rel_overall = max_rel_overall.max(rel_diff);
            let pass = abs_diff <= BINOMIAL_INVERSE_TOL * scale;
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
        test_id: "diff_special_binomial_inverses".into(),
        category: "scipy.special bdtrik/bdtrin/nbdtrik/nbdtrin".into(),
        case_count: diffs.len(),
        max_abs_diff: max_abs_overall,
        max_rel_diff: max_rel_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log)?;

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "{} mismatch: {} abs={} rel={}",
                d.func, d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    if !all_pass {
        return Err(test_error(format!(
            "binomial inverse conformance failed: {} cases, max_abs={} max_rel={}",
            diffs.len(),
            max_abs_overall,
            max_rel_overall
        )));
    }

    Ok(())
}
