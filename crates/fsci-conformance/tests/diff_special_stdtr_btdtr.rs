#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the Student-t and Beta
//! scipy-compat wrappers
//! `scipy.special.stdtr/stdtrc/stdtrit` and
//! `scipy.special.btdtr/btdtrc/btdtri`.
//!
//! Resolves [frankenscipy-8yss4]. Verifies the cdf/sf/ppf
//! wrapper layer directly; complements diff_stats_t and
//! diff_stats_beta which exercise the same kernel indirectly.
//!
//! Tolerances: 1e-12 abs cdf/sf (regularized incomplete beta);
//! 1e-9 rel ppf (betaincinv composition).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{btdtr, btdtrc, btdtri, stdtr, stdtrc, stdtrit};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const CDF_TOL: f64 = 1.0e-12;
const PPF_TOL_REL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    a: f64,
    b: f64,
    arg: f64,
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
    fs::create_dir_all(output_dir()).expect("create stdtr/btdtr diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize stdtr/btdtr diff log");
    fs::write(path, json).expect("write stdtr/btdtr diff log");
}

fn fsci_eval(func: &str, a: f64, b: f64, arg: f64) -> Option<f64> {
    // For stdtr/stdtrc: b=0 unused, a=df, arg=t
    // For stdtrit: b=0 unused, a=df, arg=p
    // For btdtr/btdtrc/btdtri: a, b are shape, arg=x or y
    let v = match func {
        "stdtr" => stdtr(a, arg),
        "stdtrc" => stdtrc(a, arg),
        "stdtrit" => stdtrit(a, arg),
        "btdtr" => btdtr(a, b, arg),
        "btdtrc" => btdtrc(a, b, arg),
        "btdtri" => btdtri(a, b, arg),
        _ => return None,
    };
    if v.is_finite() { Some(v) } else { None }
}

fn generate_query() -> OracleQuery {
    let dfs = [1.0_f64, 3.0, 10.0, 50.0];
    let ts = [-3.0_f64, -1.0, -0.3, 0.0, 0.3, 1.0, 3.0];
    let qs = [0.001_f64, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999];

    let beta_pairs = [
        (0.5_f64, 0.5),
        (1.0, 1.0),
        (2.0, 5.0),
        (5.0, 2.0),
        (3.0, 3.0),
    ];
    let xs = [0.001_f64, 0.05, 0.25, 0.5, 0.75, 0.95, 0.999];
    let ys_btdtri = [0.05_f64, 0.25, 0.5, 0.75, 0.95];

    let mut points = Vec::new();
    for &df in &dfs {
        for &t in &ts {
            for func in ["stdtr", "stdtrc"] {
                points.push(PointCase {
                    case_id: format!("{func}_df{df}_t{t}"),
                    func: func.to_string(),
                    a: df,
                    b: 0.0,
                    arg: t,
                });
            }
        }
        for &q in &qs {
            points.push(PointCase {
                case_id: format!("stdtrit_df{df}_q{q}"),
                func: "stdtrit".into(),
                a: df,
                b: 0.0,
                arg: q,
            });
        }
    }
    for &(a, b) in &beta_pairs {
        for &x in &xs {
            for func in ["btdtr", "btdtrc"] {
                points.push(PointCase {
                    case_id: format!("{func}_a{a}_b{b}_x{x}"),
                    func: func.to_string(),
                    a,
                    b,
                    arg: x,
                });
            }
        }
        for &y in &ys_btdtri {
            points.push(PointCase {
                case_id: format!("btdtri_a{a}_b{b}_y{y}"),
                func: "btdtri".into(),
                a,
                b,
                arg: y,
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
    cid = case["case_id"]; func = case["func"]
    a = float(case["a"]); b = float(case["b"]); arg = float(case["arg"])
    try:
        if func == "stdtr":     value = special.stdtr(a, arg)
        elif func == "stdtrc":  value = special.stdtrc(a, arg)
        elif func == "stdtrit": value = special.stdtrit(a, arg)
        elif func == "btdtr":   value = special.btdtr(a, b, arg)
        elif func == "btdtrc":  value = special.btdtrc(a, b, arg)
        elif func == "btdtri":  value = special.btdtri(a, b, arg)
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize stdtr/btdtr query");
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
                "failed to spawn python3 for stdtr/btdtr oracle: {e}"
            );
            eprintln!("skipping stdtr/btdtr oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open stdtr/btdtr oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "stdtr/btdtr oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping stdtr/btdtr oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for stdtr/btdtr oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "stdtr/btdtr oracle failed: {stderr}"
        );
        eprintln!("skipping stdtr/btdtr oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse stdtr/btdtr oracle JSON"))
}

#[test]
fn diff_special_stdtr_btdtr() {
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
            && let Some(rust_v) = fsci_eval(&case.func, case.a, case.b, case.arg)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            let scale = scipy_v.abs().max(1.0);
            let rel_diff = abs_diff / scale;
            max_abs_overall = max_abs_overall.max(abs_diff);
            max_rel_overall = max_rel_overall.max(rel_diff);

            let pass = match case.func.as_str() {
                "stdtr" | "stdtrc" | "btdtr" | "btdtrc" => abs_diff <= CDF_TOL,
                "stdtrit" | "btdtri" => abs_diff <= PPF_TOL_REL * scale,
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
        test_id: "diff_special_stdtr_btdtr".into(),
        category: "scipy.special.stdtr/btdtr family".into(),
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
                "stdtr/btdtr {} mismatch: {} abs={} rel={}",
                d.func, d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special stdtr/btdtr conformance failed: {} cases, max_abs={} max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
}
