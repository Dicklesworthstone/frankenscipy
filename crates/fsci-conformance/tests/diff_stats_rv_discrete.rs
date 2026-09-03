#![forbid(unsafe_code)]
//! Live scipy.stats.rv_discrete parity for fsci_stats::RvDiscrete.
//!
//! Resolves [frankenscipy-l89iz]. Tests pmf(x) and cdf(x) at a grid of
//! query points for several (xk, pk) supports (including non-unit-spaced
//! and unsorted xk).
//!
//! Tolerance: 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::RvDiscrete;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    xk: Vec<f64>,
    pk: Vec<f64>,
    queries: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    pmf: Option<Vec<f64>>,
    cdf: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
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
    fs::create_dir_all(output_dir()).expect("create rv_discrete diff dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

fn generate_query() -> OracleQuery {
    let points = vec![
        Case {
            case_id: "uniform_4".into(),
            xk: vec![1.0, 2.0, 3.0, 4.0],
            pk: vec![0.25, 0.25, 0.25, 0.25],
            queries: vec![0.5, 1.0, 1.5, 2.5, 3.0, 4.5],
        },
        Case {
            case_id: "ascending_weighted".into(),
            xk: vec![1.0, 2.0, 3.0, 4.0],
            pk: vec![0.1, 0.2, 0.3, 0.4],
            queries: vec![0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0],
        },
        Case {
            case_id: "non_unit_spaced".into(),
            xk: vec![-2.0, 0.5, 3.0, 7.5],
            pk: vec![0.4, 0.3, 0.2, 0.1],
            queries: vec![-3.0, -2.0, 0.0, 0.5, 2.0, 3.0, 7.5, 8.0],
        },
        Case {
            case_id: "unsorted_input".into(),
            xk: vec![3.0, 1.0, 4.0, 2.0],
            pk: vec![0.3, 0.1, 0.4, 0.2],
            queries: vec![0.5, 1.0, 2.0, 3.0, 4.0, 4.5],
        },
        Case {
            case_id: "single_point".into(),
            xk: vec![5.0],
            pk: vec![1.0],
            queries: vec![4.5, 5.0, 5.5],
        },
        Case {
            case_id: "unnormalized_input".into(),
            xk: vec![1.0, 2.0, 3.0],
            pk: vec![1.0, 2.0, 1.0], // sum=4, normalized to [0.25, 0.5, 0.25]
            queries: vec![0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
        },
    ];
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import stats

def finite_or_none(arr):
    out = []
    for v in arr:
        if not math.isfinite(float(v)):
            return None
        out.append(float(v))
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    xk = [float(v) for v in case["xk"]]
    pk = [float(v) for v in case["pk"]]
    # Normalize as fsci does
    total = sum(pk); pk_norm = [p / total for p in pk]
    queries = [float(v) for v in case["queries"]]
    try:
        rv = stats.rv_discrete(values=(xk, pk_norm))
        pmf = [float(rv.pmf(x)) for x in queries]
        cdf = [float(rv.cdf(x)) for x in queries]
        points.append({
            "case_id": cid,
            "pmf": finite_or_none(pmf),
            "cdf": finite_or_none(cdf),
        })
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "pmf": None, "cdf": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize query");
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
                "failed to spawn python3 for rv_discrete oracle: {e}"
            );
            eprintln!("skipping rv_discrete oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "rv_discrete oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping rv_discrete oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for rv_discrete oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "rv_discrete oracle failed: {stderr}"
        );
        eprintln!("skipping rv_discrete oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse rv_discrete oracle JSON"))
}

fn vec_max_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn diff_stats_rv_discrete() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let Some(arm) = pmap.get(&case.case_id) else {
            continue;
        };
        let (Some(epmf), Some(ecdf)) = (arm.pmf.as_ref(), arm.cdf.as_ref()) else {
            continue;
        };
        let rv = RvDiscrete::new(case.xk.clone(), case.pk.clone());
        let pmf: Vec<f64> = case.queries.iter().map(|&x| rv.pmf(x)).collect();
        let cdf: Vec<f64> = case.queries.iter().map(|&x| rv.cdf(x)).collect();

        let pmf_diff = vec_max_diff(&pmf, epmf);
        max_overall = max_overall.max(pmf_diff);
        diffs.push(CaseDiff {
            case_id: format!("{}_pmf", case.case_id),
            op: "pmf".into(),
            abs_diff: pmf_diff,
            pass: pmf_diff <= ABS_TOL,
        });

        let cdf_diff = vec_max_diff(&cdf, ecdf);
        max_overall = max_overall.max(cdf_diff);
        diffs.push(CaseDiff {
            case_id: format!("{}_cdf", case.case_id),
            op: "cdf".into(),
            abs_diff: cdf_diff,
            pass: cdf_diff <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_rv_discrete".into(),
        category: "fsci_stats::RvDiscrete vs scipy.stats.rv_discrete".into(),
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
            eprintln!("{} mismatch: {} abs_diff={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "rv_discrete conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
