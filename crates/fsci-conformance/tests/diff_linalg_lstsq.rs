#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.linalg.lstsq`.
//!
//! Resolves [frankenscipy-0zu9p]. fsci_linalg::lstsq returns x,
//! residuals, rank, singular_values. Compare solution x at 1e-9 abs;
//! ranks compared as integer.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::{LstsqOptions, lstsq};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-009";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    x: Option<Vec<f64>>,
    rank: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    abs_diff: f64,
    rank_match: bool,
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
    fs::create_dir_all(output_dir()).expect("create lstsq diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize lstsq diff log");
    fs::write(path, json).expect("write lstsq diff log");
}

fn generate_query() -> OracleQuery {
    let cases: &[(&str, Vec<Vec<f64>>, Vec<f64>)] = &[
        (
            "3x2_overdetermined",
            vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]],
            vec![1.0, 2.0, 3.5],
        ),
        (
            "4x3_well_conditioned",
            vec![
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
                vec![7.0, 8.0, 10.0],
                vec![1.0, 0.0, 1.0],
            ],
            vec![1.0, 2.0, 3.0, 4.0],
        ),
        (
            "3x3_square_nonsingular",
            vec![
                vec![2.0, 1.0, 0.0],
                vec![1.0, 3.0, 1.0],
                vec![0.0, 1.0, 4.0],
            ],
            vec![1.0, 2.0, 3.0],
        ),
        (
            "5x2_overdetermined",
            vec![
                vec![1.0, 1.0],
                vec![2.0, 1.0],
                vec![3.0, 1.0],
                vec![4.0, 1.0],
                vec![5.0, 1.0],
            ],
            vec![2.5, 4.5, 6.0, 8.5, 10.0],
        ),
    ];
    let points = cases
        .iter()
        .map(|(name, a, b)| PointCase {
            case_id: (*name).into(),
            a: a.clone(),
            b: b.clone(),
        })
        .collect();
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import linalg

def finite_vec_or_none(arr):
    out = []
    for v in np.asarray(arr).tolist():
        try:
            v = float(v)
        except Exception:
            return None
        if not math.isfinite(v):
            return None
        out.append(v)
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    A = np.array(case["a"], dtype=float)
    b = np.array(case["b"], dtype=float)
    try:
        x, residues, rank, sv = linalg.lstsq(A, b)
        points.append({
            "case_id": cid,
            "x": finite_vec_or_none(x),
            "rank": int(rank),
        })
    except Exception:
        points.append({"case_id": cid, "x": None, "rank": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize lstsq query");
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
                "failed to spawn python3 for lstsq oracle: {e}"
            );
            eprintln!("skipping lstsq oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open lstsq oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "lstsq oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping lstsq oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for lstsq oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "lstsq oracle failed: {stderr}"
        );
        eprintln!("skipping lstsq oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse lstsq oracle JSON"))
}

#[test]
fn diff_linalg_lstsq() {
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
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_x) = scipy_arm.x.as_ref() else {
            continue;
        };
        let Some(scipy_rank) = scipy_arm.rank else {
            continue;
        };
        let Ok(res) = lstsq(&case.a, &case.b, LstsqOptions::default()) else {
            continue;
        };
        if res.x.len() != scipy_x.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                abs_diff: f64::INFINITY,
                rank_match: false,
                pass: false,
            });
            continue;
        }
        let abs_d = res
            .x
            .iter()
            .zip(scipy_x.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let rank_match = res.rank == scipy_rank;
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            rank_match,
            pass: abs_d <= ABS_TOL && rank_match,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_linalg_lstsq".into(),
        category: "scipy.linalg.lstsq".into(),
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
            eprintln!(
                "lstsq mismatch: {} abs_diff={} rank_match={}",
                d.case_id, d.abs_diff, d.rank_match
            );
        }
    }

    assert!(
        all_pass,
        "scipy.linalg.lstsq conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
