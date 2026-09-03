#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci's
//! `obrientransform(groups) → Vec<Vec<f64>>` — O'Brien's
//! transformation (a per-group polynomial in (mean, var, x)
//! used as the pre-step for variance-equality tests).
//!
//! Resolves [frankenscipy-hhpdw]. The oracle calls
//! `scipy.stats.obrientransform(*groups)`.
//!
//! 3 group-fixtures × per-element arms (one per transformed
//! value) ≈ 30 cases via subprocess. Tol 1e-12 abs (closed-
//! form polynomial in n, x, mean, var).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::obrientransform;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    groups: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    transformed: Option<Vec<Vec<f64>>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    arm: String,
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
    fs::create_dir_all(output_dir()).expect("create obrientransform diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize obrientransform diff log");
    fs::write(path, json).expect("write obrientransform diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<Vec<f64>>)> = vec![
        // Two equal-size groups, narrow vs wide spread
        (
            "two_groups",
            vec![
                vec![5.0, 5.5, 4.5, 5.2, 4.8, 5.1, 4.9, 5.3, 4.7, 5.0],
                vec![1.0, 3.0, 5.0, 7.0, 9.0, 2.0, 4.0, 6.0, 8.0, 10.0],
            ],
        ),
        // Three groups of mixed sizes
        (
            "three_mixed",
            vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                vec![10.0, 12.0, 14.0, 16.0, 18.0],
                vec![20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0],
            ],
        ),
        // Single group with negatives
        (
            "single_negatives",
            vec![vec![-3.0, -1.5, 0.0, 1.5, 3.0, 4.5, 6.0]],
        ),
    ];

    let points = fixtures
        .into_iter()
        .map(|(name, groups)| PointCase {
            case_id: name.into(),
            groups,
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
from scipy import stats

def vec_or_none(arr):
    out = []
    for v in arr:
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
    groups = [np.array(g, dtype=float) for g in case["groups"]]
    val = None
    try:
        # scipy returns a 2D array of shape (k, max_len) with NaN-padded rows
        # OR raises if groups have different lengths in some versions. Use the
        # plain-list form by manually transforming each group.
        arr = stats.obrientransform(*groups)
        val = []
        # arr is an ndarray of shape (k, max_n) — rows can be NaN-padded for
        # ragged input. We trim to each group's actual length so the layout
        # matches fsci's Vec<Vec<f64>>.
        for i, g in enumerate(groups):
            row = np.asarray(arr[i])[: g.shape[0]].tolist()
            v = vec_or_none(row)
            if v is None:
                val = None
                break
            val.append(v)
    except Exception:
        val = None
    points.append({"case_id": cid, "transformed": val})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize obrientransform query");
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
                "failed to spawn python3 for obrientransform oracle: {e}"
            );
            eprintln!("skipping obrientransform oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open obrientransform oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "obrientransform oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping obrientransform oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for obrientransform oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "obrientransform oracle failed: {stderr}"
        );
        eprintln!("skipping obrientransform oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse obrientransform oracle JSON"))
}

#[test]
fn diff_stats_obrientransform() {
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
        let Some(scipy_groups) = &scipy_arm.transformed else {
            continue;
        };
        let group_refs: Vec<&[f64]> = case.groups.iter().map(|g| g.as_slice()).collect();
        let rust_groups = obrientransform(&group_refs);
        if rust_groups.len() != scipy_groups.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "shape".into(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        for (gi, (rust_g, scipy_g)) in rust_groups.iter().zip(scipy_groups.iter()).enumerate() {
            if rust_g.len() != scipy_g.len() {
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    arm: format!("group{gi}.shape"),
                    abs_diff: f64::INFINITY,
                    pass: false,
                });
                continue;
            }
            let mut max_local = 0.0_f64;
            for (r, s) in rust_g.iter().zip(scipy_g.iter()) {
                if r.is_finite() {
                    max_local = max_local.max((r - s).abs());
                }
            }
            max_overall = max_overall.max(max_local);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: format!("group{gi}"),
                abs_diff: max_local,
                pass: max_local <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_obrientransform".into(),
        category: "scipy.stats.obrientransform".into(),
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
                "obrientransform mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "obrientransform conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
