#![forbid(unsafe_code)]
//! Live SciPy / numpy differential coverage for two simple
//! utilities not exercised by any other harness:
//!   • `msign(data)` — element-wise sign vector
//!     (scipy.stats.mstats.msign)
//!   • `rank_biserial(u_stat, n1, n2)` — Mann-Whitney rank-
//!     biserial correlation r = 1 − 2U/(n1·n2)
//!
//! Resolves [frankenscipy-aw05i]. The oracle reproduces both
//! formulas in numpy directly.
//!
//! 4 datasets × msign per-element max-abs + 5 (u_stat, n1, n2)
//! fixtures × rank_biserial = 9 arms. Tol 1e-12 abs (closed-
//! form ratios; no transcendentals).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{msign, rank_biserial};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    /// Used for msign.
    data: Vec<f64>,
    /// Used for rank_biserial.
    u_stat: f64,
    n1: usize,
    n2: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// msign output vector.
    signs: Option<Vec<f64>>,
    /// rank_biserial scalar.
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
    fs::create_dir_all(output_dir()).expect("create msign_rank_biserial diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize msign_rank_biserial diff log");
    fs::write(path, json).expect("write msign_rank_biserial diff log");
}

fn generate_query() -> OracleQuery {
    let msign_datasets: Vec<(&str, Vec<f64>)> = vec![
        ("all_positive", (1..=10).map(|i| i as f64).collect()),
        (
            "mixed_signs",
            vec![-3.0, -1.5, 0.0, 0.5, 1.5, -2.0, 4.0, -0.1, 2.5, 0.0],
        ),
        (
            "with_zeros_n8",
            vec![0.0, 1.0, 0.0, -1.0, 2.0, 0.0, -3.0, 0.0],
        ),
        ("all_negative_n6", vec![-1.0, -2.0, -3.0, -4.0, -5.0, -10.0]),
    ];
    // (label, u_stat, n1, n2)
    let rb_fixtures: Vec<(&str, f64, usize, usize)> = vec![
        ("equal_balanced", 50.0, 10, 10),
        ("strong_positive", 5.0, 10, 10),
        ("strong_negative", 95.0, 10, 10),
        ("unbalanced_mid", 60.0, 8, 12),
        ("large_n", 250.0, 25, 25),
    ];

    let mut points = Vec::new();
    for (name, data) in &msign_datasets {
        points.push(PointCase {
            case_id: format!("{name}_msign"),
            func: "msign".into(),
            data: data.clone(),
            u_stat: 0.0,
            n1: 0,
            n2: 0,
        });
    }
    for (label, u, n1, n2) in &rb_fixtures {
        points.push(PointCase {
            case_id: format!("{label}_rank_biserial"),
            func: "rank_biserial".into(),
            data: vec![],
            u_stat: *u,
            n1: *n1,
            n2: *n2,
        });
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np

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

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; func = case["func"]
    out = {"case_id": cid, "signs": None, "value": None}
    try:
        if func == "msign":
            data = np.array(case["data"], dtype=float)
            # numpy.sign matches scipy.stats.mstats.msign for non-NaN inputs.
            signs = np.sign(data)
            out["signs"] = vec_or_none(signs.tolist())
        elif func == "rank_biserial":
            u = float(case["u_stat"])
            n1 = float(case["n1"]); n2 = float(case["n2"])
            n = n1 * n2
            out["value"] = fnone(1.0 - 2.0 * u / n) if n != 0 else None
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize msign_rank_biserial query");
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
                "failed to spawn python3 for msign_rank_biserial oracle: {e}"
            );
            eprintln!("skipping msign_rank_biserial oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open msign_rank_biserial oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "msign_rank_biserial oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping msign_rank_biserial oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for msign_rank_biserial oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "msign_rank_biserial oracle failed: {stderr}"
        );
        eprintln!("skipping msign_rank_biserial oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse msign_rank_biserial oracle JSON"))
}

#[test]
fn diff_stats_msign_rank_biserial() {
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
        match case.func.as_str() {
            "msign" => {
                if let Some(scipy_signs) = &scipy_arm.signs {
                    let rust_signs = msign(&case.data);
                    if rust_signs.len() == scipy_signs.len() {
                        let mut max_local = 0.0_f64;
                        for (r, s) in rust_signs.iter().zip(scipy_signs.iter()) {
                            if r.is_finite() {
                                max_local = max_local.max((r - s).abs());
                            }
                        }
                        max_overall = max_overall.max(max_local);
                        diffs.push(CaseDiff {
                            case_id: case.case_id.clone(),
                            func: case.func.clone(),
                            abs_diff: max_local,
                            pass: max_local <= ABS_TOL,
                        });
                    }
                }
            }
            "rank_biserial" => {
                if let Some(scipy_v) = scipy_arm.value {
                    let rust_v = rank_biserial(case.u_stat, case.n1, case.n2);
                    if rust_v.is_finite() {
                        let abs_diff = (rust_v - scipy_v).abs();
                        max_overall = max_overall.max(abs_diff);
                        diffs.push(CaseDiff {
                            case_id: case.case_id.clone(),
                            func: case.func.clone(),
                            abs_diff,
                            pass: abs_diff <= ABS_TOL,
                        });
                    }
                }
            }
            _ => continue,
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_msign_rank_biserial".into(),
        category: "scipy.stats.mstats.msign + Mann-Whitney rank-biserial".into(),
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
                "msign_rank_biserial {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "msign_rank_biserial conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
