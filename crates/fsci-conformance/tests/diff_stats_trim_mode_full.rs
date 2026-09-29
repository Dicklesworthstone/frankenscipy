#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci's
//! `trimboth(data, p)`, `trim1(data, p, tail)`, and
//! `mode_full(data) → ModeResult { mode, count }`.
//!
//! Resolves [frankenscipy-21j5c]. The existing
//! diff_stats_trimmed.rs / diff_stats_trim_mode.rs cover trim
//! statistics; this harness exercises the raw trim outputs and
//! the mode_full result directly against scipy.
//!
//! 4 datasets × {trimboth p=0.1, trim1 right p=0.2, trim1 left
//! p=0.2} + 4 datasets × mode_full × 2 arms = 20 cases. Tol
//! 1e-12 abs (sort + slice; integer counts).

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_stats::{mode_full, trim1, trimboth};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    data: Vec<f64>,
    proportion: f64,
    tail: String,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// trimboth/trim1 vector arm
    values: Option<Vec<f64>>,
    /// mode_full mode value
    mode: Option<f64>,
    /// mode_full count
    count: Option<i64>,
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
    compared: BTreeMap<String, ArmCounts>,
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
    fs::create_dir_all(output_dir()).expect("create trim_mode_full diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize trim_mode_full diff log");
    fs::write(path, json).expect("write trim_mode_full diff log");
}

fn generate_query() -> OracleQuery {
    let datasets: Vec<(&str, Vec<f64>)> = vec![
        ("compact_n10", (1..=10).map(|i| i as f64).collect()),
        (
            "spread_n15",
            vec![
                -3.0, -1.5, 0.0, 0.5, 1.5, 2.5, 3.5, 5.0, 7.0, 9.0, 12.0, 16.0, 21.0, 27.0, 34.0,
            ],
        ),
        (
            "ties_n12",
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 5.0, 6.0],
        ),
        (
            "double_mode_n10",
            vec![1.0, 2.0, 2.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 6.0],
        ),
    ];

    let mut points = Vec::new();
    for (name, data) in &datasets {
        points.push(PointCase {
            case_id: format!("{name}_trimboth_p10"),
            func: "trimboth".into(),
            data: data.clone(),
            proportion: 0.1,
            tail: String::new(),
        });
        points.push(PointCase {
            case_id: format!("{name}_trim1_right_p20"),
            func: "trim1".into(),
            data: data.clone(),
            proportion: 0.2,
            tail: "right".into(),
        });
        points.push(PointCase {
            case_id: format!("{name}_trim1_left_p20"),
            func: "trim1".into(),
            data: data.clone(),
            proportion: 0.2,
            tail: "left".into(),
        });
        points.push(PointCase {
            case_id: format!("{name}_mode_full"),
            func: "mode_full".into(),
            data: data.clone(),
            proportion: 0.0,
            tail: String::new(),
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
    data = np.array(case["data"], dtype=float)
    out = {"case_id": cid, "values": None, "mode": None, "count": None}
    try:
        if func == "trimboth":
            arr = stats.trimboth(data, float(case["proportion"]))
            arr = np.sort(np.asarray(arr))  # scipy returns unsorted views
            out["values"] = vec_or_none(arr.tolist())
        elif func == "trim1":
            arr = stats.trim1(
                data,
                float(case["proportion"]),
                tail=case["tail"],
            )
            arr = np.sort(np.asarray(arr))
            out["values"] = vec_or_none(arr.tolist())
        elif func == "mode_full":
            # scipy.stats.mode default keepdims=True returns ndarray; use the
            # 1-D output. scipy 1.11+ default 'keepdims' is False.
            res = stats.mode(data, keepdims=False)
            out["mode"] = fnone(res.mode)
            out["count"] = int(res.count) if res.count is not None else None
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize trim_mode_full query");
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
                "failed to spawn python3 for trim_mode_full oracle: {e}"
            );
            eprintln!("skipping trim_mode_full oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open trim_mode_full oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "trim_mode_full oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping trim_mode_full oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for trim_mode_full oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "trim_mode_full oracle failed: {stderr}"
        );
        eprintln!("skipping trim_mode_full oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse trim_mode_full oracle JSON"))
}

#[test]
fn diff_stats_trim_mode_full() {
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
    let mut ledger = CompareLedger::new(
        "diff_stats_trim_mode_full",
        &[
            "trimboth",
            "trim1.right",
            "trim1.left",
            "mode_full.mode",
            "mode_full.count",
        ],
    );

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        match case.func.as_str() {
            "trimboth" | "trim1" => {
                let (arm, rust_vec) = if case.func == "trimboth" {
                    (
                        "trimboth".to_owned(),
                        trimboth(&case.data, case.proportion).ok(),
                    )
                } else {
                    (
                        format!("trim1.{}", case.tail),
                        Some(trim1(&case.data, case.proportion, &case.tail)),
                    )
                };
                let Some((scipy_vec, rust_vec)) = ledger.slices(
                    &arm,
                    &case.case_id,
                    scipy_arm.values.as_deref(),
                    rust_vec.as_deref(),
                ) else {
                    continue;
                };
                let mut max_local = 0.0_f64;
                for (r, s) in rust_vec.iter().zip(scipy_vec.iter()) {
                    if r.is_finite() {
                        max_local = max_local.max((r - s).abs());
                    }
                }
                max_overall = max_overall.max(max_local);
                ledger.compared(&arm, &case.case_id, max_local <= ABS_TOL);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    arm,
                    abs_diff: max_local,
                    pass: max_local <= ABS_TOL,
                });
            }
            "mode_full" => {
                let r = mode_full(&case.data);
                if let Some((s_mode, r_mode)) = ledger.pair(
                    "mode_full.mode",
                    &case.case_id,
                    scipy_arm.mode,
                    Some(r.mode),
                ) {
                    let abs_diff = (r_mode - s_mode).abs();
                    max_overall = max_overall.max(abs_diff);
                    ledger.compared("mode_full.mode", &case.case_id, abs_diff <= ABS_TOL);
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        arm: "mode_full.mode".into(),
                        abs_diff,
                        pass: abs_diff <= ABS_TOL,
                    });
                }
                if let Some((s_count, r_count)) = ledger.both(
                    "mode_full.count",
                    &case.case_id,
                    scipy_arm.count,
                    Some(r.count),
                ) {
                    let abs_diff = (r_count as i64 - s_count).unsigned_abs() as f64;
                    max_overall = max_overall.max(abs_diff);
                    ledger.compared("mode_full.count", &case.case_id, abs_diff <= ABS_TOL);
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        arm: "mode_full.count".into(),
                        abs_diff,
                        pass: abs_diff <= ABS_TOL,
                    });
                }
            }
            other => panic!("unknown func {other} in {}", case.case_id),
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_trim_mode_full".into(),
        category: "scipy.stats.{trim1, trimboth, mode}".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
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
                "trim_mode_full mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "trim_mode_full conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
    // Every arm gets one case per dataset; trimboth is one of them.
    ledger.finish(query.points.iter().filter(|c| c.func == "trimboth").count());
}
