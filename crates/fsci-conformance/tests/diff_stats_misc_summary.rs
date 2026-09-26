#![forbid(unsafe_code)]
//! Live numerical reference checks for three closed-form
//! summary utilities not exercised by any other diff harness:
//!   • `coefficient_of_variation(data)` — sample_std / |mean|
//!   • `excess_kurtosis(data)` — m4/m2² - 3 (population/biased)
//!   • `expected_freq_uniform(observed)` — sum/k repeated k
//!     times (vector output)
//!
//! Resolves [frankenscipy-4s2ln]. The oracle reproduces each
//! formula directly in numpy.
//!
//! 3 datasets × (CV + excess_kurtosis + expected_freq_uniform)
//! = 9 cases via subprocess. Tol 1e-12 abs.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_stats::{coefficient_of_variation, excess_kurtosis, expected_freq_uniform};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    data: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    scalar: Option<f64>,
    vector: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create misc_summary diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize misc_summary diff log");
    fs::write(path, json).expect("write misc_summary diff log");
}

fn generate_query() -> OracleQuery {
    let datasets: Vec<(&str, Vec<f64>)> = vec![
        ("compact", (1..=10).map(|i| i as f64).collect()),
        (
            "spread",
            vec![-3.0, -1.5, 0.0, 1.5, 3.0, 5.0, 8.0, 12.0, 18.0, 25.0],
        ),
        (
            "ties",
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
        ),
    ];

    let mut points = Vec::new();
    for (name, data) in &datasets {
        for func in ["cv", "excess_kurtosis", "expected_freq_uniform"] {
            points.push(PointCase {
                case_id: format!("{name}_{func}"),
                func: func.into(),
                data: data.clone(),
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
import numpy as np

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

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
    cid = case["case_id"]; func = case["func"]
    data = np.array(case["data"], dtype=float)
    out = {"case_id": cid, "scalar": None, "vector": None}
    try:
        if func == "cv":
            n = len(data)
            mean = float(data.mean())
            std = float(data.std(ddof=1))
            out["scalar"] = fnone(std / abs(mean)) if mean != 0 else None
        elif func == "excess_kurtosis":
            n = len(data)
            mean = float(data.mean())
            m2 = float(np.mean((data - mean) ** 2))
            m4 = float(np.mean((data - mean) ** 4))
            out["scalar"] = fnone(m4 / (m2 ** 2) - 3.0) if m2 > 0 else None
        elif func == "expected_freq_uniform":
            total = float(data.sum())
            k = len(data)
            out["vector"] = vec_or_none([total / k] * k)
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize misc_summary query");
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
                "failed to spawn python3 for misc_summary oracle: {e}"
            );
            eprintln!("skipping misc_summary oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open misc_summary oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "misc_summary oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping misc_summary oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for misc_summary oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "misc_summary oracle failed: {stderr}"
        );
        eprintln!("skipping misc_summary oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse misc_summary oracle JSON"))
}

#[test]
fn diff_stats_misc_summary() {
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
        "diff_stats_misc_summary",
        &["cv", "excess_kurtosis", "expected_freq_uniform"],
    );

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let arm = case.func.as_str();
        let abs_diff = match arm {
            "cv" | "excess_kurtosis" => {
                let rust_v = if arm == "cv" {
                    coefficient_of_variation(&case.data)
                } else {
                    excess_kurtosis(&case.data)
                };
                let Some((scipy_v, rust_v)) =
                    ledger.pair(arm, &case.case_id, scipy_arm.scalar, Some(rust_v))
                else {
                    continue;
                };
                (rust_v - scipy_v).abs()
            }
            "expected_freq_uniform" => {
                let rust_vec = expected_freq_uniform(&case.data);
                let Some((scipy_vec, rust_vec)) = ledger.slices(
                    arm,
                    &case.case_id,
                    scipy_arm.vector.as_deref(),
                    Some(rust_vec.as_slice()),
                ) else {
                    continue;
                };
                let mut max_local = 0.0_f64;
                for (a, b) in rust_vec.iter().zip(scipy_vec.iter()) {
                    if b.is_finite() {
                        max_local = max_local.max((a - b).abs());
                    }
                }
                max_local
            }
            other => panic!("unknown func {other} in {}", case.case_id),
        };
        max_overall = max_overall.max(abs_diff);
        ledger.compared(arm, &case.case_id, abs_diff <= ABS_TOL);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            func: case.func.clone(),
            abs_diff,
            pass: abs_diff <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_misc_summary".into(),
        category: "coefficient_of_variation + excess_kurtosis + expected_freq_uniform".into(),
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
                "misc_summary {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "misc_summary conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
    // Every func runs on every dataset, so each arm has the same case count.
    ledger.finish(query.points.iter().filter(|c| c.func == "cv").count());
}
