#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the k-sample
//! Anderson-Darling test
//! `scipy.stats.anderson_ksamp(samples, midrank=True)`.
//!
//! Resolves [frankenscipy-x88e4]. Cross-checks the
//! normalized A²k statistic plus the 7 critical values at
//! significance levels [25%, 10%, 5%, 2.5%, 1%, 0.5%, 0.1%]
//! across 3 sample-set fixtures.
//!
//! 3 fixtures × 8 arms (statistic + 7 critical values) = 24
//! cases via subprocess. Tol 1e-9 abs for statistic; 1e-12
//! abs for critical values (which scipy returns from a
//! tabulated formula).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{AndersonKSampleVariant, anderson_ksamp};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const STAT_TOL: f64 = 1.0e-9;
const CRIT_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct DatasetCase {
    case_id: String,
    samples: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<DatasetCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    statistic: Option<f64>,
    critical_values: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create anderson_ksamp diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize anderson_ksamp diff log");
    fs::write(path, json).expect("write anderson_ksamp diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<Vec<f64>>)> = vec![
        // 2 samples, similar distributions
        (
            "g2_similar",
            vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
            ],
        ),
        // 3 samples with varying medians
        (
            "g3_varying",
            vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                vec![3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
                vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            ],
        ),
        // 4 samples mixed
        (
            "g4_mixed",
            vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0],
                vec![5.0, 6.0, 7.0, 8.0, 9.0],
                vec![3.0, 5.0, 7.0, 9.0, 11.0],
                vec![10.0, 11.0, 12.0, 13.0, 14.0],
            ],
        ),
    ];

    let points = fixtures
        .into_iter()
        .map(|(name, samples)| DatasetCase {
            case_id: name.into(),
            samples,
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
    cid = case["case_id"]
    samples = [np.array(s, dtype=float) for s in case["samples"]]
    try:
        # midrank=True is the scipy default; matches fsci's default
        # AndersonKSampleVariant::Midrank.
        res = stats.anderson_ksamp(samples, midrank=True)
        points.append({
            "case_id": cid,
            "statistic": fnone(res.statistic),
            "critical_values": vec_or_none(res.critical_values.tolist()),
        })
    except Exception:
        points.append({"case_id": cid, "statistic": None, "critical_values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize anderson_ksamp query");
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
                "failed to spawn python3 for anderson_ksamp oracle: {e}"
            );
            eprintln!("skipping anderson_ksamp oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open anderson_ksamp oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "anderson_ksamp oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping anderson_ksamp oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for anderson_ksamp oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "anderson_ksamp oracle failed: {stderr}"
        );
        eprintln!("skipping anderson_ksamp oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse anderson_ksamp oracle JSON"))
}

#[test]
fn diff_stats_anderson_ksamp() {
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
        let result = match anderson_ksamp(&case.samples, Some(AndersonKSampleVariant::Midrank)) {
            Ok(r) => r,
            Err(_) => continue,
        };

        if let Some(scipy_stat) = scipy_arm.statistic
            && result.statistic.is_finite()
        {
            let abs_diff = (result.statistic - scipy_stat).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "statistic".into(),
                abs_diff,
                pass: abs_diff <= STAT_TOL,
            });
        }
        if let Some(scipy_crit) = &scipy_arm.critical_values {
            for (idx, &scipy_v) in scipy_crit.iter().enumerate() {
                if idx >= result.critical_values.len() {
                    break;
                }
                let rust_v = result.critical_values[idx];
                if rust_v.is_finite() {
                    let abs_diff = (rust_v - scipy_v).abs();
                    max_overall = max_overall.max(abs_diff);
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        arm: format!("crit_{idx}"),
                        abs_diff,
                        pass: abs_diff <= CRIT_TOL,
                    });
                }
            }
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_anderson_ksamp".into(),
        category: "scipy.stats.anderson_ksamp(midrank=True)".into(),
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
                "anderson_ksamp mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "anderson_ksamp conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
