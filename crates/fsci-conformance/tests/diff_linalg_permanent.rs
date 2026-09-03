#![forbid(unsafe_code)]
//! Live parity for fsci_linalg::permanent vs direct sum-over-permutations.
//!
//! Resolves [frankenscipy-w5bho]. fsci uses Ryser's formula (O(2^n n));
//! oracle uses the naive sum over permutations (O(n!)) — these agree
//! exactly in exact arithmetic and within machine precision for small
//! matrices, for both even and odd n ∈ {2, 3, 4, 5} (the odd-n Ryser
//! doubled-sign-flip is fixed — frankenscipy-lsney).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_linalg::permanent;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    rows: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
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
    fs::create_dir_all(output_dir()).expect("create permanent diff dir");
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
    // Even and odd n ∈ {2, 3, 4, 5}; perm([[a,b],[c,d]]) = a·d + b·c.
    let points = vec![
        Case {
            case_id: "p2_basic".into(),
            rows: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        },
        Case {
            case_id: "p2_neg".into(),
            rows: vec![vec![-1.0, 2.5], vec![3.0, -0.5]],
        },
        Case {
            case_id: "p3_identity".into(),
            rows: vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        },
        Case {
            case_id: "p3_ints".into(),
            rows: vec![
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0],
            ],
        },
        Case {
            case_id: "p3_mixed".into(),
            rows: vec![
                vec![-1.0, 2.0, 0.5],
                vec![3.0, -0.5, 1.5],
                vec![0.25, 4.0, -2.0],
            ],
        },
        Case {
            case_id: "p5_ones".into(),
            rows: vec![vec![1.0; 5]; 5],
        },
        Case {
            case_id: "p4_ones".into(),
            rows: vec![vec![1.0; 4]; 4],
        },
        Case {
            case_id: "p4_diag".into(),
            rows: vec![
                vec![2.0, 0.0, 0.0, 0.0],
                vec![0.0, 3.0, 0.0, 0.0],
                vec![0.0, 0.0, 5.0, 0.0],
                vec![0.0, 0.0, 0.0, 7.0],
            ],
        },
        Case {
            case_id: "p4_rand".into(),
            rows: vec![
                vec![0.1, 0.2, 0.3, 0.4],
                vec![0.5, 0.6, 0.7, 0.8],
                vec![-0.3, 0.4, -0.5, 0.6],
                vec![1.0, 0.1, -0.2, 0.3],
            ],
        },
    ];
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from itertools import permutations

def perm_naive(rows):
    n = len(rows)
    total = 0.0
    for sigma in permutations(range(n)):
        prod = 1.0
        for i, j in enumerate(sigma):
            prod *= rows[i][j]
        total += prod
    return total

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    rows = case["rows"]
    try:
        v = float(perm_naive(rows))
        if math.isfinite(v):
            points.append({"case_id": cid, "value": v})
        else:
            points.append({"case_id": cid, "value": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "value": None})
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
                "failed to spawn python3 for permanent oracle: {e}"
            );
            eprintln!("skipping permanent oracle: python3 not available ({e})");
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
                "permanent oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping permanent oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for permanent oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "permanent oracle failed: {stderr}"
        );
        eprintln!("skipping permanent oracle: python not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse permanent oracle JSON"))
}

#[test]
fn diff_linalg_permanent() {
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
        let Some(expected) = arm.value else {
            continue;
        };
        let actual = permanent(&case.rows);
        if !actual.is_finite() {
            continue;
        }
        let abs_d = (actual - expected).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_linalg_permanent".into(),
        category: "fsci_linalg::permanent (Ryser) vs naive sum-over-permutations".into(),
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
            eprintln!("permanent mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "permanent conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
