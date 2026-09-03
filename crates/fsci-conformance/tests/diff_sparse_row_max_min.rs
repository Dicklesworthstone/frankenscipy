#![forbid(unsafe_code)]
//! Live scipy.sparse parity for fsci_sparse::sparse_row_max and
//! sparse_row_min.
//!
//! Resolves [frankenscipy-xmtaa]. Both ops account for implicit
//! zeros (a row's max/min considers the implicit 0s mixed in with
//! stored nonzeros). 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{CooMatrix, FormatConvertible, Shape2D, sparse_row_max, sparse_row_min};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "max" | "min"
    rows: usize,
    cols: usize,
    triplets: Vec<(usize, usize, f64)>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    values: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create row_max_min diff dir");
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
    let mut points = Vec::new();
    let a_3x4 = vec![
        (0, 0, 3.0_f64),
        (0, 2, -1.0),
        (1, 1, -2.0),
        (1, 3, 4.0),
        (2, 0, -5.0),
        (2, 1, 1.5),
        (2, 2, 0.5),
    ];
    let all_pos_4x4 = vec![
        (0, 0, 1.0_f64),
        (0, 1, 2.0),
        (1, 2, 3.0),
        (2, 0, 4.0),
        (2, 3, 5.0),
        (3, 1, 6.0),
    ];
    let all_neg_3x3 = vec![(0, 0, -2.0_f64), (1, 1, -3.0), (2, 2, -4.0)];

    for (label, rows, cols, t) in [
        ("mixed_3x4", 3_usize, 4_usize, &a_3x4),
        ("all_pos_4x4", 4, 4, &all_pos_4x4),
        ("all_neg_3x3", 3, 3, &all_neg_3x3),
    ] {
        for op in ["max", "min"] {
            points.push(Case {
                case_id: format!("{op}_{label}"),
                op: op.into(),
                rows,
                cols,
                triplets: t.clone(),
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
from scipy.sparse import csr_matrix

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    rows = int(case["rows"]); cols = int(case["cols"])
    rs = [int(t[0]) for t in case["triplets"]]
    cs = [int(t[1]) for t in case["triplets"]]
    vs = [float(t[2]) for t in case["triplets"]]
    try:
        A = csr_matrix((vs, (rs, cs)), shape=(rows, cols)).astype(float)
        if op == "max":
            r = A.max(axis=1)
        elif op == "min":
            r = A.min(axis=1)
        else:
            points.append({"case_id": cid, "values": None}); continue
        flat = [float(v) for v in np.asarray(r.todense()).flatten().tolist()]
        if all(math.isfinite(v) for v in flat):
            points.append({"case_id": cid, "values": flat})
        else:
            points.append({"case_id": cid, "values": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "values": None})
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
                "failed to spawn python3 for row_max_min oracle: {e}"
            );
            eprintln!("skipping row_max_min oracle: python3 not available ({e})");
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
                "row_max_min oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping row_max_min oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for row_max_min oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "row_max_min oracle failed: {stderr}"
        );
        eprintln!("skipping row_max_min oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse row_max_min oracle JSON"))
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
fn diff_sparse_row_max_min() {
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
        let Some(expected) = arm.values.as_ref() else {
            continue;
        };
        let mut data = Vec::new();
        let mut rs = Vec::new();
        let mut cs = Vec::new();
        for &(r, c, v) in &case.triplets {
            data.push(v);
            rs.push(r);
            cs.push(c);
        }
        let Ok(coo) =
            CooMatrix::from_triplets(Shape2D::new(case.rows, case.cols), data, rs, cs, true)
        else {
            continue;
        };
        let Ok(csr) = coo.to_csr() else {
            continue;
        };
        let result = match case.op.as_str() {
            "max" => sparse_row_max(&csr),
            "min" => sparse_row_min(&csr),
            _ => continue,
        };
        let abs_d = vec_max_diff(&result, expected);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_sparse_row_max_min".into(),
        category: "fsci_sparse::{sparse_row_max, sparse_row_min} vs scipy.sparse {max,min}(axis=1)"
            .into(),
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
        "row_max_min conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
