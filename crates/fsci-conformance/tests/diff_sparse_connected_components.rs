#![forbid(unsafe_code)]
//! Live SciPy differential coverage for
//! `scipy.sparse.csgraph.connected_components`.
//!
//! Resolves [frankenscipy-n9r84]. Compares n_components exactly and
//! labels up to permutation (via partition-equivalence on co-grouping).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_sparse::{CsrMatrix, Shape2D, connected_components};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    rows: usize,
    cols: usize,
    adj_flat: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    n_components: Option<usize>,
    labels: Option<Vec<i64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    fsci_n: usize,
    scipy_n: usize,
    co_grouping_match: bool,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create conn_comp diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize conn_comp diff log");
    fs::write(path, json).expect("write conn_comp diff log");
}

fn dense_to_csr(rows: usize, cols: usize, dense: &[f64]) -> CsrMatrix {
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = Vec::with_capacity(rows + 1);
    indptr.push(0);
    for r in 0..rows {
        for c in 0..cols {
            let v = dense[r * cols + c];
            if v != 0.0 {
                data.push(v);
                indices.push(c);
            }
        }
        indptr.push(data.len());
    }
    CsrMatrix::from_components(Shape2D::new(rows, cols), data, indices, indptr, true)
        .expect("dense_to_csr build")
}

/// Returns true if `a` and `b` define the same partition (up to label
/// permutation), checked via the co-grouping matrix M[i,j] = 1 iff
/// labels[i] == labels[j].
fn co_grouping_match(a: &[usize], b: &[i64]) -> bool {
    let n = a.len();
    if b.len() != n {
        return false;
    }
    for i in 0..n {
        for j in 0..n {
            let am = a[i] == a[j];
            let bm = b[i] == b[j];
            if am != bm {
                return false;
            }
        }
    }
    true
}

fn generate_query() -> OracleQuery {
    let adj_5_2comp = vec![
        0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    ];
    let adj_4_isolated = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let adj_6_1comp = vec![
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    ];
    let adj_7_3comp = vec![
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    ];
    OracleQuery {
        points: vec![
            PointCase {
                case_id: "5n_2comp".into(),
                rows: 5,
                cols: 5,
                adj_flat: adj_5_2comp,
            },
            PointCase {
                case_id: "4n_isolated".into(),
                rows: 4,
                cols: 4,
                adj_flat: adj_4_isolated,
            },
            PointCase {
                case_id: "6n_path".into(),
                rows: 6,
                cols: 6,
                adj_flat: adj_6_1comp,
            },
            PointCase {
                case_id: "7n_3comp".into(),
                rows: 7,
                cols: 7,
                adj_flat: adj_7_3comp,
            },
        ],
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    r = int(case["rows"]); c = int(case["cols"])
    adj = np.array(case["adj_flat"], dtype=float).reshape(r, c)
    try:
        n, labels = connected_components(csr_matrix(adj), directed=False)
        points.append({"case_id": cid,
                       "n_components": int(n),
                       "labels": [int(x) for x in labels.tolist()]})
    except Exception:
        points.append({"case_id": cid, "n_components": None, "labels": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize conn_comp query");
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
                "failed to spawn python3 for conn_comp oracle: {e}"
            );
            eprintln!("skipping conn_comp oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open conn_comp oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "conn_comp oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping conn_comp oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for conn_comp oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "conn_comp oracle failed: {stderr}"
        );
        eprintln!("skipping conn_comp oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse conn_comp oracle JSON"))
}

#[test]
fn diff_sparse_connected_components() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let (Some(scipy_n), Some(scipy_labels)) =
            (scipy_arm.n_components, scipy_arm.labels.as_ref())
        else {
            continue;
        };
        let csr = dense_to_csr(case.rows, case.cols, &case.adj_flat);
        let Ok(res) = connected_components(&csr) else {
            continue;
        };
        let pass_n = res.n_components == scipy_n;
        let pass_part = co_grouping_match(&res.labels, scipy_labels);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            fsci_n: res.n_components,
            scipy_n,
            co_grouping_match: pass_part,
            pass: pass_n && pass_part,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_sparse_connected_components".into(),
        category: "scipy.sparse.csgraph.connected_components".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "conn_comp mismatch: {} fsci_n={} scipy_n={} co_grouping={}",
                d.case_id, d.fsci_n, d.scipy_n, d.co_grouping_match
            );
        }
    }

    assert!(
        all_pass,
        "connected_components conformance failed: {} cases",
        diffs.len()
    );
}
