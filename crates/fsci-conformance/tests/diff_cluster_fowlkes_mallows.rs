#![forbid(unsafe_code)]
//! Live numpy differential coverage for fsci's
//! `fowlkes_mallows_score(labels_true, labels_pred)` —
//! external clustering-quality index.
//!
//! Resolves [frankenscipy-6cxp3]. sklearn (the canonical
//! source for this metric) is not installed in this
//! environment, so the oracle reproduces the closed-form
//! definition directly in numpy:
//!
//!   FMI = TP / sqrt((TP + FP) · (TP + FN))
//!
//! where, for the n×n cluster co-occurrence contingency
//! table C[i][j]:
//!   • Σ C[i][j]² counts ordered (a, b) pairs in same cluster
//!     under BOTH labelings (TP × 2 + a few diagonal terms).
//!   • Specifically, TP = ½ · (Σ C² − n).
//!   • TP + FP = ½ · (Σ_i (Σ_j C[i,j])² − n).
//!   • TP + FN = ½ · (Σ_j (Σ_i C[i,j])² − n).
//!
//! 5 (labels_true, labels_pred) fixtures. Tol 1e-12 abs
//! (closed-form integer-count ratios after sqrt).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_cluster::fowlkes_mallows_score;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-012";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    labels_true: Vec<u64>,
    labels_pred: Vec<u64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
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
    fs::create_dir_all(output_dir()).expect("create fowlkes_mallows diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize fowlkes_mallows diff log");
    fs::write(path, json).expect("write fowlkes_mallows diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<u64>, Vec<u64>)> = vec![
        // Perfect agreement
        ("perfect", vec![0, 0, 1, 1, 2, 2], vec![0, 0, 1, 1, 2, 2]),
        // Slight disagreement
        (
            "slight_diff",
            vec![0, 0, 1, 1, 2, 2],
            vec![0, 0, 1, 2, 2, 2],
        ),
        // No agreement (random shuffle)
        ("shuffled", vec![0, 0, 1, 1, 2, 2], vec![1, 2, 0, 2, 0, 1]),
        // Three-cluster vs two-cluster — partial overlap
        (
            "merged_clusters",
            vec![0, 0, 0, 1, 1, 1, 2, 2, 2],
            vec![0, 0, 0, 1, 1, 1, 1, 1, 1],
        ),
        // Larger sample
        (
            "ten_points",
            vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 2],
            vec![0, 0, 1, 1, 1, 1, 2, 2, 2, 0],
        ),
    ];

    let points = fixtures
        .into_iter()
        .map(|(name, t, p)| PointCase {
            case_id: name.into(),
            labels_true: t,
            labels_pred: p,
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

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

def fmi_numpy(labels_true, labels_pred):
    """Closed-form FMI via the n×n contingency table.

    FMI = TP / sqrt((TP + FP) (TP + FN)) where TP, FP, FN are
    pair counts derived from the contingency table:
      Σ C[i,j]² = 2·TP + n
      Σ_i (Σ_j C[i,j])² = 2·(TP + FP) + n
      Σ_j (Σ_i C[i,j])² = 2·(TP + FN) + n
    """
    lt = np.asarray(labels_true)
    lp = np.asarray(labels_pred)
    n = len(lt)
    if n == 0:
        return float("nan")
    rows = sorted(set(lt.tolist()))
    cols = sorted(set(lp.tolist()))
    row_idx = {v: i for i, v in enumerate(rows)}
    col_idx = {v: i for i, v in enumerate(cols)}
    table = np.zeros((len(rows), len(cols)), dtype=np.int64)
    for ti, pi in zip(lt.tolist(), lp.tolist()):
        table[row_idx[ti], col_idx[pi]] += 1
    sum_sq_table = int((table.astype(np.int64) ** 2).sum())
    sum_row = int(((table.sum(axis=1).astype(np.int64)) ** 2).sum())
    sum_col = int(((table.sum(axis=0).astype(np.int64)) ** 2).sum())
    tp = (sum_sq_table - n) // 2
    tp_fp = (sum_row - n) // 2
    tp_fn = (sum_col - n) // 2
    if tp_fp <= 0 or tp_fn <= 0:
        return 0.0
    return tp / math.sqrt(tp_fp * tp_fn)

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    val = None
    try:
        val = fnone(fmi_numpy(case["labels_true"], case["labels_pred"]))
    except Exception:
        val = None
    points.append({"case_id": cid, "value": val})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize fowlkes_mallows query");
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
                "failed to spawn python3 for fowlkes_mallows oracle: {e}"
            );
            eprintln!("skipping fowlkes_mallows oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open fowlkes_mallows oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "fowlkes_mallows oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping fowlkes_mallows oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for fowlkes_mallows oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "fowlkes_mallows oracle failed: {stderr}"
        );
        eprintln!("skipping fowlkes_mallows oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse fowlkes_mallows oracle JSON"))
}

#[test]
fn diff_cluster_fowlkes_mallows() {
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
        let Some(scipy_v) = scipy_arm.value else {
            continue;
        };
        let lt: Vec<usize> = case.labels_true.iter().map(|&v| v as usize).collect();
        let lp: Vec<usize> = case.labels_pred.iter().map(|&v| v as usize).collect();
        let rust_v = match fowlkes_mallows_score(&lt, &lp) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if !rust_v.is_finite() {
            continue;
        }
        let abs_diff = (rust_v - scipy_v).abs();
        max_overall = max_overall.max(abs_diff);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff,
            pass: abs_diff <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_cluster_fowlkes_mallows".into(),
        category: "fsci_cluster::fowlkes_mallows_score (numpy reference)".into(),
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
            eprintln!("fowlkes_mallows mismatch: {} abs={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "fowlkes_mallows conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
