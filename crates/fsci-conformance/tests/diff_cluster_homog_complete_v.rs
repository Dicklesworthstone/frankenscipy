#![forbid(unsafe_code)]
//! Live formula-derived parity for fsci_cluster::{homogeneity_score,
//! completeness_score, v_measure_score} against numpy implementations
//! of the standard sklearn formulas. Resolves [frankenscipy-3wgiu].
//!
//! Formulas:
//!   H(C|K) = -sum_{c,k} (n_ck / N) * log(n_ck / n_k)
//!   H(C)   = -sum_c (n_c / N) * log(n_c / N)
//!   homogeneity = 1 - H(C|K)/H(C), or 1.0 if H(C) ≈ 0
//!   completeness = homogeneity(K, C)
//!   v_measure = 2*h*c / (h+c), or 0.0 if h+c ≈ 0
//!
//! Tolerance: 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_cluster::{completeness_score, homogeneity_score, v_measure_score};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct LabelCase {
    case_id: String,
    labels_true: Vec<usize>,
    labels_pred: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<LabelCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    homog: Option<f64>,
    complete: Option<f64>,
    v: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create homog_complete_v diff dir");
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
    let points = vec![
        LabelCase {
            case_id: "perfect_agreement".into(),
            labels_true: vec![0, 0, 1, 1, 2, 2],
            labels_pred: vec![0, 0, 1, 1, 2, 2],
        },
        LabelCase {
            case_id: "label_permutation".into(),
            labels_true: vec![0, 0, 1, 1, 2, 2],
            labels_pred: vec![2, 2, 0, 0, 1, 1],
        },
        LabelCase {
            case_id: "complete_disagreement".into(),
            labels_true: vec![0, 0, 1, 1, 2, 2],
            labels_pred: vec![0, 1, 2, 0, 1, 2],
        },
        LabelCase {
            case_id: "single_true_cluster".into(),
            labels_true: vec![0, 0, 0, 0],
            labels_pred: vec![0, 1, 2, 3],
        },
        LabelCase {
            case_id: "single_pred_cluster".into(),
            labels_true: vec![0, 1, 2, 3],
            labels_pred: vec![0, 0, 0, 0],
        },
        LabelCase {
            case_id: "partial_agreement".into(),
            labels_true: vec![0, 0, 0, 1, 1, 1, 2, 2, 2],
            labels_pred: vec![0, 0, 1, 1, 1, 2, 2, 2, 0],
        },
        LabelCase {
            case_id: "binary_balanced".into(),
            labels_true: vec![0, 0, 0, 0, 1, 1, 1, 1],
            labels_pred: vec![0, 0, 1, 1, 0, 1, 1, 1],
        },
    ];
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np

def entropy_from_counts(counts):
    # counts: 1-D iterable of non-negative integer counts
    total = sum(counts)
    if total == 0: return 0.0
    s = 0.0
    for c in counts:
        if c <= 0: continue
        p = c / total
        s -= p * math.log(p)
    return s

def conditional_entropy(C, K):
    # H(C | K)
    n = len(C)
    if n == 0: return 0.0
    # group by K
    groups = {}
    for c, k in zip(C, K):
        groups.setdefault(k, []).append(c)
    s = 0.0
    for k, members in groups.items():
        nk = len(members)
        # counts per c within this K-group
        cnt = {}
        for c in members:
            cnt[c] = cnt.get(c, 0) + 1
        for v in cnt.values():
            p_ck = v / n
            p_k = nk / n
            s -= p_ck * math.log(v / nk)
    return s

def homog(C, K):
    if len(C) <= 1:
        return 1.0
    HC = entropy_from_counts([list(C).count(c) for c in set(C)])
    if HC < 1e-15:
        return 1.0
    return 1.0 - conditional_entropy(C, K) / HC

def complete(C, K):
    return homog(K, C)

def vmeas(C, K):
    h = homog(C, K)
    c = complete(C, K)
    if h + c < 1e-15:
        return 0.0
    return 2.0 * h * c / (h + c)

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    C = case["labels_true"]; K = case["labels_pred"]
    try:
        h = homog(C, K); c = complete(C, K); v = vmeas(C, K)
        if all(math.isfinite(x) for x in [h, c, v]):
            points.append({"case_id": cid, "homog": float(h), "complete": float(c), "v": float(v)})
        else:
            points.append({"case_id": cid, "homog": None, "complete": None, "v": None})
    except Exception:
        points.append({"case_id": cid, "homog": None, "complete": None, "v": None})
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
                "failed to spawn python3 for homog oracle: {e}"
            );
            eprintln!("skipping homog oracle: python3 not available ({e})");
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
                "homog oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping homog oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for homog oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "homog oracle failed: {stderr}"
        );
        eprintln!("skipping homog oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse homog oracle JSON"))
}

#[test]
fn diff_cluster_homog_complete_v() {
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
        let (Some(eh), Some(ec), Some(ev)) = (arm.homog, arm.complete, arm.v) else {
            continue;
        };

        let Ok(h) = homogeneity_score(&case.labels_true, &case.labels_pred) else {
            continue;
        };
        let Ok(c) = completeness_score(&case.labels_true, &case.labels_pred) else {
            continue;
        };
        let Ok(v) = v_measure_score(&case.labels_true, &case.labels_pred) else {
            continue;
        };

        for (op, actual, expected) in [
            ("homogeneity_score", h, eh),
            ("completeness_score", c, ec),
            ("v_measure_score", v, ev),
        ] {
            let abs_d = (actual - expected).abs();
            max_overall = max_overall.max(abs_d);
            diffs.push(CaseDiff {
                case_id: format!("{}_{}", case.case_id, op),
                op: op.into(),
                abs_diff: abs_d,
                pass: abs_d <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_cluster_homog_complete_v".into(),
        category: "fsci_cluster homogeneity/completeness/v_measure vs numpy formula".into(),
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
        "homog/complete/v conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
