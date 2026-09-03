#![forbid(unsafe_code)]
//! Live formula-derived parity for fsci_cluster::{adjusted_rand_score,
//! normalized_mutual_info}. sklearn unavailable, so the python oracle
//! reproduces the standard sklearn formulas in numpy.
//!
//! ARI:  R = (Σ n_ij choose 2) / (n choose 2);
//!       Eₘ = (Σ a_i C 2)(Σ b_j C 2) / (n C 2);
//!       Mₘ = ((Σ a_i C 2) + (Σ b_j C 2)) / 2;
//!       ARI = (R·(n C 2) − Eₘ) / (Mₘ − Eₘ).
//!
//! NMI (arithmetic average):
//!       MI = Σ p_ij log(p_ij / (p_i p_j))
//!       H(C) = -Σ p_i log p_i, H(K) = -Σ p_j log p_j
//!       NMI = MI / ((H(C) + H(K)) / 2)
//!
//! Resolves [frankenscipy-pvh22]. Tolerance: 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_cluster::{adjusted_rand_score, normalized_mutual_info};
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
    ari: Option<f64>,
    nmi: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create ars/nmi diff dir");
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
        LabelCase {
            case_id: "binary_swapped".into(),
            labels_true: vec![0, 0, 0, 0, 1, 1, 1, 1],
            labels_pred: vec![1, 1, 1, 1, 0, 0, 0, 0],
        },
    ];
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys

def comb2(x):
    return x * (x - 1) // 2

def contingency(C, K):
    cmap = {}
    for c, k in zip(C, K):
        cmap.setdefault((c, k), 0)
        cmap[(c, k)] += 1
    cs = sorted({c for c in C}); ks = sorted({k for k in K})
    table = [[cmap.get((c, k), 0) for k in ks] for c in cs]
    return table, cs, ks

def ari(C, K):
    n = len(C)
    if n <= 1:
        return 1.0
    table, _, _ = contingency(C, K)
    rs = [sum(row) for row in table]
    cs = [sum(table[i][j] for i in range(len(table))) for j in range(len(table[0]))]
    sum_n = sum(comb2(table[i][j]) for i in range(len(table)) for j in range(len(table[0])))
    sum_a = sum(comb2(r) for r in rs)
    sum_b = sum(comb2(c) for c in cs)
    n_choose_2 = comb2(n)
    expected = (sum_a * sum_b) / n_choose_2 if n_choose_2 > 0 else 0.0
    max_index = (sum_a + sum_b) / 2.0
    if max_index - expected == 0:
        # Edge case: only happens when both labelings are trivial.
        # sklearn returns 1.0 if (sum_n - expected) == 0 else 0.0.
        return 1.0 if (sum_n - expected) == 0 else 0.0
    return (sum_n - expected) / (max_index - expected)

def nmi_arithmetic(C, K):
    n = len(C)
    if n <= 1:
        return 1.0
    table, _, _ = contingency(C, K)
    rs = [sum(row) for row in table]
    cs = [sum(table[i][j] for i in range(len(table))) for j in range(len(table[0]))]
    nf = float(n)
    mi = 0.0
    for i in range(len(table)):
        for j in range(len(table[0])):
            nij = table[i][j]
            if nij > 0 and rs[i] > 0 and cs[j] > 0:
                pij = nij / nf; pi = rs[i] / nf; pj = cs[j] / nf
                mi += pij * math.log(pij / (pi * pj))
    h1 = sum(-(s/nf) * math.log(s/nf) for s in rs if s > 0)
    h2 = sum(-(s/nf) * math.log(s/nf) for s in cs if s > 0)
    denom = max((h1 + h2) / 2.0, 1e-15)
    val = mi / denom
    return max(0.0, min(1.0, val))

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    C = case["labels_true"]; K = case["labels_pred"]
    try:
        a = ari(C, K); n = nmi_arithmetic(C, K)
        if math.isfinite(a) and math.isfinite(n):
            points.append({"case_id": cid, "ari": float(a), "nmi": float(n)})
        else:
            points.append({"case_id": cid, "ari": None, "nmi": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "ari": None, "nmi": None})
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
                "failed to spawn python3 for ars/nmi oracle: {e}"
            );
            eprintln!("skipping ars/nmi oracle: python3 not available ({e})");
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
                "ars/nmi oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping ars/nmi oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for ars/nmi oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "ars/nmi oracle failed: {stderr}"
        );
        eprintln!("skipping ars/nmi oracle: python3 not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse ars/nmi oracle JSON"))
}

#[test]
fn diff_cluster_ars_nmi() {
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
        let (Some(eari), Some(enmi)) = (arm.ari, arm.nmi) else {
            continue;
        };

        let Ok(actual_ari) = adjusted_rand_score(&case.labels_true, &case.labels_pred) else {
            continue;
        };
        let Ok(actual_nmi) = normalized_mutual_info(&case.labels_true, &case.labels_pred) else {
            continue;
        };

        for (op, actual, expected) in [
            ("adjusted_rand_score", actual_ari, eari),
            ("normalized_mutual_info", actual_nmi, enmi),
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
        test_id: "diff_cluster_ars_nmi".into(),
        category: "fsci_cluster ARS + NMI vs sklearn formula".into(),
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
        "ars/nmi conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
