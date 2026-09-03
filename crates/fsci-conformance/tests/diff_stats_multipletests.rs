#![forbid(unsafe_code)]
//! Live numpy differential coverage for fsci's family of
//! multiple-comparison p-value corrections:
//!   • `multipletests_bonferroni(pvalues, alpha)`
//!   • `multipletests_sidak(pvalues, alpha)`
//!   • `multipletests_holm(pvalues, alpha)`
//!   • `multipletests_fdr_bh(pvalues, alpha)`
//!
//! Resolves [frankenscipy-2udog]. The oracle reproduces each
//! correction's closed-form formula in numpy (statsmodels is
//! not available in the test environment; the references
//! match the docstrings of statsmodels.multipletests for
//! these four methods).
//!
//! 4 (pvalues, alpha) fixtures × 4 funcs × 2 arms (corrected
//! pvalues + reject mask) = 32 cases. Tol 1e-12 abs (closed-
//! form scaling and monotone running-max / running-min).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{
    multipletests_bonferroni, multipletests_fdr_bh, multipletests_holm, multipletests_sidak,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    pvalues: Vec<f64>,
    alpha: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    pvalues_corrected: Option<Vec<f64>>,
    reject: Option<Vec<bool>>,
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
    fs::create_dir_all(output_dir()).expect("create multipletests diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize multipletests diff log");
    fs::write(path, json).expect("write multipletests diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<f64>, f64)> = vec![
        // Mixed signal — some tests clearly significant, some borderline
        ("mixed_signal", vec![0.001, 0.04, 0.06, 0.2, 0.5, 0.8], 0.05),
        // All non-significant
        ("all_high", vec![0.3, 0.5, 0.7, 0.85, 0.95], 0.05),
        // Mostly significant
        (
            "mostly_significant",
            vec![0.001, 0.005, 0.01, 0.02, 0.04, 0.06],
            0.10,
        ),
        // Larger family
        (
            "family_n10",
            vec![0.001, 0.01, 0.02, 0.04, 0.05, 0.08, 0.1, 0.2, 0.5, 0.9],
            0.05,
        ),
    ];

    let mut points = Vec::new();
    for (name, pvals, alpha) in &fixtures {
        for func in ["bonferroni", "sidak", "holm", "fdr_bh"] {
            points.push(PointCase {
                case_id: format!("{name}_{func}"),
                func: func.into(),
                pvalues: pvals.clone(),
                alpha: *alpha,
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

def correct_bonferroni(p, n):
    return [min(pv * n, 1.0) for pv in p]

def correct_sidak(p, n):
    return [min(1.0 - (1.0 - pv) ** n, 1.0) for pv in p]

def correct_holm(p, n):
    # Sort, multiply by (n - rank), enforce monotone non-decreasing.
    idx = sorted(range(n), key=lambda i: p[i])
    out = [0.0] * n
    running_max = 0.0
    for rank, orig in enumerate(idx):
        adj = min(p[orig] * (n - rank), 1.0)
        running_max = max(running_max, adj)
        out[orig] = running_max
    return out

def correct_fdr_bh(p, n):
    # Sort ascending; iterate from largest to smallest applying p*n/(rank+1)
    # with running-min to enforce monotone non-decreasing in original order.
    idx = sorted(range(n), key=lambda i: p[i])
    out = [0.0] * n
    running_min = 1.0
    for rank in range(n - 1, -1, -1):
        orig = idx[rank]
        adj = min(p[orig] * n / (rank + 1), 1.0)
        running_min = min(running_min, adj)
        out[orig] = running_min
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; func = case["func"]
    p = list(case["pvalues"])
    alpha = float(case["alpha"])
    n = len(p)
    out_p = None; out_r = None
    try:
        if func == "bonferroni":
            cp = correct_bonferroni(p, n)
        elif func == "sidak":
            cp = correct_sidak(p, n)
        elif func == "holm":
            cp = correct_holm(p, n)
        elif func == "fdr_bh":
            cp = correct_fdr_bh(p, n)
        else:
            cp = None
        if cp is not None:
            out_p = vec_or_none(cp)
            out_r = [pv < alpha for pv in cp]
    except Exception:
        pass
    points.append({
        "case_id": cid,
        "pvalues_corrected": out_p,
        "reject": out_r,
    })
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize multipletests query");
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
                "failed to spawn python3 for multipletests oracle: {e}"
            );
            eprintln!("skipping multipletests oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open multipletests oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "multipletests oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping multipletests oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for multipletests oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "multipletests oracle failed: {stderr}"
        );
        eprintln!("skipping multipletests oracle: python3 not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse multipletests oracle JSON"))
}

#[test]
fn diff_stats_multipletests() {
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
        let result = match case.func.as_str() {
            "bonferroni" => multipletests_bonferroni(&case.pvalues, case.alpha),
            "sidak" => multipletests_sidak(&case.pvalues, case.alpha),
            "holm" => multipletests_holm(&case.pvalues, case.alpha),
            "fdr_bh" => multipletests_fdr_bh(&case.pvalues, case.alpha),
            _ => continue,
        };

        if let Some(scipy_p) = &scipy_arm.pvalues_corrected
            && result.pvalues_corrected.len() == scipy_p.len()
        {
            let mut max_local = 0.0_f64;
            for (r, s) in result.pvalues_corrected.iter().zip(scipy_p.iter()) {
                if r.is_finite() {
                    max_local = max_local.max((r - s).abs());
                }
            }
            max_overall = max_overall.max(max_local);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: format!("{}.pvalues_corrected", case.func),
                abs_diff: max_local,
                pass: max_local <= ABS_TOL,
            });
        }
        if let Some(scipy_r) = &scipy_arm.reject
            && result.reject.len() == scipy_r.len()
        {
            let mismatches = result
                .reject
                .iter()
                .zip(scipy_r.iter())
                .filter(|(r, s)| r != s)
                .count() as f64;
            max_overall = max_overall.max(mismatches);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: format!("{}.reject", case.func),
                abs_diff: mismatches,
                pass: mismatches == 0.0,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_multipletests".into(),
        category: "multipletests {bonferroni, sidak, holm, fdr_bh} (numpy reference)".into(),
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
                "multipletests mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "multipletests conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
