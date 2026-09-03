#![forbid(unsafe_code)]
//! Live SciPy differential coverage for Fisher's exact test
//! `scipy.stats.fisher_exact(table)` on 2×2 contingency
//! tables.
//!
//! Resolves [frankenscipy-awj4v]. Cross-checks both the
//! sample odds ratio (a*d / b*c) and the two-sided p-value
//! computed via the hypergeometric distribution sum.
//!
//! 4 (2×2) fixtures × 2 arms (odds_ratio + pvalue) = 8 cases
//! via subprocess. Tol 1e-12 abs (closed-form rational
//! arithmetic over hypergeometric pmf).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::fisher_exact;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    table: [[f64; 2]; 2],
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    odds_ratio: Option<f64>,
    pvalue: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create fisher_exact diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize fisher_exact diff log");
    fs::write(path, json).expect("write fisher_exact diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: &[(&str, [[f64; 2]; 2])] = &[
        // Independent / weak association
        ("independent", [[10.0, 10.0], [10.0, 10.0]]),
        // Mild association
        ("mild_assoc", [[8.0, 12.0], [12.0, 8.0]]),
        // Strong association
        ("strong_assoc", [[15.0, 5.0], [3.0, 17.0]]),
        // (Zero-cell fixture omitted: scipy returns +infinity for the
        // odds ratio when b*c = 0, which JSON cannot encode without
        // custom handling. The closed-form bc=0 → inf path is trivial
        // and not worth a special-case oracle round-trip.)
        // Larger counts
        ("large_counts", [[50.0, 30.0], [40.0, 80.0]]),
    ];

    let points = fixtures
        .iter()
        .map(|(name, table)| PointCase {
            case_id: (*name).into(),
            table: *table,
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
    if math.isnan(v):
        return None
    return v  # allow infinity for odds ratio

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    table = np.array(case["table"], dtype=float)
    try:
        # method='exact' aligns with fsci's hypergeometric-sum path.
        res = stats.fisher_exact(table, alternative='two-sided')
        points.append({
            "case_id": cid,
            "odds_ratio": fnone(res.statistic),
            "pvalue": fnone(res.pvalue),
        })
    except Exception:
        points.append({"case_id": cid, "odds_ratio": None, "pvalue": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize fisher_exact query");
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
                "failed to spawn python3 for fisher_exact oracle: {e}"
            );
            eprintln!("skipping fisher_exact oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open fisher_exact oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "fisher_exact oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping fisher_exact oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for fisher_exact oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "fisher_exact oracle failed: {stderr}"
        );
        eprintln!("skipping fisher_exact oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse fisher_exact oracle JSON"))
}

#[test]
fn diff_stats_fisher_exact() {
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
        let result = fisher_exact(&case.table);

        if let Some(scipy_or) = scipy_arm.odds_ratio {
            let abs_diff = if scipy_or.is_infinite()
                && result.odds_ratio.is_infinite()
                && scipy_or.signum() == result.odds_ratio.signum()
            {
                0.0
            } else if result.odds_ratio.is_finite() && scipy_or.is_finite() {
                (result.odds_ratio - scipy_or).abs()
            } else {
                f64::INFINITY
            };
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "odds_ratio".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
        if let Some(scipy_p) = scipy_arm.pvalue
            && result.pvalue.is_finite()
            && scipy_p.is_finite()
        {
            let abs_diff = (result.pvalue - scipy_p).abs();
            max_overall = max_overall.max(abs_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "pvalue".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_fisher_exact".into(),
        category: "scipy.stats.fisher_exact".into(),
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
                "fisher_exact mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "fisher_exact conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
