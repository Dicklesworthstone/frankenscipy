#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the chi-squared
//! independence test
//! `scipy.stats.chi2_contingency(observed, correction=...)`.
//!
//! Resolves [frankenscipy-zfll1]. Cross-checks the chi²
//! statistic, p-value, degrees of freedom, and the
//! element-wise expected-frequencies array.
//!
//! 4 contingency-table fixtures × 4 arms (statistic + pvalue
//! + dof + expected_max_abs) = 16 cases via subprocess.
//!   Tol 1e-9 abs (closed-form expected sum + chi-squared cdf).

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_stats::chi2_contingency;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    table: Vec<Vec<f64>>,
    correction: bool,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    statistic: Option<f64>,
    pvalue: Option<f64>,
    dof: Option<i64>,
    expected: Option<Vec<Vec<f64>>>,
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
    fs::create_dir_all(output_dir()).expect("create chi2_contingency diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize chi2_contingency diff log");
    fs::write(path, json).expect("write chi2_contingency diff log");
}

fn generate_query() -> OracleQuery {
    let fixtures: Vec<(&str, Vec<Vec<f64>>, bool)> = vec![
        // 2×2 with Yates correction (scipy default for 2×2)
        ("2x2_yates", vec![vec![10.0, 20.0], vec![30.0, 40.0]], true),
        // 2×2 without Yates correction
        (
            "2x2_no_yates",
            vec![vec![10.0, 20.0], vec![30.0, 40.0]],
            false,
        ),
        // 2×3
        (
            "2x3",
            vec![vec![10.0, 20.0, 30.0], vec![25.0, 35.0, 40.0]],
            false,
        ),
        // 3×4 (df = 6)
        (
            "3x4",
            vec![
                vec![10.0, 15.0, 20.0, 25.0],
                vec![12.0, 18.0, 24.0, 30.0],
                vec![14.0, 21.0, 28.0, 35.0],
            ],
            false,
        ),
    ];

    let points = fixtures
        .into_iter()
        .map(|(name, table, correction)| PointCase {
            case_id: name.into(),
            table,
            correction,
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

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    table = np.array(case["table"], dtype=float)
    correction = bool(case["correction"])
    try:
        res = stats.chi2_contingency(table, correction=correction)
        # res is a Chi2ContingencyResult: (statistic, pvalue, dof, expected_freq)
        expected = res.expected_freq.tolist()
        points.append({
            "case_id": cid,
            "statistic": fnone(res.statistic),
            "pvalue": fnone(res.pvalue),
            "dof": int(res.dof),
            "expected": expected,
        })
    except Exception:
        points.append({"case_id": cid, "statistic": None, "pvalue": None,
                       "dof": None, "expected": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize chi2_contingency query");
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
                "failed to spawn python3 for chi2_contingency oracle: {e}"
            );
            eprintln!("skipping chi2_contingency oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open chi2_contingency oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "chi2_contingency oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping chi2_contingency oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for chi2_contingency oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "chi2_contingency oracle failed: {stderr}"
        );
        eprintln!("skipping chi2_contingency oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse chi2_contingency oracle JSON"))
}

#[test]
fn diff_stats_chi2_contingency() {
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
        "diff_stats_chi2_contingency",
        &["statistic", "pvalue", "dof", "expected_max"],
    );

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let result = chi2_contingency(&case.table, case.correction);

        let scalar_arms = [
            ("statistic", scipy_arm.statistic, result.statistic),
            ("pvalue", scipy_arm.pvalue, result.pvalue),
        ];
        for (arm, scipy, fsci) in scalar_arms {
            let Some((s, f)) = ledger.pair(arm, &case.case_id, scipy, Some(fsci)) else {
                continue;
            };
            let abs_diff = (f - s).abs();
            max_overall = max_overall.max(abs_diff);
            ledger.compared(arm, &case.case_id, abs_diff <= ABS_TOL);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: arm.into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
        if let Some((scipy_dof, fsci_dof)) =
            ledger.both("dof", &case.case_id, scipy_arm.dof, Some(result.dof))
        {
            let abs_diff = (fsci_dof as i64 - scipy_dof).unsigned_abs() as f64;
            max_overall = max_overall.max(abs_diff);
            ledger.compared("dof", &case.case_id, abs_diff <= ABS_TOL);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "dof".into(),
                abs_diff,
                pass: abs_diff <= ABS_TOL,
            });
        }
        // Element values go through slices on the row-major flattening (a NaN must match
        // SciPy's); the row structure is checked separately below.
        let scipy_flat = scipy_arm.expected.as_ref().map(|rows| rows.concat());
        let fsci_flat = result.expected.concat();
        if let Some((s, f)) = ledger.slices(
            "expected_max",
            &case.case_id,
            scipy_flat.as_deref(),
            Some(fsci_flat.as_slice()),
        ) {
            // Element-wise comparison; report max abs diff over the matrix.
            let row_lens = |rows: &[Vec<f64>]| rows.iter().map(Vec::len).collect::<Vec<usize>>();
            let shape_ok = scipy_arm.expected.as_deref().map(row_lens)
                == Some(row_lens(result.expected.as_slice()));
            let max_local = f
                .iter()
                .zip(s.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let pass = shape_ok && max_local <= ABS_TOL;
            ledger.compared("expected_max", &case.case_id, pass);
            if shape_ok {
                max_overall = max_overall.max(max_local);
            }
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "expected_max".into(),
                abs_diff: if shape_ok { max_local } else { f64::INFINITY },
                pass,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_chi2_contingency".into(),
        category: "scipy.stats.chi2_contingency".into(),
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
                "chi2_contingency mismatch: {} arm={} abs={}",
                d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "chi2_contingency conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
    ledger.finish(query.points.len());
}
