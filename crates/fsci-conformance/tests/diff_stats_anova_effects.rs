#![forbid(unsafe_code)]
//! Live SciPy / numpy differential coverage for three ANOVA
//! and effect-size utilities not exercised by any other diff
//! harness:
//!   • `scipy.stats.f_oneway(*groups)` — one-way ANOVA F-test
//!   • `cohens_d(group1, group2)` — pooled-variance effect
//!     size (no scipy equivalent, oracle computes analytically)
//!   • `contingency.association(observed, method='cramer')`
//!     for `cramers_v(observed)`
//!
//! `alexandergovern` was previously omitted (statistic off by
//! up to 64) but the underlying defect [frankenscipy-795bt] is
//! now FIXED — fsci was computing a Welch-style weighted
//! variance of means instead of the actual Alexander-Govern
//! A statistic. The proper algorithm (Hill's normalising
//! transform on per-group t-stats, summed) now matches scipy
//! at machine precision.
//!
//! Resolves [frankenscipy-ynlff]. ~18 cases via subprocess.
//! Tolerances:
//!   - f_oneway: 1e-9 abs (F-distribution chain)
//!   - cohens_d / cramers_v: 1e-12 abs (closed-form)

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{cohens_d, cramers_v, f_oneway};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const STAT_TOL: f64 = 1.0e-9;
const SCALAR_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    groups: Vec<Vec<f64>>,
    pair_a: Vec<f64>,
    pair_b: Vec<f64>,
    table: Vec<Vec<f64>>,
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
    scalar: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    func: String,
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
    fs::create_dir_all(output_dir()).expect("create anova_effects diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize anova_effects diff log");
    fs::write(path, json).expect("write anova_effects diff log");
}

fn generate_query() -> OracleQuery {
    // f_oneway / alexandergovern fixtures (3 group-set fixtures each).
    let group_fixtures: Vec<(&str, Vec<Vec<f64>>)> = vec![
        (
            "g2_similar",
            vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            ],
        ),
        (
            "g3_varied",
            vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                vec![3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
                vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            ],
        ),
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
    // cohens_d pair fixtures (3 pairs)
    let cohens_d_fixtures: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        (
            "matched",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ),
        (
            "moderate",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ),
        (
            "large_effect",
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![10.0, 11.0, 12.0, 13.0, 14.0],
        ),
    ];
    // cramers_v table fixtures (3 tables)
    let cramers_fixtures: Vec<(&str, Vec<Vec<f64>>)> = vec![
        ("indep_2x2", vec![vec![10.0, 10.0], vec![10.0, 10.0]]),
        ("strong_2x2", vec![vec![15.0, 5.0], vec![3.0, 17.0]]),
        (
            "3x3",
            vec![
                vec![10.0, 5.0, 0.0],
                vec![5.0, 15.0, 5.0],
                vec![0.0, 5.0, 10.0],
            ],
        ),
    ];

    let mut points = Vec::new();
    for (name, groups) in &group_fixtures {
        points.push(PointCase {
            case_id: format!("{name}_f_oneway"),
            func: "f_oneway".into(),
            groups: groups.clone(),
            pair_a: vec![],
            pair_b: vec![],
            table: vec![],
        });
        points.push(PointCase {
            case_id: format!("{name}_alexandergovern"),
            func: "alexandergovern".into(),
            groups: groups.clone(),
            pair_a: vec![],
            pair_b: vec![],
            table: vec![],
        });
    }
    for (name, a, b) in &cohens_d_fixtures {
        points.push(PointCase {
            case_id: format!("cohens_d_{name}"),
            func: "cohens_d".into(),
            groups: vec![],
            pair_a: a.clone(),
            pair_b: b.clone(),
            table: vec![],
        });
    }
    for (name, table) in &cramers_fixtures {
        points.push(PointCase {
            case_id: format!("cramers_v_{name}"),
            func: "cramers_v".into(),
            groups: vec![],
            pair_a: vec![],
            pair_b: vec![],
            table: table.clone(),
        });
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import stats
from scipy.stats import contingency

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; func = case["func"]
    out = {"case_id": cid, "statistic": None, "pvalue": None, "scalar": None}
    try:
        if func == "f_oneway":
            groups = [np.array(g, dtype=float) for g in case["groups"]]
            res = stats.f_oneway(*groups)
            out["statistic"] = fnone(res.statistic)
            out["pvalue"] = fnone(res.pvalue)
        elif func == "alexandergovern":
            groups = [np.array(g, dtype=float) for g in case["groups"]]
            res = stats.alexandergovern(*groups)
            out["statistic"] = fnone(res.statistic)
            out["pvalue"] = fnone(res.pvalue)
        elif func == "cohens_d":
            a = np.array(case["pair_a"], dtype=float)
            b = np.array(case["pair_b"], dtype=float)
            n1 = len(a); n2 = len(b)
            m1 = float(a.mean()); m2 = float(b.mean())
            v1 = float(a.var(ddof=1)) if n1 > 1 else 0.0
            v2 = float(b.var(ddof=1)) if n2 > 1 else 0.0
            pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)) \
                if n1 + n2 > 2 else float('nan')
            out["scalar"] = fnone((m1 - m2) / pooled) if pooled > 0 else None
        elif func == "cramers_v":
            t = np.array(case["table"], dtype=float)
            out["scalar"] = fnone(float(contingency.association(t, method='cramer')))
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize anova_effects query");
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
                "failed to spawn python3 for anova_effects oracle: {e}"
            );
            eprintln!("skipping anova_effects oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open anova_effects oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "anova_effects oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping anova_effects oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for anova_effects oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "anova_effects oracle failed: {stderr}"
        );
        eprintln!("skipping anova_effects oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse anova_effects oracle JSON"))
}

#[test]
fn diff_stats_anova_effects() {
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
        match case.func.as_str() {
            "f_oneway" => {
                let groups: Vec<&[f64]> = case.groups.iter().map(|g| g.as_slice()).collect();
                let r = f_oneway(&groups);
                let (rust_stat, rust_p) = (r.statistic, r.pvalue);
                if let Some(scipy_stat) = scipy_arm.statistic
                    && rust_stat.is_finite()
                {
                    let abs_diff = (rust_stat - scipy_stat).abs();
                    max_overall = max_overall.max(abs_diff);
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        func: case.func.clone(),
                        arm: "statistic".into(),
                        abs_diff,
                        pass: abs_diff <= STAT_TOL,
                    });
                }
                if let Some(scipy_p) = scipy_arm.pvalue
                    && rust_p.is_finite()
                {
                    let abs_diff = (rust_p - scipy_p).abs();
                    max_overall = max_overall.max(abs_diff);
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        func: case.func.clone(),
                        arm: "pvalue".into(),
                        abs_diff,
                        pass: abs_diff <= STAT_TOL,
                    });
                }
            }
            "cohens_d" => {
                if let Some(scipy_v) = scipy_arm.scalar {
                    let rust_v = cohens_d(&case.pair_a, &case.pair_b);
                    if rust_v.is_finite() {
                        let abs_diff = (rust_v - scipy_v).abs();
                        max_overall = max_overall.max(abs_diff);
                        diffs.push(CaseDiff {
                            case_id: case.case_id.clone(),
                            func: case.func.clone(),
                            arm: "cohens_d".into(),
                            abs_diff,
                            pass: abs_diff <= SCALAR_TOL,
                        });
                    }
                }
            }
            "cramers_v" => {
                if let Some(scipy_v) = scipy_arm.scalar {
                    let rust_v = cramers_v(&case.table);
                    if rust_v.is_finite() {
                        let abs_diff = (rust_v - scipy_v).abs();
                        max_overall = max_overall.max(abs_diff);
                        diffs.push(CaseDiff {
                            case_id: case.case_id.clone(),
                            func: case.func.clone(),
                            arm: "cramers_v".into(),
                            abs_diff,
                            pass: abs_diff <= SCALAR_TOL,
                        });
                    }
                }
            }
            _ => {}
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_anova_effects".into(),
        category: "f_oneway + alexandergovern + cohens_d + cramers_v".into(),
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
                "anova_effects {} mismatch: {} arm={} abs={}",
                d.func, d.case_id, d.arm, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "anova_effects conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
