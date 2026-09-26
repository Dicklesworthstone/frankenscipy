#![forbid(unsafe_code)]
//! Live SciPy differential coverage for three quantile-style
//! estimators not covered by any other diff harness:
//!   • `hdmedian(data)` — Harrell-Davis median (smooth
//!     quantile estimator).
//!   • `hdquantiles(data, prob)` — Harrell-Davis quantiles
//!     at multiple probabilities.
//!   • `expectile(data, alpha)` — generalized quantile that
//!     reduces to the mean at alpha=0.5.
//!
//! Resolves [frankenscipy-tgyvm]. Cross-checks against
//! scipy.stats.mstats.hdmedian / hdquantiles and
//! scipy.stats.expectile.
//!
//! 3 datasets × 3 funcs (multi-arm where applicable) =
//! ~18 cases. Tol 1e-9 abs (incomplete-beta weights for HD).

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_stats::{expectile, hdmedian, hdquantiles};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    data: Vec<f64>,
    probs: Vec<f64>,
    alpha: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    scalar: Option<f64>,
    vector: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    func: String,
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
    fs::create_dir_all(output_dir()).expect("create hd_expectile diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize hd_expectile diff log");
    fs::write(path, json).expect("write hd_expectile diff log");
}

fn generate_query() -> OracleQuery {
    let datasets: Vec<(&str, Vec<f64>)> = vec![
        ("compact", (1..=10).map(|i| i as f64).collect()),
        (
            "spread",
            vec![
                -3.0, -1.5, -0.7, 0.0, 0.5, 1.2, 2.0, 3.5, 4.7, 6.0, 8.5, 12.0,
            ],
        ),
        (
            "ties",
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
        ),
    ];

    let mut points = Vec::new();
    for (name, data) in &datasets {
        // hdmedian
        points.push(PointCase {
            case_id: format!("hdmedian_{name}"),
            func: "hdmedian".into(),
            data: data.clone(),
            probs: vec![],
            alpha: 0.0,
        });
        // hdquantiles at [0.25, 0.5, 0.75]
        points.push(PointCase {
            case_id: format!("hdquantiles_{name}"),
            func: "hdquantiles".into(),
            data: data.clone(),
            probs: vec![0.25, 0.5, 0.75],
            alpha: 0.0,
        });
        // expectile at alpha=0.3, 0.5, 0.7
        for alpha in [0.3, 0.5, 0.7] {
            points.push(PointCase {
                case_id: format!("expectile_{name}_a{alpha}"),
                func: "expectile".into(),
                data: data.clone(),
                probs: vec![],
                alpha,
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
from scipy import stats
from scipy.stats import mstats

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

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

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; func = case["func"]
    data = np.array(case["data"], dtype=float)
    out = {"case_id": cid, "scalar": None, "vector": None}
    try:
        if func == "hdmedian":
            out["scalar"] = fnone(float(mstats.hdmedian(data)))
        elif func == "hdquantiles":
            probs = np.array(case["probs"], dtype=float)
            out["vector"] = vec_or_none(mstats.hdquantiles(data, probs).tolist())
        elif func == "expectile":
            out["scalar"] = fnone(float(stats.expectile(data, alpha=float(case["alpha"]))))
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize hd_expectile query");
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
                "failed to spawn python3 for hd_expectile oracle: {e}"
            );
            eprintln!("skipping hd_expectile oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open hd_expectile oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "hd_expectile oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping hd_expectile oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for hd_expectile oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "hd_expectile oracle failed: {stderr}"
        );
        eprintln!("skipping hd_expectile oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse hd_expectile oracle JSON"))
}

#[test]
fn diff_stats_hd_expectile() {
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
    let arms = ["hdmedian", "hdquantiles", "expectile"];
    let mut ledger = CompareLedger::new("diff_stats_hd_expectile", &arms);

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        match case.func.as_str() {
            "hdmedian" => {
                let rust_v = hdmedian(&case.data);
                let Some((scipy_v, rust_v)) =
                    ledger.pair("hdmedian", &case.case_id, scipy_arm.scalar, Some(rust_v))
                else {
                    continue;
                };
                let abs_diff = (rust_v - scipy_v).abs();
                max_overall = max_overall.max(abs_diff);
                ledger.compared("hdmedian", &case.case_id, abs_diff <= ABS_TOL);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    func: case.func.clone(),
                    abs_diff,
                    pass: abs_diff <= ABS_TOL,
                });
            }
            "hdquantiles" => {
                let rust_vec = hdquantiles(&case.data, &case.probs);
                let Some((scipy_vec, rust_vec)) = ledger.slices(
                    "hdquantiles",
                    &case.case_id,
                    scipy_arm.vector.as_deref(),
                    Some(rust_vec.as_slice()),
                ) else {
                    continue;
                };
                let mut max_local = 0.0_f64;
                for (a, b) in rust_vec.iter().zip(scipy_vec.iter()) {
                    if a.is_finite() {
                        max_local = max_local.max((a - b).abs());
                    }
                }
                max_overall = max_overall.max(max_local);
                ledger.compared("hdquantiles", &case.case_id, max_local <= ABS_TOL);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    func: case.func.clone(),
                    abs_diff: max_local,
                    pass: max_local <= ABS_TOL,
                });
            }
            "expectile" => {
                let rust_v = expectile(&case.data, case.alpha);
                let Some((scipy_v, rust_v)) =
                    ledger.pair("expectile", &case.case_id, scipy_arm.scalar, Some(rust_v))
                else {
                    continue;
                };
                let abs_diff = (rust_v - scipy_v).abs();
                max_overall = max_overall.max(abs_diff);
                ledger.compared("expectile", &case.case_id, abs_diff <= ABS_TOL);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    func: case.func.clone(),
                    abs_diff,
                    pass: abs_diff <= ABS_TOL,
                });
            }
            other => panic!("hd_expectile: generate_query produced unknown func {other}"),
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_hd_expectile".into(),
        category: "hdmedian + hdquantiles + expectile".into(),
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
                "hd_expectile {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "hd_expectile conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
    // hdmedian and hdquantiles get one case per dataset, expectile three (one per alpha).
    let min_per_arm = arms
        .iter()
        .map(|arm| query.points.iter().filter(|c| c.func == *arm).count())
        .min()
        .expect("arms declared");
    ledger.finish(min_per_arm);
}
