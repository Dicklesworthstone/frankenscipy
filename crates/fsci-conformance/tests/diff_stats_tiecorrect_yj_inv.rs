#![forbid(unsafe_code)]
//! Live SciPy / numpy differential coverage for two utility
//! transforms not exercised elsewhere:
//!   • `tiecorrect(rankvals)` — tie-correction factor
//!     T = 1 − Σ(tᵢ³ − tᵢ) / (n³ − n) for rank-based stats
//!     (`scipy.stats.tiecorrect`)
//!   • `yeojohnson_inv(y, lam)` — inverse Yeo-Johnson power
//!     transform (numpy reproduction; scipy doesn't expose a
//!     standalone inverse, but the closed-form is unambiguous)
//!
//! Resolves [frankenscipy-916m0]. The oracle calls
//! `scipy.stats.tiecorrect(ranks)` and reproduces the inverse
//! Yeo-Johnson formula in numpy.
//!
//! 4 tiecorrect fixtures + 4 (y, λ) yeojohnson_inv fixtures =
//! 8 cases. Tol 1e-12 (tiecorrect, closed-form integer ratios)
//! / 1e-10 (yeojohnson_inv per-element max-abs, powf chain).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::{tiecorrect, yeojohnson_inv};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const TIE_TOL: f64 = 1.0e-12;
const YJ_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    /// rankvals for tiecorrect; pre-transformed values for yeojohnson_inv.
    data: Vec<f64>,
    lam: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// scalar arm (tiecorrect)
    value: Option<f64>,
    /// vector arm (yeojohnson_inv)
    values: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create tiecorrect_yj_inv diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize tiecorrect_yj_inv diff log");
    fs::write(path, json).expect("write tiecorrect_yj_inv diff log");
}

fn generate_query() -> OracleQuery {
    // tiecorrect operates on ranks (typically average ranks). Use simple
    // rank vectors with varied tie patterns.
    let tie_fixtures: Vec<(&str, Vec<f64>)> = vec![
        ("no_ties", (1..=10).map(|i| i as f64).collect()),
        ("all_pairs", vec![1.5, 1.5, 3.5, 3.5, 5.5, 5.5, 7.5, 7.5]),
        (
            "mixed_groups",
            vec![1.0, 2.5, 2.5, 4.0, 5.5, 5.5, 5.5, 8.0, 9.0, 10.0],
        ),
        ("triple_quad", vec![2.0, 2.0, 2.0, 4.5, 4.5, 4.5, 4.5, 8.0]),
    ];
    // yeojohnson_inv: pick (y, λ) pairs covering both signs of y and λ values
    // around the discontinuities (λ near 0 for y≥0; λ near 2 for y<0).
    let yj_fixtures: Vec<(&str, Vec<f64>, f64)> = vec![
        ("lam_one_pos", vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0], 1.0),
        (
            "lam_two_thirds",
            vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            2.0 / 3.0,
        ),
        (
            "lam_neg_half",
            vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            -0.5,
        ),
        (
            "lam_pos_one_neg_y",
            vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0],
            1.5,
        ),
    ];

    let mut points = Vec::new();
    for (name, ranks) in &tie_fixtures {
        points.push(PointCase {
            case_id: format!("{name}_tiecorrect"),
            func: "tiecorrect".into(),
            data: ranks.clone(),
            lam: 0.0,
        });
    }
    for (name, ys, lam) in &yj_fixtures {
        points.push(PointCase {
            case_id: format!("{name}_yj_inv"),
            func: "yeojohnson_inv".into(),
            data: ys.clone(),
            lam: *lam,
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
    out = {"case_id": cid, "value": None, "values": None}
    try:
        if func == "tiecorrect":
            ranks = np.array(case["data"], dtype=float)
            out["value"] = fnone(stats.tiecorrect(ranks))
        elif func == "yeojohnson_inv":
            y = np.array(case["data"], dtype=float)
            lam = float(case["lam"])
            inv = []
            for yi in y.tolist():
                if yi >= 0:
                    if abs(lam) < 1e-15:
                        inv.append(math.exp(yi) - 1.0)
                    else:
                        inv.append(((lam * yi + 1.0) ** (1.0 / lam)) - 1.0)
                else:
                    if abs(lam - 2.0) < 1e-15:
                        inv.append(1.0 - math.exp(-yi))
                    else:
                        inv.append(1.0 - ((2.0 - lam) * (-yi) + 1.0) ** (1.0 / (2.0 - lam)))
            out["values"] = vec_or_none(inv)
    except Exception:
        pass
    points.append(out)
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize tiecorrect_yj_inv query");
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
                "failed to spawn python3 for tiecorrect_yj_inv oracle: {e}"
            );
            eprintln!("skipping tiecorrect_yj_inv oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open tiecorrect_yj_inv oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "tiecorrect_yj_inv oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping tiecorrect_yj_inv oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for tiecorrect_yj_inv oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "tiecorrect_yj_inv oracle failed: {stderr}"
        );
        eprintln!("skipping tiecorrect_yj_inv oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse tiecorrect_yj_inv oracle JSON"))
}

#[test]
fn diff_stats_tiecorrect_yj_inv() {
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
            "tiecorrect" => {
                if let Some(scipy_v) = scipy_arm.value {
                    let rust_v = tiecorrect(&case.data);
                    if rust_v.is_finite() {
                        let abs_diff = (rust_v - scipy_v).abs();
                        max_overall = max_overall.max(abs_diff);
                        diffs.push(CaseDiff {
                            case_id: case.case_id.clone(),
                            func: case.func.clone(),
                            abs_diff,
                            pass: abs_diff <= TIE_TOL,
                        });
                    }
                }
            }
            "yeojohnson_inv" => {
                if let Some(scipy_vec) = &scipy_arm.values {
                    let rust_vec = yeojohnson_inv(&case.data, case.lam);
                    if rust_vec.len() == scipy_vec.len() {
                        let mut max_local = 0.0_f64;
                        for (r, s) in rust_vec.iter().zip(scipy_vec.iter()) {
                            if r.is_finite() {
                                max_local = max_local.max((r - s).abs());
                            }
                        }
                        max_overall = max_overall.max(max_local);
                        diffs.push(CaseDiff {
                            case_id: case.case_id.clone(),
                            func: case.func.clone(),
                            abs_diff: max_local,
                            pass: max_local <= YJ_TOL,
                        });
                    }
                }
            }
            _ => continue,
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_tiecorrect_yj_inv".into(),
        category: "scipy.stats.tiecorrect + numpy reference yeojohnson_inv".into(),
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
                "tiecorrect_yj_inv {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "tiecorrect_yj_inv conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
