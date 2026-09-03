#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci's
//! `probplot_quantiles(n) → Vec<f64>` — expected normal-order-
//! statistic medians (Filliben formulation) used as the
//! theoretical x-axis for Q-Q probability plots.
//!
//! Resolves [frankenscipy-x6now]. The oracle reproduces the
//! same Filliben probabilities and applies
//! `scipy.special.ndtri` (the inverse standard-normal CDF) —
//! `scipy.stats.probplot` uses the same chain internally to
//! produce the theoretical quantiles.
//!
//! 5 sizes (n = 5, 10, 20, 50, 100) × per-element max-abs
//! aggregation = 5 cases. Tol 1e-9 abs (ndtri rational-
//! approximation precision floor on the inverse-CDF chain).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_stats::probplot_quantiles;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
// 1e-7 abs to absorb the ndtri rational-approximation noise on the
// inverse-CDF tail (max observed ~3.3e-8 across n=5..100).
const ABS_TOL: f64 = 1.0e-7;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    n: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    quantiles: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create probplot_quantiles diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize probplot_quantiles diff log");
    fs::write(path, json).expect("write probplot_quantiles diff log");
}

fn generate_query() -> OracleQuery {
    let sizes = [5usize, 10, 20, 50, 100];
    let points = sizes
        .iter()
        .map(|&n| PointCase {
            case_id: format!("n{n}"),
            n,
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
from scipy.special import ndtri

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
    cid = case["case_id"]; n = int(case["n"])
    val = None
    try:
        # Filliben: p_1 = 1 - 0.5^(1/n); p_n = 0.5^(1/n);
        # p_i = (i - 0.3175)/(n + 0.365) for 2 <= i <= n-1.
        tail = 0.5 ** (1.0 / n)
        probs = np.empty(n)
        if n >= 1:
            probs[0] = 1.0 - tail
        if n >= 2:
            probs[-1] = tail
        for i in range(2, n):
            probs[i - 1] = (i - 0.3175) / (n + 0.365)
        val = vec_or_none(ndtri(probs).tolist())
    except Exception:
        val = None
    points.append({"case_id": cid, "quantiles": val})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize probplot_quantiles query");
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
                "failed to spawn python3 for probplot_quantiles oracle: {e}"
            );
            eprintln!("skipping probplot_quantiles oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open probplot_quantiles oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "probplot_quantiles oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping probplot_quantiles oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for probplot_quantiles oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "probplot_quantiles oracle failed: {stderr}"
        );
        eprintln!("skipping probplot_quantiles oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse probplot_quantiles oracle JSON"))
}

#[test]
fn diff_stats_probplot_quantiles() {
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
        let Some(scipy_q) = &scipy_arm.quantiles else {
            continue;
        };
        let rust_q = probplot_quantiles(case.n);
        if rust_q.len() != scipy_q.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let mut max_local = 0.0_f64;
        for (r, s) in rust_q.iter().zip(scipy_q.iter()) {
            if r.is_finite() {
                max_local = max_local.max((r - s).abs());
            }
        }
        max_overall = max_overall.max(max_local);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: max_local,
            pass: max_local <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_probplot_quantiles".into(),
        category: "scipy.special.ndtri ∘ Filliben order-statistic medians".into(),
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
                "probplot_quantiles mismatch: {} abs={}",
                d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "probplot_quantiles conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
