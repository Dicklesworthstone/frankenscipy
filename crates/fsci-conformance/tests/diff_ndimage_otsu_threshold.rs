#![forbid(unsafe_code)]
//! Live parity for fsci_ndimage::otsu_threshold against a direct
//! Python implementation of the same 256-bin between-class-variance
//! maximization algorithm.
//!
//! Resolves [frankenscipy-4vtfv]. 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_ndimage::{NdArray, otsu_threshold};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    threshold: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create otsu diff dir");
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

fn synth(n: usize, seed: u64, low: f64, hi: f64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 11) as f64) / (1u64 << 53) as f64;
            low + u * (hi - low)
        })
        .collect()
}

fn bimodal(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 11) as f64) / (1u64 << 53) as f64;
            // Two Gaussian modes
            if i % 2 == 0 {
                30.0 + 5.0 * (u - 0.5)
            } else {
                180.0 + 5.0 * (u - 0.5)
            }
        })
        .collect()
}

fn generate_query() -> OracleQuery {
    let points = vec![
        Case {
            case_id: "uniform_64x64".into(),
            rows: 64,
            cols: 64,
            data: synth(64 * 64, 0xdead, 0.0, 255.0),
        },
        Case {
            case_id: "bimodal_32x32".into(),
            rows: 32,
            cols: 32,
            data: bimodal(32 * 32, 0xfeed),
        },
        Case {
            case_id: "narrow_range_16x16".into(),
            rows: 16,
            cols: 16,
            data: synth(16 * 16, 0xcafe, 50.0, 150.0),
        },
        Case {
            case_id: "small_image_8x8".into(),
            rows: 8,
            cols: 8,
            data: synth(8 * 8, 0xbeef, 0.0, 1.0),
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

def otsu(arr):
    arr = np.asarray(arr, dtype=float).flatten()
    if arr.size == 0:
        return 0.0
    mn = float(arr.min()); mx = float(arr.max())
    if abs(mx - mn) < 1e-15:
        return mn
    nbins = 256
    bw = (mx - mn) / nbins
    bins = ((arr - mn) / bw).astype(int)
    bins = np.clip(bins, 0, nbins - 1)
    hist = np.bincount(bins, minlength=nbins).astype(float)
    total = float(arr.size)
    sum_total = float(np.sum(np.arange(nbins) * hist))
    best_thresh = 0.0
    best_var = 0.0
    weight_bg = 0.0
    sum_bg = 0.0
    for i, c in enumerate(hist):
        weight_bg += c
        if weight_bg == 0.0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0.0:
            break
        sum_bg += i * c
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > best_var:
            best_var = between
            best_thresh = mn + (i + 0.5) * bw
    return best_thresh

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    arr = np.array(case["data"], dtype=float).reshape((int(case["rows"]), int(case["cols"])))
    try:
        v = float(otsu(arr))
        if math.isfinite(v):
            points.append({"case_id": cid, "threshold": v})
        else:
            points.append({"case_id": cid, "threshold": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "threshold": None})
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
                "failed to spawn python3 for otsu oracle: {e}"
            );
            eprintln!("skipping otsu oracle: python3 not available ({e})");
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
                "otsu oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping otsu oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for otsu oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "otsu oracle failed: {stderr}"
        );
        eprintln!("skipping otsu oracle: python not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse otsu oracle JSON"))
}

#[test]
fn diff_ndimage_otsu_threshold() {
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
        let Some(expected) = arm.threshold else {
            continue;
        };
        let Ok(arr) = NdArray::new(case.data.clone(), vec![case.rows, case.cols]) else {
            continue;
        };
        let actual = otsu_threshold(&arr);
        if !actual.is_finite() {
            continue;
        }
        let abs_d = (actual - expected).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_ndimage_otsu_threshold".into(),
        category: "fsci_ndimage::otsu_threshold vs Python reference".into(),
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
            eprintln!("otsu mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "otsu conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
