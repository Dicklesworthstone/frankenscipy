#![forbid(unsafe_code)]
//! Live numpy parity for fsci_ndimage::array_histogram and
//! gradient_magnitude.
//!
//! Resolves [frankenscipy-trx9w].
//!
//! - `array_histogram(arr, bins)`: returns (counts, edges) over a
//!   uniform grid spanning [min, max]. fsci puts overflow into the
//!   last bin via `.min(bins-1)`; numpy.histogram with the same
//!   range does the same. Compare counts (Vec<usize>) and edges
//!   (Vec<f64>) at exact equality / 1e-12 abs.
//! - `gradient_magnitude(arr)`: central differences with Reflect
//!   boundary mode, then sqrt(sum_axes diff²). numpy.gradient uses
//!   central diffs in interior but forward/backward at edges, so
//!   we compare INTERIOR samples only. 1e-12 abs.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_ndimage::{NdArray, array_histogram, gradient_magnitude};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const ARMS: [&str; 2] = ["hist", "grad"];

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "hist" | "grad"
    rows: usize,
    cols: usize,
    data: Vec<f64>,
    bins: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// hist: counts + edges concatenated (counts first as f64, then edges)
    counts: Option<Vec<usize>>,
    edges: Option<Vec<f64>>,
    /// grad: interior values (row-major flatten of interior block)
    interior: Option<Vec<f64>>,
    interior_rows: Option<usize>,
    interior_cols: Option<usize>,
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
    fs::create_dir_all(output_dir()).expect("create hist_grad diff dir");
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

fn synth_image(rows: usize, cols: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    let mut out = Vec::with_capacity(rows * cols);
    for _ in 0..(rows * cols) {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((s >> 11) as f64) / (1u64 << 53) as f64;
        out.push((u - 0.5) * 8.0);
    }
    out
}

fn smooth_image(rows: usize, cols: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            let x = i as f64 / rows as f64;
            let y = j as f64 / cols as f64;
            out.push(
                (x * 2.0 * std::f64::consts::PI).sin() + (y * 2.0 * std::f64::consts::PI).cos(),
            );
        }
    }
    out
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    // histograms — 1D arrays of varying length
    let h_a = synth_image(1, 64, 0xdead);
    let h_b: Vec<f64> = (0..100).map(|i| (i as f64) * 0.5 - 25.0).collect();
    let h_c: Vec<f64> = (0..50).map(|i| (i as f64 * 0.3).sin() + 1.0).collect();

    for (label, data, bins) in [
        ("rand64_b10", &h_a, 10_usize),
        ("rand64_b20", &h_a, 20),
        ("uniform100_b5", &h_b, 5),
        ("sin50_b8", &h_c, 8),
    ] {
        points.push(Case {
            case_id: format!("hist_{label}"),
            op: "hist".into(),
            rows: 1,
            cols: data.len(),
            data: data.clone(),
            bins,
        });
    }

    // gradient_magnitude — 2D images
    let g_a = smooth_image(8, 8);
    let g_b = synth_image(10, 12, 0xfeed);
    for (label, rows, cols, data) in [
        ("smooth_8x8", 8_usize, 8_usize, &g_a),
        ("rand_10x12", 10_usize, 12_usize, &g_b),
    ] {
        points.push(Case {
            case_id: format!("grad_{label}"),
            op: "grad".into(),
            rows,
            cols,
            data: data.clone(),
            bins: 0,
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

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    rows = int(case["rows"]); cols = int(case["cols"])
    arr_flat = np.array(case["data"], dtype=float)
    try:
        if op == "hist":
            bins = int(case["bins"])
            # Mirror fsci: edges over [min, max] uniform.
            mn = float(arr_flat.min()); mx = float(arr_flat.max())
            counts, edges = np.histogram(arr_flat, bins=bins, range=(mn, mx))
            # numpy puts last edge at exactly mx and includes its values in the
            # last bin too (matches fsci's .min(bins-1) clamp).
            points.append({
                "case_id": cid,
                "counts": [int(c) for c in counts.tolist()],
                "edges": [float(e) for e in edges.tolist()],
                "interior": None,
                "interior_rows": None,
                "interior_cols": None,
            })
        elif op == "grad":
            arr = arr_flat.reshape((rows, cols))
            # numpy.gradient returns per-axis derivatives; central-diff interior
            gy, gx = np.gradient(arr)
            mag = np.hypot(gy, gx)
            # Interior block: drop first and last row/col
            interior = mag[1:-1, 1:-1]
            ir, ic = interior.shape
            flat = [float(v) for v in interior.flatten().tolist()]
            points.append({
                "case_id": cid,
                "counts": None,
                "edges": None,
                "interior": flat,
                "interior_rows": int(ir),
                "interior_cols": int(ic),
            })
        else:
            points.append({"case_id": cid, "counts": None, "edges": None, "interior": None,
                           "interior_rows": None, "interior_cols": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "counts": None, "edges": None, "interior": None,
                       "interior_rows": None, "interior_cols": None})
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
                "failed to spawn python3 for hist_grad oracle: {e}"
            );
            eprintln!("skipping hist_grad oracle: python3 not available ({e})");
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
                "hist_grad oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping hist_grad oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for hist_grad oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "hist_grad oracle failed: {stderr}"
        );
        eprintln!("skipping hist_grad oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse hist_grad oracle JSON"))
}

fn extract_interior(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    if rows < 3 || cols < 3 {
        return vec![];
    }
    let mut out = Vec::with_capacity((rows - 2) * (cols - 2));
    for r in 1..rows - 1 {
        for c in 1..cols - 1 {
            out.push(data[r * cols + c]);
        }
    }
    out
}

fn vec_max_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn diff_ndimage_array_histogram_gradient_magnitude() {
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
    let mut ledger = CompareLedger::new("diff_ndimage_array_histogram_gradient_magnitude", &ARMS);

    for case in &query.points {
        let arm = pmap.get(&case.case_id).expect("oracle row for every case");
        match case.op.as_str() {
            "hist" => {
                let shape = if case.rows == 1 {
                    vec![case.cols]
                } else {
                    vec![case.rows, case.cols]
                };
                let fsci_hist = NdArray::new(case.data.clone(), shape)
                    .ok()
                    .map(|arr| array_histogram(&arr, case.bins));
                let Some(((exp_counts, exp_edges), (counts, edges))) = ledger.both(
                    "hist",
                    &case.case_id,
                    arm.counts.as_ref().zip(arm.edges.as_ref()),
                    fsci_hist,
                ) else {
                    continue;
                };
                // Compare counts exactly
                let counts_diff = if counts.len() != exp_counts.len() {
                    f64::INFINITY
                } else {
                    counts
                        .iter()
                        .zip(exp_counts.iter())
                        .map(|(a, b)| ((*a as i64) - (*b as i64)).abs() as f64)
                        .fold(0.0_f64, f64::max)
                };
                let edges_diff = vec_max_diff(&edges, exp_edges);
                let abs_d = counts_diff.max(edges_diff);
                // vec_max_diff's max-fold swallows a NaN edge; a NaN edge must fail
                let pass = abs_d <= ABS_TOL && !edges.iter().any(|e| e.is_nan());
                max_overall = max_overall.max(abs_d);
                ledger.compared("hist", &case.case_id, pass);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    op: case.op.clone(),
                    abs_diff: abs_d,
                    pass,
                });
            }
            "grad" => {
                // The oracle reports the interior block's shape beside its values.
                let scipy_interior = match (
                    arm.interior.as_deref(),
                    arm.interior_rows,
                    arm.interior_cols,
                ) {
                    (Some(v), Some(ir), Some(ic)) if v.len() == ir * ic => Some(v),
                    _ => None,
                };
                let fsci_interior = NdArray::new(case.data.clone(), vec![case.rows, case.cols])
                    .ok()
                    .and_then(|arr| gradient_magnitude(&arr).ok())
                    .map(|g| extract_interior(&g.data, case.rows, case.cols));
                let Some((expected, interior)) = ledger.slices(
                    "grad",
                    &case.case_id,
                    scipy_interior,
                    fsci_interior.as_deref(),
                ) else {
                    continue;
                };
                let abs_d = vec_max_diff(interior, expected);
                max_overall = max_overall.max(abs_d);
                ledger.compared("grad", &case.case_id, abs_d <= ABS_TOL);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    op: case.op.clone(),
                    abs_diff: abs_d,
                    pass: abs_d <= ABS_TOL,
                });
            }
            other => panic!("unknown hist/grad op `{other}`"),
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_ndimage_array_histogram_gradient_magnitude".into(),
        category: "fsci_ndimage::{array_histogram, gradient_magnitude} vs numpy".into(),
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
            eprintln!("{} mismatch: {} abs_diff={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "hist/grad conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    // Arms have different case sets (grad has the fewest); each must compare all of its own.
    let min_per_arm = ARMS
        .iter()
        .map(|arm| query.points.iter().filter(|c| c.op == *arm).count())
        .min()
        .expect("ARMS is non-empty");
    ledger.finish(min_per_arm);
}
