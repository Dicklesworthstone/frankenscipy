#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.signal.resample` and default
//! SOS-based `scipy.signal.decimate`, including narrow-band decimation and
//! both axes of multi-channel inputs.
//!
//! Resolves [frankenscipy-t8e99]. fsci_signal::resample uses FFT
//! domain resampling. 1e-10 abs.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_signal::{decimate, decimate_axis_2d, resample};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    x: Vec<f64>,
    num: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    values: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create resample diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize resample diff log");
    fs::write(path, json).expect("write resample diff log");
}

fn generate_query() -> OracleQuery {
    let cases: &[(&str, Vec<f64>, usize)] = &[
        ("downsample_16_to_8", (0..16).map(|i| i as f64).collect(), 8),
        ("upsample_8_to_16", (0..8).map(|i| i as f64).collect(), 16),
        ("identity_16_to_16", (0..16).map(|i| i as f64).collect(), 16),
        (
            "sine_32_to_16",
            (0..32).map(|i| ((i as f64) * 0.3).sin()).collect(),
            16,
        ),
        (
            "sine_32_to_64",
            (0..32).map(|i| ((i as f64) * 0.3).sin()).collect(),
            64,
        ),
        (
            "decay_n12_to_24",
            (0..12).map(|i| (-(i as f64) / 3.0).exp()).collect(),
            24,
        ),
    ];
    let points = cases
        .iter()
        .map(|(name, x, n)| PointCase {
            case_id: (*name).into(),
            x: x.clone(),
            num: *n,
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
from scipy import signal

def finite_vec_or_none(arr):
    out = []
    for v in np.asarray(arr).tolist():
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
    cid = case["case_id"]
    x = np.array(case["x"], dtype=float)
    num = int(case["num"])
    try:
        y = signal.resample(x, num)
        points.append({"case_id": cid, "values": finite_vec_or_none(y)})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    run_scipy_or_skip(query, script)
}

fn run_scipy_or_skip<Q: Serialize, R: serde::de::DeserializeOwned>(
    query: &Q,
    script: &str,
) -> Option<R> {
    let query_json = serde_json::to_string(query).expect("serialize resample query");
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
                "failed to spawn python3 for resample oracle: {e}"
            );
            eprintln!("skipping resample oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open resample oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "resample oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping resample oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for resample oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "resample oracle failed: {stderr}"
        );
        eprintln!("skipping resample oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse resample oracle JSON"))
}

#[test]
fn diff_signal_resample() {
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
    let mut ledger = CompareLedger::new("diff_signal_resample", &["resample"]);

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let fsci_v = resample(&case.x, case.num).ok();
        let Some((scipy_v, fsci_v)) = ledger.slices(
            "resample",
            &case.case_id,
            scipy_arm.values.as_deref(),
            fsci_v.as_deref(),
        ) else {
            continue;
        };
        let abs_d = fsci_v
            .iter()
            .zip(scipy_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        max_overall = max_overall.max(abs_d);
        ledger.compared("resample", &case.case_id, abs_d <= ABS_TOL);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_resample".into(),
        category: "scipy.signal.resample".into(),
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
            eprintln!("resample mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "scipy.signal.resample conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    ledger.finish(query.points.len());
}

#[test]
fn diff_signal_decimate_sos() {
    #[derive(Serialize)]
    struct DecimateQuery {
        samples: Vec<Vec<f64>>,
        factors: Vec<usize>,
    }

    let x: Vec<f64> = (0..2000)
        .map(|i| (i as f64 * 0.003).sin() + 0.1 * (i as f64 * 0.6).cos())
        .collect();
    let shifted: Vec<f64> = x.iter().map(|&value| -2.0 * value + 0.25).collect();
    let query = DecimateQuery {
        samples: vec![x, shifted],
        factors: vec![2, 8, 13, 20, 50, 100],
    };
    // Send identical f64 inputs to both implementations, so this test measures
    // the filter and its boundary treatment rather than two libm sine curves.
    // Large q exposes the unstable degree-8 BA route; q=2 and q=8 also guard
    // ordinary decimation. An affine second channel exercises DC gain and the
    // axis-zero transpose independently of the first channel.
    let script = r#"
import json
import sys
import numpy as np
from scipy.signal import decimate

query = json.load(sys.stdin)
x = np.asarray(query["samples"], dtype=float)
points = []
for q in query["factors"]:
    for arm, data, axis in [("vector", x[0], -1), ("rows", x, -1),
                            ("columns", x.T, 0)]:
        values = decimate(data, q, axis=axis)
        assert np.all(np.isfinite(values)), (q, arm)
        points.append({"case_id": f"q={q}/{arm}",
                       "values": values.ravel().tolist()})
print(json.dumps({"points": points}, allow_nan=False))
"#;
    let Some(oracle): Option<OracleResult> = run_scipy_or_skip(&query, script) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.factors.len() * 3);
    let pmap: HashMap<_, _> = oracle
        .points
        .into_iter()
        .map(|point| (point.case_id.clone(), point))
        .collect();
    assert_eq!(pmap.len(), query.factors.len() * 3);
    let columns: Vec<Vec<f64>> = (0..query.samples[0].len())
        .map(|i| query.samples.iter().map(|row| row[i]).collect())
        .collect();
    let mut ledger = CompareLedger::new("diff_signal_decimate_sos", &["vector", "rows", "columns"]);
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;
    let start = Instant::now();
    for &q in &query.factors {
        let outputs = [
            ("vector", decimate(&query.samples[0], q)),
            (
                "rows",
                decimate_axis_2d(&query.samples, q, -1)
                    .map(|values| values.into_iter().flatten().collect()),
            ),
            (
                "columns",
                decimate_axis_2d(&columns, q, 0)
                    .map(|values| values.into_iter().flatten().collect()),
            ),
        ];
        for (arm, result) in outputs {
            let case_id = format!("q={q}/{arm}");
            let reference = pmap.get(&case_id).expect("oracle case exists");
            let values = result.ok();
            let Some((expected, actual)) = ledger.slices(
                arm,
                &case_id,
                reference.values.as_deref(),
                values.as_deref(),
            ) else {
                continue;
            };
            let abs_diff = actual
                .iter()
                .zip(expected)
                .map(|(&actual, &expected)| (actual - expected).abs())
                .fold(0.0_f64, f64::max);
            max_overall = max_overall.max(abs_diff);
            let pass = abs_diff <= ABS_TOL;
            ledger.compared(arm, &case_id, pass);
            eprintln!("decimate {case_id}: max_abs_error={abs_diff:.6e}");
            diffs.push(CaseDiff {
                case_id,
                abs_diff,
                pass,
            });
        }
    }
    emit_log(&DiffLog {
        test_id: "diff_signal_decimate_sos".into(),
        category: "scipy.signal.decimate".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        max_abs_diff: max_overall,
        pass: diffs.iter().all(|case| case.pass),
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs,
    });
    ledger.finish(query.factors.len());
}
