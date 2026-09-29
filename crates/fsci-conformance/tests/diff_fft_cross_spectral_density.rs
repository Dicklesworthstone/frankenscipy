#![forbid(unsafe_code)]
//! Formula-derived parity for fsci_fft::cross_spectral_density.
//!
//! Resolves [frankenscipy-vhigd]. CSD = X * conj(Y) / (n * fs).
//! 1e-10 abs.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_fft::cross_spectral_density;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-005";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    x: Vec<f64>,
    y: Vec<f64>,
    fs: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// Flattened (re, im) for first half (n/2 + 1) of CSD.
    csd_packed: Option<Vec<f64>>,
    freqs: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create csd diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize csd diff log");
    fs::write(path, json).expect("write csd diff log");
}

fn generate_query() -> OracleQuery {
    let n = 16;
    let x_sin: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.4).sin()).collect();
    let y_cos: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.4).cos()).collect();
    let n2 = 32;
    let x2: Vec<f64> = (0..n2).map(|i| 1.0 + ((i as f64) * 0.1).sin()).collect();
    let y2: Vec<f64> = (0..n2).map(|i| (-(i as f64) / 10.0).exp()).collect();

    OracleQuery {
        points: vec![
            PointCase {
                case_id: "n16_sin_cos_fs100".into(),
                x: x_sin.clone(),
                y: y_cos.clone(),
                fs: 100.0,
            },
            PointCase {
                case_id: "n16_self_fs50".into(),
                x: x_sin.clone(),
                y: x_sin,
                fs: 50.0,
            },
            PointCase {
                case_id: "n32_mixed_fs10".into(),
                x: x2,
                y: y2,
                fs: 10.0,
            },
        ],
    }
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
    cid = case["case_id"]
    x = np.array(case["x"], dtype=float)
    y = np.array(case["y"], dtype=float)
    fs = float(case["fs"])
    try:
        n = len(x)
        Fx = np.fft.fft(x)
        Fy = np.fft.fft(y)
        csd = Fx * np.conj(Fy) / (n * fs)
        n_freq = n // 2 + 1
        csd_half = csd[:n_freq]
        packed = []
        for c in csd_half.tolist():
            packed.append(float(c.real))
            packed.append(float(c.imag))
        freqs = [k * fs / n for k in range(n_freq)]
        # Filter NaN/inf
        if all(math.isfinite(v) for v in packed) and all(math.isfinite(v) for v in freqs):
            points.append({"case_id": cid, "csd_packed": packed, "freqs": freqs})
        else:
            points.append({"case_id": cid, "csd_packed": None, "freqs": None})
    except Exception:
        points.append({"case_id": cid, "csd_packed": None, "freqs": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize csd query");
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
                "failed to spawn python3 for csd oracle: {e}"
            );
            eprintln!("skipping csd oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open csd oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "csd oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping csd oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for csd oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "csd oracle failed: {stderr}"
        );
        eprintln!("skipping csd oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse csd oracle JSON"))
}

#[test]
fn diff_fft_cross_spectral_density() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;
    let mut ledger = CompareLedger::new("diff_fft_cross_spectral_density", &["csd", "freqs"]);

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let result = cross_spectral_density(&case.x, &case.y, case.fs).ok();
        let packed = result.as_ref().map(|(_, csd)| {
            let mut packed = Vec::with_capacity(csd.len() * 2);
            for &(re, im) in csd {
                packed.push(re);
                packed.push(im);
            }
            packed
        });
        let arms = [
            ("csd", scipy_arm.csd_packed.as_deref(), packed.as_deref()),
            (
                "freqs",
                scipy_arm.freqs.as_deref(),
                result.as_ref().map(|(freqs, _)| freqs.as_slice()),
            ),
        ];
        let mut arm_diffs = [f64::NAN; 2];
        for (slot, (arm, scipy, fsci)) in arm_diffs.iter_mut().zip(arms) {
            let Some((s, f)) = ledger.slices(arm, &case.case_id, scipy, fsci) else {
                continue;
            };
            let d = f
                .iter()
                .zip(s.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            ledger.compared(arm, &case.case_id, d <= ABS_TOL);
            *slot = d;
        }
        if arm_diffs.iter().any(|d| d.is_nan()) {
            continue; // the ledger recorded why this case was not compared
        }
        let [dc, df] = arm_diffs;
        let abs_d = dc.max(df);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_fft_cross_spectral_density".into(),
        category: "fsci_fft::cross_spectral_density".into(),
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
            eprintln!("csd mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "csd conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    ledger.finish(query.points.len());
}
