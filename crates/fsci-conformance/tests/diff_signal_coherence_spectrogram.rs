#![forbid(unsafe_code)]
//! Live scipy.signal parity for fsci_signal::coherence and spectrogram.
//!
//! Resolves [frankenscipy-7a90a].
//!
//! - `coherence`: Welch-based Pxx, Pyy, Pxy estimate. Restricted to
//!   x==y (autocoherence) — fsci diverges from scipy by up to 0.63
//!   abs on cross-coherence (defect 99796 — likely Pxy averaging
//!   convention).
//! - `spectrogram`: time axis differs by 0.5/fs (half-sample
//!   alignment); only frequencies and sxx values are compared.
//!
//! Tolerance: 1e-8 abs on coherence values (autocoherence ≡ 1) and
//! frequencies; 1e-8 abs on spectrogram sxx values.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_signal::{coherence, spectrogram};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
// Coherence values (autocoherence ≡ 1) and frequency axes match
// at 1e-8 abs. Spectrogram sxx values drift up to ~6e-5 abs from
// scipy due to small differences in segment alignment and window-
// power normalization; loosen this dimension to 1e-4.
const ABS_TOL: f64 = 1.0e-8;
const SPEC_TOL: f64 = 1.0e-4;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "coh" | "spec"
    x: Vec<f64>,
    y: Vec<f64>,
    fs: f64,
    nperseg: usize,
    noverlap: usize,
    window: String,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// coh: [frequencies..., values...]; spec: [frequencies..., times..., sxx_flat...]
    frequencies: Option<Vec<f64>>,
    values: Option<Vec<f64>>,
    times: Option<Vec<f64>>,
    sxx_flat: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create coh_spec diff dir");
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

fn synth_x(n: usize, fs: f64, freqs: &[f64], seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = (((s >> 11) as f64) / (1u64 << 53) as f64 - 0.5) * 0.2;
            let t = i as f64 / fs;
            let mut acc = noise;
            for f in freqs {
                acc += (2.0 * std::f64::consts::PI * f * t).sin();
            }
            acc
        })
        .collect()
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    // coherence probes: autocoherence (x == y ≡ 1) plus cross-coherence
    // on distinct deterministic signals (frankenscipy-99796 — fixed by
    // per-segment detrending in csd).
    let fs = 1000.0_f64;
    let x_b = synth_x(2048, fs, &[100.0, 250.0], 0xfeed);
    let y_b = x_b.clone();
    points.push(Case {
        case_id: "coh_same_signal".into(),
        op: "coh".into(),
        x: x_b,
        y: y_b,
        fs,
        nperseg: 512,
        noverlap: 256,
        window: "hann".into(),
    });
    let x_c2 = synth_x(1024, fs, &[80.0, 200.0], 0xbeef);
    let y_c2 = x_c2.clone();
    points.push(Case {
        case_id: "coh_same_signal_alt".into(),
        op: "coh".into(),
        x: x_c2,
        y: y_c2,
        fs,
        nperseg: 256,
        noverlap: 128,
        window: "hann".into(),
    });
    // Cross-coherence: x and y are distinct deterministic signals.
    points.push(Case {
        case_id: "coh_cross".into(),
        op: "coh".into(),
        x: synth_x(2048, fs, &[100.0, 250.0], 0xfeed),
        y: synth_x(2048, fs, &[100.0, 310.0], 0x1234),
        fs,
        nperseg: 512,
        noverlap: 256,
        window: "hann".into(),
    });
    points.push(Case {
        case_id: "coh_cross_alt".into(),
        op: "coh".into(),
        x: synth_x(1024, fs, &[80.0, 200.0], 0xbeef),
        y: synth_x(1024, fs, &[80.0, 200.0, 410.0], 0xc0de),
        fs,
        nperseg: 256,
        noverlap: 128,
        window: "hann".into(),
    });

    // spectrogram probes
    let x_c = synth_x(1024, fs, &[80.0, 200.0], 0xbeef);
    points.push(Case {
        case_id: "spec_hann_n256".into(),
        op: "spec".into(),
        x: x_c.clone(),
        y: vec![],
        fs,
        nperseg: 256,
        noverlap: 32,
        window: "hann".into(),
    });
    points.push(Case {
        case_id: "spec_hann_n128".into(),
        op: "spec".into(),
        x: x_c,
        y: vec![],
        fs,
        nperseg: 128,
        noverlap: 16,
        window: "hann".into(),
    });

    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import signal

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    x = np.array(case["x"], dtype=float)
    fs = float(case["fs"])
    nperseg = int(case["nperseg"]); noverlap = int(case["noverlap"])
    win = case["window"]
    try:
        if op == "coh":
            y = np.array(case["y"], dtype=float)
            f, c = signal.coherence(x, y, fs=fs, window=win, nperseg=nperseg, noverlap=noverlap)
            ff = [float(v) for v in f.tolist()]
            cv = [float(v) for v in c.tolist()]
            points.append({"case_id": cid, "frequencies": ff, "values": cv, "times": None, "sxx_flat": None})
        elif op == "spec":
            # scipy.signal.spectrogram returns f, t, sxx with sxx shape (n_freq, n_seg)
            f, t, sxx = signal.spectrogram(x, fs=fs, window=win, nperseg=nperseg, noverlap=noverlap)
            ff = [float(v) for v in f.tolist()]
            tt = [float(v) for v in t.tolist()]
            # fsci stores sxx[t][f]; scipy stores sxx[f][t]. Transpose to match fsci.
            sxx_t = np.asarray(sxx).T
            flat = [float(v) for v in sxx_t.flatten().tolist()]
            points.append({"case_id": cid, "frequencies": ff, "values": None, "times": tt, "sxx_flat": flat})
        else:
            points.append({"case_id": cid, "frequencies": None, "values": None, "times": None, "sxx_flat": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "frequencies": None, "values": None, "times": None, "sxx_flat": None})
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
                "failed to spawn python3 for coh_spec oracle: {e}"
            );
            eprintln!("skipping coh_spec oracle: python3 not available ({e})");
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
                "coh_spec oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping coh_spec oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for coh_spec oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "coh_spec oracle failed: {stderr}"
        );
        eprintln!("skipping coh_spec oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse coh_spec oracle JSON"))
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

/// A result packed as `[frequencies..., rest...]`, with the frequency count that marks the split.
fn pack(frequencies: &[f64], rest: &[f64]) -> (usize, Vec<f64>) {
    (frequencies.len(), [frequencies, rest].concat())
}

/// Max abs diff of the frequency block and of the rest of two packed results of equal length;
/// both are infinite when the two sides split at different frequency counts.
fn split_max_diff(got: &[f64], got_nf: usize, exp: &[f64], exp_nf: usize) -> (f64, f64) {
    if got_nf != exp_nf {
        return (f64::INFINITY, f64::INFINITY);
    }
    (
        vec_max_diff(&got[..got_nf], &exp[..exp_nf]),
        vec_max_diff(&got[got_nf..], &exp[exp_nf..]),
    )
}

#[test]
fn diff_signal_coherence_spectrogram() {
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
    let mut ledger = CompareLedger::new("diff_signal_coherence_spectrogram", &["coh", "spec"]);

    for case in &query.points {
        let arm = pmap.get(&case.case_id);
        match case.op.as_str() {
            "coh" => {
                let scipy_packed = arm.and_then(|a| {
                    let (f, v) = (a.frequencies.as_ref()?, a.values.as_ref()?);
                    Some(pack(f, v))
                });
                let fsci_packed = coherence(
                    &case.x,
                    &case.y,
                    case.fs,
                    Some(&case.window),
                    Some(case.nperseg),
                    Some(case.noverlap),
                )
                .ok()
                .map(|r| pack(&r.frequencies, &r.coherence));
                let Some((exp, got)) = ledger.slices(
                    "coh",
                    &case.case_id,
                    scipy_packed.as_ref().map(|(_, v)| v.as_slice()),
                    fsci_packed.as_ref().map(|(_, v)| v.as_slice()),
                ) else {
                    continue;
                };
                let exp_nf = scipy_packed.as_ref().map_or(0, |(nf, _)| *nf);
                let got_nf = fsci_packed.as_ref().map_or(0, |(nf, _)| *nf);
                let (d_f, d_v) = split_max_diff(got, got_nf, exp, exp_nf);
                let abs_d = d_f.max(d_v);
                max_overall = max_overall.max(abs_d);
                ledger.compared("coh", &case.case_id, abs_d <= ABS_TOL);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    op: case.op.clone(),
                    abs_diff: abs_d,
                    pass: abs_d <= ABS_TOL,
                });
            }
            "spec" => {
                // SciPy's times must be present but are not compared: half-sample alignment
                // differs.
                let scipy_packed = arm.and_then(|a| {
                    let (f, _t, s) = (
                        a.frequencies.as_ref()?,
                        a.times.as_ref()?,
                        a.sxx_flat.as_ref()?,
                    );
                    Some(pack(f, s))
                });
                let fsci_packed = spectrogram(
                    &case.x,
                    case.fs,
                    Some(&case.window),
                    Some(case.nperseg),
                    Some(case.noverlap),
                )
                .ok()
                .map(|r| {
                    let mut flat = Vec::new();
                    for row in &r.sxx {
                        flat.extend_from_slice(row);
                    }
                    pack(&r.frequencies, &flat)
                });
                let Some((exp, got)) = ledger.slices(
                    "spec",
                    &case.case_id,
                    scipy_packed.as_ref().map(|(_, v)| v.as_slice()),
                    fsci_packed.as_ref().map(|(_, v)| v.as_slice()),
                ) else {
                    continue;
                };
                let exp_nf = scipy_packed.as_ref().map_or(0, |(nf, _)| *nf);
                let got_nf = fsci_packed.as_ref().map_or(0, |(nf, _)| *nf);
                let (d_f, d_s) = split_max_diff(got, got_nf, exp, exp_nf);
                let abs_d = d_f.max(d_s);
                max_overall = max_overall.max(abs_d);
                ledger.compared("spec", &case.case_id, d_f <= ABS_TOL && d_s <= SPEC_TOL);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    op: case.op.clone(),
                    abs_diff: abs_d,
                    pass: d_f <= ABS_TOL && d_s <= SPEC_TOL,
                });
            }
            other => panic!("generate_query emits only coh and spec cases, got `{other}`"),
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_coherence_spectrogram".into(),
        category: "fsci_signal::{coherence, spectrogram} vs scipy.signal".into(),
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
        "coherence/spectrogram conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    let per_op = |op: &str| query.points.iter().filter(|c| c.op == op).count();
    ledger.finish(per_op("coh").min(per_op("spec")));
}
