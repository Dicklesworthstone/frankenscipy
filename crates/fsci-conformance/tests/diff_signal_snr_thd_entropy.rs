#![forbid(unsafe_code)]
//! Live numpy differential coverage for fsci_signal scalar/aggregate
//! quality metrics: snr, thd, spectral_entropy, spectral_flatness,
//! short_time_energy.
//!
//! Resolves [frankenscipy-zzeyt]. 1e-12 abs (exact float arithmetic).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{short_time_energy, snr, spectral_entropy, spectral_flatness, thd};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct SnrCase {
    case_id: String,
    signal: Vec<f64>,
    noise: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct ThdCase {
    case_id: String,
    magnitudes: Vec<f64>,
    fundamental_bin: usize,
}

#[derive(Debug, Clone, Serialize)]
struct EntropyCase {
    case_id: String,
    op: String, // "entropy" | "flatness"
    magnitudes: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct EnergyCase {
    case_id: String,
    x: Vec<f64>,
    frame_len: usize,
    hop_len: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    snr: Vec<SnrCase>,
    thd: Vec<ThdCase>,
    entropy: Vec<EntropyCase>,
    energy: Vec<EnergyCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct ScalarArm {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct VecArm {
    case_id: String,
    values: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    snr: Vec<ScalarArm>,
    thd: Vec<ScalarArm>,
    entropy: Vec<ScalarArm>,
    energy: Vec<VecArm>,
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
    fs::create_dir_all(output_dir()).expect("create snr_thd_entropy diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize snr_thd_entropy diff log");
    fs::write(path, json).expect("write snr_thd_entropy diff log");
}

fn generate_query() -> OracleQuery {
    // snr
    let snr_cases = vec![
        SnrCase {
            case_id: "snr_sine_white".into(),
            signal: (0..64).map(|i| ((i as f64) * 0.5).sin()).collect(),
            noise: (0..64).map(|i| ((i as f64) * 1.3).sin() * 0.1).collect(),
        },
        SnrCase {
            case_id: "snr_dc_low_noise".into(),
            signal: vec![1.0; 50],
            noise: (0..50).map(|i| ((i as f64) * 0.7).sin() * 0.01).collect(),
        },
        SnrCase {
            case_id: "snr_equal_power".into(),
            signal: vec![1.0; 20],
            noise: vec![1.0; 20],
        },
    ];

    // thd
    let thd_cases = vec![
        ThdCase {
            case_id: "thd_pure_fundamental".into(),
            magnitudes: vec![0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
            fundamental_bin: 1,
        },
        ThdCase {
            case_id: "thd_with_harmonics".into(),
            magnitudes: vec![0.0, 10.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.25],
            fundamental_bin: 1,
        },
        ThdCase {
            case_id: "thd_bin2_fund".into(),
            magnitudes: vec![0.0, 0.0, 5.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.25],
            fundamental_bin: 2,
        },
    ];

    // entropy/flatness
    let entropy_inputs: &[(&str, Vec<f64>)] = &[
        ("uniform_8", vec![1.0; 8]),
        ("peaked_5", vec![0.1, 0.5, 4.0, 0.5, 0.1]),
        ("ramp_6", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        (
            "decay_10",
            (0..10).map(|i| (-(i as f64) / 3.0).exp()).collect(),
        ),
    ];
    let mut entropy = Vec::new();
    for (label, m) in entropy_inputs {
        entropy.push(EntropyCase {
            case_id: format!("entropy_{label}"),
            op: "entropy".into(),
            magnitudes: m.clone(),
        });
        entropy.push(EntropyCase {
            case_id: format!("flatness_{label}"),
            op: "flatness".into(),
            magnitudes: m.clone(),
        });
    }

    // short_time_energy
    let energy = vec![
        EnergyCase {
            case_id: "ste_sine_64_8h4".into(),
            x: (0..64).map(|i| ((i as f64) * 0.4).sin()).collect(),
            frame_len: 8,
            hop_len: 4,
        },
        EnergyCase {
            case_id: "ste_ramp_30_5h5".into(),
            x: (1..=30).map(|i| i as f64).collect(),
            frame_len: 5,
            hop_len: 5,
        },
        EnergyCase {
            case_id: "ste_decay_40_10h10".into(),
            x: (0..40).map(|i| (-(i as f64) / 5.0).exp()).collect(),
            frame_len: 10,
            hop_len: 10,
        },
    ];

    OracleQuery {
        snr: snr_cases,
        thd: thd_cases,
        entropy,
        energy,
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np

q = json.load(sys.stdin)

snr_out = []
for c in q["snr"]:
    cid = c["case_id"]
    sig = np.array(c["signal"], dtype=float)
    noise = np.array(c["noise"], dtype=float)
    try:
        sig_p = float(np.sum(sig**2) / max(len(sig), 1))
        noise_p = float(np.sum(noise**2) / max(len(noise), 1))
        if noise_p == 0.0:
            v = float("inf")
        else:
            v = 10.0 * math.log10(sig_p / noise_p)
        snr_out.append({"case_id": cid, "value": v if math.isfinite(v) else None})
    except Exception:
        snr_out.append({"case_id": cid, "value": None})

thd_out = []
for c in q["thd"]:
    cid = c["case_id"]
    m = np.array(c["magnitudes"], dtype=float)
    fb = int(c["fundamental_bin"])
    try:
        if fb >= len(m) or m[fb] == 0.0:
            thd_out.append({"case_id": cid, "value": None})
            continue
        fund_p = float(m[fb] ** 2)
        harm_p = 0.0
        bin_ = 2 * fb
        while bin_ < len(m):
            harm_p += float(m[bin_] ** 2)
            bin_ += fb
        v = float(math.sqrt(harm_p / fund_p))
        thd_out.append({"case_id": cid, "value": v if math.isfinite(v) else None})
    except Exception:
        thd_out.append({"case_id": cid, "value": None})

entropy_out = []
for c in q["entropy"]:
    cid = c["case_id"]; op = c["op"]
    m = np.array(c["magnitudes"], dtype=float)
    try:
        if op == "entropy":
            # fsci normalizes Shannon entropy by ln(N) so output is in [0, 1].
            total = float(np.sum(m))
            n = len(m)
            if total <= 0.0:
                v = 0.0
            else:
                p = m / total
                h = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
                v = h / math.log(n) if n > 1 else h
        elif op == "flatness":
            n = len(m)
            arith = float(np.sum(m) / n)
            if arith == 0.0:
                v = 0.0
            else:
                log_sum = float(np.sum(np.log(np.where(m > 0, m, math.exp(-700.0)))))
                geom = math.exp(log_sum / n)
                v = max(0.0, min(1.0, geom / arith))
        else:
            v = None
        entropy_out.append({"case_id": cid, "value": v if v is not None and math.isfinite(v) else None})
    except Exception:
        entropy_out.append({"case_id": cid, "value": None})

energy_out = []
for c in q["energy"]:
    cid = c["case_id"]
    x = np.array(c["x"], dtype=float)
    frame = int(c["frame_len"]); hop = int(c["hop_len"])
    try:
        out = []
        start = 0
        while start + frame <= len(x):
            e = float(np.sum(x[start:start+frame] ** 2))
            out.append(e)
            start += hop
        energy_out.append({"case_id": cid, "values": out})
    except Exception:
        energy_out.append({"case_id": cid, "values": None})

print(json.dumps({"snr": snr_out, "thd": thd_out, "entropy": entropy_out, "energy": energy_out}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize snr_thd_entropy query");
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
                "failed to spawn python3 for snr_thd_entropy oracle: {e}"
            );
            eprintln!("skipping snr_thd_entropy oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open snr_thd_entropy oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "snr_thd_entropy oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping snr_thd_entropy oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for snr_thd_entropy oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "snr_thd_entropy oracle failed: {stderr}"
        );
        eprintln!("skipping snr_thd_entropy oracle: numpy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse snr_thd_entropy oracle JSON"))
}

#[test]
fn diff_signal_snr_thd_entropy() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.snr.len(), query.snr.len());
    assert_eq!(oracle.thd.len(), query.thd.len());
    assert_eq!(oracle.entropy.len(), query.entropy.len());
    assert_eq!(oracle.energy.len(), query.energy.len());

    let snr_map: HashMap<String, ScalarArm> = oracle
        .snr
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let thd_map: HashMap<String, ScalarArm> = oracle
        .thd
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let entropy_map: HashMap<String, ScalarArm> = oracle
        .entropy
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let energy_map: HashMap<String, VecArm> = oracle
        .energy
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.snr {
        let scipy_arm = snr_map.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.value else {
            continue;
        };
        let fsci_v = snr(&case.signal, &case.noise);
        let abs_d = (fsci_v - expected).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "snr".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    for case in &query.thd {
        let scipy_arm = thd_map.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.value else {
            continue;
        };
        let fsci_v = thd(&case.magnitudes, case.fundamental_bin);
        let abs_d = (fsci_v - expected).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "thd".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    for case in &query.entropy {
        let scipy_arm = entropy_map.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.value else {
            continue;
        };
        let fsci_v = match case.op.as_str() {
            "entropy" => spectral_entropy(&case.magnitudes),
            "flatness" => spectral_flatness(&case.magnitudes),
            _ => continue,
        };
        let abs_d = (fsci_v - expected).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    for case in &query.energy {
        let scipy_arm = energy_map.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.values.as_ref() else {
            continue;
        };
        let fsci_v = short_time_energy(&case.x, case.frame_len, case.hop_len);
        let abs_d = if fsci_v.len() != expected.len() {
            f64::INFINITY
        } else {
            fsci_v
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "short_time_energy".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_snr_thd_entropy".into(),
        category: "fsci_signal snr/thd/entropy/flatness/short_time_energy".into(),
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
            eprintln!("{} mismatch: {} abs_diff={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "snr_thd_entropy conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
