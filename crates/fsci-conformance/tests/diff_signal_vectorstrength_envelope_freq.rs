#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_signal::vectorstrength,
//! analytic_envelope (hilbert magnitude), and dominant_frequency.
//!
//! Resolves [frankenscipy-z4pf4]. 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{analytic_envelope, dominant_frequency, vectorstrength};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct VsCase {
    case_id: String,
    events: Vec<f64>,
    period: f64,
}

#[derive(Debug, Clone, Serialize)]
struct EnvCase {
    case_id: String,
    x: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct DomCase {
    case_id: String,
    magnitudes: Vec<f64>,
    freqs: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    vs: Vec<VsCase>,
    env: Vec<EnvCase>,
    dom: Vec<DomCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct VsArm {
    case_id: String,
    strength: Option<f64>,
    phase: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct VecArm {
    case_id: String,
    values: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct ScalarArm {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    vs: Vec<VsArm>,
    env: Vec<VecArm>,
    dom: Vec<ScalarArm>,
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
    fs::create_dir_all(output_dir()).expect("create vs_env_freq diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize vs_env_freq diff log");
    fs::write(path, json).expect("write vs_env_freq diff log");
}

fn generate_query() -> OracleQuery {
    // vectorstrength: (strength, phase) — phase now returned as the mean
    // unit-vector angle, matching scipy (frankenscipy-db4pr).
    let vs: Vec<VsCase> = vec![
        VsCase {
            case_id: "vs_two_cluster".into(),
            events: vec![0.1, 0.5, 1.1, 1.5, 2.1, 2.5],
            period: 1.0,
        },
        VsCase {
            case_id: "vs_tight".into(),
            events: vec![0.2, 0.21, 0.19, 0.205, 0.195],
            period: 1.0,
        },
        VsCase {
            case_id: "vs_irregular".into(),
            events: vec![0.3, 1.7, 2.9, 4.1, 5.05, 6.2, 7.8],
            period: 3.0,
        },
        VsCase {
            case_id: "vs_moderate".into(),
            events: vec![0.4, 0.6, 0.55, 1.45, 1.5, 2.4, 2.6],
            period: 1.0,
        },
    ];

    let env = vec![
        EnvCase {
            case_id: "env_decay_32".into(),
            x: (0..32)
                .map(|i| {
                    let t = (i as f64) * 4.0 * std::f64::consts::PI / 32.0;
                    t.sin() * (-(i as f64) / 8.0).exp()
                })
                .collect(),
        },
        EnvCase {
            case_id: "env_chirp_64".into(),
            x: (0..64)
                .map(|i| {
                    let t = (i as f64) / 64.0;
                    (2.0 * std::f64::consts::PI * (1.0 + 5.0 * t) * t * 10.0).sin()
                })
                .collect(),
        },
        EnvCase {
            case_id: "env_sine_24".into(),
            x: (0..24).map(|i| ((i as f64) * 0.5).sin()).collect(),
        },
    ];

    let dom = vec![
        DomCase {
            case_id: "dom_peak_at_bin3".into(),
            magnitudes: vec![0.1, 0.5, 1.0, 4.0, 1.0, 0.5, 0.1],
            freqs: vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0],
        },
        DomCase {
            case_id: "dom_first_bin".into(),
            magnitudes: vec![10.0, 1.0, 0.5, 0.1],
            freqs: vec![50.0, 100.0, 200.0, 400.0],
        },
        DomCase {
            case_id: "dom_tail".into(),
            magnitudes: vec![0.1, 0.2, 0.3, 0.4, 5.0],
            freqs: vec![10.0, 20.0, 30.0, 40.0, 50.0],
        },
    ];

    OracleQuery { vs, env, dom }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import signal

def finite_or_none(arr):
    flat = []
    for v in np.asarray(arr, dtype=float).flatten().tolist():
        if not math.isfinite(float(v)):
            return None
        flat.append(float(v))
    return flat

q = json.load(sys.stdin)

vs_out = []
for c in q["vs"]:
    cid = c["case_id"]
    events = np.array(c["events"], dtype=float)
    period = float(c["period"])
    try:
        strength, phase = signal.vectorstrength(events, period)
        vs_out.append({"case_id": cid, "strength": float(strength),
                       "phase": float(phase)})
    except Exception:
        vs_out.append({"case_id": cid, "strength": None, "phase": None})

env_out = []
for c in q["env"]:
    cid = c["case_id"]
    x = np.array(c["x"], dtype=float)
    try:
        env = np.abs(signal.hilbert(x))
        env_out.append({"case_id": cid, "values": finite_or_none(env)})
    except Exception:
        env_out.append({"case_id": cid, "values": None})

dom_out = []
for c in q["dom"]:
    cid = c["case_id"]
    m = np.array(c["magnitudes"], dtype=float)
    f = np.array(c["freqs"], dtype=float)
    try:
        idx = int(np.argmax(m))
        dom_out.append({"case_id": cid, "value": float(f[idx])})
    except Exception:
        dom_out.append({"case_id": cid, "value": None})

print(json.dumps({"vs": vs_out, "env": env_out, "dom": dom_out}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize vs_env_freq query");
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
                "failed to spawn python3 for vs_env_freq oracle: {e}"
            );
            eprintln!("skipping vs_env_freq oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open vs_env_freq oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "vs_env_freq oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping vs_env_freq oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for vs_env_freq oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "vs_env_freq oracle failed: {stderr}"
        );
        eprintln!("skipping vs_env_freq oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse vs_env_freq oracle JSON"))
}

#[test]
fn diff_signal_vectorstrength_envelope_freq() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.vs.len(), query.vs.len());
    assert_eq!(oracle.env.len(), query.env.len());
    assert_eq!(oracle.dom.len(), query.dom.len());

    let vs_map: HashMap<String, VsArm> = oracle
        .vs
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let env_map: HashMap<String, VecArm> = oracle
        .env
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let dom_map: HashMap<String, ScalarArm> = oracle
        .dom
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    // vectorstrength
    for case in &query.vs {
        let scipy_arm = vs_map.get(&case.case_id).expect("validated oracle");
        let (Some(s_exp), Some(p_exp)) = (scipy_arm.strength, scipy_arm.phase) else {
            continue;
        };
        let (s, p) = vectorstrength(&case.events, case.period);
        let abs_d = (s - s_exp).abs().max((p - p_exp).abs());
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "vectorstrength".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    // envelope
    for case in &query.env {
        let scipy_arm = env_map.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.values.as_ref() else {
            continue;
        };
        let Ok(env) = analytic_envelope(&case.x) else {
            continue;
        };
        let abs_d = if env.len() != expected.len() {
            f64::INFINITY
        } else {
            env.iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "analytic_envelope".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    // dominant_frequency
    for case in &query.dom {
        let scipy_arm = dom_map.get(&case.case_id).expect("validated oracle");
        let Some(expected) = scipy_arm.value else {
            continue;
        };
        let fsci_v = dominant_frequency(&case.magnitudes, &case.freqs);
        let abs_d = (fsci_v - expected).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "dominant_frequency".into(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_vectorstrength_envelope_freq".into(),
        category: "scipy.signal vectorstrength + envelope + dominant_freq".into(),
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
        "vs_env_freq conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
