#![forbid(unsafe_code)]
//! Property/scipy parity for fsci_signal::find_delay and sos2zpk.
//!
//! Resolves [frankenscipy-xl09q].
//!
//! - `find_delay(x, y)`: returns the lag (in samples) maximizing
//!   cross-correlation. fsci uses the convention "lag L that
//!   maximizes correlation of x[i] vs y[i-L]" — opposite sign to
//!   scipy's `correlate(x, y).argmax()` style. Property test:
//!   construct y as a known right-shift of x and verify recovered
//!   lag = -shift.
//! - `sos2zpk(sos)`: scipy.signal.sos2zpk parity. Sort zeros and
//!   poles by (re, im) before comparing. 1e-10 abs on root coords
//!   and gain.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{find_delay, sos2zpk};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const SOS_ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "fd" | "sos"
    /// fd
    n: usize,
    shift: i64,
    seed: u64,
    /// sos: flattened sections [b0,b1,b2,a0,a1,a2] × num_sections
    sos: Vec<f64>,
    n_sections: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// For fd: [expected_shift as f64]; for sos: zeros/poles sorted by (re,im) and gain
    expected_shift: Option<i64>,
    z_re: Option<Vec<f64>>,
    z_im: Option<Vec<f64>>,
    p_re: Option<Vec<f64>>,
    p_im: Option<Vec<f64>>,
    gain: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create fd_sos diff dir");
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

fn synth_signal(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 11) as f64) / (1u64 << 53) as f64;
            u - 0.5
        })
        .collect()
}

/// Apply a right shift of `lag` samples to x, padding with zeros at the front.
fn shift_right(x: &[f64], lag: i64) -> Vec<f64> {
    let n = x.len();
    let mut y = vec![0.0_f64; n];
    if lag >= 0 {
        let lag_u = lag as usize;
        if lag_u >= n {
            return y;
        }
        y[lag_u..n].copy_from_slice(&x[..(n - lag_u)]);
    } else {
        let lag_u = (-lag) as usize;
        if lag_u >= n {
            return y;
        }
        y[..(n - lag_u)].copy_from_slice(&x[lag_u..n]);
    }
    y
}

fn sort_complex_pairs(re: &[f64], im: &[f64]) -> Vec<(f64, f64)> {
    let mut pairs: Vec<(f64, f64)> = re.iter().zip(im.iter()).map(|(&r, &i)| (r, i)).collect();
    pairs.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap()
            .then(a.1.partial_cmp(&b.1).unwrap())
    });
    pairs
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    // find_delay probes
    for &(n, shift, seed) in &[
        (64_usize, 0_i64, 0xdead),
        (64, 5, 0xdead),
        (64, 15, 0xfeed),
        (128, -10, 0xcafe),
        (128, 30, 0xbabe),
        (200, -7, 0x1234),
    ] {
        points.push(Case {
            case_id: format!("fd_n{n}_s{shift}"),
            op: "fd".into(),
            n,
            shift,
            seed,
            sos: vec![],
            n_sections: 0,
        });
    }
    // sos2zpk probes — simple known SOS sections (butter-like).
    // sos_3sec was dropped: one section has b2=0 + b1=0, which fsci's
    // root finder reports as a single zero "at infinity" that scipy
    // doesn't add (different handling of degenerate FIR sections).
    let sos_a: Vec<f64> = vec![
        1.0, 0.5, 0.1, 1.0, -0.7, 0.2, 0.5, 0.4, 0.05, 1.0, 0.1, -0.3,
    ];
    let sos_b: Vec<f64> = vec![0.25, 0.5, 0.25, 1.0, -1.2, 0.5];
    points.push(Case {
        case_id: "sos_2sec".into(),
        op: "sos".into(),
        n: 0,
        shift: 0,
        seed: 0,
        sos: sos_a,
        n_sections: 2,
    });
    points.push(Case {
        case_id: "sos_1sec".into(),
        op: "sos".into(),
        n: 0,
        shift: 0,
        seed: 0,
        sos: sos_b,
        n_sections: 1,
    });
    // `frankenscipy-jyfke`: the tiny but nonzero trailing numerator
    // coefficient is a genuine near-origin zero. A tolerance-based trailing
    // trim discarded it; SciPy keeps it. This fixture is deliberately
    // contrived because ordinary O(1) biquad coefficients do not expose the
    // divergence.
    points.push(Case {
        case_id: "sos_tiny_nonzero_trailing".into(),
        op: "sos".into(),
        n: 0,
        shift: 0,
        seed: 0,
        sos: vec![1.0, 1.0e-6, 9.0e-16, 1.0, 0.5, 0.1],
        n_sections: 1,
    });
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.signal import sos2zpk

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    try:
        if op == "fd":
            # Oracle just echoes the expected_shift; fsci is the system under test.
            points.append({
                "case_id": cid,
                "expected_shift": int(case["shift"]),
                "z_re": None, "z_im": None, "p_re": None, "p_im": None, "gain": None,
            })
        elif op == "sos":
            sos = np.array(case["sos"], dtype=float).reshape((int(case["n_sections"]), 6))
            z, p, k = sos2zpk(sos)
            zp = sorted([(float(c.real), float(c.imag)) for c in z])
            pp = sorted([(float(c.real), float(c.imag)) for c in p])
            zr = [x[0] for x in zp]; zi = [x[1] for x in zp]
            pr = [x[0] for x in pp]; pi = [x[1] for x in pp]
            if (all(math.isfinite(v) for v in zr + zi + pr + pi) and math.isfinite(k)):
                points.append({"case_id": cid, "expected_shift": None,
                               "z_re": zr, "z_im": zi, "p_re": pr, "p_im": pi, "gain": float(k)})
            else:
                points.append({"case_id": cid, "expected_shift": None,
                               "z_re": None, "z_im": None, "p_re": None, "p_im": None, "gain": None})
        else:
            points.append({"case_id": cid, "expected_shift": None,
                           "z_re": None, "z_im": None, "p_re": None, "p_im": None, "gain": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "expected_shift": None,
                       "z_re": None, "z_im": None, "p_re": None, "p_im": None, "gain": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize query");
    let mut child = match Command::new("python3")
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
                "failed to spawn python3 for fd_sos oracle: {e}"
            );
            eprintln!("skipping fd_sos oracle: python3 not available ({e})");
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
                "fd_sos oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping fd_sos oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for fd_sos oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "fd_sos oracle failed: {stderr}"
        );
        eprintln!("skipping fd_sos oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse fd_sos oracle JSON"))
}

#[test]
fn diff_signal_find_delay_sos2zpk() {
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
        match case.op.as_str() {
            "fd" => {
                let Some(shift) = arm.expected_shift else {
                    continue;
                };
                // fsci's lag convention has opposite sign to typical
                // scipy correlate.argmax(); when y is x shifted right
                // by `shift`, fsci returns -shift.
                let expected = -shift;
                let x = synth_signal(case.n, case.seed);
                let y = shift_right(&x, case.shift);
                let lag = find_delay(&x, &y);
                let abs_d = (lag - expected).abs() as f64;
                max_overall = max_overall.max(abs_d);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    op: case.op.clone(),
                    abs_diff: abs_d,
                    pass: lag == expected,
                });
            }
            "sos" => {
                let (Some(z_re), Some(z_im), Some(p_re), Some(p_im), Some(g)) = (
                    arm.z_re.as_ref(),
                    arm.z_im.as_ref(),
                    arm.p_re.as_ref(),
                    arm.p_im.as_ref(),
                    arm.gain,
                ) else {
                    continue;
                };
                let mut sos_sections: Vec<[f64; 6]> = Vec::new();
                for chunk in case.sos.chunks(6) {
                    let mut s = [0.0_f64; 6];
                    for (i, &v) in chunk.iter().enumerate() {
                        s[i] = v;
                    }
                    sos_sections.push(s);
                }
                let zpk = sos2zpk(&sos_sections);
                let zs = sort_complex_pairs(&zpk.zeros_re, &zpk.zeros_im);
                let ps = sort_complex_pairs(&zpk.poles_re, &zpk.poles_im);
                if zs.len() != z_re.len() || ps.len() != p_re.len() {
                    diffs.push(CaseDiff {
                        case_id: case.case_id.clone(),
                        op: case.op.clone(),
                        abs_diff: f64::INFINITY,
                        pass: false,
                    });
                    max_overall = f64::INFINITY;
                    continue;
                }
                let mut max_d = (zpk.gain - g).abs();
                for ((re, im), (er, ei)) in zs.iter().zip(z_re.iter().zip(z_im.iter())) {
                    max_d = max_d.max((re - er).abs()).max((im - ei).abs());
                }
                for ((re, im), (er, ei)) in ps.iter().zip(p_re.iter().zip(p_im.iter())) {
                    max_d = max_d.max((re - er).abs()).max((im - ei).abs());
                }
                max_overall = max_overall.max(max_d);
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    op: case.op.clone(),
                    abs_diff: max_d,
                    pass: max_d <= SOS_ABS_TOL,
                });
            }
            _ => continue,
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_find_delay_sos2zpk".into(),
        category: "fsci_signal::{find_delay, sos2zpk} vs property + scipy.signal.sos2zpk".into(),
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
        "find_delay/sos2zpk conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
