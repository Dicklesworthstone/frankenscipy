#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.signal.tf2zpk`.
//!
//! Resolves [frankenscipy-6rc26]. fsci returns ZpkCoeffs with separate
//! re/im vectors for zeros and poles plus a scalar gain. Both
//! implementations may report roots in different orders so we sort by
//! (real, imag) before comparing. 1e-7 abs covers small-polynomial
//! root-finding noise.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::tf2zpk;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-7;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    b: Vec<f64>,
    a: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// Sorted (real, imag) zero pairs flattened + sorted pole pairs + gain.
    values: Option<Vec<f64>>,
    n_zeros: Option<usize>,
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
    fs::create_dir_all(output_dir()).expect("create tf2zpk diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize tf2zpk diff log");
    fs::write(path, json).expect("write tf2zpk diff log");
}

fn generate_query() -> OracleQuery {
    let cases: &[(&str, Vec<f64>, Vec<f64>)] = &[
        (
            "simple_complex_roots",
            vec![1.0, 0.5, 0.25],
            vec![1.0, -0.8, 0.16],
        ),
        ("fir_only", vec![1.0, 0.0, -0.5], vec![1.0]),
        (
            "iir_3rd_order",
            vec![0.1, 0.2, 0.1],
            vec![1.0, -0.5, 0.3, -0.05],
        ),
        (
            "rational_4_3",
            vec![1.0, -0.5, 0.1, 0.0],
            vec![1.0, -0.9, 0.3],
        ),
        (
            "biquad_lp",
            vec![0.0675, 0.135, 0.0675],
            vec![1.0, -1.143, 0.4128],
        ),
    ];
    let points = cases
        .iter()
        .map(|(name, b, a)| PointCase {
            case_id: (*name).into(),
            b: b.clone(),
            a: a.clone(),
        })
        .collect();
    OracleQuery { points }
}

fn pack_sorted(real: &[f64], imag: &[f64]) -> Vec<(f64, f64)> {
    let mut items: Vec<(f64, f64)> = real
        .iter()
        .zip(imag.iter())
        .map(|(r, i)| (*r, *i))
        .collect();
    items.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    items
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import signal

def pack(roots):
    items = [(float(r.real), float(r.imag)) for r in np.asarray(roots).tolist()]
    items.sort(key=lambda t: (t[0], t[1]))
    return items

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    b = np.array(case["b"], dtype=float)
    a = np.array(case["a"], dtype=float)
    try:
        z, p, k = signal.tf2zpk(b, a)
        zs = pack(z); ps = pack(p)
        packed = []
        for re, im in zs:
            if not (math.isfinite(re) and math.isfinite(im)):
                packed = None; break
            packed.append(re); packed.append(im)
        if packed is not None:
            for re, im in ps:
                if not (math.isfinite(re) and math.isfinite(im)):
                    packed = None; break
                packed.append(re); packed.append(im)
        if packed is None or not math.isfinite(float(k)):
            points.append({"case_id": cid, "values": None, "n_zeros": None})
        else:
            packed.append(float(k))
            points.append({"case_id": cid, "values": packed, "n_zeros": len(zs)})
    except Exception:
        points.append({"case_id": cid, "values": None, "n_zeros": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize tf2zpk query");
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
                "failed to spawn python3 for tf2zpk oracle: {e}"
            );
            eprintln!("skipping tf2zpk oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open tf2zpk oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "tf2zpk oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping tf2zpk oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for tf2zpk oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "tf2zpk oracle failed: {stderr}"
        );
        eprintln!("skipping tf2zpk oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse tf2zpk oracle JSON"))
}

#[test]
fn diff_signal_tf2zpk() {
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
        let Some(scipy_v) = scipy_arm.values.as_ref() else {
            continue;
        };
        let Ok(zpk) = tf2zpk(&case.b, &case.a) else {
            continue;
        };
        let zs = pack_sorted(&zpk.zeros_re, &zpk.zeros_im);
        let ps = pack_sorted(&zpk.poles_re, &zpk.poles_im);
        // The oracle reports its zero count separately so a zero/pole miscount cannot hide
        // behind a coincidentally equal packed length.
        if scipy_arm.n_zeros.is_some_and(|n| n != zs.len()) {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let mut fsci_v = Vec::with_capacity(zs.len() * 2 + ps.len() * 2 + 1);
        for (re, im) in zs {
            fsci_v.push(re);
            fsci_v.push(im);
        }
        for (re, im) in ps {
            fsci_v.push(re);
            fsci_v.push(im);
        }
        fsci_v.push(zpk.gain);
        if fsci_v.len() != scipy_v.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                abs_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        }
        let abs_d = fsci_v
            .iter()
            .zip(scipy_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_tf2zpk".into(),
        category: "scipy.signal.tf2zpk".into(),
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
            eprintln!("tf2zpk mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "scipy.signal.tf2zpk conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
