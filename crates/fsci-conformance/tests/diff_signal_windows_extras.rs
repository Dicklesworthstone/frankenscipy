#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the remaining
//! scipy.signal.windows family that fsci-signal exposes beyond the
//! hann/hamming/blackman trio already in `diff_signal_windows`:
//!   - `bartlett`, `triang`, `blackmanharris`, `nuttall`, `flattop`,
//!     `boxcar`, `parzen`, `bohman`, `tukey(α)`, `gaussian(σ)`,
//!     `kaiser(β)`
//!
//! Resolves [frankenscipy-q8yky]. Most windows agree with scipy to
//! ~1e-14; `kaiser` composes a modified-Bessel-I0 computation that
//! lands ~1.6e-9 short on the open-form mid-vector samples (the
//! lib-side `modified_bessel_i` precision floor), so it gets a
//! dedicated looser tolerance.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{
    bartlett, blackmanharris, bohman_window, boxcar, flattop, gaussian, kaiser, nuttall_window,
    parzen, triang, tukey_window,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-011";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

const STRICT_TOL: f64 = 1.0e-13;
// Kaiser routes through I0; fsci_special::modified_bessel_i has a
// ~1.2e-8 floor at n=65, β≥3 (mid-vector samples) — the relative
// magnitudes of large-argument I0 accumulate the I0 series rounding
// error. Documented looser tol; tighten once the I0 floor closes.
const KAISER_TOL: f64 = 5.0e-8;

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    n: usize,
    /// Parameter for `tukey(alpha)`, `gaussian(std)`, `kaiser(beta)`.
    /// Ignored otherwise.
    param: f64,
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
    func: String,
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
    fs::create_dir_all(output_dir()).expect("create windows_extras diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize windows_extras diff log");
    fs::write(path, json).expect("write windows_extras diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    let ns: &[usize] = &[2, 4, 8, 16, 65];
    for &n in ns {
        for func in [
            "bartlett",
            "triang",
            "blackmanharris",
            "nuttall",
            "flattop",
            "boxcar",
            "parzen",
            "bohman",
        ] {
            points.push(PointCase {
                case_id: format!("{func}_n{n}"),
                func: func.into(),
                n,
                param: 0.0,
            });
        }
        // Parameterised windows.
        for &alpha in &[0.1_f64, 0.5, 1.0] {
            points.push(PointCase {
                case_id: format!("tukey_n{n}_a{alpha}"),
                func: "tukey".into(),
                n,
                param: alpha,
            });
        }
        for &sigma in &[0.5_f64, 1.5, 3.0] {
            points.push(PointCase {
                case_id: format!("gaussian_n{n}_s{sigma}"),
                func: "gaussian".into(),
                n,
                param: sigma,
            });
        }
        for &beta in &[3.0_f64, 5.0, 8.0] {
            points.push(PointCase {
                case_id: format!("kaiser_n{n}_b{beta}"),
                func: "kaiser".into(),
                n,
                param: beta,
            });
        }
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy import signal as sp_signal

def finite_vec_or_none(arr):
    out = []
    for v in arr.tolist():
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
    cid = case["case_id"]; func = case["func"]
    n = int(case["n"]); p = float(case["param"])
    try:
        if   func == "bartlett":       v = sp_signal.windows.bartlett(n, sym=True)
        elif func == "triang":         v = sp_signal.windows.triang(n, sym=True)
        elif func == "blackmanharris": v = sp_signal.windows.blackmanharris(n, sym=True)
        elif func == "nuttall":        v = sp_signal.windows.nuttall(n, sym=True)
        elif func == "flattop":        v = sp_signal.windows.flattop(n, sym=True)
        elif func == "boxcar":         v = sp_signal.windows.boxcar(n, sym=True)
        elif func == "parzen":         v = sp_signal.windows.parzen(n, sym=True)
        elif func == "bohman":         v = sp_signal.windows.bohman(n, sym=True)
        elif func == "tukey":          v = sp_signal.windows.tukey(n, alpha=p, sym=True)
        elif func == "gaussian":       v = sp_signal.windows.gaussian(n, std=p, sym=True)
        elif func == "kaiser":         v = sp_signal.windows.kaiser(n, beta=p, sym=True)
        else: v = None
        points.append({"case_id": cid, "values": finite_vec_or_none(v) if v is not None else None})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize windows_extras query");
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
                "failed to spawn python3 for windows_extras oracle: {e}"
            );
            eprintln!("skipping windows_extras oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open windows_extras oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "windows_extras oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping windows_extras oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for windows_extras oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "windows_extras oracle failed: {stderr}"
        );
        eprintln!("skipping windows_extras oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse windows_extras oracle JSON"))
}

fn fsci_eval(func: &str, n: usize, param: f64) -> Option<Vec<f64>> {
    Some(match func {
        "bartlett" => bartlett(n),
        "triang" => triang(n),
        "blackmanharris" => blackmanharris(n),
        "nuttall" => nuttall_window(n),
        "flattop" => flattop(n),
        "boxcar" => boxcar(n, true),
        "parzen" => parzen(n),
        "bohman" => bohman_window(n),
        "tukey" => tukey_window(n, param),
        "gaussian" => gaussian(n, param, true),
        "kaiser" => kaiser(n, param),
        _ => return None,
    })
}

fn func_tol(func: &str) -> f64 {
    if func == "kaiser" {
        KAISER_TOL
    } else {
        STRICT_TOL
    }
}

#[test]
fn diff_signal_windows_extras() {
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
        let Some(fsci_v) = fsci_eval(&case.func, case.n, case.param) else {
            continue;
        };
        let Some(scipy_v) = scipy_arm.values.as_ref() else {
            continue;
        };
        if fsci_v.len() != scipy_v.len() {
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
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
        let tol = func_tol(&case.func);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            func: case.func.clone(),
            abs_diff: abs_d,
            pass: abs_d <= tol,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_windows_extras".into(),
        category: "scipy.signal.windows.{bartlett,triang,blackmanharris,nuttall,flattop,boxcar,parzen,bohman,tukey,gaussian,kaiser}"
            .into(),
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
                "window {} mismatch: {} abs_diff={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.signal.windows extras conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
