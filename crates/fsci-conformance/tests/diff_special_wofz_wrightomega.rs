#![forbid(unsafe_code)]
//! Live SciPy differential coverage for two complex / Lambert-style
//! special functions:
//!   - `wofz_real(x)` vs `(Re, Im) of scipy.special.wofz(x + 0j)`
//!     (Faddeeva function for real argument)
//!   - `wofz_scalar(x + iy)` vs `scipy.special.wofz(x + iy)` over every
//!     branch of the Faddeeva package, densest near the real axis
//!   - `wrightomega_scalar(z)` vs `scipy.special.wrightomega(z)`
//!     (Wright omega, principal real branch)
//!
//! Resolves [frankenscipy-713ui]; the complex arm is frankenscipy-k64p7.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_runtime::RuntimeMode;
use fsci_special::{Complex64, wofz_real, wofz_scalar, wrightomega_scalar};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-002";
// wofz is a port of the Faddeeva package SciPy runs, bit-identical to it. This is a TRUE relative
// tolerance per component, a few ulp. It was an absolute 1e-7 on the real axis only, and the
// complex plane was not compared at all: Re w near the axis, where it is e^{-x²} and tiny against
// |w|, was up to 100% off at |x| ≈ 4.5 (frankenscipy-k64p7).
const WOFZ_TOL_REL: f64 = 1.0e-15;
// wrightomega is xsf's, bit-identical to SciPy. This is a TRUE relative tolerance, a few ulp. It
// was an absolute 1e-12, blind to relative error on the tiny values below z = -18 where e^z
// was returned early (frankenscipy-i20cg).
const WRIGHTOMEGA_TOL_REL: f64 = 1.0e-15;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
/// One ledger arm per function.
const ARMS: [&str; 3] = ["wofz_real", "wofz_complex", "wrightomega"];

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    x: f64,
    /// Imaginary part of the argument; 0 for the real-argument arms.
    y: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// wofz_real and wofz_complex: [re, im]; wrightomega: [val].
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
    fs::create_dir_all(output_dir()).expect("create wofz_wrightomega diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize wofz_wrightomega diff log");
    fs::write(path, json).expect("write wofz_wrightomega diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    // wofz real argument samples
    let wofz_xs: &[f64] = &[
        -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0,
    ];
    for (i, x) in wofz_xs.iter().enumerate() {
        points.push(PointCase {
            case_id: format!("wofz_real_{i:02}_x{x}"),
            func: "wofz_real".into(),
            x: *x,
            y: 0.0,
        });
    }
    // wofz complex argument samples. The grid is densest near the real axis across the band
    // edges of the dispatch fsci used before (|z| = 0.5, 4, 8), where Re w lost its digits;
    // |x| up to 29.5 crosses the Faddeeva package's own switches (x = 6, 8, 10, 28).
    let near_axis_xs: &[f64] = &[
        0.3, 1.1, 2.7, 3.3, 4.23, -4.23, 4.9, 5.6, 6.5, 7.4, 9.0, 12.0, 20.0, 29.5,
    ];
    let near_axis_ys: &[f64] = &[1.0e-10, 1.0e-6, 1.0e-3, 0.05, 0.3, 2.0, 6.0];
    let mut complex_zs: Vec<(f64, f64)> = Vec::new();
    for &x in near_axis_xs {
        for &y in near_axis_ys {
            complex_zs.push((x, y));
        }
    }
    for &x in &[0.3, 2.7, 4.9, 7.4] {
        for &y in &[-1.0e-3, -0.5, -3.0] {
            complex_zs.push((x, y));
        }
    }
    complex_zs.extend_from_slice(&[
        // voigt_profile's worst point, z = (x + iγ)/(√2σ) at (-9.9622, 1.6646, 0.11926)
        (-4.23185309219969, 0.05065875922475325),
        // Re z == 0 (erfcx), x < 5e-4 (Taylor sums), y < -6, x + |y| > 4000 and > 1e7
        (0.0, 2.5),
        (0.0, -3.0),
        (1.0e-4, 0.3),
        (1.0, -6.5),
        (3000.0, 2000.0),
        (2.0e7, 1.0),
        (1.0, 2.0e7),
    ]);
    for (i, &(x, y)) in complex_zs.iter().enumerate() {
        points.push(PointCase {
            case_id: format!("wofz_complex_{i:03}_x{x}_y{y}"),
            func: "wofz_complex".into(),
            x,
            y,
        });
    }
    // wrightomega real-argument samples
    // -40 and -18.96 sit in the range where e^z used to be returned; 50 and 1e21 cover the
    // large-z seed and the z > 1e20 shortcut.
    let wo_xs: &[f64] = &[
        -3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, -40.0, -18.96, -5.0, 50.0, 1e21,
    ];
    for (i, x) in wo_xs.iter().enumerate() {
        points.push(PointCase {
            case_id: format!("wrightomega_{i:02}_z{x}"),
            func: "wrightomega".into(),
            x: *x,
            y: 0.0,
        });
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy import special

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; fn = case["func"]; x = float(case["x"]); y = float(case["y"])
    try:
        if fn in ("wofz_real", "wofz_complex"):
            v = special.wofz(complex(x, y if fn == "wofz_complex" else 0.0))
            re = fnone(v.real); im = fnone(v.imag)
            if re is None or im is None:
                points.append({"case_id": cid, "values": None})
            else:
                points.append({"case_id": cid, "values": [re, im]})
        elif fn == "wrightomega":
            v = fnone(special.wrightomega(x))
            points.append({"case_id": cid, "values": [v] if v is not None else None})
        else:
            points.append({"case_id": cid, "values": None})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize wofz_wrightomega query");
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
                "failed to spawn python3 for wofz_wrightomega oracle: {e}"
            );
            eprintln!("skipping wofz_wrightomega oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open wofz_wrightomega oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "wofz_wrightomega oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping wofz_wrightomega oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for wofz_wrightomega oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "wofz_wrightomega oracle failed: {stderr}"
        );
        eprintln!("skipping wofz_wrightomega oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse wofz_wrightomega oracle JSON"))
}

#[test]
fn diff_special_wofz_wrightomega() {
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
    let mut ledger = CompareLedger::new("diff_special_wofz_wrightomega", &ARMS);

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let arm = case.func.as_str();
        let (fsci_v, tol): (Option<Vec<f64>>, f64) = match arm {
            "wofz_real" => {
                let (re, im) = wofz_real(case.x);
                (Some(vec![re, im]), WOFZ_TOL_REL)
            }
            "wofz_complex" => (
                wofz_scalar(Complex64::new(case.x, case.y), RuntimeMode::Strict)
                    .ok()
                    .map(|w| vec![w.re, w.im]),
                WOFZ_TOL_REL,
            ),
            "wrightomega" => (Some(vec![wrightomega_scalar(case.x)]), WRIGHTOMEGA_TOL_REL),
            other => panic!("unknown func {other} in {}", case.case_id),
        };
        // The ledger rejects a length mismatch and a non-finite fsci element.
        let Some((scipy_v, fsci_v)) = ledger.slices(
            arm,
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
        // Every tolerance is relative to SciPy's value, component by component: Re w near the
        // real axis is orders of magnitude below |w|, so one tolerance on |w| would not see it.
        let pass = fsci_v
            .iter()
            .zip(scipy_v.iter())
            .all(|(a, b)| (a - b).abs() <= tol * b.abs());
        ledger.compared(arm, &case.case_id, pass);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            func: case.func.clone(),
            abs_diff: abs_d,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_wofz_wrightomega".into(),
        category: "scipy.special.wofz (real and complex) + wrightomega".into(),
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
            eprintln!(
                "wofz_wrightomega {} mismatch: {} abs_diff={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special wofz/wrightomega conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    // Arms have different case sets (wrightomega has the fewest); each must compare all of its own.
    let min_per_arm = ARMS
        .iter()
        .map(|arm| query.points.iter().filter(|c| c.func == *arm).count())
        .min()
        .expect("ARMS is non-empty");
    ledger.finish(min_per_arm);
}
