#![forbid(unsafe_code)]
//! Live scipy.constants parity for fsci_constants temperature, energy,
//! wavelength, and angle conversions.
//!
//! Resolves [frankenscipy-5lxm0]. Tolerance: 1e-12 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_constants as fc;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REL_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct ConvCase {
    case_id: String,
    op: String,
    arg: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<ConvCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    value: Option<f64>,
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
    rel_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    max_abs_diff: f64,
    max_rel_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create constants conv diff dir");
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

fn fsci_eval(op: &str, arg: f64) -> Option<f64> {
    Some(match op {
        "celsius_to_kelvin" => fc::celsius_to_kelvin(arg),
        "kelvin_to_celsius" => fc::kelvin_to_celsius(arg),
        "fahrenheit_to_kelvin" => fc::fahrenheit_to_kelvin(arg),
        "kelvin_to_fahrenheit" => fc::kelvin_to_fahrenheit(arg),
        "fahrenheit_to_celsius" => fc::fahrenheit_to_celsius(arg),
        "celsius_to_fahrenheit" => fc::celsius_to_fahrenheit(arg),
        "rankine_to_kelvin" => fc::rankine_to_kelvin(arg),
        "kelvin_to_rankine" => fc::kelvin_to_rankine(arg),
        "ev_to_joules" => fc::ev_to_joules(arg),
        "joules_to_ev" => fc::joules_to_ev(arg),
        "wavelength_to_freq" => fc::wavelength_to_freq(arg),
        "freq_to_wavelength" => fc::freq_to_wavelength(arg),
        "deg2rad" => fc::deg2rad(arg),
        "rad2deg" => fc::rad2deg(arg),
        _ => return None,
    })
}

fn generate_query() -> OracleQuery {
    let mut p = Vec::new();
    let temps_c: &[f64] = &[-273.15, -40.0, 0.0, 25.0, 100.0];
    for &c in temps_c {
        p.push(ConvCase {
            case_id: format!("c2k_{c}").replace('.', "p").replace('-', "n"),
            op: "celsius_to_kelvin".into(),
            arg: c,
        });
        p.push(ConvCase {
            case_id: format!("c2f_{c}").replace('.', "p").replace('-', "n"),
            op: "celsius_to_fahrenheit".into(),
            arg: c,
        });
    }
    let temps_k: &[f64] = &[0.0, 273.15, 300.0, 1000.0];
    for &k in temps_k {
        p.push(ConvCase {
            case_id: format!("k2c_{k}").replace('.', "p"),
            op: "kelvin_to_celsius".into(),
            arg: k,
        });
        p.push(ConvCase {
            case_id: format!("k2f_{k}").replace('.', "p"),
            op: "kelvin_to_fahrenheit".into(),
            arg: k,
        });
        p.push(ConvCase {
            case_id: format!("k2r_{k}").replace('.', "p"),
            op: "kelvin_to_rankine".into(),
            arg: k,
        });
    }
    let temps_f: &[f64] = &[-40.0, 0.0, 32.0, 212.0];
    for &f in temps_f {
        p.push(ConvCase {
            case_id: format!("f2k_{f}").replace('.', "p").replace('-', "n"),
            op: "fahrenheit_to_kelvin".into(),
            arg: f,
        });
        p.push(ConvCase {
            case_id: format!("f2c_{f}").replace('.', "p").replace('-', "n"),
            op: "fahrenheit_to_celsius".into(),
            arg: f,
        });
    }
    let temps_r: &[f64] = &[491.67, 500.0, 671.67];
    for &r in temps_r {
        p.push(ConvCase {
            case_id: format!("r2k_{r}").replace('.', "p"),
            op: "rankine_to_kelvin".into(),
            arg: r,
        });
    }
    let evs: &[f64] = &[1.0, 13.6, 1000.0, 1.0e6];
    for &e in evs {
        p.push(ConvCase {
            case_id: format!("ev2j_{e}").replace('.', "p"),
            op: "ev_to_joules".into(),
            arg: e,
        });
    }
    let joules: &[f64] = &[1.6e-19, 1.0e-15, 1.0];
    for &j in joules {
        p.push(ConvCase {
            case_id: format!("j2ev_{j}").replace('.', "p").replace('-', "n"),
            op: "joules_to_ev".into(),
            arg: j,
        });
    }
    let wavelengths: &[f64] = &[1.0e-9, 5.0e-7, 1.0e-3, 1.0];
    for &w in wavelengths {
        p.push(ConvCase {
            case_id: format!("w2f_{w}").replace('.', "p").replace('-', "n"),
            op: "wavelength_to_freq".into(),
            arg: w,
        });
    }
    let freqs: &[f64] = &[1.0e6, 1.0e9, 5.0e14];
    for &f in freqs {
        p.push(ConvCase {
            case_id: format!("f2w_{f}").replace('.', "p"),
            op: "freq_to_wavelength".into(),
            arg: f,
        });
    }
    let degs: &[f64] = &[0.0, 30.0, 45.0, 90.0, 180.0, 270.0, 360.0, -45.0];
    for &d in degs {
        p.push(ConvCase {
            case_id: format!("d2r_{d}").replace('.', "p").replace('-', "n"),
            op: "deg2rad".into(),
            arg: d,
        });
    }
    let rads: &[f64] = &[
        0.0,
        0.5,
        1.0,
        std::f64::consts::PI,
        2.0 * std::f64::consts::PI,
        -1.5,
    ];
    for &r in rads {
        p.push(ConvCase {
            case_id: format!("r2d_{r}").replace('.', "p").replace('-', "n"),
            op: "rad2deg".into(),
            arg: r,
        });
    }
    OracleQuery { points: p }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy import constants as sc

def f(op, x):
    if op == "celsius_to_kelvin":     return sc.convert_temperature(x, 'Celsius', 'Kelvin')
    if op == "kelvin_to_celsius":     return sc.convert_temperature(x, 'Kelvin', 'Celsius')
    if op == "fahrenheit_to_kelvin":  return sc.convert_temperature(x, 'Fahrenheit', 'Kelvin')
    if op == "kelvin_to_fahrenheit":  return sc.convert_temperature(x, 'Kelvin', 'Fahrenheit')
    if op == "fahrenheit_to_celsius": return sc.convert_temperature(x, 'Fahrenheit', 'Celsius')
    if op == "celsius_to_fahrenheit": return sc.convert_temperature(x, 'Celsius', 'Fahrenheit')
    if op == "rankine_to_kelvin":     return sc.convert_temperature(x, 'Rankine', 'Kelvin')
    if op == "kelvin_to_rankine":     return sc.convert_temperature(x, 'Kelvin', 'Rankine')
    if op == "ev_to_joules":          return x * sc.electron_volt
    if op == "joules_to_ev":          return x / sc.electron_volt
    if op == "wavelength_to_freq":    return sc.c / x
    if op == "freq_to_wavelength":    return sc.c / x
    if op == "deg2rad":               return math.radians(x)
    if op == "rad2deg":               return math.degrees(x)
    return float("nan")

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]; arg = float(case["arg"])
    try:
        v = float(f(op, arg))
        if math.isfinite(v):
            points.append({"case_id": cid, "value": v})
        else:
            points.append({"case_id": cid, "value": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "value": None})
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
                "failed to spawn python3 for constants conv oracle: {e}"
            );
            eprintln!("skipping constants conv oracle: python3 not available ({e})");
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
                "constants conv oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping constants conv oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for constants conv oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "constants conv oracle failed: {stderr}"
        );
        eprintln!("skipping constants conv oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse constants conv oracle JSON"))
}

#[test]
fn diff_constants_temp_ev_wavelength_deg() {
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
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;

    for case in &query.points {
        let Some(arm) = pmap.get(&case.case_id) else {
            continue;
        };
        let Some(expected) = arm.value else {
            continue;
        };
        let Some(actual) = fsci_eval(&case.op, case.arg) else {
            continue;
        };
        let abs_d = (actual - expected).abs();
        let rel_d = if expected.abs() > 1.0e-12 {
            abs_d / expected.abs()
        } else {
            abs_d
        };
        max_abs = max_abs.max(abs_d);
        max_rel = max_rel.max(rel_d);
        // Use the looser of (abs <= ABS_TOL) OR (rel <= REL_TOL) so that
        // very small reference magnitudes (deg2rad of 0) don't fail solely
        // due to absolute eps noise.
        let pass = abs_d <= ABS_TOL || rel_d <= REL_TOL;
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            abs_diff: abs_d,
            rel_diff: rel_d,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_constants_temp_ev_wavelength_deg".into(),
        category: "fsci_constants conversions vs scipy.constants".into(),
        case_count: diffs.len(),
        max_abs_diff: max_abs,
        max_rel_diff: max_rel,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "{} mismatch: {} abs={} rel={}",
                d.op, d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "constants conv conformance failed: {} cases, max_abs={}, max_rel={}",
        diffs.len(),
        max_abs,
        max_rel
    );
}
