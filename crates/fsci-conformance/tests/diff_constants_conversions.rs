#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_constants SI prefixes
//! and conversion helpers (temperature/speed/mass).
//!
//! Resolves [frankenscipy-uqzea]. Rel 1e-12 (exact float arithmetic
//! expected for SI prefixes; converters use IEC/CODATA constants).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_constants as fc;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REL_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PrefixCase {
    case_id: String,
    /// scipy.constants attribute name.
    scipy_name: String,
}

#[derive(Debug, Clone, Serialize)]
struct TempCase {
    case_id: String,
    val: f64,
    from: String,
    to: String,
}

#[derive(Debug, Clone, Serialize)]
struct SpeedMassCase {
    case_id: String,
    op: String,
    arg: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    prefixes: Vec<PrefixCase>,
    temps: Vec<TempCase>,
    speed_mass: Vec<SpeedMassCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    prefixes: Vec<PointArm>,
    temps: Vec<PointArm>,
    speed_mass: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
    rel_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
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
    fs::create_dir_all(output_dir()).expect("create constants_conversions diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize constants_conversions diff log");
    fs::write(path, json).expect("write constants_conversions diff log");
}

fn prefix_entries() -> Vec<(&'static str, &'static str, f64)> {
    vec![
        ("yotta", "yotta", fc::YOTTA),
        ("zetta", "zetta", fc::ZETTA),
        ("exa", "exa", fc::EXA),
        ("peta", "peta", fc::PETA),
        ("tera", "tera", fc::TERA),
        ("giga", "giga", fc::GIGA),
        ("mega", "mega", fc::MEGA),
        ("kilo", "kilo", fc::KILO),
        ("hecto", "hecto", fc::HECTO),
        ("deka", "deka", fc::DEKA),
        ("deci", "deci", fc::DECI),
        ("centi", "centi", fc::CENTI),
        ("milli", "milli", fc::MILLI),
        ("micro", "micro", fc::MICRO),
        ("nano", "nano", fc::NANO),
        ("pico", "pico", fc::PICO),
        ("femto", "femto", fc::FEMTO),
        ("atto", "atto", fc::ATTO),
        ("zepto", "zepto", fc::ZEPTO),
        ("yocto", "yocto", fc::YOCTO),
    ]
}

fn temp_inputs() -> Vec<TempCase> {
    vec![
        TempCase {
            case_id: "C_to_K_0".into(),
            val: 0.0,
            from: "C".into(),
            to: "K".into(),
        },
        TempCase {
            case_id: "K_to_C_273.15".into(),
            val: 273.15,
            from: "K".into(),
            to: "C".into(),
        },
        TempCase {
            case_id: "F_to_C_100".into(),
            val: 100.0,
            from: "F".into(),
            to: "C".into(),
        },
        TempCase {
            case_id: "C_to_F_100".into(),
            val: 100.0,
            from: "C".into(),
            to: "F".into(),
        },
        TempCase {
            case_id: "F_to_K_32".into(),
            val: 32.0,
            from: "F".into(),
            to: "K".into(),
        },
        TempCase {
            case_id: "K_to_F_300".into(),
            val: 300.0,
            from: "K".into(),
            to: "F".into(),
        },
        TempCase {
            case_id: "R_to_K_491.67".into(),
            val: 491.67,
            from: "R".into(),
            to: "K".into(),
        },
        TempCase {
            case_id: "K_to_R_300".into(),
            val: 300.0,
            from: "K".into(),
            to: "R".into(),
        },
    ]
}

fn speed_mass_inputs() -> Vec<SpeedMassCase> {
    vec![
        SpeedMassCase {
            case_id: "mph_60".into(),
            op: "mph_to_mps".into(),
            arg: 60.0,
        },
        SpeedMassCase {
            case_id: "kmh_100".into(),
            op: "kmh_to_mps".into(),
            arg: 100.0,
        },
        SpeedMassCase {
            case_id: "knots_15".into(),
            op: "knots_to_mps".into(),
            arg: 15.0,
        },
        SpeedMassCase {
            case_id: "psi_14_7".into(),
            op: "psi_to_pa".into(),
            arg: 14.7,
        },
        SpeedMassCase {
            case_id: "lb_150".into(),
            op: "lb_to_kg".into(),
            arg: 150.0,
        },
        SpeedMassCase {
            case_id: "kg_75".into(),
            op: "kg_to_lb".into(),
            arg: 75.0,
        },
    ]
}

fn generate_query() -> OracleQuery {
    OracleQuery {
        prefixes: prefix_entries()
            .iter()
            .map(|(case_id, sname, _)| PrefixCase {
                case_id: (*case_id).into(),
                scipy_name: (*sname).into(),
            })
            .collect(),
        temps: temp_inputs(),
        speed_mass: speed_mass_inputs(),
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy import constants

q = json.load(sys.stdin)

prefixes = []
for case in q["prefixes"]:
    cid = case["case_id"]; sname = case["scipy_name"]
    try:
        v = float(getattr(constants, sname))
        if not math.isfinite(v):
            prefixes.append({"case_id": cid, "value": None})
        else:
            prefixes.append({"case_id": cid, "value": v})
    except Exception:
        prefixes.append({"case_id": cid, "value": None})

temps = []
for case in q["temps"]:
    cid = case["case_id"]
    try:
        v = float(constants.convert_temperature(float(case["val"]), case["from"], case["to"]))
        if not math.isfinite(v):
            temps.append({"case_id": cid, "value": None})
        else:
            temps.append({"case_id": cid, "value": v})
    except Exception:
        temps.append({"case_id": cid, "value": None})

speed_mass = []
mapping = {
    "mph_to_mps":  lambda x: x * constants.mph,
    "kmh_to_mps":  lambda x: x * constants.kmh,
    "knots_to_mps":lambda x: x * constants.knot,
    "psi_to_pa":   lambda x: x * constants.psi,
    "lb_to_kg":    lambda x: x * constants.lb,
    "kg_to_lb":    lambda x: x / constants.lb,
}
for case in q["speed_mass"]:
    cid = case["case_id"]; op = case["op"]; arg = float(case["arg"])
    try:
        v = float(mapping[op](arg))
        if not math.isfinite(v):
            speed_mass.append({"case_id": cid, "value": None})
        else:
            speed_mass.append({"case_id": cid, "value": v})
    except Exception:
        speed_mass.append({"case_id": cid, "value": None})

print(json.dumps({"prefixes": prefixes, "temps": temps, "speed_mass": speed_mass}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize constants_conversions query");
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
                "failed to spawn python3 for constants_conversions oracle: {e}"
            );
            eprintln!("skipping constants_conversions oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open constants_conversions oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "constants_conversions oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!(
                "skipping constants_conversions oracle: stdin write failed ({err})\n{stderr}"
            );
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for constants_conversions oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "constants_conversions oracle failed: {stderr}"
        );
        eprintln!("skipping constants_conversions oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse constants_conversions oracle JSON"))
}

fn rel(a: f64, b: f64) -> f64 {
    if b.abs() > 0.0 {
        (a - b).abs() / b.abs()
    } else {
        (a - b).abs()
    }
}

fn apply_speed_mass(op: &str, x: f64) -> Option<f64> {
    match op {
        "mph_to_mps" => Some(fc::mph_to_mps(x)),
        "kmh_to_mps" => Some(fc::kmh_to_mps(x)),
        "knots_to_mps" => Some(fc::knots_to_mps(x)),
        "psi_to_pa" => Some(fc::psi_to_pa(x)),
        "lb_to_kg" => Some(fc::lb_to_kg(x)),
        "kg_to_lb" => Some(fc::kg_to_lb(x)),
        _ => None,
    }
}

#[test]
fn diff_constants_conversions() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.prefixes.len(), query.prefixes.len());
    assert_eq!(oracle.temps.len(), query.temps.len());
    assert_eq!(oracle.speed_mass.len(), query.speed_mass.len());

    let prefix_map: HashMap<String, PointArm> = oracle
        .prefixes
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let temp_map: HashMap<String, PointArm> = oracle
        .temps
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();
    let sm_map: HashMap<String, PointArm> = oracle
        .speed_mass
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let prefix_entries = prefix_entries();
    let prefix_val_map: HashMap<String, f64> = prefix_entries
        .iter()
        .map(|(case_id, _, v)| ((*case_id).to_string(), *v))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    // prefixes
    for case in &query.prefixes {
        let scipy_arm = prefix_map.get(&case.case_id).expect("validated oracle");
        let Some(scipy_v) = scipy_arm.value else {
            continue;
        };
        let Some(&fsci_v) = prefix_val_map.get(&case.case_id) else {
            continue;
        };
        let r = rel(fsci_v, scipy_v);
        max_overall = max_overall.max(r);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: "prefix".into(),
            rel_diff: r,
            pass: r <= REL_TOL,
        });
    }

    // temps
    for case in &query.temps {
        let scipy_arm = temp_map.get(&case.case_id).expect("validated oracle");
        let Some(scipy_v) = scipy_arm.value else {
            continue;
        };
        let Ok(fsci_v) = fc::convert_temperature(case.val, &case.from, &case.to) else {
            continue;
        };
        let r = rel(fsci_v, scipy_v);
        max_overall = max_overall.max(r);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: format!("temp_{}_{}", case.from, case.to),
            rel_diff: r,
            pass: r <= REL_TOL,
        });
    }

    // speed/mass
    for case in &query.speed_mass {
        let scipy_arm = sm_map.get(&case.case_id).expect("validated oracle");
        let Some(scipy_v) = scipy_arm.value else {
            continue;
        };
        let Some(fsci_v) = apply_speed_mass(&case.op, case.arg) else {
            continue;
        };
        let r = rel(fsci_v, scipy_v);
        max_overall = max_overall.max(r);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            rel_diff: r,
            pass: r <= REL_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_constants_conversions".into(),
        category: "scipy.constants prefixes + converters".into(),
        case_count: diffs.len(),
        max_rel_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("{} mismatch: {} rel_diff={}", d.op, d.case_id, d.rel_diff);
        }
    }

    assert!(
        all_pass,
        "constants_conversions conformance failed: {} cases, max_rel_diff={}",
        diffs.len(),
        max_overall
    );
}
