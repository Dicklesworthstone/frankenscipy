#![forbid(unsafe_code)]
//! Live SciPy/NumPy differential coverage for fsci_constants conversion
//! helpers: convert_temperature, deg2rad/rad2deg, mph_to_mps,
//! kmh_to_mps, knots_to_mps, psi_to_pa, kg_to_lb, lb_to_kg,
//! ev_to_joules round-trip and wavelength↔frequency round-trip.
//!
//! Resolves [frankenscipy-ojijj]. Rel tolerance 1e-12 (these are pure
//! algebraic conversions; the only source of drift is the underlying
//! CODATA constants, which fsci pins to CODATA-2018).

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
struct CasePoint {
    case_id: String,
    /// "convert_temperature" | "deg2rad" | "rad2deg" | ...
    op: String,
    /// First argument
    v: f64,
    /// For convert_temperature: from-scale
    from: String,
    /// For convert_temperature: to-scale
    to: String,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<CasePoint>,
}

#[derive(Debug, Clone, Deserialize)]
struct OraclePoint {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<OraclePoint>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    actual: f64,
    expected: f64,
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
    fs::create_dir_all(output_dir()).expect("create unit_conv diff dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize unit_conv log");
    fs::write(path, json).expect("write unit_conv log");
}

fn build_query() -> OracleQuery {
    let mut pts: Vec<CasePoint> = Vec::new();

    // convert_temperature combinations across C/F/K/R with several values
    let values: &[f64] = &[-40.0, 0.0, 25.0, 100.0, 273.15];
    let scales = ["c", "f", "k", "r"];
    for v in values {
        for from in &scales {
            for to in &scales {
                pts.push(CasePoint {
                    case_id: format!("convT_{from}_to_{to}_at_{v}"),
                    op: "convert_temperature".into(),
                    v: *v,
                    from: (*from).into(),
                    to: (*to).into(),
                });
            }
        }
    }

    // deg2rad / rad2deg sweep
    for v in [0.0_f64, 30.0, 45.0, 90.0, 180.0, 359.0, -90.0, 720.0] {
        pts.push(CasePoint {
            case_id: format!("deg2rad_{v}"),
            op: "deg2rad".into(),
            v,
            from: String::new(),
            to: String::new(),
        });
    }
    for v in [0.0_f64, 0.5, 1.0, 3.14159265, -1.5707963, 6.283] {
        pts.push(CasePoint {
            case_id: format!("rad2deg_{v}"),
            op: "rad2deg".into(),
            v,
            from: String::new(),
            to: String::new(),
        });
    }

    // speed conversions
    for v in [0.0_f64, 1.0, 60.0, 100.0, -25.0] {
        pts.push(CasePoint {
            case_id: format!("mph_to_mps_{v}"),
            op: "mph_to_mps".into(),
            v,
            from: String::new(),
            to: String::new(),
        });
    }
    for v in [0.0_f64, 1.0, 60.0, 100.0, -25.0] {
        pts.push(CasePoint {
            case_id: format!("kmh_to_mps_{v}"),
            op: "kmh_to_mps".into(),
            v,
            from: String::new(),
            to: String::new(),
        });
    }
    for v in [0.0_f64, 1.0, 10.0, 100.0] {
        pts.push(CasePoint {
            case_id: format!("knots_to_mps_{v}"),
            op: "knots_to_mps".into(),
            v,
            from: String::new(),
            to: String::new(),
        });
    }

    // pressure / mass
    for v in [0.0_f64, 1.0, 14.7, 100.0] {
        pts.push(CasePoint {
            case_id: format!("psi_to_pa_{v}"),
            op: "psi_to_pa".into(),
            v,
            from: String::new(),
            to: String::new(),
        });
    }
    for v in [0.0_f64, 1.0, 70.0, 100.0] {
        pts.push(CasePoint {
            case_id: format!("kg_to_lb_{v}"),
            op: "kg_to_lb".into(),
            v,
            from: String::new(),
            to: String::new(),
        });
    }
    for v in [0.0_f64, 1.0, 150.0] {
        pts.push(CasePoint {
            case_id: format!("lb_to_kg_{v}"),
            op: "lb_to_kg".into(),
            v,
            from: String::new(),
            to: String::new(),
        });
    }

    // energy round-trip (eV ↔ J): expressed as ev_to_joules; the oracle
    // computes v * scipy.constants.electron_volt
    for v in [1.0_f64, 13.6, 1000.0, 1.0e-3] {
        pts.push(CasePoint {
            case_id: format!("ev_to_joules_{v}"),
            op: "ev_to_joules".into(),
            v,
            from: String::new(),
            to: String::new(),
        });
    }
    for v in [1.602e-19_f64, 1.0e-15] {
        pts.push(CasePoint {
            case_id: format!("joules_to_ev_{v}"),
            op: "joules_to_ev".into(),
            v,
            from: String::new(),
            to: String::new(),
        });
    }

    // wavelength ↔ frequency
    for v in [1.0e-9_f64, 500.0e-9, 1.0e-6, 0.01] {
        pts.push(CasePoint {
            case_id: format!("wavelength_to_freq_{v}"),
            op: "wavelength_to_freq".into(),
            v,
            from: String::new(),
            to: String::new(),
        });
    }
    for v in [1.0e9_f64, 6.0e14, 1.0e15] {
        pts.push(CasePoint {
            case_id: format!("freq_to_wavelength_{v}"),
            op: "freq_to_wavelength".into(),
            v,
            from: String::new(),
            to: String::new(),
        });
    }

    OracleQuery { points: pts }
}

fn scipy_oracle_or_skip(q: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json, math, sys
import scipy.constants as sc
import numpy as np

q = json.load(sys.stdin)
out = []
for p in q["points"]:
    cid = p["case_id"]; op = p["op"]; v = p["v"]
    try:
        if op == "convert_temperature":
            r = float(sc.convert_temperature(np.array([v]), p["from"], p["to"])[0])
        elif op == "deg2rad":
            r = float(np.deg2rad(v))
        elif op == "rad2deg":
            r = float(np.rad2deg(v))
        elif op == "mph_to_mps":
            r = float(v * sc.mile / 3600.0)
        elif op == "kmh_to_mps":
            r = float(v * sc.kmh)  # sc.kmh == 1000/3600
        elif op == "knots_to_mps":
            r = float(v * sc.knot)
        elif op == "psi_to_pa":
            r = float(v * sc.psi)
        elif op == "kg_to_lb":
            r = float(v / sc.lb)
        elif op == "lb_to_kg":
            r = float(v * sc.lb)
        elif op == "ev_to_joules":
            r = float(v * sc.electron_volt)
        elif op == "joules_to_ev":
            r = float(v / sc.electron_volt)
        elif op == "wavelength_to_freq":
            r = float(sc.c / v)
        elif op == "freq_to_wavelength":
            r = float(sc.c / v)
        else:
            r = None
        if r is None or not math.isfinite(r):
            out.append({"case_id": cid, "value": None})
        else:
            out.append({"case_id": cid, "value": r})
    except Exception:
        out.append({"case_id": cid, "value": None})

print(json.dumps({"points": out}))
"#;
    let query_json = serde_json::to_string(q).expect("serialize unit_conv query");
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
                "failed to spawn python3: {e}"
            );
            eprintln!("skipping unit_conv oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping unit_conv oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "oracle failed: {stderr}"
        );
        eprintln!("skipping unit_conv oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse unit_conv JSON"))
}

fn fsci_compute(p: &CasePoint) -> Option<f64> {
    let r = match p.op.as_str() {
        "convert_temperature" => fc::convert_temperature(p.v, &p.from, &p.to).ok()?,
        "deg2rad" => fc::deg2rad(p.v),
        "rad2deg" => fc::rad2deg(p.v),
        "mph_to_mps" => fc::mph_to_mps(p.v),
        "kmh_to_mps" => fc::kmh_to_mps(p.v),
        "knots_to_mps" => fc::knots_to_mps(p.v),
        "psi_to_pa" => fc::psi_to_pa(p.v),
        "kg_to_lb" => fc::kg_to_lb(p.v),
        "lb_to_kg" => fc::lb_to_kg(p.v),
        "ev_to_joules" => fc::ev_to_joules(p.v),
        "joules_to_ev" => fc::joules_to_ev(p.v),
        "wavelength_to_freq" => fc::wavelength_to_freq(p.v),
        "freq_to_wavelength" => fc::freq_to_wavelength(p.v),
        _ => return None,
    };
    if r.is_finite() { Some(r) } else { None }
}

#[test]
fn diff_constants_unit_conversions() {
    let query = build_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut max_rel = 0.0_f64;

    for (q, o) in query.points.iter().zip(oracle.points.iter()) {
        assert_eq!(q.case_id, o.case_id, "oracle returned out-of-order points");
        let Some(expected) = o.value else {
            continue; // oracle skipped this case; skip
        };
        let Some(actual) = fsci_compute(q) else {
            diffs.push(CaseDiff {
                case_id: q.case_id.clone(),
                actual: f64::NAN,
                expected,
                rel_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        };
        // Relative error guarded against tiny denominators
        let denom = expected.abs().max(1.0e-300);
        let rel = (actual - expected).abs() / denom;
        max_rel = max_rel.max(rel);
        diffs.push(CaseDiff {
            case_id: q.case_id.clone(),
            actual,
            expected,
            rel_diff: rel,
            pass: rel <= REL_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_constants_unit_conversions".into(),
        category: "fsci_constants unit conversions vs scipy.constants/numpy".into(),
        case_count: diffs.len(),
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
                "unit_conv mismatch: {} actual={} expected={} rel={}",
                d.case_id, d.actual, d.expected, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "unit-conversion parity failed: {} cases, max_rel={}",
        diffs.len(),
        max_rel
    );
}
