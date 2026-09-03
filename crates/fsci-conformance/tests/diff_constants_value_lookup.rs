#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `fsci_constants::value()`
//! name-based CODATA lookup. Maps fsci's lowercase key to the
//! corresponding scipy.constants.value() canonical name.
//!
//! Resolves [frankenscipy-uquh5]. Rel 1e-7 (fsci ships CODATA-2018).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_constants as fc;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REL_TOL: f64 = 1.0e-7;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    fsci_key: String,
    scipy_key: String,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
    finds: Vec<FindCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    value: Option<f64>,
    unit: Option<String>,
    precision_bits: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
    finds: Vec<FindArm>,
}

#[derive(Debug, Clone)]
struct FindExpectation {
    case_id: &'static str,
    query: &'static str,
    fsci_name: &'static str,
    scipy_key: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct FindCase {
    case_id: String,
    query: String,
    scipy_key: String,
}

#[derive(Debug, Clone, Deserialize)]
struct FindArm {
    case_id: String,
    found: bool,
    names: Vec<String>,
    value: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create value_lookup diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize value_lookup diff log");
    fs::write(path, json).expect("write value_lookup diff log");
}

/// (case_id, fsci_key, scipy_key) for paired lookups.
fn key_pairs() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        (
            "speed_of_light",
            "speed of light",
            "speed of light in vacuum",
        ),
        ("planck", "planck", "Planck constant"),
        ("hbar", "hbar", "reduced Planck constant"),
        (
            "g_grav",
            "gravitational constant",
            "Newtonian constant of gravitation",
        ),
        (
            "elementary_charge",
            "elementary charge",
            "elementary charge",
        ),
        ("gas_constant", "gas constant", "molar gas constant"),
        ("avogadro", "avogadro", "Avogadro constant"),
        ("boltzmann", "boltzmann", "Boltzmann constant"),
        (
            "stefan_boltzmann",
            "stefan-boltzmann",
            "Stefan-Boltzmann constant",
        ),
        ("wien", "wien", "Wien wavelength displacement law constant"),
        ("electron_mass", "electron mass", "electron mass"),
        ("proton_mass", "proton mass", "proton mass"),
        ("neutron_mass", "neutron mass", "neutron mass"),
        ("rydberg", "rydberg", "Rydberg constant"),
        (
            "fine_structure",
            "fine-structure",
            "fine-structure constant",
        ),
        ("bohr_radius", "bohr radius", "Bohr radius"),
        ("faraday", "faraday constant", "Faraday constant"),
        (
            "electron_g_factor",
            "electron g factor",
            "electron g factor",
        ),
        (
            "thomson_cross_section",
            "thomson cross section",
            "Thomson cross section",
        ),
        (
            "characteristic_impedance_of_vacuum",
            "characteristic impedance of vacuum",
            "characteristic impedance of vacuum",
        ),
    ]
}

fn generate_query() -> OracleQuery {
    OracleQuery {
        points: key_pairs()
            .iter()
            .map(|(case_id, fk, sk)| PointCase {
                case_id: (*case_id).into(),
                fsci_key: (*fk).into(),
                scipy_key: (*sk).into(),
            })
            .collect(),
        finds: Vec::new(),
    }
}

fn find_expectations() -> Vec<FindExpectation> {
    vec![
        FindExpectation {
            case_id: "find_faraday",
            query: "Faraday",
            fsci_name: "Faraday constant",
            scipy_key: "Faraday constant",
        },
        FindExpectation {
            case_id: "find_muon_g_factor",
            query: "g factor",
            fsci_name: "muon g factor",
            scipy_key: "muon g factor",
        },
        FindExpectation {
            case_id: "find_thomson_cross_section",
            query: "cross section",
            fsci_name: "Thomson cross section",
            scipy_key: "Thomson cross section",
        },
        FindExpectation {
            case_id: "find_vacuum_impedance",
            query: "impedance",
            fsci_name: "characteristic impedance of vacuum",
            scipy_key: "characteristic impedance of vacuum",
        },
        FindExpectation {
            case_id: "find_alpha_mass_mev",
            query: "mass energy equivalent in MeV",
            fsci_name: "alpha particle mass energy equivalent in MeV",
            scipy_key: "alpha particle mass energy equivalent in MeV",
        },
        FindExpectation {
            case_id: "find_proton_compton_wavelength",
            query: "Compton wavelength",
            fsci_name: "proton Compton wavelength",
            scipy_key: "proton Compton wavelength",
        },
        FindExpectation {
            case_id: "find_tau_electron_ratio",
            query: "tau-electron",
            fsci_name: "tau-electron mass ratio",
            scipy_key: "tau-electron mass ratio",
        },
        FindExpectation {
            case_id: "find_wien_wavelength",
            query: "Wien",
            fsci_name: "Wien wavelength displacement law constant",
            scipy_key: "Wien wavelength displacement law constant",
        },
    ]
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import os
import struct
import sys
from scipy import constants

q = json.loads(os.environ["FSCI_ORACLE_QUERY"])
points = []
for case in q.get("points", []):
    cid = case["case_id"]; sk = case["scipy_key"]
    try:
        v = float(constants.value(sk))
        u = str(constants.unit(sk))
        p = float(constants.precision(sk))
        p_bits = struct.unpack(">Q", struct.pack(">d", p))[0]
        if not math.isfinite(v):
            points.append({"case_id": cid, "value": None, "unit": None, "precision_bits": None})
        else:
            points.append({"case_id": cid, "value": v, "unit": u, "precision_bits": p_bits})
    except Exception:
        points.append({"case_id": cid, "value": None, "unit": None, "precision_bits": None})

finds = []
for case in q.get("finds", []):
    cid = case["case_id"]; query = case["query"]; sk = case["scipy_key"]
    try:
        names = constants.find(query, disp=False)
        found = sk in names
        v = float(constants.value(sk)) if found else None
        if v is not None and not math.isfinite(v):
            v = None
        finds.append({"case_id": cid, "found": found, "names": names, "value": v})
    except Exception:
        finds.append({"case_id": cid, "found": False, "names": [], "value": None})
print(json.dumps({"points": points, "finds": finds}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize value_lookup query");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-")
        .env("FSCI_ORACLE_QUERY", query_json)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for value_lookup oracle: {e}"
            );
            eprintln!("skipping value_lookup oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open value_lookup oracle stdin");
        if let Err(err) = stdin.write_all(script.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "value_lookup oracle script write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping value_lookup oracle: script write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for value_lookup oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "value_lookup oracle failed: {stderr}"
        );
        eprintln!("skipping value_lookup oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse value_lookup oracle JSON"))
}

#[test]
fn diff_constants_value_lookup() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_v) = scipy_arm.value else {
            continue;
        };
        let Some(fsci_v) = fc::value(&case.fsci_key) else {
            // fsci doesn't know this key — count as failure to surface
            // any naming regressions.
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                rel_diff: f64::INFINITY,
                pass: false,
            });
            continue;
        };
        let r = if scipy_v.abs() > 0.0 {
            (fsci_v - scipy_v).abs() / scipy_v.abs()
        } else {
            (fsci_v - scipy_v).abs()
        };
        max_overall = max_overall.max(r);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            rel_diff: r,
            pass: r <= REL_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_constants_value_lookup".into(),
        category: "scipy.constants.value() name-based lookup".into(),
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
            eprintln!("value() mismatch: {} rel_diff={}", d.case_id, d.rel_diff);
        }
    }

    assert!(
        all_pass,
        "value_lookup conformance failed: {} cases, max_rel_diff={}",
        diffs.len(),
        max_overall
    );
}

#[test]
fn diff_constants_unit_and_precision_for_every_supported_key() {
    let supported = fc::find("");
    let query = OracleQuery {
        points: supported
            .iter()
            .enumerate()
            .map(|(index, (name, _))| PointCase {
                case_id: format!("metadata_{index:03}"),
                fsci_key: (*name).into(),
                scipy_key: (*name).into(),
            })
            .collect(),
        finds: Vec::new(),
    };
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|point| (point.case_id.clone(), point))
        .collect();

    for case in &query.points {
        let scipy = pmap.get(&case.case_id).expect("validated metadata oracle");
        let scipy_unit = scipy
            .unit
            .as_deref()
            .expect("SciPy constants unit lookup should succeed");
        let scipy_precision_bits = scipy
            .precision_bits
            .expect("SciPy constants precision lookup should succeed");
        let fsci_unit =
            fc::unit(&case.fsci_key).expect("FrankenSciPy constants unit lookup should succeed");
        let fsci_precision = fc::precision(&case.fsci_key)
            .expect("FrankenSciPy constants precision lookup should succeed");

        assert_eq!(
            fsci_unit, scipy_unit,
            "unit mismatch for {:?}",
            case.scipy_key
        );
        assert_eq!(
            fsci_precision.to_bits(),
            scipy_precision_bits,
            "precision mismatch for {:?}: fsci={fsci_precision:.17e}, scipy_bits={scipy_precision_bits:#018x}",
            case.scipy_key
        );
    }
}

#[test]
fn diff_constants_find_queries_include_scipy_values() {
    let expectations = find_expectations();
    let query = OracleQuery {
        points: Vec::new(),
        finds: expectations
            .iter()
            .map(|case| FindCase {
                case_id: case.case_id.into(),
                query: case.query.into(),
                scipy_key: case.scipy_key.into(),
            })
            .collect(),
    };
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.finds.len(), query.finds.len());

    let pmap: HashMap<String, FindArm> = oracle
        .finds
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &expectations {
        let scipy_arm = pmap.get(case.case_id).expect("validated find oracle");
        let fsci_v = fc::find(case.query)
            .into_iter()
            .find(|(name, _)| *name == case.fsci_name)
            .map(|(_, value)| value);
        let pass = if let (true, Some(scipy_v), Some(fsci_v)) =
            (scipy_arm.found, scipy_arm.value, fsci_v)
        {
            let rel_diff = if scipy_v.abs() > 0.0 {
                (fsci_v - scipy_v).abs() / scipy_v.abs()
            } else {
                (fsci_v - scipy_v).abs()
            };
            max_overall = max_overall.max(rel_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.into(),
                rel_diff,
                pass: rel_diff <= REL_TOL,
            });
            true
        } else {
            diffs.push(CaseDiff {
                case_id: case.case_id.into(),
                rel_diff: f64::INFINITY,
                pass: false,
            });
            false
        };

        if !pass {
            eprintln!(
                "find() mismatch: {} query={:?} fsci_name={:?} scipy_key={:?}",
                case.case_id, case.query, case.fsci_name, case.scipy_key
            );
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_constants_find_queries".into(),
        category: "scipy.constants.find() query membership and value lookup".into(),
        case_count: diffs.len(),
        max_rel_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    assert!(
        all_pass,
        "constants.find conformance failed: {} cases, max_rel_diff={}",
        diffs.len(),
        max_overall
    );
}

#[test]
fn diff_constants_find_preserves_scipy_order_for_supported_names() {
    let query = OracleQuery {
        points: Vec::new(),
        finds: vec![FindCase {
            case_id: "find_g_factor_order".into(),
            query: "g factor".into(),
            scipy_key: "electron g factor".into(),
        }],
    };
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    let scipy_arm = oracle
        .finds
        .into_iter()
        .find(|arm| arm.case_id == "find_g_factor_order")
        .expect("validated constants.find order oracle");
    let supported = [
        "electron g factor",
        "muon g factor",
        "neutron g factor",
        "proton g factor",
    ];
    let scipy_supported: Vec<&str> = scipy_arm
        .names
        .iter()
        .map(String::as_str)
        .filter(|name| supported.contains(name))
        .collect();
    let fsci_supported: Vec<&str> = fc::find("g factor")
        .into_iter()
        .map(|(name, _)| name)
        .filter(|name| supported.contains(name))
        .collect();

    assert_eq!(
        fsci_supported, scipy_supported,
        "constants.find should preserve scipy's sorted result order for supported keys"
    );
}
