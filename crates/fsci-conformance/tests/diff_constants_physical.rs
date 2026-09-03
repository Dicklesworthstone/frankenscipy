#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_constants top-level CODATA
//! constants. Uses relative tolerance because scipy may have moved to
//! CODATA-2022 while fsci ships CODATA-2018 values for particle masses.
//!
//! Resolves [frankenscipy-ff74z].

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
    /// scipy.constants attribute name.
    scipy_name: String,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
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
    fs::create_dir_all(output_dir()).expect("create constants_physical diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize constants_physical diff log");
    fs::write(path, json).expect("write constants_physical diff log");
}

fn entries() -> Vec<(&'static str, &'static str, f64)> {
    vec![
        ("speed_of_light", "speed_of_light", fc::SPEED_OF_LIGHT),
        ("planck_h", "h", fc::PLANCK),
        ("planck_hbar", "hbar", fc::HBAR),
        ("gravitational_constant", "G", fc::GRAVITATIONAL_CONSTANT),
        ("g_n", "g", fc::G_N),
        ("elementary_charge", "e", fc::ELEMENTARY_CHARGE),
        ("epsilon_0", "epsilon_0", fc::EPSILON_0),
        ("mu_0", "mu_0", fc::MU_0),
        ("gas_constant", "R", fc::GAS_CONSTANT),
        ("avogadro", "N_A", fc::AVOGADRO),
        ("boltzmann", "Boltzmann", fc::BOLTZMANN),
        ("stefan_boltzmann", "Stefan_Boltzmann", fc::STEFAN_BOLTZMANN),
        ("wien", "Wien", fc::WIEN),
        ("electron_mass", "m_e", fc::ELECTRON_MASS),
        ("proton_mass", "m_p", fc::PROTON_MASS),
        ("neutron_mass", "m_n", fc::NEUTRON_MASS),
        ("rydberg", "Rydberg", fc::RYDBERG),
    ]
}

fn generate_query() -> OracleQuery {
    OracleQuery {
        points: entries()
            .iter()
            .map(|(case_id, sname, _)| PointCase {
                case_id: (*case_id).into(),
                scipy_name: (*sname).into(),
            })
            .collect(),
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy import constants

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; sname = case["scipy_name"]
    try:
        v = getattr(constants, sname)
        v = float(v)
        if not math.isfinite(v):
            points.append({"case_id": cid, "value": None})
        else:
            points.append({"case_id": cid, "value": v})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize constants_physical query");
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
                "failed to spawn python3 for constants_physical oracle: {e}"
            );
            eprintln!("skipping constants_physical oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open constants_physical oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "constants_physical oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping constants_physical oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for constants_physical oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "constants_physical oracle failed: {stderr}"
        );
        eprintln!("skipping constants_physical oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse constants_physical oracle JSON"))
}

#[test]
fn diff_constants_physical() {
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

    let entries = entries();
    let entry_map: HashMap<String, f64> = entries
        .iter()
        .map(|(case_id, _, v)| ((*case_id).to_string(), *v))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_v) = scipy_arm.value else {
            continue;
        };
        let Some(&fsci_v) = entry_map.get(&case.case_id) else {
            continue;
        };
        let rel = if scipy_v.abs() > 0.0 {
            (fsci_v - scipy_v).abs() / scipy_v.abs()
        } else {
            (fsci_v - scipy_v).abs()
        };
        max_overall = max_overall.max(rel);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            rel_diff: rel,
            pass: rel <= REL_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_constants_physical".into(),
        category: "scipy.constants fundamental CODATA values".into(),
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
            eprintln!("constants mismatch: {} rel_diff={}", d.case_id, d.rel_diff);
        }
    }

    assert!(
        all_pass,
        "constants_physical conformance failed: {} cases, max_rel_diff={}",
        diffs.len(),
        max_overall
    );
}
