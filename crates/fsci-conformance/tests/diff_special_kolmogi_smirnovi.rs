#![forbid(unsafe_code)]
//! Live scipy.special.{kolmogi, smirnovi} parity for fsci_special.
//!
//! Resolves [frankenscipy-p9u4z]. Tolerances:
//!   - kolmogi: 1e-9 abs (canonical inverse series, tight)
//!   - smirnovi: 5e-3 abs (companion to smirnov's known ~3e-2
//!     asymptotic floor; smirnov defect tracked separately).

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_runtime::RuntimeMode;
use fsci_special::types::SpecialTensor;
use fsci_special::{kolmogi, smirnovi};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const KOLMOGI_TOL: f64 = 1.0e-9;
const SMIRNOVI_TOL: f64 = 5.0e-3;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
/// One ledger arm per op compared.
const ARMS: [&str; 2] = ["kolmogi", "smirnovi"];

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "kolmogi" | "smirnovi"
    n: i32,     // smirnovi only
    p: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
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
    fs::create_dir_all(output_dir()).expect("create ki/si diff dir");
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

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    // p = 0.5 included since the safeguarded Newton-bisection fix
    // (frankenscipy-or0dc); a bare Newton previously diverged there.
    let kolmogi_ps = [0.01_f64, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99];
    for &p in &kolmogi_ps {
        points.push(Case {
            case_id: format!("kolmogi_p{p}").replace('.', "p"),
            op: "kolmogi".into(),
            n: 0,
            p,
        });
    }
    let ps = [0.01_f64, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99];
    let ns = [20_i32, 50, 100, 200];
    for &n in &ns {
        for &p in &ps {
            points.push(Case {
                case_id: format!("smirnovi_n{n}_p{p}").replace('.', "p"),
                op: "smirnovi".into(),
                n,
                p,
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
from scipy import special

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    p = float(case["p"]); n = int(case["n"])
    try:
        if op == "kolmogi":
            v = float(special.kolmogi(p))
        elif op == "smirnovi":
            v = float(special.smirnovi(n, p))
        else:
            v = float("nan")
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
                "failed to spawn python3 for ki/si oracle: {e}"
            );
            eprintln!("skipping ki/si oracle: python3 not available ({e})");
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
                "ki/si oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping ki/si oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for ki/si oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "ki/si oracle failed: {stderr}"
        );
        eprintln!("skipping ki/si oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse ki/si oracle JSON"))
}

#[test]
fn diff_special_kolmogi_smirnovi() {
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
    let mut ledger = CompareLedger::new("diff_special_kolmogi_smirnovi", &ARMS);

    for case in &query.points {
        let scipy = pmap.get(&case.case_id).and_then(|a| a.value);
        let (fsci, tol) = match case.op.as_str() {
            "kolmogi" => {
                let pt = SpecialTensor::RealScalar(case.p);
                let v = match kolmogi(&pt, RuntimeMode::Strict) {
                    Ok(SpecialTensor::RealScalar(v)) => Some(v),
                    _ => None,
                };
                (v, KOLMOGI_TOL)
            }
            "smirnovi" => (Some(smirnovi(case.n, case.p)), SMIRNOVI_TOL),
            other => panic!("unknown op {other}"),
        };
        let Some((expected, actual)) = ledger.pair(&case.op, &case.case_id, scipy, fsci) else {
            continue;
        };
        let abs_d = (actual - expected).abs();
        max_overall = max_overall.max(abs_d);
        ledger.compared(&case.op, &case.case_id, abs_d <= tol);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            abs_diff: abs_d,
            pass: abs_d <= tol,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_kolmogi_smirnovi".into(),
        category: "fsci_special::kolmogi + smirnovi vs scipy.special".into(),
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
            eprintln!("{} mismatch: {} abs_diff={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "ki/si conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    // Arms have different case sets (kolmogi has the fewest); each must compare all of its own.
    let min_per_arm = ARMS
        .iter()
        .map(|arm| query.points.iter().filter(|c| c.op == *arm).count())
        .min()
        .expect("ARMS is non-empty");
    ledger.finish(min_per_arm);
}
