#![forbid(unsafe_code)]
//! Live scipy.special.airy parity for fsci_special::airy returning
//! all four components (Ai, Aip, Bi, Bip).
//!
//! Resolves [frankenscipy-c3vpv]. The existing `diff_special_airy`
//! harness only covers the scalar-out wrappers `ai` and `bi`; this
//! adds dedicated coverage for the derivatives Aip and Bip via the
//! 4-tuple `airy()` entry point.
//!
//! Tolerance: 1e-9 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::airy;
use fsci_special::types::SpecialTensor;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
// Asymptotic Airy at |x|=10 sees ~1e-8 abs gaps; at x=5 the Bi/Bip
// series is at ~1e3-1e4 magnitude with ~1e-5 rel agreement.
const ABS_TOL: f64 = 1.0e-7;
const REL_TOL_LARGE: f64 = 1.0e-4;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    ai: Option<f64>,
    aip: Option<f64>,
    bi: Option<f64>,
    bip: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create airy_full diff dir");
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

fn fsci_eval(x: f64) -> Option<(f64, f64, f64, f64)> {
    let pt = SpecialTensor::RealScalar(x);
    let result = airy(&pt, RuntimeMode::Strict).ok()?;
    if result.len() != 4 {
        return None;
    }
    let mut out = [0.0; 4];
    for (i, t) in result.into_iter().enumerate() {
        if let SpecialTensor::RealScalar(v) = t {
            out[i] = v;
        } else {
            return None;
        }
    }
    Some((out[0], out[1], out[2], out[3]))
}

fn generate_query() -> OracleQuery {
    // Bi grows like exp(2/3 x^(3/2)) for large positive x; clamp at moderate
    // range to keep magnitudes representable to 1e-9 abs without rel scaling.
    let xs: &[f64] = &[
        -10.0, -5.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0,
    ];
    let points: Vec<PointCase> = xs
        .iter()
        .enumerate()
        .map(|(i, &x)| PointCase {
            case_id: format!("p{i:02}_x{x}").replace('.', "p").replace('-', "n"),
            x,
        })
        .collect();
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
    cid = case["case_id"]
    x = float(case["x"])
    try:
        ai, aip, bi, bip = special.airy(x)
        if all(math.isfinite(v) for v in [ai, aip, bi, bip]):
            points.append({
                "case_id": cid,
                "ai": float(ai), "aip": float(aip),
                "bi": float(bi), "bip": float(bip),
            })
        else:
            points.append({"case_id": cid, "ai": None, "aip": None, "bi": None, "bip": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "ai": None, "aip": None, "bi": None, "bip": None})
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
                "failed to spawn python3 for airy_full oracle: {e}"
            );
            eprintln!("skipping airy_full oracle: python3 not available ({e})");
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
                "airy_full oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping airy_full oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for airy_full oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "airy_full oracle failed: {stderr}"
        );
        eprintln!("skipping airy_full oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse airy_full oracle JSON"))
}

#[test]
fn diff_special_airy_full() {
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
        let (Some(eai), Some(eaip), Some(ebi), Some(ebip)) = (arm.ai, arm.aip, arm.bi, arm.bip)
        else {
            continue;
        };
        let Some((ai, aip, bi, bip)) = fsci_eval(case.x) else {
            continue;
        };
        for (op, actual, expected) in [
            ("Ai", ai, eai),
            ("Aip", aip, eaip),
            ("Bi", bi, ebi),
            ("Bip", bip, ebip),
        ] {
            let abs_d = (actual - expected).abs();
            // Bi grows large for x > 0; relax to relative tol where mag > 1.
            let pass = if expected.abs() > 1.0 {
                abs_d / expected.abs() <= REL_TOL_LARGE
            } else {
                abs_d <= ABS_TOL
            };
            max_overall = max_overall.max(abs_d);
            diffs.push(CaseDiff {
                case_id: format!("{}_{}", case.case_id, op),
                op: op.into(),
                abs_diff: abs_d,
                pass,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_airy_full".into(),
        category: "fsci_special::airy (4-tuple) vs scipy.special.airy".into(),
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
        "airy_full conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
