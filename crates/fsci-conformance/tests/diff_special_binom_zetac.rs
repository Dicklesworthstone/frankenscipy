#![forbid(unsafe_code)]
//! Live SciPy differential coverage for fsci_special::binom (real-valued
//! binomial coefficient) and zetac (Riemann zeta minus 1).
//!
//! Resolves [frankenscipy-2om4f]. 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{binom, zetac_scalar};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-10;
const REL_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    a: f64,
    b: f64,
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
    fs::create_dir_all(output_dir()).expect("create binom_zetac diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize binom_zetac diff log");
    fs::write(path, json).expect("write binom_zetac diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    let binom_pairs: &[(f64, f64)] = &[
        (5.0, 2.0),
        (10.0, 3.0),
        (10.0, 5.0),
        (20.0, 7.0),
        (5.5, 2.5),
        (7.5, 3.5),
        (3.0, 0.0),
        (8.0, 8.0),
    ];
    for &(x, y) in binom_pairs {
        points.push(PointCase {
            case_id: format!("binom_{x}_{y}"),
            op: "binom".into(),
            a: x,
            b: y,
        });
    }

    // Covers s > 1, the critical strip 0 < s < 1 (Borwein eta evaluation,
    // frankenscipy-kk4vu), s = 0, and the reflected region s < 0.
    let zetac_xs = [
        2.0_f64, 3.0, 4.0, 5.0, 6.0, 1.5, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0, -1.0, -2.0, -3.0, 10.0,
    ];
    for s in zetac_xs {
        points.push(PointCase {
            case_id: format!("zetac_{s}"),
            op: "zetac".into(),
            a: s,
            b: 0.0,
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

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    a = float(case["a"]); b = float(case["b"])
    try:
        if op == "binom":
            v = float(special.binom(a, b))
        elif op == "zetac":
            v = float(special.zetac(a))
        else:
            v = None
        if v is None or not math.isfinite(v):
            points.append({"case_id": cid, "value": None})
        else:
            points.append({"case_id": cid, "value": v})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize binom_zetac query");
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
                "failed to spawn python3 for binom_zetac oracle: {e}"
            );
            eprintln!("skipping binom_zetac oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open binom_zetac oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "binom_zetac oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping binom_zetac oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for binom_zetac oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "binom_zetac oracle failed: {stderr}"
        );
        eprintln!("skipping binom_zetac oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse binom_zetac oracle JSON"))
}

#[test]
fn diff_special_binom_zetac() {
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
        let Some(expected) = scipy_arm.value else {
            continue;
        };
        let fsci_v = match case.op.as_str() {
            "binom" => binom(case.a, case.b),
            "zetac" => zetac_scalar(case.a),
            _ => continue,
        };
        let abs_d = (fsci_v - expected).abs();
        let rel = if expected.abs() > 0.0 {
            abs_d / expected.abs()
        } else {
            abs_d
        };
        let pass = abs_d <= ABS_TOL || rel <= REL_TOL;
        max_overall = max_overall.max(rel);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            rel_diff: rel,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_binom_zetac".into(),
        category: "scipy.special binom + zetac".into(),
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
        "binom_zetac conformance failed: {} cases, max_rel_diff={}",
        diffs.len(),
        max_overall
    );
}
