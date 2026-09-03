#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the normal-distribution
//! solver wrappers `scipy.special.nrdtrimn` (recover mean from
//! p, std, x) and `nrdtrisd` (recover std from mn, p, x).
//!
//! Resolves [frankenscipy-53lgi]. These are scipy's "solve for
//! one parameter" inverses of the normal cdf — composes ndtri
//! (Beasley-Springer-Moro rational).
//!
//! 6 (p, std, x) triples × 2 funcs = 12 cases via subprocess.
//! Tol 1e-7 rel.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{nrdtrimn, nrdtrisd};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REL_TOL: f64 = 1.0e-7;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    p: f64,
    std: f64,
    mn: f64,
    x: f64,
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
    func: String,
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
    fs::create_dir_all(output_dir()).expect("create nrdtr diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize nrdtr diff log");
    fs::write(path, json).expect("write nrdtr diff log");
}

fn fsci_eval(case: &PointCase) -> Option<f64> {
    let v = match case.func.as_str() {
        "nrdtrimn" => nrdtrimn(case.p, case.std, case.x),
        "nrdtrisd" => nrdtrisd(case.mn, case.p, case.x),
        _ => return None,
    };
    if v.is_finite() { Some(v) } else { None }
}

fn generate_query() -> OracleQuery {
    let triples = [
        (0.5_f64, 1.0, 0.0),
        (0.25, 1.0, -1.0),
        (0.75, 1.0, 1.0),
        (0.1, 2.0, 5.0),
        (0.9, 0.5, 3.0),
        (0.99, 1.5, 7.0),
    ];
    let mut points = Vec::new();
    for (i, &(p, sd, x)) in triples.iter().enumerate() {
        // nrdtrimn: solve for mean given (p, std, x)
        points.push(PointCase {
            case_id: format!("nrdtrimn_p{p}_s{sd}_x{x}_i{i}"),
            func: "nrdtrimn".into(),
            p,
            std: sd,
            mn: 0.0,
            x,
        });
        // nrdtrisd: solve for std given (mean, p, x). Pick mn
        // such that x > mean for p > 0.5 etc.
        let mn = if p < 0.5 { x + 1.0 } else { x - 1.0 };
        points.push(PointCase {
            case_id: format!("nrdtrisd_m{mn}_p{p}_x{x}_i{i}"),
            func: "nrdtrisd".into(),
            p,
            std: 0.0,
            mn,
            x,
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

def finite_or_none(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; func = case["func"]
    p = float(case["p"]); sd = float(case["std"]); mn = float(case["mn"]); x = float(case["x"])
    try:
        if func == "nrdtrimn":   value = special.nrdtrimn(p, sd, x)
        elif func == "nrdtrisd": value = special.nrdtrisd(mn, p, x)
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize nrdtr query");
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
                "failed to spawn python3 for nrdtr oracle: {e}"
            );
            eprintln!("skipping nrdtr oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open nrdtr oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "nrdtr oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping nrdtr oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for nrdtr oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "nrdtr oracle failed: {stderr}"
        );
        eprintln!("skipping nrdtr oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse nrdtr oracle JSON"))
}

#[test]
fn diff_special_nrdtr() {
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
    let mut max_abs_overall = 0.0_f64;
    let mut max_rel_overall = 0.0_f64;

    for case in &query.points {
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        if let Some(scipy_v) = oracle.value
            && let Some(rust_v) = fsci_eval(case)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            let scale = scipy_v.abs().max(1.0);
            let rel_diff = abs_diff / scale;
            max_abs_overall = max_abs_overall.max(abs_diff);
            max_rel_overall = max_rel_overall.max(rel_diff);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                abs_diff,
                rel_diff,
                pass: abs_diff <= REL_TOL * scale,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_nrdtr".into(),
        category: "scipy.special.nrdtrimn/nrdtrisd".into(),
        case_count: diffs.len(),
        max_abs_diff: max_abs_overall,
        max_rel_diff: max_rel_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };

    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "nrdtr {} mismatch: {} abs={} rel={}",
                d.func, d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "nrdtr conformance failed: {} cases, max_abs={} max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
}
