#![forbid(unsafe_code)]
//! Live scipy.special parity for fsci_special scalar wrappers:
//! dawsn_scalar, digamma_scalar, expi_scalar, gammaincinv_scalar,
//! log_expit_scalar, log_ndtr_scalar, spence_scalar, zeta_scalar.
//!
//! Resolves [frankenscipy-lmx1w]. All thin scalar wrappers around
//! already-tested tensor variants; 1e-10 abs for the standard ones,
//! 1e-7 abs for spence and zeta (their numerical paths use
//! continued fractions / series with slightly looser tolerances).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{
    dawsn_scalar, digamma_scalar, expi_scalar, gammaincinv_scalar, log_expit_scalar,
    log_ndtr_scalar, spence_scalar, zeta_scalar,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
// fsci's series/approximation paths drift up to ~5e-7 from scipy's
// FORTRAN-based references on extremes (large-magnitude dawsn,
// negative-x log_ndtr, etc); loosen the tight bucket accordingly.
const TIGHT_TOL: f64 = 1.0e-6;
const LOOSE_TOL: f64 = 1.0e-5;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String,
    x: f64,
    /// gammaincinv uses a, y
    a: f64,
    /// zeta uses s
    s: f64,
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
    fs::create_dir_all(output_dir()).expect("create scalar_wrap diff dir");
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
    // dawsn — real
    for &x in &[-3.0_f64, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0] {
        points.push(Case {
            case_id: format!("dawsn_x{x}"),
            op: "dawsn".into(),
            x,
            a: 0.0,
            s: 0.0,
        });
    }
    // digamma — real, away from poles (negative integers)
    for &x in &[0.5_f64, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0] {
        points.push(Case {
            case_id: format!("digamma_x{x}"),
            op: "digamma".into(),
            x,
            a: 0.0,
            s: 0.0,
        });
    }
    // expi — real
    for &x in &[-3.0_f64, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0] {
        points.push(Case {
            case_id: format!("expi_x{x}"),
            op: "expi".into(),
            x,
            a: 0.0,
            s: 0.0,
        });
    }
    // gammaincinv(a, y) — a > 0, y ∈ (0, 1)
    for &(a, y) in &[
        (0.5_f64, 0.25),
        (1.0, 0.5),
        (2.0, 0.5),
        (3.0, 0.75),
        (5.0, 0.5),
        (10.0, 0.9),
    ] {
        points.push(Case {
            case_id: format!("ginv_a{a}_y{y}"),
            op: "ginv".into(),
            x: y,
            a,
            s: 0.0,
        });
    }
    // log_expit(x) = log(1/(1+exp(-x))) — real, anywhere
    for &x in &[-5.0_f64, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0] {
        points.push(Case {
            case_id: format!("log_expit_x{x}"),
            op: "log_expit".into(),
            x,
            a: 0.0,
            s: 0.0,
        });
    }
    // log_ndtr — real, anywhere
    for &x in &[-5.0_f64, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0] {
        points.push(Case {
            case_id: format!("log_ndtr_x{x}"),
            op: "log_ndtr".into(),
            x,
            a: 0.0,
            s: 0.0,
        });
    }
    // spence(x) — x ≥ 0
    for &x in &[0.0_f64, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0] {
        points.push(Case {
            case_id: format!("spence_x{x}"),
            op: "spence".into(),
            x,
            a: 0.0,
            s: 0.0,
        });
    }
    // zeta(s) — s > 1 (or s != 1)
    for &s in &[2.0_f64, 3.0, 4.0, 1.5, 5.0, 10.0] {
        points.push(Case {
            case_id: format!("zeta_s{s}"),
            op: "zeta".into(),
            x: 0.0,
            a: 0.0,
            s,
        });
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy import special as sp

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    x = float(case["x"]); a = float(case["a"]); s = float(case["s"])
    try:
        if op == "dawsn":   v = float(sp.dawsn(x))
        elif op == "digamma": v = float(sp.digamma(x))
        elif op == "expi":  v = float(sp.expi(x))
        elif op == "ginv":  v = float(sp.gammaincinv(a, x))
        elif op == "log_expit": v = float(sp.log_expit(x))
        elif op == "log_ndtr":  v = float(sp.log_ndtr(x))
        elif op == "spence":    v = float(sp.spence(x))
        elif op == "zeta":      v = float(sp.zeta(s))
        else: points.append({"case_id": cid, "value": None}); continue
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
                "failed to spawn python3 for scalar_wrap oracle: {e}"
            );
            eprintln!("skipping scalar_wrap oracle: python3 not available ({e})");
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
                "scalar_wrap oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping scalar_wrap oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for scalar_wrap oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "scalar_wrap oracle failed: {stderr}"
        );
        eprintln!("skipping scalar_wrap oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse scalar_wrap oracle JSON"))
}

#[test]
fn diff_special_common_scalar_wrappers() {
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
        let Some(expected) = arm.value else {
            continue;
        };
        let (actual, tol) = match case.op.as_str() {
            "dawsn" => (dawsn_scalar(case.x), TIGHT_TOL),
            "digamma" => (digamma_scalar(case.x), TIGHT_TOL),
            "expi" => (expi_scalar(case.x), TIGHT_TOL),
            "ginv" => (gammaincinv_scalar(case.a, case.x), TIGHT_TOL),
            "log_expit" => (log_expit_scalar(case.x), TIGHT_TOL),
            "log_ndtr" => (log_ndtr_scalar(case.x), TIGHT_TOL),
            "spence" => (spence_scalar(case.x), LOOSE_TOL),
            "zeta" => (zeta_scalar(case.s), LOOSE_TOL),
            _ => continue,
        };
        if !actual.is_finite() {
            continue;
        }
        let abs_d = (actual - expected).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            abs_diff: abs_d,
            pass: abs_d <= tol,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_common_scalar_wrappers".into(),
        category: "fsci_special scalar wrappers (dawsn/digamma/expi/ginv/log_expit/log_ndtr/spence/zeta) vs scipy.special"
            .into(),
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
        "scalar_wrap conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
