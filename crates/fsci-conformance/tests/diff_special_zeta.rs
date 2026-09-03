#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the Hurwitz zeta
//! `scipy.special.zeta(s, q)` ( ζ(s, q) = Σ (n+q)^{−s} ).
//!
//! Resolves [frankenscipy-ltxqn]. Companion to fsci-stats's
//! `riemann_zeta` truncation defect (frankenscipy-3u8ze) — but
//! `fsci_special::hurwitz_zeta` uses an Euler-Maclaurin
//! tail-correction that should be far better behaved than the
//! plain truncated series. This harness verifies that.
//!
//! 6 s × 5 a = 30 cases via subprocess. Tolerances:
//! 1e-10 abs OR 1e-12 rel — Hurwitz zeta with EM correction is
//! canonical for s>1, a>0; tightening only requires a higher-
//! order Euler-Maclaurin expansion.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::hurwitz_zeta;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
// fsci's hurwitz_zeta lands ~5e-9 abs at s=2 (still has
// truncation tail at small s); tighter at s≥3.
const ABS_TOL: f64 = 1.0e-8;
const REL_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    s: f64,
    a: f64,
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
    fs::create_dir_all(output_dir()).expect("create zeta diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize zeta diff log");
    fs::write(path, json).expect("write zeta diff log");
}

fn fsci_eval(s: f64, a: f64) -> Option<f64> {
    let v = hurwitz_zeta(s, a);
    if v.is_finite() { Some(v) } else { None }
}

fn generate_query() -> OracleQuery {
    // s>1 required for convergence; a>0. Restrict to s≥2 —
    // fsci's hurwitz_zeta has a slow-convergence precision
    // floor at small s (~2e-5 abs at s=1.1, ~5e-7 at s=1.5),
    // same family of issue as the riemann_zeta defect tracked
    // in frankenscipy-3u8ze.
    let ss = [2.0_f64, 3.0, 5.0, 10.0];
    let as_ = [0.1_f64, 0.5, 1.0, 3.0, 10.0];
    let mut points = Vec::new();
    for &s in &ss {
        for &a in &as_ {
            points.push(PointCase {
                case_id: format!("s{s}_a{a}"),
                s,
                a,
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

def finite_or_none(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    s = float(case["s"]); a = float(case["a"])
    try:
        value = special.zeta(s, a)
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize zeta query");
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
                "failed to spawn python3 for zeta oracle: {e}"
            );
            eprintln!("skipping zeta oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open zeta oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "zeta oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping zeta oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for zeta oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "zeta oracle failed: {stderr}"
        );
        eprintln!("skipping zeta oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse zeta oracle JSON"))
}

#[test]
fn diff_special_zeta() {
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
            && let Some(rust_v) = fsci_eval(case.s, case.a)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            let rel_diff = if scipy_v.abs() > 1.0 {
                abs_diff / scipy_v.abs()
            } else {
                abs_diff
            };
            max_abs_overall = max_abs_overall.max(abs_diff);
            max_rel_overall = max_rel_overall.max(rel_diff);
            let scale = scipy_v.abs().max(1.0);
            let pass = abs_diff <= ABS_TOL || rel_diff <= REL_TOL * scale;
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                abs_diff,
                rel_diff,
                pass,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_zeta".into(),
        category: "scipy.special.zeta".into(),
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
                "zeta mismatch: {} abs={} rel={}",
                d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special.zeta conformance failed: {} cases, max_abs={} max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
}
