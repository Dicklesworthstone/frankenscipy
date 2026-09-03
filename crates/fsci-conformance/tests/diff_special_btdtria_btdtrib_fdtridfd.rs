#![forbid(unsafe_code)]
//! Live scipy parity for fsci_special::{btdtria, btdtrib, fdtridfd}.
//!
//! Resolves [frankenscipy-rmrc9]. These three routines invert the beta-
//! and F-distribution CDFs with respect to a shape parameter (a, b, or
//! dfd respectively). We compare directly against scipy.special and
//! also verify round-trip consistency:
//!   * btdtr(btdtria(p, b, x), b, x) ≈ p
//!   * btdtr(a, btdtrib(a, p, x), x) ≈ p
//!   * fdtr(dfn, fdtridfd(dfn, p, x), x) ≈ p
//!
//! The round-trip residual is the more meaningful invariant since
//! both fsci and scipy use iterative root-finders; the absolute
//! parameter value can drift more than the CDF it solves for.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{btdtr, btdtria, btdtrib, fdtr, fdtridfd};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REL_TOL: f64 = 1.0e-4;
const ROUND_TRIP_TOL: f64 = 1.0e-6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct CasePoint {
    case_id: String,
    func: String,
    /// p (for btdtria) or a (for btdtrib) or dfn (for fdtridfd)
    arg1: f64,
    /// b (for btdtria) or p (for btdtrib) or p (for fdtridfd)
    arg2: f64,
    /// x for all three
    arg3: f64,
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
    func: String,
    actual: f64,
    expected: f64,
    rel_diff: f64,
    round_trip_err: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create btdtria diff dir");
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

fn build_query() -> OracleQuery {
    let mut pts = Vec::new();

    // btdtria(p, b, x): invert for a.  Choose p ∈ (0,1)\{0.5} to exercise
    // the small-tail and big-tail branches both; x ∈ (0, 1); b > 0.
    let p_grid = [0.1_f64, 0.25, 0.4, 0.6, 0.75, 0.9];
    let b_grid = [0.5_f64, 1.0, 2.0, 5.0];
    let x_grid = [0.2_f64, 0.5, 0.8];
    for &p in &p_grid {
        for &b in &b_grid {
            for &x in &x_grid {
                pts.push(CasePoint {
                    case_id: format!("btdtria_p{p}_b{b}_x{x}"),
                    func: "btdtria".into(),
                    arg1: p,
                    arg2: b,
                    arg3: x,
                });
            }
        }
    }

    // btdtrib(a, p, x): invert for b
    let a_grid = [0.5_f64, 1.0, 2.0, 5.0];
    for &a in &a_grid {
        for &p in &p_grid {
            for &x in &x_grid {
                pts.push(CasePoint {
                    case_id: format!("btdtrib_a{a}_p{p}_x{x}"),
                    func: "btdtrib".into(),
                    arg1: a,
                    arg2: p,
                    arg3: x,
                });
            }
        }
    }

    // fdtridfd(dfn, p, x): invert for dfd.  Use moderate dfn/x and
    // p in the strictly-interior range — boundary cases (p == 0, x == 0,
    // x == inf) return CDFlib sentinel values that aren't bit-equal to
    // scipy, so we focus on the interior numerical root-find here.
    let dfn_grid = [1.0_f64, 2.0, 5.0, 10.0];
    let p_grid_f = [0.25_f64, 0.5, 0.75, 0.9];
    let x_grid_f = [0.5_f64, 1.0, 2.0, 5.0];
    for &dfn in &dfn_grid {
        for &p in &p_grid_f {
            for &x in &x_grid_f {
                pts.push(CasePoint {
                    case_id: format!("fdtridfd_dfn{dfn}_p{p}_x{x}"),
                    func: "fdtridfd".into(),
                    arg1: dfn,
                    arg2: p,
                    arg3: x,
                });
            }
        }
    }

    OracleQuery { points: pts }
}

fn scipy_oracle_or_skip(q: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json, math, sys
from scipy import special

q = json.load(sys.stdin)
out = []
for c in q["points"]:
    cid = c["case_id"]; fn = c["func"]
    a1, a2, a3 = c["arg1"], c["arg2"], c["arg3"]
    try:
        if fn == "btdtria":
            v = float(special.btdtria(a1, a2, a3))
        elif fn == "btdtrib":
            v = float(special.btdtrib(a1, a2, a3))
        elif fn == "fdtridfd":
            v = float(special.fdtridfd(a1, a2, a3))
        else:
            v = None
        if v is None or not math.isfinite(v):
            out.append({"case_id": cid, "value": None})
        else:
            out.append({"case_id": cid, "value": v})
    except Exception:
        out.append({"case_id": cid, "value": None})

print(json.dumps({"points": out}))
"#;
    let query_json = serde_json::to_string(q).expect("serialize");
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
                "python3 spawn failed: {e}"
            );
            eprintln!("skipping btdtria oracle: python3 unavailable ({e})");
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
            eprintln!("skipping btdtria oracle: stdin write failed");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "oracle failed: {stderr}"
        );
        eprintln!("skipping btdtria oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse JSON"))
}

#[test]
fn diff_special_btdtria_btdtrib_fdtridfd() {
    let query = build_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());

    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();

    for (c, o) in query.points.iter().zip(oracle.points.iter()) {
        assert_eq!(c.case_id, o.case_id);
        let Some(expected) = o.value else {
            continue;
        };

        let actual = match c.func.as_str() {
            "btdtria" => btdtria(c.arg1, c.arg2, c.arg3),
            "btdtrib" => btdtrib(c.arg1, c.arg2, c.arg3),
            "fdtridfd" => fdtridfd(c.arg1, c.arg2, c.arg3),
            other => panic!("unknown func {other}"),
        };

        // Round-trip residual
        let round_trip_err = match c.func.as_str() {
            "btdtria" => {
                // p target = c.arg1; btdtr(actual, b=c.arg2, x=c.arg3) should ≈ p
                let pr = btdtr(actual, c.arg2, c.arg3);
                (pr - c.arg1).abs()
            }
            "btdtrib" => {
                // p target = c.arg2; btdtr(a=c.arg1, actual, x=c.arg3) should ≈ p
                let pr = btdtr(c.arg1, actual, c.arg3);
                (pr - c.arg2).abs()
            }
            "fdtridfd" => {
                // p target = c.arg2; fdtr(dfn=c.arg1, actual, x=c.arg3) should ≈ p
                let pr = fdtr(c.arg1, actual, c.arg3);
                (pr - c.arg2).abs()
            }
            _ => f64::NAN,
        };

        let denom = expected.abs().max(1.0e-300);
        let rel_diff = (actual - expected).abs() / denom;
        // Parity passes if either rel_diff is small OR round-trip is small.
        // (Iterative inverse solvers in scipy use CDFlib; tiny differences
        // in the parameter are OK if the inverted CDF lands on the right p.)
        let pass = rel_diff <= REL_TOL || round_trip_err <= ROUND_TRIP_TOL;
        diffs.push(CaseDiff {
            case_id: c.case_id.clone(),
            func: c.func.clone(),
            actual,
            expected,
            rel_diff,
            round_trip_err,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_special_btdtria_btdtrib_fdtridfd".into(),
        category: "scipy.special inverse-of-shape-parameter solvers".into(),
        case_count: diffs.len(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "btdtria/btdtrib/fdtridfd mismatch: {} ({}) actual={} expected={} rel={} round_trip_err={}",
                d.case_id, d.func, d.actual, d.expected, d.rel_diff, d.round_trip_err,
            );
        }
    }

    assert!(
        all_pass,
        "btdtria/btdtrib/fdtridfd parity failed: {} cases",
        diffs.len()
    );
}
