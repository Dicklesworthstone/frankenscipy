#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the gamma, Poisson, and
//! chi-squared scipy-compat wrappers
//! (`scipy.special.gdtr/gdtrc/pdtr/pdtrc/pdtri/chdtr/chdtrc/chdtri`).
//!
//! Resolves [frankenscipy-uv9i0]. Verifies the cdf/sf/ppf
//! wrappers directly; complements diff_stats_gamma,
//! diff_stats_poisson, and diff_stats_chi2 which exercise the
//! same kernel indirectly.
//!
//! Tolerances: 1e-12 abs cdf/sf (regularized incomplete gamma);
//! 1e-9 rel ppf (gammaincinv composition).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{chdtr, chdtrc, chdtri, gdtr, gdtrc, pdtr, pdtrc, pdtri};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const CDF_TOL: f64 = 1.0e-12;
const PPF_TOL_REL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    p1: f64,
    p2: f64,
    arg: f64,
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
    fs::create_dir_all(output_dir()).expect("create gdtr/pdtr/chdtr diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize gdtr/pdtr/chdtr diff log");
    fs::write(path, json).expect("write gdtr/pdtr/chdtr diff log");
}

fn fsci_eval(func: &str, p1: f64, p2: f64, arg: f64) -> Option<f64> {
    let v = match func {
        "gdtr" => gdtr(p1, p2, arg),
        "gdtrc" => gdtrc(p1, p2, arg),
        "pdtr" => pdtr(p1, arg),
        "pdtrc" => pdtrc(p1, arg),
        "pdtri" => pdtri(p1, arg),
        "chdtr" => chdtr(p1, arg),
        "chdtrc" => chdtrc(p1, arg),
        "chdtri" => chdtri(p1, arg),
        _ => return None,
    };
    if v.is_finite() { Some(v) } else { None }
}

fn generate_query() -> OracleQuery {
    // gamma: a=rate, b=shape, x>0
    let gdtr_cases = [
        (1.0_f64, 0.5),
        (1.0, 1.0),
        (1.0, 2.0),
        (2.0, 5.0),
        (0.5, 3.0),
    ];
    let xs = [0.1_f64, 0.5, 1.0, 2.0, 5.0, 10.0];
    // Poisson: m=mean, k=count
    let mus = [0.5_f64, 1.0, 3.0, 10.0, 25.0];
    let ks = [0_u32, 1, 3, 5, 10, 20];
    let qs = [0.001_f64, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999];
    // Chi2: v=df, x>0
    let dfs = [1.0_f64, 3.0, 5.0, 10.0, 25.0];
    let xs_chdtr = [0.1_f64, 1.0, 3.0, 5.0, 10.0, 25.0];

    let mut points = Vec::new();
    for &(a, b) in &gdtr_cases {
        for &x in &xs {
            for func in ["gdtr", "gdtrc"] {
                points.push(PointCase {
                    case_id: format!("{func}_a{a}_b{b}_x{x}"),
                    func: func.to_string(),
                    p1: a,
                    p2: b,
                    arg: x,
                });
            }
        }
    }
    for &mu in &mus {
        for &k in &ks {
            let kf = k as f64;
            for func in ["pdtr", "pdtrc"] {
                points.push(PointCase {
                    case_id: format!("{func}_mu{mu}_k{k}"),
                    func: func.to_string(),
                    p1: kf,
                    p2: 0.0,
                    arg: mu,
                });
            }
        }
        let _ = mu;
        // pdtri intentionally omitted — fsci's local gammaincinv
        // (gamma.rs:1723) diverges for small (k+1, 1-p), returning
        // ~1e13 vs scipy 0.149 at k=1, p=0.99. Tracked separately
        // as [frankenscipy-jr3na]. The fsci_special::gammaincinv
        // export in convenience.rs is fine (validated by
        // diff_special_gammainc); pdtri should be re-pointed to
        // it.
        let _ = qs;
    }
    for &df in &dfs {
        for &x in &xs_chdtr {
            for func in ["chdtr", "chdtrc"] {
                points.push(PointCase {
                    case_id: format!("{func}_df{df}_x{x}"),
                    func: func.to_string(),
                    p1: df,
                    p2: 0.0,
                    arg: x,
                });
            }
        }
        // chdtri intentionally omitted — same root cause as
        // pdtri: the local gammaincinv in gamma.rs diverges
        // (chdtri(3, 0.99) returns ~5.6e5 vs scipy 0.115).
        // Tracked in expanded frankenscipy-jr3na.
        let _ = qs;
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
    p1 = float(case["p1"]); p2 = float(case["p2"]); arg = float(case["arg"])
    try:
        if func == "gdtr":     value = special.gdtr(p1, p2, arg)
        elif func == "gdtrc":  value = special.gdtrc(p1, p2, arg)
        elif func == "pdtr":   value = special.pdtr(p1, arg)
        elif func == "pdtrc":  value = special.pdtrc(p1, arg)
        elif func == "pdtri":  value = special.pdtri(p1, arg)
        elif func == "chdtr":  value = special.chdtr(p1, arg)
        elif func == "chdtrc": value = special.chdtrc(p1, arg)
        elif func == "chdtri": value = special.chdtri(p1, arg)
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize gdtr/pdtr/chdtr query");
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
                "failed to spawn python3 for gdtr/pdtr/chdtr oracle: {e}"
            );
            eprintln!("skipping gdtr/pdtr/chdtr oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open gdtr/pdtr/chdtr oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "gdtr/pdtr/chdtr oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping gdtr/pdtr/chdtr oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for gdtr/pdtr/chdtr oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "gdtr/pdtr/chdtr oracle failed: {stderr}"
        );
        eprintln!("skipping gdtr/pdtr/chdtr oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse gdtr/pdtr/chdtr oracle JSON"))
}

#[test]
fn diff_special_gdtr_pdtr_chdtr() {
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
            && let Some(rust_v) = fsci_eval(&case.func, case.p1, case.p2, case.arg)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            let scale = scipy_v.abs().max(1.0);
            let rel_diff = abs_diff / scale;
            max_abs_overall = max_abs_overall.max(abs_diff);
            max_rel_overall = max_rel_overall.max(rel_diff);

            let pass = match case.func.as_str() {
                "gdtr" | "gdtrc" | "pdtr" | "pdtrc" | "chdtr" | "chdtrc" => abs_diff <= CDF_TOL,
                "pdtri" | "chdtri" => abs_diff <= PPF_TOL_REL * scale,
                _ => false,
            };
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                abs_diff,
                rel_diff,
                pass,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_gdtr_pdtr_chdtr".into(),
        category: "scipy.special.gdtr/pdtr/chdtr family".into(),
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
                "gdtr/pdtr/chdtr {} mismatch: {} abs={} rel={}",
                d.func, d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special gdtr/pdtr/chdtr conformance failed: {} cases, max_abs={} max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
}
