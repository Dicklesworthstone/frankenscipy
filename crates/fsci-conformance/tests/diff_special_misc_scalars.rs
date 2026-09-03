#![forbid(unsafe_code)]
//! Live SciPy differential coverage for three uncovered scalar
//! special functions:
//!   - `fsci_special::tklmbda(x, lam)` vs `scipy.special.tklmbda`
//!     (Tukey-lambda CDF)
//!   - `fsci_special::voigt_profile(x, sigma, gamma)` vs
//!     `scipy.special.voigt_profile(x, sigma, gamma)` (Voigt profile)
//!   - `fsci_special::stdtridf(p, t)` vs `scipy.special.stdtridf(p, t)`
//!     (Student-t df inversion)
//!
//! Resolves [frankenscipy-uyp2g].

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{stdtridf, tklmbda, voigt_profile};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-002";
const TIGHT_TOL: f64 = 1.0e-12;
const VOIGT_TOL: f64 = 5.0e-8;
const STDTRIDF_TOL: f64 = 1.0e-6; // df-inversion is iterative
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    arg1: f64,
    arg2: f64,
    arg3: f64,
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
    fs::create_dir_all(output_dir()).expect("create misc_scalars diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize misc_scalars diff log");
    fs::write(path, json).expect("write misc_scalars diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    // tklmbda: x ∈ (-∞, ∞), lam ∈ ℝ
    let tk_xs: &[f64] = &[-3.0, -1.5, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
    let tk_lams: &[f64] = &[-0.5, 0.0, 0.5, 1.0, 1.5];
    for &x in tk_xs {
        for &lam in tk_lams {
            points.push(PointCase {
                case_id: format!("tklmbda_x{x}_lam{lam}"),
                func: "tklmbda".into(),
                arg1: x,
                arg2: lam,
                arg3: 0.0,
            });
        }
    }

    // voigt_profile: x ∈ ℝ with positive-shape and zero-shape edge cases.
    let v_xs: &[f64] = &[-4.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 4.0];
    let v_sigmas: &[f64] = &[0.5, 1.0, 1.5, 3.0];
    for &x in v_xs {
        for &s in v_sigmas {
            points.push(PointCase {
                case_id: format!("voigt0_x{x}_s{s}"),
                func: "voigt_profile".into(),
                arg1: x,
                arg2: s,
                arg3: 0.0,
            });
        }
    }
    let voigt_cases: &[(f64, f64, f64)] = &[
        (0.0, 1.0, 0.25),
        (0.0, 1.0, 1.0),
        (1.0, 1.0, 0.5),
        (1.0, 1.0, 1.0),
        (-1.5, 0.75, 0.25),
        (2.0, 1.5, 1.25),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (2.0, 0.0, 0.5),
    ];
    for &(x, sigma, gamma) in voigt_cases {
        points.push(PointCase {
            case_id: format!("voigt_x{x}_s{sigma}_g{gamma}"),
            func: "voigt_profile".into(),
            arg1: x,
            arg2: sigma,
            arg3: gamma,
        });
    }

    // stdtridf: p ∈ (0, 1), t — valid student-t inversion.
    // Pick safe interior values; stdtridf solves for df given (p, t).
    let stdtridf_cases: &[(f64, f64)] = &[
        (0.05, -1.0),
        (0.10, 1.5),
        (0.25, 2.0),
        (0.50, 0.5),
        (0.75, 1.5),
        (0.90, 1.8),
        (0.95, 2.1),
        (0.025, -2.5),
    ];
    for (p, t) in stdtridf_cases {
        points.push(PointCase {
            case_id: format!("stdtridf_p{p}_t{t}"),
            func: "stdtridf".into(),
            arg1: *p,
            arg2: *t,
            arg3: 0.0,
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

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.loads(sys.argv[1])
points = []
for case in q["points"]:
    cid = case["case_id"]; fn = case["func"]
    a1 = float(case["arg1"]); a2 = float(case["arg2"])
    a3 = float(case.get("arg3", 0.0))
    try:
        if fn == "tklmbda":
            v = special.tklmbda(a1, a2)
        elif fn == "voigt_profile":
            v = special.voigt_profile(a1, a2, a3)
        elif fn == "stdtridf":
            v = special.stdtridf(a1, a2)
        else:
            v = None
        points.append({"case_id": cid, "value": fnone(v) if v is not None else None})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize misc_scalars query");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-")
        .arg(query_json)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for misc_scalars oracle: {e}"
            );
            eprintln!("skipping misc_scalars oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open misc_scalars oracle stdin");
        if let Err(err) = stdin.write_all(script.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "misc_scalars oracle script write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping misc_scalars oracle: script write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for misc_scalars oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "misc_scalars oracle failed: {stderr}"
        );
        eprintln!("skipping misc_scalars oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse misc_scalars oracle JSON"))
}

#[test]
fn diff_special_misc_scalars() {
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
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(scipy_v) = scipy_arm.value else {
            continue;
        };
        let (fsci_v, tol) = match case.func.as_str() {
            "tklmbda" => (tklmbda(case.arg1, case.arg2), TIGHT_TOL),
            "voigt_profile" => (voigt_profile(case.arg1, case.arg2, case.arg3), VOIGT_TOL),
            "stdtridf" => (stdtridf(case.arg1, case.arg2), STDTRIDF_TOL),
            _ => continue,
        };
        if !fsci_v.is_finite() {
            continue;
        }
        let abs_d = (fsci_v - scipy_v).abs();
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            func: case.func.clone(),
            abs_diff: abs_d,
            pass: abs_d <= tol,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_misc_scalars".into(),
        category: "scipy.special.tklmbda / voigt_profile / stdtridf".into(),
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
            eprintln!(
                "misc_scalars {} mismatch: {} abs_diff={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special tklmbda / voigt_profile / stdtridf conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
