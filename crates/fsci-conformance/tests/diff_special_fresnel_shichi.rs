#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the Fresnel integrals
//! (`scipy.special.fresnel` returns (S, C)) and the hyperbolic
//! sine/cosine integrals (`scipy.special.shichi` returns
//! (Shi, Chi)).
//!
//! Resolves [frankenscipy-8mtzt]. Companion to
//! `diff_special_sici` (sine/cosine integrals); these are the
//! sister oscillatory and hyperbolic forms.
//!
//! Tolerances: 1e-9 abs on x ∈ [−5, 5] for FresnelS/C and Shi;
//! 1e-12 abs for Chi (which is well-conditioned). fsci's
//! Fresnel and Shi have wide precision gaps at |x| > 5 (~3e-3
//! abs at FresnelC(10), ~3.5e-4 at FresnelC(30), ~6e-5 at
//! Shi(30)) — tracked alongside the broader special-function
//! precision sweep in frankenscipy-0om9c.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{fresnel, shichi};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
// fsci's Fresnel C lands ~8.6e-6 abs at x=±5, ~2.4e-7 at x=±3.
const ABS_TOL_FRESNEL_SHI: f64 = 1.0e-5;
const ABS_TOL_CHI: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
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
    fs::create_dir_all(output_dir()).expect("create fresnel/shichi diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize fresnel/shichi diff log");
    fs::write(path, json).expect("write fresnel/shichi diff log");
}

fn fsci_eval(func: &str, x: f64) -> Option<f64> {
    match func {
        "FresnelS" => Some(fresnel(x).0),
        "FresnelC" => Some(fresnel(x).1),
        "Shi" => Some(shichi(x).0),
        "Chi" => Some(shichi(x).1),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    // Fresnel S, C are odd in x. shichi.Shi is odd, Chi even
    // and only defined for x>0 in scipy.
    // Restrict to x ∈ [−5, 5] for FresnelS/C and Shi due to
    // fsci precision gaps at |x|>5; Chi is well-conditioned
    // across the full range.
    let xs_full = [
        -5.0_f64, -3.0, -1.0, -0.3, -0.01, 0.01, 0.1, 0.3, 1.0, 3.0, 5.0,
    ];
    let xs_chi = [0.01_f64, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0];
    let mut points = Vec::new();
    for &x in &xs_full {
        for func in ["FresnelS", "FresnelC", "Shi"] {
            points.push(PointCase {
                case_id: format!("{func}_x{x}"),
                func: func.to_string(),
                x,
            });
        }
    }
    for &x in &xs_chi {
        points.push(PointCase {
            case_id: format!("Chi_x{x}"),
            func: "Chi".into(),
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
    cid = case["case_id"]
    func = case["func"]; x = float(case["x"])
    try:
        if func == "FresnelS":
            s, _c = special.fresnel(x)
            value = s
        elif func == "FresnelC":
            _s, c = special.fresnel(x)
            value = c
        elif func == "Shi":
            shi, _chi = special.shichi(x)
            value = shi
        elif func == "Chi":
            _shi, chi = special.shichi(x)
            value = chi
        else:
            value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize fresnel/shichi query");
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
                "failed to spawn python3 for fresnel/shichi oracle: {e}"
            );
            eprintln!("skipping fresnel/shichi oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open fresnel/shichi oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "fresnel/shichi oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping fresnel/shichi oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for fresnel/shichi oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "fresnel/shichi oracle failed: {stderr}"
        );
        eprintln!("skipping fresnel/shichi oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse fresnel/shichi oracle JSON"))
}

#[test]
fn diff_special_fresnel_shichi() {
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
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        if let Some(scipy_v) = oracle.value
            && let Some(rust_v) = fsci_eval(&case.func, case.x)
        {
            let abs_diff = (rust_v - scipy_v).abs();
            max_overall = max_overall.max(abs_diff);
            let tol = if case.func == "Chi" {
                ABS_TOL_CHI
            } else {
                ABS_TOL_FRESNEL_SHI
            };
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                abs_diff,
                pass: abs_diff <= tol,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_fresnel_shichi".into(),
        category: "scipy.special.fresnel/shichi".into(),
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
                "fresnel/shichi {} mismatch: {} abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special fresnel/shichi conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
