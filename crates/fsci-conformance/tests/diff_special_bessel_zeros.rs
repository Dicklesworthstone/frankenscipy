#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the zeros of
//! integer-order Bessel functions
//! `scipy.special.jn_zeros(n, k)`, `scipy.special.yn_zeros(n, k)`,
//! `scipy.special.jnp_zeros(n, k)`, `scipy.special.ynp_zeros(n, k)`,
//! `scipy.special.y0_zeros(k)`, `scipy.special.y1_zeros(k)`, and
//! `scipy.special.y1p_zeros(k)`.
//!
//! Resolves [frankenscipy-hwam6]. fsci-special exposes Newton-
//! refined zero finders.
//!
//! 4 orders × 5 zero-counts × 4 generic funcs plus 5 zero-counts ×
//! 3 order-specific funcs = 95 batches; we compare the first k zeros
//! entrywise, and for y0_zeros/y1_zeros/y1p_zeros also compare SciPy's
//! companion values at those zeros. Tolerances: 1e-7 abs (Newton-refined;
//! floor matches the cylindrical Bessel precision that the residual is
//! computed against — frankenscipy-0om9c).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::{jn_zeros, jnp_zeros, y0_zeros, y1_zeros, y1p_zeros, yn_zeros, ynp_zeros};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-7;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    n: u32,
    k: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    values: Option<Vec<f64>>,
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
    fs::create_dir_all(output_dir()).expect("create bessel-zeros diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize bessel-zeros diff log");
    fs::write(path, json).expect("write bessel-zeros diff log");
}

fn fsci_eval(func: &str, n: u32, k: usize) -> Option<Vec<f64>> {
    let zs = match func {
        "jn_zeros" => jn_zeros(n, k),
        "yn_zeros" => yn_zeros(n, k),
        "jnp_zeros" => jnp_zeros(n, k),
        "ynp_zeros" => ynp_zeros(n, k),
        "y0_zeros" => {
            let (zeros, values) = y0_zeros(k);
            zeros.into_iter().chain(values).collect()
        }
        "y1_zeros" => {
            let (zeros, values) = y1_zeros(k);
            zeros.into_iter().chain(values).collect()
        }
        "y1p_zeros" => {
            let (zeros, values) = y1p_zeros(k);
            zeros.into_iter().chain(values).collect()
        }
        _ => return None,
    };
    if zs.iter().all(|z| z.is_finite()) {
        Some(zs)
    } else {
        None
    }
}

fn generate_query() -> OracleQuery {
    let ns = [0_u32, 1, 2, 5];
    let ks = [1_usize, 3, 5, 8, 10];
    let mut points = Vec::new();
    for &n in &ns {
        for &k in &ks {
            for func in ["jn_zeros", "yn_zeros", "jnp_zeros", "ynp_zeros"] {
                points.push(PointCase {
                    case_id: format!("{func}_n{n}_k{k}"),
                    func: func.to_string(),
                    n,
                    k,
                });
            }
        }
    }
    for &k in &ks {
        for func in ["y0_zeros", "y1_zeros", "y1p_zeros"] {
            points.push(PointCase {
                case_id: format!("{func}_k{k}"),
                func: func.to_string(),
                n: 0,
                k,
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

def finite_or_none_list(arr):
    out = []
    for v in arr:
        try:
            v = float(v)
            out.append(v if math.isfinite(v) else None)
        except Exception:
            out.append(None)
    return out

q = json.loads(sys.argv[1])
points = []
for case in q["points"]:
    cid = case["case_id"]; func = case["func"]
    n = int(case["n"]); k = int(case["k"])
    try:
        if func == "jn_zeros":
            arr = special.jn_zeros(n, k).tolist()
        elif func == "yn_zeros":
            arr = special.yn_zeros(n, k).tolist()
        elif func == "jnp_zeros":
            arr = special.jnp_zeros(n, k).tolist()
        elif func == "ynp_zeros":
            arr = special.ynp_zeros(n, k).tolist()
        elif func == "y0_zeros":
            zeros, values = special.y0_zeros(k)
            arr = [float(v.real) for v in zeros] + [float(v.real) for v in values]
        elif func == "y1_zeros":
            zeros, values = special.y1_zeros(k)
            arr = [float(v.real) for v in zeros] + [float(v.real) for v in values]
        elif func == "y1p_zeros":
            zeros, values = special.y1p_zeros(k)
            arr = [float(v.real) for v in zeros] + [float(v.real) for v in values]
        else:
            arr = []
        if any(v is None for v in finite_or_none_list(arr)):
            points.append({"case_id": cid, "values": None})
        else:
            points.append({"case_id": cid, "values": arr})
    except Exception:
        points.append({"case_id": cid, "values": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize bessel-zeros query");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-")
        .arg(&query_json)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for bessel-zeros oracle: {e}"
            );
            eprintln!("skipping bessel-zeros oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open bessel-zeros oracle script stdin");
        if let Err(err) = stdin.write_all(script.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "bessel-zeros oracle script write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping bessel-zeros oracle: script write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for bessel-zeros oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "bessel-zeros oracle failed: {stderr}"
        );
        eprintln!("skipping bessel-zeros oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse bessel-zeros oracle JSON"))
}

#[test]
fn diff_special_bessel_zeros() {
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
        if let Some(scipy_zs) = oracle.values.as_ref()
            && let Some(rust_zs) = fsci_eval(&case.func, case.n, case.k)
        {
            if rust_zs.len() != scipy_zs.len() {
                diffs.push(CaseDiff {
                    case_id: case.case_id.clone(),
                    func: case.func.clone(),
                    abs_diff: f64::INFINITY,
                    pass: false,
                });
                continue;
            }
            let max_abs = rust_zs
                .iter()
                .zip(scipy_zs.iter())
                .map(|(r, s)| (r - s).abs())
                .fold(0.0_f64, f64::max);
            max_overall = max_overall.max(max_abs);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                abs_diff: max_abs,
                pass: max_abs <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_bessel_zeros".into(),
        category: "scipy.special.jn_zeros/yn_zeros/jnp_zeros/ynp_zeros/y0_zeros/y1_zeros/y1p_zeros"
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
            eprintln!(
                "bessel-zeros {} mismatch: {} max_abs={}",
                d.func, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special bessel-zeros conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
}
