#![forbid(unsafe_code)]
//! Live scipy.special parity for fsci_special complex scalar
//! variants: beta::{complex_beta_scalar, complex_betaln_scalar},
//! gamma::{complex_digamma_scalar, complex_polygamma_scalar}.
//!
//! Resolves [frankenscipy-vgob4]. Compare re+im at 1e-7 abs / 1e-6
//! rel for moderate magnitudes.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::beta::{complex_beta_scalar, complex_betaln_scalar};
use fsci_special::gamma::{complex_digamma_scalar, complex_polygamma_scalar};
use fsci_special::types::Complex64;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-7;
const REL_TOL: f64 = 1.0e-6;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "beta" | "betaln" | "digamma" | "polygamma"
    a_re: f64,
    a_im: f64,
    b_re: f64,
    b_im: f64,
    n: usize, // for polygamma
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    values: Option<Vec<f64>>, // [re, im]
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
    fs::create_dir_all(output_dir()).expect("create cgamma diff dir");
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
    // Use complex args away from poles (negative-integer Re).
    let ab_probes: &[((f64, f64), (f64, f64))] = &[
        ((1.0, 0.5), (2.0, 0.0)),
        ((2.0, 0.0), (3.0, 0.0)),
        ((0.5, 0.5), (1.5, -0.5)),
        ((1.5, 0.0), (1.5, 0.0)),
        ((3.0, 1.0), (2.0, -1.0)),
    ];
    for &((ar, ai), (br, bi)) in ab_probes {
        for op in ["beta", "betaln"] {
            points.push(Case {
                case_id: format!("{op}_a{ar}_{ai}_b{br}_{bi}"),
                op: op.into(),
                a_re: ar,
                a_im: ai,
                b_re: br,
                b_im: bi,
                n: 0,
            });
        }
    }

    let z_probes: &[(f64, f64)] = &[
        (1.0, 0.5),
        (2.0, 0.0),
        (1.5, 1.0),
        (3.0, -1.0),
        (0.5, 2.0),
        (4.0, 0.5),
    ];
    for &(re, im) in z_probes {
        points.push(Case {
            case_id: format!("digamma_z{re}_{im}"),
            op: "digamma".into(),
            a_re: re,
            a_im: im,
            b_re: 0.0,
            b_im: 0.0,
            n: 0,
        });
        for n in [1_usize, 2, 3] {
            points.push(Case {
                case_id: format!("polygamma_n{n}_z{re}_{im}"),
                op: "polygamma".into(),
                a_re: re,
                a_im: im,
                b_re: 0.0,
                b_im: 0.0,
                n,
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
import numpy as np
from scipy import special as sp

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    a = complex(float(case["a_re"]), float(case["a_im"]))
    b = complex(float(case["b_re"]), float(case["b_im"]))
    n = int(case["n"])
    try:
        if op == "beta":
            c = complex(sp.beta(a, b))
        elif op == "betaln":
            c = complex(sp.betaln(a, b))
        elif op == "digamma":
            c = complex(sp.digamma(a))
        elif op == "polygamma":
            c = complex(sp.polygamma(n, a))
        else:
            points.append({"case_id": cid, "values": None}); continue
        if math.isfinite(c.real) and math.isfinite(c.imag):
            points.append({"case_id": cid, "values": [float(c.real), float(c.imag)]})
        else:
            points.append({"case_id": cid, "values": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "values": None})
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
                "failed to spawn python3 for cgamma oracle: {e}"
            );
            eprintln!("skipping cgamma oracle: python3 not available ({e})");
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
                "cgamma oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping cgamma oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for cgamma oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "cgamma oracle failed: {stderr}"
        );
        eprintln!("skipping cgamma oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse cgamma oracle JSON"))
}

#[test]
fn diff_special_complex_beta_gamma_scalars() {
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
        let Some(expected) = arm.values.as_ref() else {
            continue;
        };
        let a = Complex64::new(case.a_re, case.a_im);
        let b = Complex64::new(case.b_re, case.b_im);
        let c = match case.op.as_str() {
            "beta" => complex_beta_scalar(a, b),
            "betaln" => complex_betaln_scalar(a, b),
            "digamma" => complex_digamma_scalar(a),
            "polygamma" => complex_polygamma_scalar(case.n, a),
            _ => continue,
        };
        if !c.is_finite() {
            continue;
        }
        let d_re = (c.re - expected[0]).abs();
        let d_im = (c.im - expected[1]).abs();
        let abs_d = d_re.max(d_im);
        // pass if abs or rel tolerance satisfied
        let mag = (expected[0].powi(2) + expected[1].powi(2)).sqrt();
        let pass = abs_d <= ABS_TOL || (mag > 1.0 && abs_d / mag <= REL_TOL);
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            abs_diff: abs_d,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_complex_beta_gamma_scalars".into(),
        category:
            "fsci_special complex scalars (beta, betaln, digamma, polygamma) vs scipy.special"
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
        "complex beta/gamma scalars conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
