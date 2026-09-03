#![forbid(unsafe_code)]
//! Live formula-derived parity for fsci_special::{beta, betaln} on
//! ComplexScalar inputs. scipy.special.beta does not accept complex
//! arguments, so the oracle reconstructs the reference value via
//! scipy.special.loggamma:
//!
//!   betaln(a, b) = loggamma(a) + loggamma(b) - loggamma(a+b)
//!   beta(a, b)   = exp(betaln(a, b))
//!
//! Resolves [frankenscipy-6pfey]. Tolerance: 1e-9 abs for betaln (real
//! and imaginary, mod 2π); rel 1e-9 for beta values where |β| > 1e-12.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::types::Complex64 as FsciComplex;
use fsci_special::types::SpecialTensor;
use fsci_special::{beta, betaln};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-9;
const REL_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    a_re: f64,
    a_im: f64,
    b_re: f64,
    b_im: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    betaln_re: Option<f64>,
    betaln_im: Option<f64>,
    beta_re: Option<f64>,
    beta_im: Option<f64>,
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
    fs::create_dir_all(output_dir()).expect("create beta_complex diff dir");
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

fn fsci_eval(op: &str, a_re: f64, a_im: f64, b_re: f64, b_im: f64) -> Option<(f64, f64)> {
    let a = SpecialTensor::ComplexScalar(FsciComplex::new(a_re, a_im));
    let b = SpecialTensor::ComplexScalar(FsciComplex::new(b_re, b_im));
    let result = match op {
        "beta" => beta(&a, &b, RuntimeMode::Strict),
        "betaln" => betaln(&a, &b, RuntimeMode::Strict),
        _ => return None,
    };
    match result {
        Ok(SpecialTensor::ComplexScalar(c)) => Some((c.re, c.im)),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    // (a_re, a_im, b_re, b_im). Avoid combinations where a+b is at a pole.
    let probes: &[(f64, f64, f64, f64)] = &[
        (2.0, 1.0, 3.0, 0.5),
        (1.5, 0.0, 2.5, 1.0),
        (0.7, 0.3, 1.2, -0.4),
        (3.0, 2.0, 4.0, 1.0),
        (2.5, -1.0, 3.5, -0.5),
        (1.0, 1.0, 1.0, -1.0),
        (4.0, 0.5, 2.0, 0.0),
        (5.0, 0.0, 3.0, 2.0),
        (2.5, 0.0, 2.5, 0.0),
        (1.5, 0.5, 0.5, 1.5),
        (3.0, 0.0, 3.0, 0.0),
    ];
    let points: Vec<PointCase> = probes
        .iter()
        .enumerate()
        .map(|(i, &(ar, ai, br, bi))| PointCase {
            case_id: format!("p{i:02}"),
            a_re: ar,
            a_im: ai,
            b_re: br,
            b_im: bi,
        })
        .collect();
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import cmath
from scipy import special

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    a = complex(float(case["a_re"]), float(case["a_im"]))
    b = complex(float(case["b_re"]), float(case["b_im"]))
    try:
        lg_a = complex(special.loggamma(a))
        lg_b = complex(special.loggamma(b))
        lg_ab = complex(special.loggamma(a + b))
        bln = lg_a + lg_b - lg_ab
        bv = cmath.exp(bln)
        if all(math.isfinite(v) for v in [bln.real, bln.imag, bv.real, bv.imag]):
            points.append({
                "case_id": cid,
                "betaln_re": float(bln.real), "betaln_im": float(bln.imag),
                "beta_re": float(bv.real), "beta_im": float(bv.imag),
            })
        else:
            points.append({"case_id": cid, "betaln_re": None, "betaln_im": None,
                           "beta_re": None, "beta_im": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "betaln_re": None, "betaln_im": None,
                       "beta_re": None, "beta_im": None})
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
                "failed to spawn python3 for beta_complex oracle: {e}"
            );
            eprintln!("skipping beta_complex oracle: python3 not available ({e})");
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
                "beta_complex oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping beta_complex oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for beta_complex oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "beta_complex oracle failed: {stderr}"
        );
        eprintln!("skipping beta_complex oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse beta_complex oracle JSON"))
}

#[test]
fn diff_special_beta_complex() {
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

    let two_pi = 2.0 * std::f64::consts::PI;

    for case in &query.points {
        let Some(arm) = pmap.get(&case.case_id) else {
            continue;
        };
        let (Some(eb_re), Some(eb_im), Some(ebv_re), Some(ebv_im)) =
            (arm.betaln_re, arm.betaln_im, arm.beta_re, arm.beta_im)
        else {
            continue;
        };

        // betaln
        if let Some((re, im)) = fsci_eval("betaln", case.a_re, case.a_im, case.b_re, case.b_im) {
            let im_diff_full = ((im - eb_im) / two_pi).round();
            let im_adj = im - im_diff_full * two_pi;
            let abs_d = ((re - eb_re).powi(2) + (im_adj - eb_im).powi(2)).sqrt();
            max_overall = max_overall.max(abs_d);
            diffs.push(CaseDiff {
                case_id: format!("{}_betaln", case.case_id),
                op: "betaln".into(),
                abs_diff: abs_d,
                pass: abs_d <= ABS_TOL,
            });
        }

        // beta (use rel for large magnitudes)
        if let Some((re, im)) = fsci_eval("beta", case.a_re, case.a_im, case.b_re, case.b_im) {
            let abs_d = ((re - ebv_re).powi(2) + (im - ebv_im).powi(2)).sqrt();
            let mag = (ebv_re * ebv_re + ebv_im * ebv_im).sqrt();
            let rel_d = if mag > 1.0e-12 { abs_d / mag } else { abs_d };
            max_overall = max_overall.max(rel_d);
            let pass = rel_d <= REL_TOL || abs_d <= ABS_TOL;
            diffs.push(CaseDiff {
                case_id: format!("{}_beta", case.case_id),
                op: "beta".into(),
                abs_diff: rel_d,
                pass,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_beta_complex".into(),
        category: "fsci_special::beta + betaln (ComplexScalar) vs scipy.special.loggamma formula"
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
            eprintln!("{} mismatch: {} d={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "beta_complex conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
