#![forbid(unsafe_code)]
//! Live scipy.special.loggamma parity for the explicit fsci_special::loggamma API.
//!
//! Resolves [frankenscipy-qq65q]. Tolerance: 1e-10 abs for finite values;
//! non-finite branch-cut and pole outputs compare by classification.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;
use fsci_special::loggamma;
use fsci_special::types::Complex64 as FsciComplex;
use fsci_special::types::SpecialTensor;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-011";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    re: f64,
    im: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    re_class: String,
    re: Option<f64>,
    im_class: Option<String>,
    im: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
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
    fs::create_dir_all(output_dir()).expect("create loggamma diff dir");
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

fn classify(value: f64) -> &'static str {
    if value.is_nan() {
        "nan"
    } else if value == f64::INFINITY {
        "pos_inf"
    } else if value == f64::NEG_INFINITY {
        "neg_inf"
    } else {
        "finite"
    }
}

fn finite_value(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn fsci_eval(case: &PointCase) -> PointArm {
    let result = if let Some(im) = case.im {
        loggamma(
            &SpecialTensor::ComplexScalar(FsciComplex::new(case.re, im)),
            RuntimeMode::Strict,
        )
    } else {
        loggamma(&SpecialTensor::RealScalar(case.re), RuntimeMode::Strict)
    };

    match result {
        Ok(SpecialTensor::RealScalar(value)) => PointArm {
            case_id: case.case_id.clone(),
            re_class: classify(value).into(),
            re: finite_value(value),
            im_class: None,
            im: None,
        },
        Ok(SpecialTensor::ComplexScalar(value)) => PointArm {
            case_id: case.case_id.clone(),
            re_class: classify(value.re).into(),
            re: finite_value(value.re),
            im_class: Some(classify(value.im).into()),
            im: finite_value(value.im),
        },
        _ => PointArm {
            case_id: case.case_id.clone(),
            re_class: "error".into(),
            re: None,
            im_class: case.im.map(|_| "error".into()),
            im: None,
        },
    }
}

fn generate_query() -> OracleQuery {
    let probes: &[(f64, Option<f64>)] = &[
        (-2.5, None),
        (-1.0, None),
        (-0.5, None),
        (-0.0, None),
        (0.0, None),
        (0.5, None),
        (1.0, None),
        (2.0, None),
        (10.0, None),
        (1.0, Some(0.0)),
        (0.5, Some(0.0)),
        (-0.5, Some(0.0)),
        (-1.0, Some(0.0)),
        (-0.5, Some(1.0e-12)),
        (-0.5, Some(-1.0e-12)),
        (1.0, Some(-2.0)),
        (3.0, Some(4.0)),
    ];
    let points = probes
        .iter()
        .enumerate()
        .map(|(idx, &(re, im))| PointCase {
            case_id: match im {
                Some(im) => format!(
                    "c{idx:02}_re{}_im{}",
                    re.to_string().replace('.', "p").replace('-', "n"),
                    im.to_string().replace('.', "p").replace('-', "n")
                ),
                None => format!(
                    "r{idx:02}_{}",
                    re.to_string().replace('.', "p").replace('-', "n")
                ),
            },
            re,
            im,
        })
        .collect();
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy import special

def cls(x):
    if math.isnan(x):
        return "nan", None
    if math.isinf(x):
        return ("pos_inf" if x > 0 else "neg_inf"), None
    return "finite", float(x)

q = json.loads(sys.argv[1])
points = []
for case in q["points"]:
    cid = case["case_id"]
    try:
        if case["im"] is None:
            r = float(special.loggamma(float(case["re"])))
            re_class, re = cls(r)
            points.append({"case_id": cid, "re_class": re_class, "re": re,
                           "im_class": None, "im": None})
        else:
            z = complex(float(case["re"]), float(case["im"]))
            r = complex(special.loggamma(z))
            re_class, re = cls(float(r.real))
            im_class, im = cls(float(r.imag))
            points.append({"case_id": cid, "re_class": re_class, "re": re,
                           "im_class": im_class, "im": im})
    except Exception as exc:
        sys.stderr.write(f"oracle {cid}: {exc}\n")
        points.append({"case_id": cid, "re_class": "error", "re": None,
                       "im_class": "error" if case["im"] is not None else None, "im": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize query");
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
                "failed to spawn python3 for loggamma oracle: {e}"
            );
            eprintln!("skipping loggamma oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open oracle stdin");
        if let Err(err) = stdin.write_all(script.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "loggamma oracle script write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping loggamma oracle: script write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for loggamma oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "loggamma oracle failed: {stderr}"
        );
        eprintln!("skipping loggamma oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse loggamma oracle JSON"))
}

fn compare(case: &PointCase, actual: &PointArm, expected: &PointArm) -> CaseDiff {
    if actual.re_class != expected.re_class || actual.im_class != expected.im_class {
        return CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff: f64::INFINITY,
            pass: false,
        };
    }

    let mut abs_diff = 0.0_f64;
    if let (Some(actual_re), Some(expected_re)) = (actual.re, expected.re) {
        abs_diff = abs_diff.max((actual_re - expected_re).abs());
    }
    if let (Some(actual_im), Some(expected_im)) = (actual.im, expected.im) {
        let two_pi = 2.0 * std::f64::consts::PI;
        let turns = ((actual_im - expected_im) / two_pi).round();
        abs_diff = abs_diff.max((actual_im - turns * two_pi - expected_im).abs());
    }

    CaseDiff {
        case_id: case.case_id.clone(),
        abs_diff,
        pass: abs_diff <= ABS_TOL,
    }
}

#[test]
fn diff_special_loggamma() {
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
        let expected = pmap.get(&case.case_id).expect("oracle case present");
        let actual = fsci_eval(case);
        let diff = compare(case, &actual, expected);
        max_overall = max_overall.max(diff.abs_diff);
        diffs.push(diff);
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_special_loggamma".into(),
        category: "fsci_special::loggamma vs scipy.special.loggamma".into(),
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
            eprintln!("loggamma mismatch: {} abs_diff={}", d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "loggamma conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
