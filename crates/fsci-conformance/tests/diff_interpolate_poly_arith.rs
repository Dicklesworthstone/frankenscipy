#![forbid(unsafe_code)]
//! Live SciPy/numpy differential coverage for fsci_interpolate poly
//! arithmetic helpers: `polyadd`, `polysub` (HIGH-FIRST, numpy parity)
//! and `pade`, `ratval` (LOW-FIRST, scipy.interpolate.pade parity after
//! reversing its poly1d HIGH-FIRST output).
//!
//! Resolves [frankenscipy-liw7a]. 1e-10 abs.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_interpolate::{pade, polyadd, polysub, ratval};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    op: String,
    /// For polyadd/polysub: lhs (HIGH-FIRST).
    /// For pade: Taylor coefficients (LOW-FIRST).
    /// For ratval: p (LOW-FIRST numerator).
    a: Vec<f64>,
    /// For polyadd/polysub: rhs (HIGH-FIRST).
    /// For pade: ignored.
    /// For ratval: q (LOW-FIRST denominator).
    b: Vec<f64>,
    /// For pade: m (denominator order, scipy convention). For ratval: ignored.
    m: usize,
    /// For pade: n (numerator order, scipy convention). For ratval: x.
    n: usize,
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// For polyadd/polysub: result vector (HIGH-FIRST).
    /// For pade: numerator coeffs (LOW-FIRST, length n+1).
    /// For ratval: not used.
    vec_value: Option<Vec<f64>>,
    /// For pade: denominator coeffs (LOW-FIRST, length m+1).
    /// For ratval: scalar value.
    aux_vec: Option<Vec<f64>>,
    scalar_value: Option<f64>,
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
    compared: BTreeMap<String, ArmCounts>,
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
    fs::create_dir_all(output_dir()).expect("create poly_arith diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize poly_arith diff log");
    fs::write(path, json).expect("write poly_arith diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    // polyadd / polysub (HIGH-FIRST)
    let arith: &[(&str, Vec<f64>, Vec<f64>)] = &[
        ("equal_quad", vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]),
        ("a_longer", vec![1.0, 2.0, 3.0], vec![1.0, 5.0]),
        ("b_longer", vec![1.0, 5.0], vec![1.0, 2.0, 3.0]),
        ("with_zero", vec![3.0, 0.0, -1.0], vec![0.0, 0.0, 2.0]),
        ("neg_coeffs", vec![-1.0, -2.0, -3.0], vec![1.0, 2.0, 3.0]),
    ];
    for (name, a, b) in arith {
        for op in ["polyadd", "polysub"] {
            points.push(PointCase {
                case_id: format!("{op}_{name}"),
                op: op.into(),
                a: a.clone(),
                b: b.clone(),
                m: 0,
                n: 0,
                x: 0.0,
            });
        }
    }

    // pade — Taylor series, LOW-FIRST
    let pade_cases: &[(&str, Vec<f64>, usize, usize)] = &[
        // exp(x) ≈ 1 + x + x²/2! + x³/3! + x⁴/4!  →  pade(2,2)
        ("exp_2_2", vec![1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0], 2, 2),
        // log(1+x) ≈ x - x²/2 + x³/3 - x⁴/4 + x⁵/5  →  pade(2,2)
        (
            "log1p_2_2",
            vec![0.0, 1.0, -0.5, 1.0 / 3.0, -0.25, 0.2],
            2,
            2,
        ),
        // sin(x) ≈ x - x³/6 + x⁵/120. scipy.pade(an, m, n) uses m=denominator
        // order, n=numerator order; request the [num=3/denom=2] approximant as
        // (m=2, n=3). The previous (3,2) asked scipy for [num=2/denom=3], which is
        // singular for a series with a0=0 and crashed the oracle.
        (
            "sin_2_3",
            vec![0.0, 1.0, 0.0, -1.0 / 6.0, 0.0, 1.0 / 120.0],
            2,
            3,
        ),
    ];
    for (name, taylor, m, n) in pade_cases {
        points.push(PointCase {
            case_id: format!("pade_{name}"),
            op: "pade".into(),
            a: taylor.clone(),
            b: vec![],
            m: *m,
            n: *n,
            x: 0.0,
        });
    }

    // ratval — LOW-FIRST p, q, evaluated at x
    let ratval_cases: &[(&str, Vec<f64>, Vec<f64>, f64)] = &[
        ("rat_simple", vec![1.0, 0.0, 1.0], vec![1.0, 1.0], 2.0),
        ("rat_at_zero", vec![3.0, 2.0, 1.0], vec![1.0, 0.0, 1.0], 0.0),
        (
            "rat_neg_x",
            vec![1.0, -1.0, 0.5],
            vec![1.0, 1.0, 0.25],
            -0.5,
        ),
        (
            "rat_higher_deg",
            vec![1.0, 0.0, 0.5, 0.0, 0.125],
            vec![1.0, 0.5, 0.25],
            1.5,
        ),
    ];
    for (name, p, q, x) in ratval_cases {
        points.push(PointCase {
            case_id: format!("ratval_{name}"),
            op: "ratval".into(),
            a: p.clone(),
            b: q.clone(),
            m: 0,
            n: 0,
            x: *x,
        });
    }

    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import interpolate

def finite_vec_or_none(arr):
    out = []
    for v in np.asarray(arr).flatten().tolist():
        try:
            v = float(v)
        except Exception:
            return None
        if not math.isfinite(v):
            return None
        out.append(v)
    return out

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]
    a = np.array(case["a"], dtype=float)
    b = np.array(case["b"], dtype=float)
    m = int(case["m"]); n = int(case["n"]); x = float(case["x"])
    try:
        if op == "polyadd":
            v = np.polyadd(a, b)
            points.append({"case_id": cid, "vec_value": finite_vec_or_none(v),
                           "aux_vec": None, "scalar_value": None})
        elif op == "polysub":
            v = np.polysub(a, b)
            points.append({"case_id": cid, "vec_value": finite_vec_or_none(v),
                           "aux_vec": None, "scalar_value": None})
        elif op == "pade":
            p_poly, q_poly = interpolate.pade(a, m, n)
            # poly1d returns HIGH-FIRST. fsci returns LOW-FIRST. Reverse.
            p_lo = p_poly.coeffs[::-1].tolist()
            q_lo = q_poly.coeffs[::-1].tolist()
            points.append({"case_id": cid,
                           "vec_value": finite_vec_or_none(p_lo),
                           "aux_vec": finite_vec_or_none(q_lo),
                           "scalar_value": None})
        elif op == "ratval":
            # Evaluate p(x)/q(x) with LOW-FIRST coefficients.
            num = sum(c * (x ** i) for i, c in enumerate(a.tolist()))
            den = sum(c * (x ** i) for i, c in enumerate(b.tolist()))
            val = float(num / den) if den != 0.0 else float("nan")
            if not math.isfinite(val):
                points.append({"case_id": cid, "vec_value": None,
                               "aux_vec": None, "scalar_value": None})
            else:
                points.append({"case_id": cid, "vec_value": None,
                               "aux_vec": None, "scalar_value": val})
        else:
            points.append({"case_id": cid, "vec_value": None,
                           "aux_vec": None, "scalar_value": None})
    except Exception:
        points.append({"case_id": cid, "vec_value": None,
                       "aux_vec": None, "scalar_value": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize poly_arith query");
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
                "failed to spawn python3 for poly_arith oracle: {e}"
            );
            eprintln!("skipping poly_arith oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open poly_arith oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "poly_arith oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping poly_arith oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for poly_arith oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "poly_arith oracle failed: {stderr}"
        );
        eprintln!("skipping poly_arith oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse poly_arith oracle JSON"))
}

#[test]
fn diff_interpolate_poly_arith() {
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
    let mut ledger = CompareLedger::new(
        "diff_interpolate_poly_arith",
        &["polyadd", "polysub", "pade_p", "pade_q", "ratval"],
    );

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let abs_d = match case.op.as_str() {
            "polyadd" => {
                let fsci_v = polyadd(&case.a, &case.b);
                let Some((expected, fsci_v)) = ledger.slices(
                    "polyadd",
                    &case.case_id,
                    scipy_arm.vec_value.as_deref(),
                    Some(fsci_v.as_slice()),
                ) else {
                    continue;
                };
                let d = fsci_v
                    .iter()
                    .zip(expected.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                ledger.compared("polyadd", &case.case_id, d <= ABS_TOL);
                d
            }
            "polysub" => {
                let fsci_v = polysub(&case.a, &case.b);
                let Some((expected, fsci_v)) = ledger.slices(
                    "polysub",
                    &case.case_id,
                    scipy_arm.vec_value.as_deref(),
                    Some(fsci_v.as_slice()),
                ) else {
                    continue;
                };
                let d = fsci_v
                    .iter()
                    .zip(expected.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                ledger.compared("polysub", &case.case_id, d <= ABS_TOL);
                d
            }
            "pade" => {
                let fsci = pade(&case.a, case.m, case.n).ok();
                let arms = [
                    (
                        "pade_p",
                        scipy_arm.vec_value.as_deref(),
                        fsci.as_ref().map(|(p, _)| p.as_slice()),
                    ),
                    (
                        "pade_q",
                        scipy_arm.aux_vec.as_deref(),
                        fsci.as_ref().map(|(_, q)| q.as_slice()),
                    ),
                ];
                let mut arm_diffs = [f64::NAN; 2];
                for (slot, (arm, scipy, got)) in arm_diffs.iter_mut().zip(arms) {
                    let Some((expected, actual)) = ledger.slices(arm, &case.case_id, scipy, got)
                    else {
                        continue;
                    };
                    let d = actual
                        .iter()
                        .zip(expected.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0_f64, f64::max);
                    ledger.compared(arm, &case.case_id, d <= ABS_TOL);
                    *slot = d;
                }
                if arm_diffs.iter().any(|d| d.is_nan()) {
                    continue; // the ledger recorded why this case was not compared
                }
                let [dp, dq] = arm_diffs;
                dp.max(dq)
            }
            "ratval" => {
                let Some((scalar, fsci_v)) = ledger.pair(
                    "ratval",
                    &case.case_id,
                    scipy_arm.scalar_value,
                    Some(ratval(&case.a, &case.b, case.x)),
                ) else {
                    continue;
                };
                let d = (fsci_v - scalar).abs();
                ledger.compared("ratval", &case.case_id, d <= ABS_TOL);
                d
            }
            other => panic!("unknown poly_arith op {other} in {}", case.case_id),
        };
        max_overall = max_overall.max(abs_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            abs_diff: abs_d,
            pass: abs_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_interpolate_poly_arith".into(),
        category: "numpy.polyadd/polysub + scipy.interpolate.pade + ratval".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
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
        "poly_arith conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    let per_op = |op: &str| query.points.iter().filter(|c| c.op == op).count();
    ledger.finish(
        ["polyadd", "polysub", "pade", "ratval"]
            .into_iter()
            .map(per_op)
            .min()
            .expect("four ops"),
    );
}
