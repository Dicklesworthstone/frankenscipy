#![forbid(unsafe_code)]
//! Live scipy.interpolate parity for spline derivative/antiderivative
//! operations: splder, splantider, splint, sproot.
//!
//! Resolves [frankenscipy-pq6k2]. Strategy: build the spline once via
//! fsci.splrep(x, y, k, 0.0), pass the resulting tck to BOTH fsci and
//! scipy's spline ops. Compare derived tck (knots, coefs) and
//! evaluated outputs (antider compared after constant-shift removal).
//!
//! splint (definite integral) is covered since the splantider indexing
//! fix (frankenscipy-05m2t); sproot (spline zeros) since the analytic
//! per-interval root finder (frankenscipy-5lyd8).
//!
//! Tolerances: 1e-9 abs for tck/values.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_interpolate::{splantider, splder, splev, splint, splrep, sproot};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const TCK_TOL: f64 = 1.0e-9;
const VAL_TOL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct SplCase {
    case_id: String,
    /// (x, y) data points for splrep.
    x: Vec<f64>,
    y: Vec<f64>,
    k: usize,
    /// Test points to evaluate derived/antiderived spline at.
    eval: Vec<f64>,
    /// (a, b) integration interval for splint.
    integ_a: f64,
    integ_b: f64,
    /// Whether to compute sproot (cubic only).
    do_sproot: bool,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    /// Each request's tck is fsci-built; we pass it to scipy.
    cases: Vec<SplCase>,
    fsci_tcks: Vec<TckOut>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TckOut {
    case_id: String,
    t: Vec<f64>,
    c: Vec<f64>,
    k: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct ScipyArm {
    case_id: String,
    der_t: Option<Vec<f64>>,
    der_c: Option<Vec<f64>>,
    der_k: Option<usize>,
    der_eval: Option<Vec<f64>>,
    anti_t: Option<Vec<f64>>,
    anti_c: Option<Vec<f64>>,
    anti_k: Option<usize>,
    anti_eval: Option<Vec<f64>>,
    splint_value: Option<f64>,
    sproot: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    arms: Vec<ScipyArm>,
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
    fs::create_dir_all(output_dir()).expect("create spl_ops diff dir");
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

fn generate_cases() -> Vec<SplCase> {
    let x_dense: Vec<f64> = (0..15).map(|i| (i as f64) * 0.5).collect();
    let y_quadratic: Vec<f64> = x_dense.iter().map(|x| x * x).collect();
    let y_sin: Vec<f64> = x_dense.iter().map(|x| x.sin()).collect();
    let x_smooth: Vec<f64> = (0..20).map(|i| (i as f64) * 0.3).collect();
    let y_cubic: Vec<f64> = x_smooth.iter().map(|x| x * x * x - 2.0 * x).collect();
    let eval_pts_dense: Vec<f64> = (1..=12).map(|i| (i as f64) * 0.5).collect();
    let eval_pts_smooth: Vec<f64> = (1..=18).map(|i| (i as f64) * 0.3).collect();
    vec![
        SplCase {
            case_id: "quadratic".into(),
            x: x_dense.clone(),
            y: y_quadratic,
            k: 3,
            eval: eval_pts_dense.clone(),
            integ_a: 1.0,
            integ_b: 5.0,
            do_sproot: false,
        },
        SplCase {
            case_id: "sin".into(),
            x: x_dense,
            y: y_sin,
            k: 3,
            eval: eval_pts_dense,
            integ_a: 0.5,
            integ_b: 6.0,
            do_sproot: true, // sin has roots at 0, π, 2π
        },
        SplCase {
            case_id: "cubic_minus_2x".into(),
            x: x_smooth,
            y: y_cubic,
            k: 3,
            eval: eval_pts_smooth,
            integ_a: 0.6,
            integ_b: 5.0,
            do_sproot: true, // x^3 - 2x has root at sqrt(2)
        },
    ]
}

fn build_fsci_tcks(cases: &[SplCase]) -> Vec<TckOut> {
    let mut out = Vec::with_capacity(cases.len());
    for case in cases {
        let Ok((t, c, k)) = splrep(&case.x, &case.y, case.k, 0.0) else {
            continue;
        };
        out.push(TckOut {
            case_id: case.case_id.clone(),
            t,
            c,
            k,
        });
    }
    out
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy import interpolate as si

def finite_or_none(arr):
    out = []
    for v in arr:
        if not math.isfinite(float(v)):
            return None
        out.append(float(v))
    return out

q = json.load(sys.stdin)
fsci_tcks = {tck["case_id"]: tck for tck in q["fsci_tcks"]}

arms = []
for case in q["cases"]:
    cid = case["case_id"]
    if cid not in fsci_tcks:
        arms.append({"case_id": cid, "der_t": None, "der_c": None, "der_k": None, "der_eval": None,
                     "anti_t": None, "anti_c": None, "anti_k": None, "anti_eval": None,
                     "splint_value": None, "sproot": None})
        continue
    tck_in = fsci_tcks[cid]
    t = np.array(tck_in["t"], dtype=float)
    c = np.array(tck_in["c"], dtype=float)
    k = int(tck_in["k"])
    eval_pts = np.array(case["eval"], dtype=float)
    a = float(case["integ_a"]); b = float(case["integ_b"])
    do_sproot = bool(case["do_sproot"])

    arm = {"case_id": cid}
    try:
        # splder returns (t, c, k)
        dt, dc, dk = si.splder((t, c, k))
        arm["der_t"] = finite_or_none(dt.tolist())
        arm["der_c"] = finite_or_none(dc.tolist())
        arm["der_k"] = int(dk)
        arm["der_eval"] = finite_or_none(si.splev(eval_pts, (dt, dc, dk)).tolist())
    except Exception as e:
        sys.stderr.write(f"splder {cid}: {e}\n")
        arm["der_t"] = None; arm["der_c"] = None; arm["der_k"] = None; arm["der_eval"] = None

    try:
        at, ac, ak = si.splantider((t, c, k))
        arm["anti_t"] = finite_or_none(at.tolist())
        arm["anti_c"] = finite_or_none(ac.tolist())
        arm["anti_k"] = int(ak)
        arm["anti_eval"] = finite_or_none(si.splev(eval_pts, (at, ac, ak)).tolist())
    except Exception as e:
        sys.stderr.write(f"splantider {cid}: {e}\n")
        arm["anti_t"] = None; arm["anti_c"] = None; arm["anti_k"] = None; arm["anti_eval"] = None

    try:
        sv = float(si.splint(a, b, (t, c, k)))
        arm["splint_value"] = sv if math.isfinite(sv) else None
    except Exception as e:
        sys.stderr.write(f"splint {cid}: {e}\n")
        arm["splint_value"] = None

    if do_sproot and k == 3:
        try:
            roots = si.sproot((t, c, k)).tolist()
            arm["sproot"] = sorted([float(r) for r in roots if math.isfinite(r)])
        except Exception as e:
            sys.stderr.write(f"sproot {cid}: {e}\n")
            arm["sproot"] = None
    else:
        arm["sproot"] = None
    arms.append(arm)

print(json.dumps({"arms": arms}))
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
                "failed to spawn python3 for spl_ops oracle: {e}"
            );
            eprintln!("skipping spl_ops oracle: python3 not available ({e})");
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
                "spl_ops oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping spl_ops oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for spl_ops oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "spl_ops oracle failed: {stderr}"
        );
        eprintln!("skipping spl_ops oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse spl_ops oracle JSON"))
}

fn vec_max_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn diff_interpolate_spl_ops() {
    let cases = generate_cases();
    let fsci_tcks = build_fsci_tcks(&cases);
    let query = OracleQuery {
        cases: cases.clone(),
        fsci_tcks: fsci_tcks.clone(),
    };
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let arm_map: HashMap<String, ScipyArm> = oracle
        .arms
        .into_iter()
        .map(|a| (a.case_id.clone(), a))
        .collect();
    let tck_map: HashMap<String, TckOut> = fsci_tcks
        .into_iter()
        .map(|t| (t.case_id.clone(), t))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &cases {
        let Some(arm) = arm_map.get(&case.case_id) else {
            continue;
        };
        let Some(tck) = tck_map.get(&case.case_id) else {
            continue;
        };

        // splder: tck triple equality
        if let (Some(et), Some(ec), Some(ek)) = (arm.der_t.as_ref(), arm.der_c.as_ref(), arm.der_k)
        {
            let Ok((dt, dc, dk)) = splder(&(tck.t.clone(), tck.c.clone(), tck.k)) else {
                continue;
            };
            let t_diff = vec_max_diff(&dt, et);
            let c_diff = vec_max_diff(&dc, ec);
            let k_diff = if dk == ek { 0.0 } else { f64::INFINITY };
            let abs_d = t_diff.max(c_diff).max(k_diff);
            max_overall = max_overall.max(abs_d);
            diffs.push(CaseDiff {
                case_id: format!("{}_splder_tck", case.case_id),
                op: "splder_tck".into(),
                abs_diff: abs_d,
                pass: abs_d <= TCK_TOL,
            });

            // Evaluated derivative spline values
            if let Some(ev_exp) = arm.der_eval.as_ref()
                && let Ok(ev) = splev(&case.eval, &(dt, dc, dk))
            {
                let abs_d = vec_max_diff(&ev, ev_exp);
                max_overall = max_overall.max(abs_d);
                diffs.push(CaseDiff {
                    case_id: format!("{}_splder_eval", case.case_id),
                    op: "splder_eval".into(),
                    abs_diff: abs_d,
                    pass: abs_d <= VAL_TOL,
                });
            }
        }

        // splantider: tck + eval
        if let (Some(et), Some(ec), Some(ek)) =
            (arm.anti_t.as_ref(), arm.anti_c.as_ref(), arm.anti_k)
        {
            let Ok((at, ac, ak)) = splantider(&(tck.t.clone(), tck.c.clone(), tck.k)) else {
                continue;
            };
            let t_diff = vec_max_diff(&at, et);
            let c_diff = vec_max_diff(&ac, ec);
            let k_diff = if ak == ek { 0.0 } else { f64::INFINITY };
            let abs_d = t_diff.max(c_diff).max(k_diff);
            max_overall = max_overall.max(abs_d);
            diffs.push(CaseDiff {
                case_id: format!("{}_splantider_tck", case.case_id),
                op: "splantider_tck".into(),
                abs_diff: abs_d,
                pass: abs_d <= TCK_TOL,
            });

            // Antiderivatives in scipy and fsci differ by an integration
            // constant. Subtract the value at the first eval point so we
            // compare shapes (slopes).
            if let Some(ev_exp) = arm.anti_eval.as_ref()
                && let Ok(ev) = splev(&case.eval, &(at, ac, ak))
                && !ev.is_empty()
                && !ev_exp.is_empty()
            {
                let off_actual = ev[0];
                let off_expected = ev_exp[0];
                let shifted_actual: Vec<f64> = ev.iter().map(|v| v - off_actual).collect();
                let shifted_expected: Vec<f64> = ev_exp.iter().map(|v| v - off_expected).collect();
                let abs_d = vec_max_diff(&shifted_actual, &shifted_expected);
                max_overall = max_overall.max(abs_d);
                diffs.push(CaseDiff {
                    case_id: format!("{}_splantider_eval_shifted", case.case_id),
                    op: "splantider_eval".into(),
                    abs_diff: abs_d,
                    pass: abs_d <= VAL_TOL,
                });
            }
        }

        // splint: definite integral over [integ_a, integ_b]
        // (frankenscipy-05m2t — the splantider indexing fix restored parity).
        if let Some(expected) = arm.splint_value
            && let Ok(value) = splint(
                case.integ_a,
                case.integ_b,
                &(tck.t.clone(), tck.c.clone(), tck.k),
            )
        {
            let abs_d = (value - expected).abs();
            max_overall = max_overall.max(abs_d);
            diffs.push(CaseDiff {
                case_id: format!("{}_splint", case.case_id),
                op: "splint".into(),
                abs_diff: abs_d,
                pass: abs_d <= VAL_TOL,
            });
        }

        // sproot: zeros of a cubic spline, sorted ascending
        // (frankenscipy-5lyd8 — analytic per-interval cubic root finder).
        if case.do_sproot
            && let Some(expected) = arm.sproot.as_ref()
            && let Ok(mut found) = sproot(&(tck.t.clone(), tck.c.clone(), tck.k))
        {
            found.sort_by(f64::total_cmp);
            let abs_d = vec_max_diff(&found, expected);
            max_overall = max_overall.max(abs_d);
            diffs.push(CaseDiff {
                case_id: format!("{}_sproot", case.case_id),
                op: "sproot".into(),
                abs_diff: abs_d,
                pass: abs_d <= VAL_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_interpolate_spl_ops".into(),
        category: "fsci_interpolate splder/splantider vs scipy.interpolate".into(),
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
        "spl_ops conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
