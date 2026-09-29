#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `fsci_integrate::quad` as QUADPACK
//! (frankenscipy-1ksfv.8): QAGSE on finite ranges, QAGIE on infinite ones, QAGPE with
//! `points`, and the weighted integrators behind `quad(weight=...)` (QAWOE/QAWFE for cos/sin,
//! QAWSE for the algebraic–logarithmic weights, QAWCE for Cauchy principal values), each
//! compared with `scipy.integrate.quad(..., full_output=1)`.
//!
//! Per case: QUADPACK's `ier` must match. When SciPy converged (`ier == 0`) the values must
//! agree to QUADPACK's own accuracy request `max(epsabs, epsrel·|I|)`, the error estimates to
//! within `ABSERR_RATIO_TOL`, and `neval` to within `NEVAL_RATIO_TOL`. The port is statement
//! by statement, so most cases match `(neval, last, ier)` exactly; the log counts them.
//!
//! Every case must be compared: a SciPy failure or an fsci error is a FAILED case, not a
//! skipped one (frankenscipy-olv0j.1).
//!
//! Not compared: a cos weight with `omega == 0` on `[a, ∞)`, where SciPy (as netlib QUADPACK
//! `dqawfe`) integrates from 0 instead of `a` and fsci deliberately does not.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_integrate::{
    QuadInfo, QuadOptions, QuadResult, QuadWeight, QuadWeightOptions, quad_full_output,
    quad_weighted_full_output,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-008";
const ABSERR_RATIO_TOL: f64 = 10.0;
const NEVAL_RATIO_TOL: f64 = 2.0;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

/// One comparison. `weight` is SciPy's weight name ("" for none) with `wvar`.
struct Case {
    id: &'static str,
    func: &'static str,
    a: f64,
    b: f64,
    points: Vec<f64>,
    weight: &'static str,
    wvar: Vec<f64>,
}

fn plain(id: &'static str, func: &'static str, a: f64, b: f64, points: &[f64]) -> Case {
    Case {
        id,
        func,
        a,
        b,
        points: points.to_vec(),
        weight: "",
        wvar: Vec::new(),
    }
}

fn weighted(
    id: &'static str,
    func: &'static str,
    a: f64,
    b: f64,
    weight: &'static str,
    wvar: &[f64],
) -> Case {
    Case {
        id,
        func,
        a,
        b,
        points: Vec::new(),
        weight,
        wvar: wvar.to_vec(),
    }
}

fn cases() -> Vec<Case> {
    let inf = f64::INFINITY;
    vec![
        plain("x2_0_1", "x2", 0.0, 1.0, &[]),
        plain("x2_m3_2", "x2", -3.0, 2.0, &[]),
        plain("x2_reversed", "x2", 2.0, 0.0, &[]),
        plain("sin_0_pi", "sin", 0.0, std::f64::consts::PI, &[]),
        plain("sin_0_10", "sin", 0.0, 10.0, &[]),
        plain("cos30_0_1", "cos30", 0.0, 1.0, &[]),
        plain("runge_m1_1", "runge", -1.0, 1.0, &[]),
        plain("sqrt_0_1", "sqrt", 0.0, 1.0, &[]),
        plain("pow_m05_0_1", "pow_m05", 0.0, 1.0, &[]),
        plain("pow_m09_0_1", "pow_m09", 0.0, 1.0, &[]),
        plain("pow_m099_0_1", "pow_m099", 0.0, 1.0, &[]),
        plain("log_0_1", "log", 0.0, 1.0, &[]),
        plain("cos_over_sqrt_0_1", "cos_over_sqrt", 0.0, 1.0, &[]),
        plain("abs_x1_0_3", "abs_x1", 0.0, 3.0, &[]),
        plain("abs_x1_0_3_points", "abs_x1", 0.0, 3.0, &[1.0]),
        plain("inv_sqrt_abs03_points", "inv_sqrt_abs03", 0.0, 1.0, &[0.3]),
        plain("log_abs05_points", "log_abs05", 0.0, 1.0, &[0.5]),
        plain("step_0_1", "step", 0.0, 1.0, &[]),
        plain("step_0_1_points", "step", 0.0, 1.0, &[0.5]),
        plain("sin_1e4_0_1", "sin_1e4", 0.0, 1.0, &[]),
        plain("lorentz_0_inf", "lorentz", 0.0, inf, &[]),
        plain("lorentz_R", "lorentz", -inf, inf, &[]),
        plain("gauss_R", "gauss", -inf, inf, &[]),
        plain("gauss_0_inf", "gauss", 0.0, inf, &[]),
        plain("exp_minf_1", "exp", -inf, 1.0, &[]),
        plain("x_exp_mx_0_inf", "x_exp_mx", 0.0, inf, &[]),
        plain("inv_x_1_inf", "inv_x", 1.0, inf, &[]),
        plain("inv_x2_1_inf", "inv_x2", 1.0, inf, &[]),
        weighted("cauchy_one", "one", -1.0, 2.0, "cauchy", &[0.0]),
        weighted("cauchy_exp", "exp", -1.0, 1.0, "cauchy", &[0.3]),
        weighted("cauchy_lorentz", "lorentz", 0.0, 5.0, "cauchy", &[2.0]),
        weighted("cauchy_outside", "exp", 0.0, 1.0, "cauchy", &[2.5]),
        weighted("alg_cheb", "one", -1.0, 1.0, "alg", &[-0.5, -0.5]),
        weighted("alg_exp", "exp", 0.0, 2.0, "alg", &[0.3, 1.2]),
        weighted("alg_loga_cos", "cos", 0.0, 1.0, "alg-loga", &[0.5, 0.25]),
        weighted("alg_logb_exp", "exp", 0.0, 1.0, "alg-logb", &[0.0, -0.5]),
        weighted("alg_log_one", "one", 0.0, 1.0, "alg-log", &[-0.3, -0.3]),
        weighted("cos10_expm", "expm", 0.0, 1.0, "cos", &[10.0]),
        weighted("sin50_lorentz", "lorentz", 0.0, 3.0, "sin", &[50.0]),
        weighted("sin_neg30_exp", "exp", 0.0, 2.0, "sin", &[-30.0]),
        weighted("cos200_gauss", "gauss", -3.0, 3.0, "cos", &[200.0]),
        weighted("cos1_expm_inf", "expm", 0.0, inf, "cos", &[1.0]),
        weighted("sin2_lorentz_inf", "lorentz", 0.0, inf, "sin", &[2.0]),
        weighted("sin3_invsqrt_inf", "inv_sqrt", 1.0, inf, "sin", &[3.0]),
        weighted("cos1_exp_neg_half_line", "exp", -inf, 0.0, "cos", &[1.0]),
    ]
}

fn integrand(name: &str, x: f64) -> f64 {
    match name {
        "one" => 1.0,
        "x2" => x * x,
        "sin" => x.sin(),
        "cos" => x.cos(),
        "cos30" => (30.0 * x).cos(),
        "runge" => 1.0 / (1.0 + 25.0 * x * x),
        "sqrt" => x.sqrt(),
        "inv_sqrt" => 1.0 / x.sqrt(),
        "pow_m05" => x.powf(-0.5),
        "pow_m09" => x.powf(-0.9),
        "pow_m099" => x.powf(-0.99),
        "log" => x.ln(),
        "cos_over_sqrt" => x.cos() / x.sqrt(),
        "abs_x1" => (x - 1.0).abs(),
        "inv_sqrt_abs03" => (x - 0.3).abs().powf(-0.5),
        "log_abs05" => (x - 0.5).abs().ln(),
        "step" => {
            if x < 0.5 {
                1.0
            } else {
                0.0
            }
        }
        "sin_1e4" => (1e4 * x).sin(),
        "lorentz" => 1.0 / (1.0 + x * x),
        "gauss" => (-x * x).exp(),
        "exp" => x.exp(),
        "expm" => (-x).exp(),
        "x_exp_mx" => x * (-x).exp(),
        "inv_x" => 1.0 / x,
        "inv_x2" => 1.0 / (x * x),
        _ => f64::NAN,
    }
}

fn weight_of(case: &Case) -> Option<QuadWeight> {
    let (alpha, beta) = (
        case.wvar.first().copied().unwrap_or(0.0),
        case.wvar.get(1).copied().unwrap_or(0.0),
    );
    match case.weight {
        "cos" => Some(QuadWeight::Cos(alpha)),
        "sin" => Some(QuadWeight::Sin(alpha)),
        "alg" => Some(QuadWeight::Alg { alpha, beta }),
        "alg-loga" => Some(QuadWeight::AlgLogA { alpha, beta }),
        "alg-logb" => Some(QuadWeight::AlgLogB { alpha, beta }),
        "alg-log" => Some(QuadWeight::AlgLog { alpha, beta }),
        "cauchy" => Some(QuadWeight::Cauchy(alpha)),
        _ => None,
    }
}

#[derive(Debug, Clone, Serialize)]
struct QueryCase {
    case_id: String,
    func: String,
    a: f64,
    b: f64,
    points: Vec<f64>,
    weight: String,
    wvar: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleArm {
    case_id: String,
    value: Option<f64>,
    abserr: Option<f64>,
    neval: Option<usize>,
    last: Option<usize>,
    ier: Option<u8>,
}

/// `(value, abserr, neval, last, ier)`.
type Row = (f64, f64, usize, usize, u8);

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    fsci: Row,
    scipy: Row,
    exact_structure: bool,
    pass: bool,
    reason: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    exact_structure_count: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

/// Infinite bounds travel as ±1e308 in JSON.
fn encode(v: f64) -> f64 {
    if v == f64::INFINITY {
        1e308
    } else if v == f64::NEG_INFINITY {
        -1e308
    } else {
        v
    }
}

fn scipy_oracle_or_skip(query: &[QueryCase]) -> Option<Vec<OracleArm>> {
    let script = r#"
import json, math, sys, warnings
from scipy import integrate

FUNCS = {
    "one": lambda x: 1.0,
    "x2": lambda x: x * x,
    "sin": math.sin,
    "cos": math.cos,
    "cos30": lambda x: math.cos(30.0 * x),
    "runge": lambda x: 1.0 / (1.0 + 25.0 * x * x),
    "sqrt": math.sqrt,
    "inv_sqrt": lambda x: 1.0 / math.sqrt(x),
    "pow_m05": lambda x: x ** -0.5,
    "pow_m09": lambda x: x ** -0.9,
    "pow_m099": lambda x: x ** -0.99,
    "log": math.log,
    "cos_over_sqrt": lambda x: math.cos(x) / math.sqrt(x),
    "abs_x1": lambda x: abs(x - 1.0),
    "inv_sqrt_abs03": lambda x: abs(x - 0.3) ** -0.5,
    "log_abs05": lambda x: math.log(abs(x - 0.5)),
    "step": lambda x: 1.0 if x < 0.5 else 0.0,
    "sin_1e4": lambda x: math.sin(1e4 * x),
    "lorentz": lambda x: 1.0 / (1.0 + x * x),
    "gauss": lambda x: math.exp(-x * x),
    "exp": math.exp,
    "expm": lambda x: math.exp(-x),
    "x_exp_mx": lambda x: x * math.exp(-x),
    "inv_x": lambda x: 1.0 / x,
    "inv_x2": lambda x: 1.0 / (x * x),
}

def bound(v):
    return math.inf if v >= 1e308 else (-math.inf if v <= -1e308 else v)

# quad's ier is not returned; with full_output it is recoverable from the message.
MESSAGES = [
    ("maximum number of subdivisions", 1), ("roundoff error is detected, which", 2),
    ("Extremely bad integrand", 3), ("does not converge", 4), ("divergent", 5),
    ("maximum number of cycles", 1), ("extrapolation table constructed", 4),
    ("within one or more of the cycles", 7),
]

out = []
for case in json.load(sys.stdin):
    arm = {"case_id": case["case_id"], "value": None, "abserr": None, "neval": None,
           "last": None, "ier": None}
    try:
        kw = {}
        if case["points"]:
            kw["points"] = case["points"]
        if case["weight"]:
            kw["weight"] = case["weight"]
            wvar = case["wvar"]
            kw["wvar"] = wvar[0] if len(wvar) == 1 else tuple(wvar)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = integrate.quad(FUNCS[case["func"]], bound(case["a"]), bound(case["b"]),
                               full_output=1, **kw)
        info = r[2]
        ier = 0
        if len(r) > 3:
            ier = next((code for text, code in MESSAGES if text in r[3]), 99)
        last = info["last"] if "last" in info else info["lst"]
        arm.update(value=float(r[0]), abserr=float(r[1]), neval=int(info["neval"]),
                   last=int(last), ier=ier)
    except Exception:
        pass
    out.append(arm)
print(json.dumps(out))
"#;
    let query_json = serde_json::to_string(query).expect("serialize quadpack query");
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
                "failed to spawn python3 for the quadpack oracle: {e}"
            );
            eprintln!("skipping quadpack oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open quadpack oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "quadpack oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping quadpack oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for quadpack oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "quadpack oracle failed: {stderr}"
        );
        eprintln!("skipping quadpack oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse quadpack oracle JSON"))
}

fn fsci_row(case: &Case, options: QuadOptions) -> Result<Row, String> {
    let f = |x: f64| integrand(case.func, x);
    let run: Result<(QuadResult, QuadInfo), _> = match weight_of(case) {
        Some(weight) => quad_weighted_full_output(
            f,
            case.a,
            case.b,
            weight,
            options,
            QuadWeightOptions::default(),
        ),
        None => quad_full_output(f, case.a, case.b, &case.points, options),
    };
    run.map(|(r, info)| (r.integral, r.error, r.neval, info.last, info.ier))
        .map_err(|e| e.to_string())
}

#[test]
fn diff_integrate_quad_quadpack() {
    let cases = cases();
    let query: Vec<QueryCase> = cases
        .iter()
        .map(|c| QueryCase {
            case_id: c.id.to_string(),
            func: c.func.to_string(),
            a: encode(c.a),
            b: encode(c.b),
            points: c.points.clone(),
            weight: c.weight.to_string(),
            wvar: c.wvar.clone(),
        })
        .collect();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    let arms: HashMap<String, OracleArm> = oracle
        .into_iter()
        .map(|arm| (arm.case_id.clone(), arm))
        .collect();

    let start = Instant::now();
    let options = QuadOptions::default();
    let missing: Row = (f64::NAN, f64::NAN, 0, 0, 0);
    let mut diffs = Vec::new();
    let mut ledger = CompareLedger::new("diff_integrate_quad_quadpack", &["quad"]);
    for case in &cases {
        let arm = &arms[case.id];
        let fsci = fsci_row(case, options);
        let scipy = match (arm.value, arm.abserr, arm.neval, arm.last, arm.ier) {
            (Some(v), Some(e), Some(n), Some(l), Some(i)) => Some((v, e, n, l, i)),
            _ => None,
        };
        // The ledger records a missing SciPy row or an fsci error; the case is still logged as
        // a failed diff below, as before.
        let both = ledger.both("quad", case.id, scipy, fsci.as_ref().ok().copied());
        let (pass, reason, fsci_row, scipy_row) = match (both, fsci) {
            (Some((s, f)), _) => {
                let mut problems = Vec::new();
                if f.4 != s.4 {
                    problems.push(format!("ier {} vs {}", f.4, s.4));
                }
                // The abserr ratio below takes max/min, which drop a NaN: a NaN value or error
                // estimate from fsci must match a NaN from SciPy.
                if (f.0.is_nan() && !s.0.is_nan()) || (f.1.is_nan() && !s.1.is_nan()) {
                    problems.push(format!("fsci value/abserr NaN: {f:?} vs {s:?}"));
                }
                if s.4 == 0 {
                    let request = options.epsabs.max(options.epsrel * s.0.abs());
                    if !((f.0 - s.0).abs() <= request) {
                        problems.push(format!("value off by {:e} > {request:e}", f.0 - s.0));
                    }
                    let (hi, lo) = (f.1.max(s.1), f.1.min(s.1));
                    if hi > 0.0 && !(hi <= ABSERR_RATIO_TOL * lo) {
                        problems.push(format!("abserr {:e} vs {:e}", f.1, s.1));
                    }
                    let (n_hi, n_lo) = (f.2.max(s.2) as f64, f.2.min(s.2) as f64);
                    if n_hi > NEVAL_RATIO_TOL * n_lo {
                        problems.push(format!("neval {} vs {}", f.2, s.2));
                    }
                } else if !f.0.is_finite() && s.0.is_finite() {
                    problems.push("fsci value is not finite".to_string());
                }
                ledger.compared("quad", case.id, problems.is_empty());
                (problems.is_empty(), problems.join("; "), f, s)
            }
            (None, Err(e)) => (
                false,
                format!("fsci error {e}"),
                missing,
                scipy.unwrap_or(missing),
            ),
            (None, Ok(f)) => (false, "SciPy produced no result".to_string(), f, missing),
        };
        let exact =
            pass && (fsci_row.2, fsci_row.3, fsci_row.4) == (scipy_row.2, scipy_row.3, scipy_row.4);
        diffs.push(CaseDiff {
            case_id: case.id.to_string(),
            fsci: fsci_row,
            scipy: scipy_row,
            exact_structure: exact,
            pass,
            reason,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let exact_structure_count = diffs.iter().filter(|d| d.exact_structure).count();
    let log = DiffLog {
        test_id: "diff_integrate_quad_quadpack".into(),
        category:
            "scipy.integrate.quad full_output (QUADPACK qagse/qagie/qagpe/qawoe/qawfe/qawse/qawce)"
                .into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        exact_structure_count,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    fs::create_dir_all(output_dir()).expect("create quadpack diff dir");
    fs::write(
        output_dir().join("diff_integrate_quad_quadpack.json"),
        serde_json::to_string_pretty(&log).expect("serialize quadpack log"),
    )
    .expect("write quadpack log");

    for d in &diffs {
        println!(
            "{} fsci={:?} scipy={:?} exact={} {}",
            d.case_id, d.fsci, d.scipy, d.exact_structure, d.reason
        );
    }
    println!(
        "{} cases compared, {exact_structure_count} with identical (neval, last, ier)",
        diffs.len()
    );
    assert_eq!(diffs.len(), cases.len(), "every case must be compared");
    assert!(all_pass, "quad vs scipy.integrate.quad (QUADPACK) failed");
    ledger.finish(cases.len());
}
