#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.stats.genpareto`.
//!
//! Covers bounded support (including endpoint density and survival), the
//! exponential limit, heavy tails, log probabilities, inverse tails, and
//! fixed-location/scale maximum likelihood (frankenscipy-1ksfv.14).
//! Every requested output is counted, including matching infinities.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::compare_ledger::oracle_f64;
use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_stats::{ContinuousDistribution, GenPareto};
use rand::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const PDF_TOL: f64 = 1.0e-12;
const CDF_TOL: f64 = 1.0e-12;
const PPF_TOL_REL: f64 = 1.0e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    c: f64,
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct PpfCase {
    case_id: String,
    c: f64,
    q: f64,
}

#[derive(Debug, Clone, Serialize)]
struct FitCase {
    case_id: String,
    data: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
    ppf: Vec<PpfCase>,
    fit: Vec<FitCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    #[serde(default, deserialize_with = "oracle_f64")]
    pdf: Option<f64>,
    #[serde(default, deserialize_with = "oracle_f64")]
    logpdf: Option<f64>,
    #[serde(default, deserialize_with = "oracle_f64")]
    cdf: Option<f64>,
    #[serde(default, deserialize_with = "oracle_f64")]
    logcdf: Option<f64>,
    #[serde(default, deserialize_with = "oracle_f64")]
    sf: Option<f64>,
    #[serde(default, deserialize_with = "oracle_f64")]
    logsf: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct PpfArm {
    case_id: String,
    #[serde(default, deserialize_with = "oracle_f64")]
    ppf: Option<f64>,
    #[serde(default, deserialize_with = "oracle_f64")]
    isf: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct FitArm {
    case_id: String,
    c: Option<f64>,
    log_likelihood: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
    ppf: Vec<PpfArm>,
    fit: Vec<FitArm>,
    scipy_version: String,
    numpy_version: String,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    family: String,
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
    fs::create_dir_all(output_dir()).expect("create genpareto diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize genpareto diff log");
    fs::write(path, json).expect("write genpareto diff log");
}

fn generate_query() -> OracleQuery {
    // GenPareto support depends on c:
    //   c >= 0: x in [0, ∞)
    //   c <  0: x in [0, -1/c]
    let cs = [-2.0_f64, -1.0, -0.5, -1e-12, 0.0, 1e-12, 0.3, 0.5, 1.0];
    let qs = [
        0.0_f64, 1e-30, 1e-16, 1e-8, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0,
    ];
    let mut points = Vec::new();
    for &c in &cs {
        let upper = if c < 0.0 { (-1.0 / c) * 0.95 } else { 5.0 };
        for k in 0..7u32 {
            let frac = 0.05 + 0.9 * (k as f64) / 6.0;
            let x = frac * upper;
            points.push(PointCase {
                case_id: format!("c{c}_x{x:.4}"),
                c,
                x,
            });
        }
        let mut xs = vec![-0.1, 0.0, 1e-30];
        if c < 0.0 {
            let endpoint = -1.0 / c;
            xs.extend([endpoint, endpoint.next_up(), 2.0 * endpoint]);
        } else {
            xs.extend([100.0, 1e100]);
        }
        for (index, x) in xs.into_iter().enumerate() {
            points.push(PointCase {
                case_id: format!("boundary_c{c}_{index}"),
                c,
                x,
            });
        }
    }
    let mut ppf_cases = Vec::new();
    for &c in &cs {
        for &q in &qs {
            ppf_cases.push(PpfCase {
                case_id: format!("c{c}_q{q}"),
                c,
                q,
            });
        }
    }
    let mut fit = vec![
        FitCase {
            case_id: "fit_mom_is_not_mle".into(),
            data: vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        },
        FitCase {
            case_id: "fit_uniform_endpoint".into(),
            data: vec![0.1, 0.2, 1.0],
        },
        FitCase {
            case_id: "fit_zero_shape".into(),
            data: vec![2.0; 8],
        },
        FitCase {
            case_id: "fit_mom_excludes_observation".into(),
            data: vec![0.0, 0.1, 0.2, 0.9, 1.1],
        },
        FitCase {
            case_id: "fit_heavy_tail".into(),
            data: vec![0.1, 0.4, 2.0, 10.0, 100.0, 1000.0],
        },
    ];
    for c in [-0.75_f64, -0.25, 0.0, 0.25, 0.75, 2.0] {
        for seed in [17, 29, 43] {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let data = (0..2_000)
                .map(|_| {
                    let q = rng.random::<f64>();
                    let t = -(-q).ln_1p();
                    if c == 0.0 { t } else { (c * t).exp_m1() / c }
                })
                .collect();
            fit.push(FitCase {
                case_id: format!("fit_c{c}_seed{seed}"),
                data,
            });
        }
    }
    OracleQuery {
        points,
        ppf: ppf_cases,
        fit,
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
import numpy as np
import scipy
from scipy.optimize import fmin
from scipy.stats import genpareto

def fval(value):
    value = float(value)
    if math.isfinite(value):
        return value
    return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")

def optimizer(func, x0, args=(), disp=0):
    result = fmin(func, x0, args=args, disp=0, xtol=1e-12, ftol=1e-12,
                  maxiter=4000, maxfun=8000, full_output=True)
    if result[4] != 0:
        raise RuntimeError("SciPy likelihood fit did not converge")
    return result[0]

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    c = float(case["c"]); x = float(case["x"])
    try:
        points.append({
            "case_id": cid,
            "pdf": fval(genpareto.pdf(x, c)),
            "logpdf": fval(genpareto.logpdf(x, c)),
            "cdf": fval(genpareto.cdf(x, c)),
            "logcdf": fval(genpareto.logcdf(x, c)),
            "sf": fval(genpareto.sf(x, c)),
            "logsf": fval(genpareto.logsf(x, c)),
        })
    except Exception:
        points.append({"case_id": cid, "pdf": None, "cdf": None, "sf": None})
ppf = []
for case in q["ppf"]:
    cid = case["case_id"]; c = float(case["c"]); qv = float(case["q"])
    try:
        ppf.append({"case_id": cid, "ppf": fval(genpareto.ppf(qv, c)),
                    "isf": fval(genpareto.isf(qv, c))})
    except Exception:
        ppf.append({"case_id": cid, "ppf": None, "isf": None})
fit = []
for case in q["fit"]:
    try:
        c, loc, scale = genpareto.fit(case["data"], floc=0, fscale=1, optimizer=optimizer)
        fit.append({"case_id": case["case_id"], "c": float(c),
                    "log_likelihood": float(np.sum(genpareto.logpdf(case["data"], c)))})
    except Exception as error:
        print(case["case_id"], repr(error), file=sys.stderr)
        fit.append({"case_id": case["case_id"], "c": None, "log_likelihood": None})
print(json.dumps({"points": points, "ppf": ppf, "fit": fit,
                  "scipy_version": scipy.__version__, "numpy_version": np.__version__},
                 allow_nan=False))
"#;

    let query_json = serde_json::to_string(query).expect("serialize genpareto query");
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
                "failed to spawn python3 for genpareto oracle: {e}"
            );
            eprintln!("skipping genpareto oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open genpareto oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "genpareto oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping genpareto oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for genpareto oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "genpareto oracle failed: {stderr}"
        );
        eprintln!("skipping genpareto oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse genpareto oracle JSON"))
}

#[test]
fn diff_stats_genpareto() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());
    assert_eq!(oracle.ppf.len(), query.ppf.len());
    assert_eq!(oracle.fit.len(), query.fit.len());
    eprintln!(
        "GenPareto oracle: SciPy {}, NumPy {}",
        oracle.scipy_version, oracle.numpy_version
    );

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();
    let ppfmap: HashMap<String, PpfArm> = oracle
        .ppf
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();
    let fitmap: HashMap<String, FitArm> = oracle
        .fit
        .into_iter()
        .map(|row| (row.case_id.clone(), row))
        .collect();
    assert_eq!(pmap.len(), query.points.len());
    assert_eq!(ppfmap.len(), query.ppf.len());
    assert_eq!(fitmap.len(), query.fit.len());

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;
    let mut ledger = CompareLedger::new(
        "diff_stats_genpareto",
        &[
            "pdf", "logpdf", "cdf", "logcdf", "sf", "logsf", "ppf", "isf", "fit_c", "fit_ll",
        ],
    );

    for case in &query.points {
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        let dist = GenPareto::new(case.c);
        let arms = [
            ("pdf", oracle.pdf, dist.pdf(case.x), PDF_TOL),
            ("logpdf", oracle.logpdf, dist.logpdf(case.x), PDF_TOL),
            ("cdf", oracle.cdf, dist.cdf(case.x), CDF_TOL),
            ("logcdf", oracle.logcdf, dist.logcdf(case.x), CDF_TOL),
            ("sf", oracle.sf, dist.sf(case.x), CDF_TOL),
            ("logsf", oracle.logsf, dist.logsf(case.x), CDF_TOL),
        ];
        for (family, scipy, fsci, tol) in arms {
            let Some((s, f)) = ledger.pair(family, &case.case_id, scipy, Some(fsci)) else {
                continue;
            };
            let d = (f - s).abs();
            let tol = if family.starts_with("log") {
                tol * s.abs().max(1.0)
            } else {
                tol
            };
            max_overall = max_overall.max(d);
            ledger.compared(family, &case.case_id, d <= tol);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: family.into(),
                abs_diff: d,
                pass: d <= tol,
            });
        }
    }

    for case in &query.ppf {
        let oracle = ppfmap.get(&case.case_id).expect("validated oracle");
        let dist = GenPareto::new(case.c);
        for (family, scipy, rust) in [
            ("ppf", oracle.ppf, dist.ppf(case.q)),
            ("isf", oracle.isf, dist.isf(case.q)),
        ] {
            let Some((scipy, rust)) = ledger.pair(family, &case.case_id, scipy, Some(rust)) else {
                continue;
            };
            let d = (rust - scipy).abs();
            let pass = d <= PPF_TOL_REL * scipy.abs();
            max_overall = max_overall.max(d);
            ledger.compared(family, &case.case_id, pass);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: family.into(),
                abs_diff: d,
                pass,
            });
        }
    }

    for case in &query.fit {
        let oracle = fitmap.get(&case.case_id).expect("validated fit oracle");
        let fitted = GenPareto::try_fit(&case.data);
        if let Err(error) = &fitted {
            eprintln!("{}: Rust fit failed: {error}", case.case_id);
        }
        let shape = fitted.as_ref().ok().map(|dist| dist.c);
        let likelihood = fitted
            .as_ref()
            .ok()
            .map(|dist| case.data.iter().map(|&x| dist.logpdf(x)).sum::<f64>());
        eprintln!(
            "{}: c rust={shape:?} scipy={:?}; LL rust={likelihood:?} scipy={:?}",
            case.case_id, oracle.c, oracle.log_likelihood
        );
        for (family, scipy, rust, relative_tolerance) in [
            ("fit_c", oracle.c, shape, 2e-6),
            ("fit_ll", oracle.log_likelihood, likelihood, 1e-8),
        ] {
            let Some((scipy, rust)) = ledger.pair(family, &case.case_id, scipy, rust) else {
                continue;
            };
            let d = (rust - scipy).abs();
            let pass = d <= relative_tolerance * scipy.abs().max(1.0);
            max_overall = max_overall.max(d);
            ledger.compared(family, &case.case_id, pass);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                family: family.into(),
                abs_diff: d,
                pass,
            });
        }
    }

    let all_pass = ledger.counts().values().all(|counts| {
        counts.compared_failed == 0 && counts.rust_failed == 0 && counts.oracle_missing == 0
    });

    let log = DiffLog {
        test_id: "diff_stats_genpareto".into(),
        category: "scipy.stats.genpareto".into(),
        case_count: ledger.counts().values().map(|arm| arm.compared_cases).sum(),
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
            eprintln!(
                "genpareto {} mismatch: {} abs_diff={}",
                d.family, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.stats.genpareto conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    let counts = ledger.finish(1);
    for arm in ["pdf", "logpdf", "cdf", "logcdf", "sf", "logsf"] {
        assert_eq!(counts[arm].compared_cases, query.points.len(), "{arm}");
    }
    for arm in ["ppf", "isf"] {
        assert_eq!(counts[arm].compared_cases, query.ppf.len(), "{arm}");
    }
    for arm in ["fit_c", "fit_ll"] {
        assert_eq!(counts[arm].compared_cases, query.fit.len(), "{arm}");
    }
}
