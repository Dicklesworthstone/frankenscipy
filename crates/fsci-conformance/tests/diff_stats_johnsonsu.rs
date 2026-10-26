#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.stats.johnsonsu`.
//!
//! Resolves [frankenscipy-k8ilu]. JohnsonSU has anchor tests in
//! `fsci-stats/src/lib.rs` but no dedicated scipy diff harness.
//! 4 a-values × 4 b-values × 7 x-values × 3 families (pdf/cdf/sf)
//! + 4 × 4 × 5 ppf cases via subprocess.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_stats::{ContinuousDistribution, JohnsonSU};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
// JohnsonSU composes Φ(a + b·sinh⁻¹(x)) and the standard normal
// helper has its own precision floor; loosen cdf slightly to
// absorb that, and ppf which inverts the same path.
const PDF_TOL: f64 = 1.0e-12;
const CDF_TOL: f64 = 1.0e-11;
const PPF_TOL_REL: f64 = 1.0e-7;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    a: f64,
    b: f64,
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct PpfCase {
    case_id: String,
    a: f64,
    b: f64,
    q: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
    ppf: Vec<PpfCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    pdf: Option<f64>,
    cdf: Option<f64>,
    sf: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct PpfArm {
    case_id: String,
    ppf: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
    ppf: Vec<PpfArm>,
    fits: Vec<FitArm>,
}

#[derive(Debug, Clone, Deserialize)]
struct FitArm {
    case_id: String,
    data: Vec<f64>,
    a: f64,
    b: f64,
    log_likelihood: f64,
    scipy_fit_log_likelihood: f64,
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
    fs::create_dir_all(output_dir()).expect("create johnsonsu diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize johnsonsu diff log");
    fs::write(path, json).expect("write johnsonsu diff log");
}

fn generate_query() -> OracleQuery {
    let aas = [-1.0_f64, 0.0, 0.5, 1.5];
    let bs = [0.7_f64, 1.0, 2.0, 3.0];
    let xs = [-3.0_f64, -1.0, -0.3, 0.0, 0.5, 1.5, 3.0];
    let qs = [0.05_f64, 0.25, 0.5, 0.75, 0.95];
    let mut points = Vec::new();
    for &a in &aas {
        for &b in &bs {
            for &x in &xs {
                points.push(PointCase {
                    case_id: format!("a{a}_b{b}_x{x}"),
                    a,
                    b,
                    x,
                });
            }
        }
    }
    let mut ppf_cases = Vec::new();
    for &a in &aas {
        for &b in &bs {
            for &q in &qs {
                ppf_cases.push(PpfCase {
                    case_id: format!("a{a}_b{b}_q{q}"),
                    a,
                    b,
                    q,
                });
            }
        }
    }
    OracleQuery {
        points,
        ppf: ppf_cases,
    }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
import numpy as np
from scipy.stats import johnsonsu, norm

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    a = float(case["a"]); b = float(case["b"]); x = float(case["x"])
    try:
        points.append({
            "case_id": cid,
            "pdf": float(johnsonsu.pdf(x, a, b)),
            "cdf": float(johnsonsu.cdf(x, a, b)),
            "sf":  float(johnsonsu.sf(x, a, b)),
        })
    except Exception:
        points.append({"case_id": cid, "pdf": None, "cdf": None, "sf": None})
ppf = []
for case in q["ppf"]:
    cid = case["case_id"]; a = float(case["a"]); b = float(case["b"]); qv = float(case["q"])
    try:
        ppf.append({"case_id": cid, "ppf": float(johnsonsu.ppf(qv, a, b))})
    except Exception:
        ppf.append({"case_id": cid, "ppf": None})
fits = []
for seed, (a, b) in enumerate([(0.0, 1.0), (-1.5, 0.6), (2.0, 3.0)], 843):
    data = johnsonsu.rvs(a, b, size=1024, random_state=np.random.default_rng(seed))
    # Under the fixed location/scale model, asinh(X) is normal.
    # Use SciPy's independent normal MLE instead of its generic optimizer's
    # looser parameter stopping tolerance; check that optimizer's likelihood too.
    mean, scale = norm.fit(np.arcsinh(data))
    fitted_a, fitted_b = -mean / scale, 1.0 / scale
    generic = johnsonsu.fit(data, floc=0, fscale=1)
    fits.append({"case_id": f"fit_a{a}_b{b}", "data": data.tolist(),
                 "a": float(fitted_a), "b": float(fitted_b),
                 "log_likelihood": float(johnsonsu.logpdf(data, fitted_a, fitted_b).sum()),
                 "scipy_fit_log_likelihood": float(johnsonsu.logpdf(data, *generic).sum())})
print(json.dumps({"points": points, "ppf": ppf, "fits": fits}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize johnsonsu query");
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
                "failed to spawn python3 for johnsonsu oracle: {e}"
            );
            eprintln!("skipping johnsonsu oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open johnsonsu oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "johnsonsu oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping johnsonsu oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for johnsonsu oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "johnsonsu oracle failed: {stderr}"
        );
        eprintln!("skipping johnsonsu oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse johnsonsu oracle JSON"))
}

#[test]
fn diff_stats_johnsonsu_fit_maximum_likelihood() {
    let Some(oracle) = scipy_oracle_or_skip(&generate_query()) else {
        return;
    };
    assert_eq!(oracle.fits.len(), 3);
    let mut ledger = CompareLedger::new("diff_stats_johnsonsu_fit", &["a", "b", "likelihood"]);
    for case in oracle.fits {
        assert_eq!(case.data.len(), 1024);
        let fitted = JohnsonSU::try_fit(&case.data).expect("finite transformed-normal MLE");
        let log_likelihood: f64 = case.data.iter().map(|&x| fitted.logpdf(x)).sum();
        for (arm, rust, reference) in [
            ("a", fitted.a, case.a),
            ("b", fitted.b, case.b),
            ("likelihood", log_likelihood, case.log_likelihood),
        ] {
            let (reference, rust) = ledger
                .pair(arm, &case.case_id, Some(reference), Some(rust))
                .expect("finite fit parameters and likelihood");
            let pass = (rust - reference).abs() <= 1e-10 * reference.abs().max(1.0);
            ledger.compared(arm, &case.case_id, pass);
            assert!(
                pass,
                "{} {arm}: Rust={rust}, SciPy={reference}",
                case.case_id
            );
        }
        assert!(
            log_likelihood >= case.scipy_fit_log_likelihood - 1e-10 * log_likelihood.abs().max(1.0)
        );
        let perturbed = JohnsonSU::new(fitted.a + 0.25, fitted.b * 1.25);
        let perturbed_ll: f64 = case.data.iter().map(|&x| perturbed.logpdf(x)).sum();
        assert!(log_likelihood > perturbed_ll + 1.0);
    }
    ledger.finish(3);
}

#[test]
fn diff_stats_johnsonsu() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());
    assert_eq!(oracle.ppf.len(), query.ppf.len());

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

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;
    let mut ledger = CompareLedger::new("diff_stats_johnsonsu", &["pdf", "cdf", "sf", "ppf"]);

    for case in &query.points {
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        let dist = JohnsonSU::new(case.a, case.b);
        let arms = [
            ("pdf", oracle.pdf, dist.pdf(case.x), PDF_TOL),
            ("cdf", oracle.cdf, dist.cdf(case.x), CDF_TOL),
            ("sf", oracle.sf, dist.sf(case.x), CDF_TOL),
        ];
        for (family, scipy, fsci, tol) in arms {
            let Some((s, f)) = ledger.pair(family, &case.case_id, scipy, Some(fsci)) else {
                continue;
            };
            let d = (f - s).abs();
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
        let dist = JohnsonSU::new(case.a, case.b);
        let Some((sppf, rust)) =
            ledger.pair("ppf", &case.case_id, oracle.ppf, Some(dist.ppf(case.q)))
        else {
            continue;
        };
        let d = (rust - sppf).abs();
        let scale = sppf.abs().max(1.0);
        max_overall = max_overall.max(d);
        ledger.compared("ppf", &case.case_id, d <= PPF_TOL_REL * scale);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            family: "ppf".into(),
            abs_diff: d,
            pass: d <= PPF_TOL_REL * scale,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_johnsonsu".into(),
        category: "scipy.stats.johnsonsu".into(),
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
            eprintln!(
                "johnsonsu {} mismatch: {} abs_diff={}",
                d.family, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.stats.johnsonsu conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    // pdf/cdf/sf compare every point case; ppf has its own (smaller) q-grid.
    ledger.finish(query.points.len().min(query.ppf.len()));
}
