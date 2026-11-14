#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.stats.invgamma`.
//!
//! Resolves [frankenscipy-xre7v]. InverseGamma has anchor tests
//! in `fsci-stats/src/lib.rs` but no dedicated scipy diff
//! harness. 6 a-values × 7 x-values × 3 families + 6 × 5 ppf
//! cases via subprocess.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_stats::{ContinuousDistribution, InverseGamma};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
// InverseGamma cdf composes regularised upper incomplete gamma
// — same series-precision floor as Pearson3/Erlang siblings.
const PDF_TOL: f64 = 1.0e-12;
const CDF_TOL: f64 = 1.0e-11;
const PPF_TOL_REL: f64 = 1.0e-7;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    a: f64,
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct PpfCase {
    case_id: String,
    a: f64,
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
}

#[derive(Debug, Deserialize)]
struct FitArm {
    case_id: String,
    data: Vec<f64>,
    shape: f64,
    log_likelihood: f64,
    moment_log_likelihood: f64,
}

#[test]
fn diff_stats_invgamma_shape_mle() {
    let script = r#"
import json
import numpy as np
from scipy.stats import invgamma

rows = []
for i, shape in enumerate((0.3, 0.8, 4.0)):
    data = invgamma.rvs(shape, size=2000, random_state=np.random.default_rng(14311+i))
    fitted, _, _ = invgamma.fit(data, floc=0, fscale=1)
    moment = 1 + 1/data.mean()
    rows.append(dict(case_id=f"shape_{shape}", data=data.tolist(), shape=float(fitted),
                     log_likelihood=float(invgamma.logpdf(data, fitted).sum()),
                     moment_log_likelihood=float(invgamma.logpdf(data, moment).sum())))
print(json.dumps(rows, allow_nan=False))
"#;
    let output = match fsci_conformance::scipy_oracle_command()
        .args(["-c", script])
        .output()
    {
        Ok(output) if output.status.success() => output,
        output => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "required inverse-gamma MLE oracle failed: {output:?}"
            );
            eprintln!("skipping inverse-gamma MLE oracle: {output:?}");
            return;
        }
    };
    let rows: Vec<FitArm> = serde_json::from_slice(&output.stdout).expect("finite MLE oracle rows");
    assert_eq!(rows.len(), 3);
    let mut ledger = CompareLedger::new("diff_stats_invgamma_shape_mle", &["shape", "likelihood"]);
    let mut moment_failures = 0;
    for row in rows {
        assert_eq!(row.data.len(), 2000);
        let fitted = InverseGamma::try_fit(&row.data).expect("every valid oracle sample must fit");
        let log_likelihood = row.data.iter().map(|&x| fitted.logpdf(x)).sum::<f64>();
        for (arm, rust, reference) in [
            ("shape", fitted.a, row.shape),
            ("likelihood", log_likelihood, row.log_likelihood),
        ] {
            let (reference, rust) = ledger
                .pair(arm, &row.case_id, Some(reference), Some(rust))
                .expect("finite MLE parameters and likelihoods");
            let pass = if arm == "likelihood" {
                rust >= reference - 1e-6 * reference.abs().max(1.0)
            } else {
                (rust - reference).abs() <= 1e-4 * reference.abs()
            };
            ledger.compared(arm, &row.case_id, pass);
            assert!(
                pass,
                "{} {arm}: Rust={rust}, SciPy={reference}",
                row.case_id
            );
        }
        if row.moment_log_likelihood < row.log_likelihood - 1e-6 * row.log_likelihood.abs().max(1.0)
        {
            moment_failures += 1;
        }
        eprintln!(
            "{}: MLE={}, SciPy={}, LL={log_likelihood}, SciPy LL={}, old moment LL={}",
            row.case_id, fitted.a, row.shape, row.log_likelihood, row.moment_log_likelihood
        );
    }
    assert_eq!(
        moment_failures, 3,
        "the old moment fit must fail every new MLE row"
    );
    ledger.finish(3);
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
    fs::create_dir_all(output_dir()).expect("create invgamma diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize invgamma diff log");
    fs::write(path, json).expect("write invgamma diff log");
}

fn generate_query() -> OracleQuery {
    let aas = [0.5_f64, 1.0, 2.0, 3.0, 5.0, 10.0];
    let xs = [0.1_f64, 0.3, 0.5, 1.0, 2.0, 5.0, 20.0];
    let qs = [0.05_f64, 0.25, 0.5, 0.75, 0.95];
    let mut points = Vec::new();
    for &a in &aas {
        for &x in &xs {
            points.push(PointCase {
                case_id: format!("a{a}_x{x}"),
                a,
                x,
            });
        }
    }
    let mut ppf_cases = Vec::new();
    for &a in &aas {
        for &q in &qs {
            ppf_cases.push(PpfCase {
                case_id: format!("a{a}_q{q}"),
                a,
                q,
            });
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
from scipy.stats import invgamma

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    a = float(case["a"]); x = float(case["x"])
    try:
        points.append({
            "case_id": cid,
            "pdf": float(invgamma.pdf(x, a)),
            "cdf": float(invgamma.cdf(x, a)),
            "sf":  float(invgamma.sf(x, a)),
        })
    except Exception:
        points.append({"case_id": cid, "pdf": None, "cdf": None, "sf": None})
ppf = []
for case in q["ppf"]:
    cid = case["case_id"]; a = float(case["a"]); qv = float(case["q"])
    try:
        ppf.append({"case_id": cid, "ppf": float(invgamma.ppf(qv, a))})
    except Exception:
        ppf.append({"case_id": cid, "ppf": None})
print(json.dumps({"points": points, "ppf": ppf}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize invgamma query");
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
                "failed to spawn python3 for invgamma oracle: {e}"
            );
            eprintln!("skipping invgamma oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open invgamma oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "invgamma oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping invgamma oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for invgamma oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "invgamma oracle failed: {stderr}"
        );
        eprintln!("skipping invgamma oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse invgamma oracle JSON"))
}

#[test]
fn diff_stats_invgamma() {
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
    let mut ledger = CompareLedger::new("diff_stats_invgamma", &["pdf", "cdf", "sf", "ppf"]);

    for case in &query.points {
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        let dist = InverseGamma::new(case.a);
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
        let dist = InverseGamma::new(case.a);
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
        test_id: "diff_stats_invgamma".into(),
        category: "scipy.stats.invgamma".into(),
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
                "invgamma {} mismatch: {} abs_diff={}",
                d.family, d.case_id, d.abs_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.stats.invgamma conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    // pdf/cdf/sf compare every point case; ppf has its own (smaller) q-grid.
    ledger.finish(query.points.len().min(query.ppf.len()));
}
