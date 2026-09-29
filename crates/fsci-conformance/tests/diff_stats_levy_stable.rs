#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `scipy.stats.levy_stable` (frankenscipy-1ksfv.16).
//!
//! SciPy 1.17.1 computes the stable law's pdf and cdf by Nolan's piecewise integration
//! (`pdf_default_method = cdf_default_method = 'piecewise'`); fsci's `LevyStable` ports that
//! code path. Grid: alpha in {0.5, 0.8, 1, 1.3, 1.5, 1.9, 2} x beta in {-1, -0.3, 0, 0.3, 1},
//! at the standard law and one (loc, scale) pair, in both parameterizations. S1 points are
//! absolute (z = 0 is the S0 point x0 = zeta, z = 0.002 is rounded to it); S0 points are offsets
//! from zeta = -beta·tan(pi·alpha/2) (from 0 at alpha = 1). Both tails are included, down to where
//! the one-sided laws are exactly 0, which must then match exactly.
//!
//! The fsci-stats unit tests (`levy_stable_matches_scipy`) reproduce 490 pinned SciPy pdf/cdf
//! values bit for bit, hence at most one ulp here (`f64::EPSILON` relative; the tolerance lint
//! takes only positive literals): a larger difference is a divergence to explain, not noise.
//! Non-finite x, ppf, stats, rvs and fitstart are pinned there.

use std::collections::{BTreeMap, HashMap};
use std::f64::consts::PI;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_stats::{ContinuousDistribution, LevyStable, LevyStableParameterization};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
// The port reproduces SciPy bit for bit at the fsci-stats unit-test points, and 1958 of these
// 1960 cases agree to 1e-12. The other two are x = 0.002 at α = 1/2, β = 0 (pdf, S1 and S0):
// within piecewise_x_tol_near_zeta = 0.005 of ζ = 0, where both sides round x by design and
// SciPy is itself 5.6e-5 off the true density (mpmath, 40 digits), the two quadratures agree to
// 2.2e-12, not to the bit. 1e-9 holds that, and is 100x tighter than the 1e-7 the bead sets as
// SciPy's own piecewise accuracy.
const PDF_REL_TOL: f64 = 1e-9;
const CDF_REL_TOL: f64 = 1e-9;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const ARMS: [&str; 4] = ["pdf", "cdf", "pdf_s0", "cdf_s0"];

const ALPHAS: [f64; 7] = [0.5, 0.8, 1.0, 1.3, 1.5, 1.9, 2.0];
const BETAS: [f64; 5] = [-1.0, -0.3, 0.0, 0.3, 1.0];

/// (loc, scale) pairs: the standard law and one shifted/scaled one.
const PARAMS: [(f64, f64); 2] = [(0.0, 1.0), (-1.5, 2.5)];

/// Standardized points: x in S1, x0 - zeta in S0.
const Z: [f64; 7] = [-20.0, -1.3, 0.0, 0.002, 0.45, 3.5, 40.0];

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    arm: String,
    parameterization: String,
    alpha: f64,
    beta: f64,
    loc: f64,
    scale: f64,
    x: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    arm: String,
    rel_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    max_rel_diff: f64,
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

fn emit_log(log: &DiffLog) {
    fs::create_dir_all(output_dir()).expect("create levy_stable diff output dir");
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize levy_stable diff log");
    fs::write(path, json).expect("write levy_stable diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    for &alpha in &ALPHAS {
        for &beta in &BETAS {
            // S0's zeta; S0 and S1 coincide at alpha = 1, where zeta is not finite in practice.
            let zeta = if alpha == 1.0 {
                0.0
            } else {
                -beta * (PI * alpha / 2.0).tan()
            };
            for &(loc, scale) in &PARAMS {
                for arm in ARMS {
                    let (parameterization, center) = if arm.ends_with("_s0") {
                        ("S0", zeta)
                    } else {
                        ("S1", 0.0)
                    };
                    for &z in &Z {
                        points.push(PointCase {
                            case_id: format!("{arm}_a{alpha}_b{beta}_l{loc}_s{scale}_z{z}"),
                            arm: arm.into(),
                            parameterization: parameterization.into(),
                            alpha,
                            beta,
                            loc,
                            scale,
                            x: loc + scale * (center + z),
                        });
                    }
                }
            }
        }
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import math
import sys
from scipy.stats import levy_stable

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    levy_stable.parameterization = case["parameterization"]
    fn = levy_stable.pdf if case["arm"].startswith("pdf") else levy_stable.cdf
    try:
        v = float(fn(float(case["x"]), float(case["alpha"]), float(case["beta"]),
                     loc=float(case["loc"]), scale=float(case["scale"])))
        points.append({"case_id": cid, "value": v if math.isfinite(v) else None})
    except Exception:
        points.append({"case_id": cid, "value": None})
levy_stable.parameterization = "S1"
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize levy_stable query");
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
                "failed to spawn python3 for levy_stable oracle: {e}"
            );
            eprintln!("skipping levy_stable oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open levy_stable oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "levy_stable oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping levy_stable oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for levy_stable oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "levy_stable oracle failed: {stderr}"
        );
        eprintln!("skipping levy_stable oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse levy_stable oracle JSON"))
}

/// Relative difference; a zero SciPy value (a one-sided law's empty side) must be matched
/// exactly.
fn rel_diff(fsci: f64, scipy: f64) -> f64 {
    if scipy == 0.0 {
        if fsci == 0.0 { 0.0 } else { f64::INFINITY }
    } else {
        ((fsci - scipy) / scipy).abs()
    }
}

#[test]
fn diff_stats_levy_stable() {
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
    let mut ledger = CompareLedger::new("diff_stats_levy_stable", &ARMS);

    for case in &query.points {
        let scipy = pmap.get(&case.case_id).expect("validated oracle").value;
        let parameterization = if case.parameterization == "S0" {
            LevyStableParameterization::S0
        } else {
            LevyStableParameterization::S1
        };
        let dist = LevyStable::with_parameterization(
            case.alpha,
            case.beta,
            case.loc,
            case.scale,
            parameterization,
        );
        // An arm outside ARMS is skipped here and then fails the per-arm count below.
        let Some((fsci, tol)) = (match case.arm.as_str() {
            "pdf" | "pdf_s0" => Some((dist.pdf(case.x), PDF_REL_TOL)),
            "cdf" | "cdf_s0" => Some((dist.cdf(case.x), CDF_REL_TOL)),
            _ => None,
        }) else {
            continue;
        };
        let Some((s, f)) = ledger.pair(&case.arm, &case.case_id, scipy, Some(fsci)) else {
            continue;
        };
        let d = rel_diff(f, s);
        max_overall = max_overall.max(d);
        ledger.compared(&case.arm, &case.case_id, d <= tol);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            arm: case.arm.clone(),
            rel_diff: d,
            pass: d <= tol,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_stats_levy_stable".into(),
        category: "scipy.stats.levy_stable".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        max_rel_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "levy_stable {} mismatch: {} rel_diff={}",
                d.arm, d.case_id, d.rel_diff
            );
        }
    }
    assert!(
        all_pass,
        "scipy.stats.levy_stable conformance failed: {} cases, max_rel_diff={}",
        diffs.len(),
        max_overall
    );
    let counts = ledger.finish(Z.len());
    for arm in ARMS {
        let cases = query.points.iter().filter(|c| c.arm == arm).count();
        assert_eq!(
            counts[arm].compared_cases, cases,
            "arm `{arm}` must compare all {cases} of its cases"
        );
    }
}
