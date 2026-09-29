#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `fsci_stats::ks_2samp_alternative` against
//! `scipy.stats.ks_2samp(x, y, alternative=...)` at SciPy's DEFAULT `method='auto'`, for all
//! three alternatives. diff_stats_ks_2samp.rs covers the two-sided `ks_2samp` entry point.
//!
//! Resolves [frankenscipy-pzp32]; oracle un-pinned by [frankenscipy-qwa3t]. The oracle used to
//! call `method='asymp'`, a non-default method chosen because it matched fsci's one-sided
//! implementation, which was always Hodges' asymptotic formula. SciPy's default is the EXACT
//! null distribution whenever max(n1, n2) <= 10000, for every alternative: on the n = 10
//! shifted fixture that is 0.6818 where Hodges gives 0.5488.
//!
//! Fixtures exercise every branch of SciPy's 'auto', and most of them are ones where the exact
//! and asymptotic answers differ materially:
//!   - equal sizes (the closed-form product), tied data, small and coprime unequal sizes;
//!   - 300 vs 450 and 10000 vs 50: the lattice path count, whose binomials go through
//!     `special.binom`'s log-gamma route and carry its rounding into the p-value;
//!   - 600 vs 700: SciPy's exact attempt overflows, so it uses Hodges at the SNAPPED statistic;
//!   - 12000 vs 300 and 20000 vs 15000: max > 10000, so 'asymp' (Hodges one-sided,
//!     `kstwo.sf(d, round(en))` two-sided).
//!
//! MUST-HIT control: the oracle also returns SciPy's `method='asymp'` p-value, and at least
//! `MIN_EXACT_SEPARATED` one-sided cases must have exact and asymptotic answers more than 1e-3
//! apart (relative). Without such cases an implementation that is asymptotic everywhere passes.
//!
//! Tolerances, from the measured agreement (scipy 1.17.1, numpy 2.4.3, glibc 2.43): statistic
//! exactly equal on all 51 cases (both sides compute `i/n1 − j/n2` and snap it to `h / lcm`);
//! p-value bit-identical on 47 of 51, worst 1.9e-15 relative. The residual is
//! `fsci_special::binom` on its direct Γ route, where `gamma_core` evaluates Γ(x > 33) by a
//! log-form Lanczos sum that is up to 2.6e-13 off the exact binomial (SciPy's Cephes
//! Stirling form: 6.7e-16); over a wider sweep of 283 one-sided exact (n1, n2, h) cases that
//! reached 1.2e-13 relative on the p-value. Hence 1e-13 relative: 50x over what these fixtures
//! measure. The previous 1e-7 absolute would have accepted the 1.1e-12 relative error a
//! regrouped log-beta put into the one-sided p-value, and any error at all in the 1e-15 tail.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_stats::ks_2samp_alternative;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
// The statistic is compared for equality (`==`, so SciPy's `-0.0` from `np.clip(-0.0, 0, 1)`
// matches fsci's `0.0`); it has no tolerance const because the tolerance lint rejects a zero one.
const PVALUE_REL_TOL: f64 = 1.0e-13;
/// One-sided cases whose exact and asymptotic SciPy p-values must differ by > 1e-3 relative:
/// every non-trivial (D > 0) one-sided case below the size cap, 13 of them, other than the
/// 600 vs 700 fallback (where SciPy's answer IS the asymptotic one).
const MIN_EXACT_SEPARATED: usize = 13;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    alternative: String,
    data1: Vec<f64>,
    data2: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    statistic: Option<f64>,
    pvalue: Option<f64>,
    /// SciPy's `method='asymp'` p-value for the same data: the must-hit control's reference.
    pvalue_asymp: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    arm: String,
    abs_diff: f64,
    rel_diff: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    max_abs_diff: f64,
    max_pvalue_rel_diff: f64,
    exact_vs_asymp_separated: usize,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create ks_2samp_alt diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize ks_2samp_alt diff log");
    fs::write(path, json).expect("write ks_2samp_alt diff log");
}

/// `i/n + shift` for i in 0..n: exactly representable arithmetic, so both sides see the same
/// samples, and the empirical CDFs of two such grids cross wherever the shift puts them.
fn grid(n: usize, shift: f64) -> Vec<f64> {
    (0..n).map(|i| i as f64 / n as f64 + shift).collect()
}

fn generate_query() -> OracleQuery {
    let ints = |lo: f64, n: u32| -> Vec<f64> { (0..n).map(|i| lo + f64::from(i)).collect() };
    let fixtures: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        ("identical", ints(1.0, 10), ints(1.0, 10)),
        ("x_left_of_y", ints(1.0, 10), ints(3.0, 10)),
        ("x_right_of_y", ints(3.0, 10), ints(1.0, 10)),
        (
            "equal_30",
            (0..30).map(|i| f64::from(i) / 10.0).collect(),
            (0..30).map(|i| 0.3 + f64::from(i) / 10.0).collect(),
        ),
        (
            "ties",
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
            vec![2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0],
        ),
        ("small_3_4", vec![0.0, 1.0, 2.0], vec![0.0, 2.0, 4.0, 6.0]),
        ("coprime_7_11", grid(7, 0.0), grid(11, 0.2)),
        ("n300_450_shift002", grid(300, 0.0), grid(450, 0.02)),
        ("n300_450_shift01", grid(300, 0.0), grid(450, 0.1)),
        ("n300_450_shift03", grid(300, 0.0), grid(450, 0.3)),
        // Swapped samples: the non-trivial directed statistic is then D-, so 'less' is tested
        // at these sizes too (on the fixtures above it is 0 and its p-value is 1).
        ("n450_300_shift01_swapped", grid(450, 0.1), grid(300, 0.0)),
        ("n10000_50_shift005", grid(10_000, 0.0), grid(50, 0.05)),
        ("n10000_50_shift03", grid(10_000, 0.0), grid(50, 0.3)),
        (
            "n50_10000_shift03_swapped",
            grid(50, 0.3),
            grid(10_000, 0.0),
        ),
        ("n600_700_fallback", grid(600, 0.0), grid(700, 0.04)),
        ("n12000_300_asymp", grid(12_000, 0.0), grid(300, 0.03)),
        ("n20000_15000_asymp", grid(20_000, 0.0), grid(15_000, 0.012)),
    ];
    let alternatives = ["less", "greater", "two-sided"];

    let mut points = Vec::new();
    for (name, d1, d2) in &fixtures {
        for alt in alternatives {
            points.push(PointCase {
                case_id: format!("{name}_{alt}"),
                alternative: alt.into(),
                data1: d1.clone(),
                data2: d2.clone(),
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
import warnings
import numpy as np
from scipy import stats

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; alt = case["alternative"]
    d1 = np.array(case["data1"], dtype=float)
    d2 = np.array(case["data2"], dtype=float)
    try:
        with warnings.catch_warnings():
            # 'auto' warns when its exact attempt fails and it switches to 'asymp'.
            warnings.simplefilter("ignore", RuntimeWarning)
            res = stats.ks_2samp(d1, d2, alternative=alt)
            asymp = stats.ks_2samp(d1, d2, alternative=alt, method='asymp')
        points.append({
            "case_id": cid,
            "statistic": fnone(res.statistic),
            "pvalue": fnone(res.pvalue),
            "pvalue_asymp": fnone(asymp.pvalue),
        })
    except Exception:
        points.append({"case_id": cid, "statistic": None, "pvalue": None, "pvalue_asymp": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize ks_2samp_alt query");
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
                "failed to spawn python3 for ks_2samp_alt oracle: {e}"
            );
            eprintln!("skipping ks_2samp_alt oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open ks_2samp_alt oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "ks_2samp_alt oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping ks_2samp_alt oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for ks_2samp_alt oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "ks_2samp_alt oracle failed: {stderr}"
        );
        eprintln!("skipping ks_2samp_alt oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse ks_2samp_alt oracle JSON"))
}

/// |f − s| / |s|, with a zero SciPy value matched only exactly.
fn relative_diff(f: f64, s: f64) -> f64 {
    if f == s {
        0.0
    } else if s == 0.0 {
        f64::INFINITY
    } else {
        (f - s).abs() / s.abs()
    }
}

#[test]
fn diff_stats_ks_2samp_alt() {
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
    let mut max_abs = 0.0_f64;
    let mut max_pvalue_rel = 0.0_f64;
    let mut separated = 0_usize;
    let mut ledger = CompareLedger::new("diff_stats_ks_2samp_alt", &["statistic", "pvalue"]);

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let result = ks_2samp_alternative(&case.data1, &case.data2, &case.alternative);

        if case.alternative != "two-sided"
            && let (Some(exact), Some(asymp)) = (scipy_arm.pvalue, scipy_arm.pvalue_asymp)
            && relative_diff(asymp, exact) > 1e-3
        {
            separated += 1;
        }

        if let Some((s, f)) = ledger.pair(
            "statistic",
            &case.case_id,
            scipy_arm.statistic,
            Some(result.statistic),
        ) {
            let abs_diff = (f - s).abs();
            let pass = f == s;
            max_abs = max_abs.max(abs_diff);
            ledger.compared("statistic", &case.case_id, pass);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "statistic".into(),
                abs_diff,
                rel_diff: relative_diff(f, s),
                pass,
            });
        }
        if let Some((s, f)) = ledger.pair(
            "pvalue",
            &case.case_id,
            scipy_arm.pvalue,
            Some(result.pvalue),
        ) {
            let abs_diff = (f - s).abs();
            let rel_diff = relative_diff(f, s);
            let pass = rel_diff <= PVALUE_REL_TOL;
            max_abs = max_abs.max(abs_diff);
            max_pvalue_rel = max_pvalue_rel.max(rel_diff);
            ledger.compared("pvalue", &case.case_id, pass);
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                arm: "pvalue".into(),
                abs_diff,
                rel_diff,
                pass,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_stats_ks_2samp_alt".into(),
        category: "scipy.stats.ks_2samp(alternative=less/greater/two-sided, method='auto')".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        max_abs_diff: max_abs,
        max_pvalue_rel_diff: max_pvalue_rel,
        exact_vs_asymp_separated: separated,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };

    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "ks_2samp_alt mismatch: {} arm={} abs={} rel={}",
                d.case_id, d.arm, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "ks_2samp_alt conformance failed: {} cases, max_abs={}, max pvalue rel={}",
        diffs.len(),
        max_abs,
        max_pvalue_rel
    );
    assert!(
        separated >= MIN_EXACT_SEPARATED,
        "only {separated} one-sided fixtures separate SciPy's exact and asymptotic p-values \
         (need {MIN_EXACT_SEPARATED}); the fixtures no longer distinguish an exact \
         implementation from an asymptotic one"
    );
    ledger.finish(query.points.len());
}
