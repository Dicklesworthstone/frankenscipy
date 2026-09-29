#![forbid(unsafe_code)]
//! Live SciPy differential coverage for Owen's T function
//! `scipy.special.owens_t`.
//!
//! Resolves [frankenscipy-siziw]. Owen's T is used by SkewNormal
//! cdf and other tail-correction integrals. fsci's
//! `owens_t_scalar` had no dedicated diff harness.
//!
//! fsci's owens_t is xsf's Patefield-Tandy `owens_t.h`, the kernel
//! SciPy 1.17.1 calls, so every case must be SciPy's value to the bit
//! (frankenscipy-nb55y). The former 5e-6 absolute gate could not see
//! the Gauss-Legendre rule's 2.5e-9 relative error at
//! (h, a) = (-4.872, -0.922), or anything at all in values of 1e-11.
//!
//! Cases: the original 9 × 7 grid; an 8 × 8 grid over the region the
//! Gauss-Legendre rule got worst (|h| 3.7-5, |a| 0.8-1); the midpoint
//! of every cell of the method table (15 h × 8 a cells, T1..T6), with
//! alternating signs; a > 1 through both branches of the reflection;
//! and 400 points uniform on [-5, 5]², the sweep the bead measured.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_runtime::RuntimeMode;
use fsci_special::owens_t;
use fsci_special::types::SpecialTensor;
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    h: f64,
    a: f64,
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
    max_rel_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create owens_t diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize owens_t diff log");
    fs::write(path, json).expect("write owens_t diff log");
}

fn fsci_eval(h: f64, a: f64) -> Option<f64> {
    let ph = SpecialTensor::RealScalar(h);
    let pa = SpecialTensor::RealScalar(a);
    match owens_t(&ph, &pa, RuntimeMode::Strict) {
        Ok(SpecialTensor::RealScalar(v)) => Some(v),
        _ => None,
    }
}

/// Upper bounds of the h and a cells of xsf's method table (`owens_t_HRANGE`,
/// `owens_t_ARANGE`), with 0 below and a last h cell closed at 8.
const H_CELL_BOUNDS: [f64; 16] = [
    0.0, 0.02, 0.06, 0.09, 0.125, 0.26, 0.4, 0.6, 1.6, 1.7, 2.33, 2.4, 3.36, 3.4, 4.8, 8.0,
];
const A_CELL_BOUNDS: [f64; 9] = [0.0, 0.025, 0.09, 0.15, 0.36, 0.5, 0.9, 0.99999, 1.0];

/// Uniform on [0, 1) from a fixed 64-bit LCG, so the sweep is the same every run.
fn next_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 11) as f64 / (1_u64 << 53) as f64
}

fn generate_query() -> OracleQuery {
    let mut pairs: Vec<(f64, f64)> = Vec::new();
    // T(h, a) is even in h and odd in a; both signs are sampled anyway to verify the
    // implementation.
    let hs = [-3.0_f64, -1.0, -0.3, 0.0, 0.3, 1.0, 3.0, 5.0, 10.0];
    let as_ = [-2.0_f64, -0.5, -0.1, 0.5, 1.0, 2.0, 5.0];
    for &h in &hs {
        for &a in &as_ {
            pairs.push((h, a));
        }
    }
    // frankenscipy-nb55y: where the Gauss-Legendre rule was worst, with the bead's
    // (-4.872, -0.922) and a above the table's last a bound (T6, T5 and T3 cells).
    for &h in &[-4.872_f64, -4.6, -4.3, -3.9, 3.7, 4.1, 4.5, 4.95] {
        for &a in &[
            -0.999_995_f64,
            -0.97,
            -0.922,
            -0.86,
            0.81,
            0.9,
            0.95,
            0.999_99,
        ] {
            pairs.push((h, a));
        }
    }
    // Every cell of the method table at its midpoint, so each of T1..T6 is reached.
    let mut k = 0_usize;
    for hc in H_CELL_BOUNDS.windows(2) {
        for ac in A_CELL_BOUNDS.windows(2) {
            let h = f64::midpoint(hc[0], hc[1]);
            let a = f64::midpoint(ac[0], ac[1]);
            let h = if k.is_multiple_of(2) { h } else { -h };
            let a = if k % 4 < 2 { a } else { -a };
            pairs.push((h, a));
            k += 1;
        }
    }
    // a > 1 maps to 1/a through Owen's reflection: ah <= 0.67 takes Phi, above it the
    // complementary Phi(-x).
    for &a in &[1.000_004_f64, 1.2, 1.9, 3.5, 12.0, 60.0] {
        for &h in &[0.005_f64, 0.05, 0.3, 0.9, 2.0, 5.0] {
            pairs.push((h, a));
        }
    }
    // The sweep the bead measured: h and a uniform on [-5, 5].
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    for _ in 0..400 {
        let h = next_unit(&mut state).mul_add(10.0, -5.0);
        let a = next_unit(&mut state).mul_add(10.0, -5.0);
        pairs.push((h, a));
    }
    let points = pairs
        .into_iter()
        .enumerate()
        .map(|(i, (h, a))| PointCase {
            case_id: format!("{i:04}_h{h}_a{a}"),
            h,
            a,
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

def finite_or_none(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    h = float(case["h"]); a = float(case["a"])
    try:
        value = special.owens_t(h, a)
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize owens_t query");
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
                "failed to spawn python3 for owens_t oracle: {e}"
            );
            eprintln!("skipping owens_t oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open owens_t oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "owens_t oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping owens_t oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for owens_t oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "owens_t oracle failed: {stderr}"
        );
        eprintln!("skipping owens_t oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse owens_t oracle JSON"))
}

#[test]
fn diff_special_owens_t() {
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
    let mut max_rel = 0.0_f64;
    let mut ledger = CompareLedger::new("diff_special_owens_t", &["owens_t"]);

    for case in &query.points {
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        let Some((scipy_v, rust_v)) = ledger.pair(
            "owens_t",
            &case.case_id,
            oracle.value,
            fsci_eval(case.h, case.a),
        ) else {
            continue;
        };
        let abs_diff = (rust_v - scipy_v).abs();
        let rel_diff = if scipy_v == 0.0 {
            abs_diff
        } else {
            abs_diff / scipy_v.abs()
        };
        max_overall = max_overall.max(abs_diff);
        max_rel = max_rel.max(rel_diff);
        // Exact: SciPy's bits, so a zero's sign counts too.
        let pass = rust_v.to_bits() == scipy_v.to_bits();
        ledger.compared("owens_t", &case.case_id, pass);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            abs_diff,
            rel_diff,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_owens_t".into(),
        category: "scipy.special.owens_t".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        max_abs_diff: max_overall,
        max_rel_diff: max_rel,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };

    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "owens_t mismatch: {} abs={} rel={}",
                d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }
    let failed = diffs.iter().filter(|d| !d.pass).count();
    eprintln!(
        "owens_t: {} cases, {failed} not SciPy's bits, max_abs={max_overall:e} max_rel={max_rel:e}",
        diffs.len()
    );

    assert!(
        all_pass,
        "scipy.special.owens_t conformance failed: {failed} of {} cases not SciPy's bits, \
         max_abs={max_overall:e}, max_rel={max_rel:e}",
        diffs.len()
    );
    ledger.finish(query.points.len());
}
