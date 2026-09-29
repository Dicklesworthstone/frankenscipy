#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the Kolmogorov-Smirnov
//! special functions
//! `scipy.special.kolmogorov` and `scipy.special.smirnov`.
//!
//! Resolves [frankenscipy-7ps6s].
//!   • kolmogorov(y) is the asymptotic two-sided KS sf at y.
//!   • smirnov(n, d) is the one-sided KS sf for sample size n
//!     at deviation d.
//!
//! 17 y for kolmogorov; for smirnov, n from 1 to 2·10^6 (every branch of xsf `_smirnov`:
//! d ≤ 1/n, the lower and upper Birnbaum–Tingey sums, d ≥ 1 − 1/n, the underflow cut-off
//! and the n > 10^6 approximation) against a fixed d grid, points on the 1/n and 1/√n
//! scales, and 1 − 1/2n. Tolerances: 1e-15 relative for both; kolmogorov is xsf's
//! `_kolmogorov` (frankenscipy-11wqg) and smirnov ports xsf's double-double sum operation
//! for operation (frankenscipy-k4c2c). The C `int` products SciPy overflows from n = 46342
//! on, and which fsci keeps exact, feed only the pdf (and through it smirnovi), never the
//! sf, so n = 10^5 is compared here like every other n.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_runtime::RuntimeMode;
use fsci_special::types::SpecialTensor;
use fsci_special::{kolmogorov, smirnov};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
// kolmogorov is xsf's `_kolmogorov`, bit-identical to SciPy. This is a TRUE relative tolerance,
// a few ulp. It was an absolute 1e-9 over y >= 0.1, which never reached the small-y region where
// the old truncated series was 0.1376 off (frankenscipy-11wqg).
const KOLMOGOROV_TOL_REL: f64 = 1.0e-15;
// Relative to SciPy's value: fsci's smirnov is xsf's algorithm evaluated in the same order,
// so it agrees to the last bit; an exact 0 must be matched exactly.
const SMIRNOV_TOL_REL: f64 = 1.0e-15;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
/// One ledger arm per SciPy function compared.
const ARMS: [&str; 2] = ["kolmogorov", "smirnov"];

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    n: i32,
    arg: f64,
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
    func: String,
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
    max_smirnov_rel_diff: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create ks diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize ks diff log");
    fs::write(path, json).expect("write ks diff log");
}

fn fsci_eval(func: &str, n: i32, arg: f64) -> Option<f64> {
    match func {
        "kolmogorov" => {
            let arg_t = SpecialTensor::RealScalar(arg);
            match kolmogorov(&arg_t, RuntimeMode::Strict) {
                Ok(SpecialTensor::RealScalar(v)) => Some(v),
                _ => None,
            }
        }
        // A non-finite value is returned as is: the ledger classifies it against SciPy's.
        "smirnov" => Some(smirnov(n, arg)),
        _ => None,
    }
}

fn generate_query() -> OracleQuery {
    // Small y first: below ~0.035 a truncated alternating series has not converged, and
    // SciPy's theta-series branch runs up to 0.82.
    let ys = [
        0.01_f64, 0.02, 0.035, 0.05, 0.1, 0.3, 0.5, 0.75, 0.82, 1.0, 1.36, 1.5, 1.95, 2.0, 2.5,
        3.0, 5.0,
    ];
    let ns = [
        1_i32, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 10_000, 100_000, 2_000_000,
    ];
    let grid = [0.02_f64, 0.05, 0.08, 0.12, 0.18, 0.25, 0.4, 0.6, 0.9];

    let mut points = Vec::new();
    for &y in &ys {
        points.push(PointCase {
            case_id: format!("kolmogorov_y{y}"),
            func: "kolmogorov".into(),
            n: 0,
            arg: y,
        });
    }
    for &n in &ns {
        let nf = f64::from(n);
        // Below, at and just above 1/n (d ≤ 1/n closed form, then the upper sum), across the
        // 1/√n scale where the probability moves, and within 1/n of 1.
        let mut ds: Vec<f64> = grid.to_vec();
        ds.extend([0.5 / nf, 1.0 / nf, 2.5 / nf, 1.0 - 0.5 / nf]);
        ds.extend([0.3, 0.8, 1.5].map(|c| c / nf.sqrt()));
        ds.retain(|d| (0.0..=1.0).contains(d));
        ds.sort_by(f64::total_cmp);
        ds.dedup_by(|a, b| a.to_bits() == b.to_bits());
        for d in ds {
            points.push(PointCase {
                case_id: format!("smirnov_n{n}_d{d}"),
                func: "smirnov".into(),
                n,
                arg: d,
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
    cid = case["case_id"]; func = case["func"]
    n = int(case["n"]); arg = float(case["arg"])
    try:
        if func == "kolmogorov": value = special.kolmogorov(arg)
        elif func == "smirnov":  value = special.smirnov(n, arg)
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize ks query");
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
                "failed to spawn python3 for ks oracle: {e}"
            );
            eprintln!("skipping ks oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open ks oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "ks oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping ks oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for ks oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "ks oracle failed: {stderr}"
        );
        eprintln!("skipping ks oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse ks oracle JSON"))
}

#[test]
fn diff_special_ks() {
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
    let mut max_smirnov_rel = 0.0_f64;
    let mut ledger = CompareLedger::new("diff_special_ks", &ARMS);

    for case in &query.points {
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        let arm = case.func.as_str();
        let Some((scipy_v, rust_v)) = ledger.pair(
            arm,
            &case.case_id,
            oracle.value,
            fsci_eval(&case.func, case.n, case.arg),
        ) else {
            continue;
        };
        let abs_diff = (rust_v - scipy_v).abs();
        max_overall = max_overall.max(abs_diff);
        let pass = match case.func.as_str() {
            "kolmogorov" => abs_diff <= KOLMOGOROV_TOL_REL * scipy_v.abs(),
            "smirnov" => {
                if scipy_v != 0.0 {
                    max_smirnov_rel = max_smirnov_rel.max(abs_diff / scipy_v.abs());
                }
                abs_diff <= SMIRNOV_TOL_REL * scipy_v.abs()
            }
            _ => false,
        };
        ledger.compared(arm, &case.case_id, pass);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            func: case.func.clone(),
            abs_diff,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_ks".into(),
        category: "scipy.special.kolmogorov/smirnov".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        max_abs_diff: max_overall,
        max_smirnov_rel_diff: max_smirnov_rel,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };

    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("ks {} mismatch: {} abs={}", d.func, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "scipy.special ks conformance failed: {} cases, max_abs={}",
        diffs.len(),
        max_overall
    );
    // Arms have different case sets (kolmogorov has the fewest); each must compare all of its
    // own.
    let min_per_arm = ARMS
        .iter()
        .map(|arm| query.points.iter().filter(|c| c.func == *arm).count())
        .min()
        .expect("ARMS is non-empty");
    ledger.finish(min_per_arm);
}
