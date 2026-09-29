#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the gamma, Poisson, and
//! chi-squared scipy-compat wrappers
//! (`scipy.special.gdtr/gdtrc/pdtr/pdtrc/pdtri/chdtr/chdtrc/chdtri`).
//!
//! Resolves [frankenscipy-uv9i0]. Verifies the cdf/sf/ppf
//! wrappers directly; complements diff_stats_gamma,
//! diff_stats_poisson, and diff_stats_chi2 which exercise the
//! same kernel indirectly.
//!
//! Gates are exact: each wrapper is xsf's `cephes/gdtr.h`, `pdtr.h` or `chdtr.h` over the
//! ported `igam`/`igamc`/`igamci` (frankenscipy-449uv), so every value must be SciPy's to the
//! bit. The inverses are also compared in their tails, p from 1e-300 to 1 − 2^-53, where a
//! hard-region sweep found the old kernels 6e-2 (chdtri) and 1.9e84 (pdtri) off.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_special::{chdtr, chdtrc, chdtri, gdtr, gdtrc, pdtr, pdtrc, pdtri};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
/// One ledger arm per generated function.
const ARMS: [&str; 8] = [
    "gdtr", "gdtrc", "pdtr", "pdtrc", "pdtri", "chdtr", "chdtrc", "chdtri",
];

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    p1: f64,
    p2: f64,
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
    fs::create_dir_all(output_dir()).expect("create gdtr/pdtr/chdtr diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize gdtr/pdtr/chdtr diff log");
    fs::write(path, json).expect("write gdtr/pdtr/chdtr diff log");
}

fn fsci_eval(func: &str, p1: f64, p2: f64, arg: f64) -> Option<f64> {
    let v = match func {
        "gdtr" => gdtr(p1, p2, arg),
        "gdtrc" => gdtrc(p1, p2, arg),
        "pdtr" => pdtr(p1, arg),
        "pdtrc" => pdtrc(p1, arg),
        "pdtri" => pdtri(p1, arg),
        "chdtr" => chdtr(p1, arg),
        "chdtrc" => chdtrc(p1, arg),
        "chdtri" => chdtri(p1, arg),
        _ => return None,
    };
    // A non-finite value reaches the ledger, which records it as an fsci failure.
    Some(v)
}

fn generate_query() -> OracleQuery {
    // gamma: a=rate, b=shape, x>0
    let gdtr_cases = [
        (1.0_f64, 0.5),
        (1.0, 1.0),
        (1.0, 2.0),
        (2.0, 5.0),
        (0.5, 3.0),
    ];
    let xs = [0.1_f64, 0.5, 1.0, 2.0, 5.0, 10.0];
    // Poisson: m=mean, k=count
    let mus = [0.5_f64, 1.0, 3.0, 10.0, 25.0];
    let ks = [0_u32, 1, 3, 5, 10, 20];
    let qs = [0.001_f64, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999];
    // Chi2: v=df, x>0
    let dfs = [1.0_f64, 3.0, 5.0, 10.0, 25.0];
    let xs_chdtr = [0.1_f64, 1.0, 3.0, 5.0, 10.0, 25.0];

    let mut points = Vec::new();
    for &(a, b) in &gdtr_cases {
        for &x in &xs {
            for func in ["gdtr", "gdtrc"] {
                points.push(PointCase {
                    case_id: format!("{func}_a{a}_b{b}_x{x}"),
                    func: func.to_string(),
                    p1: a,
                    p2: b,
                    arg: x,
                });
            }
        }
    }
    for &mu in &mus {
        for &k in &ks {
            let kf = k as f64;
            for func in ["pdtr", "pdtrc"] {
                points.push(PointCase {
                    case_id: format!("{func}_mu{mu}_k{k}"),
                    func: func.to_string(),
                    p1: kf,
                    p2: 0.0,
                    arg: mu,
                });
            }
        }
    }
    // pdtri(k, q): the mean with pdtr(k, m) = q. It used to be omitted: the local gammaincinv
    // returned ~1e13 where SciPy gives 0.149 at k=1, q=0.99. frankenscipy-jr3na re-pointed it
    // at the validated gammaincinv.
    for &k in &ks {
        for &q in &qs {
            let kf = f64::from(k);
            points.push(PointCase {
                case_id: format!("pdtri_k{k}_q{q}"),
                func: "pdtri".to_string(),
                p1: kf,
                p2: 0.0,
                arg: q,
            });
        }
    }
    // Non-integer counts: SciPy floors k in pdtr/pdtrc and truncates it in pdtri (a C int),
    // so -0.5 is pdtri's count 0. With the raw k + 1 these were up to 0.136 (pdtr) and 1.69
    // (pdtri) off (frankenscipy-uyhhv).
    for &kf in &[0.5_f64, 2.7, 7.3] {
        for &mu in &mus {
            for func in ["pdtr", "pdtrc"] {
                points.push(PointCase {
                    case_id: format!("{func}_mu{mu}_kfrac{kf}"),
                    func: func.to_string(),
                    p1: kf,
                    p2: 0.0,
                    arg: mu,
                });
            }
        }
    }
    for &kf in &[-0.5_f64, 0.5, 2.7, 7.3] {
        for &q in &qs {
            points.push(PointCase {
                case_id: format!("pdtri_kfrac{kf}_q{q}"),
                func: "pdtri".to_string(),
                p1: kf,
                p2: 0.0,
                arg: q,
            });
        }
    }
    for &df in &dfs {
        for &x in &xs_chdtr {
            for func in ["chdtr", "chdtrc"] {
                points.push(PointCase {
                    case_id: format!("{func}_df{df}_x{x}"),
                    func: func.to_string(),
                    p1: df,
                    p2: 0.0,
                    arg: x,
                });
            }
        }
        // chdtri(df, q): x with chdtrc(df, x) = q; it was omitted with pdtri, same cause
        // (chdtri(3, 0.99) returned ~5.6e5 where SciPy gives 0.115), same fix.
        for &q in &qs {
            points.push(PointCase {
                case_id: format!("chdtri_df{df}_q{q}"),
                func: "chdtri".to_string(),
                p1: df,
                p2: 0.0,
                arg: q,
            });
        }
    }

    // frankenscipy-449uv: the inverses in their tails, p from 1e-300 to 1 − 2^-53, where a
    // hard-region sweep found the old kernels 6e-2 (chdtri) and 1.9e84 (pdtri) off, and the
    // forward CDFs across the Temme zones and the Lanczos igam_fac above a = 200.
    let tails = [
        1e-300_f64,
        1e-200,
        1e-100,
        1e-50,
        1e-20,
        1e-10,
        1.0 - 1e-10,
        1.0 - f64::EPSILON / 2.0,
    ];
    for &df in &[0.5_f64, 3.0, 20.0, 100.0] {
        for &q in &tails {
            points.push(PointCase {
                case_id: format!("tail_chdtri_df{df}_q{q:e}"),
                func: "chdtri".to_string(),
                p1: df,
                p2: 0.0,
                arg: q,
            });
        }
    }
    for &k in &[0.0_f64, 5.0, 30.0, 2.7] {
        for &q in &tails {
            points.push(PointCase {
                case_id: format!("tail_pdtri_k{k}_q{q:e}"),
                func: "pdtri".to_string(),
                p1: k,
                p2: 0.0,
                arg: q,
            });
        }
    }
    for &(a, x) in &[
        (50.0_f64, 45.0),
        (150.0, 160.0),
        (300.0, 400.0),
        (1e4, 1.01e4),
        (250.0, 20.0),
    ] {
        for func in ["chdtr", "chdtrc"] {
            points.push(PointCase {
                case_id: format!("wide_{func}_df{}_x{}", 2.0 * a, 2.0 * x),
                func: func.to_string(),
                p1: 2.0 * a,
                p2: 0.0,
                arg: 2.0 * x,
            });
        }
        for func in ["pdtr", "pdtrc"] {
            points.push(PointCase {
                case_id: format!("wide_{func}_k{a}_mu{x}"),
                func: func.to_string(),
                p1: a,
                p2: 0.0,
                arg: x,
            });
        }
        for func in ["gdtr", "gdtrc"] {
            points.push(PointCase {
                case_id: format!("wide_{func}_a0.5_b{a}_x{}", 2.0 * x),
                func: func.to_string(),
                p1: 0.5,
                p2: a,
                arg: 2.0 * x,
            });
        }
    }
    // The complements for x ≫ a, x from 1e-10 to 1e4: a forward-CDF sweep over that range found
    // the old pdtrc 6.05e-3, gdtrc 2.6e-4, pdtr 1.5e-4 and chdtrc 6.4e-5 off.
    let wide_xs = [1e-10_f64, 1e-3, 50.0, 1e3, 1e4];
    for &k in &[0.0_f64, 3.0, 12.0, 29.0] {
        for &mu in &wide_xs {
            for func in ["pdtr", "pdtrc"] {
                points.push(PointCase {
                    case_id: format!("widex_{func}_k{k}_mu{mu:e}"),
                    func: func.to_string(),
                    p1: k,
                    p2: 0.0,
                    arg: mu,
                });
            }
        }
    }
    for &shape in &[0.5_f64, 7.0, 19.0] {
        for &x in &wide_xs {
            for func in ["gdtr", "gdtrc"] {
                points.push(PointCase {
                    case_id: format!("widex_{func}_a0.7_b{shape}_x{x:e}"),
                    func: func.to_string(),
                    p1: 0.7,
                    p2: shape,
                    arg: x,
                });
            }
            for func in ["chdtr", "chdtrc"] {
                points.push(PointCase {
                    case_id: format!("widex_{func}_df{}_x{x:e}", 2.0 * shape),
                    func: func.to_string(),
                    p1: 2.0 * shape,
                    p2: 0.0,
                    arg: x,
                });
            }
        }
    }
    // SciPy's edges that a plain domain check gets wrong: a zero degree of freedom, shape or
    // rate is not NaN (chdtr(0, 1) = 1, gdtr(1, 0, 3) = 1, gdtr(0, 2, 3) = 0), and
    // chdtri(0, 1) is 0.
    let edges: [(&str, f64, f64, f64); 7] = [
        ("chdtr", 0.0, 0.0, 1.0),
        ("chdtrc", 0.0, 0.0, 1.0),
        ("gdtr", 1.0, 0.0, 3.0),
        ("gdtrc", 1.0, 0.0, 3.0),
        ("gdtr", 0.0, 2.0, 3.0),
        ("gdtrc", 0.0, 2.0, 3.0),
        ("chdtri", 0.0, 0.0, 1.0),
    ];
    for (func, p1, p2, arg) in edges {
        points.push(PointCase {
            case_id: format!("edge_{func}_{p1}_{p2}_{arg}"),
            func: func.to_string(),
            p1,
            p2,
            arg,
        });
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
    p1 = float(case["p1"]); p2 = float(case["p2"]); arg = float(case["arg"])
    try:
        if func == "gdtr":     value = special.gdtr(p1, p2, arg)
        elif func == "gdtrc":  value = special.gdtrc(p1, p2, arg)
        elif func == "pdtr":   value = special.pdtr(p1, arg)
        elif func == "pdtrc":  value = special.pdtrc(p1, arg)
        elif func == "pdtri":  value = special.pdtri(p1, arg)
        elif func == "chdtr":  value = special.chdtr(p1, arg)
        elif func == "chdtrc": value = special.chdtrc(p1, arg)
        elif func == "chdtri": value = special.chdtri(p1, arg)
        else: value = None
        points.append({"case_id": cid, "value": finite_or_none(value)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;

    let query_json = serde_json::to_string(query).expect("serialize gdtr/pdtr/chdtr query");
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
                "failed to spawn python3 for gdtr/pdtr/chdtr oracle: {e}"
            );
            eprintln!("skipping gdtr/pdtr/chdtr oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open gdtr/pdtr/chdtr oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "gdtr/pdtr/chdtr oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping gdtr/pdtr/chdtr oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for gdtr/pdtr/chdtr oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "gdtr/pdtr/chdtr oracle failed: {stderr}"
        );
        eprintln!("skipping gdtr/pdtr/chdtr oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse gdtr/pdtr/chdtr oracle JSON"))
}

#[test]
fn diff_special_gdtr_pdtr_chdtr() {
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
    let mut max_abs_overall = 0.0_f64;
    let mut max_rel_overall = 0.0_f64;
    let mut ledger = CompareLedger::new("diff_special_gdtr_pdtr_chdtr", &ARMS);

    for case in &query.points {
        let oracle = pmap.get(&case.case_id).expect("validated oracle");
        let arm = case.func.as_str();
        let Some((scipy_v, rust_v)) = ledger.pair(
            arm,
            &case.case_id,
            oracle.value,
            fsci_eval(&case.func, case.p1, case.p2, case.arg),
        ) else {
            continue;
        };
        let abs_diff = (rust_v - scipy_v).abs();
        let scale = scipy_v.abs().max(1.0);
        let rel_diff = abs_diff / scale;
        max_abs_overall = max_abs_overall.max(abs_diff);
        max_rel_overall = max_rel_overall.max(rel_diff);

        // Bit for bit, the sign of a zero included (frankenscipy-449uv).
        let pass = rust_v.to_bits() == scipy_v.to_bits();
        ledger.compared(arm, &case.case_id, pass);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            func: case.func.clone(),
            abs_diff,
            rel_diff,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_gdtr_pdtr_chdtr".into(),
        category: "scipy.special.gdtr/pdtr/chdtr family".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        max_abs_diff: max_abs_overall,
        max_rel_diff: max_rel_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };

    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "gdtr/pdtr/chdtr {} mismatch: {} abs={} rel={}",
                d.func, d.case_id, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special gdtr/pdtr/chdtr conformance failed: {} cases, max_abs={} max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
    // Arms have different case sets (gdtr/gdtrc and chdtr/chdtrc have the fewest); each must
    // compare all of its own.
    let min_per_arm = ARMS
        .iter()
        .map(|arm| query.points.iter().filter(|c| c.func == *arm).count())
        .min()
        .expect("ARMS is non-empty");
    ledger.finish(min_per_arm);
}
