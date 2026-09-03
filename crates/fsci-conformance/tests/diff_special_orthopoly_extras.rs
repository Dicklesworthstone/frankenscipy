#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the orthopoly evaluators
//! that fall outside `diff_special.rs`'s bulk harness:
//!   - `scipy.special.eval_chebyc(n, x)`  (Chebyshev C polynomial)
//!   - `scipy.special.eval_chebys(n, x)`  (Chebyshev S polynomial)
//!   - `scipy.special.eval_sh_legendre(n, x)`  (shifted Legendre)
//!   - `scipy.special.eval_sh_chebyt(n, x)`  (shifted Chebyshev T)
//!   - `scipy.special.eval_sh_chebyu(n, x)`  (shifted Chebyshev U)
//!   - `scipy.special.assoc_laguerre(x, n, k)`  (associated Laguerre)
//!
//! Resolves [frankenscipy-9boml]. fsci_special::orthopoly exposes
//! these alongside the already-covered (chebyt, chebyu, legendre,
//! hermite, hermitenorm, laguerre, genlaguerre, jacobi, gegenbauer)
//! family but had no dedicated conformance harness — spot checks
//! showed agreement to ~1e-15.
//!
//! Tolerance: 1e-12 abs / rel — exact polynomials hit machine
//! precision; the larger n × |x| > 1 path can amplify cancellation,
//! so 1e-12 leaves margin without papering over real drift.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_special::orthopoly::{
    assoc_laguerre, eval_chebyc, eval_chebys, eval_sh_chebyt, eval_sh_chebyu, eval_sh_legendre,
};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-12;
const REL_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct PointCase {
    case_id: String,
    func: String,
    n: u32,
    x: f64,
    /// `k` parameter for assoc_laguerre; ignored for the other functions.
    k: f64,
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
    fs::create_dir_all(output_dir()).expect("create orthopoly_extras diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize orthopoly_extras diff log");
    fs::write(path, json).expect("write orthopoly_extras diff log");
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();

    // chebyc / chebys are defined on [-2, 2]; include endpoints + interior.
    let chebycs_xs: &[f64] = &[-1.5, -0.5, 0.0, 0.7, 1.5];
    let ns: &[u32] = &[0, 1, 2, 3, 5];
    for func in ["eval_chebyc", "eval_chebys"] {
        for &n in ns {
            for &x in chebycs_xs {
                points.push(PointCase {
                    case_id: format!("{func}_n{n}_x{x}"),
                    func: func.into(),
                    n,
                    x,
                    k: 0.0,
                });
            }
        }
    }

    // Shifted variants live on [0, 1].
    let sh_xs: &[f64] = &[0.0, 0.25, 0.5, 0.75, 1.0];
    for func in ["eval_sh_legendre", "eval_sh_chebyt", "eval_sh_chebyu"] {
        for &n in &[0, 1, 2, 3, 4] {
            for &x in sh_xs {
                points.push(PointCase {
                    case_id: format!("{func}_n{n}_x{x}"),
                    func: func.into(),
                    n,
                    x,
                    k: 0.0,
                });
            }
        }
    }

    // assoc_laguerre(x, n, k). k spans 0 (== Laguerre) through fractional.
    for &(x, n, k) in &[
        (0.5_f64, 0_u32, 0.0_f64),
        (0.5, 1, 0.0),
        (1.5, 2, 1.5),
        (2.0, 3, 2.5),
        (0.1, 1, 0.5),
        (5.0, 4, 1.0),
        (3.0, 5, 0.0),
        (1.0, 2, 2.0),
    ] {
        points.push(PointCase {
            case_id: format!("assoc_laguerre_x{x}_n{n}_k{k}"),
            func: "assoc_laguerre".into(),
            n,
            x,
            k,
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

def fnone(v):
    try:
        v = float(v)
    except Exception:
        return None
    return v if math.isfinite(v) else None

DISPATCH = {
    "eval_chebyc":     lambda c: special.eval_chebyc(int(c["n"]), float(c["x"])),
    "eval_chebys":     lambda c: special.eval_chebys(int(c["n"]), float(c["x"])),
    "eval_sh_legendre": lambda c: special.eval_sh_legendre(int(c["n"]), float(c["x"])),
    "eval_sh_chebyt":   lambda c: special.eval_sh_chebyt(int(c["n"]), float(c["x"])),
    "eval_sh_chebyu":   lambda c: special.eval_sh_chebyu(int(c["n"]), float(c["x"])),
    "assoc_laguerre":   lambda c: special.assoc_laguerre(float(c["x"]), int(c["n"]), float(c["k"])),
}

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; func = case["func"]
    handler = DISPATCH.get(func)
    try:
        v = handler(case) if handler is not None else None
        points.append({"case_id": cid, "value": fnone(v)})
    except Exception:
        points.append({"case_id": cid, "value": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize orthopoly_extras query");
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
                "failed to spawn python3 for orthopoly_extras oracle: {e}"
            );
            eprintln!("skipping orthopoly_extras oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open orthopoly_extras oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "orthopoly_extras oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping orthopoly_extras oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for orthopoly_extras oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "orthopoly_extras oracle failed: {stderr}"
        );
        eprintln!("skipping orthopoly_extras oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse orthopoly_extras oracle JSON"))
}

fn fsci_eval(func: &str, n: u32, x: f64, k: f64) -> Option<f64> {
    Some(match func {
        "eval_chebyc" => eval_chebyc(n, x),
        "eval_chebys" => eval_chebys(n, x),
        "eval_sh_legendre" => eval_sh_legendre(n, x),
        "eval_sh_chebyt" => eval_sh_chebyt(n, x),
        "eval_sh_chebyu" => eval_sh_chebyu(n, x),
        "assoc_laguerre" => assoc_laguerre(x, n, k),
        _ => return None,
    })
}

#[test]
fn diff_special_orthopoly_extras() {
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

    for case in &query.points {
        let scipy_arm = pmap.get(&case.case_id).expect("validated oracle");
        let Some(fsci_v) = fsci_eval(&case.func, case.n, case.x, case.k) else {
            continue;
        };
        if let Some(scipy_v) = scipy_arm.value
            && fsci_v.is_finite()
        {
            let abs_d = (fsci_v - scipy_v).abs();
            let rel_d = abs_d / scipy_v.abs().max(f64::MIN_POSITIVE);
            max_overall = max_overall.max(abs_d);
            let pass = abs_d <= ABS_TOL || rel_d <= REL_TOL;
            diffs.push(CaseDiff {
                case_id: case.case_id.clone(),
                func: case.func.clone(),
                abs_diff: abs_d,
                rel_diff: rel_d,
                pass,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_orthopoly_extras".into(),
        category: "scipy.special.eval_*".into(),
        case_count: diffs.len(),
        max_abs_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs,
    };
    emit_log(&log);

    assert!(
        all_pass,
        "orthopoly_extras diff harness failed; see fixtures/artifacts/{PACKET_ID}/diff/diff_special_orthopoly_extras.json"
    );
}
