#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the Carlson symmetric elliptic
//! integral family — RC, RF, RD, RG, RJ.
//!
//! Resolves [frankenscipy-q5es6]. Drives a curated set of (x, y, z)
//! cases — including the just-shipped RC(x, y<0) Cauchy-PV branch — and
//! compares against `scipy.special.elliprc/d/f/g/j` via subprocess oracle.
//! Skips cleanly when scipy/python3 is unavailable unless
//! `FSCI_REQUIRE_SCIPY_ORACLE` is set.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_special::{elliprc, elliprd, elliprf, elliprg, elliprj};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
// elliprc is SciPy's own `ellint_carlson::rc` (frankenscipy-2f8h7), so it must be SciPy's bits.
// RF, RD, RG and RJ are fsci's duplication kernels: over 200,000 random points each, their
// largest relative difference from SciPy 1.17.1 was 2.8e-14 (RG), and RJ's was 1.7e-15 once RC
// was exact. The former 1e-7 absolute-or-pseudo-relative gate could not see RC's 1.9e-11 error
// near its diagonal, nor the 8e-5 it caused in RJ where p is close to x.
const REL_TOL: f64 = 1.0e-13;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
/// One ledger arm per Carlson integral.
const ARMS: [&str; 5] = ["elliprc", "elliprf", "elliprd", "elliprg", "elliprj"];

#[derive(Debug, Clone, Serialize)]
struct CarlsonCase {
    case_id: String,
    op: String,
    x: f64,
    y: f64,
    z: Option<f64>,
    p: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct CarlsonOracleResult {
    case_id: String,
    value: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
    rust: f64,
    scipy: f64,
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
    rel_tol: f64,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn carlson_binary_case(case_id: String, op: &str, x: f64, y: f64) -> CarlsonCase {
    CarlsonCase {
        case_id,
        op: op.into(),
        x,
        y,
        z: None,
        p: None,
    }
}

fn carlson_ternary_case(case_id: String, op: &str, x: f64, y: f64, z: f64) -> CarlsonCase {
    CarlsonCase {
        case_id,
        op: op.into(),
        x,
        y,
        z: Some(z),
        p: None,
    }
}

fn carlson_rj_case(case_id: String, x: f64, y: f64, z: f64, p: f64) -> CarlsonCase {
    CarlsonCase {
        case_id,
        op: "elliprj".into(),
        x,
        y,
        z: Some(z),
        p: Some(p),
    }
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create carlson diff output dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize carlson diff log");
    fs::write(path, json).expect("write carlson diff log");
}

fn generate_carlson_cases() -> Vec<CarlsonCase> {
    let mut cases = Vec::new();
    let mut idx = 0_usize;

    // RC(x, y) — positive y branch (acos / acosh / equal-arg).
    for &(x, y) in &[
        (1.0_f64, 1.0),
        (1.0, 2.0),
        (2.0, 1.0),
        (4.0, 1.0),
        (0.5, 4.0),
        (3.0, 7.5),
        (0.25, 0.75),
        (10.0, 0.1),
        (0.0, 1.0),
        (0.0, 4.0),
        // Near the diagonal, where the arccos/arccosh closed forms cancelled; the third pair
        // is the one elliprj's duplication handed RC at the RJ point below.
        (4.0, 4.000000000003638),
        (4.0, 3.999999999996362),
        (9.955270134189597, 9.955274904175287),
        (1e-300, 1.0000000010000002e-300),
        (1e300, 1e-300),
        (169768.46417187434, 2557356.0496944487),
    ] {
        cases.push(carlson_binary_case(
            format!("rc_pos_{idx:02}"),
            "elliprc",
            x,
            y,
        ));
        idx += 1;
    }

    // RC(x, y<0) — Cauchy-PV branch [frankenscipy-43vts].
    idx = 0;
    for &(x, y) in &[
        (1.0_f64, -1.0),
        (3.0, -1.0),
        (2.0, -0.5),
        (0.5, -2.0),
        (0.0, -1.0),
        (0.0, -3.0),
        (5.0, -0.25),
    ] {
        cases.push(carlson_binary_case(
            format!("rc_pv_{idx:02}"),
            "elliprc",
            x,
            y,
        ));
        idx += 1;
    }

    // RF(x, y, z) — symmetric in three non-negative args, ≤ 1 zero.
    idx = 0;
    for &(x, y, z) in &[
        (1.0_f64, 2.0, 3.0),
        (0.5, 1.5, 4.0),
        (0.25, 0.5, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.0, 2.0, 4.0),
        (3.0, 5.0, 7.0),
        (0.1, 0.2, 0.3),
        (10.0, 20.0, 30.0),
    ] {
        cases.push(carlson_ternary_case(
            format!("rf_{idx:02}"),
            "elliprf",
            x,
            y,
            z,
        ));
        idx += 1;
    }

    // RD(x, y, z) — symmetric only in (x, y); z must be > 0; ≤ 1 zero in (x, y).
    idx = 0;
    for &(x, y, z) in &[
        (1.0_f64, 2.0, 3.0),
        (0.5, 1.5, 4.0),
        (0.25, 0.5, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.0, 2.0, 4.0),
        (3.0, 5.0, 7.0),
        (0.1, 0.2, 0.3),
    ] {
        cases.push(carlson_ternary_case(
            format!("rd_{idx:02}"),
            "elliprd",
            x,
            y,
            z,
        ));
        idx += 1;
    }

    // RG(x, y, z) — fully symmetric, all three non-negative.
    idx = 0;
    for &(x, y, z) in &[
        (1.0_f64, 2.0, 3.0),
        (0.5, 1.5, 4.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.0, 0.0, 4.0),
        (3.0, 5.0, 7.0),
        (0.1, 0.5, 2.0),
    ] {
        cases.push(carlson_ternary_case(
            format!("rg_{idx:02}"),
            "elliprg",
            x,
            y,
            z,
        ));
        idx += 1;
    }

    // RJ(x, y, z, p) — symmetric in (x, y, z), p is distinct and must be positive.
    idx = 0;
    for &(x, y, z, p) in &[
        (1.0_f64, 2.0, 3.0, 4.0),
        (1.0, 1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0, 1.0),
        (0.5, 1.5, 4.0, 2.0),
        (0.25, 0.5, 1.0, 0.75),
        (3.0, 5.0, 7.0, 11.0),
        (0.1, 0.2, 0.3, 0.4),
        // p close to one of x, y, z: RJ's duplication then calls RC near its diagonal. The
        // first point is the n-ary perf sweep's worst, 7.95e-5 off before RC was exact.
        (
            9.955270134189597,
            2.8704213887358336,
            9.51902144293567,
            9.955274904175287,
        ),
        (1.0, 2.0, 3.0, 1.000001),
        (0.5, 4.0, 2.0, 0.5000001),
        (2.0, 3.0, 4.0, 3.000000001),
        (1.0, 2.0, 5.0, 4.9999999),
    ] {
        cases.push(carlson_rj_case(format!("rj_{idx:02}"), x, y, z, p));
        idx += 1;
    }

    cases
}

fn scipy_oracle_or_skip(cases: &[CarlsonCase]) -> Vec<CarlsonOracleResult> {
    let script = r#"
import json
import sys
from scipy import special

cases = json.load(sys.stdin)
results = []
for c in cases:
    cid = c["case_id"]
    op = c["op"]
    x = c["x"]; y = c["y"]; z = c.get("z"); p = c.get("p")
    try:
        if op == "elliprc":
            val = float(special.elliprc(x, y))
        elif op == "elliprf":
            val = float(special.elliprf(x, y, z))
        elif op == "elliprd":
            val = float(special.elliprd(x, y, z))
        elif op == "elliprg":
            val = float(special.elliprg(x, y, z))
        elif op == "elliprj":
            val = float(special.elliprj(x, y, z, p))
        else:
            val = None
        results.append({"case_id": cid, "value": val})
    except Exception:
        results.append({"case_id": cid, "value": None})

print(json.dumps(results))
"#;

    let cases_json = serde_json::to_string(cases).expect("serialize carlson cases");

    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for carlson oracle: {e}"
            );
            eprintln!("skipping carlson oracle: python3 not available ({e})");
            return Vec::new();
        }
    };

    {
        let stdin = child.stdin.as_mut().expect("open carlson oracle stdin");
        if let Err(err) = stdin.write_all(cases_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "carlson oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping carlson oracle: stdin write failed ({err})\n{stderr}");
            return Vec::new();
        }
    }

    let output = child.wait_with_output().expect("wait for carlson oracle");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "carlson oracle failed: {stderr}"
        );
        eprintln!("skipping carlson oracle: scipy not available\n{stderr}");
        return Vec::new();
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(&stdout).expect("parse carlson oracle JSON")
}

fn rust_eval(case: &CarlsonCase) -> f64 {
    match case.op.as_str() {
        "elliprc" => elliprc(case.x, case.y),
        "elliprf" => elliprf(case.x, case.y, case.z.expect("rf needs z")),
        "elliprd" => elliprd(case.x, case.y, case.z.expect("rd needs z")),
        "elliprg" => elliprg(case.x, case.y, case.z.expect("rg needs z")),
        "elliprj" => elliprj(
            case.x,
            case.y,
            case.z.expect("rj needs z"),
            case.p.expect("rj needs p"),
        ),
        other => panic!("unknown carlson op {other}"),
    }
}

#[test]
fn diff_special_carlson() {
    let cases = generate_carlson_cases();
    let oracle_results = scipy_oracle_or_skip(&cases);

    if oracle_results.is_empty() {
        return;
    }

    assert_eq!(
        oracle_results.len(),
        cases.len(),
        "SciPy carlson oracle returned partial coverage"
    );

    let oracle_map: HashMap<String, CarlsonOracleResult> = oracle_results
        .into_iter()
        .map(|r| (r.case_id.clone(), r))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_abs_overall = 0.0_f64;
    let mut max_rel_overall = 0.0_f64;
    let mut ledger = CompareLedger::new("diff_special_carlson", &ARMS);

    for case in &cases {
        let scipy = oracle_map.get(&case.case_id).and_then(|r| r.value);
        // A SciPy refusal is recorded as a missing oracle value, and a non-finite Rust value
        // against a finite SciPy one as an fsci failure, instead of being skipped.
        let Some((scipy_value, rust_value)) =
            ledger.pair(&case.op, &case.case_id, scipy, Some(rust_eval(case)))
        else {
            continue;
        };

        let abs_diff = (rust_value - scipy_value).abs();
        let rel_diff = abs_diff / scipy_value.abs().max(f64::MIN_POSITIVE);
        let pass = if case.op == "elliprc" {
            rust_value.to_bits() == scipy_value.to_bits()
        } else {
            rel_diff <= REL_TOL
        };
        ledger.compared(&case.op, &case.case_id, pass);

        max_abs_overall = max_abs_overall.max(abs_diff);
        max_rel_overall = max_rel_overall.max(rel_diff);

        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            rust: rust_value,
            scipy: scipy_value,
            abs_diff,
            rel_diff,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_special_carlson".into(),
        category: "scipy.special.elliprc/d/f/g/j".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        max_abs_diff: max_abs_overall,
        max_rel_diff: max_rel_overall,
        rel_tol: REL_TOL,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };

    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!(
                "carlson mismatch: {} {} rust={} scipy={} abs={} rel={}",
                d.op, d.case_id, d.rust, d.scipy, d.abs_diff, d.rel_diff
            );
        }
    }

    assert!(
        all_pass,
        "scipy.special carlson conformance failed: {} cases, max_abs={}, max_rel={}",
        diffs.len(),
        max_abs_overall,
        max_rel_overall
    );
    // Arms have different case sets (elliprg/elliprj have the fewest); each must compare all
    // of its own.
    let min_per_arm = ARMS
        .iter()
        .map(|arm| cases.iter().filter(|c| c.op == *arm).count())
        .min()
        .expect("ARMS is non-empty");
    ledger.finish(min_per_arm);
}
