#![forbid(unsafe_code)]
//! Live scipy.signal parity for fsci_signal lowpass-prototype
//! transformations and the bilinear ZPK transform.
//!
//! Resolves [frankenscipy-j9cgc].
//!
//! Restricted to `lp2lp_zpk` and `bilinear_zpk` (with n=2..4 only).
//! `lp2hp_zpk`, `lp2bp_zpk`, `lp2bs_zpk` diverge from scipy by
//! sqrt(2)–7 abs (defect frankenscipy-8fw59 — likely a sign or
//! gain-multiplier convention mismatch in how zeros-at-infinity
//! propagate). bilinear at n=5/fs=1 also diverges; restricted to
//! n≤4 here.
//!
//! Each transform takes a normalized lowpass prototype from
//! buttap(n). Roots are sorted by (re, im) before comparing.
//! Tolerance 1e-10 abs.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{bilinear_zpk, buttap, lp2bp_zpk, lp2bs_zpk, lp2hp_zpk, lp2lp_zpk};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-10;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Debug, Clone, Serialize)]
struct Case {
    case_id: String,
    op: String, // "lplp" | "lphp" | "lpbp" | "lpbs" | "bilinear"
    n: u32,
    /// lplp/lphp
    wo: f64,
    /// lpbp/lpbs
    bw: f64,
    /// bilinear
    fs: f64,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<Case>,
}

#[derive(Debug, Clone, Deserialize)]
struct PointArm {
    case_id: String,
    /// zeros sorted by (re, im) flat; then poles sorted; then [k]
    z_re: Option<Vec<f64>>,
    z_im: Option<Vec<f64>>,
    p_re: Option<Vec<f64>>,
    p_im: Option<Vec<f64>>,
    k: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    op: String,
    abs_diff: f64,
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
    fs::create_dir_all(output_dir()).expect("create lp2_zpk diff dir");
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

fn emit_log(log: &DiffLog) {
    ensure_output_dir();
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize log");
    fs::write(path, json).expect("write log");
}

/// Max residual of a tolerance-greedy bijection between two root sets.
///
/// zpk roots have no canonical order, so comparing them position-wise
/// after a lexicographic sort is fragile: a sub-ULP difference in the
/// real part of a conjugate pair flips the sort and pairs `+iω` against
/// `−iω`. Matching each root to its nearest unused partner is
/// order-independent and robust to that noise.
fn root_set_max_diff(a: &[(f64, f64)], b: &[(f64, f64)]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    let mut used = vec![false; b.len()];
    let mut worst = 0.0_f64;
    for &(ar, ai) in a {
        let mut best = f64::INFINITY;
        let mut best_j = None;
        for (j, &(br, bi)) in b.iter().enumerate() {
            if used[j] {
                continue;
            }
            let d = (ar - br).abs().max((ai - bi).abs());
            if d < best {
                best = d;
                best_j = Some(j);
            }
        }
        match best_j {
            Some(j) => {
                used[j] = true;
                worst = worst.max(best);
            }
            None => return f64::INFINITY,
        }
    }
    worst
}

fn generate_query() -> OracleQuery {
    let mut points = Vec::new();
    // All five zpk transforms; roots are sorted before comparison so
    // emission order does not matter (frankenscipy-8fw59).
    for &n in &[2_u32, 3, 4] {
        for &wo in &[1.0_f64, 0.5, 2.5] {
            points.push(Case {
                case_id: format!("lplp_n{n}_wo{wo}"),
                op: "lplp".into(),
                n,
                wo,
                bw: 0.0,
                fs: 0.0,
            });
            points.push(Case {
                case_id: format!("lphp_n{n}_wo{wo}"),
                op: "lphp".into(),
                n,
                wo,
                bw: 0.0,
                fs: 0.0,
            });
            for &bw in &[0.5_f64, 1.5] {
                points.push(Case {
                    case_id: format!("lpbp_n{n}_wo{wo}_bw{bw}"),
                    op: "lpbp".into(),
                    n,
                    wo,
                    bw,
                    fs: 0.0,
                });
                points.push(Case {
                    case_id: format!("lpbs_n{n}_wo{wo}_bw{bw}"),
                    op: "lpbs".into(),
                    n,
                    wo,
                    bw,
                    fs: 0.0,
                });
            }
        }
        for &fs in &[2.0_f64, 10.0] {
            points.push(Case {
                case_id: format!("bilinear_n{n}_fs{fs}"),
                op: "bilinear".into(),
                n,
                wo: 0.0,
                bw: 0.0,
                fs,
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
import numpy as np
from scipy.signal import buttap, lp2lp_zpk, lp2hp_zpk, lp2bp_zpk, lp2bs_zpk, bilinear_zpk

def sort_pairs(arr):
    pairs = [(float(c.real), float(c.imag)) for c in arr]
    pairs.sort()
    re = [p[0] for p in pairs]
    im = [p[1] for p in pairs]
    return re, im

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]; op = case["op"]; n = int(case["n"])
    try:
        z, p, k = buttap(n)
        if op == "lplp":
            wo = float(case["wo"])
            zo, po, ko = lp2lp_zpk(z, p, k, wo=wo)
        elif op == "lphp":
            wo = float(case["wo"])
            zo, po, ko = lp2hp_zpk(z, p, k, wo=wo)
        elif op == "lpbp":
            wo = float(case["wo"]); bw = float(case["bw"])
            zo, po, ko = lp2bp_zpk(z, p, k, wo=wo, bw=bw)
        elif op == "lpbs":
            wo = float(case["wo"]); bw = float(case["bw"])
            zo, po, ko = lp2bs_zpk(z, p, k, wo=wo, bw=bw)
        elif op == "bilinear":
            fs = float(case["fs"])
            zo, po, ko = bilinear_zpk(z, p, k, fs=fs)
        else:
            points.append({"case_id": cid, "z_re": None, "z_im": None, "p_re": None, "p_im": None, "k": None}); continue
        z_re, z_im = sort_pairs(zo)
        p_re, p_im = sort_pairs(po)
        if (all(math.isfinite(v) for v in z_re + z_im + p_re + p_im) and math.isfinite(ko)):
            points.append({"case_id": cid, "z_re": z_re, "z_im": z_im, "p_re": p_re, "p_im": p_im, "k": float(ko)})
        else:
            points.append({"case_id": cid, "z_re": None, "z_im": None, "p_re": None, "p_im": None, "k": None})
    except Exception as e:
        sys.stderr.write(f"oracle {cid}: {e}\n")
        points.append({"case_id": cid, "z_re": None, "z_im": None, "p_re": None, "p_im": None, "k": None})
print(json.dumps({"points": points}))
"#;
    let query_json = serde_json::to_string(query).expect("serialize query");
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
                "failed to spawn python3 for lp2_zpk oracle: {e}"
            );
            eprintln!("skipping lp2_zpk oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "lp2_zpk oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping lp2_zpk oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child.wait_with_output().expect("wait for lp2_zpk oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "lp2_zpk oracle failed: {stderr}"
        );
        eprintln!("skipping lp2_zpk oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse lp2_zpk oracle JSON"))
}

#[test]
fn diff_signal_lp2_zpk_transforms() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };

    let pmap: HashMap<String, PointArm> = oracle
        .points
        .into_iter()
        .map(|d| (d.case_id.clone(), d))
        .collect();

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &query.points {
        let Some(arm) = pmap.get(&case.case_id) else {
            continue;
        };
        let (Some(z_re), Some(z_im), Some(p_re), Some(p_im), Some(k_exp)) = (
            arm.z_re.as_ref(),
            arm.z_im.as_ref(),
            arm.p_re.as_ref(),
            arm.p_im.as_ref(),
            arm.k,
        ) else {
            continue;
        };
        let Ok((z, p, k)) = buttap(case.n) else {
            continue;
        };
        let res = match case.op.as_str() {
            "lplp" => lp2lp_zpk(&z, &p, k, case.wo),
            "lphp" => lp2hp_zpk(&z, &p, k, case.wo),
            "lpbp" => lp2bp_zpk(&z, &p, k, case.wo, case.bw),
            "lpbs" => lp2bs_zpk(&z, &p, k, case.wo, case.bw),
            "bilinear" => bilinear_zpk(&z, &p, k, case.fs),
            _ => continue,
        };
        let Ok((zout, pout, kout)) = res else {
            continue;
        };
        let scipy_z: Vec<(f64, f64)> = z_re.iter().copied().zip(z_im.iter().copied()).collect();
        let scipy_p: Vec<(f64, f64)> = p_re.iter().copied().zip(p_im.iter().copied()).collect();
        let max_d = root_set_max_diff(&zout, &scipy_z)
            .max(root_set_max_diff(&pout, &scipy_p))
            .max((kout - k_exp).abs());
        max_overall = max_overall.max(max_d);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            op: case.op.clone(),
            abs_diff: max_d,
            pass: max_d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_signal_lp2_zpk_transforms".into(),
        category:
            "fsci_signal::{lp2lp_zpk, lp2hp_zpk, lp2bp_zpk, lp2bs_zpk, bilinear_zpk} vs scipy.signal"
                .into(),
        case_count: diffs.len(),
        max_abs_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("{} mismatch: {} abs_diff={}", d.op, d.case_id, d.abs_diff);
        }
    }

    assert!(
        all_pass,
        "lp2_zpk conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
}
