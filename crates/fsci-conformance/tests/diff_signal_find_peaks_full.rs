#![forbid(unsafe_code)]
//! Live SciPy differential coverage for the FULL `fsci_signal::find_peaks` surface:
//! height / threshold / distance / prominence / width / wlen / rel_height / plateau_size, in
//! scalar, `(min, max)` and `(None, max)` interval forms, and every property SciPy returns.
//!
//! Resolves [frankenscipy-szq1n.10]: `width` used to be ignored and `distance` ran after
//! `prominence`, so the order-sensitive cases here disagreed with SciPy.
//!
//! 500 seeded signals, lengths 5-200. Most are small integers, so ties, plateaus and equal-height
//! neighbours are common; the rest are continuous. Peak indices and integer properties must
//! match exactly; float properties to 1e-12 (relative above 1, absolute below).
//!
//! SciPy's `distance` filter visits peaks in `np.argsort(priority)` order, and numpy 2.4's
//! default argsort breaks TIES differently depending on the CPU it dispatches to: x86-simd-sort
//! on X86_V3 (AVX2) / X86_V4 (AVX-512), numpy's portable introsort otherwise. So the incumbent
//! is not one function on equal-height candidates (frankenscipy-80z9v). Measured 2026-09-24 with
//! the same SciPy 1.17.1 / numpy 2.4.3 on one AVX2 machine: 23 of these 500 cases change between
//! the default dispatch and `NPY_DISABLE_CPU_FEATURES` = every dispatch group (fp_437: [.., 34,
//! ..] vs [.., 37, ..]). The oracle therefore runs twice — numpy's default dispatch and its
//! portable path — and a case passes when fsci equals SciPy under either; a case matching neither
//! fails as before. The split (both / default only / portable only / neither) is printed and
//! logged, and a canary in the oracle fails the test if the portable arm did not really take the
//! portable argsort.
//!
//! The admission is not what made the cases pass. CI's pinned run failed 20 of 500. With the
//! same peaks the portable path would have admitted only 5 of them: fsci ordered ties by index,
//! but numpy's portable argsort is an introsort whose tie order is index order only below 17
//! elements. fsci-signal now ports that argsort (`numpy_argsort`). Measured locally on the AVX2
//! machine, fsci then agrees with SciPy on all 500: 477 under both dispatches, 23 under the
//! portable path only, 0 under neither.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_signal::{FindPeaksOptions, FindPeaksResult, PeakCondition, find_peaks};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
const ABS_TOL: f64 = 1.0e-12;
const REL_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const CASES: usize = 500;
/// numpy 2.4's x86-64 SIMD dispatch groups. Naming all of them in `NPY_DISABLE_CPU_FEATURES`
/// leaves numpy's portable `argsort`; naming a feature outside the list (`AVX2`, `AVX512F`) is
/// rejected with an ImportWarning and the variable is ignored, which the canary catches.
const NUMPY_DISPATCH_GROUPS: &str = "X86_V3 X86_V4 AVX512_ICL AVX512_SPR";

#[derive(Debug, Clone, Copy, Serialize)]
struct Cond {
    min: Option<f64>,
    max: Option<f64>,
}

impl From<Cond> for PeakCondition {
    fn from(c: Cond) -> Self {
        Self {
            min: c.min,
            max: c.max,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct PeakCase {
    case_id: String,
    x: Vec<f64>,
    height: Option<Cond>,
    threshold: Option<Cond>,
    distance: Option<usize>,
    prominence: Option<Cond>,
    width: Option<Cond>,
    wlen: Option<f64>,
    rel_height: f64,
    plateau_size: Option<Cond>,
}

#[derive(Debug, Clone, Serialize)]
struct OracleQuery {
    points: Vec<PeakCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct PeakArm {
    case_id: String,
    error: Option<String>,
    peaks: Option<Vec<usize>>,
    props: Option<HashMap<String, Vec<f64>>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleResult {
    points: Vec<PeakArm>,
    /// numpy's default `argsort` broke the canary's ties in index order in this process.
    argsort_is_stable: bool,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    max_diff: f64,
    pass: bool,
    detail: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: usize,
    max_diff: f64,
    pass: bool,
    /// Cases where SciPy's default-dispatch and portable numpy disagree.
    isa_sensitive: usize,
    agree_both: usize,
    agree_default_only: usize,
    agree_portable_only: usize,
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
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn emit_log(log: &DiffLog) {
    fs::create_dir_all(output_dir()).expect("create find_peaks diff output dir");
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize find_peaks diff log");
    fs::write(path, json).expect("write find_peaks diff log");
}

/// xorshift64*: deterministic, dependency-free case generation.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    fn chance(&mut self, p: f64) -> bool {
        self.unit() < p
    }
    /// A scalar, a `(min, max)` or a `(None, max)` condition around `[lo, hi]`.
    fn cond(&mut self, lo: f64, hi: f64) -> Cond {
        let a = lo + (hi - lo) * self.unit();
        let b = lo + (hi - lo) * self.unit();
        match self.below(3) {
            0 => Cond {
                min: Some(a),
                max: None,
            },
            1 => Cond {
                min: Some(a.min(b)),
                max: Some(a.max(b)),
            },
            _ => Cond {
                min: None,
                max: Some(a.max(b)),
            },
        }
    }
}

fn generate_query() -> OracleQuery {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut points = Vec::with_capacity(CASES);
    for i in 0..CASES {
        let n = 5 + rng.below(196);
        let integer = rng.chance(0.65);
        let x: Vec<f64> = (0..n)
            .map(|_| {
                if integer {
                    rng.below(7) as f64
                } else {
                    (rng.unit() - 0.5) * 8.0
                }
            })
            .collect();
        let (lo, hi) = x
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| {
                (l.min(v), h.max(v))
            });
        let span = (hi - lo).max(1.0);
        points.push(PeakCase {
            case_id: format!("fp_{i:03}_n{n}"),
            height: rng.chance(0.35).then(|| rng.cond(lo, hi)),
            threshold: rng.chance(0.3).then(|| rng.cond(0.0, span / 2.0)),
            distance: rng.chance(0.35).then(|| 1 + rng.below(8)),
            prominence: rng.chance(0.4).then(|| rng.cond(0.0, span)),
            width: rng.chance(0.4).then(|| rng.cond(0.0, 6.0)),
            wlen: rng.chance(0.3).then(|| 1.5 + 30.0 * rng.unit()),
            rel_height: if rng.chance(0.4) { rng.unit() } else { 0.5 },
            plateau_size: rng.chance(0.25).then(|| rng.cond(1.0, 4.0)),
            x,
        });
    }
    OracleQuery { points }
}

/// `portable` switches numpy's SIMD dispatch off (see the module comment), so the oracle takes
/// numpy's portable `argsort`.
fn scipy_oracle_or_skip(query: &OracleQuery, portable: bool) -> Option<OracleResult> {
    let script = r#"
import json
import sys
import numpy as np
from scipy.signal import find_peaks

def cond(c):
    if c is None:
        return None
    if c["max"] is None:
        return c["min"]
    return (c["min"], c["max"])

q = json.load(sys.stdin)
points = []
for case in q["points"]:
    cid = case["case_id"]
    try:
        peaks, props = find_peaks(
            np.asarray(case["x"], dtype=float),
            height=cond(case["height"]),
            threshold=cond(case["threshold"]),
            distance=case["distance"],
            prominence=cond(case["prominence"]),
            width=cond(case["width"]),
            wlen=case["wlen"],
            rel_height=case["rel_height"],
            plateau_size=cond(case["plateau_size"]),
        )
        points.append({
            "case_id": cid,
            "error": None,
            "peaks": [int(p) for p in peaks],
            "props": {k: [float(v) for v in np.asarray(a).tolist()] for k, a in props.items()},
        })
    except Exception as e:
        points.append({"case_id": cid, "error": repr(e), "peaks": None, "props": None})
# Canary: does this process's default argsort break ties in index order (the portable path)?
witness = np.array([3, 6, 4, 5, 4, 6, 5, 6, 3, 5, 5, 5, 6, 4], dtype=float)
stable = np.argsort(witness).tolist() == np.argsort(witness, kind="stable").tolist()
print(json.dumps({"points": points, "argsort_is_stable": stable}))
"#;
    // The query goes over stdin: 500 cases of JSON exceed Linux's 128 KiB limit on a single
    // environment string, and the spawn failed with E2BIG when it was passed as one.
    let query_json = serde_json::to_string(query).expect("serialize find_peaks query");
    let mut command = fsci_conformance::scipy_oracle_command();
    if portable {
        command.env("NPY_DISABLE_CPU_FEATURES", NUMPY_DISPATCH_GROUPS);
    }
    let mut child = match command
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
                "failed to spawn python3 for find_peaks oracle: {e}"
            );
            eprintln!("skipping find_peaks oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child.stdin.as_mut().expect("open find_peaks oracle stdin");
        if let Err(err) = stdin.write_all(query_json.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "find_peaks oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping find_peaks oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for find_peaks oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "find_peaks oracle failed: {stderr}"
        );
        eprintln!("skipping find_peaks oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse find_peaks oracle JSON"))
}

fn options_for(case: &PeakCase) -> FindPeaksOptions {
    FindPeaksOptions {
        height: case.height.map(Into::into),
        threshold: case.threshold.map(Into::into),
        distance: case.distance,
        prominence: case.prominence.map(Into::into),
        width: case.width.map(Into::into),
        wlen: case.wlen,
        rel_height: case.rel_height,
        plateau_size: case.plateau_size.map(Into::into),
    }
}

/// Every property fsci returned, as f64 columns keyed by SciPy's names.
fn fsci_props(r: &FindPeaksResult) -> HashMap<String, Vec<f64>> {
    let mut out = HashMap::new();
    let mut ints = |name: &str, v: &Option<Vec<usize>>| {
        if let Some(v) = v {
            out.insert(name.to_string(), v.iter().map(|&u| u as f64).collect());
        }
    };
    ints("plateau_sizes", &r.plateau_sizes);
    ints("left_edges", &r.left_edges);
    ints("right_edges", &r.right_edges);
    ints("left_bases", &r.left_bases);
    ints("right_bases", &r.right_bases);
    for (name, v) in [
        ("peak_heights", &r.peak_heights),
        ("left_thresholds", &r.left_thresholds),
        ("right_thresholds", &r.right_thresholds),
        ("prominences", &r.prominences),
        ("widths", &r.widths),
        ("width_heights", &r.width_heights),
        ("left_ips", &r.left_ips),
        ("right_ips", &r.right_ips),
    ] {
        if let Some(v) = v {
            out.insert(name.to_string(), v.clone());
        }
    }
    out
}

/// Compare one case; returns (max scaled difference, mismatch description or empty).
fn compare(
    peaks: &[usize],
    props: &HashMap<String, Vec<f64>>,
    ours: &FindPeaksResult,
) -> (f64, String) {
    if ours.peaks != peaks {
        return (
            f64::INFINITY,
            format!("peaks: fsci {:?} scipy {:?}", ours.peaks, peaks),
        );
    }
    let mine = fsci_props(ours);
    let mut theirs: Vec<&String> = props.keys().collect();
    theirs.sort();
    let mut mine_keys: Vec<&String> = mine.keys().collect();
    mine_keys.sort();
    if theirs != mine_keys {
        return (
            f64::INFINITY,
            format!("property sets: fsci {mine_keys:?} scipy {theirs:?}"),
        );
    }
    let mut worst = 0.0_f64;
    for key in theirs {
        let (a, b) = (&mine[key], &props[key]);
        if a.len() != b.len() {
            return (
                f64::INFINITY,
                format!("{key}: lengths {} vs {}", a.len(), b.len()),
            );
        }
        for (i, (&p, &q)) in a.iter().zip(b).enumerate() {
            let scale = q.abs().max(1.0);
            let d = (p - q).abs() / scale;
            if d.is_nan() || d > REL_TOL.max(ABS_TOL) {
                return (d, format!("{key}[{i}]: fsci {p} scipy {q}"));
            }
            worst = worst.max(d);
        }
    }
    (worst, String::new())
}

/// fsci's `find_peaks` on `case` against one SciPy arm: (max scaled difference, mismatch
/// description, empty or "both reject: …" when it passes).
fn judge(arm: &PeakArm, case: &PeakCase) -> (f64, String) {
    match (&arm.error, find_peaks(&case.x, options_for(case))) {
        (Some(err), Err(_)) => (0.0, format!("both reject: {err}")),
        (Some(err), Ok(r)) => (
            f64::INFINITY,
            format!("scipy raised {err}; fsci returned peaks {:?}", r.peaks),
        ),
        (None, Err(e)) => (f64::INFINITY, format!("fsci error {e}; scipy succeeded")),
        (None, Ok(r)) => {
            let peaks = arm.peaks.as_deref().expect("peaks when no error");
            let props = arm.props.as_ref().expect("props when no error");
            compare(peaks, props, &r)
        }
    }
}

fn passes(detail: &str) -> bool {
    detail.is_empty() || detail.starts_with("both reject")
}

#[test]
fn diff_signal_find_peaks_full() {
    let query = generate_query();
    let Some(default) = scipy_oracle_or_skip(&query, false) else {
        return;
    };
    let Some(portable) = scipy_oracle_or_skip(&query, true) else {
        return;
    };
    assert!(
        portable.argsort_is_stable,
        "the portable oracle arm still took numpy's SIMD argsort: \
         NPY_DISABLE_CPU_FEATURES=\"{NUMPY_DISPATCH_GROUPS}\" was not honoured"
    );
    assert_eq!(default.points.len(), query.points.len());
    assert_eq!(portable.points.len(), query.points.len());
    let default_argsort_is_stable = default.argsort_is_stable;
    let by_case = |oracle: OracleResult| -> HashMap<String, PeakArm> {
        oracle
            .points
            .into_iter()
            .map(|a| (a.case_id.clone(), a))
            .collect()
    };
    let default_arms = by_case(default);
    let portable_arms = by_case(portable);

    let start = Instant::now();
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;
    let (mut agree_both, mut agree_default_only, mut agree_portable_only) = (0, 0, 0);
    let mut isa_sensitive = 0;
    for case in &query.points {
        let default_arm = default_arms.get(&case.case_id).expect("validated oracle");
        let portable_arm = portable_arms.get(&case.case_id).expect("validated oracle");
        if default_arm.peaks != portable_arm.peaks || default_arm.error != portable_arm.error {
            isa_sensitive += 1;
        }
        let (default_diff, default_detail) = judge(default_arm, case);
        let (portable_diff, portable_detail) = judge(portable_arm, case);
        let (default_ok, portable_ok) = (passes(&default_detail), passes(&portable_detail));
        match (default_ok, portable_ok) {
            (true, true) => agree_both += 1,
            (true, false) => agree_default_only += 1,
            (false, true) => agree_portable_only += 1,
            (false, false) => {}
        }
        let pass = default_ok || portable_ok;
        let (max_diff, detail) = if default_ok {
            (default_diff, default_detail)
        } else if portable_ok {
            (portable_diff, portable_detail)
        } else {
            (
                default_diff,
                format!("default dispatch: {default_detail}; portable: {portable_detail}"),
            )
        };
        if !pass {
            eprintln!(
                "find_peaks mismatch {}: {detail}\n  x={:?}\n  case={case:?}",
                case.case_id, case.x
            );
        }
        max_overall = max_overall.max(max_diff);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            max_diff,
            pass,
            detail,
        });
    }

    let compared = diffs.len();
    let failures = diffs.iter().filter(|d| !d.pass).count();
    let with_width = query.points.iter().filter(|c| c.width.is_some()).count();
    let with_threshold = query
        .points
        .iter()
        .filter(|c| c.threshold.is_some())
        .count();
    println!(
        "find_peaks full: {compared} cases compared ({with_width} with width, {with_threshold} with threshold), {failures} failures, max scaled diff {max_overall:e}"
    );
    println!(
        "find_peaks tie order: SciPy's two numpy dispatches disagree on {isa_sensitive} cases \
         (default argsort stable: {}); fsci agrees with both on {agree_both}, with the default \
         dispatch only on {agree_default_only}, with the portable path only on \
         {agree_portable_only}",
        default_argsort_is_stable
    );
    emit_log(&DiffLog {
        test_id: "diff_signal_find_peaks_full".into(),
        category: "scipy.signal.find_peaks".into(),
        case_count: query.points.len(),
        compared,
        max_diff: max_overall,
        pass: failures == 0,
        isa_sensitive,
        agree_both,
        agree_default_only,
        agree_portable_only,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs,
    });

    assert_eq!(
        compared, CASES,
        "find_peaks: compared {compared} of {CASES} cases"
    );
    assert!(
        with_width > 100 && with_threshold > 100,
        "generator must exercise every filter"
    );
    assert_eq!(
        failures, 0,
        "find_peaks: {failures} of {compared} cases disagree with SciPy"
    );
}
