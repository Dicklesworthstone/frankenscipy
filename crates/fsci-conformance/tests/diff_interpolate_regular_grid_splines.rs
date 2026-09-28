#![forbid(unsafe_code)]
//! Live SciPy differential coverage for `RegularGridInterpolator`'s tensor-product methods:
//! `slinear`, `cubic`, `quintic` and `pchip`, on 1-D to 3-D grids, including queries outside
//! the grid (`bounds_error=False, fill_value=None`, so both sides extrapolate).
//!
//! fsci computes the exact interpolating spline. SciPy 1.17.1's default `slinear`, `cubic` and
//! `quintic` build the same spline with `make_ndbspl`, which solves the tensor collocation
//! system with the iterative `gcrotmk` at `rtol = 1e-5`, so those values carry that solver's
//! error (1.5e-4 for cubic and 1.4e-2 for quintic on an O(1) grid in the unit tests). SciPy's
//! `*_legacy` methods fit the exact spline one axis at a time. Each arm therefore compares
//! against `<method>_legacy` (`pchip` is exact in SciPy already) at `REL_TOL`. The distance
//! from SciPy's default to its own exact value is logged as `scipy_default_gap`.
//!
//! frankenscipy-b7891: before this file RGI cubic and quintic ran a local Catmull-Rom tensor
//! that agreed with SciPy only to ~1e-3, and no live test covered any of these methods.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_interpolate::{RegularGridInterpolator, RegularGridMethod};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-006";
/// Against SciPy's exact interpolant, relative to max(|expected|, 1).
const REL_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const METHODS: [&str; 4] = ["slinear", "cubic", "quintic", "pchip"];

#[derive(Debug, Clone, Serialize)]
struct GridCase {
    case_id: String,
    method: String,
    points: Vec<Vec<f64>>,
    /// Row-major values over the grid.
    values: Vec<f64>,
    xi: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
struct OracleArm {
    case_id: String,
    /// `<method>_legacy` (or `pchip`): the exact interpolant.
    exact: Option<Vec<f64>>,
    /// The default `<method>`, built with gcrotmk; equal to `exact` for pchip.
    default: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    method: String,
    max_rel_diff: f64,
    scipy_default_gap: f64,
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
    fs::create_dir_all(output_dir()).expect("create regular grid diff output dir");
    let path = output_dir().join(format!("{}.json", log.test_id));
    let json = serde_json::to_string_pretty(log).expect("serialize regular grid diff log");
    fs::write(path, json).expect("write regular grid diff log");
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1_u64 << 53) as f64
    }
}

/// A strictly increasing axis on [lo, hi] with `n` points and jittered spacing.
fn axis(rng: &mut Lcg, lo: f64, hi: f64, n: usize) -> Vec<f64> {
    let mut gaps: Vec<f64> = (0..n - 1).map(|_| 0.4 + rng.next()).collect();
    let total: f64 = gaps.iter().sum();
    for g in &mut gaps {
        *g *= (hi - lo) / total;
    }
    let mut out = Vec::with_capacity(n);
    let mut x = lo;
    out.push(x);
    for g in gaps {
        x += g;
        out.push(x);
    }
    out
}

fn grid_values(points: &[Vec<f64>], rough: bool, rng: &mut Lcg) -> Vec<f64> {
    let total: usize = points.iter().map(Vec::len).product();
    let mut values = Vec::with_capacity(total);
    let mut idx = vec![0_usize; points.len()];
    for _ in 0..total {
        let coords: Vec<f64> = idx.iter().zip(points).map(|(&i, ax)| ax[i]).collect();
        let smooth = coords
            .iter()
            .enumerate()
            .map(|(d, &c)| ((d as f64 + 1.0) * 0.7 * c).sin() + 0.2 * c * c)
            .sum::<f64>();
        values.push(if rough {
            10.0 * rng.next() - 5.0
        } else {
            smooth
        });
        for d in (0..idx.len()).rev() {
            idx[d] += 1;
            if idx[d] < points[d].len() {
                break;
            }
            idx[d] = 0;
        }
    }
    values
}

/// Interior points, every grid node of the first axis at a random place on the others, and
/// points up to 20% of each axis's span outside it.
fn queries(points: &[Vec<f64>], rng: &mut Lcg) -> Vec<Vec<f64>> {
    let mut out = Vec::new();
    for _ in 0..12 {
        out.push(
            points
                .iter()
                .map(|ax| ax[0] + rng.next() * (ax[ax.len() - 1] - ax[0]))
                .collect(),
        );
    }
    for &node in &points[0] {
        let mut q: Vec<f64> = points
            .iter()
            .map(|ax| ax[0] + rng.next() * (ax[ax.len() - 1] - ax[0]))
            .collect();
        q[0] = node;
        out.push(q);
    }
    for _ in 0..6 {
        out.push(
            points
                .iter()
                .map(|ax| {
                    let span = ax[ax.len() - 1] - ax[0];
                    ax[0] - 0.2 * span + rng.next() * 1.4 * span
                })
                .collect(),
        );
    }
    out
}

fn generate_cases() -> Vec<GridCase> {
    let mut rng = Lcg(0x5e_ed0f_9e1d);
    let shapes: [&[usize]; 4] = [&[11], &[8, 7], &[13, 9], &[6, 7, 6]];
    let mut cases = Vec::new();
    for (si, shape) in shapes.iter().enumerate() {
        for rough in [false, true] {
            let points: Vec<Vec<f64>> = shape
                .iter()
                .enumerate()
                .map(|(d, &n)| axis(&mut rng, -1.0 + d as f64, 2.0 + 0.5 * d as f64, n))
                .collect();
            let values = grid_values(&points, rough, &mut rng);
            let xi = queries(&points, &mut rng);
            for method in METHODS {
                let kind = if rough { "rough" } else { "smooth" };
                cases.push(GridCase {
                    case_id: format!("{method}_{}d_s{si}_{kind}", shape.len()),
                    method: method.to_string(),
                    points: points.clone(),
                    values: values.clone(),
                    xi: xi.clone(),
                });
            }
        }
    }
    cases
}

fn scipy_oracle_or_skip(cases: &[GridCase]) -> Option<Vec<OracleArm>> {
    let script = r#"
import json
import math
import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def finite_or_none(arr):
    out = [float(v) for v in np.asarray(arr).ravel()]
    return out if all(math.isfinite(v) for v in out) else None

def run(grid, vals, xi, method):
    try:
        rgi = RegularGridInterpolator(grid, vals, method=method, bounds_error=False,
                                      fill_value=None)
        return finite_or_none(rgi(xi))
    except Exception:
        return None

cases = json.load(sys.stdin)
out = []
for c in cases:
    grid = tuple(np.array(ax, dtype=float) for ax in c["points"])
    vals = np.array(c["values"], dtype=float).reshape(tuple(len(ax) for ax in grid))
    xi = np.array(c["xi"], dtype=float)
    m = c["method"]
    exact = run(grid, vals, xi, m if m == "pchip" else m + "_legacy")
    default = run(grid, vals, xi, m)
    out.append({"case_id": c["case_id"], "exact": exact, "default": default})
print(json.dumps(out))
"#;
    let payload = serde_json::to_string(cases).expect("serialize regular grid cases");
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
                "failed to spawn python3 for the regular grid oracle: {e}"
            );
            eprintln!("skipping regular grid oracle: python3 not available ({e})");
            return None;
        }
    };
    {
        let stdin = child
            .stdin
            .as_mut()
            .expect("open regular grid oracle stdin");
        if let Err(err) = stdin.write_all(payload.as_bytes()) {
            let output = child.wait_with_output().expect("wait for failed oracle");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "regular grid oracle stdin write failed: {err}; stderr: {stderr}"
            );
            eprintln!("skipping regular grid oracle: stdin write failed ({err})\n{stderr}");
            return None;
        }
    }
    let output = child
        .wait_with_output()
        .expect("wait for regular grid oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "regular grid oracle failed: {stderr}"
        );
        eprintln!("skipping regular grid oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse regular grid oracle JSON"))
}

fn method_of(name: &str) -> RegularGridMethod {
    match name {
        "slinear" => RegularGridMethod::Slinear,
        "cubic" => RegularGridMethod::Cubic,
        "quintic" => RegularGridMethod::Quintic,
        "pchip" => RegularGridMethod::Pchip,
        other => panic!("unknown method {other}"),
    }
}

/// `|a - b| / max(|b|, 1)`, infinite when exactly one side is NaN.
fn rel_diff(a: f64, b: f64) -> f64 {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => 0.0,
        (false, false) => (a - b).abs() / b.abs().max(1.0),
        _ => f64::INFINITY,
    }
}

#[test]
fn diff_interpolate_regular_grid_splines() {
    let cases = generate_cases();
    let Some(oracle) = scipy_oracle_or_skip(&cases) else {
        return;
    };
    assert_eq!(
        oracle.len(),
        cases.len(),
        "oracle returned partial coverage"
    );
    let by_id: HashMap<String, OracleArm> = oracle
        .into_iter()
        .map(|arm| (arm.case_id.clone(), arm))
        .collect();

    let start = Instant::now();
    let mut ledger = CompareLedger::new("diff_interpolate_regular_grid_splines", &METHODS);
    let mut diffs = Vec::new();
    let mut max_overall = 0.0_f64;

    for case in &cases {
        let arm = by_id.get(&case.case_id).expect("validated oracle map");
        let fsci = RegularGridInterpolator::new(
            case.points.clone(),
            case.values.clone(),
            method_of(&case.method),
            false,
            None,
        )
        .and_then(|rgi| rgi.eval_many(&case.xi))
        .ok();
        let Some((exact, fsci)) = ledger.slices(
            &case.method,
            &case.case_id,
            arm.exact.as_deref(),
            fsci.as_deref(),
        ) else {
            continue;
        };
        let max_rel = fsci
            .iter()
            .zip(exact)
            .fold(0.0_f64, |acc, (&f, &e)| acc.max(rel_diff(f, e)));
        let scipy_default_gap = arm.default.as_deref().map_or(f64::NAN, |default| {
            default
                .iter()
                .zip(exact)
                .fold(0.0_f64, |acc, (&d, &e)| acc.max(rel_diff(d, e)))
        });
        let pass = max_rel <= REL_TOL;
        max_overall = max_overall.max(max_rel);
        ledger.compared(&case.method, &case.case_id, pass);
        diffs.push(CaseDiff {
            case_id: case.case_id.clone(),
            method: case.method.clone(),
            max_rel_diff: max_rel,
            scipy_default_gap,
            pass,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    emit_log(&DiffLog {
        test_id: "diff_interpolate_regular_grid_splines".into(),
        category: "scipy.interpolate.RegularGridInterpolator (slinear, cubic, quintic, pchip)"
            .into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        max_rel_diff: max_overall,
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    });
    for d in &diffs {
        eprintln!(
            "{} {}: max_rel_diff={:e} scipy_default_gap={:e}{}",
            d.method,
            d.case_id,
            d.max_rel_diff,
            d.scipy_default_gap,
            if d.pass { "" } else { "  MISMATCH" }
        );
    }
    assert!(
        all_pass,
        "RegularGridInterpolator spline methods diverged from SciPy's exact interpolant: max {max_overall:e}"
    );
    let min_per_arm = METHODS
        .iter()
        .map(|m| cases.iter().filter(|c| c.method == *m).count())
        .min()
        .expect("METHODS is non-empty");
    ledger.finish(min_per_arm);
}
