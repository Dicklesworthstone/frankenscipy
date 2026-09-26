#![forbid(unsafe_code)]
//! Live SciPy differential coverage for discrete distributions on the signed integers through
//! the generic `DiscreteDistribution` trait (frankenscipy-szq1n.8): `skellam(2, 3)`,
//! `dlaplace(0.8)` and `randint(-3, 3)` — pmf / cdf / sf at k = −5..=5, ppf at
//! q ∈ {0.01, 0.25, 0.5, 0.75, 0.99}, mean, var and support. The trait used to be `u64`-only, so
//! negative k was unreachable and `rvs` cast −1 to 18446744073709551615.
//!
//! Values must agree to `REL_TOL` (relative to max(|scipy|, 1e-300)); quantiles and supports
//! exactly. Every row must be compared: a SciPy failure is a FAILED row, not a skipped one
//! (frankenscipy-olv0j.1).

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::process::Stdio;

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_stats::{DiscreteDistribution, DiscreteLaplace, RandInt, Skellam};
use serde::{Deserialize, Serialize};

const PACKET_ID: &str = "FSCI-P2C-007";
const REL_TOL: f64 = 1.0e-12;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";
const KS: [i64; 11] = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5];
const QS: [f64; 5] = [0.01, 0.25, 0.5, 0.75, 0.99];

#[derive(Debug, Clone, Deserialize)]
struct OracleFamily {
    name: String,
    pmf: Vec<f64>,
    cdf: Vec<f64>,
    sf: Vec<f64>,
    ppf: Vec<f64>,
    mean: f64,
    var: f64,
    /// Python's `str(float(end))`: "-inf", "inf" or a finite value.
    support: (String, String),
}

#[derive(Debug, Clone, Serialize)]
struct Row {
    family: String,
    quantity: String,
    at: f64,
    fsci: f64,
    scipy: f64,
    pass: bool,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    compared: BTreeMap<String, ArmCounts>,
    rows: Vec<Row>,
}

const ARMS: [&str; 8] = [
    "pmf",
    "cdf",
    "sf",
    "ppf",
    "mean",
    "var",
    "support_lo",
    "support_hi",
];

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn scipy_oracle_or_skip() -> Option<Vec<OracleFamily>> {
    let script = r#"
import json, math
from scipy import stats
KS = list(range(-5, 6))
QS = [0.01, 0.25, 0.5, 0.75, 0.99]
fams = [("skellam", stats.skellam(2, 3)), ("dlaplace", stats.dlaplace(0.8)),
        ("randint", stats.randint(-3, 3))]
out = []
for name, d in fams:
    lo, hi = d.support()
    out.append({"name": name,
                "pmf": [float(d.pmf(k)) for k in KS],
                "cdf": [float(d.cdf(k)) for k in KS],
                "sf": [float(d.sf(k)) for k in KS],
                "ppf": [float(d.ppf(q)) for q in QS],
                "mean": float(d.mean()), "var": float(d.var()),
                # JSON has no infinity: the (possibly infinite) ends travel as "-inf" / "inf".
                "support": [str(float(lo)), str(float(hi))]})
print(json.dumps(out, allow_nan=False))
"#;
    let child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "failed to spawn python3 for the signed-support oracle: {e}"
            );
            eprintln!("skipping signed-support oracle: python3 not available ({e})");
            return None;
        }
    };
    let output = child
        .wait_with_output()
        .expect("wait for signed-support oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "signed-support oracle failed: {stderr}"
        );
        eprintln!("skipping signed-support oracle: scipy not available\n{stderr}");
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Some(serde_json::from_str(&stdout).expect("parse signed-support oracle JSON"))
}

fn check(
    ledger: &mut CompareLedger,
    rows: &mut Vec<Row>,
    family: &str,
    quantity: &str,
    at: f64,
    fsci: f64,
    scipy: f64,
) {
    let case_id = if at.is_nan() {
        family.to_string()
    } else {
        format!("{family}@{at}")
    };
    let exact = matches!(quantity, "ppf" | "support_lo" | "support_hi");
    // A support end is infinite on both sides for skellam and dlaplace: `both` keeps that row,
    // and the exact equality below still fails a NaN on either side. Every other value SciPy
    // sends is finite (`allow_nan=False`), so `pair` hands back both or records why not.
    let values = if exact {
        ledger.both(quantity, &case_id, Some(scipy), Some(fsci))
    } else {
        ledger.pair(quantity, &case_id, Some(scipy), Some(fsci))
    };
    let Some((scipy, fsci)) = values else {
        return;
    };
    let pass = if exact {
        fsci == scipy
    } else {
        (fsci - scipy).abs() <= REL_TOL * scipy.abs().max(1e-300)
    };
    ledger.compared(quantity, &case_id, pass);
    rows.push(Row {
        family: family.to_string(),
        quantity: quantity.to_string(),
        at,
        fsci,
        scipy,
        pass,
    });
}

fn compare<D: DiscreteDistribution>(
    ledger: &mut CompareLedger,
    rows: &mut Vec<Row>,
    d: &D,
    oracle: &OracleFamily,
) {
    let name = oracle.name.as_str();
    for (i, &k) in KS.iter().enumerate() {
        check(ledger, rows, name, "pmf", k as f64, d.pmf(k), oracle.pmf[i]);
        check(ledger, rows, name, "cdf", k as f64, d.cdf(k), oracle.cdf[i]);
        check(ledger, rows, name, "sf", k as f64, d.sf(k), oracle.sf[i]);
    }
    for (i, &q) in QS.iter().enumerate() {
        check(ledger, rows, name, "ppf", q, d.ppf(q), oracle.ppf[i]);
    }
    check(ledger, rows, name, "mean", f64::NAN, d.mean(), oracle.mean);
    check(ledger, rows, name, "var", f64::NAN, d.var(), oracle.var);
    let (lo, hi) = d.support();
    // An unparseable end becomes NaN, which fails the exact comparison.
    let end = |s: &str| s.parse::<f64>().unwrap_or(f64::NAN);
    check(
        ledger,
        rows,
        name,
        "support_lo",
        f64::NAN,
        lo,
        end(&oracle.support.0),
    );
    check(
        ledger,
        rows,
        name,
        "support_hi",
        f64::NAN,
        hi,
        end(&oracle.support.1),
    );
}

#[test]
fn diff_stats_discrete_signed_support() {
    let Some(oracle) = scipy_oracle_or_skip() else {
        return;
    };
    assert_eq!(oracle.len(), 3, "the oracle must return every family");
    let mut rows = Vec::new();
    let mut ledger = CompareLedger::new("diff_stats_discrete_signed_support", &ARMS);
    for family in &oracle {
        match family.name.as_str() {
            "skellam" => compare(&mut ledger, &mut rows, &Skellam::new(2.0, 3.0), family),
            "dlaplace" => compare(&mut ledger, &mut rows, &DiscreteLaplace::new(0.8), family),
            "randint" => compare(&mut ledger, &mut rows, &RandInt::new(-3, 3), family),
            other => {
                unreachable!("the oracle script emits only skellam, dlaplace, randint: {other}")
            }
        }
    }

    let dir = output_dir();
    fs::create_dir_all(&dir).expect("create diff output dir");
    let log = DiffLog {
        compared: ledger.counts().clone(),
        rows: rows.clone(),
    };
    fs::write(
        dir.join("diff_stats_discrete_signed_support.json"),
        serde_json::to_string_pretty(&log).expect("serialize rows"),
    )
    .expect("write diff log");
    for r in rows.iter().filter(|r| !r.pass) {
        println!(
            "{} {}({}) fsci={:e} scipy={:e}",
            r.family, r.quantity, r.at, r.fsci, r.scipy
        );
    }
    // 3 families × (3·11 point values + 5 quantiles + mean + var + 2 support ends).
    assert_eq!(rows.len(), 3 * (33 + 5 + 4), "every row must be compared");
    let failed = rows.iter().filter(|r| !r.pass).count();
    assert_eq!(
        failed,
        0,
        "{failed} of {} rows disagree with SciPy",
        rows.len()
    );
    // mean, var and the support ends have one case per family, the fewest of any arm.
    ledger.finish(oracle.len());
}
