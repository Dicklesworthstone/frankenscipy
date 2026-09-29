#![forbid(unsafe_code)]
//! Cover fsci_signal::iirfilter dispatch across all 5 IIR families.
//!
//! Resolves [frankenscipy-he0h9]. iirfilter is a dispatch wrapper
//! that routes to butter / cheby1 / cheby2 / bessel / ellip based on
//! the IirFamily enum, validating that the required ripple/attenuation
//! parameters are supplied. Verifies:
//!   * Each family produces a BaCoeffs equal to its direct designer
//!   * Cheby1/Elliptic without rp → error
//!   * Cheby2/Elliptic without rs → error

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_signal::{
    BaCoeffs, FilterType, IirFamily, bessel, butter, cheby1, cheby2, ellip, iirfilter,
};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-14;

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    pass: bool,
    note: String,
}

#[derive(Debug, Clone, Serialize)]
struct DiffLog {
    test_id: String,
    category: String,
    case_count: usize,
    compared: BTreeMap<String, ArmCounts>,
    pass: bool,
    timestamp_ms: u128,
    duration_ns: u128,
    cases: Vec<CaseDiff>,
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/artifacts/{PACKET_ID}/diff"))
}

fn ensure_output_dir() {
    fs::create_dir_all(output_dir()).expect("create iirfilter diff dir");
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

fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// b then a in one vector, so a NaN or a length mismatch goes through the ledger's slice check
/// instead of vanishing in `max_abs`'s fold; the caller checks the b/a split.
fn packed(c: Option<&BaCoeffs>) -> Option<Vec<f64>> {
    c.map(|c| {
        let mut v = c.b.clone();
        v.extend(c.a.iter().copied());
        v
    })
}

#[test]
fn diff_signal_iirfilter_dispatch() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut check = |id: &str, ok: bool, note: String| {
        diffs.push(CaseDiff {
            case_id: id.into(),
            pass: ok,
            note,
        });
    };

    let order = 3;
    let wn = vec![0.3_f64];
    let btype = FilterType::Lowpass;
    let rp = 0.5;
    let rs = 40.0;
    let mut ledger = CompareLedger::new(
        "diff_signal_iirfilter_dispatch",
        &["dispatch", "missing_param"],
    );

    // Each family's dispatched design against its direct designer, which is the reference side:
    // a direct designer that fails is recorded as a missing reference.
    let families = [
        (
            "butterworth_matches_direct",
            IirFamily::Butterworth,
            None,
            None,
            butter(order, &wn, btype).ok(),
        ),
        (
            "chebyshev1_matches_direct",
            IirFamily::Chebyshev1,
            Some(rp),
            None,
            cheby1(order, rp, &wn, btype).ok(),
        ),
        (
            "chebyshev2_matches_direct",
            IirFamily::Chebyshev2,
            None,
            Some(rs),
            cheby2(order, rs, &wn, btype).ok(),
        ),
        (
            "bessel_matches_direct",
            IirFamily::Bessel,
            None,
            None,
            bessel(order, &wn, btype).ok(),
        ),
        (
            "elliptic_matches_direct",
            IirFamily::Elliptic,
            Some(rp),
            Some(rs),
            ellip(order, rp, rs, &wn, btype).ok(),
        ),
    ];
    let n_families = families.len();
    for (id, family, family_rp, family_rs, direct) in families {
        let dispatched = iirfilter(order, &wn, btype, family, family_rp, family_rs).ok();
        let (direct_v, dispatched_v) = (packed(direct.as_ref()), packed(dispatched.as_ref()));
        let Some((direct_v, dispatched_v)) =
            ledger.slices("dispatch", id, direct_v.as_deref(), dispatched_v.as_deref())
        else {
            continue;
        };
        let nb = direct.as_ref().map_or(0, |c| c.b.len());
        let (mab, maa) = if dispatched.as_ref().map(|c| c.b.len()) == Some(nb) {
            (
                max_abs(&direct_v[..nb], &dispatched_v[..nb]),
                max_abs(&direct_v[nb..], &dispatched_v[nb..]),
            )
        } else {
            (f64::INFINITY, f64::INFINITY)
        };
        let pass = mab <= ABS_TOL && maa <= ABS_TOL;
        ledger.compared("dispatch", id, pass);
        check(id, pass, format!("b_max={mab} a_max={maa}"));
    }

    // === Missing-rp errors for cheby1 and elliptic; missing-rs errors for cheby2 and elliptic ===
    // SciPy 1.17.1's iirfilter raises ValueError for each of these too.
    let missing = [
        (
            "chebyshev1_missing_rp_errors",
            IirFamily::Chebyshev1,
            None,
            None,
        ),
        (
            "elliptic_missing_rp_errors",
            IirFamily::Elliptic,
            None,
            Some(rs),
        ),
        (
            "chebyshev2_missing_rs_errors",
            IirFamily::Chebyshev2,
            None,
            None,
        ),
        (
            "elliptic_missing_rs_errors",
            IirFamily::Elliptic,
            Some(rp),
            None,
        ),
    ];
    let n_missing = missing.len();
    for (id, family, family_rp, family_rs) in missing {
        let r = iirfilter(order, &wn, btype, family, family_rp, family_rs);
        ledger.expected_raise("missing_param", id, r.is_err());
        check(id, r.is_err(), format!("res={r:?}"));
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_signal_iirfilter_dispatch".into(),
        category: "fsci_signal::iirfilter dispatch coverage".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        pass: all_pass,
        timestamp_ms: timestamp_ms(),
        duration_ns: start.elapsed().as_nanos(),
        cases: diffs.clone(),
    };
    emit_log(&log);

    for d in &diffs {
        if !d.pass {
            eprintln!("iirfilter mismatch: {} — {}", d.case_id, d.note);
        }
    }

    assert!(
        all_pass,
        "iirfilter dispatch coverage failed: {} cases",
        diffs.len()
    );
    ledger.finish(n_families.min(n_missing));
}
