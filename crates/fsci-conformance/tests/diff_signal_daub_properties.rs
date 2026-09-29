#![forbid(unsafe_code)]
//! Property-based coverage for fsci_signal::daub(p) wavelet filters.
//!
//! Resolves [frankenscipy-lwpup]. scipy.signal.daub was removed in
//! scipy >= 1.12 so we cannot do live oracle parity, but the
//! Daubechies wavelet coefficients must satisfy strict mathematical
//! invariants that are stronger than any oracle:
//!   * length(h) == 2p
//!   * sum(h) == sqrt(2)         (low-pass DC normalization)
//!   * sum(h_k²) == 1            (orthonormality)
//!   * sum_k h[k] · h[k+2m] == δ_m   (orthogonality at even shifts)
//!
//! These four invariants together pin down the filter up to a single
//! sign flip per p — strong enough to catch any coefficient corruption.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_signal::daub;
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
// The tabulated coefficients are quoted to ~16 significant digits and
// the sum / energy invariants drift by ~1e-11 for higher p (truncation
// error in the source table). 1e-10 is tight enough to catch any
// coefficient corruption and loose enough to accept the table itself.
const ABS_TOL: f64 = 1.0e-10;

#[derive(Debug, Clone, Serialize)]
struct CaseDiff {
    case_id: String,
    p: usize,
    length: usize,
    sum_minus_sqrt2: f64,
    energy_minus_1: f64,
    max_offdiag_corr: f64,
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
    fs::create_dir_all(output_dir()).expect("create daub diff dir");
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

/// Compute Σ_k h[k] · h[k+2m] for an integer shift m. Returns 0 if the
/// shifted index is out of range.
fn shifted_inner(h: &[f64], m: isize) -> f64 {
    let n = h.len() as isize;
    let mut s = 0.0_f64;
    for k in 0..n {
        let j = k + 2 * m;
        if (0..n).contains(&j) {
            s += h[k as usize] * h[j as usize];
        }
    }
    s
}

#[test]
fn diff_signal_daub_properties() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let sqrt2 = 2.0_f64.sqrt();
    // The reference side is analytic (length 2p, sum sqrt(2), energy 1, zero correlation at
    // even shifts), so it is always present; a daub error is an fsci failure.
    let mut ledger = CompareLedger::new("diff_signal_daub_properties", &["daub"]);
    let orders: Vec<usize> = (1..=10).collect();

    for &p in &orders {
        let case_id = format!("daub_p{p}");
        let result = daub(p);
        if let Err(e) = &result {
            diffs.push(CaseDiff {
                case_id: case_id.clone(),
                p,
                length: 0,
                sum_minus_sqrt2: f64::INFINITY,
                energy_minus_1: f64::INFINITY,
                max_offdiag_corr: f64::INFINITY,
                pass: false,
                note: format!("daub error: {e:?}"),
            });
        }
        let Some((expected_len, h)) = ledger.both("daub", &case_id, Some(2 * p), result.ok())
        else {
            continue;
        };

        let length_ok = h.len() == expected_len;
        let sum: f64 = h.iter().sum();
        let sum_minus_sqrt2 = (sum - sqrt2).abs();
        let energy: f64 = h.iter().map(|v| v * v).sum();
        let energy_minus_1 = (energy - 1.0).abs();

        // Orthogonality at even shifts m ∈ {1, .., p-1} (m=0 is energy).
        // Σ_k h[k] · h[k+2m] should be ~0 for all m != 0.
        let max_offdiag = (1..p as isize)
            .map(|m| shifted_inner(&h, m).abs())
            .fold(0.0_f64, f64::max);

        let pass = length_ok
            && sum_minus_sqrt2 <= ABS_TOL
            && energy_minus_1 <= ABS_TOL
            && max_offdiag <= ABS_TOL;
        ledger.compared("daub", &case_id, pass);

        diffs.push(CaseDiff {
            case_id,
            p,
            length: h.len(),
            sum_minus_sqrt2,
            energy_minus_1,
            max_offdiag_corr: max_offdiag,
            pass,
            note: if length_ok {
                String::new()
            } else {
                format!("expected length 2p={}", 2 * p)
            },
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_signal_daub_properties".into(),
        category: "fsci_signal::daub(p) property-based: length, sum, energy, orthogonality".into(),
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
            eprintln!(
                "daub property violation: p={} len={} sum_err={} energy_err={} offdiag={} note={}",
                d.p, d.length, d.sum_minus_sqrt2, d.energy_minus_1, d.max_offdiag_corr, d.note
            );
        }
    }

    assert!(
        all_pass,
        "daub property coverage failed: {} cases",
        diffs.len()
    );
    ledger.finish(orders.len());
}
