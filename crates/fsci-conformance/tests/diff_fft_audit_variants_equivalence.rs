#![forbid(unsafe_code)]
//! Property test: fsci_fft audit variants (fft_with_audit,
//! ifft_with_audit, rfft_with_audit, irfft_with_audit, etc) must
//! produce numerically identical output to their non-audit
//! counterparts.
//!
//! Resolves [frankenscipy-z6stf]. Both code paths share the same
//! `*_impl` worker; the only difference is whether an audit ledger
//! is passed. Output should be bit-identical.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_fft::{
    Complex64, FftOptions, fft, fft_with_audit, ifft, ifft_with_audit, irfft, irfft_with_audit,
    rfft, rfft_with_audit, sync_audit_ledger,
};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const ABS_TOL: f64 = 1.0e-15;

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
    compared: BTreeMap<String, ArmCounts>,
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
    fs::create_dir_all(output_dir()).expect("create audit_equiv diff dir");
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

/// Complex output as [re0, im0, re1, im1, ...]: `real_max_diff` over it is the per-element
/// max(|d re|, |d im|) folded by max.
fn flatten(v: &[Complex64]) -> Vec<f64> {
    v.iter().flat_map(|&(re, im)| [re, im]).collect()
}

fn real_max_diff(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn diff_fft_audit_variants_equivalence() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut max_overall = 0.0_f64;
    let opts = FftOptions::default();
    let audit_ledger = sync_audit_ledger();
    let sizes = [16_usize, 32, 64, 128];
    // The reference side is fsci's own non-audit variant, not SciPy: a failed plain call is
    // recorded as a missing reference value, a failed audited call as an fsci failure.
    let mut ledger = CompareLedger::new(
        "diff_fft_audit_variants_equivalence",
        &["fft", "ifft", "rfft", "irfft"],
    );

    for &n in &sizes {
        // fft
        let signal: Vec<Complex64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (
                    (2.0 * std::f64::consts::PI * t).sin(),
                    (4.0 * std::f64::consts::PI * t).cos(),
                )
            })
            .collect();
        let plain = fft(&signal, &opts).ok();
        let audited = fft_with_audit(&signal, &opts, &audit_ledger).ok();

        // ifft (round trip through fft)
        let p_ifft = plain.as_ref().and_then(|x| ifft(x, &opts).ok());
        let a_ifft = plain
            .as_ref()
            .and_then(|x| ifft_with_audit(x, &opts, &audit_ledger).ok());

        // rfft
        let real_sig: Vec<f64> = signal.iter().map(|(re, _)| *re).collect();
        let p_rfft = rfft(&real_sig, &opts).ok();
        let a_rfft = rfft_with_audit(&real_sig, &opts, &audit_ledger).ok();

        // irfft
        let p_irfft = p_rfft.as_ref().and_then(|x| irfft(x, Some(n), &opts).ok());
        let a_irfft = p_rfft
            .as_ref()
            .and_then(|x| irfft_with_audit(x, Some(n), &opts, &audit_ledger).ok());

        let arms = [
            (
                "fft",
                plain.as_deref().map(flatten),
                audited.as_deref().map(flatten),
            ),
            (
                "ifft",
                p_ifft.as_deref().map(flatten),
                a_ifft.as_deref().map(flatten),
            ),
            (
                "rfft",
                p_rfft.as_deref().map(flatten),
                a_rfft.as_deref().map(flatten),
            ),
            ("irfft", p_irfft, a_irfft),
        ];
        for (op, reference, with_audit) in arms {
            let case_id = format!("{op}_n{n}");
            let Some((r, a)) =
                ledger.slices(op, &case_id, reference.as_deref(), with_audit.as_deref())
            else {
                continue;
            };
            let d = real_max_diff(r, a);
            max_overall = max_overall.max(d);
            ledger.compared(op, &case_id, d <= ABS_TOL);
            diffs.push(CaseDiff {
                case_id,
                op: op.into(),
                abs_diff: d,
                pass: d <= ABS_TOL,
            });
        }
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_fft_audit_variants_equivalence".into(),
        category: "fsci_fft::*_with_audit equivalent to non-audit variants".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
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
        "audit_equiv conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    ledger.finish(sizes.len());
}
