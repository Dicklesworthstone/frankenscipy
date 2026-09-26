#![forbid(unsafe_code)]
//! Property test: fsci_fft 2D/N-D audit variants produce numerically
//! identical output to their non-audit counterparts.
//!
//! Resolves [frankenscipy-r28td]. Covers fft2/ifft2/fftn/ifftn,
//! rfft2/irfft2/rfftn/irfftn, hfft/ihfft. 1e-15 abs.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_fft::{
    Complex64, FftOptions, fft2, fft2_with_audit, fftn, fftn_with_audit, hfft, hfft_with_audit,
    ifft2, ifft2_with_audit, ifftn, ifftn_with_audit, ihfft, ihfft_with_audit, irfft2,
    irfft2_with_audit, irfftn, irfftn_with_audit, rfft2, rfft2_with_audit, rfftn, rfftn_with_audit,
    sync_audit_ledger,
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
    fs::create_dir_all(output_dir()).expect("create audit_nd diff dir");
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

const ARMS: [&str; 10] = [
    "fft2", "ifft2", "rfft2", "irfft2", "fftn", "ifftn", "rfftn", "irfftn", "hfft", "ihfft",
];

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
fn diff_fft_audit_variants_nd_equivalence() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut max_overall = 0.0_f64;
    let opts = FftOptions::default();
    let audit_ledger = sync_audit_ledger();
    let shapes_2d = [(8_usize, 8_usize), (16, 16), (8, 16)];
    let shapes_nd = [vec![4_usize, 4, 4], vec![8, 8, 4]];
    let hfft_sizes = [16_usize, 32, 64];
    // (op, case id, plain, audited). The reference side is fsci's own non-audit variant, not
    // SciPy: a failed plain call is recorded as a missing reference value, a failed audited call
    // as an fsci failure. An inverse runs on the plain forward result, so a failed plain forward
    // call leaves its inverse without a reference too.
    let mut probes: Vec<(&str, String, Option<Vec<f64>>, Option<Vec<f64>>)> = Vec::new();

    // 2D probes
    for &shape in &shapes_2d {
        let (r, c) = shape;
        let signal: Vec<Complex64> = (0..r * c)
            .map(|i| {
                let t = i as f64 / (r * c) as f64;
                (
                    (2.0 * std::f64::consts::PI * t).sin(),
                    (4.0 * std::f64::consts::PI * t).cos(),
                )
            })
            .collect();
        let real_sig: Vec<f64> = signal.iter().map(|(re, _)| *re).collect();

        let p = fft2(&signal, shape, &opts).ok();
        let a = fft2_with_audit(&signal, shape, &opts, &audit_ledger).ok();
        // ifft2 on the fft2 result
        let pi = p.as_ref().and_then(|x| ifft2(x, shape, &opts).ok());
        let ai = p
            .as_ref()
            .and_then(|x| ifft2_with_audit(x, shape, &opts, &audit_ledger).ok());
        probes.push((
            "fft2",
            format!("fft2_{r}x{c}"),
            p.as_deref().map(flatten),
            a.as_deref().map(flatten),
        ));
        probes.push((
            "ifft2",
            format!("ifft2_{r}x{c}"),
            pi.as_deref().map(flatten),
            ai.as_deref().map(flatten),
        ));

        // rfft2
        let p = rfft2(&real_sig, shape, &opts).ok();
        let a = rfft2_with_audit(&real_sig, shape, &opts, &audit_ledger).ok();
        let pi = p.as_ref().and_then(|x| irfft2(x, shape, &opts).ok());
        let ai = p
            .as_ref()
            .and_then(|x| irfft2_with_audit(x, shape, &opts, &audit_ledger).ok());
        probes.push((
            "rfft2",
            format!("rfft2_{r}x{c}"),
            p.as_deref().map(flatten),
            a.as_deref().map(flatten),
        ));
        probes.push(("irfft2", format!("irfft2_{r}x{c}"), pi, ai));
    }

    // N-D probes
    for shape in &shapes_nd {
        let n: usize = shape.iter().product();
        let signal: Vec<Complex64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (
                    (2.0 * std::f64::consts::PI * t).sin(),
                    (4.0 * std::f64::consts::PI * t).cos(),
                )
            })
            .collect();
        let real_sig: Vec<f64> = signal.iter().map(|(re, _)| *re).collect();
        let s = shape.as_slice();

        let p = fftn(&signal, s, &opts).ok();
        let a = fftn_with_audit(&signal, s, &opts, &audit_ledger).ok();
        let pi = p.as_ref().and_then(|x| ifftn(x, s, &opts).ok());
        let ai = p
            .as_ref()
            .and_then(|x| ifftn_with_audit(x, s, &opts, &audit_ledger).ok());
        probes.push((
            "fftn",
            format!("fftn_{shape:?}"),
            p.as_deref().map(flatten),
            a.as_deref().map(flatten),
        ));
        probes.push((
            "ifftn",
            format!("ifftn_{shape:?}"),
            pi.as_deref().map(flatten),
            ai.as_deref().map(flatten),
        ));

        let p = rfftn(&real_sig, s, &opts).ok();
        let a = rfftn_with_audit(&real_sig, s, &opts, &audit_ledger).ok();
        let pi = p.as_ref().and_then(|x| irfftn(x, s, &opts).ok());
        let ai = p
            .as_ref()
            .and_then(|x| irfftn_with_audit(x, s, &opts, &audit_ledger).ok());
        probes.push((
            "rfftn",
            format!("rfftn_{shape:?}"),
            p.as_deref().map(flatten),
            a.as_deref().map(flatten),
        ));
        probes.push(("irfftn", format!("irfftn_{shape:?}"), pi, ai));
    }

    // hfft / ihfft (1D)
    for &n in &hfft_sizes {
        let cmpx: Vec<Complex64> = (0..n / 2 + 1)
            .map(|i| (i as f64 * 0.3, i as f64 * 0.2))
            .collect();
        probes.push((
            "hfft",
            format!("hfft_n{n}"),
            hfft(&cmpx, Some(n), &opts).ok(),
            hfft_with_audit(&cmpx, Some(n), &opts, &audit_ledger).ok(),
        ));
        let real_sig: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let p = ihfft(&real_sig, Some(n), &opts).ok();
        let a = ihfft_with_audit(&real_sig, Some(n), &opts, &audit_ledger).ok();
        probes.push((
            "ihfft",
            format!("ihfft_n{n}"),
            p.as_deref().map(flatten),
            a.as_deref().map(flatten),
        ));
    }

    let mut ledger = CompareLedger::new("diff_fft_audit_variants_nd_equivalence", &ARMS);
    for (op, case_id, plain, audited) in probes {
        let Some((p, a)) = ledger.slices(op, &case_id, plain.as_deref(), audited.as_deref()) else {
            continue;
        };
        let d = real_max_diff(p, a);
        max_overall = max_overall.max(d);
        ledger.compared(op, &case_id, d <= ABS_TOL);
        diffs.push(CaseDiff {
            case_id,
            op: op.into(),
            abs_diff: d,
            pass: d <= ABS_TOL,
        });
    }

    let all_pass = diffs.iter().all(|d| d.pass);

    let log = DiffLog {
        test_id: "diff_fft_audit_variants_nd_equivalence".into(),
        category: "fsci_fft N-D audit variants equivalent to non-audit".into(),
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
        "audit_nd_equiv conformance failed: {} cases, max_diff={}",
        diffs.len(),
        max_overall
    );
    ledger.finish(shapes_2d.len().min(shapes_nd.len()).min(hfft_sizes.len()));
}
