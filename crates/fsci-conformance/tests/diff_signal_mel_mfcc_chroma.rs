#![forbid(unsafe_code)]
//! Property-based coverage for fsci_signal audio features.
//!
//! Resolves [frankenscipy-b97q1]. Covers:
//!   * hz_to_mel + mel_to_hz round-trip (Hz → Mel → Hz)
//!   * mel_filterbank shape (n_mels rows × n_freq cols) and bounded
//!     [0, 1] triangular weights
//!   * mfcc output shape (n_frames × n_mfcc)
//!   * chroma fixed length 12 (one per pitch class)
//!   * spectral_contrast output length n_bands

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_signal::{chroma, hz_to_mel, mel_filterbank, mel_to_hz, mfcc, spectral_contrast};
use serde::Serialize;

const PACKET_ID: &str = "FSCI-P2C-007";
const REL_TOL: f64 = 1.0e-10;
/// One ledger arm per property. `hz_mel_roundtrip` compares the round-tripped value against the
/// input Hz (one case per input); every other arm is one fixed structural case recorded under its
/// own name, whose reference is the analytic shape or range stated in its check.
const ARMS: [&str; 11] = [
    "hz_mel_roundtrip",
    "mel_filterbank_shape",
    "mel_filterbank_weights_in_unit_interval",
    "mel_filterbank_each_filter_nonzero",
    "mfcc_frame_count",
    "mfcc_per_frame_dim_eq_n_mfcc",
    "mfcc_empty_signal_returns_empty",
    "chroma_length_12",
    "chroma_values_finite",
    "spectral_contrast_length_eq_n_bands",
    "spectral_contrast_values_finite",
];

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
    fs::create_dir_all(output_dir()).expect("create mel_mfcc diff dir");
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

#[test]
fn diff_signal_mel_mfcc_chroma() {
    let start = Instant::now();
    let mut diffs: Vec<CaseDiff> = Vec::new();
    let mut ledger = CompareLedger::new("diff_signal_mel_mfcc_chroma", &ARMS);
    let mut check = |id: &str, ok: bool, note: String| {
        diffs.push(CaseDiff {
            case_id: id.into(),
            pass: ok,
            note,
        });
    };

    // === Hz ↔ Mel round-trip ===
    let roundtrip_hz = [0.0_f64, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 20000.0];
    for &hz in &roundtrip_hz {
        let mel = hz_to_mel(hz);
        let hz_back = mel_to_hz(mel);
        let rel = ((hz_back - hz).abs()) / hz.abs().max(1.0e-6);
        let case_id = format!("hz_mel_roundtrip_{hz}");
        // The reference is the input Hz, always finite; pair records a non-finite hz_back as an
        // fsci failure instead of letting `rel <= REL_TOL` be the only guard.
        if ledger
            .pair("hz_mel_roundtrip", &case_id, Some(hz), Some(hz_back))
            .is_some()
        {
            ledger.compared("hz_mel_roundtrip", &case_id, rel <= REL_TOL);
        }
        check(
            &case_id,
            rel <= REL_TOL,
            format!("hz={hz} mel={mel} hz_back={hz_back}"),
        );
    }

    // === mel_filterbank shape ===
    {
        let n_mels = 13;
        let n_fft = 512;
        let sr = 16000.0;
        let fb = mel_filterbank(n_mels, n_fft, sr, 0.0, sr / 2.0);
        let n_freq = n_fft / 2 + 1;
        let shape_ok = fb.len() == n_mels && fb.iter().all(|r| r.len() == n_freq);
        ledger.compared("mel_filterbank_shape", "mel_filterbank_shape", shape_ok);
        check(
            "mel_filterbank_shape",
            shape_ok,
            format!(
                "rows={} cols0={}",
                fb.len(),
                fb.first().map_or(0, |r| r.len())
            ),
        );
        // All weights in [0, 1] (a NaN weight is outside the range, so it fails here)
        let in_range = fb
            .iter()
            .flat_map(|r| r.iter())
            .all(|&v| (0.0..=1.0).contains(&v));
        ledger.compared(
            "mel_filterbank_weights_in_unit_interval",
            "mel_filterbank_weights_in_unit_interval",
            in_range,
        );
        check(
            "mel_filterbank_weights_in_unit_interval",
            in_range,
            String::new(),
        );
        // Each filter has at least one non-zero entry (peaks at center)
        let all_have_peak = fb.iter().all(|r| r.iter().any(|&v| v > 0.0));
        ledger.compared(
            "mel_filterbank_each_filter_nonzero",
            "mel_filterbank_each_filter_nonzero",
            all_have_peak,
        );
        check(
            "mel_filterbank_each_filter_nonzero",
            all_have_peak,
            String::new(),
        );
    }

    // === mfcc output shape ===
    {
        let sr = 16000.0;
        let n_mfcc = 13;
        let n_mels = 26;
        let frame_len = 512;
        let hop_len = 256;
        // 2 seconds of sinusoidal signal
        let n_samples = sr as usize * 2;
        let signal: Vec<f64> = (0..n_samples)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / sr).sin())
            .collect();
        let result = mfcc(&signal, sr, n_mfcc, n_mels, frame_len, hop_len);
        // n_frames = floor((n_samples - frame_len) / hop_len) + 1
        let expected_frames = (n_samples - frame_len) / hop_len + 1;
        ledger.compared(
            "mfcc_frame_count",
            "mfcc_frame_count",
            result.len() == expected_frames,
        );
        check(
            "mfcc_frame_count",
            result.len() == expected_frames,
            format!("got {} expected {}", result.len(), expected_frames),
        );
        let mfcc_dim_ok = result.iter().all(|frame| frame.len() == n_mfcc);
        ledger.compared(
            "mfcc_per_frame_dim_eq_n_mfcc",
            "mfcc_per_frame_dim_eq_n_mfcc",
            mfcc_dim_ok,
        );
        check(
            "mfcc_per_frame_dim_eq_n_mfcc",
            mfcc_dim_ok,
            format!("first_len={}", result.first().map_or(0, |f| f.len())),
        );
    }

    // === mfcc on empty signal returns empty ===
    {
        let result = mfcc(&[], 16000.0, 13, 26, 512, 256);
        ledger.compared(
            "mfcc_empty_signal_returns_empty",
            "mfcc_empty_signal_returns_empty",
            result.is_empty(),
        );
        check(
            "mfcc_empty_signal_returns_empty",
            result.is_empty(),
            String::new(),
        );
    }

    // === chroma: always 12 ===
    {
        let n_fft = 512;
        let sr = 16000.0;
        let mags: Vec<f64> = (0..n_fft / 2 + 1).map(|i| (i as f64).sin().abs()).collect();
        let result = chroma(&mags, sr, n_fft);
        ledger.compared("chroma_length_12", "chroma_length_12", result.len() == 12);
        check(
            "chroma_length_12",
            result.len() == 12,
            format!("got {}", result.len()),
        );
        let all_finite = result.iter().all(|v| v.is_finite());
        ledger.compared("chroma_values_finite", "chroma_values_finite", all_finite);
        check("chroma_values_finite", all_finite, String::new());
    }

    // === spectral_contrast: output length == n_bands ===
    {
        let n_bands = 6;
        let mags: Vec<f64> = (0..257).map(|i| (i as f64 + 1.0).sqrt()).collect();
        let result = spectral_contrast(&mags, n_bands);
        ledger.compared(
            "spectral_contrast_length_eq_n_bands",
            "spectral_contrast_length_eq_n_bands",
            result.len() == n_bands,
        );
        check(
            "spectral_contrast_length_eq_n_bands",
            result.len() == n_bands,
            format!("got {}", result.len()),
        );
        let all_finite = result.iter().all(|v| v.is_finite());
        ledger.compared(
            "spectral_contrast_values_finite",
            "spectral_contrast_values_finite",
            all_finite,
        );
        check("spectral_contrast_values_finite", all_finite, String::new());
    }

    let all_pass = diffs.iter().all(|d| d.pass);
    let log = DiffLog {
        test_id: "diff_signal_mel_mfcc_chroma".into(),
        category:
            "fsci_signal::{hz_to_mel, mel_to_hz, mel_filterbank, mfcc, chroma, spectral_contrast} coverage"
                .into(),
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
            eprintln!("mel/mfcc/chroma mismatch: {} — {}", d.case_id, d.note);
        }
    }

    assert!(
        all_pass,
        "audio feature coverage failed: {} cases",
        diffs.len()
    );
    // Arms have different case sets: the structural arms are one fixed case each (the smallest
    // designed count), the round-trip arm has one case per input Hz, each recorded through pair
    // with an always-present reference, so a case it did not compare is a failure, not a gap.
    ledger.finish(1);
}
