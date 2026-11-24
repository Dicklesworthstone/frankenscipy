#![forbid(unsafe_code)]
//! Live, bidirectional SciPy STFT/ISTFT conformance, including reconstruction
//! of SciPy-generated and modified coefficients rather than only self-roundtrips.

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fsci_conformance::{ArmCounts, CompareLedger};
use fsci_signal::{
    DetrendType, IstftOptions, StftBoundary, StftOptions, StftResult, StftScaling, StftWindow,
    istft_with_options, stft_with_options,
};
use serde::{Deserialize, Serialize};

const ABS_TOL: f64 = 5e-11;
const REL_TOL: f64 = 2e-11;
const REQUIRE_SCIPY_ENV: &str = "FSCI_REQUIRE_SCIPY_ORACLE";

#[derive(Clone, Debug, Serialize)]
struct PointCase {
    case_id: String,
    x: Vec<f64>,
    fs: f64,
    window: String,
    coefficients: Option<Vec<f64>>,
    nperseg: Option<usize>,
    noverlap: Option<usize>,
    nfft: Option<usize>,
    detrend: Option<String>,
    boundary: Option<String>,
    padded: bool,
    scaling: String,
    infer_inverse: bool,
    modify: bool,
}

#[derive(Debug, Serialize)]
struct OracleQuery {
    points: Vec<PointCase>,
}

#[derive(Debug, Deserialize)]
struct PointArm {
    case_id: String,
    frequencies: Vec<f64>,
    times: Vec<f64>,
    zxx: Vec<Vec<[f64; 2]>>,
    inverse_zxx: Vec<Vec<[f64; 2]>>,
    inverse: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct OracleResult {
    points: Vec<PointArm>,
}

#[derive(Clone, Debug, Serialize)]
struct CaseDiff {
    case_id: String,
    abs_diff: f64,
    pass: bool,
}

#[derive(Debug, Serialize)]
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

fn generate_query() -> OracleQuery {
    let base = PointCase {
        case_id: "centered_even".into(),
        x: (0..37)
            .map(|i| 1.3 + (i as f64 * 0.31).sin() + i as f64 * 0.01)
            .collect(),
        fs: 23.0,
        window: "hann".into(),
        coefficients: None,
        nperseg: Some(8),
        noverlap: None,
        nfft: None,
        detrend: None,
        boundary: Some("zeros".into()),
        padded: true,
        scaling: "spectrum".into(),
        infer_inverse: false,
        modify: false,
    };
    let mut points = vec![
        base.clone(),
        PointCase {
            case_id: "centered_odd".into(),
            nperseg: Some(7),
            ..base.clone()
        },
        PointCase {
            case_id: "inverse_inferred_even".into(),
            infer_inverse: true,
            ..base.clone()
        },
        PointCase {
            case_id: "short_named_clamped".into(),
            x: vec![2.0, -1.0, 3.0],
            nperseg: Some(256),
            ..base.clone()
        },
        PointCase {
            case_id: "single_sample".into(),
            x: vec![4.5],
            nperseg: None,
            boundary: None,
            ..base.clone()
        },
        PointCase {
            case_id: "default_segment_length".into(),
            nperseg: None,
            ..base.clone()
        },
        PointCase {
            case_id: "psd_scaling".into(),
            fs: 8000.0,
            scaling: "psd".into(),
            ..base.clone()
        },
        PointCase {
            case_id: "zero_padded_fft".into(),
            nperseg: Some(5),
            nfft: Some(16),
            noverlap: Some(3),
            window: "hamming".into(),
            ..base.clone()
        },
        PointCase {
            case_id: "custom_odd_fft".into(),
            coefficients: Some(vec![0.1, 0.5, 1.0, 0.5, 0.1]),
            nperseg: None,
            nfft: Some(9),
            noverlap: Some(3),
            scaling: "psd".into(),
            ..base.clone()
        },
        PointCase {
            case_id: "custom_even_fft".into(),
            coefficients: Some(vec![0.1, 0.5, 1.0, 0.5, 0.1]),
            nperseg: Some(5),
            nfft: Some(8),
            noverlap: Some(3),
            ..base.clone()
        },
        PointCase {
            case_id: "negative_sum_window".into(),
            coefficients: Some(vec![-1.0; 4]),
            nperseg: None,
            ..base.clone()
        },
        PointCase {
            case_id: "constant_detrend".into(),
            detrend: Some("constant".into()),
            ..base.clone()
        },
        PointCase {
            case_id: "linear_detrend".into(),
            detrend: Some("linear".into()),
            nperseg: Some(7),
            ..base.clone()
        },
        PointCase {
            case_id: "modified_coefficients".into(),
            window: "blackman".into(),
            modify: true,
            ..base.clone()
        },
        PointCase {
            case_id: "nola_gaps".into(),
            noverlap: Some(0),
            boundary: None,
            ..base.clone()
        },
        PointCase {
            case_id: "zero_signal".into(),
            x: vec![0.0; 31],
            ..base.clone()
        },
    ];
    for boundary in [None, Some("even"), Some("odd"), Some("constant")] {
        for padded in [false, true] {
            for nperseg in [5, 6] {
                points.push(PointCase {
                    case_id: format!(
                        "boundary_{}_padded_{padded}_n_{nperseg}",
                        boundary.unwrap_or("none")
                    ),
                    boundary: boundary.map(str::to_string),
                    padded,
                    nperseg: Some(nperseg),
                    window: "hamming".into(),
                    ..base.clone()
                });
            }
        }
    }
    OracleQuery { points }
}

fn scipy_oracle_or_skip(query: &OracleQuery) -> Option<OracleResult> {
    let script = r#"
import json
import sys
import numpy as np
from scipy import signal

def pairs(z):
    return [[[float(v.real), float(v.imag)] for v in row] for row in z.T]

query = json.load(sys.stdin)
points = []
for case in query["points"]:
    x = np.asarray(case["x"], dtype=float)
    window = case["window"] if case["coefficients"] is None else np.asarray(case["coefficients"])
    nperseg = case["nperseg"]
    if nperseg is None:
        nperseg = min(256, len(x)) if isinstance(window, str) else len(window)
    elif isinstance(window, str):
        nperseg = min(nperseg, len(x))
    f, t, z = signal.stft(
        x, fs=case["fs"], window=window,
        nperseg=case["nperseg"] if case["nperseg"] is not None else (256 if isinstance(window, str) else None),
        noverlap=case["noverlap"], nfft=case["nfft"],
        detrend=case["detrend"] if case["detrend"] is not None else False,
        boundary=case["boundary"], padded=case["padded"], scaling=case["scaling"],
    )
    inverse_z = z.copy()
    if case["modify"]:
        inverse_z[1:-1, ::2] *= 0.35
    _, y = signal.istft(
        inverse_z, fs=case["fs"], window=window,
        nperseg=None if case["infer_inverse"] else nperseg,
        noverlap=case["noverlap"], nfft=case["nfft"],
        boundary=case["boundary"] is not None, scaling=case["scaling"],
    )
    points.append({
        "case_id": case["case_id"], "frequencies": f.tolist(), "times": t.tolist(),
        "zxx": pairs(z), "inverse_zxx": pairs(inverse_z), "inverse": y.tolist(),
    })
print(json.dumps({"points": points}, allow_nan=False))
"#;
    let query_json = serde_json::to_string(query).expect("serialize STFT oracle query");
    let mut child = match fsci_conformance::scipy_oracle_command()
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            assert!(
                std::env::var(REQUIRE_SCIPY_ENV).is_err(),
                "cannot start STFT oracle: {error}"
            );
            eprintln!("skipping STFT oracle: {error}");
            return None;
        }
    };
    if let Err(error) = child
        .stdin
        .as_mut()
        .expect("oracle stdin")
        .write_all(query_json.as_bytes())
    {
        let output = child
            .wait_with_output()
            .expect("wait for failed STFT oracle");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "STFT oracle write failed: {error}; {stderr}"
        );
        eprintln!("skipping STFT oracle: {error}; {stderr}");
        return None;
    }
    let output = child.wait_with_output().expect("wait for STFT oracle");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            std::env::var(REQUIRE_SCIPY_ENV).is_err(),
            "STFT oracle failed: {stderr}"
        );
        eprintln!("skipping STFT oracle: {stderr}");
        return None;
    }
    Some(serde_json::from_slice(&output.stdout).expect("parse STFT oracle output"))
}

#[test]
fn diff_signal_stft_istft() {
    let query = generate_query();
    let Some(oracle) = scipy_oracle_or_skip(&query) else {
        return;
    };
    assert_eq!(oracle.points.len(), query.points.len());
    let references: HashMap<_, _> = oracle
        .points
        .into_iter()
        .map(|point| (point.case_id.clone(), point))
        .collect();
    assert_eq!(
        references.len(),
        query.points.len(),
        "duplicate oracle case IDs"
    );
    let started = Instant::now();
    let mut ledger = CompareLedger::new(
        "diff_signal_stft_istft",
        &["stft_frequencies", "stft_times", "stft", "istft"],
    );
    let mut diffs = Vec::new();
    let mut max_abs_diff = 0.0_f64;
    for case in &query.points {
        let reference = references.get(&case.case_id).expect("oracle case exists");
        let window = case.coefficients.as_deref().map_or(
            StftWindow::Named(case.window.as_str()),
            StftWindow::Coefficients,
        );
        let scaling = match case.scaling.as_str() {
            "spectrum" => StftScaling::Magnitude,
            "psd" => StftScaling::Psd,
            unexpected => panic!("invalid test scaling: {unexpected}"),
        };
        let boundary = match case.boundary.as_deref() {
            None => StftBoundary::None,
            Some("zeros") => StftBoundary::Zeros,
            Some("even") => StftBoundary::Even,
            Some("odd") => StftBoundary::Odd,
            Some("constant") => StftBoundary::Constant,
            unexpected => panic!("invalid test boundary: {unexpected:?}"),
        };
        let detrend = match case.detrend.as_deref() {
            None => None,
            Some("constant") => Some(DetrendType::Constant),
            Some("linear") => Some(DetrendType::Linear),
            unexpected => panic!("invalid test detrend: {unexpected:?}"),
        };
        let actual = stft_with_options(
            &case.x,
            &StftOptions {
                fs: case.fs,
                window,
                nperseg: case.nperseg,
                noverlap: case.noverlap,
                nfft: case.nfft,
                detrend,
                boundary,
                padded: case.padded,
                scaling,
            },
        )
        .unwrap_or_else(|error| panic!("STFT {}: {error}", case.case_id));
        let effective_segment = case
            .coefficients
            .as_ref()
            .map_or_else(|| case.nperseg.unwrap_or(256).min(case.x.len()), Vec::len);
        // Feed SciPy's coefficients into Rust's inverse. Matching mistakes in
        // the Rust forward/inverse pair cannot hide behind a successful roundtrip.
        let inverse_input = StftResult {
            frequencies: reference.frequencies.clone(),
            times: reference.times.clone(),
            zxx: reference
                .inverse_zxx
                .iter()
                .map(|frame| frame.iter().map(|&[re, im]| (re, im)).collect())
                .collect(),
        };
        let inverse = istft_with_options(
            &inverse_input,
            &IstftOptions {
                fs: case.fs,
                window,
                nperseg: if case.infer_inverse {
                    None
                } else {
                    Some(effective_segment)
                },
                noverlap: case.noverlap,
                nfft: case.nfft,
                boundary: boundary != StftBoundary::None,
                scaling,
            },
        )
        .unwrap_or_else(|error| panic!("ISTFT {}: {error}", case.case_id));
        assert_eq!(
            actual.zxx.len(),
            reference.zxx.len(),
            "{} frame count",
            case.case_id
        );
        for (actual_frame, expected_frame) in actual.zxx.iter().zip(&reference.zxx) {
            assert_eq!(
                actual_frame.len(),
                expected_frame.len(),
                "{} frequency count",
                case.case_id
            );
        }
        let actual_z: Vec<f64> = actual
            .zxx
            .iter()
            .flatten()
            .flat_map(|&(re, im)| [re, im])
            .collect();
        let reference_z: Vec<f64> = reference.zxx.iter().flatten().flatten().copied().collect();
        for (arm, expected, actual) in [
            (
                "stft_frequencies",
                reference.frequencies.as_slice(),
                actual.frequencies.as_slice(),
            ),
            (
                "stft_times",
                reference.times.as_slice(),
                actual.times.as_slice(),
            ),
            ("stft", reference_z.as_slice(), actual_z.as_slice()),
            ("istft", reference.inverse.as_slice(), inverse.as_slice()),
        ] {
            let Some((expected, actual)) =
                ledger.slices(arm, &case.case_id, Some(expected), Some(actual))
            else {
                continue;
            };
            let abs_diff = actual
                .iter()
                .zip(expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0_f64, f64::max);
            let pass = actual.iter().zip(expected).all(|(actual, expected)| {
                (actual - expected).abs() <= ABS_TOL + REL_TOL * expected.abs()
            });
            max_abs_diff = max_abs_diff.max(abs_diff);
            ledger.compared(arm, &case.case_id, pass);
            diffs.push(CaseDiff {
                case_id: format!("{arm}/{}", case.case_id),
                abs_diff,
                pass,
            });
        }
    }
    let pass = diffs.iter().all(|case| case.pass);
    let log = DiffLog {
        test_id: "diff_signal_stft_istft".into(),
        category: "scipy.signal.stft/istft".into(),
        case_count: diffs.len(),
        compared: ledger.counts().clone(),
        max_abs_diff,
        pass,
        timestamp_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |time| time.as_millis()),
        duration_ns: started.elapsed().as_nanos(),
        cases: diffs,
    };
    let directory = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/artifacts/FSCI-P2C-011/diff");
    std::fs::create_dir_all(&directory).expect("create STFT differential artifact directory");
    std::fs::write(
        directory.join("diff_signal_stft_istft.json"),
        serde_json::to_string_pretty(&log).expect("serialize STFT differential log"),
    )
    .expect("write STFT differential log");
    assert!(pass, "STFT/ISTFT SciPy comparison failed: {log:?}");
    ledger.finish(query.points.len());
}
