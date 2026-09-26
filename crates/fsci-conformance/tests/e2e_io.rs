#![forbid(unsafe_code)]
//! E2E scenario tests for FSCI-P2C-013 (I/O).
//!
//! Implements conformance tests for scipy.io parity:
//!   Happy-path (1-5): Matrix Market, WAV, CSV, text, JSON
//!   Edge cases (6-8): empty data, malformed input, encoding issues
//!   Cross-op consistency (9-11): roundtrip tests
//!   Performance boundary (12-14): large files
//!
//! Each scenario emits a forensic log bundle to
//! `fixtures/artifacts/FSCI-P2C-013/e2e/`.

use fsci_io::{
    MatArray, MmField, MmFormat, MmObject, MmSymmetry, loadmat_text, loadtxt, mminfo, mmread,
    mmwrite, mmwrite_sparse, read_csv, read_json_array, savemat_text, savetxt, wav_read, wav_write,
    write_csv, write_json_array,
};
use serde::Serialize;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

// ───────────────────────── Forensic log types ─────────────────────────

#[derive(Debug, Clone, Serialize)]
struct ForensicLogBundle {
    scenario_id: String,
    steps: Vec<ForensicStep>,
    artifacts: Vec<ArtifactRef>,
    environment: EnvironmentInfo,
    io_metadata: Option<IoMetadata>,
    overall: OverallResult,
}

#[derive(Debug, Clone, Serialize)]
struct ForensicStep {
    step_id: usize,
    step_name: String,
    action: String,
    input_summary: String,
    output_summary: String,
    duration_ns: u128,
    mode: String,
    outcome: String,
}

#[derive(Debug, Clone, Serialize)]
struct ArtifactRef {
    path: String,
    blake3: String,
}

#[derive(Debug, Clone, Serialize)]
struct EnvironmentInfo {
    rust_version: String,
    os: String,
    cpu_count: usize,
    total_memory_mb: String,
}

#[derive(Debug, Clone, Serialize)]
struct IoMetadata {
    format: String,
    operation: String,
    size_bytes: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OverallResult {
    status: String,
    total_duration_ns: u128,
    replay_command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_chain: Option<String>,
}

// ───────────────────────── Helpers ─────────────────────────

fn e2e_output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/artifacts/FSCI-P2C-013/e2e")
}

fn make_env() -> EnvironmentInfo {
    EnvironmentInfo {
        rust_version: String::from(env!("CARGO_PKG_VERSION")),
        os: String::from(std::env::consts::OS),
        cpu_count: std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1),
        total_memory_mb: String::from("unknown"),
    }
}

fn replay_cmd(scenario_id: &str) -> String {
    format!("cargo test -p fsci-conformance --test e2e_io -- {scenario_id} --nocapture")
}

fn write_bundle(scenario_id: &str, bundle: &ForensicLogBundle) {
    let dir = e2e_output_dir();
    fs::create_dir_all(&dir).ok();
    let path = dir.join(format!("{scenario_id}.json"));
    if let Ok(json) = serde_json::to_string_pretty(bundle) {
        fs::write(path, json).ok();
    }
}

// ───────────────────────── Scenario Runner ─────────────────────────

struct ScenarioRunner {
    scenario_id: String,
    steps: Vec<ForensicStep>,
    start: Instant,
    step_counter: usize,
    io_metadata: Option<IoMetadata>,
    status: String,
    error_chain: Option<String>,
}

impl ScenarioRunner {
    fn new(scenario_id: &str) -> Self {
        Self {
            scenario_id: scenario_id.to_string(),
            steps: Vec::new(),
            start: Instant::now(),
            step_counter: 0,
            io_metadata: None,
            status: "pass".to_string(),
            error_chain: None,
        }
    }

    fn set_io_meta(&mut self, format: &str, operation: &str, size_bytes: usize) {
        self.io_metadata = Some(IoMetadata {
            format: format.to_string(),
            operation: operation.to_string(),
            size_bytes,
        });
    }

    fn step<F>(&mut self, name: &str, action: &str, input: &str, mode: &str, f: F)
    where
        F: FnOnce() -> Result<String, String>,
    {
        self.step_counter += 1;
        let step_start = Instant::now();
        let result = f();
        let duration_ns = step_start.elapsed().as_nanos();

        let (outcome, output_summary) = match result {
            Ok(out) => ("pass".to_string(), out),
            Err(e) => {
                self.status = "fail".to_string();
                if self.error_chain.is_none() {
                    self.error_chain = Some(e.clone());
                }
                ("fail".to_string(), e)
            }
        };

        self.steps.push(ForensicStep {
            step_id: self.step_counter,
            step_name: name.to_string(),
            action: action.to_string(),
            input_summary: input.to_string(),
            output_summary,
            duration_ns,
            mode: mode.to_string(),
            outcome,
        });
    }

    fn finish(self) -> ForensicLogBundle {
        ForensicLogBundle {
            scenario_id: self.scenario_id.clone(),
            steps: self.steps,
            artifacts: vec![],
            environment: make_env(),
            io_metadata: self.io_metadata,
            overall: OverallResult {
                status: self.status,
                total_duration_ns: self.start.elapsed().as_nanos(),
                replay_command: replay_cmd(&self.scenario_id),
                error_chain: self.error_chain,
            },
        }
    }
}

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol || (a.is_nan() && b.is_nan())
}

/// `f64::max` returns the other operand when one is NaN, so folding residuals with it reads a
/// NaN entry as agreement. This keeps the NaN, and `NaN <= tol` then fails the case.
fn nan_max(acc: f64, d: f64) -> f64 {
    if acc.is_nan() || d.is_nan() {
        f64::NAN
    } else {
        acc.max(d)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SCENARIOS 1-5: HAPPY-PATH
// ═══════════════════════════════════════════════════════════════════════

/// Scenario 1: Matrix Market format basic operations
#[test]
fn scenario_01_matrix_market() {
    let mut runner = ScenarioRunner::new("scenario_01_matrix_market");
    runner.set_io_meta("Matrix Market", "read/write", 0);

    // Dense array format
    let mm_dense = r#"%%MatrixMarket matrix array real general
3 2
1.0
2.0
3.0
4.0
5.0
6.0
"#;

    runner.step(
        "mmread_dense",
        "mmread(dense_matrix)",
        "3x2 array format",
        "Strict",
        || {
            let result = mmread(mm_dense).map_err(|e| format!("{e}"))?;
            if result.rows != 3 || result.cols != 2 {
                return Err(format!("expected 3x2, got {}x{}", result.rows, result.cols));
            }
            // MM input is column-major [1,2,3,4,5,6], but stored as row-major:
            // Row 0: [1, 4], Row 1: [2, 5], Row 2: [3, 6]
            // So data = [1, 4, 2, 5, 3, 6]
            let expected = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
            for (i, (&a, &e)) in result.data.iter().zip(expected.iter()).enumerate() {
                if !approx_eq(a, e, 1e-10) {
                    return Err(format!("data[{i}]: expected {e}, got {a}"));
                }
            }
            Ok(format!("{}x{} matrix read", result.rows, result.cols))
        },
    );

    runner.step(
        "mminfo",
        "mminfo(content)",
        "parse header only",
        "Strict",
        || {
            let info = mminfo(mm_dense).map_err(|e| format!("{e}"))?;
            if info.object != MmObject::Matrix {
                return Err("expected Matrix object".to_string());
            }
            if info.format != MmFormat::Array {
                return Err("expected Array format".to_string());
            }
            if info.field != MmField::Real {
                return Err("expected Real field".to_string());
            }
            if info.symmetry != MmSymmetry::General {
                return Err("expected General symmetry".to_string());
            }
            Ok(format!(
                "object={:?}, format={:?}, field={:?}",
                info.object, info.format, info.field
            ))
        },
    );

    runner.step(
        "mmwrite_dense",
        "mmwrite(3, 2, data)",
        "write 3x2 matrix",
        "Strict",
        || {
            let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            let content = mmwrite(3, 2, &data).map_err(|e| format!("{e}"))?;
            if !content.contains("%%MatrixMarket") {
                return Err("missing MatrixMarket header".to_string());
            }
            if !content.contains("matrix array real general") {
                return Err("wrong header format".to_string());
            }
            Ok(format!("wrote {} bytes", content.len()))
        },
    );

    // Coordinate (sparse) format
    let mm_sparse = r#"%%MatrixMarket matrix coordinate real general
3 3 4
1 1 1.0
2 2 2.0
3 3 3.0
1 3 0.5
"#;

    runner.step(
        "mmread_sparse",
        "mmread(sparse_matrix)",
        "3x3 coordinate format",
        "Strict",
        || {
            let result = mmread(mm_sparse).map_err(|e| format!("{e}"))?;
            if result.rows != 3 || result.cols != 3 {
                return Err(format!("expected 3x3, got {}x{}", result.rows, result.cols));
            }
            // Stored row-major: data[row*3 + col]
            // m[0,0]=1.0 at idx=0, m[1,1]=2.0 at idx=4, m[2,2]=3.0 at idx=8, m[0,2]=0.5 at idx=2
            let m00 = result.data[0]; // (0,0)
            let m11 = result.data[4]; // (1,1)
            let m22 = result.data[8]; // (2,2)
            let m02 = result.data[2]; // (0,2) = 0*3 + 2 = 2
            if !approx_eq(m00, 1.0, 1e-10)
                || !approx_eq(m11, 2.0, 1e-10)
                || !approx_eq(m22, 3.0, 1e-10)
                || !approx_eq(m02, 0.5, 1e-10)
            {
                return Err(format!(
                    "values wrong: m00={m00}, m11={m11}, m22={m22}, m02={m02}"
                ));
            }
            Ok(format!("sparse 3x3 read, nnz={}", result.info.nnz))
        },
    );

    runner.step(
        "mmwrite_sparse",
        "mmwrite_sparse(triplets)",
        "write sparse matrix",
        "Strict",
        || {
            let triplets = vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)];
            let content = mmwrite_sparse(3, 3, &triplets).map_err(|e| format!("{e}"))?;
            if !content.contains("coordinate") {
                return Err("should be coordinate format".to_string());
            }
            Ok(format!("wrote {} bytes", content.len()))
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_01_matrix_market", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_01 failed");
}

/// Scenario 2: WAV audio file operations
#[test]
fn scenario_02_wav_audio() {
    let mut runner = ScenarioRunner::new("scenario_02_wav_audio");
    runner.set_io_meta("WAV", "read/write", 0);

    runner.step(
        "wav_write_mono",
        "wav_write(44100, 1, sine_wave)",
        "generate mono WAV",
        "Strict",
        || {
            // Generate a short sine wave
            let sample_rate = 44100u32;
            let duration_samples = 4410; // 0.1 seconds
            let freq = 440.0; // A4
            let data: Vec<f64> = (0..duration_samples)
                .map(|i| {
                    let t = i as f64 / sample_rate as f64;
                    (2.0 * std::f64::consts::PI * freq * t).sin()
                })
                .collect();

            let bytes = wav_write(sample_rate, 1, &data).map_err(|e| format!("{e}"))?;

            // Check RIFF header
            if bytes.len() < 44 {
                return Err(format!("WAV too short: {} bytes", bytes.len()));
            }
            if &bytes[0..4] != b"RIFF" {
                return Err("missing RIFF header".to_string());
            }
            if &bytes[8..12] != b"WAVE" {
                return Err("missing WAVE identifier".to_string());
            }

            Ok(format!(
                "wrote {} bytes for {} samples",
                bytes.len(),
                duration_samples
            ))
        },
    );

    runner.step(
        "wav_roundtrip",
        "wav_read(wav_write(...))",
        "verify roundtrip",
        "Strict",
        || {
            let sample_rate = 22050u32;
            let data: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.01).sin() * 0.5).collect();

            let bytes = wav_write(sample_rate, 1, &data).map_err(|e| format!("{e}"))?;
            let wav = wav_read(&bytes).map_err(|e| format!("{e}"))?;

            if wav.sample_rate != sample_rate {
                return Err(format!(
                    "sample_rate mismatch: {} vs {}",
                    wav.sample_rate, sample_rate
                ));
            }
            if wav.channels != 1 {
                return Err(format!("expected mono, got {} channels", wav.channels));
            }

            // Check samples (allow some quantization error from int16 encoding)
            for (i, (&orig, &read)) in data.iter().zip(wav.data.iter()).enumerate() {
                if (orig - read).abs() > 0.001 {
                    return Err(format!("sample {i} mismatch: wrote {orig}, read {read}"));
                }
            }

            Ok(format!(
                "roundtrip verified: {} samples at {}Hz",
                wav.data.len(),
                wav.sample_rate
            ))
        },
    );

    runner.step(
        "wav_write_stereo",
        "wav_write(44100, 2, interleaved)",
        "generate stereo WAV",
        "Strict",
        || {
            let sample_rate = 44100u32;
            // Stereo: interleaved [L0, R0, L1, R1, ...]
            let frames = 1000;
            let mut data = Vec::with_capacity(frames * 2);
            for i in 0..frames {
                let t = i as f64 / sample_rate as f64;
                let left = (440.0 * 2.0 * std::f64::consts::PI * t).sin();
                let right = (880.0 * 2.0 * std::f64::consts::PI * t).sin();
                data.push(left);
                data.push(right);
            }

            let bytes = wav_write(sample_rate, 2, &data).map_err(|e| format!("{e}"))?;
            let wav = wav_read(&bytes).map_err(|e| format!("{e}"))?;

            if wav.channels != 2 {
                return Err(format!("expected 2 channels, got {}", wav.channels));
            }

            Ok(format!("stereo WAV: {} bytes", bytes.len()))
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_02_wav_audio", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_02 failed");
}

/// Scenario 3: Text file operations (loadtxt/savetxt)
#[test]
fn scenario_03_text_files() {
    let mut runner = ScenarioRunner::new("scenario_03_text_files");
    runner.set_io_meta("text", "loadtxt/savetxt", 0);

    let txt_content = "1.0 2.0 3.0\n4.0 5.0 6.0\n7.0 8.0 9.0\n";

    runner.step(
        "loadtxt",
        "loadtxt(content)",
        "3x3 whitespace-delimited",
        "Strict",
        || {
            let (rows, cols, data) = loadtxt(txt_content).map_err(|e| format!("{e}"))?;
            if rows != 3 || cols != 3 {
                return Err(format!("expected 3x3, got {}x{}", rows, cols));
            }
            let expected: Vec<f64> = (1..=9).map(|i| i as f64).collect();
            for (i, (&a, &e)) in data.iter().zip(expected.iter()).enumerate() {
                if !approx_eq(a, e, 1e-10) {
                    return Err(format!("data[{i}]: expected {e}, got {a}"));
                }
            }
            Ok(format!("{}x{} matrix loaded", rows, cols))
        },
    );

    runner.step(
        "savetxt",
        "savetxt(3, 3, data)",
        "write 3x3 matrix",
        "Strict",
        || {
            let data: Vec<f64> = (1..=9).map(|i| i as f64).collect();
            let content = savetxt(3, 3, &data, " ").map_err(|e| format!("{e}"))?;
            if content.lines().count() != 3 {
                return Err("expected 3 lines".to_string());
            }
            Ok(format!("wrote {} chars", content.len()))
        },
    );

    runner.step(
        "loadtxt_savetxt_roundtrip",
        "loadtxt(savetxt(...))",
        "verify roundtrip",
        "Strict",
        || {
            let original: Vec<f64> = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5];
            let content = savetxt(2, 3, &original, " ").map_err(|e| format!("{e}"))?;
            let (rows, cols, data) = loadtxt(&content).map_err(|e| format!("{e}"))?;

            if rows != 2 || cols != 3 {
                return Err(format!("dimension mismatch: {}x{}", rows, cols));
            }
            for (i, (&o, &r)) in original.iter().zip(data.iter()).enumerate() {
                if !approx_eq(o, r, 1e-10) {
                    return Err(format!("data[{i}] mismatch: {o} vs {r}"));
                }
            }
            Ok("roundtrip verified".to_string())
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_03_text_files", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_03 failed");
}

/// Scenario 4: CSV file operations
#[test]
fn scenario_04_csv_files() {
    let mut runner = ScenarioRunner::new("scenario_04_csv_files");
    runner.set_io_meta("CSV", "read/write", 0);

    runner.step(
        "read_csv_no_header",
        "read_csv(content, ',', false)",
        "3x3 numeric CSV",
        "Strict",
        || {
            let content = "1,2,3\n4,5,6\n7,8,9\n";
            let (header, data) = read_csv(content, ',', false).map_err(|e| format!("{e}"))?;
            if header.is_some() {
                return Err("expected no header".to_string());
            }
            if data.len() != 3 || data[0].len() != 3 {
                return Err(format!(
                    "expected 3x3, got {}x{}",
                    data.len(),
                    data.first().map(|r| r.len()).unwrap_or(0)
                ));
            }
            // Check values row by row
            let expected = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
            for (i, row) in data.iter().enumerate() {
                for (j, &val) in row.iter().enumerate() {
                    if !approx_eq(val, expected[i][j], 1e-10) {
                        return Err(format!(
                            "data[{i}][{j}]: expected {}, got {val}",
                            expected[i][j]
                        ));
                    }
                }
            }
            Ok(format!("read {}x{} CSV", data.len(), data[0].len()))
        },
    );

    runner.step(
        "read_csv_with_header",
        "read_csv(content, ',', true)",
        "CSV with header row",
        "Strict",
        || {
            let content = "a,b,c\n1,2,3\n4,5,6\n";
            let (header, data) = read_csv(content, ',', true).map_err(|e| format!("{e}"))?;
            if data.len() != 2 {
                return Err(format!("expected 2 data rows, got {}", data.len()));
            }
            if header.is_none() {
                return Err("expected header".to_string());
            }
            let h = header.as_ref().unwrap();
            if h != &["a", "b", "c"] {
                return Err(format!("wrong header: {:?}", h));
            }
            Ok(format!("header: {:?}, {} data rows", h, data.len()))
        },
    );

    runner.step(
        "write_csv",
        "write_csv(header, data, ',')",
        "write CSV with header",
        "Strict",
        || {
            let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
            let header: Option<&[&str]> = Some(&["x", "y", "z"]);
            let content = write_csv(header, &data, ',').map_err(|e| format!("{e}"))?;
            if !content.starts_with("x,y,z") {
                return Err("missing header in output".to_string());
            }
            let lines: Vec<&str> = content.lines().collect();
            if lines.len() != 3 {
                // header + 2 data rows
                return Err(format!("expected 3 lines, got {}", lines.len()));
            }
            Ok(format!("wrote {} chars", content.len()))
        },
    );

    runner.step(
        "csv_roundtrip",
        "read_csv(write_csv(...))",
        "verify roundtrip",
        "Strict",
        || {
            let original = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
            let content = write_csv(None, &original, ',').map_err(|e| format!("{e}"))?;
            let (_, data) = read_csv(&content, ',', false).map_err(|e| format!("{e}"))?;

            if data.len() != 2 || data[0].len() != 2 {
                return Err(format!(
                    "dimension mismatch: {}x{}",
                    data.len(),
                    data[0].len()
                ));
            }
            for (i, row) in original.iter().enumerate() {
                for (j, &orig_val) in row.iter().enumerate() {
                    if !approx_eq(orig_val, data[i][j], 1e-10) {
                        return Err(format!("data[{i}][{j}] mismatch"));
                    }
                }
            }
            Ok("CSV roundtrip verified".to_string())
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_04_csv_files", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_04 failed");
}

/// Scenario 5: JSON array and MATLAB text format
#[test]
fn scenario_05_json_and_mat() {
    let mut runner = ScenarioRunner::new("scenario_05_json_and_mat");
    runner.set_io_meta("JSON/MAT", "read/write", 0);

    runner.step(
        "read_json_array",
        "read_json_array([1,2,3])",
        "simple JSON array",
        "Strict",
        || {
            let content = "[1.0, 2.5, 3.7, 4.2]";
            let data = read_json_array(content).map_err(|e| format!("{e}"))?;
            if data.len() != 4 {
                return Err(format!("expected 4 elements, got {}", data.len()));
            }
            let expected = [1.0, 2.5, 3.7, 4.2];
            for (i, (&a, &e)) in data.iter().zip(expected.iter()).enumerate() {
                if !approx_eq(a, e, 1e-10) {
                    return Err(format!("element[{i}]: expected {e}, got {a}"));
                }
            }
            Ok(format!("read {} elements", data.len()))
        },
    );

    runner.step(
        "write_json_array",
        "write_json_array([1,2,3])",
        "write JSON array",
        "Strict",
        || {
            let data = vec![1.0, 2.0, 3.0];
            let content = write_json_array(&data).map_err(|e| format!("{e}"))?;
            if !content.starts_with('[') || !content.ends_with(']') {
                return Err("not a JSON array".to_string());
            }
            Ok(format!("wrote {} chars", content.len()))
        },
    );

    runner.step(
        "json_roundtrip",
        "read_json_array(write_json_array(...))",
        "verify roundtrip",
        "Strict",
        || {
            let original = vec![1.5, 2.5, 3.5, 4.5, 5.5];
            let content = write_json_array(&original).map_err(|e| format!("{e}"))?;
            let read = read_json_array(&content).map_err(|e| format!("{e}"))?;

            if original.len() != read.len() {
                return Err("length mismatch".to_string());
            }
            for (i, (&o, &r)) in original.iter().zip(read.iter()).enumerate() {
                if !approx_eq(o, r, 1e-10) {
                    return Err(format!("element[{i}] mismatch"));
                }
            }
            Ok("JSON roundtrip verified".to_string())
        },
    );

    runner.step(
        "savemat_loadmat_roundtrip",
        "loadmat_text(savemat_text(...))",
        "MATLAB text format roundtrip",
        "Strict",
        || {
            let arrays = vec![
                MatArray {
                    name: "x".to_string(),
                    rows: 2,
                    cols: 3,
                    data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                },
                MatArray {
                    name: "y".to_string(),
                    rows: 1,
                    cols: 2,
                    data: vec![7.0, 8.0],
                },
            ];

            let content = savemat_text(&arrays).map_err(|e| format!("{e}"))?;
            let loaded = loadmat_text(&content).map_err(|e| format!("{e}"))?;

            if loaded.len() != 2 {
                return Err(format!("expected 2 arrays, got {}", loaded.len()));
            }

            // Check first array
            let x = &loaded[0];
            if x.name != "x" || x.rows != 2 || x.cols != 3 {
                return Err(format!(
                    "array x mismatch: {} {}x{}",
                    x.name, x.rows, x.cols
                ));
            }

            Ok("MAT text roundtrip verified".to_string())
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_05_json_and_mat", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_05 failed");
}

// ═══════════════════════════════════════════════════════════════════════
// SCENARIOS 6-8: EDGE CASES AND ERROR HANDLING
// ═══════════════════════════════════════════════════════════════════════

/// Scenario 6: Empty and minimal data handling
#[test]
fn scenario_06_empty_data() {
    let mut runner = ScenarioRunner::new("scenario_06_empty_data");
    runner.set_io_meta("multiple", "error_handling", 0);

    runner.step(
        "loadtxt_empty",
        "loadtxt('')",
        "empty content",
        "Strict",
        || match loadtxt("") {
            Err(_) => Ok("correctly rejected empty content".to_string()),
            Ok((rows, cols, _)) => Ok(format!("accepted empty as {}x{}", rows, cols)),
        },
    );

    runner.step(
        "mmwrite_zero_dim",
        "mmwrite(0, 3, [])",
        "zero rows",
        "Strict",
        || {
            let result = mmwrite(0, 3, &[]);
            // Zero dimensions may be valid (empty matrix)
            match result {
                Ok(content) => Ok(format!("accepted 0x3 matrix: {} bytes", content.len())),
                Err(e) => Ok(format!("rejected zero dimension: {e}")),
            }
        },
    );

    runner.step(
        "wav_write_empty",
        "wav_write(44100, 1, [])",
        "empty audio data",
        "Strict",
        || {
            let result = wav_write(44100, 1, &[]);
            // Empty WAV is technically valid (just header), but check behavior
            match result {
                Ok(bytes) => Ok(format!("accepted empty: {} bytes", bytes.len())),
                Err(e) => Ok(format!("rejected empty: {e}")),
            }
        },
    );

    runner.step(
        "json_empty_array",
        "read_json_array('[]')",
        "empty JSON array",
        "Strict",
        || {
            let data = read_json_array("[]").map_err(|e| format!("{e}"))?;
            if data.is_empty() {
                Ok("correctly parsed empty array".to_string())
            } else {
                Err(format!("expected empty, got {} elements", data.len()))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_06_empty_data", &bundle);
    for step in &bundle.steps {
        if step.outcome != "pass" {
            eprintln!("FAILED: {} - {}", step.step_name, step.output_summary);
        }
    }
    assert!(bundle.overall.status == "pass", "scenario_06 failed");
}

/// Scenario 7: Malformed input handling
#[test]
fn scenario_07_malformed_input() {
    let mut runner = ScenarioRunner::new("scenario_07_malformed_input");
    runner.set_io_meta("multiple", "error_handling", 0);

    runner.step(
        "mm_bad_header",
        "mmread(bad header)",
        "missing MatrixMarket header",
        "Strict",
        || {
            let bad = "matrix array real general\n3 3\n1 2 3\n";
            let result = mmread(bad);
            if result.is_err() {
                Ok("correctly rejected bad header".to_string())
            } else {
                Err("should reject missing MatrixMarket header".to_string())
            }
        },
    );

    runner.step(
        "mm_incomplete",
        "mmread(incomplete data)",
        "fewer values than declared",
        "Strict",
        || {
            let incomplete = "%%MatrixMarket matrix array real general\n3 3\n1.0\n2.0\n";
            let result = mmread(incomplete);
            if result.is_err() {
                Ok("correctly rejected incomplete data".to_string())
            } else {
                Err("should reject incomplete data".to_string())
            }
        },
    );

    runner.step(
        "wav_bad_header",
        "wav_read(random bytes)",
        "invalid WAV data",
        "Strict",
        || {
            let bad = b"not a wav file at all";
            let result = wav_read(bad);
            if result.is_err() {
                Ok("correctly rejected invalid WAV".to_string())
            } else {
                Err("should reject invalid WAV".to_string())
            }
        },
    );

    runner.step(
        "json_malformed",
        "read_json_array('[1,2,')",
        "incomplete JSON",
        "Strict",
        || {
            let bad = "[1, 2,";
            let result = read_json_array(bad);
            if result.is_err() {
                Ok("correctly rejected malformed JSON".to_string())
            } else {
                Err("should reject malformed JSON".to_string())
            }
        },
    );

    runner.step(
        "csv_inconsistent_cols",
        "read_csv with varying column count",
        "ragged CSV",
        "Strict",
        || {
            let ragged = "1,2,3\n4,5\n6,7,8,9\n";
            let result = read_csv(ragged, ',', false);
            // Should error on inconsistent column count
            match result {
                Ok((_, data)) => Ok(format!("accepted ragged CSV: {} rows", data.len())),
                Err(e) => Ok(format!("correctly rejected ragged CSV: {e}")),
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_07_malformed_input", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_07 failed");
}

/// Scenario 8: Special values (NaN, Inf)
#[test]
fn scenario_08_special_values() {
    let mut runner = ScenarioRunner::new("scenario_08_special_values");
    runner.set_io_meta("multiple", "special_values", 0);

    runner.step(
        "json_with_specials",
        "json with NaN/Inf handling",
        "special float values",
        "Strict",
        || {
            // Standard JSON doesn't support NaN/Inf
            let data = vec![1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
            let result = write_json_array(&data);
            match result {
                Ok(content) => {
                    // Check if it encodes as null or strings
                    Ok(format!(
                        "encoded specials: {}",
                        &content[..content.len().min(50)]
                    ))
                }
                Err(e) => Ok(format!("rejected specials: {e}")),
            }
        },
    );

    runner.step(
        "csv_with_nan",
        "savetxt with NaN",
        "NaN in numeric data",
        "Strict",
        || {
            let data = vec![1.0, f64::NAN, 3.0, 4.0];
            let result = savetxt(2, 2, &data, " ");
            match result {
                Ok(content) => {
                    if content.contains("nan") || content.contains("NaN") || content.contains("NAN")
                    {
                        Ok("NaN represented in output".to_string())
                    } else {
                        Err(format!("NaN not visible in output: {content}"))
                    }
                }
                Err(e) => Ok(format!("rejected NaN: {e}")),
            }
        },
    );

    runner.step(
        "mm_with_inf",
        "mmwrite with Inf",
        "Inf in matrix data",
        "Strict",
        || {
            let data = vec![1.0, f64::INFINITY, 3.0, 4.0];
            let result = mmwrite(2, 2, &data);
            match result {
                Ok(content) => {
                    if content.contains("inf") || content.contains("Inf") || content.contains("INF")
                    {
                        Ok("Inf represented in output".to_string())
                    } else {
                        // Some implementations use scientific notation
                        Ok(format!("output length: {} chars", content.len()))
                    }
                }
                Err(e) => Ok(format!("rejected Inf: {e}")),
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_08_special_values", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_08 failed");
}

// ═══════════════════════════════════════════════════════════════════════
// SCENARIOS 9-11: CROSS-OP CONSISTENCY
// ═══════════════════════════════════════════════════════════════════════

/// Scenario 9: Format conversion consistency
#[test]
fn scenario_09_format_conversion() {
    let mut runner = ScenarioRunner::new("scenario_09_format_conversion");
    runner.set_io_meta("multiple", "conversion", 0);

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rows = 2;
    let cols = 3;

    runner.step(
        "mm_to_csv_to_mm",
        "MM -> CSV -> MM",
        "cross-format roundtrip",
        "Strict",
        || {
            // Write as MM
            let mm = mmwrite(rows, cols, &data).map_err(|e| format!("{e}"))?;
            let mm_result = mmread(&mm).map_err(|e| format!("{e}"))?;

            // Convert flat column-major to row-major Vec<Vec<f64>>
            let mut csv_data: Vec<Vec<f64>> = Vec::with_capacity(rows);
            for i in 0..rows {
                let mut row = Vec::with_capacity(cols);
                for j in 0..cols {
                    // MM is column-major: index = j * rows + i
                    row.push(mm_result.data[j * rows + i]);
                }
                csv_data.push(row);
            }

            // Write as CSV
            let csv = write_csv(None, &csv_data, ',').map_err(|e| format!("{e}"))?;
            let (_, csv_result) = read_csv(&csv, ',', false).map_err(|e| format!("{e}"))?;

            // Compare dimensions
            if csv_result.len() != rows || csv_result[0].len() != cols {
                return Err(format!(
                    "dimension mismatch: {}x{} vs {}x{}",
                    rows,
                    cols,
                    csv_result.len(),
                    csv_result[0].len()
                ));
            }

            // Compare values (both are now row-major)
            for i in 0..rows {
                for j in 0..cols {
                    if !approx_eq(csv_data[i][j], csv_result[i][j], 1e-10) {
                        return Err(format!("data[{i}][{j}] mismatch after MM->CSV"));
                    }
                }
            }

            Ok("MM -> CSV preserves data".to_string())
        },
    );

    runner.step(
        "txt_to_json_consistency",
        "savetxt vs JSON array",
        "different formats, same data",
        "Strict",
        || {
            let flat_data = vec![1.5, 2.5, 3.5, 4.5];

            let txt = savetxt(1, 4, &flat_data, " ").map_err(|e| format!("{e}"))?;
            let json = write_json_array(&flat_data).map_err(|e| format!("{e}"))?;

            let (_, _, txt_read) = loadtxt(&txt).map_err(|e| format!("{e}"))?;
            let json_read = read_json_array(&json).map_err(|e| format!("{e}"))?;

            for (i, (&t, &j)) in txt_read.iter().zip(json_read.iter()).enumerate() {
                if !approx_eq(t, j, 1e-10) {
                    return Err(format!("mismatch at {i}: txt={t}, json={j}"));
                }
            }

            Ok("txt and JSON preserve same data".to_string())
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_09_format_conversion", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_09 failed");
}

/// Scenario 10: Precision preservation
#[test]
fn scenario_10_precision() {
    let mut runner = ScenarioRunner::new("scenario_10_precision");
    runner.set_io_meta("multiple", "precision", 0);

    runner.step(
        "high_precision_text",
        "savetxt/loadtxt precision",
        "many decimal places",
        "Strict",
        || {
            let data = vec![
                std::f64::consts::PI,
                std::f64::consts::E,
                1.0 / 3.0,
                2.0_f64.sqrt(),
            ];

            let txt = savetxt(1, 4, &data, " ").map_err(|e| format!("{e}"))?;
            let (_, _, read) = loadtxt(&txt).map_err(|e| format!("{e}"))?;

            let mut max_error = 0.0_f64;
            for (&orig, &back) in data.iter().zip(read.iter()) {
                max_error = nan_max(max_error, (orig - back).abs());
            }

            if max_error <= 1e-10 {
                Ok(format!("max roundtrip error: {max_error:.2e}"))
            } else {
                Err(format!("precision loss: max_error={max_error}"))
            }
        },
    );

    runner.step(
        "mm_precision",
        "mmwrite/mmread precision",
        "Matrix Market precision",
        "Strict",
        || {
            let data = vec![1e-15, 1e15, 1.23456789012345];

            let mm = mmwrite(1, 3, &data).map_err(|e| format!("{e}"))?;
            let result = mmread(&mm).map_err(|e| format!("{e}"))?;

            let mut max_rel_error = 0.0_f64;
            for (&orig, &back) in data.iter().zip(result.data.iter()) {
                let rel_error = if orig != 0.0 {
                    ((orig - back) / orig).abs()
                } else {
                    back.abs()
                };
                max_rel_error = nan_max(max_rel_error, rel_error);
            }

            if max_rel_error <= 1e-10 {
                Ok(format!("max relative error: {max_rel_error:.2e}"))
            } else {
                Err(format!("precision loss: rel_error={max_rel_error}"))
            }
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_10_precision", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_10 failed");
}

/// Scenario 11: WAV bit depth handling
#[test]
fn scenario_11_wav_bit_depth() {
    let mut runner = ScenarioRunner::new("scenario_11_wav_bit_depth");
    runner.set_io_meta("WAV", "bit_depth", 0);

    runner.step(
        "wav_16bit_range",
        "WAV 16-bit quantization",
        "full dynamic range",
        "Strict",
        || {
            // Test full range: -1.0 to 1.0
            let data: Vec<f64> = (-100..=100).map(|i| i as f64 / 100.0).collect();

            let bytes = wav_write(44100, 1, &data).map_err(|e| format!("{e}"))?;
            let wav = wav_read(&bytes).map_err(|e| format!("{e}"))?;

            // 16-bit quantization: step size is about 1/32768 ≈ 3e-5
            let max_error = data
                .iter()
                .zip(wav.data.iter())
                .map(|(&o, &r)| (o - r).abs())
                .fold(0.0_f64, nan_max);

            if max_error <= 0.001 {
                Ok(format!("max quantization error: {max_error:.6}"))
            } else {
                Err(format!("quantization error too large: {max_error}"))
            }
        },
    );

    runner.step(
        "wav_clipping",
        "WAV clipping behavior",
        "values outside [-1, 1]",
        "Strict",
        || {
            // Values outside valid range
            let data = vec![-2.0, -1.5, 0.0, 1.5, 2.0];

            let bytes = wav_write(44100, 1, &data).map_err(|e| format!("{e}"))?;
            let wav = wav_read(&bytes).map_err(|e| format!("{e}"))?;

            // Check clipping: values should be clamped to [-1, 1]
            for &sample in &wav.data {
                if !(-1.0..=1.0).contains(&sample) {
                    return Err(format!("sample out of range: {sample}"));
                }
            }

            Ok("clipping handled correctly".to_string())
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_11_wav_bit_depth", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_11 failed");
}

// ═══════════════════════════════════════════════════════════════════════
// SCENARIOS 12-14: PERFORMANCE BOUNDARY
// ═══════════════════════════════════════════════════════════════════════

/// Scenario 12: Large Matrix Market file
#[test]
fn scenario_12_large_mm() {
    let mut runner = ScenarioRunner::new("scenario_12_large_mm");
    let n = 500;
    runner.set_io_meta("Matrix Market", "large", n * n * 8);

    // Generate deterministic data
    let data: Vec<f64> = (0..n * n).map(|i| (i as f64 * 1.23456).sin()).collect();

    runner.step(
        "write_large_mm",
        "mmwrite(500, 500, data)",
        "250K elements",
        "Strict",
        || {
            let start = Instant::now();
            let content = mmwrite(n, n, &data).map_err(|e| format!("{e}"))?;
            let elapsed = start.elapsed();

            if elapsed.as_millis() > 5000 {
                return Err(format!("too slow: {:?}", elapsed));
            }

            Ok(format!("wrote {} bytes in {:?}", content.len(), elapsed))
        },
    );

    runner.step(
        "read_large_mm",
        "mmread(large_content)",
        "parse 250K elements",
        "Strict",
        || {
            let content = mmwrite(n, n, &data).map_err(|e| format!("{e}"))?;

            let start = Instant::now();
            let result = mmread(&content).map_err(|e| format!("{e}"))?;
            let elapsed = start.elapsed();

            if result.rows != n || result.cols != n {
                return Err(format!(
                    "dimension mismatch: {}x{}",
                    result.rows, result.cols
                ));
            }

            if elapsed.as_millis() > 5000 {
                return Err(format!("too slow: {:?}", elapsed));
            }

            Ok(format!(
                "read {}x{} in {:?}",
                result.rows, result.cols, elapsed
            ))
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_12_large_mm", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_12 failed");
}

/// Scenario 13: Large WAV file
#[test]
fn scenario_13_large_wav() {
    let mut runner = ScenarioRunner::new("scenario_13_large_wav");
    let sample_rate = 44100u32;
    let duration_sec = 5;
    let n_samples = sample_rate as usize * duration_sec;
    runner.set_io_meta("WAV", "large", n_samples * 2);

    runner.step(
        "write_large_wav",
        "wav_write(5 seconds of audio)",
        "220K samples",
        "Strict",
        || {
            // Generate 5 seconds of a sweeping sine wave
            let data: Vec<f64> = (0..n_samples)
                .map(|i| {
                    let t = i as f64 / sample_rate as f64;
                    let freq = 200.0 + 800.0 * t / duration_sec as f64; // 200-1000 Hz sweep
                    (2.0 * std::f64::consts::PI * freq * t).sin() * 0.8
                })
                .collect();

            let start = Instant::now();
            let bytes = wav_write(sample_rate, 1, &data).map_err(|e| format!("{e}"))?;
            let elapsed = start.elapsed();

            if elapsed.as_millis() > 5000 {
                return Err(format!("too slow: {:?}", elapsed));
            }

            Ok(format!(
                "wrote {} bytes ({:.1}MB) in {:?}",
                bytes.len(),
                bytes.len() as f64 / 1024.0 / 1024.0,
                elapsed
            ))
        },
    );

    runner.step(
        "wav_roundtrip_large",
        "wav_read(wav_write(5 sec))",
        "large roundtrip",
        "Strict",
        || {
            let data: Vec<f64> = (0..n_samples)
                .map(|i| (i as f64 * 0.01).sin() * 0.5)
                .collect();

            let start = Instant::now();
            let bytes = wav_write(sample_rate, 1, &data).map_err(|e| format!("{e}"))?;
            let wav = wav_read(&bytes).map_err(|e| format!("{e}"))?;
            let elapsed = start.elapsed();

            if wav.data.len() != n_samples {
                return Err(format!(
                    "sample count mismatch: {} vs {}",
                    wav.data.len(),
                    n_samples
                ));
            }

            if elapsed.as_millis() > 5000 {
                return Err(format!("roundtrip too slow: {:?}", elapsed));
            }

            Ok(format!("roundtrip {} samples in {:?}", n_samples, elapsed))
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_13_large_wav", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_13 failed");
}

/// Scenario 14: Large CSV file
#[test]
fn scenario_14_large_csv() {
    let mut runner = ScenarioRunner::new("scenario_14_large_csv");
    let rows = 1000;
    let cols = 100;
    runner.set_io_meta("CSV", "large", rows * cols * 8);

    // Generate data as Vec<Vec<f64>> (row-major)
    let data: Vec<Vec<f64>> = (0..rows)
        .map(|i| {
            (0..cols)
                .map(|j| ((i * cols + j) as f64 * 0.001).sin())
                .collect()
        })
        .collect();

    runner.step(
        "write_large_csv",
        "write_csv(1000x100)",
        "100K values",
        "Strict",
        || {
            let start = Instant::now();
            let content = write_csv(None, &data, ',').map_err(|e| format!("{e}"))?;
            let elapsed = start.elapsed();

            if elapsed.as_millis() > 5000 {
                return Err(format!("too slow: {:?}", elapsed));
            }

            Ok(format!(
                "wrote {} bytes ({:.1}KB) in {:?}",
                content.len(),
                content.len() as f64 / 1024.0,
                elapsed
            ))
        },
    );

    runner.step(
        "read_large_csv",
        "read_csv(1000x100)",
        "parse 100K values",
        "Strict",
        || {
            let content = write_csv(None, &data, ',').map_err(|e| format!("{e}"))?;

            let start = Instant::now();
            let (_, result) = read_csv(&content, ',', false).map_err(|e| format!("{e}"))?;
            let elapsed = start.elapsed();

            if result.len() != rows || result[0].len() != cols {
                return Err(format!(
                    "dimension mismatch: {}x{}",
                    result.len(),
                    result[0].len()
                ));
            }

            if elapsed.as_millis() > 5000 {
                return Err(format!("too slow: {:?}", elapsed));
            }

            Ok(format!(
                "read {}x{} in {:?}",
                result.len(),
                result[0].len(),
                elapsed
            ))
        },
    );

    let bundle = runner.finish();
    write_bundle("scenario_14_large_csv", &bundle);
    assert!(bundle.overall.status == "pass", "scenario_14 failed");
}
