#![forbid(unsafe_code)]

//! Input/Output routines for FrankenSciPy.
//!
//! Matches `scipy.io` core functions:
//! - `savemat` / `loadmat` — MATLAB .mat file v4 real double matrix read/write
//! - `mmread` / `mmwrite` — Matrix Market format read/write
//! - `wavfile.read` / `wavfile.write` — WAV audio file read/write
//! - `netcdf_file` — NetCDF (simplified) read/write
//! - `readsav` — IDL SAVE scalar and primitive array read support

/// Error type for I/O operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IoError {
    InvalidFormat(String),
    IoFailed(String),
    UnsupportedFeature(String),
}

impl std::fmt::Display for IoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFormat(msg) => write!(f, "invalid format: {msg}"),
            Self::IoFailed(msg) => write!(f, "I/O failed: {msg}"),
            Self::UnsupportedFeature(msg) => write!(f, "unsupported: {msg}"),
        }
    }
}

impl std::error::Error for IoError {}

// ══════════════════════════════════════════════════════════════════════
// Matrix Market Format
// ══════════════════════════════════════════════════════════════════════

/// Matrix Market object type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmObject {
    Matrix,
    Vector,
}

/// Matrix Market format type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmFormat {
    Coordinate,
    Array,
}

/// Matrix Market field type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmField {
    Real,
    Integer,
    Complex,
    Pattern,
}

/// Matrix Market symmetry type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmSymmetry {
    General,
    Symmetric,
    SkewSymmetric,
    Hermitian,
}

/// Matrix Market header information.
#[derive(Debug, Clone)]
pub struct MmInfo {
    pub object: MmObject,
    pub format: MmFormat,
    pub field: MmField,
    pub symmetry: MmSymmetry,
    pub rows: usize,
    pub cols: usize,
    pub nnz: usize,
}

/// Dense matrix result from Matrix Market.
#[derive(Debug, Clone)]
pub struct MmMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
    pub complex_data: Option<Vec<(f64, f64)>>,
    pub info: MmInfo,
}

const MAX_MM_DENSE_ELEMENTS: usize = 128 * 1024 * 1024;

fn checked_matrix_len(rows: usize, cols: usize, context: &str) -> Result<usize, IoError> {
    rows.checked_mul(cols).ok_or_else(|| {
        IoError::InvalidFormat(format!(
            "{context} dimensions {rows}x{cols} overflowed usize"
        ))
    })
}

fn checked_mm_dense_read_len(rows: usize, cols: usize) -> Result<usize, IoError> {
    let dense_len = checked_matrix_len(rows, cols, "Matrix Market matrix")?;
    if dense_len > MAX_MM_DENSE_ELEMENTS {
        return Err(IoError::InvalidFormat(format!(
            "Matrix Market matrix dimensions {rows}x{cols} exceed dense read safety bound of {MAX_MM_DENSE_ELEMENTS} elements"
        )));
    }
    Ok(dense_len)
}

fn parse_mm_info(lines: &mut std::str::Lines<'_>) -> Result<MmInfo, IoError> {
    let header = lines
        .next()
        .ok_or_else(|| IoError::InvalidFormat("empty file".to_string()))?;
    if !header.starts_with("%%MatrixMarket") {
        return Err(IoError::InvalidFormat(
            "missing %%MatrixMarket header".to_string(),
        ));
    }

    let parts: Vec<&str> = header.split_whitespace().collect();
    if parts.len() < 5 {
        return Err(IoError::InvalidFormat("incomplete header line".to_string()));
    }

    let object = match parts[1].to_lowercase().as_str() {
        "matrix" => MmObject::Matrix,
        "vector" => MmObject::Vector,
        other => {
            return Err(IoError::InvalidFormat(format!(
                "unknown object type: {other}"
            )));
        }
    };

    let format = match parts[2].to_lowercase().as_str() {
        "coordinate" => MmFormat::Coordinate,
        "array" => MmFormat::Array,
        other => return Err(IoError::InvalidFormat(format!("unknown format: {other}"))),
    };

    let field = match parts[3].to_lowercase().as_str() {
        "real" => MmField::Real,
        "integer" => MmField::Integer,
        "complex" => MmField::Complex,
        "pattern" => MmField::Pattern,
        other => {
            return Err(IoError::InvalidFormat(format!(
                "unknown field type: {other}"
            )));
        }
    };

    let symmetry = match parts[4].to_lowercase().as_str() {
        "general" => MmSymmetry::General,
        "symmetric" => MmSymmetry::Symmetric,
        "skew-symmetric" => MmSymmetry::SkewSymmetric,
        "hermitian" => MmSymmetry::Hermitian,
        other => return Err(IoError::InvalidFormat(format!("unknown symmetry: {other}"))),
    };

    let mut size_line = None;
    for line in lines.by_ref() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('%') {
            continue;
        }
        size_line = Some(trimmed.to_string());
        break;
    }

    let size_str =
        size_line.ok_or_else(|| IoError::InvalidFormat("missing size line".to_string()))?;
    let size_parts: Vec<&str> = size_str.split_whitespace().collect();

    match format {
        MmFormat::Coordinate => {
            if size_parts.len() < 3 {
                return Err(IoError::InvalidFormat(
                    "coordinate format requires rows cols nnz".to_string(),
                ));
            }
            let rows: usize = size_parts[0]
                .parse()
                .map_err(|e| IoError::InvalidFormat(format!("bad rows: {e}")))?;
            let cols: usize = size_parts[1]
                .parse()
                .map_err(|e| IoError::InvalidFormat(format!("bad cols: {e}")))?;
            let nnz: usize = size_parts[2]
                .parse()
                .map_err(|e| IoError::InvalidFormat(format!("bad nnz: {e}")))?;

            Ok(MmInfo {
                object,
                format,
                field,
                symmetry,
                rows,
                cols,
                nnz,
            })
        }
        MmFormat::Array => {
            if size_parts.len() < 2 {
                return Err(IoError::InvalidFormat(
                    "array format requires rows cols".to_string(),
                ));
            }
            let rows: usize = size_parts[0]
                .parse()
                .map_err(|e| IoError::InvalidFormat(format!("bad rows: {e}")))?;
            let cols: usize = size_parts[1]
                .parse()
                .map_err(|e| IoError::InvalidFormat(format!("bad cols: {e}")))?;
            let nnz = rows.checked_mul(cols).ok_or_else(|| {
                IoError::InvalidFormat("array dimensions overflowed nnz computation".to_string())
            })?;

            Ok(MmInfo {
                object,
                format,
                field,
                symmetry,
                rows,
                cols,
                nnz,
            })
        }
    }
}

/// Read a Matrix Market file.
///
/// Matches `scipy.io.mmread`.
pub fn mmread(content: &str) -> Result<MmMatrix, IoError> {
    let mut lines = content.lines();
    let info = parse_mm_info(&mut lines)?;

    if info.field == MmField::Complex {
        return Err(IoError::UnsupportedFeature(
            "Matrix Market complex field is not supported".to_string(),
        ));
    }
    if info.symmetry != MmSymmetry::General && info.rows != info.cols {
        let symmetry = match info.symmetry {
            MmSymmetry::General => "general",
            MmSymmetry::Symmetric => "symmetric",
            MmSymmetry::SkewSymmetric => "skew-symmetric",
            MmSymmetry::Hermitian => "hermitian",
        };
        return Err(IoError::InvalidFormat(format!(
            "Matrix Market {symmetry} symmetry requires a square matrix, got {}x{}",
            info.rows, info.cols
        )));
    }

    match info.format {
        MmFormat::Coordinate => {
            let rows = info.rows;
            let cols = info.cols;
            let nnz = info.nnz;
            let dense_len = checked_mm_dense_read_len(rows, cols)?;
            let mut data = vec![0.0; dense_len];
            let mut complex_data: Option<Vec<(f64, f64)>> = None;
            let mut seen_nnz = 0usize;

            for line in lines {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('%') {
                    continue;
                }
                let vals: Vec<&str> = trimmed.split_whitespace().collect();
                if vals.len() < 2 {
                    continue;
                }
                let r_one_based = vals[0]
                    .parse::<usize>()
                    .map_err(|e| IoError::InvalidFormat(format!("bad row index: {e}")))?;
                let c_one_based = vals[1]
                    .parse::<usize>()
                    .map_err(|e| IoError::InvalidFormat(format!("bad col index: {e}")))?;
                let r = r_one_based.checked_sub(1).ok_or_else(|| {
                    IoError::InvalidFormat(
                        "Matrix Market row indices must be 1-based and >= 1".to_string(),
                    )
                })?;
                let c = c_one_based.checked_sub(1).ok_or_else(|| {
                    IoError::InvalidFormat(
                        "Matrix Market col indices must be 1-based and >= 1".to_string(),
                    )
                })?;
                let v: f64;
                if info.field == MmField::Pattern {
                    v = 1.0;
                } else if vals.len() >= 3 {
                    v = vals[2]
                        .parse()
                        .map_err(|e| IoError::InvalidFormat(format!("bad value: {e}")))?;
                } else {
                    return Err(IoError::InvalidFormat(
                        "coordinate entry missing value for non-pattern field".to_string(),
                    ));
                }
                let v_im = 0.0;

                if r >= rows || c >= cols {
                    return Err(IoError::InvalidFormat(format!(
                        "coordinate entry ({r}, {c}) out of bounds for {rows}x{cols}"
                    )));
                }

                let add_val = |cd: &mut Option<Vec<(f64, f64)>>,
                               d: &mut Vec<f64>,
                               r: usize,
                               c: usize,
                               vr: f64,
                               vi: f64| {
                    let i = r * cols + c;
                    if let Some(cdata) = cd {
                        cdata[i].0 += vr;
                        cdata[i].1 += vi;
                    } else {
                        d[i] += vr;
                    }
                };
                let sub_val = |cd: &mut Option<Vec<(f64, f64)>>,
                               d: &mut Vec<f64>,
                               r: usize,
                               c: usize,
                               vr: f64,
                               vi: f64| {
                    let i = r * cols + c;
                    if let Some(cdata) = cd {
                        cdata[i].0 -= vr;
                        cdata[i].1 -= vi;
                    } else {
                        d[i] -= vr;
                    }
                };

                match info.symmetry {
                    MmSymmetry::General => {
                        add_val(&mut complex_data, &mut data, r, c, v, v_im);
                    }
                    MmSymmetry::Symmetric | MmSymmetry::Hermitian => {
                        add_val(&mut complex_data, &mut data, r, c, v, v_im);
                        if r != c {
                            if info.symmetry == MmSymmetry::Hermitian {
                                add_val(&mut complex_data, &mut data, c, r, v, -v_im);
                            } else {
                                add_val(&mut complex_data, &mut data, c, r, v, v_im);
                            }
                        }
                    }
                    MmSymmetry::SkewSymmetric => {
                        if r == c {
                            if v != 0.0 || v_im != 0.0 {
                                return Err(IoError::InvalidFormat(
                                    "skew-symmetric diagonal entries must be zero".to_string(),
                                ));
                            }
                        } else {
                            add_val(&mut complex_data, &mut data, r, c, v, v_im);
                            sub_val(&mut complex_data, &mut data, c, r, v, v_im);
                        }
                    }
                }
                seen_nnz += 1;
            }

            if seen_nnz != nnz {
                return Err(IoError::InvalidFormat(format!(
                    "coordinate format expected {nnz} entries but found {seen_nnz}"
                )));
            }

            Ok(MmMatrix {
                rows,
                cols,
                data,
                complex_data,
                info,
            })
        }
        MmFormat::Array => {
            let rows = info.rows;
            let cols = info.cols;
            let dense_len = checked_mm_dense_read_len(rows, cols)?;
            // Array format: column-major order
            let mut data = vec![0.0; dense_len];
            let mut complex_data: Option<Vec<(f64, f64)>> = None;
            let mut idx = 0;

            for line in lines {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('%') {
                    continue;
                }
                if idx >= dense_len {
                    return Err(IoError::InvalidFormat(format!(
                        "array format has more than the declared {} values",
                        dense_len
                    )));
                }
                let v = trimmed
                    .parse()
                    .map_err(|e| IoError::InvalidFormat(format!("bad value: {e}")))?;
                let v_im = 0.0;

                // Column-major to row-major conversion
                let col = idx / rows;
                let row = idx % rows;
                if row < rows && col < cols {
                    if let Some(ref mut cd) = complex_data {
                        cd[row * cols + col] = (v, v_im);
                    } else {
                        data[row * cols + col] = v;
                    }
                }
                idx += 1;
            }
            if idx != dense_len {
                return Err(IoError::InvalidFormat(format!(
                    "array format expected {} values but found {idx}",
                    dense_len
                )));
            }

            Ok(MmMatrix {
                rows,
                cols,
                data,
                complex_data,
                info,
            })
        }
    }
}

/// Write a dense matrix in Matrix Market format.
///
/// Matches `scipy.io.mmwrite`.
pub fn mmwrite(rows: usize, cols: usize, data: &[f64]) -> Result<String, IoError> {
    let expected_len = checked_matrix_len(rows, cols, "Matrix Market matrix")?;
    if data.len() != expected_len {
        return Err(IoError::InvalidFormat(format!(
            "data length {} doesn't match {}x{}",
            data.len(),
            rows,
            cols
        )));
    }

    let mut out = String::new();
    out.push_str("%%MatrixMarket matrix array real general\n");
    out.push_str(&format!("{rows} {cols}\n"));

    // Column-major order (Matrix Market convention)
    for c in 0..cols {
        for r in 0..rows {
            let v = data[r * cols + c];
            out.push_str(&format!("{v}\n"));
        }
    }

    Ok(out)
}

/// Write a sparse matrix in coordinate Matrix Market format.
pub fn mmwrite_sparse(
    rows: usize,
    cols: usize,
    entries: &[(usize, usize, f64)],
) -> Result<String, IoError> {
    let mut out = String::new();
    out.push_str("%%MatrixMarket matrix coordinate real general\n");
    out.push_str(&format!("{rows} {cols} {}\n", entries.len()));

    for &(r, c, v) in entries {
        if r >= rows || c >= cols {
            return Err(IoError::InvalidFormat(format!(
                "sparse entry ({r}, {c}) out of bounds for {rows}x{cols}"
            )));
        }
        let row = r.checked_add(1).ok_or_else(|| {
            IoError::InvalidFormat("sparse row index overflowed Matrix Market encoding".to_string())
        })?;
        let col = c.checked_add(1).ok_or_else(|| {
            IoError::InvalidFormat("sparse col index overflowed Matrix Market encoding".to_string())
        })?;
        out.push_str(&format!("{row} {col} {v}\n"));
    }

    Ok(out)
}

/// Write a complex dense matrix in Matrix Market format.
pub fn mmwrite_complex(rows: usize, cols: usize, data: &[(f64, f64)]) -> Result<String, IoError> {
    let expected_len = checked_matrix_len(rows, cols, "Matrix Market matrix")?;
    if data.len() != expected_len {
        return Err(IoError::InvalidFormat(format!(
            "data length {} doesn't match {}x{}",
            data.len(),
            rows,
            cols
        )));
    }

    let mut out = String::new();
    out.push_str("%%MatrixMarket matrix array complex general\n");
    out.push_str(&format!("{rows} {cols}\n"));

    // Column-major order (Matrix Market convention)
    for c in 0..cols {
        for r in 0..rows {
            let (vr, vi) = data[r * cols + c];
            out.push_str(&format!("{vr} {vi}\n"));
        }
    }

    Ok(out)
}

/// Write a complex sparse matrix in coordinate Matrix Market format.
pub fn mmwrite_sparse_complex(
    rows: usize,
    cols: usize,
    entries: &[(usize, usize, (f64, f64))],
) -> Result<String, IoError> {
    let mut out = String::new();
    out.push_str("%%MatrixMarket matrix coordinate complex general\n");
    out.push_str(&format!("{rows} {cols} {}\n", entries.len()));

    for &(r, c, (vr, vi)) in entries {
        if r >= rows || c >= cols {
            return Err(IoError::InvalidFormat(format!(
                "sparse entry ({r}, {c}) out of bounds for {rows}x{cols}"
            )));
        }
        let row = r.checked_add(1).ok_or_else(|| {
            IoError::InvalidFormat("sparse row index overflowed Matrix Market encoding".to_string())
        })?;
        let col = c.checked_add(1).ok_or_else(|| {
            IoError::InvalidFormat("sparse col index overflowed Matrix Market encoding".to_string())
        })?;
        out.push_str(&format!("{row} {col} {vr} {vi}\n"));
    }

    Ok(out)
}

/// Read Matrix Market info (header only).
///
/// Matches `scipy.io.mminfo`.
pub fn mminfo(content: &str) -> Result<MmInfo, IoError> {
    let mut lines = content.lines();
    parse_mm_info(&mut lines)
}

// ══════════════════════════════════════════════════════════════════════
// WAV File Format
// ══════════════════════════════════════════════════════════════════════

/// WAV file data.
#[derive(Debug, Clone)]
pub struct WavData {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub data: Vec<f64>,
}

/// Read a WAV file from bytes.
///
/// Matches `scipy.io.wavfile.read`.
pub fn wav_read(bytes: &[u8]) -> Result<WavData, IoError> {
    if bytes.len() < 44 {
        return Err(IoError::InvalidFormat("WAV file too short".to_string()));
    }

    // RIFF header
    if &bytes[0..4] != b"RIFF" {
        return Err(IoError::InvalidFormat("missing RIFF header".to_string()));
    }
    if &bytes[8..12] != b"WAVE" {
        return Err(IoError::InvalidFormat(
            "missing WAVE identifier".to_string(),
        ));
    }

    // Find fmt chunk
    let mut pos = 12;
    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    let mut bits_per_sample = 0u16;
    let mut audio_format = 0u16;

    while pos + 8 <= bytes.len() {
        let chunk_id = &bytes[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;

        if chunk_id == b"fmt " {
            if chunk_size < 16 || pos + 8 + chunk_size > bytes.len() {
                return Err(IoError::InvalidFormat("fmt chunk too small".to_string()));
            }
            let fmt = &bytes[pos + 8..];
            audio_format = u16::from_le_bytes([fmt[0], fmt[1]]);
            channels = u16::from_le_bytes([fmt[2], fmt[3]]);
            if channels == 0 {
                return Err(IoError::InvalidFormat(
                    "fmt chunk declares zero channels".to_string(),
                ));
            }
            sample_rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
            bits_per_sample = u16::from_le_bytes([fmt[14], fmt[15]]);
        } else if chunk_id == b"data" {
            if pos + 8 + chunk_size > bytes.len() {
                return Err(IoError::InvalidFormat(
                    "data chunk extends past file".to_string(),
                ));
            }
            let data_bytes = &bytes[pos + 8..pos + 8 + chunk_size];
            if sample_rate == 0 || channels == 0 || bits_per_sample == 0 {
                return Err(IoError::InvalidFormat(
                    "encountered data chunk before a valid fmt chunk".to_string(),
                ));
            }

            if audio_format != 1 && audio_format != 3 {
                return Err(IoError::UnsupportedFeature(format!(
                    "unsupported audio format: {audio_format} (only PCM=1 and IEEE_FLOAT=3 supported)"
                )));
            }
            if audio_format == 3 && bits_per_sample != 32 {
                return Err(IoError::UnsupportedFeature(format!(
                    "unsupported IEEE float bits per sample: {bits_per_sample}"
                )));
            }

            let bytes_per_sample = match bits_per_sample {
                8 => 1usize,
                16 => 2,
                24 => 3,
                32 => 4,
                _ => {
                    return Err(IoError::UnsupportedFeature(format!(
                        "unsupported bits per sample: {bits_per_sample}"
                    )));
                }
            };
            if !data_bytes.len().is_multiple_of(bytes_per_sample) {
                return Err(IoError::InvalidFormat(format!(
                    "data chunk size {} is not aligned to {}-byte samples",
                    data_bytes.len(),
                    bytes_per_sample
                )));
            }
            let frame_bytes = bytes_per_sample
                .checked_mul(channels as usize)
                .ok_or_else(|| {
                    IoError::InvalidFormat("WAV frame size overflowed usize".to_string())
                })?;
            if !data_bytes.len().is_multiple_of(frame_bytes) {
                return Err(IoError::InvalidFormat(format!(
                    "data chunk size {} does not contain whole {}-channel frames",
                    data_bytes.len(),
                    channels
                )));
            }

            let samples = match (bits_per_sample, audio_format) {
                (8, _) => data_bytes
                    .iter()
                    .map(|&b| (b as f64 - 128.0) / 128.0)
                    .collect(),
                (16, _) => data_bytes
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]) as f64 / 32768.0)
                    .collect(),
                (24, _) => data_bytes
                    .chunks_exact(3)
                    .map(|c| {
                        let sign = if c[2] & 0x80 != 0 { 0xFF } else { 0x00 };
                        let raw = i32::from_le_bytes([c[0], c[1], c[2], sign]);
                        raw as f64 / 8_388_608.0
                    })
                    .collect(),
                (32, 3) => data_bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
                    .collect(),
                (32, _) => data_bytes
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64 / 2_147_483_648.0)
                    .collect(),
                _ => {
                    return Err(IoError::UnsupportedFeature(format!(
                        "unsupported bits per sample: {bits_per_sample}"
                    )));
                }
            };

            return Ok(WavData {
                sample_rate,
                channels,
                bits_per_sample,
                data: samples,
            });
        }

        pos += 8 + chunk_size;
        // Chunks are word-aligned
        if !chunk_size.is_multiple_of(2) {
            pos += 1;
        }
    }

    Err(IoError::InvalidFormat("no data chunk found".to_string()))
}

/// Write WAV file data as bytes (16-bit PCM).
///
/// Matches `scipy.io.wavfile.write`.
pub fn wav_write(sample_rate: u32, channels: u16, data: &[f64]) -> Result<Vec<u8>, IoError> {
    if sample_rate == 0 {
        return Err(IoError::InvalidFormat(
            "WAV sample rate must be nonzero".to_string(),
        ));
    }
    if channels == 0 {
        return Err(IoError::InvalidFormat(
            "WAV channel count must be nonzero".to_string(),
        ));
    }
    if !data.len().is_multiple_of(channels as usize) {
        return Err(IoError::InvalidFormat(format!(
            "data length {} does not contain whole frames for {channels} channels",
            data.len()
        )));
    }
    let bits_per_sample: u16 = 16;
    let bytes_per_sample = bits_per_sample / 8;
    let data_size = data
        .len()
        .checked_mul(bytes_per_sample as usize)
        .and_then(|size| u32::try_from(size).ok())
        .ok_or_else(|| IoError::InvalidFormat("WAV data chunk too large".to_string()))?;
    let file_size = 36u32
        .checked_add(data_size)
        .ok_or_else(|| IoError::InvalidFormat("WAV file too large".to_string()))?;

    let mut buf = Vec::with_capacity(file_size as usize + 8);

    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // fmt chunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    buf.extend_from_slice(&channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    let byte_rate = sample_rate
        .checked_mul(channels as u32)
        .and_then(|rate| rate.checked_mul(bytes_per_sample as u32))
        .ok_or_else(|| IoError::InvalidFormat("WAV byte rate overflowed u32".to_string()))?;
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    let block_align = channels
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| IoError::InvalidFormat("WAV block align overflowed u16".to_string()))?;
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());

    for &sample in data {
        let clamped = sample.clamp(-1.0, 1.0);
        let val = (clamped * 32767.0) as i16;
        buf.extend_from_slice(&val.to_le_bytes());
    }

    Ok(buf)
}

// ══════════════════════════════════════════════════════════════════════
// MAT-file v4 (simple numeric arrays)
// ══════════════════════════════════════════════════════════════════════

/// A named array loaded from a MAT file.
#[derive(Debug, Clone)]
pub struct MatArray {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

const MAT4_MI_DOUBLE: i32 = 0;
const MAT4_MX_FULL_CLASS: i32 = 0;
const MAT4_MAX_ELEMENTS: usize = 128 * 1024 * 1024;

fn checked_mat_dense_len(rows: usize, cols: usize) -> Result<usize, IoError> {
    let len = checked_matrix_len(rows, cols, "MAT v4 matrix")?;
    if len > MAT4_MAX_ELEMENTS {
        return Err(IoError::InvalidFormat(format!(
            "MAT v4 matrix dimensions {rows}x{cols} exceed dense read safety bound of {MAT4_MAX_ELEMENTS} elements"
        )));
    }
    Ok(len)
}

fn validate_mat_array(arr: &MatArray) -> Result<usize, IoError> {
    if arr.name.is_empty() {
        return Err(IoError::InvalidFormat(
            "MAT v4 variable name cannot be empty".to_string(),
        ));
    }
    if arr.name.contains('\0') {
        return Err(IoError::InvalidFormat(format!(
            "array name '{}' contains a NUL byte and cannot be encoded safely",
            arr.name.escape_debug()
        )));
    }
    if arr.name.chars().any(|ch| u32::from(ch) > 0xff) {
        return Err(IoError::UnsupportedFeature(format!(
            "array name '{}' is not Latin-1 encodable for MAT v4",
            arr.name.escape_debug()
        )));
    }
    i32::try_from(arr.rows).map_err(|_| {
        IoError::InvalidFormat(format!(
            "array '{}' row count {} exceeds MAT v4 i32 header range",
            arr.name, arr.rows
        ))
    })?;
    i32::try_from(arr.cols).map_err(|_| {
        IoError::InvalidFormat(format!(
            "array '{}' column count {} exceeds MAT v4 i32 header range",
            arr.name, arr.cols
        ))
    })?;
    let expected_len = checked_mat_dense_len(arr.rows, arr.cols)?;
    if arr.data.len() != expected_len {
        return Err(IoError::InvalidFormat(format!(
            "array '{}' expected {} values but found {}",
            arr.name,
            expected_len,
            arr.data.len()
        )));
    }
    Ok(expected_len)
}

fn mat4_name_bytes(name: &str) -> Result<Vec<u8>, IoError> {
    let mut bytes = Vec::with_capacity(name.len() + 1);
    for ch in name.chars() {
        if u32::from(ch) > 0xff {
            return Err(IoError::UnsupportedFeature(format!(
                "array name '{}' is not Latin-1 encodable for MAT v4",
                name.escape_debug()
            )));
        }
        bytes.push(ch as u8);
    }
    bytes.push(0);
    Ok(bytes)
}

fn read_mat4_i32(bytes: &[u8], offset: &mut usize, field: &str) -> Result<i32, IoError> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| IoError::InvalidFormat(format!("MAT v4 {field} offset overflowed usize")))?;
    let slice = bytes.get(*offset..end).ok_or_else(|| {
        IoError::InvalidFormat(format!("truncated MAT v4 header while reading {field}"))
    })?;
    *offset = end;
    Ok(i32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_mat4_f64(bytes: &[u8], offset: &mut usize, name: &str) -> Result<f64, IoError> {
    let end = offset.checked_add(8).ok_or_else(|| {
        IoError::InvalidFormat(format!("MAT v4 data offset overflowed for '{name}'"))
    })?;
    let slice = bytes.get(*offset..end).ok_or_else(|| {
        IoError::InvalidFormat(format!("truncated MAT v4 data payload for '{name}'"))
    })?;
    *offset = end;
    Ok(f64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

fn mat4_nonnegative_usize(value: i32, field: &str) -> Result<usize, IoError> {
    if value < 0 {
        return Err(IoError::InvalidFormat(format!(
            "MAT v4 {field} cannot be negative: {value}"
        )));
    }
    usize::try_from(value).map_err(|_| {
        IoError::InvalidFormat(format!(
            "MAT v4 {field} {value} cannot be represented as usize"
        ))
    })
}

/// Save arrays to MATLAB MAT-file Level 4 bytes.
///
/// This intentionally supports the SciPy-compatible `format="4"` subset for
/// full real double matrices. Structs, cells, sparse matrices, complex values,
/// character arrays, compression, and MAT v5/v7.3 are outside this narrow
/// contract and fail closed elsewhere.
pub fn savemat(arrays: &[MatArray]) -> Result<Vec<u8>, IoError> {
    let mut out = Vec::new();
    for arr in arrays {
        validate_mat_array(arr)?;
        let name = mat4_name_bytes(&arr.name)?;
        let name_len = i32::try_from(name.len()).map_err(|_| {
            IoError::InvalidFormat(format!(
                "array '{}' name length {} exceeds MAT v4 i32 header range",
                arr.name,
                name.len()
            ))
        })?;
        let rows = i32::try_from(arr.rows).map_err(|_| {
            IoError::InvalidFormat(format!(
                "array '{}' row count {} exceeds MAT v4 i32 header range",
                arr.name, arr.rows
            ))
        })?;
        let cols = i32::try_from(arr.cols).map_err(|_| {
            IoError::InvalidFormat(format!(
                "array '{}' column count {} exceeds MAT v4 i32 header range",
                arr.name, arr.cols
            ))
        })?;

        out.extend_from_slice(&0i32.to_le_bytes());
        out.extend_from_slice(&rows.to_le_bytes());
        out.extend_from_slice(&cols.to_le_bytes());
        out.extend_from_slice(&0i32.to_le_bytes());
        out.extend_from_slice(&name_len.to_le_bytes());
        out.extend_from_slice(&name);
        for col in 0..arr.cols {
            for row in 0..arr.rows {
                out.extend_from_slice(&arr.data[row * arr.cols + col].to_le_bytes());
            }
        }
    }
    Ok(out)
}

/// Load arrays from MATLAB MAT-file Level 4 bytes.
///
/// The returned `MatArray::data` uses the same row-major layout as the rest of
/// `fsci-io`; MAT v4 stores full matrices in column-major order on disk.
pub fn loadmat(bytes: &[u8]) -> Result<Vec<MatArray>, IoError> {
    let mut arrays = Vec::new();
    let mut offset = 0usize;
    while offset < bytes.len() {
        let mopt = read_mat4_i32(bytes, &mut offset, "mopt")?;
        let rows_raw = read_mat4_i32(bytes, &mut offset, "mrows")?;
        let cols_raw = read_mat4_i32(bytes, &mut offset, "ncols")?;
        let imagf = read_mat4_i32(bytes, &mut offset, "imagf")?;
        let name_len_raw = read_mat4_i32(bytes, &mut offset, "namlen")?;

        if !(0..=5000).contains(&mopt) {
            return Err(IoError::InvalidFormat(format!(
                "MAT v4 mopt {mopt} is outside the supported header range"
            )));
        }
        let order = mopt / 1000;
        let after_order = mopt % 1000;
        let unused = after_order / 100;
        let after_unused = after_order % 100;
        let data_type = after_unused / 10;
        let matrix_class = after_unused % 10;
        if order != 0 {
            return Err(IoError::UnsupportedFeature(
                "MAT v4 big-endian variables are not supported".to_string(),
            ));
        }
        if unused != 0 {
            return Err(IoError::InvalidFormat(format!(
                "MAT v4 mopt reserved O field must be 0, got {unused}"
            )));
        }
        if data_type != MAT4_MI_DOUBLE || matrix_class != MAT4_MX_FULL_CLASS {
            return Err(IoError::UnsupportedFeature(format!(
                "only MAT v4 full real double matrices are supported (P={data_type}, T={matrix_class})"
            )));
        }
        if imagf != 0 {
            return Err(IoError::UnsupportedFeature(
                "MAT v4 complex matrices are not supported".to_string(),
            ));
        }

        let rows = mat4_nonnegative_usize(rows_raw, "mrows")?;
        let cols = mat4_nonnegative_usize(cols_raw, "ncols")?;
        let name_len = mat4_nonnegative_usize(name_len_raw, "namlen")?;
        if name_len == 0 {
            return Err(IoError::InvalidFormat(
                "MAT v4 variable name length cannot be zero".to_string(),
            ));
        }
        let name_end = offset.checked_add(name_len).ok_or_else(|| {
            IoError::InvalidFormat("MAT v4 name offset overflowed usize".to_string())
        })?;
        let name_bytes = bytes
            .get(offset..name_end)
            .ok_or_else(|| IoError::InvalidFormat("truncated MAT v4 variable name".to_string()))?;
        offset = name_end;
        let trimmed_name_len = name_bytes
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(name_bytes.len());
        if trimmed_name_len == 0 {
            return Err(IoError::InvalidFormat(
                "MAT v4 variable name cannot be empty".to_string(),
            ));
        }
        let name: String = name_bytes[..trimmed_name_len]
            .iter()
            .map(|&byte| char::from(byte))
            .collect();

        let expected_len = checked_mat_dense_len(rows, cols)?;
        let mut data = vec![0.0; expected_len];
        for col in 0..cols {
            for row in 0..rows {
                data[row * cols + col] = read_mat4_f64(bytes, &mut offset, &name)?;
            }
        }
        arrays.push(MatArray {
            name,
            rows,
            cols,
            data,
        });
    }
    Ok(arrays)
}

/// Save arrays to a simple text-based format (similar to MATLAB ASCII).
///
/// This provides a basic `savemat`-like interface. Full .mat v5 binary
/// format requires extensive implementation; this provides a portable
/// text alternative.
pub fn savemat_text(arrays: &[MatArray]) -> Result<String, IoError> {
    let mut out = String::new();
    for arr in arrays {
        if arr.name.contains(['\n', '\r']) {
            return Err(IoError::InvalidFormat(format!(
                "array name '{}' contains a newline and cannot be encoded safely",
                arr.name.escape_debug()
            )));
        }
        let expected_len = arr.rows.checked_mul(arr.cols).ok_or_else(|| {
            IoError::InvalidFormat(format!(
                "array '{}' dimensions {}x{} overflow usize",
                arr.name, arr.rows, arr.cols
            ))
        })?;
        if arr.data.len() != expected_len {
            return Err(IoError::InvalidFormat(format!(
                "array '{}' expected {} values but found {}",
                arr.name,
                expected_len,
                arr.data.len()
            )));
        }
        out.push_str(&format!(
            "# name: {}\n# type: matrix\n# rows: {}\n# columns: {}\n",
            arr.name, arr.rows, arr.cols
        ));
        for r in 0..arr.rows {
            for c in 0..arr.cols {
                if c > 0 {
                    out.push(' ');
                }
                out.push_str(&format!("{}", arr.data[r * arr.cols + c]));
            }
            out.push('\n');
        }
        out.push('\n');
    }
    Ok(out)
}

/// Load arrays from the text-based format.
pub fn loadmat_text(content: &str) -> Result<Vec<MatArray>, IoError> {
    let mut arrays = Vec::new();
    let mut lines = content.lines().peekable();

    while lines.peek().is_some() {
        // Find "# name:" line
        let mut name = None;
        let mut rows = 0usize;
        let mut cols = 0usize;

        loop {
            match lines.next() {
                None => {
                    if name.is_some() || rows != 0 || cols != 0 {
                        return Err(IoError::InvalidFormat(
                            "incomplete MAT text block at end of file".to_string(),
                        ));
                    }
                    return Ok(arrays);
                }
                Some(line) => {
                    let trimmed = line.trim();
                    if let Some(stripped) = trimmed.strip_prefix("# name:") {
                        name = Some(stripped.trim().to_string());
                    } else if let Some(stripped) = trimmed.strip_prefix("# rows:") {
                        rows = stripped
                            .trim()
                            .parse()
                            .map_err(|e| IoError::InvalidFormat(format!("bad rows: {e}")))?;
                    } else if let Some(stripped) = trimmed.strip_prefix("# columns:") {
                        cols = stripped
                            .trim()
                            .parse()
                            .map_err(|e| IoError::InvalidFormat(format!("bad cols: {e}")))?;
                    } else if trimmed.starts_with("# type:") {
                        // Skip type line
                    } else if !trimmed.is_empty() && !trimmed.starts_with('#') {
                        // Data line — we've hit the matrix data
                        let n = name.as_ref().ok_or_else(|| {
                            IoError::InvalidFormat(
                                "encountered matrix data before '# name:' header".to_string(),
                            )
                        })?;
                        if rows == 0 || cols == 0 {
                            return Err(IoError::InvalidFormat(format!(
                                "array '{n}' is missing nonzero '# rows:' and '# columns:' headers before data"
                            )));
                        }
                        let expected_len = checked_matrix_len(rows, cols, "MAT text matrix")?;
                        let mut data = Vec::with_capacity(expected_len);
                        let parse_row = |line: &str| -> Result<Vec<f64>, IoError> {
                            line.split_whitespace()
                                .map(|val_str| {
                                    val_str.parse::<f64>().map_err(|e| {
                                        IoError::InvalidFormat(format!("bad value: {e}"))
                                    })
                                })
                                .collect()
                        };
                        let first_vals = parse_row(trimmed)?;
                        if first_vals.len() != cols {
                            return Err(IoError::InvalidFormat(format!(
                                "array '{n}' row 0 has {} columns, expected {cols}",
                                first_vals.len()
                            )));
                        }
                        data.extend_from_slice(&first_vals);
                        // Read remaining rows
                        for row_idx in 1..rows {
                            let line = lines.next().ok_or_else(|| {
                                IoError::InvalidFormat(format!(
                                    "array '{n}' expected {rows} rows but found {row_idx}"
                                ))
                            })?;
                            let row_vals = parse_row(line)?;
                            if row_vals.len() != cols {
                                return Err(IoError::InvalidFormat(format!(
                                    "array '{n}' row {row_idx} has {} columns, expected {cols}",
                                    row_vals.len()
                                )));
                            }
                            data.extend_from_slice(&row_vals);
                        }
                        if data.len() != expected_len {
                            return Err(IoError::InvalidFormat(format!(
                                "array '{n}' expected {} values but found {}",
                                expected_len,
                                data.len()
                            )));
                        }
                        arrays.push(MatArray {
                            name: n.clone(),
                            rows,
                            cols,
                            data,
                        });
                        break;
                    }
                }
            }
        }
    }

    Ok(arrays)
}

// ══════════════════════════════════════════════════════════════════════
// IDL SAVE files
// ══════════════════════════════════════════════════════════════════════

const IDL_MAX_ARRAY_ELEMENTS: usize = 64 * 1024 * 1024;

/// Primitive IDL SAVE type codes supported by `read_idl_save`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdlType {
    Byte,
    Int16,
    Int32,
    Float32,
    Float64,
    Complex32,
    String,
    Complex64,
    UInt16,
    UInt32,
    Int64,
    UInt64,
}

impl IdlType {
    fn from_code(code: i32) -> Result<Self, IoError> {
        match code {
            1 => Ok(Self::Byte),
            2 => Ok(Self::Int16),
            3 => Ok(Self::Int32),
            4 => Ok(Self::Float32),
            5 => Ok(Self::Float64),
            6 => Ok(Self::Complex32),
            7 => Ok(Self::String),
            9 => Ok(Self::Complex64),
            12 => Ok(Self::UInt16),
            13 => Ok(Self::UInt32),
            14 => Ok(Self::Int64),
            15 => Ok(Self::UInt64),
            8 => Err(IoError::UnsupportedFeature(
                "IDL SAVE structure type code 8 is not supported".to_string(),
            )),
            10 => Err(IoError::UnsupportedFeature(
                "IDL SAVE heap pointer type code 10 is not supported".to_string(),
            )),
            11 => Err(IoError::UnsupportedFeature(
                "IDL SAVE object pointer type code 11 is not supported".to_string(),
            )),
            other => Err(IoError::InvalidFormat(format!(
                "IDL SAVE unknown type code {other}"
            ))),
        }
    }
}

/// Scalar value read from an IDL SAVE file.
#[derive(Debug, Clone, PartialEq)]
pub enum IdlScalar {
    Byte(u8),
    Int16(i16),
    Int32(i32),
    Float32(f32),
    Float64(f64),
    Complex32 { real: f32, imag: f32 },
    String(Vec<u8>),
    Complex64 { real: f64, imag: f64 },
    UInt16(u16),
    UInt32(u32),
    Int64(i64),
    UInt64(u64),
}

/// Primitive IDL array. Dimensions follow SciPy's observable order: the IDL
/// descriptor dimensions are reversed before being exposed.
#[derive(Debug, Clone, PartialEq)]
pub struct IdlArray {
    pub element_type: IdlType,
    pub dims: Vec<usize>,
    pub values: Vec<IdlScalar>,
}

/// Variable value read from an IDL SAVE file.
#[derive(Debug, Clone, PartialEq)]
pub enum IdlValue {
    Null,
    Scalar(IdlScalar),
    Array(IdlArray),
}

/// Named IDL SAVE variable.
#[derive(Debug, Clone, PartialEq)]
pub struct IdlVariable {
    pub name: String,
    pub value: IdlValue,
}

/// Parsed IDL SAVE file.
#[derive(Debug, Clone, PartialEq)]
pub struct IdlSaveFile {
    pub variables: Vec<IdlVariable>,
}

impl IdlSaveFile {
    /// Case-insensitive variable lookup, matching SciPy's `readsav` access
    /// behavior for variable names.
    pub fn get(&self, name: &str) -> Option<&IdlValue> {
        self.variables
            .iter()
            .find(|variable| variable.name.eq_ignore_ascii_case(name))
            .map(|variable| &variable.value)
    }
}

#[derive(Debug, Clone)]
struct IdlTypeDesc {
    type_code: i32,
    is_array: bool,
    is_structure: bool,
    array_desc: Option<IdlArrayDesc>,
}

#[derive(Debug, Clone)]
struct IdlArrayDesc {
    nbytes: usize,
    nelements: usize,
    dims: Vec<usize>,
}

struct IdlReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> IdlReader<'a> {
    fn new(bytes: &'a [u8], offset: usize) -> Self {
        Self { bytes, offset }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }

    fn read_exact(&mut self, len: usize, context: &str) -> Result<&'a [u8], IoError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| IoError::InvalidFormat(format!("IDL SAVE {context} offset overflow")))?;
        if end > self.bytes.len() {
            return Err(IoError::InvalidFormat(format!(
                "IDL SAVE {context} truncated: need {len} bytes, have {}",
                self.remaining()
            )));
        }
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn skip(&mut self, len: usize, context: &str) -> Result<(), IoError> {
        self.read_exact(len, context).map(|_| ())
    }

    fn seek(&mut self, offset: usize, context: &str) -> Result<(), IoError> {
        if offset > self.bytes.len() {
            return Err(IoError::InvalidFormat(format!(
                "IDL SAVE {context} seeks past end: {offset} > {}",
                self.bytes.len()
            )));
        }
        self.offset = offset;
        Ok(())
    }

    fn align_32(&mut self, context: &str) -> Result<(), IoError> {
        let aligned = match self.offset % 4 {
            0 => self.offset,
            rem => self.offset + 4 - rem,
        };
        self.seek(aligned, context)
    }

    fn read_i32(&mut self, context: &str) -> Result<i32, IoError> {
        let bytes = self.read_exact(4, context)?;
        Ok(i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u32(&mut self, context: &str) -> Result<u32, IoError> {
        let bytes = self.read_exact(4, context)?;
        Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_i64(&mut self, context: &str) -> Result<i64, IoError> {
        let bytes = self.read_exact(8, context)?;
        Ok(i64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_u64(&mut self, context: &str) -> Result<u64, IoError> {
        let bytes = self.read_exact(8, context)?;
        Ok(u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_f32(&mut self, context: &str) -> Result<f32, IoError> {
        Ok(f32::from_bits(self.read_u32(context)?))
    }

    fn read_f64(&mut self, context: &str) -> Result<f64, IoError> {
        Ok(f64::from_bits(self.read_u64(context)?))
    }

    fn read_padded_u8(&mut self, context: &str) -> Result<u8, IoError> {
        Ok(self.read_exact(4, context)?[0])
    }

    fn read_padded_i16(&mut self, context: &str) -> Result<i16, IoError> {
        let bytes = self.read_exact(4, context)?;
        Ok(i16::from_be_bytes([bytes[2], bytes[3]]))
    }

    fn read_padded_u16(&mut self, context: &str) -> Result<u16, IoError> {
        let bytes = self.read_exact(4, context)?;
        Ok(u16::from_be_bytes([bytes[2], bytes[3]]))
    }
}

/// Read an IDL SAVE (`.sav`) byte stream.
///
/// This covers the uncompressed `scipy.io.readsav` scalar and primitive-array
/// surface: numeric scalars, byte strings, complex numbers, and arrays of those
/// primitive values. Structures, heap/object pointers, and compressed SAVE
/// files fail closed with `UnsupportedFeature`.
pub fn read_idl_save(bytes: &[u8]) -> Result<IdlSaveFile, IoError> {
    if bytes.len() < 4 {
        return Err(IoError::InvalidFormat(
            "IDL SAVE header truncated".to_string(),
        ));
    }
    if &bytes[..2] != b"SR" {
        return Err(IoError::InvalidFormat(format!(
            "IDL SAVE invalid signature: {:02x?}",
            &bytes[..2]
        )));
    }
    match &bytes[2..4] {
        b"\x00\x04" => {}
        b"\x00\x06" => {
            return Err(IoError::UnsupportedFeature(
                "compressed IDL SAVE files are not supported".to_string(),
            ));
        }
        recfmt => {
            return Err(IoError::InvalidFormat(format!(
                "IDL SAVE invalid record format: {recfmt:02x?}"
            )));
        }
    }

    let mut reader = IdlReader::new(bytes, 4);
    let mut variables = Vec::new();
    let mut saw_end = false;

    while reader.offset < bytes.len() {
        let record_start = reader.offset;
        let rectype = reader.read_i32("record type")?;
        let next_low = u64::from(reader.read_u32("next record low word")?);
        let next_high = u64::from(reader.read_u32("next record high word")?);
        reader.skip(4, "record header padding")?;

        if rectype == 6 {
            saw_end = true;
            break;
        }

        let next_record = checked_idl_next_record(next_low, next_high, reader.offset)?;
        if next_record > bytes.len() {
            return Err(IoError::InvalidFormat(format!(
                "IDL SAVE record at offset {record_start} points past end: {next_record} > {}",
                bytes.len()
            )));
        }

        match rectype {
            0 | 1 | 3 | 10 | 12 | 13 | 14 | 15 | 17 | 19 | 20 => {}
            2 => {
                let variable = read_idl_variable_record(&mut reader, next_record)?;
                variables.push(variable);
            }
            16 => {
                return Err(IoError::UnsupportedFeature(
                    "IDL SAVE heap data records are not supported".to_string(),
                ));
            }
            other => {
                return Err(IoError::InvalidFormat(format!(
                    "IDL SAVE unknown record type {other} at offset {record_start}"
                )));
            }
        }

        if reader.offset > next_record {
            return Err(IoError::InvalidFormat(format!(
                "IDL SAVE record type {rectype} over-read next record boundary"
            )));
        }
        reader.seek(next_record, "record boundary")?;
    }

    if !saw_end {
        return Err(IoError::InvalidFormat(
            "IDL SAVE missing END_MARKER record".to_string(),
        ));
    }

    Ok(IdlSaveFile { variables })
}

/// Alias matching `scipy.io.readsav`.
pub fn readsav(bytes: &[u8]) -> Result<IdlSaveFile, IoError> {
    read_idl_save(bytes)
}

fn checked_idl_next_record(low: u64, high: u64, current: usize) -> Result<usize, IoError> {
    let next = high
        .checked_mul(1u64 << 32)
        .and_then(|base| base.checked_add(low))
        .ok_or_else(|| IoError::InvalidFormat("IDL SAVE next record overflow".to_string()))?;
    let next = usize::try_from(next).map_err(|_| {
        IoError::InvalidFormat("IDL SAVE next record does not fit usize".to_string())
    })?;
    if next < current {
        return Err(IoError::InvalidFormat(format!(
            "IDL SAVE next record {next} precedes current offset {current}"
        )));
    }
    Ok(next)
}

fn read_idl_variable_record(
    reader: &mut IdlReader<'_>,
    next_record: usize,
) -> Result<IdlVariable, IoError> {
    let name = read_idl_string(reader, "variable name")?;
    let typedesc = read_idl_typedesc(reader)?;
    let value = if typedesc.type_code == 0 {
        if reader.offset != next_record {
            return Err(IoError::InvalidFormat(
                "IDL SAVE null typedesc has trailing payload".to_string(),
            ));
        }
        IdlValue::Null
    } else {
        let varstart = reader.read_i32("VARSTART")?;
        if varstart != 7 {
            return Err(IoError::InvalidFormat(format!(
                "IDL SAVE VARSTART must be 7, got {varstart}"
            )));
        }
        read_idl_value(reader, &typedesc)?
    };
    Ok(IdlVariable { name, value })
}

fn read_idl_typedesc(reader: &mut IdlReader<'_>) -> Result<IdlTypeDesc, IoError> {
    let type_code = reader.read_i32("type descriptor type code")?;
    let varflags = reader.read_i32("type descriptor flags")?;

    if varflags & 2 == 2 {
        return Err(IoError::UnsupportedFeature(
            "IDL SAVE system variables are not supported".to_string(),
        ));
    }
    let is_array = varflags & 4 == 4;
    let is_structure = varflags & 32 == 32;
    if is_structure {
        return Err(IoError::UnsupportedFeature(
            "IDL SAVE structure variables are not supported".to_string(),
        ));
    }
    let array_desc = if is_array {
        Some(read_idl_arraydesc(reader)?)
    } else {
        None
    };

    Ok(IdlTypeDesc {
        type_code,
        is_array,
        is_structure,
        array_desc,
    })
}

fn read_idl_arraydesc(reader: &mut IdlReader<'_>) -> Result<IdlArrayDesc, IoError> {
    let arrstart = reader.read_i32("array descriptor start")?;
    if arrstart == 18 {
        return Err(IoError::UnsupportedFeature(
            "IDL SAVE 64-bit array descriptors are not supported".to_string(),
        ));
    }
    if arrstart != 8 {
        return Err(IoError::InvalidFormat(format!(
            "IDL SAVE unknown array descriptor start {arrstart}"
        )));
    }

    reader.skip(4, "array descriptor padding")?;
    let nbytes = read_idl_nonnegative_usize(reader, "array byte count")?;
    let nelements = read_idl_nonnegative_usize(reader, "array element count")?;
    let ndims = read_idl_nonnegative_usize(reader, "array dimension count")?;
    reader.skip(8, "array descriptor reserved fields")?;
    let nmax = read_idl_nonnegative_usize(reader, "array max dimension count")?;
    if ndims > nmax {
        return Err(IoError::InvalidFormat(format!(
            "IDL SAVE array ndims {ndims} exceeds nmax {nmax}"
        )));
    }
    if nelements > IDL_MAX_ARRAY_ELEMENTS {
        return Err(IoError::InvalidFormat(format!(
            "IDL SAVE array element count {nelements} exceeds safety bound {IDL_MAX_ARRAY_ELEMENTS}"
        )));
    }

    let mut raw_dims = Vec::with_capacity(nmax);
    for _ in 0..nmax {
        raw_dims.push(read_idl_nonnegative_usize(reader, "array dimension")?);
    }
    validate_idl_array_shape(&raw_dims, ndims, nelements)?;
    let dims = raw_dims
        .iter()
        .take(ndims)
        .rev()
        .copied()
        .collect::<Vec<_>>();

    Ok(IdlArrayDesc {
        nbytes,
        nelements,
        dims,
    })
}

fn read_idl_nonnegative_usize(reader: &mut IdlReader<'_>, context: &str) -> Result<usize, IoError> {
    let value = reader.read_i32(context)?;
    if value < 0 {
        return Err(IoError::InvalidFormat(format!(
            "IDL SAVE {context} is negative: {value}"
        )));
    }
    usize::try_from(value).map_err(|_| {
        IoError::InvalidFormat(format!("IDL SAVE {context} does not fit usize: {value}"))
    })
}

fn validate_idl_array_shape(
    raw_dims: &[usize],
    ndims: usize,
    nelements: usize,
) -> Result<(), IoError> {
    let shape_product = raw_dims.iter().take(ndims).try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| IoError::InvalidFormat("IDL SAVE array shape overflow".to_string()))
    })?;
    if shape_product != nelements {
        return Err(IoError::InvalidFormat(format!(
            "IDL SAVE array shape product {shape_product} does not match element count {nelements}"
        )));
    }
    Ok(())
}

fn read_idl_value(reader: &mut IdlReader<'_>, typedesc: &IdlTypeDesc) -> Result<IdlValue, IoError> {
    if typedesc.is_structure {
        return Err(IoError::UnsupportedFeature(
            "IDL SAVE structure values are not supported".to_string(),
        ));
    }
    let idl_type = IdlType::from_code(typedesc.type_code)?;
    if typedesc.is_array {
        let array_desc = typedesc.array_desc.as_ref().ok_or_else(|| {
            IoError::InvalidFormat("IDL SAVE array flag without descriptor".to_string())
        })?;
        Ok(IdlValue::Array(read_idl_array(
            reader, idl_type, array_desc,
        )?))
    } else {
        Ok(IdlValue::Scalar(read_idl_scalar(reader, idl_type)?))
    }
}

fn read_idl_scalar(reader: &mut IdlReader<'_>, idl_type: IdlType) -> Result<IdlScalar, IoError> {
    match idl_type {
        IdlType::Byte => {
            let byte_count = reader.read_i32("byte scalar marker")?;
            if byte_count != 1 {
                return Err(IoError::InvalidFormat(format!(
                    "IDL SAVE byte scalar marker must be 1, got {byte_count}"
                )));
            }
            Ok(IdlScalar::Byte(reader.read_padded_u8("byte scalar")?))
        }
        IdlType::Int16 => Ok(IdlScalar::Int16(reader.read_padded_i16("int16 scalar")?)),
        IdlType::Int32 => Ok(IdlScalar::Int32(reader.read_i32("int32 scalar")?)),
        IdlType::Float32 => Ok(IdlScalar::Float32(reader.read_f32("float32 scalar")?)),
        IdlType::Float64 => Ok(IdlScalar::Float64(reader.read_f64("float64 scalar")?)),
        IdlType::Complex32 => {
            let real = reader.read_f32("complex32 real")?;
            let imag = reader.read_f32("complex32 imag")?;
            Ok(IdlScalar::Complex32 { real, imag })
        }
        IdlType::String => Ok(IdlScalar::String(read_idl_string_data(
            reader,
            "string scalar",
        )?)),
        IdlType::Complex64 => {
            let real = reader.read_f64("complex64 real")?;
            let imag = reader.read_f64("complex64 imag")?;
            Ok(IdlScalar::Complex64 { real, imag })
        }
        IdlType::UInt16 => Ok(IdlScalar::UInt16(reader.read_padded_u16("uint16 scalar")?)),
        IdlType::UInt32 => Ok(IdlScalar::UInt32(reader.read_u32("uint32 scalar")?)),
        IdlType::Int64 => Ok(IdlScalar::Int64(reader.read_i64("int64 scalar")?)),
        IdlType::UInt64 => Ok(IdlScalar::UInt64(reader.read_u64("uint64 scalar")?)),
    }
}

fn read_idl_array(
    reader: &mut IdlReader<'_>,
    idl_type: IdlType,
    desc: &IdlArrayDesc,
) -> Result<IdlArray, IoError> {
    let values = match idl_type {
        IdlType::Byte => read_idl_byte_array(reader, desc)?,
        IdlType::Int16 => read_idl_repeated(reader, desc.nelements, |reader| {
            Ok(IdlScalar::Int16(
                reader.read_padded_i16("int16 array element")?,
            ))
        })?,
        IdlType::Int32 => read_idl_repeated(reader, desc.nelements, |reader| {
            Ok(IdlScalar::Int32(reader.read_i32("int32 array element")?))
        })?,
        IdlType::Float32 => read_idl_repeated(reader, desc.nelements, |reader| {
            Ok(IdlScalar::Float32(
                reader.read_f32("float32 array element")?,
            ))
        })?,
        IdlType::Float64 => read_idl_repeated(reader, desc.nelements, |reader| {
            Ok(IdlScalar::Float64(
                reader.read_f64("float64 array element")?,
            ))
        })?,
        IdlType::Complex32 => read_idl_repeated(reader, desc.nelements, |reader| {
            let real = reader.read_f32("complex32 array real")?;
            let imag = reader.read_f32("complex32 array imag")?;
            Ok(IdlScalar::Complex32 { real, imag })
        })?,
        IdlType::String => read_idl_repeated(reader, desc.nelements, |reader| {
            Ok(IdlScalar::String(read_idl_string_data(
                reader,
                "string array element",
            )?))
        })?,
        IdlType::Complex64 => read_idl_repeated(reader, desc.nelements, |reader| {
            let real = reader.read_f64("complex64 array real")?;
            let imag = reader.read_f64("complex64 array imag")?;
            Ok(IdlScalar::Complex64 { real, imag })
        })?,
        IdlType::UInt16 => read_idl_repeated(reader, desc.nelements, |reader| {
            Ok(IdlScalar::UInt16(
                reader.read_padded_u16("uint16 array element")?,
            ))
        })?,
        IdlType::UInt32 => read_idl_repeated(reader, desc.nelements, |reader| {
            Ok(IdlScalar::UInt32(reader.read_u32("uint32 array element")?))
        })?,
        IdlType::Int64 => read_idl_repeated(reader, desc.nelements, |reader| {
            Ok(IdlScalar::Int64(reader.read_i64("int64 array element")?))
        })?,
        IdlType::UInt64 => read_idl_repeated(reader, desc.nelements, |reader| {
            Ok(IdlScalar::UInt64(reader.read_u64("uint64 array element")?))
        })?,
    };
    reader.align_32("array payload alignment")?;
    Ok(IdlArray {
        element_type: idl_type,
        dims: desc.dims.clone(),
        values,
    })
}

fn read_idl_repeated<F>(
    reader: &mut IdlReader<'_>,
    count: usize,
    mut read_one: F,
) -> Result<Vec<IdlScalar>, IoError>
where
    F: FnMut(&mut IdlReader<'_>) -> Result<IdlScalar, IoError>,
{
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(read_one(reader)?);
    }
    Ok(values)
}

fn read_idl_byte_array(
    reader: &mut IdlReader<'_>,
    desc: &IdlArrayDesc,
) -> Result<Vec<IdlScalar>, IoError> {
    let byte_count = read_idl_nonnegative_usize(reader, "byte array payload byte count")?;
    if byte_count != desc.nbytes || byte_count != desc.nelements {
        return Err(IoError::InvalidFormat(format!(
            "IDL SAVE byte array count {byte_count} does not match descriptor nbytes={} nelements={}",
            desc.nbytes, desc.nelements
        )));
    }
    reader
        .read_exact(byte_count, "byte array payload")?
        .iter()
        .map(|&value| Ok(IdlScalar::Byte(value)))
        .collect()
}

fn read_idl_string(reader: &mut IdlReader<'_>, context: &str) -> Result<String, IoError> {
    let len = read_idl_nonnegative_usize(reader, context)?;
    if len == 0 {
        return Ok(String::new());
    }
    let bytes = reader.read_exact(len, context)?;
    reader.align_32(context)?;
    Ok(bytes.iter().map(|&byte| char::from(byte)).collect())
}

fn read_idl_string_data(reader: &mut IdlReader<'_>, context: &str) -> Result<Vec<u8>, IoError> {
    let len = read_idl_nonnegative_usize(reader, context)?;
    if len == 0 {
        return Ok(Vec::new());
    }
    let repeated_len = read_idl_nonnegative_usize(reader, context)?;
    if repeated_len != len {
        return Err(IoError::InvalidFormat(format!(
            "IDL SAVE string length marker mismatch: {len} != {repeated_len}"
        )));
    }
    let bytes = reader.read_exact(len, context)?.to_vec();
    reader.align_32(context)?;
    Ok(bytes)
}

// ══════════════════════════════════════════════════════════════════════
// Text matrix utility
// ══════════════════════════════════════════════════════════════════════

/// Load a whitespace-delimited text file as a matrix.
///
/// Like `numpy.loadtxt`.
pub fn loadtxt(content: &str) -> Result<(usize, usize, Vec<f64>), IoError> {
    let mut data = Vec::new();
    let mut cols = 0usize;
    let mut rows = 0usize;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('%') {
            continue;
        }

        let vals: Result<Vec<f64>, _> = trimmed
            .split_whitespace()
            .map(|s| s.parse::<f64>())
            .collect();
        let vals = vals.map_err(|e| IoError::InvalidFormat(format!("parse error: {e}")))?;

        if rows == 0 {
            cols = vals.len();
        } else if vals.len() != cols {
            return Err(IoError::InvalidFormat(format!(
                "row {rows} has {} columns, expected {cols}",
                vals.len()
            )));
        }

        data.extend_from_slice(&vals);
        rows += 1;
    }

    Ok((rows, cols, data))
}

/// Save a matrix as whitespace-delimited text.
///
/// Like `numpy.savetxt`.
pub fn savetxt(rows: usize, cols: usize, data: &[f64], delimiter: &str) -> Result<String, IoError> {
    if delimiter.contains(['\n', '\r']) {
        return Err(IoError::InvalidFormat(format!(
            "delimiter {:?} contains a newline and cannot be encoded safely",
            delimiter
        )));
    }
    let expected_len = checked_matrix_len(rows, cols, "text matrix")?;
    if data.len() != expected_len {
        return Err(IoError::InvalidFormat(format!(
            "data length {} doesn't match {}x{}",
            data.len(),
            rows,
            cols
        )));
    }
    let mut out = String::new();
    for r in 0..rows {
        for c in 0..cols {
            if c > 0 {
                out.push_str(delimiter);
            }
            out.push_str(&format!("{}", data[r * cols + c]));
        }
        out.push('\n');
    }
    Ok(out)
}

/// Read a CSV file into rows of f64 values.
///
/// Simple CSV reader for numerical data.
pub type CsvResult = Result<(Option<Vec<String>>, Vec<Vec<f64>>), IoError>;

pub fn read_csv(content: &str, delimiter: char, has_header: bool) -> CsvResult {
    let mut lines = content.lines();
    let header = if has_header {
        Some(
            lines
                .next()
                .ok_or_else(|| {
                    IoError::InvalidFormat(
                        "CSV header row is required but the input is empty".to_string(),
                    )
                })?
                .split(delimiter)
                .map(|s| s.trim().to_string())
                .collect(),
        )
    } else {
        None
    };
    let header_cols = header.as_ref().map(std::vec::Vec::len);

    let mut data = Vec::new();
    let mut expected_cols = None;
    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let row: Result<Vec<f64>, _> = trimmed
            .split(delimiter)
            .map(|s| s.trim().parse::<f64>())
            .collect();
        match row {
            Ok(r) => {
                if let Some(cols) = expected_cols {
                    if r.len() != cols {
                        return Err(IoError::InvalidFormat(format!(
                            "CSV row has {} columns, expected {cols}",
                            r.len()
                        )));
                    }
                } else {
                    if let Some(header_cols) = header_cols
                        && r.len() != header_cols
                    {
                        return Err(IoError::InvalidFormat(format!(
                            "CSV header has {header_cols} columns but first data row has {}",
                            r.len()
                        )));
                    }
                    expected_cols = Some(r.len());
                }
                data.push(r);
            }
            Err(e) => {
                return Err(IoError::InvalidFormat(format!("CSV parse error: {e}")));
            }
        }
    }

    Ok((header, data))
}

/// Write data to CSV format.
pub fn write_csv(
    header: Option<&[&str]>,
    data: &[Vec<f64>],
    delimiter: char,
) -> Result<String, IoError> {
    let mut out = String::new();
    let header_cols = header.map(<[&str]>::len);
    if let Some(h) = header {
        for cell in h {
            if cell.contains(['\n', '\r']) {
                return Err(IoError::InvalidFormat(format!(
                    "CSV header cell {:?} contains a newline and cannot be encoded safely",
                    cell
                )));
            }
            if cell.contains(delimiter) {
                return Err(IoError::InvalidFormat(format!(
                    "CSV header cell {:?} contains the delimiter {:?} and cannot be encoded safely",
                    cell, delimiter
                )));
            }
        }
        out.push_str(&h.join(&delimiter.to_string()));
        out.push('\n');
    }
    let mut expected_cols = None;
    for row in data {
        if let Some(cols) = expected_cols {
            if row.len() != cols {
                return Err(IoError::InvalidFormat(format!(
                    "CSV row has {} columns, expected {cols}",
                    row.len()
                )));
            }
        } else {
            if let Some(header_cols) = header_cols
                && row.len() != header_cols
            {
                return Err(IoError::InvalidFormat(format!(
                    "CSV header has {header_cols} columns but first data row has {}",
                    row.len()
                )));
            }
            expected_cols = Some(row.len());
        }
        let row_str: Vec<String> = row.iter().map(|v| format!("{v}")).collect();
        out.push_str(&row_str.join(&delimiter.to_string()));
        out.push('\n');
    }
    Ok(out)
}

/// Read a simple JSON array of numbers.
pub fn read_json_array(content: &str) -> Result<Vec<f64>, IoError> {
    let trimmed = content.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return Err(IoError::InvalidFormat("expected JSON array".to_string()));
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<f64>()
                .map_err(|e| IoError::InvalidFormat(format!("JSON parse error: {e}")))
                .and_then(|v| {
                    if v.is_finite() {
                        Ok(v)
                    } else {
                        Err(IoError::InvalidFormat(format!(
                            "JSON parse error: non-finite value {v}"
                        )))
                    }
                })
        })
        .collect()
}

/// Write a vector as a JSON array.
pub fn write_json_array(data: &[f64]) -> Result<String, IoError> {
    if let Some((idx, value)) = data
        .iter()
        .copied()
        .enumerate()
        .find(|(_, v)| !v.is_finite())
    {
        return Err(IoError::InvalidFormat(format!(
            "JSON array value at index {idx} is not finite: {value}"
        )));
    }
    let items: Vec<String> = data.iter().map(|v| format!("{v}")).collect();
    Ok(format!("[{}]", items.join(", ")))
}

/// Read a simple NPY-like header (shape + dtype) from text representation.
///
/// Returns (shape, data).
pub fn read_npy_text(content: &str) -> Result<(Vec<usize>, Vec<f64>), IoError> {
    let mut lines = content.lines();

    // Read shape line
    let shape_line = lines
        .next()
        .ok_or_else(|| IoError::InvalidFormat("missing shape line".to_string()))?;
    let trimmed_shape = shape_line.trim();
    if trimmed_shape.is_empty() {
        return Err(IoError::InvalidFormat(
            "shape declaration must contain at least one dimension".to_string(),
        ));
    }
    if trimmed_shape
        .split(',')
        .any(|segment| segment.trim().is_empty())
    {
        return Err(IoError::InvalidFormat(
            "shape declaration contains an empty dimension".to_string(),
        ));
    }
    let shape: Result<Vec<usize>, _> = shape_line
        .trim()
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect();
    let shape = shape.map_err(|e| IoError::InvalidFormat(format!("bad shape: {e}")))?;
    // Read data
    let mut data = Vec::new();
    for line in lines {
        for val in line.split_whitespace() {
            let v: f64 = val
                .parse()
                .map_err(|e| IoError::InvalidFormat(format!("bad value: {e}")))?;
            data.push(v);
        }
    }

    let expected_len = shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| IoError::InvalidFormat("shape product overflowed usize".to_string()))
    })?;
    if data.len() != expected_len {
        return Err(IoError::InvalidFormat(format!(
            "shape {:?} expects {expected_len} values but found {}",
            shape,
            data.len()
        )));
    }

    Ok((shape, data))
}

// ══════════════════════════════════════════════════════════════════════
// NetCDF classic v3
// ══════════════════════════════════════════════════════════════════════

const NC_DIMENSION: u32 = 10;
const NC_VARIABLE: u32 = 11;
const NC_ATTRIBUTE: u32 = 12;
const NC_BYTE: u32 = 1;
const NC_CHAR: u32 = 2;
const NC_SHORT: u32 = 3;
const NC_INT: u32 = 4;
const NC_FLOAT: u32 = 5;
const NC_DOUBLE: u32 = 6;

/// NetCDF classic scalar type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetcdfType {
    Byte,
    Char,
    Short,
    Int,
    Float,
    Double,
}

/// NetCDF dimension. `len == None` represents the classic unlimited dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetcdfDimension {
    pub name: String,
    pub len: Option<usize>,
}

/// NetCDF typed payload.
#[derive(Debug, Clone, PartialEq)]
pub enum NetcdfValue {
    Byte(Vec<i8>),
    Char(String),
    Short(Vec<i16>),
    Int(Vec<i32>),
    Float(Vec<f32>),
    Double(Vec<f64>),
}

impl NetcdfValue {
    fn value_type(&self) -> NetcdfType {
        match self {
            Self::Byte(_) => NetcdfType::Byte,
            Self::Char(_) => NetcdfType::Char,
            Self::Short(_) => NetcdfType::Short,
            Self::Int(_) => NetcdfType::Int,
            Self::Float(_) => NetcdfType::Float,
            Self::Double(_) => NetcdfType::Double,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Byte(v) => v.len(),
            Self::Char(s) => s.len(),
            Self::Short(v) => v.len(),
            Self::Int(v) => v.len(),
            Self::Float(v) => v.len(),
            Self::Double(v) => v.len(),
        }
    }
}

/// NetCDF attribute.
#[derive(Debug, Clone, PartialEq)]
pub struct NetcdfAttribute {
    pub name: String,
    pub value: NetcdfValue,
}

/// NetCDF variable. `dim_ids` indexes into `NetcdfFile::dimensions`.
#[derive(Debug, Clone, PartialEq)]
pub struct NetcdfVariable {
    pub name: String,
    pub dim_ids: Vec<usize>,
    pub attributes: Vec<NetcdfAttribute>,
    pub data: NetcdfValue,
}

/// Parsed NetCDF classic file.
#[derive(Debug, Clone, PartialEq)]
pub struct NetcdfFile {
    pub dimensions: Vec<NetcdfDimension>,
    pub attributes: Vec<NetcdfAttribute>,
    pub variables: Vec<NetcdfVariable>,
}

#[derive(Debug, Clone)]
struct NetcdfVariableHeader {
    name: String,
    dim_ids: Vec<usize>,
    attributes: Vec<NetcdfAttribute>,
    value_type: NetcdfType,
    begin: usize,
}

struct NetcdfReader<'a> {
    bytes: &'a [u8],
    offset: usize,
    version: u8,
}

/// Read a NetCDF classic or 64-bit-offset file from bytes.
///
/// This covers the fixed-size NetCDF v3 subset exposed by
/// `scipy.io.netcdf_file`: dimensions, global attributes, variables, and
/// primitive numeric/character arrays. Unlimited record variables are rejected
/// until the record-interleaving path is implemented.
pub fn read_netcdf_classic(bytes: &[u8]) -> Result<NetcdfFile, IoError> {
    if bytes.len() < 4 {
        return Err(IoError::InvalidFormat("NetCDF file too short".to_string()));
    }
    if &bytes[0..3] != b"CDF" {
        return Err(IoError::InvalidFormat(
            "NetCDF missing CDF magic".to_string(),
        ));
    }
    let version = bytes[3];
    if version != 1 && version != 2 {
        return Err(IoError::UnsupportedFeature(format!(
            "NetCDF version byte {version} is not supported"
        )));
    }

    let mut reader = NetcdfReader {
        bytes,
        offset: 4,
        version,
    };
    let numrecs = read_netcdf_u32(&mut reader)? as usize;
    let dimensions = read_netcdf_dimensions(&mut reader)?;
    if dimensions.iter().filter(|dim| dim.len.is_none()).count() > 1 {
        return Err(IoError::InvalidFormat(
            "NetCDF classic permits at most one unlimited dimension".to_string(),
        ));
    }
    let attributes = read_netcdf_attributes(&mut reader)?;
    let variable_headers = read_netcdf_variable_headers(&mut reader)?;

    let mut variables = Vec::with_capacity(variable_headers.len());
    for header in variable_headers {
        let element_count = netcdf_element_count_for_dims(&dimensions, &header.dim_ids, numrecs)?;
        if header
            .dim_ids
            .iter()
            .any(|&dim_id| dimensions[dim_id].len.is_none())
        {
            return Err(IoError::UnsupportedFeature(
                "NetCDF unlimited record variables are not supported".to_string(),
            ));
        }
        let raw_len = netcdf_value_raw_len(header.value_type, element_count)?;
        let end = header.begin.checked_add(raw_len).ok_or_else(|| {
            IoError::InvalidFormat("NetCDF variable payload offset overflowed usize".to_string())
        })?;
        let payload = bytes.get(header.begin..end).ok_or_else(|| {
            IoError::InvalidFormat(format!(
                "NetCDF variable '{}' payload extends past file",
                header.name
            ))
        })?;
        let data = decode_netcdf_values(header.value_type, payload, element_count, &header.name)?;
        variables.push(NetcdfVariable {
            name: header.name,
            dim_ids: header.dim_ids,
            attributes: header.attributes,
            data,
        });
    }

    Ok(NetcdfFile {
        dimensions,
        attributes,
        variables,
    })
}

/// Write a fixed-size NetCDF classic file.
///
/// The writer emits NetCDF classic v1 files and fails closed for unlimited
/// dimensions or payload offsets that require the 64-bit-offset variant.
pub fn write_netcdf_classic(file: &NetcdfFile) -> Result<Vec<u8>, IoError> {
    validate_netcdf_fixed_file(file)?;

    let placeholder_begins = vec![0usize; file.variables.len()];
    let header = encode_netcdf_header(file, &placeholder_begins)?;
    let mut cursor = align4(header.len());
    let mut begins = Vec::with_capacity(file.variables.len());
    let mut payloads = Vec::with_capacity(file.variables.len());
    for variable in &file.variables {
        let payload = encode_netcdf_padded_values(&variable.data)?;
        begins.push(cursor);
        cursor = cursor.checked_add(payload.len()).ok_or_else(|| {
            IoError::InvalidFormat("NetCDF output size overflowed usize".to_string())
        })?;
        payloads.push(payload);
    }
    if begins.iter().any(|&begin| u32::try_from(begin).is_err()) {
        return Err(IoError::UnsupportedFeature(
            "NetCDF output requires 64-bit variable offsets".to_string(),
        ));
    }

    let mut out = encode_netcdf_header(file, &begins)?;
    while out.len() < align4(out.len()) {
        out.push(0);
    }
    for payload in payloads {
        out.extend_from_slice(&payload);
    }
    Ok(out)
}

/// Alias matching the SciPy surface name.
pub fn netcdf_file_read(bytes: &[u8]) -> Result<NetcdfFile, IoError> {
    read_netcdf_classic(bytes)
}

/// Alias matching the SciPy surface name.
pub fn netcdf_file_write(file: &NetcdfFile) -> Result<Vec<u8>, IoError> {
    write_netcdf_classic(file)
}

fn netcdf_type_code(value_type: NetcdfType) -> u32 {
    match value_type {
        NetcdfType::Byte => NC_BYTE,
        NetcdfType::Char => NC_CHAR,
        NetcdfType::Short => NC_SHORT,
        NetcdfType::Int => NC_INT,
        NetcdfType::Float => NC_FLOAT,
        NetcdfType::Double => NC_DOUBLE,
    }
}

fn netcdf_type_from_code(code: u32) -> Result<NetcdfType, IoError> {
    match code {
        NC_BYTE => Ok(NetcdfType::Byte),
        NC_CHAR => Ok(NetcdfType::Char),
        NC_SHORT => Ok(NetcdfType::Short),
        NC_INT => Ok(NetcdfType::Int),
        NC_FLOAT => Ok(NetcdfType::Float),
        NC_DOUBLE => Ok(NetcdfType::Double),
        other => Err(IoError::UnsupportedFeature(format!(
            "NetCDF type code {other} is not supported"
        ))),
    }
}

fn netcdf_type_size(value_type: NetcdfType) -> usize {
    match value_type {
        NetcdfType::Byte | NetcdfType::Char => 1,
        NetcdfType::Short => 2,
        NetcdfType::Int | NetcdfType::Float => 4,
        NetcdfType::Double => 8,
    }
}

fn align4(value: usize) -> usize {
    value + ((4 - (value % 4)) % 4)
}

fn netcdf_value_raw_len(value_type: NetcdfType, count: usize) -> Result<usize, IoError> {
    count
        .checked_mul(netcdf_type_size(value_type))
        .ok_or_else(|| {
            IoError::InvalidFormat("NetCDF value byte length overflowed usize".to_string())
        })
}

fn read_netcdf_u32(reader: &mut NetcdfReader<'_>) -> Result<u32, IoError> {
    let end = reader
        .offset
        .checked_add(4)
        .ok_or_else(|| IoError::InvalidFormat("NetCDF u32 offset overflowed usize".to_string()))?;
    let slice = reader
        .bytes
        .get(reader.offset..end)
        .ok_or_else(|| IoError::InvalidFormat("truncated NetCDF u32".to_string()))?;
    reader.offset = end;
    Ok(u32::from_be_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_netcdf_begin(reader: &mut NetcdfReader<'_>) -> Result<usize, IoError> {
    if reader.version == 1 {
        Ok(read_netcdf_u32(reader)? as usize)
    } else {
        let hi = read_netcdf_u32(reader)? as u64;
        let lo = read_netcdf_u32(reader)? as u64;
        let value = (hi << 32) | lo;
        usize::try_from(value).map_err(|_| {
            IoError::InvalidFormat(format!(
                "NetCDF 64-bit offset {value} cannot be represented as usize"
            ))
        })
    }
}

fn read_netcdf_name(reader: &mut NetcdfReader<'_>, context: &str) -> Result<String, IoError> {
    let len = read_netcdf_u32(reader)? as usize;
    let end = reader.offset.checked_add(len).ok_or_else(|| {
        IoError::InvalidFormat(format!("NetCDF {context} name offset overflowed usize"))
    })?;
    let name_bytes = reader
        .bytes
        .get(reader.offset..end)
        .ok_or_else(|| IoError::InvalidFormat(format!("truncated NetCDF {context} name")))?;
    reader.offset = align4(end);
    if reader.offset > reader.bytes.len() {
        return Err(IoError::InvalidFormat(format!(
            "truncated NetCDF {context} name padding"
        )));
    }
    let name = String::from_utf8(name_bytes.to_vec())
        .map_err(|e| IoError::InvalidFormat(format!("NetCDF {context} name is not UTF-8: {e}")))?;
    validate_netcdf_name(&name, context)?;
    Ok(name)
}

fn read_netcdf_list_count(
    reader: &mut NetcdfReader<'_>,
    expected_tag: u32,
    context: &str,
) -> Result<usize, IoError> {
    let tag = read_netcdf_u32(reader)?;
    let count = read_netcdf_u32(reader)? as usize;
    if tag == 0 && count == 0 {
        return Ok(0);
    }
    if tag != expected_tag {
        return Err(IoError::InvalidFormat(format!(
            "NetCDF {context} list tag {tag} does not match expected {expected_tag}"
        )));
    }
    Ok(count)
}

fn read_netcdf_dimensions(reader: &mut NetcdfReader<'_>) -> Result<Vec<NetcdfDimension>, IoError> {
    let count = read_netcdf_list_count(reader, NC_DIMENSION, "dimension")?;
    let mut dimensions = Vec::with_capacity(count);
    for _ in 0..count {
        let name = read_netcdf_name(reader, "dimension")?;
        let raw_len = read_netcdf_u32(reader)?;
        let len = if raw_len == 0 {
            None
        } else {
            Some(raw_len as usize)
        };
        dimensions.push(NetcdfDimension { name, len });
    }
    Ok(dimensions)
}

fn read_netcdf_attributes(reader: &mut NetcdfReader<'_>) -> Result<Vec<NetcdfAttribute>, IoError> {
    let count = read_netcdf_list_count(reader, NC_ATTRIBUTE, "attribute")?;
    let mut attributes = Vec::with_capacity(count);
    for _ in 0..count {
        let name = read_netcdf_name(reader, "attribute")?;
        let value_type = netcdf_type_from_code(read_netcdf_u32(reader)?)?;
        let value_count = read_netcdf_u32(reader)? as usize;
        let raw_len = netcdf_value_raw_len(value_type, value_count)?;
        let value_end = reader.offset.checked_add(raw_len).ok_or_else(|| {
            IoError::InvalidFormat("NetCDF attribute payload offset overflowed usize".to_string())
        })?;
        let payload = reader.bytes.get(reader.offset..value_end).ok_or_else(|| {
            IoError::InvalidFormat(format!("truncated NetCDF attribute '{name}'"))
        })?;
        let value = decode_netcdf_values(value_type, payload, value_count, &name)?;
        reader.offset = align4(value_end);
        if reader.offset > reader.bytes.len() {
            return Err(IoError::InvalidFormat(format!(
                "truncated NetCDF attribute '{name}' padding"
            )));
        }
        attributes.push(NetcdfAttribute { name, value });
    }
    Ok(attributes)
}

fn read_netcdf_variable_headers(
    reader: &mut NetcdfReader<'_>,
) -> Result<Vec<NetcdfVariableHeader>, IoError> {
    let count = read_netcdf_list_count(reader, NC_VARIABLE, "variable")?;
    let mut variables = Vec::with_capacity(count);
    for _ in 0..count {
        let name = read_netcdf_name(reader, "variable")?;
        let dim_count = read_netcdf_u32(reader)? as usize;
        let mut dim_ids = Vec::with_capacity(dim_count);
        for _ in 0..dim_count {
            dim_ids.push(read_netcdf_u32(reader)? as usize);
        }
        let attributes = read_netcdf_attributes(reader)?;
        let value_type = netcdf_type_from_code(read_netcdf_u32(reader)?)?;
        let _vsize = read_netcdf_u32(reader)?;
        let begin = read_netcdf_begin(reader)?;
        variables.push(NetcdfVariableHeader {
            name,
            dim_ids,
            attributes,
            value_type,
            begin,
        });
    }
    Ok(variables)
}

fn netcdf_element_count_for_dims(
    dimensions: &[NetcdfDimension],
    dim_ids: &[usize],
    numrecs: usize,
) -> Result<usize, IoError> {
    if dim_ids.is_empty() {
        return Ok(1);
    }
    let mut count = 1usize;
    for &dim_id in dim_ids {
        let dimension = dimensions.get(dim_id).ok_or_else(|| {
            IoError::InvalidFormat(format!(
                "NetCDF variable references bad dimension id {dim_id}"
            ))
        })?;
        let dim_len = dimension.len.unwrap_or(numrecs);
        if dim_len == 0 {
            return Err(IoError::InvalidFormat(format!(
                "NetCDF dimension '{}' has zero length",
                dimension.name
            )));
        }
        count = count.checked_mul(dim_len).ok_or_else(|| {
            IoError::InvalidFormat("NetCDF variable shape overflowed usize".to_string())
        })?;
    }
    Ok(count)
}

fn decode_netcdf_values(
    value_type: NetcdfType,
    payload: &[u8],
    count: usize,
    name: &str,
) -> Result<NetcdfValue, IoError> {
    let expected_len = netcdf_value_raw_len(value_type, count)?;
    if payload.len() < expected_len {
        return Err(IoError::InvalidFormat(format!(
            "NetCDF value '{name}' expected {expected_len} bytes but found {}",
            payload.len()
        )));
    }
    match value_type {
        NetcdfType::Byte => Ok(NetcdfValue::Byte(
            payload[..count].iter().map(|&byte| byte as i8).collect(),
        )),
        NetcdfType::Char => {
            let s = String::from_utf8(payload[..count].to_vec()).map_err(|e| {
                IoError::InvalidFormat(format!("NetCDF char value '{name}' is not UTF-8: {e}"))
            })?;
            Ok(NetcdfValue::Char(s))
        }
        NetcdfType::Short => {
            let mut values = Vec::with_capacity(count);
            for chunk in payload[..expected_len].chunks_exact(2) {
                values.push(i16::from_be_bytes([chunk[0], chunk[1]]));
            }
            Ok(NetcdfValue::Short(values))
        }
        NetcdfType::Int => {
            let mut values = Vec::with_capacity(count);
            for chunk in payload[..expected_len].chunks_exact(4) {
                values.push(i32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(NetcdfValue::Int(values))
        }
        NetcdfType::Float => {
            let mut values = Vec::with_capacity(count);
            for chunk in payload[..expected_len].chunks_exact(4) {
                values.push(f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(NetcdfValue::Float(values))
        }
        NetcdfType::Double => {
            let mut values = Vec::with_capacity(count);
            for chunk in payload[..expected_len].chunks_exact(8) {
                values.push(f64::from_be_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]));
            }
            Ok(NetcdfValue::Double(values))
        }
    }
}

fn validate_netcdf_name(name: &str, context: &str) -> Result<(), IoError> {
    if name.is_empty() {
        return Err(IoError::InvalidFormat(format!(
            "NetCDF {context} name cannot be empty"
        )));
    }
    if name.contains('\0') {
        return Err(IoError::InvalidFormat(format!(
            "NetCDF {context} name '{}' contains NUL",
            name.escape_debug()
        )));
    }
    Ok(())
}

fn validate_netcdf_fixed_file(file: &NetcdfFile) -> Result<(), IoError> {
    let unlimited_count = file
        .dimensions
        .iter()
        .filter(|dimension| dimension.len.is_none())
        .count();
    if unlimited_count > 0 {
        return Err(IoError::UnsupportedFeature(
            "NetCDF writer does not yet support unlimited record dimensions".to_string(),
        ));
    }
    for dimension in &file.dimensions {
        validate_netcdf_name(&dimension.name, "dimension")?;
        if dimension.len == Some(0) {
            return Err(IoError::InvalidFormat(format!(
                "NetCDF dimension '{}' has zero length",
                dimension.name
            )));
        }
    }
    for attribute in &file.attributes {
        validate_netcdf_name(&attribute.name, "attribute")?;
    }
    for variable in &file.variables {
        validate_netcdf_name(&variable.name, "variable")?;
        for attribute in &variable.attributes {
            validate_netcdf_name(&attribute.name, "attribute")?;
        }
        let expected = netcdf_element_count_for_dims(&file.dimensions, &variable.dim_ids, 0)?;
        if variable.data.len() != expected {
            return Err(IoError::InvalidFormat(format!(
                "NetCDF variable '{}' expected {expected} values from its dimensions but found {}",
                variable.name,
                variable.data.len()
            )));
        }
    }
    Ok(())
}

fn write_netcdf_u32(out: &mut Vec<u8>, value: usize, context: &str) -> Result<(), IoError> {
    let value = u32::try_from(value).map_err(|_| {
        IoError::InvalidFormat(format!("NetCDF {context} {value} exceeds u32 range"))
    })?;
    out.extend_from_slice(&value.to_be_bytes());
    Ok(())
}

fn write_netcdf_name(out: &mut Vec<u8>, name: &str, context: &str) -> Result<(), IoError> {
    validate_netcdf_name(name, context)?;
    write_netcdf_u32(out, name.len(), "name length")?;
    out.extend_from_slice(name.as_bytes());
    while !out.len().is_multiple_of(4) {
        out.push(0);
    }
    Ok(())
}

fn encode_netcdf_header(file: &NetcdfFile, begins: &[usize]) -> Result<Vec<u8>, IoError> {
    if begins.len() != file.variables.len() {
        return Err(IoError::InvalidFormat(
            "NetCDF begin-offset count does not match variable count".to_string(),
        ));
    }
    let mut out = Vec::new();
    out.extend_from_slice(b"CDF");
    out.push(1);
    out.extend_from_slice(&0u32.to_be_bytes());

    if file.dimensions.is_empty() {
        out.extend_from_slice(&0u32.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
    } else {
        out.extend_from_slice(&NC_DIMENSION.to_be_bytes());
        write_netcdf_u32(&mut out, file.dimensions.len(), "dimension count")?;
        for dimension in &file.dimensions {
            write_netcdf_name(&mut out, &dimension.name, "dimension")?;
            let len = dimension.len.ok_or_else(|| {
                IoError::UnsupportedFeature(
                    "NetCDF classic writer does not support unlimited dimensions".to_string(),
                )
            })?;
            write_netcdf_u32(&mut out, len, "dimension length")?;
        }
    }

    encode_netcdf_attribute_list(&mut out, &file.attributes)?;

    if file.variables.is_empty() {
        out.extend_from_slice(&0u32.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
    } else {
        out.extend_from_slice(&NC_VARIABLE.to_be_bytes());
        write_netcdf_u32(&mut out, file.variables.len(), "variable count")?;
        for (idx, variable) in file.variables.iter().enumerate() {
            write_netcdf_name(&mut out, &variable.name, "variable")?;
            write_netcdf_u32(&mut out, variable.dim_ids.len(), "variable dimension count")?;
            for &dim_id in &variable.dim_ids {
                if dim_id >= file.dimensions.len() {
                    return Err(IoError::InvalidFormat(format!(
                        "NetCDF variable '{}' references bad dimension id {dim_id}",
                        variable.name
                    )));
                }
                write_netcdf_u32(&mut out, dim_id, "variable dimension id")?;
            }
            encode_netcdf_attribute_list(&mut out, &variable.attributes)?;
            out.extend_from_slice(&netcdf_type_code(variable.data.value_type()).to_be_bytes());
            write_netcdf_u32(
                &mut out,
                encode_netcdf_padded_values(&variable.data)?.len(),
                "variable byte size",
            )?;
            write_netcdf_u32(&mut out, begins[idx], "variable begin offset")?;
        }
    }
    Ok(out)
}

fn encode_netcdf_attribute_list(
    out: &mut Vec<u8>,
    attributes: &[NetcdfAttribute],
) -> Result<(), IoError> {
    if attributes.is_empty() {
        out.extend_from_slice(&0u32.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
        return Ok(());
    }
    out.extend_from_slice(&NC_ATTRIBUTE.to_be_bytes());
    write_netcdf_u32(out, attributes.len(), "attribute count")?;
    for attribute in attributes {
        write_netcdf_name(out, &attribute.name, "attribute")?;
        out.extend_from_slice(&netcdf_type_code(attribute.value.value_type()).to_be_bytes());
        write_netcdf_u32(out, attribute.value.len(), "attribute value count")?;
        out.extend_from_slice(&encode_netcdf_padded_values(&attribute.value)?);
    }
    Ok(())
}

fn encode_netcdf_padded_values(value: &NetcdfValue) -> Result<Vec<u8>, IoError> {
    let mut out = Vec::new();
    match value {
        NetcdfValue::Byte(values) => {
            out.reserve(values.len());
            for &value in values {
                out.push(value as u8);
            }
        }
        NetcdfValue::Char(value) => out.extend_from_slice(value.as_bytes()),
        NetcdfValue::Short(values) => {
            out.reserve(values.len() * 2);
            for &value in values {
                out.extend_from_slice(&value.to_be_bytes());
            }
        }
        NetcdfValue::Int(values) => {
            out.reserve(values.len() * 4);
            for &value in values {
                out.extend_from_slice(&value.to_be_bytes());
            }
        }
        NetcdfValue::Float(values) => {
            out.reserve(values.len() * 4);
            for &value in values {
                out.extend_from_slice(&value.to_be_bytes());
            }
        }
        NetcdfValue::Double(values) => {
            out.reserve(values.len() * 8);
            for &value in values {
                out.extend_from_slice(&value.to_be_bytes());
            }
        }
    }
    while !out.len().is_multiple_of(4) {
        out.push(0);
    }
    Ok(out)
}

// ══════════════════════════════════════════════════════════════════════
// Harwell-Boeing sparse matrix format
// ══════════════════════════════════════════════════════════════════════

/// Harwell-Boeing matrix data type from the title-line "Type" code (RUA/RSA/etc).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HbType {
    /// "R" — real values, "U" — unsymmetric, "A" — assembled.
    RealUnsymmetricAssembled,
    /// "R" — real values, "S" — symmetric, "A" — assembled. Only the lower
    /// triangle is stored; callers wanting the full matrix must symmetrize.
    RealSymmetricAssembled,
}

/// Result of reading a Harwell-Boeing sparse matrix.
///
/// Returns the matrix in CSC (compressed sparse column) form: the canonical
/// on-disk Harwell-Boeing layout. `col_ptr` has `cols + 1` entries; row
/// indices and values are 0-indexed (the on-disk format is 1-indexed).
#[derive(Debug, Clone, PartialEq)]
pub struct HbMatrix {
    pub title: String,
    pub key: String,
    pub matrix_type: HbType,
    pub rows: usize,
    pub cols: usize,
    pub nnz: usize,
    pub col_ptr: Vec<usize>,
    pub row_idx: Vec<usize>,
    pub values: Vec<f64>,
}

/// Read a real, assembled Harwell-Boeing sparse matrix file.
///
/// Supports the dominant subset of `scipy.io.hb_read`: real (R), assembled (A),
/// either unsymmetric (U) or symmetric (S). The unassembled (E), pattern-only
/// (P), and complex (C) variants intentionally return an UnsupportedFeature
/// error and are tracked under the parent bead `frankenscipy-vsas0`.
///
/// The on-disk format uses 1-based indexing per the spec; the returned
/// `col_ptr` and `row_idx` are converted to 0-based for parity with the
/// rest of `fsci-io` and `fsci-sparse`.
pub fn read_harwell_boeing(content: &str) -> Result<HbMatrix, IoError> {
    let mut lines = content.lines();
    let title_line = lines.next().ok_or_else(|| {
        IoError::InvalidFormat("Harwell-Boeing file missing title line".to_string())
    })?;
    if title_line.len() < 72 {
        return Err(IoError::InvalidFormat(format!(
            "Harwell-Boeing title line must be ≥72 chars, got {}",
            title_line.len()
        )));
    }
    let title = title_line[..72].trim_end().to_string();
    let key = title_line[72..].trim().to_string();

    let totcrd_line = lines.next().ok_or_else(|| {
        IoError::InvalidFormat("Harwell-Boeing file missing totcrd line".to_string())
    })?;
    let counts = parse_hb_int_fields(totcrd_line, 5, "totcrd")?;
    let _totcrd = counts[0];
    let ptrcrd = counts[1] as usize;
    let indcrd = counts[2] as usize;
    let valcrd = counts[3] as usize;
    let rhscrd = counts[4] as usize;

    let mxtype_line = lines.next().ok_or_else(|| {
        IoError::InvalidFormat("Harwell-Boeing file missing mxtype line".to_string())
    })?;
    let mxtype_field = mxtype_line.get(..3).map(str::trim_start).unwrap_or("");
    let dims = parse_hb_int_fields(&mxtype_line[3.min(mxtype_line.len())..], 4, "mxtype")?;
    let rows = dims[0] as usize;
    let cols = dims[1] as usize;
    let nnz = dims[2] as usize;
    let neltvl = dims[3];

    let matrix_type = match mxtype_field {
        "RUA" => HbType::RealUnsymmetricAssembled,
        "RSA" => HbType::RealSymmetricAssembled,
        other => {
            return Err(IoError::UnsupportedFeature(format!(
                "Harwell-Boeing type '{other}' not supported (only RUA and RSA today; \
                 see frankenscipy-vsas0 for follow-on)"
            )));
        }
    };

    if neltvl != 0 {
        return Err(IoError::UnsupportedFeature(
            "Harwell-Boeing unassembled (NELTVL != 0) is not supported".to_string(),
        ));
    }

    // Format-line records — we don't need to parse Fortran format specs because
    // the value/index lists are whitespace-tolerant when read as flat token streams.
    // SciPy is similarly relaxed in hb_read for fixed-format real assembled files.
    let _ptrfmt = lines
        .next()
        .ok_or_else(|| IoError::InvalidFormat("Harwell-Boeing missing ptrfmt line".to_string()))?;
    let _indfmt = lines
        .next()
        .ok_or_else(|| IoError::InvalidFormat("Harwell-Boeing missing indfmt line".to_string()))?;
    let _valfmt = lines
        .next()
        .ok_or_else(|| IoError::InvalidFormat("Harwell-Boeing missing valfmt line".to_string()))?;
    if valcrd == 0 {
        return Err(IoError::UnsupportedFeature(
            "Harwell-Boeing pattern-only (valcrd == 0) is not supported".to_string(),
        ));
    }

    // Optional rhsfmt line iff rhscrd > 0; we don't consume an RHS payload.
    let mut remaining: Vec<&str> = lines.collect();
    if rhscrd > 0 && !remaining.is_empty() {
        // First rhsfmt line; we then ignore the RHS payload past the matrix data.
        remaining.remove(0);
    }

    // ptrcrd lines hold col_ptr (cols+1 ints), then indcrd lines row_idx (nnz ints),
    // then valcrd lines hold the values (nnz reals). All whitespace-separated within
    // each card group.
    if remaining.len() < ptrcrd + indcrd + valcrd {
        return Err(IoError::InvalidFormat(format!(
            "Harwell-Boeing payload truncated: need {} cards, got {}",
            ptrcrd + indcrd + valcrd,
            remaining.len()
        )));
    }
    let ptr_lines = &remaining[..ptrcrd];
    let ind_lines = &remaining[ptrcrd..ptrcrd + indcrd];
    let val_lines = &remaining[ptrcrd + indcrd..ptrcrd + indcrd + valcrd];

    let raw_col_ptr = parse_hb_int_stream(ptr_lines, cols + 1, "col_ptr")?;
    let raw_row_idx = parse_hb_int_stream(ind_lines, nnz, "row_idx")?;
    let raw_values = parse_hb_real_stream(val_lines, nnz, "values")?;

    // Convert 1-based → 0-based and sanity-check.
    let mut col_ptr = Vec::with_capacity(cols + 1);
    for (i, &p) in raw_col_ptr.iter().enumerate() {
        if p < 1 {
            return Err(IoError::InvalidFormat(format!(
                "Harwell-Boeing col_ptr[{i}] = {p} is below 1 (1-based input expected)"
            )));
        }
        col_ptr.push((p - 1) as usize);
    }
    if col_ptr[0] != 0 {
        return Err(IoError::InvalidFormat(format!(
            "Harwell-Boeing col_ptr[0] = {} (after 0-based shift) but must be 0",
            col_ptr[0]
        )));
    }
    if *col_ptr.last().unwrap() != nnz {
        return Err(IoError::InvalidFormat(format!(
            "Harwell-Boeing col_ptr[last] = {} but nnz = {nnz}",
            col_ptr.last().unwrap()
        )));
    }
    let mut row_idx = Vec::with_capacity(nnz);
    for (i, &r) in raw_row_idx.iter().enumerate() {
        if r < 1 || (r as usize) > rows {
            return Err(IoError::InvalidFormat(format!(
                "Harwell-Boeing row_idx[{i}] = {r} out of range [1, {rows}]"
            )));
        }
        row_idx.push((r - 1) as usize);
    }

    Ok(HbMatrix {
        title,
        key,
        matrix_type,
        rows,
        cols,
        nnz,
        col_ptr,
        row_idx,
        values: raw_values,
    })
}

fn parse_hb_int_fields(line: &str, expected: usize, ctx: &str) -> Result<Vec<i64>, IoError> {
    let toks: Vec<i64> = line
        .split_whitespace()
        .map(|s| s.parse::<i64>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| IoError::InvalidFormat(format!("Harwell-Boeing {ctx}: {e}")))?;
    if toks.len() < expected {
        return Err(IoError::InvalidFormat(format!(
            "Harwell-Boeing {ctx} line expected {expected} integers, got {}",
            toks.len()
        )));
    }
    Ok(toks.into_iter().take(expected).collect())
}

fn parse_hb_int_stream(lines: &[&str], expected: usize, ctx: &str) -> Result<Vec<i64>, IoError> {
    let mut out = Vec::with_capacity(expected);
    for line in lines {
        for tok in line.split_whitespace() {
            let v: i64 = tok
                .parse()
                .map_err(|e| IoError::InvalidFormat(format!("Harwell-Boeing {ctx}: {e}")))?;
            out.push(v);
        }
    }
    if out.len() < expected {
        return Err(IoError::InvalidFormat(format!(
            "Harwell-Boeing {ctx} expected {expected} integers, got {}",
            out.len()
        )));
    }
    out.truncate(expected);
    Ok(out)
}

fn parse_hb_real_stream(lines: &[&str], expected: usize, ctx: &str) -> Result<Vec<f64>, IoError> {
    // Harwell-Boeing values are written in Fortran D-format (e.g. "1.0D+00").
    // Tokenize on whitespace and rewrite the Fortran exponent marker before parsing.
    let mut out = Vec::with_capacity(expected);
    for line in lines {
        for tok in line.split_whitespace() {
            let normalized = tok.replace('D', "E").replace('d', "e");
            let v: f64 = normalized
                .parse()
                .map_err(|e| IoError::InvalidFormat(format!("Harwell-Boeing {ctx}: {e}")))?;
            out.push(v);
        }
    }
    if out.len() < expected {
        return Err(IoError::InvalidFormat(format!(
            "Harwell-Boeing {ctx} expected {expected} reals, got {}",
            out.len()
        )));
    }
    out.truncate(expected);
    Ok(out)
}

// ══════════════════════════════════════════════════════════════════════
// Fortran sequential unformatted binary records
// ══════════════════════════════════════════════════════════════════════

/// Endianness of the i32 length-marker words framing each record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FortranEndian {
    Little,
    Big,
}

/// Read a Fortran sequential unformatted file.
///
/// Each record on disk is framed as `<len:i32><payload:len><len:i32>` with
/// the leading and trailing length words matching. Returns the concatenated
/// list of payloads in order. Matches the most common subset of
/// `scipy.io.FortranFile.read_record()` driven across the whole file.
///
/// Errors:
/// - `InvalidFormat` if a record header is truncated or the trailing length
///   does not match the leading length.
/// - `InvalidFormat` if the input contains trailing bytes that do not form
///   a complete record.
pub fn read_fortran_unformatted(
    bytes: &[u8],
    endian: FortranEndian,
) -> Result<Vec<Vec<u8>>, IoError> {
    let mut records = Vec::new();
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        let header_end = cursor
            .checked_add(4)
            .ok_or_else(|| IoError::InvalidFormat("Fortran record offset overflow".to_string()))?;
        if header_end > bytes.len() {
            return Err(IoError::InvalidFormat(format!(
                "Fortran record at offset {cursor}: header truncated ({} bytes remaining)",
                bytes.len() - cursor
            )));
        }
        let header_bytes: [u8; 4] = bytes[cursor..header_end].try_into().expect("4-byte slice");
        let length = match endian {
            FortranEndian::Little => i32::from_le_bytes(header_bytes),
            FortranEndian::Big => i32::from_be_bytes(header_bytes),
        };
        if length < 0 {
            return Err(IoError::InvalidFormat(format!(
                "Fortran record at offset {cursor}: negative length {length}"
            )));
        }
        let length = length as usize;
        let payload_end = header_end
            .checked_add(length)
            .ok_or_else(|| IoError::InvalidFormat("Fortran payload offset overflow".into()))?;
        let trailer_end = payload_end
            .checked_add(4)
            .ok_or_else(|| IoError::InvalidFormat("Fortran trailer offset overflow".into()))?;
        if trailer_end > bytes.len() {
            return Err(IoError::InvalidFormat(format!(
                "Fortran record at offset {cursor}: payload+trailer truncated \
                 (need {trailer_end}, have {})",
                bytes.len()
            )));
        }
        let payload = bytes[header_end..payload_end].to_vec();
        let trailer_bytes: [u8; 4] = bytes[payload_end..trailer_end]
            .try_into()
            .expect("4-byte slice");
        let trailer = match endian {
            FortranEndian::Little => i32::from_le_bytes(trailer_bytes),
            FortranEndian::Big => i32::from_be_bytes(trailer_bytes),
        };
        if trailer as usize != length {
            return Err(IoError::InvalidFormat(format!(
                "Fortran record at offset {cursor}: header length {length} does not match \
                 trailer {trailer}"
            )));
        }
        records.push(payload);
        cursor = trailer_end;
    }
    Ok(records)
}

/// Frame a payload with the Fortran sequential unformatted header+trailer.
/// Useful for building fixtures and for round-trip tests.
pub fn write_fortran_record(payload: &[u8], endian: FortranEndian) -> Vec<u8> {
    let length = payload.len() as i32;
    let length_bytes = match endian {
        FortranEndian::Little => length.to_le_bytes(),
        FortranEndian::Big => length.to_be_bytes(),
    };
    let mut out = Vec::with_capacity(payload.len() + 8);
    out.extend_from_slice(&length_bytes);
    out.extend_from_slice(payload);
    out.extend_from_slice(&length_bytes);
    out
}

// ══════════════════════════════════════════════════════════════════════
// ARFF (Weka attribute-relation file format)
// ══════════════════════════════════════════════════════════════════════

/// ARFF attribute type. Relational attributes are intentionally not supported
/// and return `UnsupportedFeature`.
#[derive(Debug, Clone, PartialEq)]
pub enum ArffAttribute {
    /// `@attribute name numeric` / `real` / `integer`.
    Numeric { name: String },
    /// `@attribute name {a, b, c}`. Domain is the ordered list of allowed
    /// nominal values (case-sensitive).
    Nominal { name: String, domain: Vec<String> },
    /// `@attribute name string`. Free-form text.
    String { name: String },
    /// `@attribute name date "yyyy-MM-dd"`. Format uses Weka's Java-style
    /// `SimpleDateFormat` subset, matching `scipy.io.arff` for simple fields.
    Date {
        name: String,
        format: String,
        unit: ArffDateUnit,
    },
}

impl ArffAttribute {
    pub fn name(&self) -> &str {
        match self {
            Self::Numeric { name }
            | Self::Nominal { name, .. }
            | Self::String { name }
            | Self::Date { name, .. } => name,
        }
    }
}

/// Precision selected for an ARFF date attribute. This mirrors the numpy
/// `datetime64` unit SciPy chooses from the declared format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ArffDateUnit {
    Year,
    Month,
    Day,
    Hour,
    Minute,
    Second,
}

/// Parsed ARFF date value. `raw` preserves the input token after unquoting;
/// `normalized` matches numpy `datetime64` display at the selected unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArffDateTime {
    pub raw: String,
    pub normalized: String,
    pub unit: ArffDateUnit,
}

/// One ARFF data cell. Missing values are encoded as `?` in the file and
/// surface as `Missing` here.
#[derive(Debug, Clone, PartialEq)]
pub enum ArffValue {
    Numeric(f64),
    Nominal(String),
    String(String),
    Date(ArffDateTime),
    Missing,
}

/// Parsed ARFF file. Sparse rows are expanded to their dense equivalent
/// (missing positions filled with the type-appropriate zero/empty value)
/// for parity with `scipy.io.arff.loadarff`.
#[derive(Debug, Clone, PartialEq)]
pub struct ArffData {
    pub relation: String,
    pub attributes: Vec<ArffAttribute>,
    pub rows: Vec<Vec<ArffValue>>,
}

/// Read an ARFF (Weka attribute-relation file format) string.
///
/// Supports numeric, nominal, string, and simple date attributes; comments
/// (`%`); sparse rows (`{ idx val, idx val }`); single- and double-quoted
/// values. Relational attributes return `UnsupportedFeature`.
pub fn read_arff(content: &str) -> Result<ArffData, IoError> {
    let mut relation: Option<String> = None;
    let mut attributes: Vec<ArffAttribute> = Vec::new();
    let mut rows: Vec<Vec<ArffValue>> = Vec::new();
    let mut in_data_section = false;

    for raw_line in content.lines() {
        let line = strip_arff_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }

        if !in_data_section {
            let lower = line.to_ascii_lowercase();
            if lower.starts_with("@relation") {
                relation = Some(unquote_arff(line["@relation".len()..].trim()));
            } else if lower.starts_with("@attribute") {
                attributes.push(parse_arff_attribute(line)?);
            } else if lower == "@data" {
                in_data_section = true;
            } else {
                return Err(IoError::InvalidFormat(format!(
                    "ARFF: unexpected directive in header: {line}"
                )));
            }
        } else {
            rows.push(parse_arff_data_row(line, &attributes)?);
        }
    }

    let relation = relation
        .ok_or_else(|| IoError::InvalidFormat("ARFF: missing @relation header".to_string()))?;
    if attributes.is_empty() {
        return Err(IoError::InvalidFormat(
            "ARFF: at least one @attribute is required".to_string(),
        ));
    }
    Ok(ArffData {
        relation,
        attributes,
        rows,
    })
}

fn strip_arff_comment(line: &str) -> &str {
    // Comments start with '%' but not inside quotes. ARFF doesn't escape
    // '%' inside quotes by anything other than scanning past the quote.
    let bytes = line.as_bytes();
    let mut in_single = false;
    let mut in_double = false;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'\'' if !in_double => in_single = !in_single,
            b'"' if !in_single => in_double = !in_double,
            b'%' if !in_single && !in_double => return &line[..i],
            _ => {}
        }
    }
    line
}

fn unquote_arff(raw: &str) -> String {
    let trimmed = raw.trim();
    if (trimmed.starts_with('\'') && trimmed.ends_with('\'') && trimmed.len() >= 2)
        || (trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2)
    {
        trimmed[1..trimmed.len() - 1].to_string()
    } else {
        trimmed.to_string()
    }
}

fn parse_arff_attribute(line: &str) -> Result<ArffAttribute, IoError> {
    // After '@attribute', tokens: name TYPE_OR_DOMAIN
    let body = &line[10.min(line.len())..].trim_start();
    let (name, rest) = split_arff_first_token(body)?;
    let name = unquote_arff(&name);
    let rest = rest.trim();
    if rest.starts_with('{') {
        let close = rest.find('}').ok_or_else(|| {
            IoError::InvalidFormat(format!("ARFF nominal attribute '{name}': missing '}}'"))
        })?;
        let domain: Vec<String> = rest[1..close]
            .split(',')
            .map(|s| unquote_arff(s.trim()))
            .filter(|s| !s.is_empty())
            .collect();
        if domain.is_empty() {
            return Err(IoError::InvalidFormat(format!(
                "ARFF nominal attribute '{name}': empty domain"
            )));
        }
        Ok(ArffAttribute::Nominal { name, domain })
    } else {
        let type_part = rest
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();
        match type_part.as_str() {
            "numeric" | "real" | "integer" => Ok(ArffAttribute::Numeric { name }),
            "string" => Ok(ArffAttribute::String { name }),
            "date" => {
                let (format, unit) = parse_arff_date_format(&name, rest)?;
                Ok(ArffAttribute::Date { name, format, unit })
            }
            "relational" => Err(IoError::UnsupportedFeature(format!(
                "ARFF relational attribute '{name}' (frankenscipy-vsas0 follow-on)"
            ))),
            other => Err(IoError::InvalidFormat(format!(
                "ARFF attribute '{name}': unknown type '{other}'"
            ))),
        }
    }
}

fn split_arff_first_token(input: &str) -> Result<(String, &str), IoError> {
    let bytes = input.as_bytes();
    if bytes.is_empty() {
        return Err(IoError::InvalidFormat(
            "ARFF: missing attribute name".into(),
        ));
    }
    let (end, _quote) = match bytes[0] {
        b'\'' | b'"' => {
            let quote = bytes[0];
            let mut i = 1usize;
            while i < bytes.len() && bytes[i] != quote {
                i += 1;
            }
            if i >= bytes.len() {
                return Err(IoError::InvalidFormat(
                    "ARFF: unterminated quoted attribute name".into(),
                ));
            }
            (i + 1, Some(quote))
        }
        _ => (
            bytes
                .iter()
                .position(|b| b.is_ascii_whitespace())
                .unwrap_or(bytes.len()),
            None,
        ),
    };
    Ok((input[..end].to_string(), &input[end..]))
}

fn parse_arff_data_row(
    line: &str,
    attributes: &[ArffAttribute],
) -> Result<Vec<ArffValue>, IoError> {
    let trimmed = line.trim();
    if trimmed.starts_with('{') {
        let close = trimmed
            .find('}')
            .ok_or_else(|| IoError::InvalidFormat("ARFF sparse row: missing '}'".into()))?;
        let mut row = vec![ArffValue::Missing; attributes.len()];
        // Initialize numeric cells to 0.0 per ARFF sparse semantics.
        for (i, attr) in attributes.iter().enumerate() {
            if matches!(attr, ArffAttribute::Numeric { .. }) {
                row[i] = ArffValue::Numeric(0.0);
            }
        }
        for entry in trimmed[1..close].split(',') {
            let entry = entry.trim();
            if entry.is_empty() {
                continue;
            }
            let mut parts = entry.splitn(2, char::is_whitespace);
            let idx_part = parts.next().unwrap_or("");
            let value_part = parts.next().unwrap_or("").trim();
            let idx: usize = idx_part.parse().map_err(|e| {
                IoError::InvalidFormat(format!("ARFF sparse index '{idx_part}': {e}"))
            })?;
            if idx >= attributes.len() {
                return Err(IoError::InvalidFormat(format!(
                    "ARFF sparse index {idx} out of range for {} attributes",
                    attributes.len()
                )));
            }
            row[idx] = parse_arff_cell(value_part, &attributes[idx])?;
        }
        Ok(row)
    } else {
        let cells = split_arff_csv_cells(trimmed);
        if cells.len() != attributes.len() {
            return Err(IoError::InvalidFormat(format!(
                "ARFF dense row: got {} cells but {} attributes declared",
                cells.len(),
                attributes.len()
            )));
        }
        cells
            .iter()
            .zip(attributes.iter())
            .map(|(cell, attr)| parse_arff_cell(cell.trim(), attr))
            .collect()
    }
}

fn split_arff_csv_cells(line: &str) -> Vec<String> {
    let mut cells = Vec::new();
    let mut current = String::new();
    let mut in_single = false;
    let mut in_double = false;
    for ch in line.chars() {
        match ch {
            '\'' if !in_double => {
                in_single = !in_single;
                current.push(ch);
            }
            '"' if !in_single => {
                in_double = !in_double;
                current.push(ch);
            }
            ',' if !in_single && !in_double => {
                cells.push(std::mem::take(&mut current));
            }
            _ => current.push(ch),
        }
    }
    cells.push(current);
    cells
}

fn parse_arff_cell(token: &str, attribute: &ArffAttribute) -> Result<ArffValue, IoError> {
    if matches!(token.as_bytes(), [b'?']) {
        return Ok(ArffValue::Missing);
    }
    match attribute {
        ArffAttribute::Numeric { name } => {
            let v: f64 = token.parse().map_err(|e| {
                IoError::InvalidFormat(format!("ARFF numeric '{name}' value '{token}': {e}"))
            })?;
            Ok(ArffValue::Numeric(v))
        }
        ArffAttribute::Nominal { name, domain } => {
            let unquoted = unquote_arff(token);
            if !domain.contains(&unquoted) {
                return Err(IoError::InvalidFormat(format!(
                    "ARFF nominal '{name}': value '{unquoted}' not in domain"
                )));
            }
            Ok(ArffValue::Nominal(unquoted))
        }
        ArffAttribute::String { .. } => Ok(ArffValue::String(unquote_arff(token))),
        ArffAttribute::Date { name, format, unit } => {
            parse_arff_date_cell(token, name, format, *unit)
        }
    }
}

fn parse_arff_date_format(
    name: &str,
    attribute_spec: &str,
) -> Result<(String, ArffDateUnit), IoError> {
    let raw_format = attribute_spec["date".len()..].trim();
    let format = unquote_arff(raw_format);
    if format.is_empty() {
        return Err(IoError::InvalidFormat(format!(
            "ARFF date attribute '{name}': missing date format"
        )));
    }
    if format.contains('z') || format.contains('Z') {
        return Err(IoError::UnsupportedFeature(format!(
            "ARFF date attribute '{name}': timezone formats are not supported"
        )));
    }

    let mut unit = None;
    let mut i = 0usize;
    while i < format.len() {
        let rest = &format[i..];
        if rest.starts_with("yyyy") {
            unit = Some(max_arff_date_unit(unit, ArffDateUnit::Year));
            i += 4;
        } else if rest.starts_with("yy") {
            unit = Some(max_arff_date_unit(unit, ArffDateUnit::Year));
            i += 2;
        } else if rest.starts_with("MM") {
            unit = Some(max_arff_date_unit(unit, ArffDateUnit::Month));
            i += 2;
        } else if rest.starts_with("dd") {
            unit = Some(max_arff_date_unit(unit, ArffDateUnit::Day));
            i += 2;
        } else if rest.starts_with("HH") {
            unit = Some(max_arff_date_unit(unit, ArffDateUnit::Hour));
            i += 2;
        } else if rest.starts_with("mm") {
            unit = Some(max_arff_date_unit(unit, ArffDateUnit::Minute));
            i += 2;
        } else if rest.starts_with("ss") {
            unit = Some(max_arff_date_unit(unit, ArffDateUnit::Second));
            i += 2;
        } else {
            let ch = rest.chars().next().expect("nonempty format tail");
            i += ch.len_utf8();
        }
    }

    let unit = unit.ok_or_else(|| {
        IoError::InvalidFormat(format!(
            "ARFF date attribute '{name}': invalid or unsupported date format '{format}'"
        ))
    })?;
    Ok((format, unit))
}

fn max_arff_date_unit(current: Option<ArffDateUnit>, candidate: ArffDateUnit) -> ArffDateUnit {
    current.map_or(candidate, |unit| unit.max(candidate))
}

fn parse_arff_date_cell(
    token: &str,
    name: &str,
    format: &str,
    unit: ArffDateUnit,
) -> Result<ArffValue, IoError> {
    let raw = unquote_arff(token);
    let components = parse_arff_date_components(&raw, format).map_err(|message| {
        IoError::InvalidFormat(format!("ARFF date '{name}' value '{raw}': {message}"))
    })?;
    let normalized = normalize_arff_date(&components, unit).map_err(|message| {
        IoError::InvalidFormat(format!("ARFF date '{name}' value '{raw}': {message}"))
    })?;
    Ok(ArffValue::Date(ArffDateTime {
        raw,
        normalized,
        unit,
    }))
}

#[derive(Debug, Default)]
struct ArffDateComponents {
    year: Option<i32>,
    month: Option<u8>,
    day: Option<u8>,
    hour: Option<u8>,
    minute: Option<u8>,
    second: Option<u8>,
}

fn parse_arff_date_components(value: &str, format: &str) -> Result<ArffDateComponents, String> {
    let mut components = ArffDateComponents::default();
    let mut format_pos = 0usize;
    let mut value_pos = 0usize;

    while format_pos < format.len() {
        let rest = &format[format_pos..];
        if rest.starts_with("yyyy") {
            components.year = Some(read_arff_date_number(value, &mut value_pos, 4)?);
            format_pos += 4;
        } else if rest.starts_with("yy") {
            let year: i32 = read_arff_date_number(value, &mut value_pos, 2)?;
            components.year = Some(if year <= 68 { 2000 + year } else { 1900 + year });
            format_pos += 2;
        } else if rest.starts_with("MM") {
            components.month = Some(read_arff_date_number(value, &mut value_pos, 2)?);
            format_pos += 2;
        } else if rest.starts_with("dd") {
            components.day = Some(read_arff_date_number(value, &mut value_pos, 2)?);
            format_pos += 2;
        } else if rest.starts_with("HH") {
            components.hour = Some(read_arff_date_number(value, &mut value_pos, 2)?);
            format_pos += 2;
        } else if rest.starts_with("mm") {
            components.minute = Some(read_arff_date_number(value, &mut value_pos, 2)?);
            format_pos += 2;
        } else if rest.starts_with("ss") {
            components.second = Some(read_arff_date_number(value, &mut value_pos, 2)?);
            format_pos += 2;
        } else if rest.starts_with('\'') {
            format_pos += 1;
            while format_pos < format.len() && !format[format_pos..].starts_with('\'') {
                let ch = format[format_pos..]
                    .chars()
                    .next()
                    .expect("nonempty quoted format literal");
                consume_arff_date_literal(value, &mut value_pos, ch)?;
                format_pos += ch.len_utf8();
            }
            if format_pos >= format.len() {
                return Err("unterminated quoted literal in date format".to_string());
            }
            format_pos += 1;
        } else {
            let ch = rest.chars().next().expect("nonempty date format literal");
            consume_arff_date_literal(value, &mut value_pos, ch)?;
            format_pos += ch.len_utf8();
        }
    }

    if value_pos != value.len() {
        return Err(format!("trailing input '{}'", &value[value_pos..]));
    }

    Ok(components)
}

fn read_arff_date_number<T>(value: &str, value_pos: &mut usize, width: usize) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    if *value_pos + width > value.len() {
        return Err(format!("expected {width} digits"));
    }
    let field = &value[*value_pos..*value_pos + width];
    if !field.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!("expected {width} digits, got '{field}'"));
    }
    *value_pos += width;
    field
        .parse()
        .map_err(|e| format!("invalid number '{field}': {e}"))
}

fn consume_arff_date_literal(
    value: &str,
    value_pos: &mut usize,
    expected: char,
) -> Result<(), String> {
    let actual = value[*value_pos..]
        .chars()
        .next()
        .ok_or_else(|| format!("expected literal '{expected}'"))?;
    if actual != expected {
        return Err(format!("expected literal '{expected}', got '{actual}'"));
    }
    *value_pos += actual.len_utf8();
    Ok(())
}

fn normalize_arff_date(
    components: &ArffDateComponents,
    unit: ArffDateUnit,
) -> Result<String, String> {
    let year = components
        .year
        .ok_or_else(|| "date format must include a year".to_string())?;
    let month = components.month.unwrap_or(1);
    let day = components.day.unwrap_or(1);
    let hour = components.hour.unwrap_or(0);
    let minute = components.minute.unwrap_or(0);
    let second = components.second.unwrap_or(0);

    validate_arff_date_components(year, month, day, hour, minute, second)?;

    match unit {
        ArffDateUnit::Year => Ok(format!("{year:04}")),
        ArffDateUnit::Month => {
            require_arff_date_component(components.month, "month")?;
            Ok(format!("{year:04}-{month:02}"))
        }
        ArffDateUnit::Day => {
            require_arff_date_component(components.month, "month")?;
            require_arff_date_component(components.day, "day")?;
            Ok(format!("{year:04}-{month:02}-{day:02}"))
        }
        ArffDateUnit::Hour => {
            require_arff_date_component(components.month, "month")?;
            require_arff_date_component(components.day, "day")?;
            require_arff_date_component(components.hour, "hour")?;
            Ok(format!("{year:04}-{month:02}-{day:02}T{hour:02}"))
        }
        ArffDateUnit::Minute => {
            require_arff_date_component(components.month, "month")?;
            require_arff_date_component(components.day, "day")?;
            require_arff_date_component(components.hour, "hour")?;
            require_arff_date_component(components.minute, "minute")?;
            Ok(format!(
                "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}"
            ))
        }
        ArffDateUnit::Second => {
            require_arff_date_component(components.month, "month")?;
            require_arff_date_component(components.day, "day")?;
            require_arff_date_component(components.hour, "hour")?;
            require_arff_date_component(components.minute, "minute")?;
            require_arff_date_component(components.second, "second")?;
            Ok(format!(
                "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}"
            ))
        }
    }
}

fn require_arff_date_component<T>(component: Option<T>, name: &str) -> Result<(), String> {
    if component.is_some() {
        Ok(())
    } else {
        Err(format!("date format is missing required {name} field"))
    }
}

fn validate_arff_date_components(
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: u8,
) -> Result<(), String> {
    if !(1..=12).contains(&month) {
        return Err(format!("month {month} out of range"));
    }
    let max_day = days_in_arff_month(year, month);
    if !(1..=max_day).contains(&day) {
        return Err(format!("day {day} out of range for month {month}"));
    }
    if hour > 23 {
        return Err(format!("hour {hour} out of range"));
    }
    if minute > 59 {
        return Err(format!("minute {minute} out of range"));
    }
    if second > 59 {
        return Err(format!("second {second} out of range"));
    }
    Ok(())
}

fn days_in_arff_month(year: i32, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_arff_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

fn is_arff_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mmread_coordinate() {
        let content = "%%MatrixMarket matrix coordinate real general\n\
                        3 3 3\n\
                        1 1 1.0\n\
                        2 2 2.0\n\
                        3 3 3.0\n";
        let mat = mmread(content).unwrap();
        assert_eq!(mat.rows, 3);
        assert_eq!(mat.cols, 3);
        assert_eq!(mat.data[0], 1.0); // (0,0)
        assert_eq!(mat.data[4], 2.0); // (1,1)
        assert_eq!(mat.data[8], 3.0); // (2,2)
        assert_eq!(mat.data[1], 0.0); // off-diagonal is zero
    }

    #[test]
    fn mmread_symmetric() {
        let content = "%%MatrixMarket matrix coordinate real symmetric\n\
                        3 3 2\n\
                        1 1 5.0\n\
                        2 1 3.0\n";
        let mat = mmread(content).unwrap();
        assert_eq!(mat.data[0], 5.0); // (0,0)
        assert_eq!(mat.data[3], 3.0); // (1,0)
        assert_eq!(mat.data[1], 3.0); // (0,1) = symmetric
    }

    #[test]
    fn mmread_coordinate_sums_duplicates() {
        let content = "%%MatrixMarket matrix coordinate real general\n\
                        2 2 3\n\
                        1 1 1.0\n\
                        1 1 2.5\n\
                        2 1 -1.0\n";
        let mat = mmread(content).unwrap();
        assert_eq!(mat.data[0], 3.5);
        assert_eq!(mat.data[2], -1.0);
        assert_eq!(mat.data[1], 0.0);
    }

    #[test]
    fn mmread_skew_symmetric() {
        let content = "%%MatrixMarket matrix coordinate real skew-symmetric\n\
                        3 3 1\n\
                        1 3 2.0\n";
        let mat = mmread(content).unwrap();
        assert_eq!(mat.data[2], 2.0); // (0,2)
        assert_eq!(mat.data[6], -2.0); // (2,0)
    }

    #[test]
    fn mmread_skew_symmetric_rejects_nonzero_diagonal() {
        let content = "%%MatrixMarket matrix coordinate real skew-symmetric\n\
                        2 2 1\n\
                        1 1 1.0\n";
        let err = mmread(content).expect_err("skew-symmetric diagonal must be zero");
        assert_eq!(
            err,
            IoError::InvalidFormat("skew-symmetric diagonal entries must be zero".to_string())
        );
    }

    #[test]
    fn mmread_rejects_rectangular_symmetric_coordinate_matrix() {
        let content = "%%MatrixMarket matrix coordinate real symmetric\n\
                        2 3 1\n\
                        1 3 2.0\n";
        let err = mmread(content).expect_err("rectangular symmetric Matrix Market should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "Matrix Market symmetric symmetry requires a square matrix, got 2x3".to_string()
            )
        );
    }

    #[test]
    fn mmread_rejects_zero_based_coordinate_indices() {
        let content = "%%MatrixMarket matrix coordinate real general\n\
                        3 3 1\n\
                        0 1 5.0\n";
        let err = mmread(content).expect_err("zero-based row index should be rejected");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "Matrix Market row indices must be 1-based and >= 1".to_string()
            )
        );
    }

    #[test]
    fn mmread_rejects_out_of_bounds_coordinate_indices() {
        let content = "%%MatrixMarket matrix coordinate real general\n\
                        3 3 1\n\
                        4 1 5.0\n";
        let err = mmread(content).expect_err("out-of-bounds row index should be rejected");
        assert_eq!(
            err,
            IoError::InvalidFormat("coordinate entry (3, 0) out of bounds for 3x3".to_string())
        );
    }

    #[test]
    fn mmread_rejects_coordinate_nnz_mismatch() {
        let content = "%%MatrixMarket matrix coordinate real general\n\
                        3 3 2\n\
                        1 1 1.0\n";
        let err = mmread(content).expect_err("declared nnz mismatch should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("coordinate format expected 2 entries but found 1".to_string())
        );
    }

    #[test]
    fn mmread_rejects_missing_coordinate_value_for_real_field() {
        let content = "%%MatrixMarket matrix coordinate real general\n\
                        3 3 1\n\
                        1 1\n";
        let err = mmread(content).expect_err("real coordinate entries need explicit values");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "coordinate entry missing value for non-pattern field".to_string()
            )
        );
    }

    #[test]
    fn mmread_rejects_complex_coordinate_payloads() {
        let content = "%%MatrixMarket matrix coordinate complex general\n\
                        2 2 1\n\
                        1 1 1.0 2.0\n";
        let err = mmread(content).expect_err("complex Matrix Market payloads should fail closed");
        assert_eq!(
            err,
            IoError::UnsupportedFeature("Matrix Market complex field is not supported".to_string())
        );
    }

    #[test]
    fn mmread_array() {
        let content = "%%MatrixMarket matrix array real general\n\
                        2 3\n\
                        1.0\n2.0\n3.0\n4.0\n5.0\n6.0\n";
        let mat = mmread(content).unwrap();
        assert_eq!(mat.rows, 2);
        assert_eq!(mat.cols, 3);
        // Column-major: 1,2 are col 0, 3,4 are col 1, 5,6 are col 2
        assert_eq!(mat.data[0], 1.0); // (0,0)
        assert_eq!(mat.data[3], 2.0); // (1,0)
        assert_eq!(mat.data[1], 3.0); // (0,1)
    }

    #[test]
    fn mminfo_reads_coordinate_header_without_body() {
        let content = "%%MatrixMarket matrix coordinate real general\n\
                        3 4 2\n";
        let info = mminfo(content).expect("header-only coordinate metadata should parse");
        assert_eq!(info.object, MmObject::Matrix);
        assert_eq!(info.format, MmFormat::Coordinate);
        assert_eq!(info.field, MmField::Real);
        assert_eq!(info.symmetry, MmSymmetry::General);
        assert_eq!(info.rows, 3);
        assert_eq!(info.cols, 4);
        assert_eq!(info.nnz, 2);
    }

    #[test]
    fn mminfo_reads_array_header_without_body() {
        let content = "%%MatrixMarket matrix array real general\n\
                        2 3\n";
        let info = mminfo(content).expect("header-only array metadata should parse");
        assert_eq!(info.object, MmObject::Matrix);
        assert_eq!(info.format, MmFormat::Array);
        assert_eq!(info.field, MmField::Real);
        assert_eq!(info.symmetry, MmSymmetry::General);
        assert_eq!(info.rows, 2);
        assert_eq!(info.cols, 3);
        assert_eq!(info.nnz, 6);
    }

    #[test]
    fn mmread_rejects_complex_array_payloads() {
        let content = "%%MatrixMarket matrix array complex general\n\
                        2 1\n\
                        1.0 2.0\n\
                        3.0 4.0\n";
        let err = mmread(content).expect_err("complex Matrix Market payloads should fail closed");
        assert_eq!(
            err,
            IoError::UnsupportedFeature("Matrix Market complex field is not supported".to_string())
        );
    }

    #[test]
    fn mmread_array_rejects_too_few_values() {
        let content = "%%MatrixMarket matrix array real general\n\
                        2 3\n\
                        1.0\n2.0\n3.0\n4.0\n5.0\n";
        let err = mmread(content).expect_err("underfilled array payload should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("array format expected 6 values but found 5".to_string())
        );
    }

    #[test]
    fn mmread_array_rejects_too_many_values() {
        let content = "%%MatrixMarket matrix array real general\n\
                        2 3\n\
                        1.0\n2.0\n3.0\n4.0\n5.0\n6.0\n7.0\n";
        let err = mmread(content).expect_err("overfilled array payload should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("array format has more than the declared 6 values".to_string())
        );
    }

    #[test]
    fn mmread_rejects_dense_size_overflow() {
        let content = format!(
            "%%MatrixMarket matrix coordinate real general\n{} 2 0\n",
            usize::MAX
        );
        let err = mmread(&content).expect_err("overflowing dense dimensions should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(format!(
                "Matrix Market matrix dimensions {}x2 overflowed usize",
                usize::MAX
            ))
        );
    }

    #[test]
    fn mmread_rejects_coordinate_dense_allocation_dos() {
        let content = "%%MatrixMarket matrix coordinate real general\n1000000 1000000 1\n1 1 1.0\n";
        let err = mmread(content).expect_err("hostile dense allocation should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(format!(
                "Matrix Market matrix dimensions 1000000x1000000 exceed dense read safety bound of {MAX_MM_DENSE_ELEMENTS} elements"
            ))
        );
    }

    #[test]
    fn mmwrite_roundtrip() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let content = mmwrite(2, 3, &data).unwrap();
        let mat = mmread(&content).unwrap();
        assert_eq!(mat.rows, 2);
        assert_eq!(mat.cols, 3);
        for (i, (&orig, &read)) in data.iter().zip(mat.data.iter()).enumerate() {
            assert!(
                (orig - read).abs() < 1e-10,
                "mismatch at {i}: orig={orig}, read={read}"
            );
        }
    }

    #[test]
    fn metamorphic_text_roundtrips_preserve_payload_and_metadata() {
        let dense_cases = [
            (1, 1, vec![42.0]),
            (2, 3, vec![1.0, -2.5, 3.25, 4.0, 0.0, 9.5]),
            (3, 2, vec![0.125, 1.25, 2.5, 5.0, 10.0, 20.0]),
        ];
        for (rows, cols, data) in dense_cases {
            let encoded = mmwrite(rows, cols, &data).expect("Matrix Market encode succeeds");
            let info = mminfo(&encoded).expect("mminfo should parse writer output");
            let decoded = mmread(&encoded).expect("mmread should parse writer output");

            assert_eq!(info.rows, rows);
            assert_eq!(info.cols, cols);
            assert_eq!(info.nnz, rows * cols);
            assert_eq!(info.rows, decoded.info.rows);
            assert_eq!(info.cols, decoded.info.cols);
            assert_eq!(info.nnz, decoded.info.nnz);
            assert_eq!(decoded.data, data);
        }

        let csv_rows = vec![vec![0.0, 1.5], vec![1.0, 2.25], vec![2.0, 3.0]];
        let csv = write_csv(Some(&["time", "value"]), &csv_rows, ',').expect("CSV encode");
        let (header, decoded_csv) = read_csv(&csv, ',', true).expect("CSV decode");
        assert_eq!(header, Some(vec!["time".to_string(), "value".to_string()]));
        assert_eq!(decoded_csv, csv_rows);

        let json_values = vec![1.5, -2.25, 0.0, 3.75];
        let json = write_json_array(&json_values).expect("JSON array encode");
        let decoded_json = read_json_array(&json).expect("JSON array decode");
        assert_eq!(decoded_json, json_values);
    }

    #[test]
    fn mmwrite_rejects_dense_size_overflow() {
        let err = mmwrite(usize::MAX, 2, &[])
            .expect_err("overflowing dense dimensions should fail before length comparison");
        assert_eq!(
            err,
            IoError::InvalidFormat(format!(
                "Matrix Market matrix dimensions {}x2 overflowed usize",
                usize::MAX
            ))
        );
    }

    #[test]
    fn wav_roundtrip() {
        let samples = vec![0.0, 0.5, 1.0, -1.0, -0.5, 0.0];
        let bytes = wav_write(44100, 1, &samples).expect("mono samples should encode");
        let wav = wav_read(&bytes).unwrap();
        assert_eq!(wav.sample_rate, 44100);
        assert_eq!(wav.channels, 1);
        assert_eq!(wav.data.len(), samples.len());
        // 16-bit quantization: ~1/32768 precision
        for (i, (&orig, &read)) in samples.iter().zip(wav.data.iter()).enumerate() {
            assert!(
                (orig - read).abs() < 0.001,
                "sample {i}: orig={orig}, read={read}"
            );
        }
    }

    #[test]
    fn wav_write_rejects_partial_frames() {
        let err = wav_write(44_100, 2, &[0.0, 0.5, 1.0])
            .expect_err("stereo data with odd sample count should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "data length 3 does not contain whole frames for 2 channels".to_string()
            )
        );
    }

    #[test]
    fn wav_write_rejects_zero_sample_rate() {
        let err =
            wav_write(0, 1, &[0.0]).expect_err("zero sample rate should fail before encoding");
        assert_eq!(
            err,
            IoError::InvalidFormat("WAV sample rate must be nonzero".to_string())
        );
    }

    #[test]
    fn wav_write_rejects_block_align_overflow() {
        let err = wav_write(1, 32_768, &[])
            .expect_err("oversized channel count should fail before header overflow");
        assert_eq!(
            err,
            IoError::InvalidFormat("WAV block align overflowed u16".to_string())
        );
    }

    #[test]
    fn wav_read_rejects_partial_sample_bytes() {
        let mut bytes = wav_write(44_100, 1, &[0.0, 0.5]).expect("mono samples should encode");
        bytes[40..44].copy_from_slice(&3u32.to_le_bytes());
        bytes.truncate(44 + 3);

        let err = wav_read(&bytes).expect_err("misaligned sample bytes should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "data chunk size 3 is not aligned to 2-byte samples".to_string()
            )
        );
    }

    #[test]
    fn wav_read_rejects_partial_frames() {
        let mut bytes =
            wav_write(44_100, 2, &[0.0, 0.5, 1.0, -1.0]).expect("stereo samples should encode");
        bytes[40..44].copy_from_slice(&6u32.to_le_bytes());
        bytes.truncate(44 + 6);

        let err = wav_read(&bytes).expect_err("partial stereo frame should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "data chunk size 6 does not contain whole 2-channel frames".to_string()
            )
        );
    }

    #[test]
    fn wav_read_24bit_pcm_sign_extends_negative_samples() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36u32 + 3u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&44100u32.to_le_bytes());
        bytes.extend_from_slice(&(44100u32 * 3).to_le_bytes());
        bytes.extend_from_slice(&3u16.to_le_bytes());
        bytes.extend_from_slice(&24u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&[0x00, 0x00, 0x80]); // -1.0 in signed 24-bit PCM
        bytes.push(0); // pad odd-sized chunk

        let wav = wav_read(&bytes).expect("24-bit wav should decode");
        assert_eq!(wav.bits_per_sample, 24);
        assert!(
            (wav.data[0] + 1.0).abs() < 1e-6,
            "expected -1.0 sample, got {}",
            wav.data[0]
        );
    }

    #[test]
    fn wav_read_rejects_zero_channel_fmt() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36u32 + 2u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&44_100u32.to_le_bytes());
        bytes.extend_from_slice(&(44_100u32 * 2).to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0i16.to_le_bytes());

        let err = wav_read(&bytes).expect_err("zero-channel WAV should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("fmt chunk declares zero channels".to_string())
        );
    }

    #[test]
    fn wav_read_rejects_data_before_fmt_chunk() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36u32 + 2u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0i16.to_le_bytes());
        bytes.resize(44, 0);

        let err = wav_read(&bytes).expect_err("data before fmt should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("encountered data chunk before a valid fmt chunk".to_string())
        );
    }

    #[test]
    fn wav_read_rejects_float_with_non_32bit_samples() {
        let mut bytes = wav_write(44_100, 1, &[0.0]).expect("mono samples should encode");
        bytes[20..22].copy_from_slice(&3u16.to_le_bytes());

        let err = wav_read(&bytes).expect_err("float WAV with 16-bit samples should fail");
        assert_eq!(
            err,
            IoError::UnsupportedFeature("unsupported IEEE float bits per sample: 16".to_string())
        );
    }

    #[test]
    fn savemat_loadmat_roundtrip() {
        let arrays = vec![
            MatArray {
                name: "A".to_string(),
                rows: 2,
                cols: 2,
                data: vec![1.0, 2.0, 3.0, 4.0],
            },
            MatArray {
                name: "b".to_string(),
                rows: 3,
                cols: 1,
                data: vec![10.0, 20.0, 30.0],
            },
        ];
        let text = savemat_text(&arrays).expect("well-formed arrays should serialize");
        let loaded = loadmat_text(&text).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].name, "A");
        assert_eq!(loaded[0].data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(loaded[1].name, "b");
        assert_eq!(loaded[1].data, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn savemat_loadmat_binary_matches_scipy_semantics() {
        // Test that our MAT format roundtrip preserves values like scipy.io.savemat/loadmat
        // scipy.io.loadmat returns arrays: {'x': [[1.0, 2.0, 3.0]], 'y': [[1, 2], [3, 4]]}
        let arrays = vec![
            MatArray {
                name: "x".to_string(),
                rows: 1,
                cols: 3,
                data: vec![1.0, 2.0, 3.0],
            },
            MatArray {
                name: "y".to_string(),
                rows: 2,
                cols: 2,
                data: vec![1.0, 2.0, 3.0, 4.0],
            },
        ];
        let bytes = savemat(&arrays).expect("savemat should succeed");
        let loaded = loadmat(&bytes).expect("loadmat should succeed");
        assert_eq!(loaded.len(), 2);
        let x = loaded
            .iter()
            .find(|a| a.name == "x")
            .expect("x should exist");
        let y = loaded
            .iter()
            .find(|a| a.name == "y")
            .expect("y should exist");
        assert_eq!(x.data, vec![1.0, 2.0, 3.0]);
        assert_eq!(y.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(x.rows, 1);
        assert_eq!(x.cols, 3);
        assert_eq!(y.rows, 2);
        assert_eq!(y.cols, 2);
    }

    #[test]
    fn savemat_binary_roundtrip_mat4_real_double() {
        let arrays = vec![
            MatArray {
                name: "A".to_string(),
                rows: 2,
                cols: 2,
                data: vec![1.0, 2.0, 3.0, 4.0],
            },
            MatArray {
                name: "b".to_string(),
                rows: 1,
                cols: 3,
                data: vec![5.0, 6.0, 7.0],
            },
        ];

        let bytes = savemat(&arrays).expect("MAT v4 real doubles should serialize");
        let loaded = loadmat(&bytes).expect("MAT v4 real doubles should parse");

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].name, "A");
        assert_eq!(loaded[0].rows, 2);
        assert_eq!(loaded[0].cols, 2);
        assert_eq!(loaded[0].data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(loaded[1].name, "b");
        assert_eq!(loaded[1].rows, 1);
        assert_eq!(loaded[1].cols, 3);
        assert_eq!(loaded[1].data, vec![5.0, 6.0, 7.0]);
    }

    #[test]
    fn savemat_binary_uses_mat4_column_major_double_layout() {
        let bytes = savemat(&[MatArray {
            name: "A".to_string(),
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        }])
        .expect("MAT v4 real doubles should serialize");

        let expected = [
            0, 0, 0, 0, // mopt: little-endian double full matrix
            2, 0, 0, 0, // rows
            2, 0, 0, 0, // cols
            0, 0, 0, 0, // imagf
            2, 0, 0, 0, // namlen, including trailing NUL
            b'A', 0, // name
            0, 0, 0, 0, 0, 0, 240, 63, // 1.0
            0, 0, 0, 0, 0, 0, 8, 64, // 3.0
            0, 0, 0, 0, 0, 0, 0, 64, // 2.0
            0, 0, 0, 0, 0, 0, 16, 64, // 4.0
        ];
        assert_eq!(bytes, expected);
    }

    #[test]
    fn loadmat_binary_rejects_complex_mat4_payload() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&1i32.to_le_bytes());
        bytes.extend_from_slice(&1i32.to_le_bytes());
        bytes.extend_from_slice(&1i32.to_le_bytes());
        bytes.extend_from_slice(&2i32.to_le_bytes());
        bytes.extend_from_slice(b"z\0");
        bytes.extend_from_slice(&1.0f64.to_le_bytes());
        bytes.extend_from_slice(&2.0f64.to_le_bytes());

        let err = loadmat(&bytes).expect_err("complex MAT v4 payload should fail closed");
        assert_eq!(
            err,
            IoError::UnsupportedFeature("MAT v4 complex matrices are not supported".to_string())
        );
    }

    #[test]
    fn loadmat_binary_rejects_truncated_header() {
        let err = loadmat(&[0, 0, 0]).expect_err("short MAT v4 header should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("truncated MAT v4 header while reading mopt".to_string())
        );
    }

    #[test]
    fn savemat_rejects_wrong_element_count() {
        let arrays = vec![MatArray {
            name: "A".to_string(),
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0, 3.0],
        }];
        let err = savemat_text(&arrays).expect_err("truncated matrix payload should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("array 'A' expected 4 values but found 3".to_string())
        );
    }

    #[test]
    fn savemat_rejects_multiline_names() {
        let arrays = vec![MatArray {
            name: "bad\nname".to_string(),
            rows: 1,
            cols: 1,
            data: vec![1.0],
        }];
        let err = savemat_text(&arrays).expect_err("multiline names should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "array name 'bad\\nname' contains a newline and cannot be encoded safely"
                    .to_string()
            )
        );
    }

    #[test]
    fn loadmat_rejects_wrong_element_count() {
        let text = "# name: A\n# type: matrix\n# rows: 2\n# columns: 2\n1 2\n";
        let err = loadmat_text(text).expect_err("truncated matrix payload should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("array 'A' expected 2 rows but found 1".to_string())
        );
    }

    #[test]
    fn loadmat_rejects_ragged_rows() {
        let text = "# name: A\n# type: matrix\n# rows: 2\n# columns: 2\n1 2 3\n4\n";
        let err = loadmat_text(text).expect_err("ragged rows should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("array 'A' row 0 has 3 columns, expected 2".to_string())
        );
    }

    #[test]
    fn loadmat_rejects_data_without_name_header() {
        let err = loadmat_text("1 2\n3 4\n")
            .expect_err("data block without a name header should fail closed");
        assert_eq!(
            err,
            IoError::InvalidFormat("encountered matrix data before '# name:' header".to_string())
        );
    }

    #[test]
    fn loadmat_rejects_incomplete_trailing_header_block() {
        let err = loadmat_text("# name: A\n# rows: 2\n# columns: 2\n")
            .expect_err("trailing header-only block should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("incomplete MAT text block at end of file".to_string())
        );
    }

    #[test]
    fn loadmat_rejects_data_before_dimension_headers_are_complete() {
        let err = loadmat_text("# name: A\n# rows: 2\n1 2\n3 4\n")
            .expect_err("data before full dimension metadata should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "array 'A' is missing nonzero '# rows:' and '# columns:' headers before data"
                    .to_string()
            )
        );
    }

    #[test]
    fn loadmat_rejects_dimension_overflow() {
        let text = format!(
            "# name: A\n# type: matrix\n# rows: {}\n# columns: 2\n1 2\n",
            usize::MAX
        );
        let err = loadmat_text(&text).expect_err("overflowing MAT dimensions should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(format!(
                "MAT text matrix dimensions {}x2 overflowed usize",
                usize::MAX
            ))
        );
    }

    #[test]
    fn read_idl_save_scalar_int32_matches_readsav_case_lookup() {
        let mut bytes = idl_save_header();
        idl_push_variable_record(&mut bytes, "I32S", 3, 0, |out| {
            idl_push_i32(out, -1_234_567_890);
        });
        idl_push_end_record(&mut bytes);

        let parsed = readsav(&bytes).expect("IDL SAVE scalar int32 should parse");

        assert_eq!(
            parsed.get("i32s"),
            Some(&IdlValue::Scalar(IdlScalar::Int32(-1_234_567_890)))
        );
        assert_eq!(
            parsed.get("I32S"),
            Some(&IdlValue::Scalar(IdlScalar::Int32(-1_234_567_890)))
        );
    }

    #[test]
    fn read_idl_save_scalar_string_preserves_bytes() {
        let mut bytes = idl_save_header();
        idl_push_variable_record(&mut bytes, "S", 7, 0, |out| {
            idl_push_string_data(out, b"The quick brown fox");
        });
        idl_push_end_record(&mut bytes);

        let parsed = read_idl_save(&bytes).expect("IDL SAVE string should parse");

        assert_eq!(
            parsed.get("s"),
            Some(&IdlValue::Scalar(IdlScalar::String(
                b"The quick brown fox".to_vec()
            )))
        );
    }

    #[test]
    fn read_idl_save_float32_array_reverses_dimensions_like_scipy() {
        let mut bytes = idl_save_header();
        idl_push_array_variable_record(&mut bytes, "ARRAY2D", 4, 24, 6, &[3, 2], |out| {
            for value in [1.0_f32, 2.0, 3.5, 4.5, 5.25, 6.75] {
                out.extend_from_slice(&value.to_be_bytes());
            }
        });
        idl_push_end_record(&mut bytes);

        let parsed = read_idl_save(&bytes).expect("IDL SAVE float32 array should parse");
        let value = parsed.get("array2d").expect("array variable present");

        let IdlValue::Array(array) = value else {
            assert!(
                matches!(value, IdlValue::Array(_)),
                "expected IDL array, got {value:?}"
            );
            return;
        };
        assert_eq!(array.element_type, IdlType::Float32);
        assert_eq!(array.dims, vec![2, 3]);
        assert_eq!(
            array.values,
            vec![
                IdlScalar::Float32(1.0),
                IdlScalar::Float32(2.0),
                IdlScalar::Float32(3.5),
                IdlScalar::Float32(4.5),
                IdlScalar::Float32(5.25),
                IdlScalar::Float32(6.75),
            ]
        );
    }

    #[test]
    fn read_idl_save_rejects_bad_signature_and_compressed_streams() {
        let bad_signature =
            read_idl_save(b"NO\x00\x04").expect_err("bad signature should fail closed");
        assert!(matches!(bad_signature, IoError::InvalidFormat(_)));

        let compressed =
            read_idl_save(b"SR\x00\x06").expect_err("compressed IDL SAVE should be unsupported");
        assert!(matches!(compressed, IoError::UnsupportedFeature(_)));
    }

    #[test]
    fn loadtxt_basic() {
        let content = "# comment\n1 2 3\n4 5 6\n";
        let (rows, cols, data) = loadtxt(content).unwrap();
        assert_eq!(rows, 2);
        assert_eq!(cols, 3);
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn savetxt_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let text = savetxt(2, 2, &data, " ").expect("matching shape should succeed");
        assert_eq!(text, "1 2\n3 4\n");
    }

    #[test]
    fn savetxt_rejects_shape_length_mismatch() {
        let err = savetxt(2, 2, &[1.0, 2.0, 3.0], " ").expect_err("mismatched shape should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("data length 3 doesn't match 2x2".to_string())
        );
    }

    #[test]
    fn savetxt_rejects_dimension_overflow() {
        let err = savetxt(usize::MAX, 2, &[], " ")
            .expect_err("overflowing savetxt dimensions should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(format!(
                "text matrix dimensions {}x2 overflowed usize",
                usize::MAX
            ))
        );
    }

    #[test]
    fn savetxt_rejects_multiline_delimiter() {
        let err = savetxt(1, 2, &[1.0, 2.0], "\n").expect_err("multiline delimiters should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "delimiter \"\\n\" contains a newline and cannot be encoded safely".to_string()
            )
        );
    }

    #[test]
    fn read_npy_text_rejects_shape_payload_mismatch() {
        let err = read_npy_text("2,2\n1 2 3\n").expect_err("truncated payload should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("shape [2, 2] expects 4 values but found 3".to_string())
        );
    }

    #[test]
    fn read_npy_text_rejects_empty_shape() {
        let err = read_npy_text("\n1\n").expect_err("empty shape line should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "shape declaration must contain at least one dimension".to_string()
            )
        );
    }

    #[test]
    fn read_npy_text_rejects_empty_shape_dimension() {
        let err = read_npy_text("2,,2\n1 2 3 4\n")
            .expect_err("shape declarations with empty dimensions should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("shape declaration contains an empty dimension".to_string())
        );
    }

    #[test]
    fn read_csv_rejects_ragged_rows() {
        let err = read_csv("1,2,3\n4,5\n", ',', false).expect_err("ragged CSV should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("CSV row has 2 columns, expected 3".to_string())
        );
    }

    #[test]
    fn read_csv_rejects_empty_input_when_header_is_required() {
        let err =
            read_csv("", ',', true).expect_err("empty input with required header should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("CSV header row is required but the input is empty".to_string())
        );
    }

    #[test]
    fn read_csv_rejects_header_data_column_mismatch() {
        let err =
            read_csv("a,b,c\n1,2\n3,4\n", ',', true).expect_err("header/data mismatch should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("CSV header has 3 columns but first data row has 2".to_string())
        );
    }

    #[test]
    fn write_csv_basic() {
        let text = write_csv(None, &[vec![1.0, 2.0], vec![3.0, 4.0]], ',')
            .expect("rectangular CSV should serialize");
        assert_eq!(text, "1,2\n3,4\n");
    }

    #[test]
    fn write_csv_rejects_header_data_column_mismatch() {
        let err = write_csv(Some(&["a", "b", "c"]), &[vec![1.0, 2.0]], ',')
            .expect_err("header/data mismatch should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("CSV header has 3 columns but first data row has 2".to_string())
        );
    }

    #[test]
    fn write_csv_rejects_header_cells_with_delimiters() {
        let err = write_csv(Some(&["bad,header"]), &[vec![1.0]], ',')
            .expect_err("header cells containing the delimiter should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "CSV header cell \"bad,header\" contains the delimiter ',' and cannot be encoded safely"
                    .to_string()
            )
        );
    }

    #[test]
    fn write_csv_rejects_header_cells_with_newlines() {
        let err = write_csv(Some(&["bad\nheader"]), &[vec![1.0]], ',')
            .expect_err("header cells containing newlines should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat(
                "CSV header cell \"bad\\nheader\" contains a newline and cannot be encoded safely"
                    .to_string()
            )
        );
    }

    #[test]
    fn write_csv_rejects_ragged_rows() {
        let err = write_csv(None, &[vec![1.0, 2.0, 3.0], vec![4.0, 5.0]], ',')
            .expect_err("ragged CSV should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("CSV row has 2 columns, expected 3".to_string())
        );
    }

    #[test]
    fn read_json_array_accepts_empty_array() {
        let data = read_json_array("[]").expect("empty array should parse");
        assert!(data.is_empty());
    }

    #[test]
    fn read_json_array_rejects_non_finite_values() {
        let err = read_json_array("[1.0, NaN]").expect_err("NaN is not valid JSON");
        assert_eq!(
            err,
            IoError::InvalidFormat("JSON parse error: non-finite value NaN".to_string())
        );
    }

    #[test]
    fn write_json_array_rejects_non_finite_values() {
        let err = write_json_array(&[1.0, f64::NAN]).expect_err("NaN is not valid JSON");
        assert_eq!(
            err,
            IoError::InvalidFormat("JSON array value at index 1 is not finite: NaN".to_string())
        );
    }

    #[test]
    fn mmwrite_sparse_format() {
        let entries = vec![(0, 0, 1.0), (1, 1, 2.0)];
        let content = mmwrite_sparse(3, 3, &entries).unwrap();
        assert!(content.contains("coordinate"));
        let mat = mmread(&content).unwrap();
        assert_eq!(mat.data[0], 1.0);
        assert_eq!(mat.data[4], 2.0);
    }

    #[test]
    fn mmwrite_sparse_rejects_out_of_bounds_entries() {
        let err = mmwrite_sparse(2, 2, &[(2, 0, 1.0)])
            .expect_err("out-of-bounds sparse entry should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("sparse entry (2, 0) out of bounds for 2x2".to_string())
        );
    }

    #[test]
    fn read_csv_single_column_no_header() {
        let (header, data) = read_csv("1\n2\n3\n", ',', false).expect("single column CSV");
        assert_eq!(data.len(), 3);
        assert!(header.is_none());
        assert_eq!(data[0], vec![1.0]);
        assert_eq!(data[2], vec![3.0]);
    }

    #[test]
    fn read_csv_with_tab_delimiter() {
        let (header, data) = read_csv("a\tb\n1\t2\n3\t4\n", '\t', true).expect("TSV");
        assert_eq!(header, Some(vec!["a".to_string(), "b".to_string()]));
        assert_eq!(data[0], vec![1.0, 2.0]);
    }

    #[test]
    fn read_csv_empty_no_header() {
        let (_header, data) = read_csv("", ',', false).expect("empty without header is ok");
        assert!(data.is_empty());
    }

    #[test]
    fn netcdf_classic_roundtrip_double_matrix_with_attributes() {
        let file = NetcdfFile {
            dimensions: vec![
                NetcdfDimension {
                    name: "time".to_string(),
                    len: Some(2),
                },
                NetcdfDimension {
                    name: "station".to_string(),
                    len: Some(3),
                },
            ],
            attributes: vec![NetcdfAttribute {
                name: "title".to_string(),
                value: NetcdfValue::Char("demo".to_string()),
            }],
            variables: vec![NetcdfVariable {
                name: "temperature".to_string(),
                dim_ids: vec![0, 1],
                attributes: vec![NetcdfAttribute {
                    name: "units".to_string(),
                    value: NetcdfValue::Char("K".to_string()),
                }],
                data: NetcdfValue::Double(vec![280.0, 281.5, 282.25, 283.0, 284.5, 285.25]),
            }],
        };

        let bytes = write_netcdf_classic(&file).expect("NetCDF classic encode");
        assert_eq!(&bytes[..4], b"CDF\x01");
        let parsed = read_netcdf_classic(&bytes).expect("NetCDF classic decode");
        assert_eq!(parsed, file);
    }

    #[test]
    fn netcdf_file_aliases_roundtrip_int_scalar() {
        let file = NetcdfFile {
            dimensions: Vec::new(),
            attributes: Vec::new(),
            variables: vec![NetcdfVariable {
                name: "answer".to_string(),
                dim_ids: Vec::new(),
                attributes: Vec::new(),
                data: NetcdfValue::Int(vec![42]),
            }],
        };

        let bytes = netcdf_file_write(&file).expect("NetCDF alias encode");
        let parsed = netcdf_file_read(&bytes).expect("NetCDF alias decode");
        assert_eq!(parsed.variables.len(), 1);
        assert_eq!(parsed.variables[0].name, "answer");
        assert_eq!(parsed.variables[0].data, NetcdfValue::Int(vec![42]));
    }

    #[test]
    fn netcdf_classic_metamorphic_variable_shape_matches_dimension_product() {
        let file = NetcdfFile {
            dimensions: vec![
                NetcdfDimension {
                    name: "x".to_string(),
                    len: Some(4),
                },
                NetcdfDimension {
                    name: "y".to_string(),
                    len: Some(2),
                },
            ],
            attributes: Vec::new(),
            variables: vec![NetcdfVariable {
                name: "mask".to_string(),
                dim_ids: vec![0, 1],
                attributes: Vec::new(),
                data: NetcdfValue::Byte(vec![1, 0, 1, 0, 1, 0, 1, 0]),
            }],
        };
        let bytes = write_netcdf_classic(&file).expect("NetCDF encode");
        let parsed = read_netcdf_classic(&bytes).expect("NetCDF decode");
        let variable = &parsed.variables[0];
        let expected_len = variable
            .dim_ids
            .iter()
            .map(|&dim_id| parsed.dimensions[dim_id].len.unwrap_or(0))
            .product::<usize>();
        assert_eq!(variable.data.len(), expected_len);
    }

    #[test]
    fn netcdf_classic_rejects_bad_magic() {
        let err = read_netcdf_classic(b"BAD\x01\0\0\0\0").expect_err("bad magic should fail");
        assert_eq!(
            err,
            IoError::InvalidFormat("NetCDF missing CDF magic".to_string())
        );
    }

    #[test]
    fn netcdf_classic_writer_rejects_unlimited_dimension() {
        let file = NetcdfFile {
            dimensions: vec![NetcdfDimension {
                name: "time".to_string(),
                len: None,
            }],
            attributes: Vec::new(),
            variables: Vec::new(),
        };
        let err = write_netcdf_classic(&file).expect_err("unlimited dims are follow-on work");
        assert!(matches!(err, IoError::UnsupportedFeature(_)));
    }

    /// Tiny canonical HB file: 3x3 RUA, 4 nonzeros at (0,0)=1.0, (1,1)=2.0,
    /// (2,1)=3.0, (2,2)=4.0. Title is exactly 72 chars long (padded), key 8.
    fn sample_hb_rua_3x3_4nnz() -> String {
        let title = format!("{:<72}", "Test 3x3 RUA");
        format!(
            "{title}KEY00001\n\
             4 1 1 1 0\n\
             RUA            3            3            4            0\n\
             (4I20)\n\
             (4I20)\n\
             (4D20.13)\n\
             1 2 3 5\n\
             1 2 3 3\n\
             1.0D+00 2.0D+00 3.0D+00 4.0D+00\n"
        )
    }

    #[test]
    fn read_harwell_boeing_rua_basic_dimensions() {
        let content = sample_hb_rua_3x3_4nnz();
        let mat = read_harwell_boeing(&content).expect("RUA parse");
        assert_eq!(mat.rows, 3);
        assert_eq!(mat.cols, 3);
        assert_eq!(mat.nnz, 4);
        assert_eq!(mat.matrix_type, HbType::RealUnsymmetricAssembled);
        assert_eq!(mat.col_ptr, vec![0, 1, 2, 4]);
        assert_eq!(mat.row_idx, vec![0, 1, 2, 2]);
        assert_eq!(mat.values, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(mat.title, "Test 3x3 RUA");
        assert_eq!(mat.key, "KEY00001");
    }

    #[test]
    fn read_harwell_boeing_metamorphic_value_sum_invariant() {
        let mat = read_harwell_boeing(&sample_hb_rua_3x3_4nnz()).unwrap();
        let direct: f64 = mat.values.iter().sum();
        let by_column: f64 = (0..mat.cols)
            .map(|j| {
                mat.values[mat.col_ptr[j]..mat.col_ptr[j + 1]]
                    .iter()
                    .sum::<f64>()
            })
            .sum();
        assert!((direct - by_column).abs() < 1e-12);
    }

    #[test]
    fn read_harwell_boeing_metamorphic_col_ptr_monotone() {
        let mat = read_harwell_boeing(&sample_hb_rua_3x3_4nnz()).unwrap();
        for w in mat.col_ptr.windows(2) {
            assert!(w[0] <= w[1], "col_ptr must be monotone non-decreasing");
        }
        assert_eq!(*mat.col_ptr.last().unwrap(), mat.nnz);
    }

    #[test]
    fn read_harwell_boeing_rejects_unsupported_complex() {
        let title = format!("{:<72}", "Bad type");
        let content = format!(
            "{title}KEY00002\n\
             4 1 1 1 0\n\
             CUA            3            3            4            0\n\
             (4I20)\n(4I20)\n(4D20.13)\n\
             1 2 3 5\n1 2 3 3\n1.0D+00 2.0D+00 3.0D+00 4.0D+00\n"
        );
        let err = read_harwell_boeing(&content)
            .expect_err("complex unsupported variant must be rejected");
        assert!(matches!(err, IoError::UnsupportedFeature(ref m) if m.contains("CUA")));
    }

    #[test]
    fn read_harwell_boeing_rejects_truncated_payload() {
        let title = format!("{:<72}", "Truncated");
        let content = format!(
            "{title}KEY00003\n\
             4 1 1 1 0\n\
             RUA            3            3            4            0\n\
             (4I20)\n(4I20)\n(4D20.13)\n\
             1 2 3 5\n"
        );
        let err = read_harwell_boeing(&content).expect_err("truncated payload must be rejected");
        assert!(matches!(err, IoError::InvalidFormat(_)));
    }

    #[test]
    fn read_harwell_boeing_handles_lowercase_d_exponent() {
        let title = format!("{:<72}", "Lowercase D");
        let content = format!(
            "{title}KEY00004\n\
             4 1 1 1 0\n\
             RUA            2            2            2            0\n\
             (4I20)\n(4I20)\n(4D20.13)\n\
             1 2 3\n1 2\n1.5d+00 -2.5d-01\n"
        );
        let mat = read_harwell_boeing(&content).expect("lowercase d parse");
        assert_eq!(mat.values, vec![1.5, -0.25]);
    }

    #[test]
    fn read_arff_minimal_dense_numeric_and_nominal() {
        let arff = "@relation iris\n\
                    @attribute sepal_length numeric\n\
                    @attribute class {Iris-setosa, Iris-versicolor, Iris-virginica}\n\
                    @data\n\
                    5.1, Iris-setosa\n\
                    7.0, Iris-versicolor\n";
        let parsed = read_arff(arff).expect("ARFF parse");
        assert_eq!(parsed.relation, "iris");
        assert_eq!(parsed.attributes.len(), 2);
        assert_eq!(parsed.rows.len(), 2);
        assert!(matches!(&parsed.rows[0][0], ArffValue::Numeric(v) if (v - 5.1).abs() < 1e-12));
        assert_eq!(
            parsed.rows[0][1],
            ArffValue::Nominal("Iris-setosa".to_string())
        );
    }

    #[test]
    fn read_arff_metamorphic_attribute_count_invariant() {
        let arff = "@relation r\n\
                    @attribute a numeric\n\
                    @attribute b numeric\n\
                    @attribute c {x, y}\n\
                    @data\n\
                    1, 2, x\n\
                    3, 4, y\n";
        let parsed = read_arff(arff).unwrap();
        for (idx, row) in parsed.rows.iter().enumerate() {
            assert_eq!(row.len(), parsed.attributes.len(), "row {idx}");
        }
    }

    #[test]
    fn read_arff_metamorphic_nominal_domain_consistency() {
        // Every nominal cell in a row must match its column's declared domain.
        let arff = "@relation r\n\
                    @attribute c {a, b, c}\n\
                    @data\n\
                    a\n\
                    b\n\
                    c\n";
        let parsed = read_arff(arff).unwrap();
        assert!(matches!(
            &parsed.attributes[0],
            ArffAttribute::Nominal { .. }
        ));
        for row in &parsed.rows {
            assert!(matches!(
                (&parsed.attributes[0], &row[0]),
                (ArffAttribute::Nominal { domain, .. }, ArffValue::Nominal(v)) if domain.contains(v)
            ));
        }
    }

    #[test]
    fn read_arff_sparse_row_expansion() {
        let arff = "@relation r\n\
                    @attribute a numeric\n\
                    @attribute b numeric\n\
                    @attribute c {x, y}\n\
                    @data\n\
                    {0 5.5, 2 y}\n";
        let parsed = read_arff(arff).unwrap();
        assert_eq!(parsed.rows.len(), 1);
        let row = &parsed.rows[0];
        // Per ARFF sparse semantics: missing numeric cells default to 0.0.
        assert!(matches!(&row[0], ArffValue::Numeric(v) if (v - 5.5).abs() < 1e-12));
        assert_eq!(row[1], ArffValue::Numeric(0.0));
        assert_eq!(row[2], ArffValue::Nominal("y".to_string()));
    }

    #[test]
    fn read_arff_handles_comments_and_quotes() {
        let arff = "% top comment\n\
                    @relation 'fancy name'\n\
                    @attribute a numeric % inline numeric\n\
                    @attribute b string\n\
                    @data\n\
                    1.0, 'hello world'\n\
                    2.0, \"with comma, here\"\n\
                    ?, ?\n";
        let parsed = read_arff(arff).expect("ARFF parse with comments");
        assert_eq!(parsed.relation, "fancy name");
        assert_eq!(parsed.rows.len(), 3);
        assert_eq!(
            parsed.rows[1][1],
            ArffValue::String("with comma, here".to_string())
        );
        assert_eq!(parsed.rows[2][0], ArffValue::Missing);
        assert_eq!(parsed.rows[2][1], ArffValue::Missing);
    }

    #[test]
    fn read_arff_dense_date_attributes_match_scipy_units() {
        let arff = "@relation r\n\
                    @attribute day date \"yyyy-MM-dd\"\n\
                    @attribute instant date \"yyyy-MM-dd HH:mm:ss\"\n\
                    @data\n\
                    \"2026-05-03\", \"2026-05-03 14:05:09\"\n\
                    ?, ?\n";
        let parsed = read_arff(arff).expect("date attributes parse");
        assert_eq!(
            parsed.attributes[0],
            ArffAttribute::Date {
                name: "day".to_string(),
                format: "yyyy-MM-dd".to_string(),
                unit: ArffDateUnit::Day,
            }
        );
        assert_eq!(
            parsed.rows[0][0],
            ArffValue::Date(ArffDateTime {
                raw: "2026-05-03".to_string(),
                normalized: "2026-05-03".to_string(),
                unit: ArffDateUnit::Day,
            })
        );
        assert_eq!(
            parsed.rows[0][1],
            ArffValue::Date(ArffDateTime {
                raw: "2026-05-03 14:05:09".to_string(),
                normalized: "2026-05-03T14:05:09".to_string(),
                unit: ArffDateUnit::Second,
            })
        );
        assert_eq!(parsed.rows[1][0], ArffValue::Missing);
        assert_eq!(parsed.rows[1][1], ArffValue::Missing);
    }

    #[test]
    fn read_arff_sparse_date_rows_keep_missing_dates() {
        let arff = "@relation r\n\
                    @attribute score numeric\n\
                    @attribute when date \"yyyy-MM-dd\"\n\
                    @data\n\
                    {1 \"2026-05-03\"}\n\
                    {0 2.5}\n";
        let parsed = read_arff(arff).expect("sparse date row parse");
        assert_eq!(parsed.rows[0][0], ArffValue::Numeric(0.0));
        assert_eq!(
            parsed.rows[0][1],
            ArffValue::Date(ArffDateTime {
                raw: "2026-05-03".to_string(),
                normalized: "2026-05-03".to_string(),
                unit: ArffDateUnit::Day,
            })
        );
        assert_eq!(parsed.rows[1][0], ArffValue::Numeric(2.5));
        assert_eq!(parsed.rows[1][1], ArffValue::Missing);
    }

    #[test]
    fn read_arff_rejects_date_attribute_with_timezone() {
        let arff = "@relation r\n\
                    @attribute when date \"yyyy-MM-dd Z\"\n\
                    @data\n\
                    \"2026-05-03 +0000\"\n";
        let err = read_arff(arff).expect_err("timezone date should be unsupported");
        assert!(matches!(err, IoError::UnsupportedFeature(_)));
    }

    #[test]
    fn read_arff_rejects_relational_attribute_as_unsupported() {
        let arff = "@relation r\n\
                    @attribute nested relational\n\
                    @attribute child numeric\n\
                    @end nested\n\
                    @data\n\
                    \"1\"\n";
        let err = read_arff(arff).expect_err("relational should be unsupported");
        assert!(matches!(err, IoError::UnsupportedFeature(_)));
    }

    #[test]
    fn read_arff_rejects_value_outside_nominal_domain() {
        let arff = "@relation r\n\
                    @attribute c {a, b}\n\
                    @data\n\
                    z\n";
        let err = read_arff(arff).expect_err("z not in {a,b}");
        assert!(matches!(err, IoError::InvalidFormat(_)));
    }

    #[test]
    fn fortran_roundtrip_two_little_endian_records() {
        let r1 = b"hello".to_vec();
        let r2 = b"world!".to_vec();
        let mut bytes = write_fortran_record(&r1, FortranEndian::Little);
        bytes.extend(write_fortran_record(&r2, FortranEndian::Little));
        let parsed = read_fortran_unformatted(&bytes, FortranEndian::Little).expect("two records");
        assert_eq!(parsed, vec![r1, r2]);
    }

    #[test]
    fn fortran_roundtrip_big_endian() {
        let payload = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0xFF];
        let bytes = write_fortran_record(&payload, FortranEndian::Big);
        let parsed = read_fortran_unformatted(&bytes, FortranEndian::Big).expect("BE record");
        assert_eq!(parsed, vec![payload.clone()]);
        // Reading the same bytes with the wrong endian must fail at the
        // length-mismatch check (or report a header far larger than the
        // input).
        let err = read_fortran_unformatted(&bytes, FortranEndian::Little)
            .expect_err("wrong endian must fail");
        assert!(matches!(err, IoError::InvalidFormat(_)));
    }

    #[test]
    fn fortran_metamorphic_record_count_invariant() {
        // For any sequence of payloads, the number of decoded records equals
        // the number we framed.
        let payloads: Vec<Vec<u8>> = (1..=5).map(|i| vec![i as u8; i * 3]).collect();
        let mut bytes = Vec::new();
        for p in &payloads {
            bytes.extend(write_fortran_record(p, FortranEndian::Little));
        }
        let parsed = read_fortran_unformatted(&bytes, FortranEndian::Little).unwrap();
        assert_eq!(parsed.len(), payloads.len());
        for (orig, got) in payloads.iter().zip(parsed.iter()) {
            assert_eq!(orig, got);
        }
    }

    #[test]
    fn fortran_rejects_truncated_payload() {
        let mut bytes = write_fortran_record(b"abcdef", FortranEndian::Little);
        bytes.truncate(bytes.len() - 4); // drop the trailer + 1
        let err = read_fortran_unformatted(&bytes, FortranEndian::Little)
            .expect_err("truncated must fail");
        assert!(matches!(err, IoError::InvalidFormat(_)));
    }

    #[test]
    fn fortran_rejects_length_mismatch() {
        // Hand-craft a record whose trailer disagrees with the header.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&5_i32.to_le_bytes());
        bytes.extend_from_slice(b"hello");
        bytes.extend_from_slice(&7_i32.to_le_bytes()); // wrong trailer
        let err = read_fortran_unformatted(&bytes, FortranEndian::Little)
            .expect_err("trailer mismatch must fail");
        assert!(matches!(err, IoError::InvalidFormat(ref msg) if msg.contains("trailer")));
    }

    #[test]
    fn fortran_empty_input_produces_no_records() {
        let parsed = read_fortran_unformatted(&[], FortranEndian::Little).unwrap();
        assert!(parsed.is_empty());
    }

    #[test]
    fn fortran_zero_length_record_is_supported() {
        let bytes = write_fortran_record(&[], FortranEndian::Little);
        let parsed = read_fortran_unformatted(&bytes, FortranEndian::Little).unwrap();
        assert_eq!(parsed.len(), 1);
        assert!(parsed[0].is_empty());
    }

    #[test]
    fn mmread_matches_scipy_reference_values() {
        // scipy.io.mmread on a Matrix Market coordinate format file
        // import scipy.io as spio; import io
        // mm = "%%MatrixMarket matrix coordinate real general\n3 3 3\n1 1 1.0\n2 2 2.0\n3 3 3.0\n"
        // spio.mmread(io.StringIO(mm)).toarray()
        // -> array([[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
        let mm_content =
            "%%MatrixMarket matrix coordinate real general\n3 3 3\n1 1 1.0\n2 2 2.0\n3 3 3.0\n";
        let result = mmread(mm_content).expect("mmread should succeed");
        assert_eq!(result.rows, 3);
        assert_eq!(result.cols, 3);
        // Expanded to dense row-major: data[i*cols+j]
        // (0,0)=1.0, (1,1)=2.0, (2,2)=3.0, all others 0.0
        assert!((result.data[0] - 1.0).abs() < 1e-10, "data[0,0] = 1.0");
        assert!((result.data[4] - 2.0).abs() < 1e-10, "data[1,1] = 2.0");
        assert!((result.data[8] - 3.0).abs() < 1e-10, "data[2,2] = 3.0");
        assert!((result.data[1] - 0.0).abs() < 1e-10, "off-diagonal is 0");
    }

    #[test]
    fn mmwrite_matches_scipy_output_format() {
        // scipy.io.mmwrite produces Matrix Market array format for dense matrices
        // import scipy.io as spio; import io
        // spio.mmwrite(io.StringIO(), [[1, 2], [3, 4]])
        // Result format: header + "1\n2\n3\n4\n" (column-major)
        let output = mmwrite(2, 2, &[1.0, 2.0, 3.0, 4.0]).expect("mmwrite should succeed");
        assert!(output.contains("%%MatrixMarket matrix array real general"));
        assert!(output.contains("2 2"));
        // Verify roundtrip
        let parsed = mmread(&output).expect("roundtrip should work");
        assert_eq!(parsed.rows, 2);
        assert_eq!(parsed.cols, 2);
        let expected = [1.0, 2.0, 3.0, 4.0];
        for (i, &want) in expected.iter().enumerate() {
            let got = parsed.data[i];
            assert!(
                (got - want).abs() < 1e-10,
                "data[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn loadtxt_matches_scipy_reference_values() {
        // np.loadtxt(StringIO("1 2\n3 4")) -> [[1, 2], [3, 4]]
        let content = "1 2\n3 4\n";
        let (rows, cols, data) = loadtxt(content).expect("loadtxt should succeed");
        assert_eq!(rows, 2);
        assert_eq!(cols, 2);
        let expected = [1.0, 2.0, 3.0, 4.0];
        for (i, (&got, &want)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "data[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn savetxt_matches_scipy_output_format() {
        // np.savetxt produces space-delimited text by default
        let output = savetxt(2, 2, &[1.0, 2.0, 3.0, 4.0], " ").expect("savetxt should succeed");
        // Verify roundtrip
        let (rows, cols, data) = loadtxt(&output).expect("roundtrip should work");
        assert_eq!(rows, 2);
        assert_eq!(cols, 2);
        let expected = [1.0, 2.0, 3.0, 4.0];
        for (i, (&got, &want)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "data[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn mminfo_matches_scipy_reference_values() {
        // scipy.io.mminfo returns (rows, cols, nnz, format, field, symmetry)
        let content = "%%MatrixMarket matrix coordinate real general\n2 3 4\n1 1 1.0\n1 2 2.0\n2 2 3.0\n2 3 4.0\n";
        let info = mminfo(content).expect("mminfo");
        assert_eq!(info.rows, 2, "mminfo rows");
        assert_eq!(info.cols, 3, "mminfo cols");
        assert_eq!(info.nnz, 4, "mminfo nnz");
    }

    #[test]
    fn wav_write_read_roundtrip_matches_scipy_semantics() {
        // scipy.io.wavfile roundtrip preserves data
        let original = vec![0.5, -0.5, 0.25, -0.25];
        let bytes = wav_write(44100, 1, &original).expect("wav_write");
        let result = wav_read(&bytes).expect("wav_read");
        assert_eq!(result.sample_rate, 44100, "sample rate preserved");
        assert_eq!(result.data.len(), original.len(), "data length preserved");
    }

    fn idl_save_header() -> Vec<u8> {
        b"SR\x00\x04".to_vec()
    }

    fn idl_push_variable_record<F>(
        out: &mut Vec<u8>,
        name: &str,
        type_code: i32,
        varflags: i32,
        write_payload: F,
    ) where
        F: FnOnce(&mut Vec<u8>),
    {
        let record_start = idl_begin_record(out, 2);
        idl_push_string(out, name.as_bytes());
        idl_push_i32(out, type_code);
        idl_push_i32(out, varflags);
        idl_push_i32(out, 7);
        write_payload(out);
        idl_finish_record(out, record_start);
    }

    fn idl_push_array_variable_record<F>(
        out: &mut Vec<u8>,
        name: &str,
        type_code: i32,
        nbytes: usize,
        nelements: usize,
        dims: &[usize],
        write_payload: F,
    ) where
        F: FnOnce(&mut Vec<u8>),
    {
        let record_start = idl_begin_record(out, 2);
        idl_push_string(out, name.as_bytes());
        idl_push_i32(out, type_code);
        idl_push_i32(out, 4);
        idl_push_array_desc(out, nbytes, nelements, dims);
        idl_push_i32(out, 7);
        write_payload(out);
        idl_finish_record(out, record_start);
    }

    fn idl_push_end_record(out: &mut Vec<u8>) {
        idl_push_i32(out, 6);
        out.extend_from_slice(&0_u32.to_be_bytes());
        out.extend_from_slice(&0_u32.to_be_bytes());
        out.extend_from_slice(&0_u32.to_be_bytes());
    }

    fn idl_begin_record(out: &mut Vec<u8>, rectype: i32) -> usize {
        let start = out.len();
        idl_push_i32(out, rectype);
        out.extend_from_slice(&0_u32.to_be_bytes());
        out.extend_from_slice(&0_u32.to_be_bytes());
        out.extend_from_slice(&0_u32.to_be_bytes());
        start
    }

    fn idl_finish_record(out: &mut [u8], record_start: usize) {
        let next = u64::try_from(out.len()).expect("test fixture length fits u64");
        let low = u32::try_from(next & 0xffff_ffff).expect("low word fits u32");
        let high = u32::try_from(next >> 32).expect("high word fits u32");
        out[record_start + 4..record_start + 8].copy_from_slice(&low.to_be_bytes());
        out[record_start + 8..record_start + 12].copy_from_slice(&high.to_be_bytes());
    }

    fn idl_push_array_desc(out: &mut Vec<u8>, nbytes: usize, nelements: usize, dims: &[usize]) {
        idl_push_i32(out, 8);
        idl_push_i32(out, 0);
        idl_push_usize_as_i32(out, nbytes);
        idl_push_usize_as_i32(out, nelements);
        idl_push_usize_as_i32(out, dims.len());
        idl_push_i32(out, 0);
        idl_push_i32(out, 0);
        idl_push_i32(out, 8);
        for idx in 0..8 {
            idl_push_usize_as_i32(out, dims.get(idx).copied().unwrap_or(0));
        }
    }

    fn idl_push_string(out: &mut Vec<u8>, bytes: &[u8]) {
        idl_push_usize_as_i32(out, bytes.len());
        out.extend_from_slice(bytes);
        idl_pad_32(out);
    }

    fn idl_push_string_data(out: &mut Vec<u8>, bytes: &[u8]) {
        idl_push_usize_as_i32(out, bytes.len());
        if !bytes.is_empty() {
            idl_push_usize_as_i32(out, bytes.len());
            out.extend_from_slice(bytes);
            idl_pad_32(out);
        }
    }

    fn idl_pad_32(out: &mut Vec<u8>) {
        while !out.len().is_multiple_of(4) {
            out.push(0);
        }
    }

    fn idl_push_usize_as_i32(out: &mut Vec<u8>, value: usize) {
        let value = i32::try_from(value).expect("IDL test fixture field fits i32");
        idl_push_i32(out, value);
    }

    fn idl_push_i32(out: &mut Vec<u8>, value: i32) {
        out.extend_from_slice(&value.to_be_bytes());
    }
}
