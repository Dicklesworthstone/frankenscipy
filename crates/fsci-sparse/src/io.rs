//! SciPy-compatible sparse NPZ persistence.
//!
//! SciPy stores sparse objects as a ZIP archive of NPY members. This module
//! preserves its format-specific wire layout, including the `_is_array`
//! discriminator and N-dimensional COO `coords` member.

use crate::construct::SparseArrayOutput;
use crate::formats::{
    BsrMatrix, CooArray, CooMatrix, CscMatrix, CsrMatrix, DiaMatrix, Shape2D, SparseArray2D,
    SparseError, SparseFormat, SparseResult,
};
use crate::ops::FormatConvertible;
use crate::{SparseArray, SparseObject};
use npyz::{DType, NpyFile, Serialize, WriterBuilder};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, Write};
use std::path::{Path, PathBuf};
use zip::read::ZipFile;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

struct NpzArchive<R: Read + Seek> {
    zip: ZipArchive<R>,
}

impl<R: Read + Seek> NpzArchive<R> {
    fn new(reader: R) -> io::Result<Self> {
        ZipArchive::new(reader)
            .map(|zip| Self { zip })
            .map_err(zip_io)
    }

    fn array_names(&self) -> impl Iterator<Item = &str> {
        self.zip
            .file_names()
            .filter_map(|name| name.strip_suffix(".npy"))
    }

    fn by_name<'a>(&'a mut self, name: &str) -> io::Result<Option<NpyFile<ZipFile<'a, R>>>> {
        match self.zip.by_name(&format!("{name}.npy")) {
            Ok(file) => NpyFile::new(file).map(Some),
            Err(zip::result::ZipError::FileNotFound) => Ok(None),
            Err(error) => Err(zip_io(error)),
        }
    }
}

#[doc(hidden)]
pub struct NpzWriter<W: Write + Seek> {
    zip: ZipWriter<W>,
}

impl<W: Write + Seek> NpzWriter<W> {
    fn new(writer: W) -> Self {
        Self {
            zip: ZipWriter::new(writer),
        }
    }

    fn array<T: Serialize + ?Sized>(
        &mut self,
        name: &str,
        options: SimpleFileOptions,
    ) -> io::Result<
        npyz::write_options::WithWriter<&mut ZipWriter<W>, npyz::write_options::WriteOptions<T>>,
    > {
        self.zip
            .start_file(format!("{name}.npy"), options)
            .map_err(zip_io)?;
        Ok(npyz::WriteOptions::new().writer(&mut self.zip))
    }

    fn finish(self) -> io::Result<W> {
        self.zip.finish().map_err(zip_io)
    }
}

/// A format-polymorphic sparse matrix loaded from an NPZ archive.
#[derive(Debug, Clone, PartialEq)]
pub enum SparseMatrixOutput {
    Csr(CsrMatrix),
    Csc(CscMatrix),
    Coo(CooMatrix),
    Bsr(BsrMatrix),
    Dia(DiaMatrix),
}

impl SparseMatrixOutput {
    /// Return the SciPy format spelling for the contained matrix.
    #[must_use]
    pub fn format_name(&self) -> &'static str {
        format_name(self.format())
    }

    /// Convert the contained matrix to COO without changing its values.
    pub fn to_coo(&self) -> SparseResult<CooMatrix> {
        match self {
            Self::Csr(matrix) => matrix.to_coo(),
            Self::Csc(matrix) => matrix.to_coo(),
            Self::Coo(matrix) => Ok(matrix.clone()),
            Self::Bsr(matrix) => matrix.to_coo(),
            Self::Dia(matrix) => matrix.to_coo(),
        }
    }
}

impl SparseObject for SparseMatrixOutput {
    fn format(&self) -> SparseFormat {
        match self {
            Self::Csr(_) => SparseFormat::Csr,
            Self::Csc(_) => SparseFormat::Csc,
            Self::Coo(_) => SparseFormat::Coo,
            Self::Bsr(_) => SparseFormat::Bsr,
            Self::Dia(_) => SparseFormat::Dia,
        }
    }

    fn shape_nd(&self) -> Vec<usize> {
        let shape = match self {
            Self::Csr(matrix) => matrix.shape(),
            Self::Csc(matrix) => matrix.shape(),
            Self::Coo(matrix) => matrix.shape(),
            Self::Bsr(matrix) => matrix.shape(),
            Self::Dia(matrix) => matrix.shape(),
        };
        vec![shape.rows, shape.cols]
    }

    fn nnz(&self) -> usize {
        match self {
            Self::Csr(matrix) => matrix.nnz(),
            Self::Csc(matrix) => matrix.nnz(),
            Self::Coo(matrix) => matrix.nnz(),
            Self::Bsr(matrix) => matrix.nnz(),
            Self::Dia(matrix) => matrix.nnz(),
        }
    }

    fn is_matrix(&self) -> bool {
        true
    }
}

/// A sparse matrix or sparse array loaded from an NPZ archive.
#[derive(Debug, Clone, PartialEq)]
pub enum SparseNpz {
    Matrix(SparseMatrixOutput),
    Array(SparseArrayOutput),
}

impl SparseNpz {
    /// Return the SciPy format spelling stored in the archive.
    #[must_use]
    pub fn format_name(&self) -> &'static str {
        format_name(self.format())
    }

    /// Convert either family to the N-dimensional COO foundation.
    pub fn to_coo_array(&self) -> SparseResult<CooArray> {
        match self {
            Self::Matrix(matrix) => matrix.to_coo().map(|coo| CooArray::from_coo_matrix(&coo)),
            Self::Array(array) => array.to_coo_array(),
        }
    }
}

impl SparseObject for SparseNpz {
    fn format(&self) -> SparseFormat {
        match self {
            Self::Matrix(matrix) => matrix.format(),
            Self::Array(array) => array.format(),
        }
    }

    fn shape_nd(&self) -> Vec<usize> {
        match self {
            Self::Matrix(matrix) => matrix.shape_nd(),
            Self::Array(array) => array.shape_nd(),
        }
    }

    fn nnz(&self) -> usize {
        match self {
            Self::Matrix(matrix) => matrix.nnz(),
            Self::Array(array) => array.nnz(),
        }
    }

    fn is_matrix(&self) -> bool {
        matches!(self, Self::Matrix(_))
    }
}

/// A sparse object that can emit SciPy's NPZ member layout.
pub trait NpzWritable {
    #[doc(hidden)]
    fn write_npz_entries<W: Write + Seek>(
        &self,
        npz: &mut NpzWriter<W>,
        compressed: bool,
    ) -> io::Result<()>;

    #[doc(hidden)]
    fn npz_is_array(&self) -> bool {
        false
    }
}

impl NpzWritable for CsrMatrix {
    fn write_npz_entries<W: Write + Seek>(
        &self,
        npz: &mut NpzWriter<W>,
        compressed: bool,
    ) -> io::Result<()> {
        write_format(npz, "csr", compressed)?;
        write_shape(npz, &[self.shape().rows, self.shape().cols], compressed)?;
        write_usize_member(npz, "indices", self.indices(), &[self.nnz()], compressed)?;
        write_usize_member(
            npz,
            "indptr",
            self.indptr(),
            &[self.indptr().len()],
            compressed,
        )?;
        write_f64_member(npz, "data", self.data(), &[self.nnz()], compressed)
    }
}

impl NpzWritable for CscMatrix {
    fn write_npz_entries<W: Write + Seek>(
        &self,
        npz: &mut NpzWriter<W>,
        compressed: bool,
    ) -> io::Result<()> {
        write_format(npz, "csc", compressed)?;
        write_shape(npz, &[self.shape().rows, self.shape().cols], compressed)?;
        write_usize_member(npz, "indices", self.indices(), &[self.nnz()], compressed)?;
        write_usize_member(
            npz,
            "indptr",
            self.indptr(),
            &[self.indptr().len()],
            compressed,
        )?;
        write_f64_member(npz, "data", self.data(), &[self.nnz()], compressed)
    }
}

impl NpzWritable for CooMatrix {
    fn write_npz_entries<W: Write + Seek>(
        &self,
        npz: &mut NpzWriter<W>,
        compressed: bool,
    ) -> io::Result<()> {
        write_format(npz, "coo", compressed)?;
        write_shape(npz, &[self.shape().rows, self.shape().cols], compressed)?;
        write_usize_member(npz, "row", self.row_indices(), &[self.nnz()], compressed)?;
        write_usize_member(npz, "col", self.col_indices(), &[self.nnz()], compressed)?;
        write_f64_member(npz, "data", self.data(), &[self.nnz()], compressed)
    }
}

impl NpzWritable for CooArray {
    fn write_npz_entries<W: Write + Seek>(
        &self,
        npz: &mut NpzWriter<W>,
        compressed: bool,
    ) -> io::Result<()> {
        write_format(npz, "coo", compressed)?;
        write_shape(npz, self.shape(), compressed)?;
        if self.ndim() == 2 {
            write_usize_member(npz, "row", &self.coords()[0], &[self.nnz()], compressed)?;
            write_usize_member(npz, "col", &self.coords()[1], &[self.nnz()], compressed)?;
        } else {
            let coordinate_len = self
                .ndim()
                .checked_mul(self.nnz())
                .ok_or_else(|| SparseError::IndexOverflow {
                    message: "COO coordinate member length overflows usize".to_string(),
                })
                .map_err(sparse_io)?;
            let mut coordinates = Vec::with_capacity(coordinate_len);
            for axis in self.coords() {
                coordinates.extend_from_slice(axis);
            }
            write_usize_member(
                npz,
                "coords",
                &coordinates,
                &[self.ndim(), self.nnz()],
                compressed,
            )?;
        }
        write_f64_member(npz, "data", self.data(), &[self.nnz()], compressed)
    }

    fn npz_is_array(&self) -> bool {
        true
    }
}

impl NpzWritable for BsrMatrix {
    fn write_npz_entries<W: Write + Seek>(
        &self,
        npz: &mut NpzWriter<W>,
        compressed: bool,
    ) -> io::Result<()> {
        write_format(npz, "bsr", compressed)?;
        write_shape(npz, &[self.shape().rows, self.shape().cols], compressed)?;
        write_usize_member(
            npz,
            "indices",
            self.indices(),
            &[self.indices().len()],
            compressed,
        )?;
        write_usize_member(
            npz,
            "indptr",
            self.indptr(),
            &[self.indptr().len()],
            compressed,
        )?;
        let block_shape = self.block_shape();
        let block_area = block_shape
            .rows
            .checked_mul(block_shape.cols)
            .ok_or_else(|| SparseError::IndexOverflow {
                message: "BSR block area overflows usize".to_string(),
            })
            .map_err(sparse_io)?;
        let data_len = self
            .data()
            .len()
            .checked_mul(block_area)
            .ok_or_else(|| SparseError::IndexOverflow {
                message: "BSR data member length overflows usize".to_string(),
            })
            .map_err(sparse_io)?;
        let mut data = Vec::with_capacity(data_len);
        for block in self.data() {
            data.extend_from_slice(block);
        }
        write_f64_member(
            npz,
            "data",
            &data,
            &[self.data().len(), block_shape.rows, block_shape.cols],
            compressed,
        )
    }
}

impl NpzWritable for DiaMatrix {
    fn write_npz_entries<W: Write + Seek>(
        &self,
        npz: &mut NpzWriter<W>,
        compressed: bool,
    ) -> io::Result<()> {
        write_format(npz, "dia", compressed)?;
        let shape = self.shape();
        write_shape(npz, &[shape.rows, shape.cols], compressed)?;
        write_isize_member(
            npz,
            "offsets",
            self.offsets(),
            &[self.offsets().len()],
            compressed,
        )?;

        let data_len = self
            .offsets()
            .len()
            .checked_mul(shape.cols)
            .ok_or_else(|| SparseError::IndexOverflow {
                message: "DIA data member length overflows usize".to_string(),
            })
            .map_err(sparse_io)?;
        let mut data = vec![0.0; data_len];
        for (diagonal_index, (&offset, diagonal)) in
            self.offsets().iter().zip(self.data()).enumerate()
        {
            let first_col = if offset > 0 {
                usize::try_from(offset)
                    .map_err(|_| invalid_data("positive DIA offset does not fit in usize"))?
            } else {
                0
            };
            for (entry, &value) in diagonal.iter().enumerate() {
                let column = first_col
                    .checked_add(entry)
                    .ok_or_else(|| invalid_data("DIA column offset overflows usize"))?;
                let target = diagonal_index
                    .checked_mul(shape.cols)
                    .and_then(|base| base.checked_add(column))
                    .ok_or_else(|| invalid_data("DIA data offset overflows usize"))?;
                data[target] = value;
            }
        }
        write_f64_member(
            npz,
            "data",
            &data,
            &[self.offsets().len(), shape.cols],
            compressed,
        )
    }
}

impl<M: NpzWritable> NpzWritable for SparseArray2D<M> {
    fn write_npz_entries<W: Write + Seek>(
        &self,
        npz: &mut NpzWriter<W>,
        compressed: bool,
    ) -> io::Result<()> {
        self.as_matrix().write_npz_entries(npz, compressed)
    }

    fn npz_is_array(&self) -> bool {
        true
    }
}

impl NpzWritable for SparseMatrixOutput {
    fn write_npz_entries<W: Write + Seek>(
        &self,
        npz: &mut NpzWriter<W>,
        compressed: bool,
    ) -> io::Result<()> {
        match self {
            Self::Csr(matrix) => matrix.write_npz_entries(npz, compressed),
            Self::Csc(matrix) => matrix.write_npz_entries(npz, compressed),
            Self::Coo(matrix) => matrix.write_npz_entries(npz, compressed),
            Self::Bsr(matrix) => matrix.write_npz_entries(npz, compressed),
            Self::Dia(matrix) => matrix.write_npz_entries(npz, compressed),
        }
    }
}

impl NpzWritable for SparseArrayOutput {
    fn write_npz_entries<W: Write + Seek>(
        &self,
        npz: &mut NpzWriter<W>,
        compressed: bool,
    ) -> io::Result<()> {
        match self {
            Self::Csr(array) => array.write_npz_entries(npz, compressed),
            Self::Csc(array) => array.write_npz_entries(npz, compressed),
            Self::Coo(array) => array.write_npz_entries(npz, compressed),
            Self::Bsr(array) => array.write_npz_entries(npz, compressed),
            Self::Dia(array) => array.write_npz_entries(npz, compressed),
            Self::Dok(_) | Self::Lil(_) => Err(invalid_data(
                "SciPy NPZ persistence supports CSR, CSC, COO, BSR, and DIA only",
            )),
        }
    }

    fn npz_is_array(&self) -> bool {
        true
    }
}

impl NpzWritable for SparseNpz {
    fn write_npz_entries<W: Write + Seek>(
        &self,
        npz: &mut NpzWriter<W>,
        compressed: bool,
    ) -> io::Result<()> {
        match self {
            Self::Matrix(matrix) => matrix.write_npz_entries(npz, compressed),
            Self::Array(array) => array.write_npz_entries(npz, compressed),
        }
    }

    fn npz_is_array(&self) -> bool {
        matches!(self, Self::Array(_))
    }
}

/// Save a sparse matrix or array to a SciPy-compatible NPZ file.
///
/// Like SciPy, this appends `.npz` when the supplied path does not already end
/// with that extension.
pub fn save_npz<P, S>(file: P, sparse: &S, compressed: bool) -> io::Result<()>
where
    P: AsRef<Path>,
    S: NpzWritable + ?Sized,
{
    let path = with_npz_extension(file.as_ref());
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    save_npz_to_writer(&mut writer, sparse, compressed)?;
    writer.flush()
}

/// Save a sparse matrix or array to a seekable output stream.
pub fn save_npz_to_writer<W, S>(mut writer: W, sparse: &S, compressed: bool) -> io::Result<()>
where
    W: Write + Seek,
    S: NpzWritable + ?Sized,
{
    {
        let mut npz = NpzWriter::new(&mut writer);
        sparse.write_npz_entries(&mut npz, compressed)?;
        if sparse.npz_is_array() {
            write_array_flag(&mut npz, compressed)?;
        }
        let _ = npz.finish()?;
    }
    writer.flush()
}

/// Load a SciPy-compatible sparse NPZ file.
pub fn load_npz<P: AsRef<Path>>(file: P) -> io::Result<SparseNpz> {
    let reader = BufReader::new(File::open(file)?);
    load_npz_from_reader(reader)
}

/// Load a SciPy-compatible sparse NPZ archive from a seekable stream.
pub fn load_npz_from_reader<R: Read + Seek>(reader: R) -> io::Result<SparseNpz> {
    let mut npz = NpzArchive::new(reader)?;
    let format = read_format(&mut npz)?;
    let is_array = read_array_flag(&mut npz)?;
    let shape = read_shape(&mut npz)?;

    match format.as_str() {
        "csr" => load_csr(&mut npz, shape, is_array),
        "csc" => load_csc(&mut npz, shape, is_array),
        "coo" => load_coo(&mut npz, shape, is_array),
        "bsr" => load_bsr(&mut npz, shape, is_array),
        "dia" => load_dia(&mut npz, shape, is_array),
        _ => Err(invalid_data(format!(
            "load_npz does not support sparse format {format:?}"
        ))),
    }
}

fn load_csr<R: Read + Seek>(
    npz: &mut NpzArchive<R>,
    shape: Vec<usize>,
    is_array: bool,
) -> io::Result<SparseNpz> {
    let shape = shape_2d(&shape)?;
    let (data, data_shape) = read_numeric_member(npz, "data")?;
    expect_shape("data", &data_shape, &[data.len()])?;
    let (indices, indices_shape) = read_usize_member(npz, "indices")?;
    expect_shape("indices", &indices_shape, &[indices.len()])?;
    let (indptr, indptr_shape) = read_usize_member(npz, "indptr")?;
    expect_shape("indptr", &indptr_shape, &[indptr.len()])?;
    let matrix =
        CsrMatrix::from_components(shape, data, indices, indptr, false).map_err(sparse_io)?;
    Ok(if is_array {
        SparseNpz::Array(SparseArrayOutput::Csr(SparseArray2D::new(matrix)))
    } else {
        SparseNpz::Matrix(SparseMatrixOutput::Csr(matrix))
    })
}

fn load_csc<R: Read + Seek>(
    npz: &mut NpzArchive<R>,
    shape: Vec<usize>,
    is_array: bool,
) -> io::Result<SparseNpz> {
    let shape = shape_2d(&shape)?;
    let (data, data_shape) = read_numeric_member(npz, "data")?;
    expect_shape("data", &data_shape, &[data.len()])?;
    let (indices, indices_shape) = read_usize_member(npz, "indices")?;
    expect_shape("indices", &indices_shape, &[indices.len()])?;
    let (indptr, indptr_shape) = read_usize_member(npz, "indptr")?;
    expect_shape("indptr", &indptr_shape, &[indptr.len()])?;
    let matrix =
        CscMatrix::from_components(shape, data, indices, indptr, false).map_err(sparse_io)?;
    Ok(if is_array {
        SparseNpz::Array(SparseArrayOutput::Csc(SparseArray2D::new(matrix)))
    } else {
        SparseNpz::Matrix(SparseMatrixOutput::Csc(matrix))
    })
}

fn load_coo<R: Read + Seek>(
    npz: &mut NpzArchive<R>,
    shape: Vec<usize>,
    is_array: bool,
) -> io::Result<SparseNpz> {
    let (data, data_shape) = read_numeric_member(npz, "data")?;
    expect_shape("data", &data_shape, &[data.len()])?;
    let has_coords = npz.array_names().any(|name| name == "coords");

    let coordinates = if has_coords {
        let (flat, coordinate_shape) = read_usize_member(npz, "coords")?;
        expect_shape("coords", &coordinate_shape, &[shape.len(), data.len()])?;
        let expected_len = shape
            .len()
            .checked_mul(data.len())
            .ok_or_else(|| SparseError::IndexOverflow {
                message: "COO coordinate member length overflows usize".to_string(),
            })
            .map_err(sparse_io)?;
        if flat.len() != expected_len {
            return Err(invalid_data(format!(
                "coords contains {} indices, expected {expected_len}",
                flat.len()
            )));
        }
        (0..shape.len())
            .map(|axis| {
                let start = axis * data.len();
                flat[start..start + data.len()].to_vec()
            })
            .collect()
    } else {
        let shape_2d = shape_2d(&shape)?;
        let (row, row_shape) = read_usize_member(npz, "row")?;
        let (col, col_shape) = read_usize_member(npz, "col")?;
        expect_shape("row", &row_shape, &[data.len()])?;
        expect_shape("col", &col_shape, &[data.len()])?;
        if row.len() != data.len() || col.len() != data.len() {
            return Err(invalid_data(
                "COO row, col, and data members must have equal lengths",
            ));
        }
        let _ = shape_2d;
        vec![row, col]
    };

    if is_array {
        let array = CooArray::from_coords(shape, data, coordinates, false).map_err(sparse_io)?;
        Ok(SparseNpz::Array(SparseArrayOutput::Coo(array)))
    } else {
        let shape = shape_2d(&shape)?;
        let matrix = CooMatrix::from_triplets(
            shape,
            data,
            coordinates[0].clone(),
            coordinates[1].clone(),
            false,
        )
        .map_err(sparse_io)?;
        Ok(SparseNpz::Matrix(SparseMatrixOutput::Coo(matrix)))
    }
}

fn load_bsr<R: Read + Seek>(
    npz: &mut NpzArchive<R>,
    shape: Vec<usize>,
    is_array: bool,
) -> io::Result<SparseNpz> {
    let shape = shape_2d(&shape)?;
    let (indices, indices_shape) = read_usize_member(npz, "indices")?;
    expect_shape("indices", &indices_shape, &[indices.len()])?;
    let (indptr, indptr_shape) = read_usize_member(npz, "indptr")?;
    expect_shape("indptr", &indptr_shape, &[indptr.len()])?;
    let (flat_data, data_shape) = read_numeric_member(npz, "data")?;
    if data_shape.len() != 3 {
        return Err(invalid_data(format!(
            "BSR data must be three-dimensional, got shape {data_shape:?}"
        )));
    }
    if data_shape[0] != indices.len() {
        return Err(invalid_data(format!(
            "BSR data has {} blocks but indices has {}",
            data_shape[0],
            indices.len()
        )));
    }
    let block_shape = Shape2D::new(data_shape[1], data_shape[2]);
    let block_area = block_shape
        .rows
        .checked_mul(block_shape.cols)
        .ok_or_else(|| SparseError::IndexOverflow {
            message: "BSR block area overflows usize".to_string(),
        })
        .map_err(sparse_io)?;
    let expected_len = indices
        .len()
        .checked_mul(block_area)
        .ok_or_else(|| SparseError::IndexOverflow {
            message: "BSR data member length overflows usize".to_string(),
        })
        .map_err(sparse_io)?;
    if flat_data.len() != expected_len {
        return Err(invalid_data(format!(
            "BSR data contains {} values, expected {expected_len}",
            flat_data.len()
        )));
    }
    if block_area == 0 {
        return Err(invalid_data("BSR blocks must have nonzero dimensions"));
    }
    let data = flat_data.chunks(block_area).map(<[f64]>::to_vec).collect();
    let matrix = BsrMatrix::from_components(shape, block_shape, data, indices, indptr, false)
        .map_err(sparse_io)?;
    Ok(if is_array {
        SparseNpz::Array(SparseArrayOutput::Bsr(SparseArray2D::new(matrix)))
    } else {
        SparseNpz::Matrix(SparseMatrixOutput::Bsr(matrix))
    })
}

fn load_dia<R: Read + Seek>(
    npz: &mut NpzArchive<R>,
    shape: Vec<usize>,
    is_array: bool,
) -> io::Result<SparseNpz> {
    let shape = shape_2d(&shape)?;
    let (raw_offsets, offsets_shape) = read_signed_member(npz, "offsets")?;
    expect_shape("offsets", &offsets_shape, &[raw_offsets.len()])?;
    let offsets: Vec<isize> = raw_offsets
        .into_iter()
        .map(|offset| {
            isize::try_from(offset)
                .map_err(|_| invalid_data("DIA offset does not fit in platform isize"))
        })
        .collect::<io::Result<_>>()?;
    let (flat_data, data_shape) = read_numeric_member(npz, "data")?;
    if data_shape.len() != 2 || data_shape[0] != offsets.len() {
        return Err(invalid_data(format!(
            "DIA data shape {data_shape:?} does not match {} offsets",
            offsets.len()
        )));
    }
    let width = data_shape[1];
    let expected_len = offsets
        .len()
        .checked_mul(width)
        .ok_or_else(|| SparseError::IndexOverflow {
            message: "DIA data member length overflows usize".to_string(),
        })
        .map_err(sparse_io)?;
    if flat_data.len() != expected_len {
        return Err(invalid_data(format!(
            "DIA data contains {} values, expected {expected_len}",
            flat_data.len()
        )));
    }

    let mut diagonals = Vec::with_capacity(offsets.len());
    for (diagonal_index, &offset) in offsets.iter().enumerate() {
        let row = &flat_data[diagonal_index * width..(diagonal_index + 1) * width];
        let mut values = Vec::new();
        let mut matrix_row = if offset < 0 { offset.unsigned_abs() } else { 0 };
        let mut matrix_col = if offset > 0 {
            usize::try_from(offset)
                .map_err(|_| invalid_data("positive DIA offset does not fit in usize"))?
        } else {
            0
        };
        while matrix_row < shape.rows && matrix_col < shape.cols {
            values.push(row.get(matrix_col).copied().unwrap_or(0.0));
            matrix_row += 1;
            matrix_col += 1;
        }
        diagonals.push(values);
    }
    let matrix = DiaMatrix::from_diagonals(shape, offsets, diagonals).map_err(sparse_io)?;
    Ok(if is_array {
        SparseNpz::Array(SparseArrayOutput::Dia(SparseArray2D::new(matrix)))
    } else {
        SparseNpz::Matrix(SparseMatrixOutput::Dia(matrix))
    })
}

fn write_format<W: Write + Seek>(
    npz: &mut NpzWriter<W>,
    format: &str,
    compressed: bool,
) -> io::Result<()> {
    let dtype = "|S3"
        .parse()
        .map(DType::Plain)
        .map_err(|error| invalid_data(format!("invalid fixed NPZ format dtype: {error}")))?;
    let mut writer = npz
        .array("format", file_options(compressed))?
        .dtype(dtype)
        .shape(&[])
        .begin_nd()?;
    writer.push(format.as_bytes())?;
    writer.finish()
}

fn write_array_flag<W: Write + Seek>(npz: &mut NpzWriter<W>, compressed: bool) -> io::Result<()> {
    let mut writer = npz
        .array("_is_array", file_options(compressed))?
        .default_dtype()
        .shape(&[])
        .begin_nd()?;
    writer.push(&true)?;
    writer.finish()
}

fn write_shape<W: Write + Seek>(
    npz: &mut NpzWriter<W>,
    shape: &[usize],
    compressed: bool,
) -> io::Result<()> {
    let values = shape
        .iter()
        .copied()
        .map(|extent| {
            i64::try_from(extent)
                .map_err(|_| invalid_data("sparse extent does not fit in NumPy int64"))
        })
        .collect::<io::Result<Vec<_>>>()?;
    write_i64_member(npz, "shape", &values, &[values.len()], compressed)
}

fn write_usize_member<W: Write + Seek>(
    npz: &mut NpzWriter<W>,
    name: &str,
    values: &[usize],
    shape: &[usize],
    compressed: bool,
) -> io::Result<()> {
    if values.iter().all(|&value| i32::try_from(value).is_ok()) {
        let values: Vec<i32> = values.iter().map(|&value| value as i32).collect();
        write_i32_member(npz, name, &values, shape, compressed)
    } else {
        let values = values
            .iter()
            .copied()
            .map(|value| {
                i64::try_from(value)
                    .map_err(|_| invalid_data(format!("{name} index does not fit in NumPy int64")))
            })
            .collect::<io::Result<Vec<_>>>()?;
        write_i64_member(npz, name, &values, shape, compressed)
    }
}

fn write_isize_member<W: Write + Seek>(
    npz: &mut NpzWriter<W>,
    name: &str,
    values: &[isize],
    shape: &[usize],
    compressed: bool,
) -> io::Result<()> {
    if values.iter().all(|&value| i32::try_from(value).is_ok()) {
        let values: Vec<i32> = values.iter().map(|&value| value as i32).collect();
        write_i32_member(npz, name, &values, shape, compressed)
    } else {
        let values: Vec<i64> = values
            .iter()
            .map(|&value| {
                i64::try_from(value)
                    .map_err(|_| invalid_data(format!("{name} index does not fit in NumPy int64")))
            })
            .collect::<io::Result<_>>()?;
        write_i64_member(npz, name, &values, shape, compressed)
    }
}

fn write_i32_member<W: Write + Seek>(
    npz: &mut NpzWriter<W>,
    name: &str,
    values: &[i32],
    shape: &[usize],
    compressed: bool,
) -> io::Result<()> {
    let shape = npy_shape(shape)?;
    let mut writer = npz
        .array(name, file_options(compressed))?
        .default_dtype()
        .shape(&shape)
        .begin_nd()?;
    writer.extend(values.iter().copied())?;
    writer.finish()
}

fn write_i64_member<W: Write + Seek>(
    npz: &mut NpzWriter<W>,
    name: &str,
    values: &[i64],
    shape: &[usize],
    compressed: bool,
) -> io::Result<()> {
    let shape = npy_shape(shape)?;
    let mut writer = npz
        .array(name, file_options(compressed))?
        .default_dtype()
        .shape(&shape)
        .begin_nd()?;
    writer.extend(values.iter().copied())?;
    writer.finish()
}

fn write_f64_member<W: Write + Seek>(
    npz: &mut NpzWriter<W>,
    name: &str,
    values: &[f64],
    shape: &[usize],
    compressed: bool,
) -> io::Result<()> {
    let shape = npy_shape(shape)?;
    let mut writer = npz
        .array(name, file_options(compressed))?
        .default_dtype()
        .shape(&shape)
        .begin_nd()?;
    writer.extend(values.iter().copied())?;
    writer.finish()
}

fn read_format<R: Read + Seek>(npz: &mut NpzArchive<R>) -> io::Result<String> {
    let npy = required_member(npz, "format")?;
    expect_shape("format", &npy_shape_from_header(&npy)?, &[])?;
    let values = npy.into_vec::<Vec<u8>>()?;
    let value = values
        .into_iter()
        .next()
        .ok_or_else(|| invalid_data("format scalar is empty"))?;
    String::from_utf8(value).map_err(|error| invalid_data(format!("format is not ASCII: {error}")))
}

fn read_array_flag<R: Read + Seek>(npz: &mut NpzArchive<R>) -> io::Result<bool> {
    let Some(npy) = npz.by_name("_is_array")? else {
        return Ok(false);
    };
    expect_shape("_is_array", &npy_shape_from_header(&npy)?, &[])?;
    let values = npy.into_vec::<bool>()?;
    values
        .into_iter()
        .next()
        .ok_or_else(|| invalid_data("_is_array scalar is empty"))
}

fn read_shape<R: Read + Seek>(npz: &mut NpzArchive<R>) -> io::Result<Vec<usize>> {
    let (values, member_shape) = read_signed_member(npz, "shape")?;
    if member_shape.len() != 1 || member_shape[0] != values.len() {
        return Err(invalid_data(format!(
            "shape member must be one-dimensional, got {member_shape:?}"
        )));
    }
    values
        .into_iter()
        .map(|value| {
            if value < 0 {
                return Err(invalid_data(format!(
                    "sparse shape contains negative extent {value}"
                )));
            }
            usize::try_from(value)
                .map_err(|_| invalid_data("sparse extent does not fit in platform usize"))
        })
        .collect()
}

fn read_usize_member<R: Read + Seek>(
    npz: &mut NpzArchive<R>,
    name: &str,
) -> io::Result<(Vec<usize>, Vec<usize>)> {
    let (values, shape) = read_signed_member(npz, name)?;
    let values = values
        .into_iter()
        .map(|value| {
            if value < 0 {
                return Err(invalid_data(format!(
                    "{name} contains negative index {value}"
                )));
            }
            usize::try_from(value)
                .map_err(|_| invalid_data(format!("{name} index does not fit in platform usize")))
        })
        .collect::<io::Result<_>>()?;
    Ok((values, shape))
}

fn read_signed_member<R: Read + Seek>(
    npz: &mut NpzArchive<R>,
    name: &str,
) -> io::Result<(Vec<i64>, Vec<usize>)> {
    let npy = required_member(npz, name)?;
    let shape = npy_shape_from_header(&npy)?;
    let npy = match npy.try_data::<i32>() {
        Ok(data) => {
            let values = data
                .map(|value| value.map(i64::from))
                .collect::<io::Result<_>>()?;
            return Ok((values, shape));
        }
        Err(npy) => npy,
    };
    match npy.try_data::<i64>() {
        Ok(data) => Ok((data.collect::<io::Result<_>>()?, shape)),
        Err(npy) => Err(invalid_data(format!(
            "{name} has unsupported integer dtype {}",
            npy.dtype().descr()
        ))),
    }
}

fn read_numeric_member<R: Read + Seek>(
    npz: &mut NpzArchive<R>,
    name: &str,
) -> io::Result<(Vec<f64>, Vec<usize>)> {
    let npy = required_member(npz, name)?;
    let shape = npy_shape_from_header(&npy)?;

    macro_rules! try_numeric_type {
        ($npy:ident, $type:ty) => {
            let $npy = match $npy.try_data::<$type>() {
                Ok(data) => {
                    let values = data
                        .map(|value| value.map(|value| value as f64))
                        .collect::<io::Result<_>>()?;
                    return Ok((values, shape));
                }
                Err(npy) => npy,
            };
        };
    }

    try_numeric_type!(npy, f64);
    try_numeric_type!(npy, f32);
    try_numeric_type!(npy, i64);
    try_numeric_type!(npy, i32);
    try_numeric_type!(npy, i16);
    try_numeric_type!(npy, i8);
    try_numeric_type!(npy, u64);
    try_numeric_type!(npy, u32);
    try_numeric_type!(npy, u16);
    try_numeric_type!(npy, u8);
    let npy = match npy.try_data::<bool>() {
        Ok(data) => {
            let values = data
                .map(|value| value.map(|value| f64::from(u8::from(value))))
                .collect::<io::Result<_>>()?;
            return Ok((values, shape));
        }
        Err(npy) => npy,
    };

    Err(invalid_data(format!(
        "{name} has unsupported numeric dtype {}",
        npy.dtype().descr()
    )))
}

fn required_member<'a, R: Read + Seek>(
    npz: &'a mut NpzArchive<R>,
    name: &str,
) -> io::Result<NpyFile<ZipFile<'a, R>>> {
    npz.by_name(name)?
        .ok_or_else(|| invalid_data(format!("sparse NPZ archive is missing {name}.npy")))
}

fn npy_shape_from_header<R: Read>(npy: &NpyFile<R>) -> io::Result<Vec<usize>> {
    npy.shape()
        .iter()
        .copied()
        .map(|extent| {
            usize::try_from(extent)
                .map_err(|_| invalid_data("NPY member extent does not fit in platform usize"))
        })
        .collect()
}

fn npy_shape(shape: &[usize]) -> io::Result<Vec<u64>> {
    shape
        .iter()
        .copied()
        .map(|extent| {
            u64::try_from(extent).map_err(|_| invalid_data("NPY member extent does not fit in u64"))
        })
        .collect()
}

fn shape_2d(shape: &[usize]) -> io::Result<Shape2D> {
    match shape {
        [rows, cols] => Ok(Shape2D::new(*rows, *cols)),
        _ => Err(invalid_data(format!(
            "sparse matrix format requires a two-dimensional shape, got {shape:?}"
        ))),
    }
}

fn expect_shape(name: &str, actual: &[usize], expected: &[usize]) -> io::Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(invalid_data(format!(
            "{name} has NPY shape {actual:?}, expected {expected:?}"
        )))
    }
}

fn file_options(compressed: bool) -> SimpleFileOptions {
    SimpleFileOptions::default().compression_method(if compressed {
        CompressionMethod::Deflated
    } else {
        CompressionMethod::Stored
    })
}

fn with_npz_extension(path: &Path) -> PathBuf {
    if path.extension().is_some_and(|extension| extension == "npz") {
        path.to_path_buf()
    } else {
        let mut path = path.as_os_str().to_os_string();
        path.push(".npz");
        PathBuf::from(path)
    }
}

const fn format_name(format: SparseFormat) -> &'static str {
    match format {
        SparseFormat::Csr => "csr",
        SparseFormat::Csc => "csc",
        SparseFormat::Coo => "coo",
        SparseFormat::Bsr => "bsr",
        SparseFormat::Dia => "dia",
        SparseFormat::Dok => "dok",
        SparseFormat::Lil => "lil",
    }
}

fn sparse_io(error: SparseError) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error)
}

fn zip_io(error: zip::result::ZipError) -> io::Error {
    match error {
        zip::result::ZipError::Io(error) => error,
        error => invalid_data(error.to_string()),
    }
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn fixture_coo() -> CooMatrix {
        CooMatrix::from_triplets(
            Shape2D::new(4, 4),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![0, 0, 1, 1, 2, 2, 3, 3],
            vec![0, 2, 1, 3, 0, 2, 1, 3],
            false,
        )
        .expect("valid COO fixture")
    }

    fn fixture_matrices() -> Vec<SparseMatrixOutput> {
        let coo = fixture_coo();
        vec![
            SparseMatrixOutput::Csr(coo.to_csr().expect("CSR fixture")),
            SparseMatrixOutput::Csc(coo.to_csc().expect("CSC fixture")),
            SparseMatrixOutput::Coo(coo.clone()),
            SparseMatrixOutput::Bsr(
                BsrMatrix::from_triplets(
                    coo.shape(),
                    Shape2D::new(2, 2),
                    coo.data().to_vec(),
                    coo.row_indices().to_vec(),
                    coo.col_indices().to_vec(),
                )
                .expect("BSR fixture"),
            ),
            SparseMatrixOutput::Dia(
                DiaMatrix::from_triplets(
                    coo.shape(),
                    coo.data().to_vec(),
                    coo.row_indices().to_vec(),
                    coo.col_indices().to_vec(),
                )
                .expect("DIA fixture"),
            ),
        ]
    }

    fn round_trip<S: NpzWritable + ?Sized>(value: &S, compressed: bool) -> SparseNpz {
        let mut archive = Cursor::new(Vec::new());
        save_npz_to_writer(&mut archive, value, compressed).expect("write NPZ");
        archive.set_position(0);
        load_npz_from_reader(archive).expect("read NPZ")
    }

    #[test]
    fn matrix_formats_round_trip_compressed_and_stored() {
        for matrix in fixture_matrices() {
            for compressed in [false, true] {
                let loaded = round_trip(&matrix, compressed);
                assert_eq!(loaded, SparseNpz::Matrix(matrix.clone()));
            }
        }
    }

    #[test]
    fn array_formats_preserve_array_identity() {
        for matrix in fixture_matrices() {
            let array = match matrix {
                SparseMatrixOutput::Csr(matrix) => {
                    SparseArrayOutput::Csr(SparseArray2D::new(matrix))
                }
                SparseMatrixOutput::Csc(matrix) => {
                    SparseArrayOutput::Csc(SparseArray2D::new(matrix))
                }
                SparseMatrixOutput::Coo(matrix) => {
                    SparseArrayOutput::Coo(CooArray::from_coo_matrix(&matrix))
                }
                SparseMatrixOutput::Bsr(matrix) => {
                    SparseArrayOutput::Bsr(SparseArray2D::new(matrix))
                }
                SparseMatrixOutput::Dia(matrix) => {
                    SparseArrayOutput::Dia(SparseArray2D::new(matrix))
                }
            };
            let loaded = round_trip(&array, true);
            assert_eq!(loaded, SparseNpz::Array(array));
        }
    }

    #[test]
    fn nd_coo_array_round_trip_preserves_coords() {
        let array = CooArray::from_coords(
            vec![2, 3, 2],
            vec![1.0, 2.0, 3.0],
            vec![vec![0, 1, 1], vec![1, 0, 2], vec![0, 1, 1]],
            false,
        )
        .expect("valid N-D COO array");
        assert_eq!(
            round_trip(&array, true),
            SparseNpz::Array(SparseArrayOutput::Coo(array))
        );
    }

    #[test]
    fn malformed_archive_is_rejected() {
        let error = load_npz_from_reader(Cursor::new(b"not a zip archive".to_vec()))
            .expect_err("invalid archive must fail");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn scipy_style_extension_is_appended_only_when_needed() {
        assert_eq!(
            with_npz_extension(Path::new("fixture")),
            PathBuf::from("fixture.npz")
        );
        assert_eq!(
            with_npz_extension(Path::new("fixture.npz")),
            PathBuf::from("fixture.npz")
        );
        assert_eq!(
            with_npz_extension(Path::new("fixture.bin")),
            PathBuf::from("fixture.bin.npz")
        );
    }
}
