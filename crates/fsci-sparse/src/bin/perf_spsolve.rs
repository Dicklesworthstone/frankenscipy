//! Same-process A/B + parity harness for spsolve routing.
//!
//! Before: spsolve densified any sparse A (n<=32768) into an n×n dense matrix and
//! ran O(n³) nalgebra dense LU. After: genuinely-sparse A routes to the native
//! sparse LU (~O(n·fill)). On a diagonally-dominant pentadiagonal system the fill
//! is O(n), so the sparse path is orders of magnitude cheaper. The solution is
//! unique, so x matches the dense path to rounding (PARITY block prints max|Δx|).
//! Run: `cargo run --profile release-perf -p fsci-sparse --bin perf_spsolve`.

use std::hint::black_box;
use std::time::Instant;

use fsci_sparse::{
    CooMatrix, CscMatrix, CsrMatrix, FormatConvertible, LuOptions, PermutationOrdering, Shape2D,
    SolveOptions, splu, splu_solve, spsolve,
};
use nalgebra::{DMatrix, DVector};

// Pentadiagonal whose row/col labels are scrambled by a fixed pseudo-random symmetric
// permutation: same nnz (~5/row) but huge bandwidth in natural order, so natural-order
// sparse LU fills toward dense — while a fill-reducing reorder (RCM) recovers the band.
fn scattered_pentadiagonal(n: usize, seed: u64) -> CsrMatrix {
    // Fisher-Yates shuffle of 0..n with an LCG.
    let mut q: Vec<usize> = (0..n).collect();
    let mut s = seed;
    for i in (1..n).rev() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (s >> 11) as usize % (i + 1);
        q.swap(i, j);
    }
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    for i in 0..n {
        for off in [-2i64, -1, 0, 1, 2] {
            let j = i as i64 + off;
            if j >= 0 && (j as usize) < n {
                rows.push(q[i]);
                cols.push(q[j as usize]);
                data.push(if off == 0 { 6.0 } else { -1.0 });
            }
        }
    }
    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .unwrap()
        .to_csr()
        .unwrap()
}

fn opts_with(ordering: PermutationOrdering) -> SolveOptions {
    SolveOptions {
        ordering,
        ..SolveOptions::default()
    }
}

// Diagonally-dominant banded matrix, half-bandwidth `hb` (2·hb+1 nnz/row). For hb>8 this
// exceeds the old nnz<=16n gate and used to densify to O(n³); the bandwidth gate now
// routes it to the sparse LU (fill bounded by the band).
fn banded(n: usize, hb: usize) -> CsrMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    for i in 0..n {
        let lo = i.saturating_sub(hb);
        let hi = (i + hb).min(n - 1);
        for j in lo..=hi {
            rows.push(i);
            cols.push(j);
            data.push(if i == j { 2.0 * hb as f64 + 2.0 } else { -1.0 });
        }
    }
    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .unwrap()
        .to_csr()
        .unwrap()
}

// 2D 5-point Laplacian on a k×k grid (n=k²): the canonical fill-reduction benchmark.
// RCM keeps bandwidth ~k -> fill O(n·k)=O(n^1.5); minimum-degree/nested-dissection
// achieve O(n log n) fill. Diagonally dominant (diag 4+eps, neighbors -1) -> stable.
fn laplacian_2d(k: usize) -> CsrMatrix {
    let n = k * k;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    let idx = |r: usize, c: usize| r * k + c;
    for r in 0..k {
        for c in 0..k {
            let i = idx(r, c);
            rows.push(i);
            cols.push(i);
            data.push(4.001);
            for (dr, dc) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                let (nr, nc) = (r as i64 + dr, c as i64 + dc);
                if nr >= 0 && nr < k as i64 && nc >= 0 && nc < k as i64 {
                    rows.push(i);
                    cols.push(idx(nr as usize, nc as usize));
                    data.push(-1.0);
                }
            }
        }
    }
    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .unwrap()
        .to_csr()
        .unwrap()
}

// Rectangular counterpart used by the profile-first widening campaign. The
// insertion order matches the square fixture above and produces canonical CSR.
fn laplacian_2d_rectangular(rows_count: usize, cols_count: usize) -> CsrMatrix {
    let n = rows_count * cols_count;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    let idx = |row: usize, col: usize| row * cols_count + col;
    for row in 0..rows_count {
        for col in 0..cols_count {
            let i = idx(row, col);
            rows.push(i);
            cols.push(i);
            data.push(4.001);
            for (row_delta, col_delta) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                let neighbor_row = row as i64 + row_delta;
                let neighbor_col = col as i64 + col_delta;
                if neighbor_row >= 0
                    && neighbor_row < rows_count as i64
                    && neighbor_col >= 0
                    && neighbor_col < cols_count as i64
                {
                    rows.push(i);
                    cols.push(idx(neighbor_row as usize, neighbor_col as usize));
                    data.push(-1.0);
                }
            }
        }
    }
    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .unwrap()
        .to_csr()
        .unwrap()
}

// Arrowhead: diagonal + a dense hub row/col through node 0. nnz ~= 3n. Eliminating the
// hub early (natural/RCM, which can't isolate it) fills the whole trailing block O(n²);
// minimum-degree eliminates the degree-1 spokes first (no fill) and the hub last (no
// fill) -> O(n). The showcase where min-degree crushes bandwidth ordering.
fn arrowhead(n: usize) -> CsrMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    for i in 0..n {
        rows.push(i);
        cols.push(i);
        data.push(n as f64 + 4.0); // strong diagonal -> diagonally dominant, stable
        if i != 0 {
            rows.push(0);
            cols.push(i);
            data.push(-1.0);
            rows.push(i);
            cols.push(0);
            data.push(-1.0);
        }
    }
    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .unwrap()
        .to_csr()
        .unwrap()
}

// Diagonally-dominant pentadiagonal A (bandwidth 2): diag 6, off-diagonals -1 at
// ±1, ±2. nnz/row ~= 5, so a.nnz() <= 16n -> routes to the native sparse LU.
fn pentadiagonal(n: usize) -> CsrMatrix {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();
    for i in 0..n {
        for off in [-2i64, -1, 0, 1, 2] {
            let j = i as i64 + off;
            if j >= 0 && (j as usize) < n {
                rows.push(i);
                cols.push(j as usize);
                data.push(if off == 0 { 6.0 } else { -1.0 });
            }
        }
    }
    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .unwrap()
        .to_csr()
        .unwrap()
}

// Verbatim of the OLD dense path: densify the CSR and solve with nalgebra LU.
fn dense_solve_baseline(a: &CsrMatrix, b: &[f64]) -> Vec<f64> {
    let n = a.shape().rows;
    let mut dense = vec![0.0f64; n * n];
    let indptr = a.indptr();
    let indices = a.indices();
    let data = a.data();
    for i in 0..n {
        for idx in indptr[i]..indptr[i + 1] {
            dense[i * n + indices[idx]] = data[idx];
        }
    }
    let matrix = DMatrix::from_row_slice(n, n, &dense);
    let rhs = DVector::from_column_slice(b);
    let x = matrix.lu().solve(&rhs).expect("dense lu");
    x.iter().copied().collect()
}

fn time<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() * 1e3 / reps as f64
}

fn profile_rectangular_rust(repetitions: usize) {
    let rows = 32usize;
    let cols = 128usize;
    let n = rows * cols;
    let matrix = laplacian_2d_rectangular(rows, cols);
    let rhs: Vec<f64> = (0..n).map(|i| 1.0 + (i % 13) as f64 * 0.5).collect();
    let warm = spsolve(&matrix, &rhs, SolveOptions::default()).expect("rectangular warmup");
    let mut checksum = warm.solution.iter().sum::<f64>();
    let started = Instant::now();
    for _ in 0..repetitions {
        let solved = spsolve(black_box(&matrix), black_box(&rhs), SolveOptions::default())
            .expect("rectangular profile solve");
        checksum += black_box(solved.solution[n / 2]);
    }
    println!(
        "RECTANGULAR_PROFILE rows={rows} cols={cols} n={n} nnz={} repetitions={repetitions} elapsed_seconds={:.9} checksum={checksum:.17e}",
        matrix.nnz(),
        started.elapsed().as_secs_f64(),
    );
}

#[cfg(feature = "sparse-incumbent-bench")]
mod rectangular_live {
    use super::laplacian_2d_rectangular;
    use fsci_sparse::{
        CsrMatrix, SPSOLVE_RECTANGULAR_SPECTRAL_DISABLE, SPSOLVE_RECTANGULAR_SPECTRAL_HITS,
        SolveOptions, spsolve,
    };
    use sha2::{Digest, Sha256};
    use std::collections::{BTreeMap, HashSet};
    use std::hint::black_box;
    use std::io::{BufRead, BufReader, Write};
    use std::path::{Path, PathBuf};
    use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
    use std::sync::atomic::Ordering;
    use std::time::{Duration, Instant};

    const DIMENSIONS: [(usize, usize); 3] = [(32, 128), (48, 96), (64, 96)];
    const EXPECTED_COMPONENTS: usize = 14_848;
    const RESIDUAL_LIMIT: f64 = 1.0e-8;
    const L2_LIMIT: f64 = 1.0e-10;
    const MINIMUM_ROUNDS: usize = 21;
    const NULL_MEDIAN_LIMIT: f64 = 0.02;
    const HOST_BUSY_LIMIT: f64 = 0.20;
    const HOST_SAMPLE: Duration = Duration::from_secs(1);
    const LINALG_SOURCE_BYTES: &[u8] = include_bytes!("../linalg.rs");
    const HARNESS_SOURCE_BYTES: &[u8] = include_bytes!("perf_spsolve.rs");

    struct Fixture {
        rows: usize,
        cols: usize,
        matrix: CsrMatrix,
        rhs: Vec<f64>,
    }

    struct Scipy {
        child: Child,
        stdin: ChildStdin,
        stdout: BufReader<ChildStdout>,
        components: usize,
        maximum_threads: usize,
    }

    impl Scipy {
        fn start(script: &Path) -> Result<(Self, String), String> {
            let mut child = Command::new("python3")
                .arg("-u")
                .arg(script)
                .arg("--live")
                .arg("spsolve")
                .env("OPENBLAS_NUM_THREADS", "1")
                .env("OMP_NUM_THREADS", "1")
                .env("MKL_NUM_THREADS", "1")
                .env("BLIS_NUM_THREADS", "1")
                .env("VECLIB_MAXIMUM_THREADS", "1")
                .env("NUMEXPR_NUM_THREADS", "1")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .map_err(|error| format!("spawn live SciPy arm: {error}"))?;
            let stdin = child
                .stdin
                .take()
                .ok_or_else(|| "live SciPy arm has no stdin".to_string())?;
            let mut stdout = BufReader::new(
                child
                    .stdout
                    .take()
                    .ok_or_else(|| "live SciPy arm has no stdout".to_string())?,
            );
            let mut identity = String::new();
            stdout
                .read_line(&mut identity)
                .map_err(|error| format!("read live SciPy identity: {error}"))?;
            if identity.is_empty() {
                return Err("live SciPy arm exited before reporting identity".to_string());
            }
            Ok((
                Self {
                    child,
                    stdin,
                    stdout,
                    components: 0,
                    maximum_threads: 0,
                },
                identity.trim().to_string(),
            ))
        }

        fn read_reply(&mut self, context: &str) -> Result<String, String> {
            let mut reply = String::new();
            self.stdout
                .read_line(&mut reply)
                .map_err(|error| format!("read {context}: {error}"))?;
            if reply.is_empty() {
                return Err(format!("live SciPy arm exited while reading {context}"));
            }
            Ok(reply.trim().to_string())
        }

        fn initialize(&mut self, fixture: &Fixture) -> Result<(), String> {
            let n = fixture.rows * fixture.cols;
            writeln!(self.stdin, "INIT {n} {} 0.0 1", fixture.matrix.nnz())
                .map_err(|error| format!("write SciPy INIT: {error}"))?;
            write_usize_vector(&mut self.stdin, "INDPTR", fixture.matrix.indptr())?;
            write_usize_vector(&mut self.stdin, "INDICES", fixture.matrix.indices())?;
            write_f64_vector(&mut self.stdin, "DATA", fixture.matrix.data())?;
            write_f64_vector(&mut self.stdin, "B", &fixture.rhs)?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush SciPy INIT: {error}"))?;
            let reply = self.read_reply("SciPy CASE")?;
            if !reply.starts_with("CASE method=spsolve ")
                || !reply.contains(&format!("n={n} "))
                || !reply.contains(&format!("nnz={} ", fixture.matrix.nnz()))
                || !reply.contains("sorted=True ")
                || !reply.contains("canonical=True ")
                || !reply.contains("finite=True ")
                || !reply.ends_with("nonsymmetric=False")
            {
                return Err(format!("inadmissible SciPy fixture: {reply}"));
            }
            writeln!(self.stdin, "INPUT_SHA256")
                .map_err(|error| format!("write SciPy INPUT_SHA256: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush SciPy INPUT_SHA256: {error}"))?;
            let reported_hash = self.read_reply("SciPy input SHA-256")?;
            let expected_hash = fixture_input_sha256(fixture);
            if reported_hash != format!("INPUT_SHA256 {expected_hash}") {
                return Err(format!(
                    "SciPy input SHA-256 mismatch: expected {expected_hash}, received {reported_hash}"
                ));
            }
            self.components = n;
            Ok(())
        }

        fn parity(&mut self) -> Result<(Vec<f64>, f64), String> {
            writeln!(self.stdin, "PARITY")
                .map_err(|error| format!("write SciPy PARITY: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush SciPy PARITY: {error}"))?;
            let result = self.read_reply("SciPy parity result")?;
            if !result.starts_with("RESULT info=0 iterations=0 ") {
                return Err(format!("inadmissible SciPy parity result: {result}"));
            }
            let residual = result
                .split_whitespace()
                .find_map(|field| field.strip_prefix("residual="))
                .ok_or_else(|| "SciPy parity omitted residual".to_string())?
                .parse::<f64>()
                .map_err(|error| format!("parse SciPy parity residual: {error}"))?;
            if !result.contains(&format!("components={}", self.components)) {
                return Err(format!("SciPy parity component mismatch: {result}"));
            }
            let output = self.read_reply("SciPy parity solution")?;
            let payload = output
                .strip_prefix("X ")
                .ok_or_else(|| format!("invalid SciPy parity solution: {output}"))?;
            let solution = payload
                .split(',')
                .map(|value| {
                    value
                        .parse::<f64>()
                        .map_err(|error| format!("parse SciPy solution: {error}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if solution.len() != self.components || solution.iter().any(|value| !value.is_finite())
            {
                return Err("SciPy parity solution is incomplete or non-finite".to_string());
            }
            Ok((solution, residual))
        }

        fn time_one(&mut self) -> Result<(f64, f64), String> {
            writeln!(self.stdin, "SOLVE 1")
                .map_err(|error| format!("write SciPy SOLVE: {error}"))?;
            self.stdin
                .flush()
                .map_err(|error| format!("flush SciPy SOLVE: {error}"))?;
            let reply = self.read_reply("timed SciPy solve")?;
            let fields = reply.split_whitespace().collect::<Vec<_>>();
            if fields.len() != 6 || fields[0] != "TIME" {
                return Err(format!("invalid timed SciPy reply: {reply}"));
            }
            let elapsed = parse::<f64>(fields[1], "SciPy elapsed")?;
            let info = parse::<i32>(fields[2], "SciPy status")?;
            let components = parse::<usize>(fields[3], "SciPy components")?;
            let threads = parse::<usize>(fields[4], "SciPy observed threads")?;
            let checksum = parse::<f64>(fields[5], "SciPy checksum")?;
            self.maximum_threads = self.maximum_threads.max(threads);
            if info != 0
                || components != self.components
                || threads != 1
                || !elapsed.is_finite()
                || elapsed <= 0.0
                || !checksum.is_finite()
            {
                return Err(format!("inadmissible timed SciPy reply: {reply}"));
            }
            Ok((elapsed, checksum))
        }
    }

    impl Drop for Scipy {
        fn drop(&mut self) {
            let _ = writeln!(self.stdin, "QUIT");
            let _ = self.stdin.flush();
            let _ = self.child.wait();
        }
    }

    fn parse<T: std::str::FromStr>(value: &str, label: &str) -> Result<T, String>
    where
        T::Err: std::fmt::Display,
    {
        value
            .parse::<T>()
            .map_err(|error| format!("parse {label}: {error}"))
    }

    fn write_usize_vector(
        output: &mut ChildStdin,
        label: &str,
        values: &[usize],
    ) -> Result<(), String> {
        write!(output, "{label} ").map_err(|error| format!("write {label}: {error}"))?;
        for (index, value) in values.iter().enumerate() {
            if index != 0 {
                write!(output, ",").map_err(|error| format!("write {label}: {error}"))?;
            }
            write!(output, "{value}").map_err(|error| format!("write {label}: {error}"))?;
        }
        writeln!(output).map_err(|error| format!("write {label}: {error}"))
    }

    fn write_f64_vector(
        output: &mut ChildStdin,
        label: &str,
        values: &[f64],
    ) -> Result<(), String> {
        write!(output, "{label} ").map_err(|error| format!("write {label}: {error}"))?;
        for (index, value) in values.iter().enumerate() {
            if index != 0 {
                write!(output, ",").map_err(|error| format!("write {label}: {error}"))?;
            }
            write!(output, "{value:.17e}").map_err(|error| format!("write {label}: {error}"))?;
        }
        writeln!(output).map_err(|error| format!("write {label}: {error}"))
    }

    fn fixtures() -> Vec<Fixture> {
        DIMENSIONS
            .into_iter()
            .map(|(rows, cols)| {
                let n = rows * cols;
                Fixture {
                    rows,
                    cols,
                    matrix: laplacian_2d_rectangular(rows, cols),
                    rhs: (0..n).map(|i| 1.0 + (i % 13) as f64 * 0.5).collect(),
                }
            })
            .collect()
    }

    fn input_sha256(fixtures: &[Fixture]) -> String {
        let mut hasher = Sha256::new();
        for fixture in fixtures {
            hasher.update((fixture.rows as u64).to_le_bytes());
            hasher.update((fixture.cols as u64).to_le_bytes());
            hasher.update((fixture.matrix.nnz() as u64).to_le_bytes());
            for &value in fixture.matrix.data() {
                hasher.update(value.to_le_bytes());
            }
            for &index in fixture.matrix.indices() {
                hasher.update((index as u64).to_le_bytes());
            }
            for &pointer in fixture.matrix.indptr() {
                hasher.update((pointer as u64).to_le_bytes());
            }
            for &value in &fixture.rhs {
                hasher.update(value.to_le_bytes());
            }
        }
        format!("{:x}", hasher.finalize())
    }

    fn fixture_input_sha256(fixture: &Fixture) -> String {
        let mut hasher = Sha256::new();
        hasher.update(((fixture.rows * fixture.cols) as u64).to_le_bytes());
        hasher.update((fixture.matrix.nnz() as u64).to_le_bytes());
        for &value in fixture.matrix.data() {
            hasher.update(value.to_le_bytes());
        }
        for &index in fixture.matrix.indices() {
            hasher.update((index as u64).to_le_bytes());
        }
        for &pointer in fixture.matrix.indptr() {
            hasher.update((pointer as u64).to_le_bytes());
        }
        for &value in &fixture.rhs {
            hasher.update(value.to_le_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    fn sha256_file(path: &Path) -> Result<String, String> {
        let bytes = std::fs::read(path)
            .map_err(|error| format!("read {} for SHA-256: {error}", path.display()))?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    fn sha256_of_self() -> Result<String, String> {
        let executable =
            std::env::current_exe().map_err(|error| format!("current executable: {error}"))?;
        sha256_file(&executable)
    }

    fn relative_residual(fixture: &Fixture, solution: &[f64]) -> f64 {
        let mut residual_squared = 0.0;
        let mut rhs_squared = 0.0;
        for (row, &rhs) in fixture.rhs.iter().enumerate() {
            let mut product = 0.0;
            for index in fixture.matrix.indptr()[row]..fixture.matrix.indptr()[row + 1] {
                product += fixture.matrix.data()[index] * solution[fixture.matrix.indices()[index]];
            }
            residual_squared += (product - rhs).powi(2);
            rhs_squared += rhs * rhs;
        }
        residual_squared.sqrt() / rhs_squared.sqrt()
    }

    fn relative_l2(left: &[Vec<f64>], right: &[Vec<f64>]) -> f64 {
        let mut difference_squared = 0.0;
        let mut reference_squared = 0.0;
        for (left_solution, right_solution) in left.iter().zip(right) {
            for (&left_value, &right_value) in left_solution.iter().zip(right_solution) {
                difference_squared += (left_value - right_value).powi(2);
                reference_squared += right_value * right_value;
            }
        }
        difference_squared.sqrt() / reference_squared.sqrt()
    }

    fn rust_solutions(fixtures: &[Fixture], disable: bool) -> Result<Vec<Vec<f64>>, String> {
        SPSOLVE_RECTANGULAR_SPECTRAL_DISABLE.store(disable, Ordering::Relaxed);
        let result = fixtures
            .iter()
            .map(|fixture| {
                spsolve(&fixture.matrix, &fixture.rhs, SolveOptions::default())
                    .map(|result| result.solution)
                    .map_err(|error| format!("FrankenSciPy spsolve: {error}"))
            })
            .collect::<Result<Vec<_>, _>>();
        SPSOLVE_RECTANGULAR_SPECTRAL_DISABLE.store(false, Ordering::Relaxed);
        result
    }

    fn time_rust_job(fixtures: &[Fixture], disable: bool) -> Result<f64, String> {
        SPSOLVE_RECTANGULAR_SPECTRAL_DISABLE.store(disable, Ordering::Relaxed);
        let result = (|| {
            let started = Instant::now();
            let mut checksum = 0u64;
            for fixture in fixtures {
                let solution = spsolve(
                    black_box(&fixture.matrix),
                    black_box(&fixture.rhs),
                    SolveOptions::default(),
                )
                .map_err(|error| format!("timed FrankenSciPy spsolve: {error}"))?
                .solution;
                for value in solution {
                    checksum = checksum.rotate_left(1) ^ value.to_bits();
                }
            }
            let elapsed = started.elapsed().as_secs_f64();
            black_box(checksum);
            Ok(elapsed)
        })();
        SPSOLVE_RECTANGULAR_SPECTRAL_DISABLE.store(false, Ordering::Relaxed);
        result
    }

    fn time_scipy_job(oracles: &mut [Scipy]) -> Result<f64, String> {
        let mut elapsed = 0.0;
        let mut checksum = 0u64;
        for oracle in oracles {
            let (fixture_elapsed, fixture_checksum) = oracle.time_one()?;
            elapsed += fixture_elapsed;
            checksum = checksum.rotate_left(1) ^ fixture_checksum.to_bits();
        }
        black_box(checksum);
        Ok(elapsed)
    }

    fn median(mut values: Vec<f64>) -> f64 {
        values.sort_by(f64::total_cmp);
        if values.len().is_multiple_of(2) {
            0.5 * (values[values.len() / 2 - 1] + values[values.len() / 2])
        } else {
            values[values.len() / 2]
        }
    }

    fn percentile(mut values: Vec<f64>, quantile: f64) -> f64 {
        values.sort_by(f64::total_cmp);
        let index = ((values.len() - 1) as f64 * quantile).ceil() as usize;
        values[index.min(values.len() - 1)]
    }

    fn bootstrap_median_ci(values: &[f64]) -> (f64, f64) {
        let mut state = 0x6a09_e667_f3bc_c909u64;
        let mut medians = Vec::with_capacity(10_000);
        for _ in 0..10_000 {
            let mut sample = Vec::with_capacity(values.len());
            for _ in 0..values.len() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                sample.push(values[(state as usize) % values.len()]);
            }
            medians.push(median(sample));
        }
        medians.sort_by(f64::total_cmp);
        (medians[250], medians[9_750])
    }

    fn cv(values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / values.len().saturating_sub(1).max(1) as f64;
        variance.sqrt() / mean
    }

    fn ratios(numerator: &[f64], denominator: &[f64]) -> Vec<f64> {
        numerator
            .iter()
            .zip(denominator)
            .map(|(left, right)| left / right)
            .collect()
    }

    fn csv(values: &[f64]) -> String {
        values
            .iter()
            .map(|value| format!("{value:.9}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    #[derive(Clone, Copy)]
    enum Arm {
        Candidate,
        Control,
        Live,
    }

    #[derive(Clone, Copy)]
    enum Sample {
        Headline(Arm),
        NullLeft(Arm),
        NullRight(Arm),
    }

    #[derive(Default)]
    struct Measurement {
        candidate: Vec<f64>,
        control: Vec<f64>,
        live: Vec<f64>,
        candidate_null_left: Vec<f64>,
        candidate_null_right: Vec<f64>,
        control_null_left: Vec<f64>,
        control_null_right: Vec<f64>,
        live_null_left: Vec<f64>,
        live_null_right: Vec<f64>,
    }

    impl Measurement {
        fn push(&mut self, sample: Sample, elapsed: f64) {
            match sample {
                Sample::Headline(Arm::Candidate) => self.candidate.push(elapsed),
                Sample::Headline(Arm::Control) => self.control.push(elapsed),
                Sample::Headline(Arm::Live) => self.live.push(elapsed),
                Sample::NullLeft(Arm::Candidate) => self.candidate_null_left.push(elapsed),
                Sample::NullRight(Arm::Candidate) => self.candidate_null_right.push(elapsed),
                Sample::NullLeft(Arm::Control) => self.control_null_left.push(elapsed),
                Sample::NullRight(Arm::Control) => self.control_null_right.push(elapsed),
                Sample::NullLeft(Arm::Live) => self.live_null_left.push(elapsed),
                Sample::NullRight(Arm::Live) => self.live_null_right.push(elapsed),
            }
        }
    }

    fn time_arm(arm: Arm, fixtures: &[Fixture], oracles: &mut [Scipy]) -> Result<f64, String> {
        match arm {
            Arm::Candidate => time_rust_job(fixtures, false),
            Arm::Control => time_rust_job(fixtures, true),
            Arm::Live => time_scipy_job(oracles),
        }
    }

    fn measure(
        fixtures: &[Fixture],
        oracles: &mut [Scipy],
        rounds: usize,
    ) -> Result<Measurement, String> {
        const ORDER: [Sample; 9] = [
            Sample::Headline(Arm::Candidate),
            Sample::NullLeft(Arm::Control),
            Sample::NullRight(Arm::Live),
            Sample::Headline(Arm::Control),
            Sample::NullLeft(Arm::Live),
            Sample::NullRight(Arm::Candidate),
            Sample::Headline(Arm::Live),
            Sample::NullLeft(Arm::Candidate),
            Sample::NullRight(Arm::Control),
        ];
        let mut measurement = Measurement::default();
        for round in 0..rounds {
            for offset in 0..ORDER.len() {
                let sample = ORDER[(offset + round) % ORDER.len()];
                let arm = match sample {
                    Sample::Headline(arm) | Sample::NullLeft(arm) | Sample::NullRight(arm) => arm,
                };
                measurement.push(sample, time_arm(arm, fixtures, oracles)?);
            }
        }
        Ok(measurement)
    }

    fn print_distribution(label: &str, values: &[f64]) {
        println!(
            "{label}: p50_ms={:.6} p95_ms={:.6} p99_ms={:.6} cv_percent={:.3}",
            median(values.to_vec()) * 1e3,
            percentile(values.to_vec(), 0.95) * 1e3,
            percentile(values.to_vec(), 0.99) * 1e3,
            cv(values) * 100.0
        );
    }

    fn print_measurement(measurement: &Measurement) -> bool {
        let control_ratios = ratios(&measurement.control, &measurement.candidate);
        let live_ratios = ratios(&measurement.live, &measurement.candidate);
        let candidate_nulls = ratios(
            &measurement.candidate_null_left,
            &measurement.candidate_null_right,
        );
        let control_nulls = ratios(
            &measurement.control_null_left,
            &measurement.control_null_right,
        );
        let live_nulls = ratios(&measurement.live_null_left, &measurement.live_null_right);
        let (control_low, control_high) = bootstrap_median_ci(&control_ratios);
        let (live_low, live_high) = bootstrap_median_ci(&live_ratios);
        let (candidate_null_low, candidate_null_high) = bootstrap_median_ci(&candidate_nulls);
        let (control_null_low, control_null_high) = bootstrap_median_ci(&control_nulls);
        let (live_null_low, live_null_high) = bootstrap_median_ci(&live_nulls);

        print_distribution("candidate_whole_job", &measurement.candidate);
        print_distribution("same_elf_control_whole_job", &measurement.control);
        print_distribution("live_scipy_whole_job", &measurement.live);
        println!(
            "raw_samples_seconds: candidate={} control={} live={} \
             candidate_null_left={} candidate_null_right={} \
             control_null_left={} control_null_right={} \
             live_null_left={} live_null_right={}",
            csv(&measurement.candidate),
            csv(&measurement.control),
            csv(&measurement.live),
            csv(&measurement.candidate_null_left),
            csv(&measurement.candidate_null_right),
            csv(&measurement.control_null_left),
            csv(&measurement.control_null_right),
            csv(&measurement.live_null_left),
            csv(&measurement.live_null_right),
        );
        println!("control_over_candidate_ratios={}", csv(&control_ratios));
        println!("live_over_candidate_ratios={}", csv(&live_ratios));

        let candidate_null_median = median(candidate_nulls.clone());
        let control_null_median = median(control_nulls.clone());
        let live_null_median = median(live_nulls.clone());
        println!(
            "candidate_A/A: median={candidate_null_median:.6} \
             ci95=[{candidate_null_low:.6},{candidate_null_high:.6}] raw={}",
            csv(&candidate_nulls)
        );
        println!(
            "control_A/A: median={control_null_median:.6} \
             ci95=[{control_null_low:.6},{control_null_high:.6}] raw={}",
            csv(&control_nulls)
        );
        println!(
            "live_A/A: median={live_null_median:.6} \
             ci95=[{live_null_low:.6},{live_null_high:.6}] raw={}",
            csv(&live_nulls)
        );

        let widest_null_edge = candidate_null_high
            .max(control_null_high)
            .max(live_null_high)
            .max(1.0 / candidate_null_low.max(1.0e-12))
            .max(1.0 / control_null_low.max(1.0e-12))
            .max(1.0 / live_null_low.max(1.0e-12))
            .max(1.0);
        let twice_null_threshold = 1.0 + 2.0 * (widest_null_edge - 1.0);
        let null_medians_pass = [candidate_null_median, control_null_median, live_null_median]
            .into_iter()
            .all(|value| (value - 1.0).abs() <= NULL_MEDIAN_LIMIT);
        let maintenance_pass = control_low >= 1.20 && control_low > twice_null_threshold;
        let competitive_pass = live_low > 1.0 && live_low > twice_null_threshold;
        println!(
            "maintenance_ratio: control/candidate median={:.6} \
             bootstrap_median_ci95=[{control_low:.6},{control_high:.6}] \
             registered_minimum=1.200000 twice_widest_null_threshold={twice_null_threshold:.6}",
            median(control_ratios)
        );
        println!(
            "competitive_ratio: live_scipy/candidate median={:.6} \
             bootstrap_median_ci95=[{live_low:.6},{live_high:.6}] \
             null_corrected_threshold={twice_null_threshold:.6}",
            median(live_ratios)
        );
        println!(
            "decision_gate: null_medians_within_2pct={null_medians_pass} \
             maintenance_ci_low_at_least_1_20_and_beyond_2x_null={maintenance_pass} \
             competitive_ci_low_above_1_and_beyond_2x_null={competitive_pass} \
             cv_used_for_decision=false"
        );
        let keep = null_medians_pass && maintenance_pass;
        println!(
            "RECTANGULAR_SPSOLVE_DECISION={} competitive_claim={}",
            if keep { "KEEP" } else { "REVERT" },
            if competitive_pass { "PASS" } else { "FAIL" }
        );
        keep
    }

    #[derive(Clone, Copy)]
    struct CpuTicks {
        total: u64,
        idle: u64,
    }

    fn read_cpu_ticks() -> Result<BTreeMap<usize, CpuTicks>, String> {
        let stat = std::fs::read_to_string("/proc/stat")
            .map_err(|error| format!("read /proc/stat: {error}"))?;
        let mut cpus = BTreeMap::new();
        for line in stat.lines() {
            let mut fields = line.split_whitespace();
            let Some(name) = fields.next() else {
                continue;
            };
            let Some(cpu_text) = name.strip_prefix("cpu") else {
                continue;
            };
            if cpu_text.is_empty() || !cpu_text.bytes().all(|byte| byte.is_ascii_digit()) {
                continue;
            }
            let cpu = parse::<usize>(cpu_text, "CPU index")?;
            let ticks = fields
                .map(|field| parse::<u64>(field, "CPU tick"))
                .collect::<Result<Vec<_>, _>>()?;
            if ticks.len() < 5 {
                return Err(format!("CPU {cpu} has an incomplete /proc/stat row"));
            }
            cpus.insert(
                cpu,
                CpuTicks {
                    total: ticks.iter().sum(),
                    idle: ticks[3].saturating_add(ticks[4]),
                },
            );
        }
        if cpus.is_empty() {
            return Err("/proc/stat exposed no per-CPU rows".to_string());
        }
        Ok(cpus)
    }

    fn host_load_sample(label: &str, attempt: usize) -> Result<bool, String> {
        let before = read_cpu_ticks()?;
        std::thread::sleep(HOST_SAMPLE);
        let after = read_cpu_ticks()?;
        if before.len() != after.len() {
            return Err("CPU topology changed during host-wide load sample".to_string());
        }
        let mut maximum_busy = 0.0f64;
        let mut busy = Vec::new();
        for (cpu, first) in &before {
            let second = after
                .get(cpu)
                .ok_or_else(|| format!("CPU {cpu} disappeared during load sample"))?;
            let total = second.total.saturating_sub(first.total);
            let idle = second.idle.saturating_sub(first.idle);
            if total == 0 {
                return Err(format!("CPU {cpu} accumulated no ticks during load sample"));
            }
            let fraction = 1.0 - idle as f64 / total as f64;
            maximum_busy = maximum_busy.max(fraction);
            if fraction > HOST_BUSY_LIMIT {
                busy.push((*cpu, fraction));
            }
        }
        println!(
            "host_wide_load: phase={label} attempt={attempt} sampled_cpus={} \
             maximum_busy_fraction={maximum_busy:.3} busy_cpu_count={} limit={HOST_BUSY_LIMIT:.3}",
            before.len(),
            busy.len()
        );
        if !busy.is_empty() {
            println!(
                "host_wide_load_busy: phase={label} detail={}",
                busy.iter()
                    .map(|(cpu, fraction)| format!("{cpu}:{:.1}%", fraction * 100.0))
                    .collect::<Vec<_>>()
                    .join(",")
            );
        }
        Ok(busy.is_empty())
    }

    fn bounded_preflight() -> Result<(), String> {
        for attempt in 1..=12 {
            if host_load_sample("preflight", attempt)? {
                println!("host_wide_preflight=ADMITTED attempt={attempt} maximum_attempts=12");
                return Ok(());
            }
        }
        Err("host-wide preflight exhausted twelve one-second samples".to_string())
    }

    fn require_single_load_gate(label: &str) -> Result<(), String> {
        if host_load_sample(label, 1)? {
            Ok(())
        } else {
            Err(format!("single-shot host-wide {label} load gate failed"))
        }
    }

    fn cpu_affinity() -> Result<String, String> {
        let status = std::fs::read_to_string("/proc/self/status")
            .map_err(|error| format!("read /proc/self/status: {error}"))?;
        status
            .lines()
            .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
            .map(str::trim)
            .map(str::to_string)
            .ok_or_else(|| "Cpus_allowed_list missing from /proc/self/status".to_string())
    }

    fn affinity_cpus(affinity: &str) -> Result<HashSet<usize>, String> {
        let mut cpus = HashSet::new();
        for segment in affinity.split(',') {
            if let Some((start, end)) = segment.split_once('-') {
                let start = parse::<usize>(start, "affinity range start")?;
                let end = parse::<usize>(end, "affinity range end")?;
                if start > end {
                    return Err(format!("invalid CPU affinity range {segment}"));
                }
                cpus.extend(start..=end);
            } else {
                cpus.insert(parse::<usize>(segment, "affinity CPU")?);
            }
        }
        Ok(cpus)
    }

    fn observed_os_threads() -> Result<usize, String> {
        std::fs::read_dir("/proc/self/task")
            .map_err(|error| format!("read /proc/self/task: {error}"))
            .map(Iterator::count)
    }

    fn required_env(name: &str) -> Result<String, String> {
        std::env::var(name).map_err(|_| format!("required provenance variable {name} is absent"))
    }

    fn ready_value<'a>(identity: &'a str, prefix: &str) -> Option<&'a str> {
        identity
            .split_whitespace()
            .find_map(|field| field.strip_prefix(prefix))
    }

    fn is_sha256(value: &str) -> bool {
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
    }

    fn oracle_script(argument: Option<&String>) -> Result<PathBuf, String> {
        if let Some(argument) = argument {
            let path = PathBuf::from(argument);
            if path.is_file() {
                return Ok(path);
            }
            return Err(format!(
                "explicit SciPy oracle is unavailable: {}",
                path.display()
            ));
        }
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python/scipy_sparse_arm.py");
        if path.is_file() {
            Ok(path)
        } else {
            Err(format!("SciPy oracle is unavailable: {}", path.display()))
        }
    }

    fn print_hardware_provenance(cpu: usize) -> Result<(), String> {
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo")
            .map_err(|error| format!("read /proc/cpuinfo: {error}"))?;
        let model = cpuinfo
            .lines()
            .find_map(|line| line.strip_prefix("model name\t: "))
            .unwrap_or("unknown");
        let flags = cpuinfo
            .lines()
            .find_map(|line| line.strip_prefix("flags\t\t: "))
            .unwrap_or("");
        let meminfo = std::fs::read_to_string("/proc/meminfo")
            .map_err(|error| format!("read /proc/meminfo: {error}"))?;
        let memory_kib = meminfo
            .lines()
            .find_map(|line| line.strip_prefix("MemTotal:"))
            .and_then(|line| line.split_whitespace().next())
            .ok_or_else(|| "MemTotal missing from /proc/meminfo".to_string())?;
        let memory_bytes = parse::<u64>(memory_kib, "MemTotal KiB")?.saturating_mul(1024);
        let numa_nodes = std::fs::read_dir("/sys/devices/system/node")
            .map_err(|error| format!("read NUMA topology: {error}"))?
            .filter_map(Result::ok)
            .filter(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                name.strip_prefix("node")
                    .is_some_and(|suffix| suffix.bytes().all(|byte| byte.is_ascii_digit()))
            })
            .count();
        let frequency_base = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq");
        let read_frequency = |name: &str| {
            std::fs::read_to_string(format!("{frequency_base}/{name}"))
                .map(|value| value.trim().to_string())
                .unwrap_or_else(|_| "unavailable".to_string())
        };
        println!(
            "hardware_provenance: cpu_model={model:?} memory_bytes={memory_bytes} \
             numa_nodes={numa_nodes} avx2={} fma={} rust_observed_os_threads={}",
            flags.split_whitespace().any(|flag| flag == "avx2"),
            flags.split_whitespace().any(|flag| flag == "fma"),
            observed_os_threads()?
        );
        println!(
            "cpu_frequency_policy: cpu={cpu} scaling_driver={} scaling_governor={} \
             energy_performance_preference={} scaling_min_freq_khz={} scaling_max_freq_khz={}",
            read_frequency("scaling_driver"),
            read_frequency("scaling_governor"),
            read_frequency("energy_performance_preference"),
            read_frequency("scaling_min_freq"),
            read_frequency("scaling_max_freq"),
        );
        Ok(())
    }

    pub fn run(arguments: &[String]) -> Result<(), String> {
        let rounds = arguments
            .first()
            .map(|value| parse::<usize>(value, "rounds"))
            .transpose()?
            .unwrap_or(MINIMUM_ROUNDS);
        if rounds < MINIMUM_ROUNDS {
            return Err(format!(
                "rectangular live gate requires at least {MINIMUM_ROUNDS} rounds"
            ));
        }

        let elf_sha256 = sha256_of_self()?;
        let source_commit = required_env("BINARY_SOURCE_COMMIT")?;
        let builder_identity = required_env("BINARY_BUILDER_IDENTITY")?;
        let build_route = required_env("BINARY_BUILD_ROUTE")?;
        let booking_claim = required_env("TRJ_BOOKING_CLAIM_MESSAGE_ID")?;
        println!("elf_sha256={elf_sha256}");
        println!("frankenscipy_engine_sha256={elf_sha256}");
        println!(
            "binary_provenance: source_commit={source_commit} \
             builder_identity={builder_identity} build_route={build_route}"
        );
        println!("trj_booking_claim_message_id={booking_claim}");

        println!(
            "linalg_source_sha256={}",
            format!("{:x}", Sha256::digest(LINALG_SOURCE_BYTES))
        );
        println!(
            "harness_source_sha256={}",
            format!("{:x}", Sha256::digest(HARNESS_SOURCE_BYTES))
        );

        let affinity = cpu_affinity()?;
        let cpus = affinity_cpus(&affinity)?;
        if cpus.len() != 1 {
            return Err(format!(
                "all benchmark arms require one pinned physical CPU, observed affinity {affinity}"
            ));
        }
        let cpu = *cpus.iter().next().expect("one affinity CPU");
        if observed_os_threads()? != 1 {
            return Err("FrankenSciPy harness started with more than one OS thread".to_string());
        }
        println!(
            "thread_provenance: cpu_affinity={affinity} requested_frankenscipy_threads=1 \
             actual_observed_frankenscipy_threads=1 requested_scipy_threads=1"
        );
        print_hardware_provenance(cpu)?;
        bounded_preflight()?;

        let fixtures = fixtures();
        let total_components = fixtures
            .iter()
            .map(|fixture| fixture.rows * fixture.cols)
            .sum::<usize>();
        if total_components != EXPECTED_COMPONENTS {
            return Err(format!(
                "fixture components {total_components} != {EXPECTED_COMPONENTS}"
            ));
        }
        let shared_input_sha256 = input_sha256(&fixtures);
        println!(
            "fixture: dimensions=32x128,48x96,64x96 diagonal=4.001 \
             horizontal=-1 vertical=-1 rhs=1+0.5*(i_mod_13) matrices=3 \
             materialized_components={total_components} rounds={rounds}"
        );
        println!(
            "whole_job_boundary: INCLUDED=3_public_spsolve_calls,14848_materialized_outputs,\
             folded_output_checksum; EXCLUDED=matrix_rhs_construction,python_startup,\
             scipy_import,csr_transport,warmup,parity,provenance,bootstrap"
        );
        println!("shared_matrix_rhs_sha256={shared_input_sha256}");
        println!(
            "live_verified_fixture_sha256={}",
            fixtures
                .iter()
                .map(fixture_input_sha256)
                .collect::<Vec<_>>()
                .join(",")
        );

        SPSOLVE_RECTANGULAR_SPECTRAL_HITS.store(0, Ordering::Relaxed);
        let candidate = rust_solutions(&fixtures, false)?;
        let candidate_hits = SPSOLVE_RECTANGULAR_SPECTRAL_HITS.load(Ordering::Relaxed);
        SPSOLVE_RECTANGULAR_SPECTRAL_HITS.store(0, Ordering::Relaxed);
        let control = rust_solutions(&fixtures, true)?;
        let control_hits = SPSOLVE_RECTANGULAR_SPECTRAL_HITS.load(Ordering::Relaxed);
        if candidate_hits != 3 || control_hits != 0 {
            return Err(format!(
                "dispatch proof failed: candidate_hits={candidate_hits} control_hits={control_hits}"
            ));
        }
        let candidate_residual = fixtures
            .iter()
            .zip(&candidate)
            .map(|(fixture, solution)| relative_residual(fixture, solution))
            .fold(0.0f64, f64::max);
        let control_residual = fixtures
            .iter()
            .zip(&control)
            .map(|(fixture, solution)| relative_residual(fixture, solution))
            .fold(0.0f64, f64::max);
        let candidate_control_l2 = relative_l2(&candidate, &control);
        if candidate_residual > RESIDUAL_LIMIT
            || control_residual > RESIDUAL_LIMIT
            || candidate_control_l2 > L2_LIMIT
        {
            return Err(format!(
                "candidate/control conformance failed: candidate_residual={candidate_residual:.3e} \
                 control_residual={control_residual:.3e} relative_l2={candidate_control_l2:.3e}"
            ));
        }
        println!(
            "candidate_control_proof: candidate_hits={candidate_hits} control_hits={control_hits} \
             candidate_max_relative_residual={candidate_residual:.3e} \
             control_max_relative_residual={control_residual:.3e} \
             relative_l2={candidate_control_l2:.3e}"
        );

        let script = oracle_script(arguments.get(1))?;
        println!("scipy_oracle_script={}", script.display());
        println!("scipy_oracle_script_sha256={}", sha256_file(&script)?);
        let mut oracles = Vec::with_capacity(fixtures.len());
        let mut engine_sha256 = None;
        for (index, fixture) in fixtures.iter().enumerate() {
            let (mut oracle, identity) = Scipy::start(&script)?;
            println!("scipy_arm_{index}: {identity}");
            if !identity.starts_with("READY scipy=1.17.1 ")
                || !identity.contains("method=spsolve ")
                || !identity.contains("solver_mod=scipy.sparse.linalg._dsolve")
                || !identity.contains("actual_observed_worker_threads=1")
                || !identity.contains("fsci_loaded=False")
                || !identity.ends_with("genuine=True")
            {
                return Err(format!("live SciPy arm failed identity gate: {identity}"));
            }
            let reported_engine = ready_value(&identity, "scipy_engine_sha256=")
                .ok_or_else(|| "SciPy identity omitted engine SHA-256".to_string())?;
            if !is_sha256(reported_engine) {
                return Err("SciPy identity reported an invalid engine SHA-256".to_string());
            }
            if engine_sha256
                .as_deref()
                .is_some_and(|expected| expected != reported_engine)
            {
                return Err("SciPy oracle processes reported different engines".to_string());
            }
            engine_sha256 = Some(reported_engine.to_string());
            oracle.initialize(fixture)?;
            oracles.push(oracle);
        }
        println!(
            "scipy_engine_sha256={}",
            engine_sha256.expect("three SciPy engine identities")
        );

        let mut live = Vec::with_capacity(fixtures.len());
        let mut live_reported_residual = 0.0f64;
        for oracle in &mut oracles {
            let (solution, residual) = oracle.parity()?;
            live_reported_residual = live_reported_residual.max(residual);
            live.push(solution);
        }
        let live_recomputed_residual = fixtures
            .iter()
            .zip(&live)
            .map(|(fixture, solution)| relative_residual(fixture, solution))
            .fold(0.0f64, f64::max);
        let candidate_live_l2 = relative_l2(&candidate, &live);
        if live_reported_residual > RESIDUAL_LIMIT
            || live_recomputed_residual > RESIDUAL_LIMIT
            || candidate_live_l2 > L2_LIMIT
        {
            return Err(format!(
                "candidate/live conformance failed: reported_residual={live_reported_residual:.3e} \
                 recomputed_residual={live_recomputed_residual:.3e} relative_l2={candidate_live_l2:.3e}"
            ));
        }
        println!(
            "candidate_live_proof: genuine_scipy=1.17.1 input_sha_match=true \
             live_reported_max_relative_residual={live_reported_residual:.3e} \
             live_recomputed_max_relative_residual={live_recomputed_residual:.3e} \
             relative_l2={candidate_live_l2:.3e}"
        );

        black_box(time_rust_job(&fixtures, false)?);
        black_box(time_rust_job(&fixtures, true)?);
        black_box(time_scipy_job(&mut oracles)?);
        require_single_load_gate("measurement")?;
        let measurement = measure(&fixtures, &mut oracles, rounds)?;
        require_single_load_gate("post")?;
        if observed_os_threads()? != 1 || oracles.iter().any(|oracle| oracle.maximum_threads != 1) {
            return Err("observed worker count changed during measurement".to_string());
        }
        println!(
            "observed_workers: candidate=1 control=1 live_scipy=1 \
             matrix_rhs_sha256={shared_input_sha256}"
        );
        let _keep = print_measurement(&measurement);
        Ok(())
    }
}

fn main() {
    let raw_arguments = std::env::args().collect::<Vec<_>>();
    if raw_arguments.get(1).map(String::as_str) == Some("--rectangular-live") {
        #[cfg(feature = "sparse-incumbent-bench")]
        {
            if let Err(error) = rectangular_live::run(&raw_arguments[2..]) {
                eprintln!("RECTANGULAR_LIVE_FATAL {error}");
                std::process::exit(1);
            }
            return;
        }
        #[cfg(not(feature = "sparse-incumbent-bench"))]
        {
            eprintln!("--rectangular-live requires --features sparse-incumbent-bench");
            std::process::exit(2);
        }
    }

    let mut arguments = raw_arguments.into_iter().skip(1);
    if arguments.next().as_deref() == Some("--profile-rectangular-rust") {
        let repetitions = arguments
            .next()
            .map(|value| value.parse::<usize>().expect("positive repetition count"))
            .unwrap_or(50);
        assert!(repetitions > 0, "repetition count must be positive");
        profile_rectangular_rust(repetitions);
        return;
    }

    // Wider-banded routing: matrices with >16 nnz/row but a narrow band now route to the
    // sparse LU (bandwidth gate) instead of densifying to an O(n³) dense LU.
    println!("--- wider-banded routing: dense(old) vs sparse(bandwidth gate) ---");
    for &(n, hb) in &[(1024usize, 16usize), (2048, 24), (3000, 30)] {
        let a = banded(n, hb);
        let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 13) as f64 * 0.5).collect();
        let x_sparse = spsolve(&a, &b, SolveOptions::default())
            .expect("spsolve")
            .solution;
        let x_dense = dense_solve_baseline(&a, &b);
        let max_dx = x_sparse
            .iter()
            .zip(&x_dense)
            .map(|(s, d)| (s - d).abs())
            .fold(0.0_f64, f64::max);
        let reps_s = (20_000_000 / (n + 1)).clamp(10, 3000);
        let t_sparse = time(reps_s, || {
            black_box(spsolve(black_box(&a), black_box(&b), SolveOptions::default()).unwrap());
        });
        let reps_d = if n >= 2048 { 2 } else { 4 };
        let t_dense = time(reps_d, || {
            black_box(dense_solve_baseline(black_box(&a), black_box(&b)));
        });
        println!(
            "banded n={n:>5} hb={hb:>3} ({} nnz/row): dense={t_dense:>10.4}ms  sparse={t_sparse:>9.5}ms  speedup={:>9.1}x  max|dx|={max_dx:.2e}",
            2 * hb + 1,
            t_dense / t_sparse,
        );
    }

    println!("===PARITY+AB===");
    for &n in &[512usize, 1024, 2048] {
        let a = pentadiagonal(n);
        let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 13) as f64 * 0.5).collect();

        // correctness: sparse-routed result vs old dense result
        let x_sparse = spsolve(&a, &b, SolveOptions::default())
            .expect("spsolve")
            .solution;
        let x_dense = dense_solve_baseline(&a, &b);
        let max_dx = x_sparse
            .iter()
            .zip(x_dense.iter())
            .map(|(s, d)| (s - d).abs())
            .fold(0.0_f64, f64::max);

        let reps_sparse = (50_000_000 / (n + 1)).clamp(20, 5000);
        let t_after = time(reps_sparse, || {
            black_box(spsolve(black_box(&a), black_box(&b), SolveOptions::default()).unwrap());
        });
        let reps_dense = if n >= 2048 { 1 } else { 3 };
        let t_before = time(reps_dense, || {
            black_box(dense_solve_baseline(black_box(&a), black_box(&b)));
        });

        println!(
            "spsolve n={n:>5}: dense={t_before:>10.4}ms  sparse={t_after:>9.5}ms  speedup={:>9.1}x  max|dx|={max_dx:.2e}",
            t_before / t_after
        );

        // splu factorization: same routing. Time factorize-only (the dominant cost).
        let a_csc: CscMatrix = a.to_csc().unwrap();
        let fac = splu(&a_csc, LuOptions::default()).expect("splu");
        let x_splu = splu_solve(&fac, &b).expect("splu_solve");
        let max_dx2 = x_splu
            .iter()
            .zip(x_dense.iter())
            .map(|(s, d)| (s - d).abs())
            .fold(0.0_f64, f64::max);
        let t_splu = time(reps_sparse, || {
            black_box(splu(black_box(&a_csc), LuOptions::default()).unwrap());
        });
        let t_dense_fac = time(reps_dense, || {
            let n = a.shape().rows;
            let mut dense = vec![0.0f64; n * n];
            let indptr = a.indptr();
            let indices = a.indices();
            let data = a.data();
            for i in 0..n {
                for idx in indptr[i]..indptr[i + 1] {
                    dense[i * n + indices[idx]] = data[idx];
                }
            }
            black_box(DMatrix::from_row_slice(n, n, &dense).lu());
        });
        println!(
            "splu    n={n:>5}: dense={t_dense_fac:>10.4}ms  sparse={t_splu:>9.5}ms  speedup={:>9.1}x  max|dx|={max_dx2:.2e}",
            t_dense_fac / t_splu
        );
    }

    // ── NEW LEVER: fill-reducing ordering on a SCATTERED sparse matrix ──
    // natural-order sparse LU fills toward dense; RCM (default Colamd→RCM) recovers
    // the band. Both routes solve the SAME unique system (parity to rounding).
    println!("--- fill-reducing ordering (scattered pentadiagonal) ---");
    for &n in &[300usize, 600, 1000] {
        let a = scattered_pentadiagonal(n, 0x1234 ^ n as u64);
        let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 13) as f64 * 0.5).collect();

        let x_nat = spsolve(&a, &b, opts_with(PermutationOrdering::Natural))
            .expect("spsolve natural")
            .solution;
        let x_rcm = spsolve(&a, &b, opts_with(PermutationOrdering::Colamd))
            .expect("spsolve rcm")
            .solution;
        let max_dx = x_nat
            .iter()
            .zip(x_rcm.iter())
            .map(|(s, d)| (s - d).abs())
            .fold(0.0_f64, f64::max);

        let reps = (5_000_000 / (n + 1)).clamp(5, 2000);
        let t_nat = time(reps, || {
            black_box(
                spsolve(
                    black_box(&a),
                    black_box(&b),
                    opts_with(PermutationOrdering::Natural),
                )
                .unwrap(),
            );
        });
        let t_rcm = time(reps, || {
            black_box(
                spsolve(
                    black_box(&a),
                    black_box(&b),
                    opts_with(PermutationOrdering::Colamd),
                )
                .unwrap(),
            );
        });
        println!(
            "ordering n={n:>5}: natural={t_nat:>10.4}ms  rcm={t_rcm:>9.5}ms  speedup={:>9.1}x  max|dx|={max_dx:.2e}",
            t_nat / t_rcm
        );
    }

    // ── NEW LEVER: minimum-degree (MmdAtPlusA) vs RCM on a 2D Laplacian ──
    println!("--- minimum-degree vs RCM (2D 5-point Laplacian) ---");
    for &k in &[20usize, 32, 45, 64] {
        let a = laplacian_2d(k);
        let n = k * k;
        let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 13) as f64 * 0.5).collect();
        let x_rcm = spsolve(&a, &b, opts_with(PermutationOrdering::Colamd))
            .expect("rcm")
            .solution;
        let x_mmd = spsolve(&a, &b, opts_with(PermutationOrdering::MmdAtPlusA))
            .expect("mmd")
            .solution;
        let max_dx = x_rcm
            .iter()
            .zip(&x_mmd)
            .map(|(s, d)| (s - d).abs())
            .fold(0.0_f64, f64::max);
        let reps = (8_000_000 / (n + 1)).clamp(3, 2000);
        let t_rcm = time(reps, || {
            black_box(
                spsolve(
                    black_box(&a),
                    black_box(&b),
                    opts_with(PermutationOrdering::Colamd),
                )
                .unwrap(),
            );
        });
        let t_mmd = time(reps, || {
            black_box(
                spsolve(
                    black_box(&a),
                    black_box(&b),
                    opts_with(PermutationOrdering::MmdAtPlusA),
                )
                .unwrap(),
            );
        });
        println!(
            "lap2d k={k:>3} n={n:>5}: rcm={t_rcm:>10.4}ms  mmd={t_mmd:>9.5}ms  speedup={:>7.2}x  max|dx|={max_dx:.2e}",
            t_rcm / t_mmd
        );
    }

    // ── factor-once-solve-many: min-degree's smaller factor pays off per-solve ──
    println!("--- splu factor + 200 solves: RCM vs min-degree (2D Laplacian) ---");
    for &k in &[32usize, 45, 64] {
        let a = laplacian_2d(k);
        let a_csc: CscMatrix = a.to_csc().unwrap();
        let n = k * k;
        let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 13) as f64 * 0.5).collect();
        let m = 200usize;

        let lu_rcm = splu(
            &a_csc,
            LuOptions {
                ordering: PermutationOrdering::Colamd,
                ..LuOptions::default()
            },
        )
        .expect("rcm");
        let lu_mmd = splu(
            &a_csc,
            LuOptions {
                ordering: PermutationOrdering::MmdAtPlusA,
                ..LuOptions::default()
            },
        )
        .expect("mmd");
        let xr = splu_solve(&lu_rcm, &b).unwrap();
        let xm = splu_solve(&lu_mmd, &b).unwrap();
        let max_dx = xr
            .iter()
            .zip(&xm)
            .map(|(s, d)| (s - d).abs())
            .fold(0.0_f64, f64::max);

        let reps = 30;
        let t_rcm = time(reps, || {
            for _ in 0..m {
                black_box(splu_solve(black_box(&lu_rcm), black_box(&b)).unwrap());
            }
        });
        let t_mmd = time(reps, || {
            for _ in 0..m {
                black_box(splu_solve(black_box(&lu_mmd), black_box(&b)).unwrap());
            }
        });
        println!(
            "solve×{m} k={k:>3} n={n:>5}: rcm={t_rcm:>9.4}ms  mmd={t_mmd:>9.4}ms  per-solve speedup={:>6.2}x  max|dx|={max_dx:.2e}",
            t_rcm / t_mmd
        );
    }

    println!("--- minimum-degree ordering (arrowhead) ---");
    for &n in &[300usize, 600, 1000] {
        let a = arrowhead(n);
        let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 13) as f64 * 0.5).collect();

        let x_rcm = spsolve(&a, &b, opts_with(PermutationOrdering::Colamd))
            .expect("spsolve rcm")
            .solution;
        let x_mmd = spsolve(&a, &b, opts_with(PermutationOrdering::MmdAtPlusA))
            .expect("spsolve mmd")
            .solution;
        let max_dx = x_rcm
            .iter()
            .zip(x_mmd.iter())
            .map(|(s, d)| (s - d).abs())
            .fold(0.0_f64, f64::max);

        let reps = (5_000_000 / (n + 1)).clamp(5, 2000);
        let t_rcm = time(reps, || {
            black_box(
                spsolve(
                    black_box(&a),
                    black_box(&b),
                    opts_with(PermutationOrdering::Colamd),
                )
                .unwrap(),
            );
        });
        let t_mmd = time(reps, || {
            black_box(
                spsolve(
                    black_box(&a),
                    black_box(&b),
                    opts_with(PermutationOrdering::MmdAtPlusA),
                )
                .unwrap(),
            );
        });
        println!(
            "arrowhd n={n:>5}: rcm={t_rcm:>10.4}ms  mmd={t_mmd:>9.5}ms  speedup={:>9.1}x  max|dx|={max_dx:.2e}",
            t_rcm / t_mmd
        );
    }
}
