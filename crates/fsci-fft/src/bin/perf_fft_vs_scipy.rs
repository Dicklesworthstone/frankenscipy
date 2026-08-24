//! Whole-job FFT/RFFT incumbent harness.
//!
//! This binary owns both timing arms in one invocation: FrankenSciPy is timed
//! in-process, while one long-lived Python child times live `scipy.fft`. For
//! every `(operation, size)` it executes interleaved A-B-B-A rounds, so each
//! side has its own A/A control and neither arm is measured in isolation.

use std::hint::black_box;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use fsci_fft::{Complex64, FftOptions, WorkerPolicy, fft, rfft};

const SIZES: &[usize] = &[
    1 << 16,
    1 << 17,
    1 << 18,
    1 << 19,
    1 << 20,
    1 << 21,
    1 << 22,
];
const ROUNDS: usize = 3;

const PYTHON_DRIVER: &str = r#"
import sys
import time
import numpy as np
import scipy
import scipy.fft

def real_input(n):
    phase = np.arange(n, dtype=np.float64) / float(n)
    return np.sin(2.0 * np.pi * phase) + 0.5 * np.cos(5.0 * np.pi * phase)

def complex_input(n):
    phase = np.arange(n, dtype=np.float64) / float(n)
    real = real_input(n)
    imag = np.cos(3.0 * np.pi * phase) - 0.25 * np.sin(7.0 * np.pi * phase)
    return real + 1j * imag

def checksum(values):
    real_bits = int(np.float64(np.sum(values.real, dtype=np.float64)).view(np.uint64))
    imag_bits = int(np.float64(np.sum(values.imag, dtype=np.float64)).view(np.uint64))
    return real_bits ^ ((imag_bits << 1) & 0xffffffffffffffff)

print(f"READY scipy={scipy.__version__} numpy={np.__version__}", flush=True)
for line in sys.stdin:
    fields = line.split()
    if not fields:
        continue
    if fields[0] == "quit":
        break
    mode, n, repeats = fields[0], int(fields[1]), int(fields[2])
    values = real_input(n) if mode == "rfft" else complex_input(n)
    transform = scipy.fft.rfft if mode == "rfft" else scipy.fft.fft
    for _ in range(2):
        transform(values)
    start = time.perf_counter_ns()
    digest = 0
    for _ in range(repeats):
        output = transform(values)
        digest = ((digest << 7) | (digest >> 57)) & 0xffffffffffffffff
        digest ^= checksum(output)
    elapsed = time.perf_counter_ns() - start
    print(f"{elapsed} {digest}", flush=True)
"#;

struct ScipyServer {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    version_line: String,
}

impl ScipyServer {
    fn start() -> Self {
        let mut child = Command::new("python3")
            .args(["-u", "-c", PYTHON_DRIVER])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("start live scipy.fft timing child");
        let stdin = child.stdin.take().expect("python stdin");
        let mut stdout = BufReader::new(child.stdout.take().expect("python stdout"));
        let mut version_line = String::new();
        stdout
            .read_line(&mut version_line)
            .expect("read live scipy readiness");
        assert!(
            version_line.starts_with("READY scipy="),
            "live scipy child did not report readiness: {version_line:?}"
        );
        Self {
            child,
            stdin,
            stdout,
            version_line: version_line.trim().to_owned(),
        }
    }

    fn time(&mut self, mode: &str, n: usize, repeats: usize) -> (f64, u64) {
        writeln!(self.stdin, "{mode} {n} {repeats}").expect("request scipy timing");
        self.stdin.flush().expect("flush scipy timing request");
        let mut response = String::new();
        self.stdout
            .read_line(&mut response)
            .expect("read scipy timing response");
        let mut fields = response.split_whitespace();
        let elapsed_ns: f64 = fields
            .next()
            .expect("scipy elapsed field")
            .parse()
            .expect("numeric scipy elapsed");
        let checksum: u64 = fields
            .next()
            .expect("scipy checksum field")
            .parse()
            .expect("numeric scipy checksum");
        assert!(
            fields.next().is_none(),
            "unexpected scipy response: {response:?}"
        );
        (elapsed_ns / repeats as f64 / 1e6, checksum)
    }
}

impl Drop for ScipyServer {
    fn drop(&mut self) {
        let _ = self.stdin.write_all(b"quit\n");
        let _ = self.stdin.flush();
        let _ = self.child.wait();
    }
}

fn real_input(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let phase = i as f64 / n as f64;
            (2.0 * std::f64::consts::PI * phase).sin()
                + 0.5 * (5.0 * std::f64::consts::PI * phase).cos()
        })
        .collect()
}

fn complex_input(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|i| {
            let phase = i as f64 / n as f64;
            let real = (2.0 * std::f64::consts::PI * phase).sin()
                + 0.5 * (5.0 * std::f64::consts::PI * phase).cos();
            let imag = (3.0 * std::f64::consts::PI * phase).cos()
                - 0.25 * (7.0 * std::f64::consts::PI * phase).sin();
            (real, imag)
        })
        .collect()
}

fn checksum(values: &[Complex64]) -> u64 {
    values.iter().fold(0_u64, |digest, &(real, imag)| {
        digest.rotate_left(7) ^ real.to_bits().rotate_left(13) ^ imag.to_bits().rotate_left(31)
    })
}

fn extend_digest(digest: u64, next: u64) -> u64 {
    digest.rotate_left(7) ^ next
}

fn repeats_for(n: usize) -> usize {
    match n {
        0..=262_144 => 6,
        262_145..=1_048_576 => 4,
        _ => 2,
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    assert!(!values.is_empty(), "median needs observations");
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn options_from_environment() -> FftOptions {
    let options = FftOptions::default();
    let Ok(value) = std::env::var("FSCI_FFT_WORKERS") else {
        return options;
    };
    let workers: usize = value
        .parse()
        .expect("FSCI_FFT_WORKERS must be a positive integer");
    assert!(workers > 0, "FSCI_FFT_WORKERS must be positive");
    options.with_workers(WorkerPolicy::Exact(workers))
}

fn executable_sha256() -> String {
    let executable = std::env::current_exe().expect("resolve timing executable");
    let output = Command::new("sha256sum")
        .arg(executable)
        .output()
        .expect("run sha256sum for timing executable");
    assert!(output.status.success(), "sha256sum timing executable");
    String::from_utf8(output.stdout)
        .expect("sha256sum output is UTF-8")
        .split_whitespace()
        .next()
        .expect("sha256sum digest")
        .to_owned()
}

fn time_fsci(
    mode: &str,
    real: &[f64],
    complex: &[Complex64],
    options: &FftOptions,
    repeats: usize,
) -> (f64, u64) {
    for _ in 0..2 {
        let values = if mode == "rfft" {
            rfft(black_box(real), black_box(options)).expect("warm rfft")
        } else {
            fft(black_box(complex), black_box(options)).expect("warm fft")
        };
        black_box(values);
    }

    let start = Instant::now();
    let mut digest = 0_u64;
    for _ in 0..repeats {
        let values = if mode == "rfft" {
            rfft(black_box(real), black_box(options)).expect("timed rfft")
        } else {
            fft(black_box(complex), black_box(options)).expect("timed fft")
        };
        assert!(
            values
                .iter()
                .all(|&(real, imag)| real.is_finite() && imag.is_finite()),
            "{mode} emitted a non-finite value"
        );
        digest = extend_digest(digest, checksum(&values));
        black_box(values);
    }
    (start.elapsed().as_secs_f64() * 1e3 / repeats as f64, digest)
}

fn main() {
    let mut scipy = ScipyServer::start();
    let options = options_from_environment();
    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    println!(
        "METADATA harness=fft-whole-job-aa-bbaa rounds={ROUNDS} rust_threads={threads} configured_workers={:?} elf_sha256={} {}",
        options.workers,
        executable_sha256(),
        scipy.version_line
    );

    for &n in SIZES {
        let real = real_input(n);
        let complex = complex_input(n);
        let repeats = repeats_for(n);
        for mode in ["rfft", "fft"] {
            let mut fsci_first = Vec::with_capacity(ROUNDS);
            let mut fsci_second = Vec::with_capacity(ROUNDS);
            let mut scipy_first = Vec::with_capacity(ROUNDS);
            let mut scipy_second = Vec::with_capacity(ROUNDS);
            let mut fsci_digest = 0_u64;
            let mut scipy_digest = 0_u64;

            for _ in 0..ROUNDS {
                let (first_fsci_ms, first_fsci_digest) =
                    time_fsci(mode, &real, &complex, &options, repeats);
                let (first_scipy_ms, first_scipy_digest) = scipy.time(mode, n, repeats);
                let (second_scipy_ms, second_scipy_digest) = scipy.time(mode, n, repeats);
                let (second_fsci_ms, second_fsci_digest) =
                    time_fsci(mode, &real, &complex, &options, repeats);
                fsci_first.push(first_fsci_ms);
                fsci_second.push(second_fsci_ms);
                scipy_first.push(first_scipy_ms);
                scipy_second.push(second_scipy_ms);
                fsci_digest = extend_digest(fsci_digest, first_fsci_digest);
                fsci_digest = extend_digest(fsci_digest, second_fsci_digest);
                scipy_digest = extend_digest(scipy_digest, first_scipy_digest);
                scipy_digest = extend_digest(scipy_digest, second_scipy_digest);
            }

            let fsci_first_median = median(fsci_first);
            let fsci_second_median = median(fsci_second);
            let scipy_first_median = median(scipy_first);
            let scipy_second_median = median(scipy_second);
            let fsci_ms = (fsci_first_median + fsci_second_median) / 2.0;
            let scipy_ms = (scipy_first_median + scipy_second_median) / 2.0;
            println!(
                "RESULT mode={mode} n={n} repeats={repeats} fsci_ms={fsci_ms:.6} scipy_ms={scipy_ms:.6} fsci_over_scipy={:.6} fsci_aa={:.6} scipy_aa={:.6} fsci_checksum={fsci_digest:016x} scipy_checksum={scipy_digest:016x}",
                fsci_ms / scipy_ms,
                fsci_first_median / fsci_second_median,
                scipy_first_median / scipy_second_median,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{checksum, complex_input, extend_digest, median, real_input, repeats_for};

    #[test]
    fn deterministic_inputs_and_digest_are_stable() {
        let real = real_input(16);
        let complex = complex_input(16);
        assert_eq!(real, real_input(16));
        assert_eq!(complex, complex_input(16));
        assert_eq!(checksum(&complex), checksum(&complex_input(16)));
        assert_ne!(
            extend_digest(1, 1),
            0,
            "digest extension must not XOR-cancel"
        );
    }

    #[test]
    fn size_schedule_covers_the_requested_curve_without_zero_repetitions() {
        assert_eq!(repeats_for(1 << 16), 6);
        assert_eq!(repeats_for(1 << 20), 4);
        assert_eq!(repeats_for(1 << 22), 2);
        assert_eq!(median(vec![3.0, 1.0, 2.0]), 2.0);
    }
}
