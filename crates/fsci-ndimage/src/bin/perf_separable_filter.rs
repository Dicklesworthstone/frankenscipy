//! Persistent live-SciPy whole-job benchmark for separable ndimage filters.

use std::hint::black_box;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use fsci_ndimage::{BoundaryMode, NdArray, gaussian_filter, gaussian_filter1d, uniform_filter};

const SCIPY_SITE_PACKAGES: &str =
    "/data/projects/.python-incumbents/frankenscipy-scipy-1.17.1/site-packages";
const PYTHON: &str = r#"
import hashlib, os, sys, time
import numpy as np
import scipy
from scipy import ndimage

shape = tuple(map(int, os.environ['FSCI_NDIMAGE_SHAPE'].split(',')))
op = os.environ['FSCI_NDIMAGE_OP']
nbytes = int(np.prod(shape)) * 8
raw = sys.stdin.buffer.read(nbytes)
if len(raw) != nbytes: raise RuntimeError('short fixture')
src = np.frombuffer(raw, dtype='<f8').reshape(shape).copy()
def run():
    if op == 'gaussian': return ndimage.gaussian_filter(src, sigma=2.0, mode='reflect', truncate=4.0)
    return ndimage.uniform_filter(src, size=9, mode='reflect')
ref = run()
try: blas = np.__config__.CONFIG['Build Dependencies']['blas']['name']
except Exception: blas = 'unknown'
print(f'READY scipy={scipy.__version__} numpy={np.__version__} blas={blas} fixture_sha256={hashlib.sha256(raw).hexdigest()} tasks={len(os.listdir("/proc/self/task"))} genuine={scipy.__version__ == "1.17.1"}', flush=True)
for line in sys.stdin.buffer:
    cmd = line.decode('ascii').strip().split()
    if cmd[0] == 'TIME':
        reps, minimum = map(int, cmd[1:])
        best = float('inf')
        for _ in range(minimum):
            t0 = time.perf_counter()
            for _ in range(reps): out = run()
            best = min(best, time.perf_counter() - t0)
        print(f'TIME {best:.17e}', flush=True)
    elif cmd[0] == 'CHECK':
        raw_ours = sys.stdin.buffer.read(nbytes)
        if len(raw_ours) != nbytes: raise RuntimeError('short Rust result')
        ours = np.frombuffer(raw_ours, dtype='<f8').reshape(shape)
        diff = np.abs(ours - ref)
        rel = diff / np.maximum(np.abs(ref), np.finfo(np.float64).tiny)
        print(f'CHECK max_abs={np.max(diff):.17e} max_rel={np.max(rel):.17e}', flush=True)
    else: raise RuntimeError(f'bad command {cmd}')
"#;

#[derive(Clone, Copy)]
enum Op {
    Gaussian,
    Uniform,
}

impl Op {
    fn name(self) -> &'static str {
        match self {
            Self::Gaussian => "gaussian",
            Self::Uniform => "uniform",
        }
    }
    fn run(self, input: &NdArray) -> NdArray {
        match self {
            Self::Gaussian => gaussian_filter(input, 2.0, BoundaryMode::Reflect, 0.0),
            Self::Uniform => uniform_filter(input, 9, BoundaryMode::Reflect, 0.0),
        }
        .expect("fixed valid fixture")
    }
}

struct Scipy {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Scipy {
    fn start(input: &NdArray, op: Op) -> Self {
        let shape = input
            .shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let mut child = Command::new("python3")
            .arg("-u")
            .arg("-c")
            .arg(PYTHON)
            .env("PYTHONPATH", SCIPY_SITE_PACKAGES)
            .env("FSCI_NDIMAGE_SHAPE", shape)
            .env("FSCI_NDIMAGE_OP", op.name())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("start SciPy");
        let mut stdin = child.stdin.take().expect("SciPy stdin");
        for &value in &input.data {
            stdin.write_all(&value.to_le_bytes()).expect("send fixture");
        }
        stdin.flush().expect("flush fixture");
        let stdout = BufReader::new(child.stdout.take().expect("SciPy stdout"));
        let mut this = Self {
            child,
            stdin,
            stdout,
        };
        let ready = this.line();
        assert!(
            ready.starts_with("READY ") && ready.contains("genuine=True"),
            "bad incumbent: {ready}"
        );
        println!("op={} {ready}", op.name());
        this
    }
    fn line(&mut self) -> String {
        let mut line = String::new();
        self.stdout.read_line(&mut line).expect("SciPy response");
        line.trim().to_owned()
    }
    fn time(&mut self, reps: usize, min_of: usize) -> f64 {
        writeln!(self.stdin, "TIME {reps} {min_of}").expect("TIME request");
        self.stdin.flush().expect("TIME flush");
        self.line()
            .strip_prefix("TIME ")
            .expect("TIME response")
            .parse()
            .expect("finite time")
    }
    fn check(&mut self, ours: &NdArray) -> String {
        writeln!(self.stdin, "CHECK").expect("CHECK request");
        for &value in &ours.data {
            self.stdin
                .write_all(&value.to_le_bytes())
                .expect("send result");
        }
        self.stdin.flush().expect("CHECK flush");
        let line = self.line();
        assert!(line.starts_with("CHECK "), "CHECK response: {line}");
        line
    }
}

impl Drop for Scipy {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn fixture(shape: &[usize]) -> NdArray {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let data = (0..shape.iter().product())
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5) + (index % 257) as f64 * 0.001
        })
        .collect();
    NdArray::new(data, shape.to_vec()).expect("fixture shape")
}

fn time_fsci(op: Op, input: &NdArray, reps: usize, min_of: usize) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..min_of {
        let start = Instant::now();
        for _ in 0..reps {
            let output = op.run(input);
            black_box(output.data[0]);
        }
        best = best.min(start.elapsed().as_secs_f64());
    }
    best
}

fn time_gaussian_axis(input: &NdArray, axis: usize, min_of: usize) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..min_of {
        let start = Instant::now();
        let output = gaussian_filter1d(input, 2.0, axis, 0, BoundaryMode::Reflect, 0.0)
            .expect("fixed valid fixture");
        black_box(output.data[0]);
        best = best.min(start.elapsed().as_secs_f64());
    }
    best
}

fn elf_sha256() -> String {
    let output = Command::new("sha256sum")
        .arg(std::env::current_exe().expect("executable"))
        .output()
        .expect("sha256sum");
    assert!(output.status.success(), "sha256sum failed");
    String::from_utf8(output.stdout)
        .expect("sha256 output")
        .split_whitespace()
        .next()
        .expect("digest")
        .to_owned()
}

fn main() {
    let cases: &[(&str, &[usize])] = &[("2d-4096", &[4096, 4096]), ("3d-256", &[256, 256, 256])];
    let reps = 1;
    let min_of = 2;
    println!("# live scipy.ndimage whole-job separable filters");
    println!("elf_sha256={}", elf_sha256());
    println!(
        "host={} affinity={} reps={reps} min_of={min_of}",
        std::fs::read_to_string("/proc/sys/kernel/hostname")
            .unwrap_or_default()
            .trim(),
        std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get)
    );
    if std::env::var_os("FSCI_NDIMAGE_PROFILE_3D_GAUSSIAN").is_some() {
        let input = fixture(&[256, 256, 256]);
        for axis in 0..input.ndim() {
            println!(
                "profile case=3d-256 op=gaussian axis={axis} fsci_pass={:.3}ms",
                time_gaussian_axis(&input, axis, min_of) * 1e3
            );
        }
        return;
    }
    for &(case, shape) in cases {
        let input = fixture(shape);
        for op in [Op::Gaussian, Op::Uniform] {
            let mut scipy = Scipy::start(&input, op);
            let parity = scipy.check(&op.run(&input));
            if case == "3d-256" && matches!(op, Op::Gaussian) {
                for axis in 0..input.ndim() {
                    println!(
                        "profile case={case} op=gaussian axis={axis} fsci_pass={:.3}ms",
                        time_gaussian_axis(&input, axis, min_of) * 1e3
                    );
                }
            }
            black_box(time_fsci(op, &input, 1, 1));
            let f_a = time_fsci(op, &input, reps, min_of);
            let s_a = scipy.time(reps, min_of);
            let s_b = scipy.time(reps, min_of);
            let f_b = time_fsci(op, &input, reps, min_of);
            let fsci = f_a.min(f_b);
            let incumbent = s_a.min(s_b);
            println!(
                "case={case} op={} fsci={:.3}ms scipy={:.3}ms scipy/fsci={:.3}x null_fsci={:.3} null_scipy={:.3} {parity}",
                op.name(),
                fsci * 1e3,
                incumbent * 1e3,
                incumbent / fsci,
                f_a.max(f_b) / f_a.min(f_b),
                s_a.max(s_b) / s_a.min(s_b)
            );
        }
    }
}
