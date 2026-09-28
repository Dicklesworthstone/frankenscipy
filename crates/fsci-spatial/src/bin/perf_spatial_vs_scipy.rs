//! Persistent live-SciPy whole-job benchmark for `fsci-spatial`.
//!
//! WHY THIS FILE EXISTS. `fsci-spatial` carries 37 perf binaries and none of them puts a live
//! SciPy arm next to ours; `fsci-cluster` and `fsci-interpolate` are in the same position.
//! This is the second harness written to close that gap, after `perf_signal_vs_scipy`.
//!
//! The two cases are chosen because SciPy is at its STRONGEST there, so a loss is the likely
//! outcome and therefore the informative one: `scipy.spatial.distance.pdist` is a hand-written
//! C loop, and `cKDTree.query` is C++ with its own node layout. Comparing against SciPy's
//! numpy-level code would mostly measure Python overhead.
//!
//! `delaunay` and `tsearch` (opt-in through `FSCI_SPATIAL_OPS`, 2-D) put Qhull itself on the
//! other side: the triangulation build, and batched point location in a prebuilt one.
//!
//! Gates on nothing a running measurement cannot supply for itself — twelve harnesses in this
//! tree abort on a booking claim that is unsatisfiable (bead fr78g), and copying that would
//! make this one unrunnable too. Agreement is CHECKED by shipping our result back to the
//! Python child rather than compared through two separately-computed digests.

use std::hint::black_box;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use fsci_runtime::scipy_incumbent::ScipyIncumbent;
use fsci_spatial::{Delaunay, DistanceMetric, KDTree, pdist, tsearch};

/// Submodules the oracle actually uses. A bare `import scipy` can succeed on an
/// installation whose compiled submodules do not load, and that difference would otherwise
/// only surface mid-run.
const SCIPY_REQUIRED_MODULES: &[&str] = &["scipy.spatial"];

/// The one live-SciPy incumbent this process compares against, resolved once and PROVEN by
/// running the import rather than by a name resolving on `PATH`.
///
/// This harness used to spawn a bare `python3`. On `thinkstation1` that is 3.14 with no
/// SciPy at all, so the oracle died on its first write with `BrokenPipe` and the run read as
/// a flaky pipe rather than as a missing incumbent (frankenscipy-m5s54). Resolving names the
/// interpreter, and prints the scipy AND numpy versions it proved, before anything is timed.
fn incumbent() -> &'static ScipyIncumbent {
    static INCUMBENT: std::sync::OnceLock<ScipyIncumbent> = std::sync::OnceLock::new();
    INCUMBENT.get_or_init(|| {
        let resolved = ScipyIncumbent::resolve_with(&[], SCIPY_REQUIRED_MODULES)
            .unwrap_or_else(|error| panic!("{error}"));
        println!("{}", resolved.provenance_line());
        resolved
    })
}

const PYTHON: &str = r#"
import hashlib, os, sys, time
import numpy as np
import scipy
from scipy.spatial import Delaunay, cKDTree
from scipy.spatial.distance import pdist

op = os.environ['FSCI_SPATIAL_OP']
n = int(os.environ['FSCI_SPATIAL_N'])
dim = int(os.environ['FSCI_SPATIAL_DIM'])
nq = int(os.environ['FSCI_SPATIAL_NQ'])

raw = sys.stdin.buffer.read(n * dim * 8)
if len(raw) != n * dim * 8: raise RuntimeError('short fixture')
pts = np.frombuffer(raw, dtype='<f8').reshape(n, dim).copy()
qraw = sys.stdin.buffer.read(nq * dim * 8)
if len(qraw) != nq * dim * 8: raise RuntimeError('short queries')
qry = np.frombuffer(qraw, dtype='<f8').reshape(nq, dim).copy()

tree = cKDTree(pts) if op == 'kdtree' else None
tri = Delaunay(pts) if op == 'tsearch' else None

def simplex_keys(s):
    # A triangle's identity independent of Qhull's order: its sorted vertices packed into one
    # float, exact while n**3 < 2**53.
    s = np.sort(s, axis=1).astype(np.float64)
    return (s[:, 0] * n + s[:, 1]) * n + s[:, 2]

def run():
    if op == 'pdist':
        return pdist(pts, metric='euclidean')
    if op == 'delaunay':
        return Delaunay(pts).simplices
    if op == 'tsearch':
        return tri.find_simplex(qry)
    d, _i = tree.query(qry, k=1)
    return d

def key(out):
    # Applied to the checked result only, never inside the timed loop.
    if op == 'delaunay':
        return np.sort(simplex_keys(out))
    if op == 'tsearch':
        return np.where(out >= 0, simplex_keys(tri.simplices)[out], -1.0)
    return out

ref = np.ascontiguousarray(key(run()), dtype='<f8')
print(f'READY scipy={scipy.__version__} numpy={np.__version__} op={op} n={n} dim={dim} nq={nq} '
      f'fixture_sha256={hashlib.sha256(raw).hexdigest()} '
      f'tasks={len(os.listdir("/proc/self/task"))} '
      f'genuine={scipy.__version__ == "1.17.1" and np.__version__ == "2.4.3"} out_len={ref.size}', flush=True)

for line in sys.stdin.buffer:
    cmd = line.decode('ascii').strip().split()
    if not cmd: continue
    if cmd[0] == 'TIME':
        reps, minimum = int(cmd[1]), int(cmd[2])
        best = float('inf')
        for _ in range(minimum):
            t0 = time.perf_counter()
            for _ in range(reps): out = run()
            best = min(best, time.perf_counter() - t0)
        print(f'TIME {best:.17e}', flush=True)
    elif cmd[0] == 'CHECK':
        m = ref.size
        raw_ours = sys.stdin.buffer.read(m * 8)
        if len(raw_ours) != m * 8: raise RuntimeError('short Rust result')
        ours = np.frombuffer(raw_ours, dtype='<f8')
        diff = np.abs(ours - ref)
        rel = diff / np.maximum(np.abs(ref), np.finfo(np.float64).tiny)
        print(f'CHECK max_abs={np.max(diff):.17e} max_rel={np.max(rel):.17e}', flush=True)
    elif cmd[0] == 'quit':
        break
    else:
        raise RuntimeError(f'bad command {cmd}')
"#;

struct Scipy {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    ready: String,
}

impl Scipy {
    fn start(op: &str, points: &[Vec<f64>], queries: &[Vec<f64>]) -> Self {
        let dim = points[0].len();
        let mut child = incumbent()
            .command()
            .args(["-u", "-c", PYTHON])
            .env("FSCI_SPATIAL_OP", op)
            .env("FSCI_SPATIAL_N", points.len().to_string())
            .env("FSCI_SPATIAL_DIM", dim.to_string())
            .env("FSCI_SPATIAL_NQ", queries.len().to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn live scipy.spatial child");
        let mut stdin = child.stdin.take().expect("python stdin");
        let mut bytes = Vec::new();
        for row in points.iter().chain(queries.iter()) {
            for value in row {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        stdin.write_all(&bytes).expect("send fixture");
        stdin.flush().expect("flush fixture");

        let mut stdout = BufReader::new(child.stdout.take().expect("python stdout"));
        let mut ready = String::new();
        stdout.read_line(&mut ready).expect("read readiness");
        assert!(ready.starts_with("READY scipy="), "not ready: {ready:?}");
        Self {
            child,
            stdin,
            stdout,
            ready: ready.trim().to_owned(),
        }
    }

    fn time(&mut self, reps: usize, minimum: usize) -> f64 {
        writeln!(self.stdin, "TIME {reps} {minimum}").expect("request timing");
        self.stdin.flush().expect("flush timing");
        let mut line = String::new();
        self.stdout.read_line(&mut line).expect("read timing");
        let value: f64 = line
            .trim()
            .strip_prefix("TIME ")
            .expect("TIME reply")
            .parse()
            .expect("numeric timing");
        value * 1.0e3 / reps as f64
    }

    fn check(&mut self, ours: &[f64]) -> String {
        writeln!(self.stdin, "CHECK").expect("request check");
        let mut bytes = Vec::with_capacity(ours.len() * 8);
        for value in ours {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        self.stdin.write_all(&bytes).expect("send result");
        self.stdin.flush().expect("flush result");
        let mut line = String::new();
        self.stdout.read_line(&mut line).expect("read check");
        line.trim().to_owned()
    }
}

impl Drop for Scipy {
    fn drop(&mut self) {
        let _ = self.stdin.write_all(b"quit\n");
        let _ = self.stdin.flush();
        let _ = self.child.wait();
    }
}

/// SHA-256 of this running executable, via `sha256sum` — `fsci-spatial` has no `sha2`, and
/// adding one so a benchmark can print a digest would be a production dependency taken on for
/// a measurement.
fn elf_sha256() -> String {
    let exe = std::env::current_exe().expect("current exe");
    let output = Command::new("sha256sum")
        .arg(exe)
        .output()
        .expect("run sha256sum");
    assert!(output.status.success(), "sha256sum failed");
    String::from_utf8(output.stdout)
        .expect("sha256sum output is UTF-8")
        .split_whitespace()
        .next()
        .expect("digest")
        .to_owned()
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// The Python side's `simplex_keys` for one triangle.
fn simplex_key(vertices: &[usize], n: usize) -> f64 {
    let mut v = [vertices[0], vertices[1], vertices[2]];
    v.sort_unstable();
    ((v[0] * n + v[1]) * n + v[2]) as f64
}

/// Uniform points in the unit square for the triangulation ops. The lattice fixture the
/// other ops use puts every 2-D point on the line y = x + c (mod 1), which is degenerate for
/// a triangulation.
fn uniform_square(count: usize, state: &mut u64) -> Vec<Vec<f64>> {
    let mut next = || {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*state >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..count).map(|_| vec![next(), next()]).collect()
}

fn main() {
    let n: usize = std::env::var("FSCI_SPATIAL_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);
    let dim: usize = std::env::var("FSCI_SPATIAL_DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let nq: usize = std::env::var("FSCI_SPATIAL_NQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);
    let rounds: usize = std::env::var("FSCI_SPATIAL_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    println!("elf_sha256={}", elf_sha256());

    // `FSCI_SPATIAL_PDIST_SOA=0` restores the per-pair `metric_distance` scan.
    // Only-override-when-asked: storing the parsed value unconditionally would overwrite a
    // newly flipped library default with `false` whenever the variable is unset.
    match std::env::var("FSCI_SPATIAL_PDIST_SOA").ok().as_deref() {
        Some("1") | Some("true") => {
            fsci_spatial::PDIST_SOA_EUCLIDEAN.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        Some("0") | Some("false") => {
            fsci_spatial::PDIST_SOA_EUCLIDEAN.store(false, std::sync::atomic::Ordering::Relaxed);
        }
        _ => {}
    }
    // `FSCI_SPATIAL_OPS=pdist` restricts the run to one op, so a `perf stat` total can be
    // attributed to it; `FSCI_SPATIAL_FIXED_REPS=N` pins repetitions so two arms being
    // compared on instructions retired do the SAME work (calibration derives reps from
    // measured time, so a faster arm would otherwise retire fewer for that reason alone).
    let selected = std::env::var("FSCI_SPATIAL_OPS").unwrap_or_else(|_| "pdist,kdtree".to_owned());
    let fixed_reps: Option<usize> = std::env::var("FSCI_SPATIAL_FIXED_REPS")
        .ok()
        .and_then(|v| v.parse().ok());
    println!(
        "pdist_soa_euclidean={} available_parallelism={}",
        fsci_spatial::PDIST_SOA_EUCLIDEAN.load(std::sync::atomic::Ordering::Relaxed),
        std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get),
    );

    // Deterministic and well spread, so the tree is balanced and no metric degenerates.
    let coord = |i: usize, d: usize| -> f64 {
        let k = (i * 2_654_435_761usize).wrapping_add(d * 40_503) % 100_003;
        k as f64 / 100_003.0 - 0.5
    };
    let points: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..dim).map(|d| coord(i, d)).collect())
        .collect();
    let queries: Vec<Vec<f64>> = (0..nq)
        .map(|i| (0..dim).map(|d| coord(i + n, d)).collect())
        .collect();

    // `delaunay` times the 2-D triangulation build and `tsearch` the batched point location on
    // a prebuilt one (both select with FSCI_SPATIAL_OPS and need FSCI_SPATIAL_DIM=2). Each is
    // checked on order-free simplex keys, because simplex numbering is Qhull's artifact.
    let mut lcg_state = 11_u64;
    let triangulation_points = uniform_square(n, &mut lcg_state);
    let triangulation_queries = uniform_square(nq, &mut lcg_state);

    for op in ["pdist", "kdtree", "delaunay", "tsearch"] {
        if !selected.split(',').any(|name| name.trim() == op) {
            continue;
        }
        let triangulates = matches!(op, "delaunay" | "tsearch");
        assert!(
            !triangulates || dim == 2,
            "op={op} needs FSCI_SPATIAL_DIM=2"
        );
        let (points, queries) = if triangulates {
            (&triangulation_points, &triangulation_queries)
        } else {
            (&points, &queries)
        };
        let mut scipy = Scipy::start(op, points, queries);
        println!("{}", scipy.ready);

        let tree = (op == "kdtree").then(|| KDTree::new(points).expect("build fsci KDTree"));
        let tri = (op == "tsearch").then(|| Delaunay::new(points).expect("build fsci Delaunay"));

        // The timed call, which returns what it computed so nothing can be optimised away.
        let run = || -> Vec<f64> {
            match op {
                "pdist" => pdist(points, DistanceMetric::Euclidean).expect("fsci pdist"),
                "kdtree" => {
                    let tree = tree.as_ref().expect("tree");
                    queries
                        .iter()
                        .map(|q| tree.query(q).expect("fsci query").1)
                        .collect()
                }
                "delaunay" => {
                    let built = Delaunay::new(points).expect("fsci Delaunay");
                    vec![built.simplices.len() as f64]
                }
                _ => {
                    let found = tsearch(tri.as_ref().expect("tri"), queries).expect("tsearch");
                    vec![found.len() as f64]
                }
            }
        };
        // The checked result, keyed like the Python side's `key`, outside any timed region.
        let ours = || -> Vec<f64> {
            match op {
                "delaunay" => {
                    let built = Delaunay::new(points).expect("fsci Delaunay");
                    let mut keys: Vec<f64> =
                        built.simplices.iter().map(|s| simplex_key(s, n)).collect();
                    keys.sort_by(f64::total_cmp);
                    keys
                }
                "tsearch" => {
                    let tri = tri.as_ref().expect("tri");
                    tsearch(tri, queries)
                        .expect("tsearch")
                        .iter()
                        .map(|&s| {
                            usize::try_from(s).map_or(-1.0, |s| simplex_key(&tri.simplices[s], n))
                        })
                        .collect()
                }
                _ => run(),
            }
        };

        black_box(run());
        let _ = scipy.time(1, 1);

        // `FSCI_SPATIAL_FIXED_REPS` repeats the call inside one sample. Default stays 1 so
        // existing rows keep their meaning; the knob exists so a `perf stat` comparison can
        // pin both arms to identical work.
        let reps = fixed_reps.unwrap_or(1).max(1);
        let time_ours = || -> f64 {
            let started = Instant::now();
            for _ in 0..reps {
                black_box(run());
            }
            started.elapsed().as_secs_f64() * 1.0e3 / reps as f64
        };

        // A-B-B-A per round so each arm carries its own A/A null across the same window.
        let (mut fsci, mut sp, mut null_f, mut null_s) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        // FLIP THE QUARTET EACH ROUND, because A-B-B-A alone balances DRIFT but not
        // POSITION. In `A B B A` our two samples are the outermost and are separated by both
        // SciPy samples, while SciPy's two are adjacent — so its A/A null measures a much
        // shorter interval than ours and is optimistic by construction. That asymmetry made
        // our null read 1.09-1.12 against SciPy's 1.003-1.012 in every row of this harness,
        // which looks like our arm being unstable when it is the schedule. Alternating to
        // `B A A B` on odd rounds gives each arm the inner and outer slots equally often, so
        // the two nulls become comparable and the medians mean the same thing.
        for round in 0..rounds {
            let (a1, s1, s2, a2) = if round % 2 == 0 {
                let a1 = time_ours();
                let s1 = scipy.time(reps, 1);
                let s2 = scipy.time(reps, 1);
                let a2 = time_ours();
                (a1, s1, s2, a2)
            } else {
                let s1 = scipy.time(reps, 1);
                let a1 = time_ours();
                let a2 = time_ours();
                let s2 = scipy.time(reps, 1);
                (a1, s1, s2, a2)
            };
            fsci.push(a1.min(a2));
            sp.push(s1.min(s2));
            null_f.push(a1.max(a2) / a1.min(a2));
            null_s.push(s1.max(s2) / s1.min(s2));
        }

        let (fsci_ms, scipy_ms) = (median(fsci), median(sp));
        // A triangulation with a different simplex count would desynchronise the CHECK
        // exchange, which reads exactly SciPy's length; report it instead.
        let checked = ours();
        let scipy_len = scipy
            .ready
            .split_whitespace()
            .find_map(|field| field.strip_prefix("out_len="))
            .and_then(|v| v.parse::<usize>().ok())
            .expect("READY line carries out_len");
        let check = if checked.len() == scipy_len {
            scipy.check(&checked)
        } else {
            format!(
                "CHECK length_mismatch fsci={} scipy={scipy_len}",
                checked.len()
            )
        };
        println!(
            "case=n{n}d{dim} op={op} fsci={fsci_ms:.3}ms scipy={scipy_ms:.3}ms \
             scipy/fsci={:.3}x null_fsci={:.3} null_scipy={:.3} {check}",
            scipy_ms / fsci_ms,
            median(null_f),
            median(null_s),
        );
    }
}
