//! Live-SciPy `griddata` harness with A/A nulls, in ONE invocation.
//!
//! WHY THIS EXISTS. `perf_griddata_scipy.rs` times fsci only, has no SciPy arm in-process, has no
//! A/A null, and dumps its fixture to a hardcoded path in a scratchpad belonging to a session that
//! no longer exists. It cannot produce an admissible ratio. This can.
//!
//! ADMISSIBILITY IS ENFORCED, NOT ASSUMED. All three supported methods are parity-gated against
//! the persistent live SciPy arm before timing. `Cubic` is enabled only because
//! `CloughTocher2DInterpolator` now uses SciPy's global iterative curvature-minimising gradient
//! solve (frankenscipy-keeck); the same gate rejects any future semantic drift before a ratio can
//! be printed. `Linear` and `Nearest` are likewise measured only after this live comparison.
//!
//! PROTOCOL. One process. A persistent SciPy child holds the fixture; each round runs the two arms
//! in an interleaved A-B-B-A schedule so drift cancels rather than accumulating into one arm. Each
//! arm is ALSO sampled twice per round to give an A/A null: a self-ratio that must sit at 1.0. A
//! row whose null is off 1.0 is measuring the window, not the code.
//!
//! Usage: `perf_griddata_live [rounds] [npoints] [nqueries] [method]`
use fsci_interpolate::{GriddataMethod, griddata};
use fsci_runtime::scipy_incumbent::ScipyIncumbent;
use std::collections::BTreeSet;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

/// Submodules the oracle actually uses. A bare `import scipy` can succeed on an
/// installation whose compiled submodules do not load, and that difference would otherwise
/// only surface mid-run.
const SCIPY_REQUIRED_MODULES: &[&str] = &["scipy.interpolate"];

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

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    if v.is_empty() {
        return f64::NAN;
    }
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Percentile bootstrap over the RATIO samples. Deterministic LCG so a row is reproducible.
fn bootstrap_ci(samples: &[f64], iters: usize) -> (f64, f64) {
    if samples.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut medians = Vec::with_capacity(iters);
    for _ in 0..iters {
        let draw: Vec<f64> = (0..samples.len())
            .map(|_| samples[(lcg(&mut state) * samples.len() as f64) as usize % samples.len()])
            .collect();
        medians.push(median(draw));
    }
    medians.sort_by(f64::total_cmp);
    (medians[iters / 40], medians[iters - 1 - iters / 40])
}

fn coefficient_of_variation(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return f64::NAN;
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    if mean == 0.0 {
        return f64::NAN;
    }
    let variance = samples
        .iter()
        .map(|sample| (sample - mean).powi(2))
        .sum::<f64>()
        / samples.len() as f64;
    variance.sqrt() / mean.abs()
}

fn proc_stat_totals() -> Option<(u64, u64)> {
    let line = std::fs::read_to_string("/proc/stat")
        .ok()?
        .lines()
        .next()?
        .to_owned();
    let mut fields = line.split_whitespace();
    if fields.next()? != "cpu" {
        return None;
    }
    let values: Vec<u64> = fields.filter_map(|field| field.parse().ok()).collect();
    let total = values.iter().sum();
    let idle = values.get(3).copied().unwrap_or(0) + values.get(4).copied().unwrap_or(0);
    Some((total, idle))
}

/// This is an observation, not an admission gate: the A/A controls decide whether a busy host
/// contaminated this particular window.
fn host_mean_busy() -> Option<f64> {
    let (total_before, idle_before) = proc_stat_totals()?;
    std::thread::sleep(Duration::from_millis(100));
    let (total_after, idle_after) = proc_stat_totals()?;
    let total_delta = total_after.checked_sub(total_before)?;
    if total_delta == 0 {
        return None;
    }
    let idle_delta = idle_after.checked_sub(idle_before)?;
    Some(1.0 - idle_delta as f64 / total_delta as f64)
}

fn physical_cores() -> usize {
    let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") else {
        return std::thread::available_parallelism().map_or(0, usize::from);
    };
    let mut cores = BTreeSet::new();
    for processor in cpuinfo.split("\n\n") {
        let physical = processor
            .lines()
            .find_map(|line| {
                line.strip_prefix("physical id")
                    .and_then(|field| field.split_once(':'))
            })
            .and_then(|(_, value)| value.trim().parse::<u32>().ok());
        let core = processor
            .lines()
            .find_map(|line| {
                line.strip_prefix("core id")
                    .and_then(|field| field.split_once(':'))
            })
            .and_then(|(_, value)| value.trim().parse::<u32>().ok());
        if let (Some(physical), Some(core)) = (physical, core) {
            cores.insert((physical, core));
        }
    }
    if cores.is_empty() {
        std::thread::available_parallelism().map_or(0, usize::from)
    } else {
        cores.len()
    }
}

fn sysfs_count(prefix: &str, path: &str) -> usize {
    std::fs::read_dir(path)
        .ok()
        .map(|entries| {
            entries
                .flatten()
                .filter(|entry| {
                    entry.file_name().to_str().is_some_and(|name| {
                        name.strip_prefix(prefix).is_some_and(|suffix| {
                            !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
                        })
                    })
                })
                .count()
        })
        .unwrap_or(0)
}

fn mem_total_bytes() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|meminfo| {
            meminfo.lines().find_map(|line| {
                line.strip_prefix("MemTotal:")?
                    .split_whitespace()
                    .next()?
                    .parse::<u64>()
                    .ok()
                    .map(|kib| kib * 1024)
            })
        })
        .unwrap_or(0)
}

fn status_value(name: &str) -> Option<String> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| {
            line.strip_prefix(name)
                .and_then(|value| value.trim_start_matches(':').split_whitespace().next())
                .map(str::to_owned)
        })
}

fn task_count() -> usize {
    std::fs::read_dir("/proc/self/task")
        .map(|entries| entries.flatten().count())
        .unwrap_or(0)
}

fn runtime_isa() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            "avx2+fma"
        } else if std::is_x86_feature_detected!("sse4.2") {
            "sse4.2"
        } else {
            "x86_64-baseline"
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        "aarch64-neon"
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        "unknown"
    }
}

fn decision(nulls_ok: bool, lo: f64, hi: f64, null_edge: f64) -> &'static str {
    let margin = 2.0 * null_edge;
    if !nulls_ok {
        "VOID(null gate failed)"
    } else if lo > 1.0 + margin {
        "fsci FASTER"
    } else if hi < 1.0 - margin {
        "fsci SLOWER"
    } else {
        "INDISTINGUISHABLE(null margin)"
    }
}

struct ScipyArm {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

/// SHA-256 of the executable that is currently running this harness. This records the real
/// in-process candidate artifact without adding a production hashing dependency just for a bench.
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

impl ScipyArm {
    fn start(script: &str, fixture: &str) -> Self {
        let mut child = incumbent()
            .command()
            .arg(script)
            .arg(fixture)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("spawn scipy arm");
        let stdin = child.stdin.take().expect("stdin");
        let mut stdout = BufReader::new(child.stdout.take().expect("stdout"));
        let mut ready = String::new();
        stdout.read_line(&mut ready).expect("ready line");
        print!("scipy_arm: {ready}");
        assert!(
            ready.starts_with("READY"),
            "scipy arm did not come up: {ready}"
        );
        assert!(
            ready.contains("fsci_loaded=False"),
            "the SciPy arm must be an unpolluted interpreter"
        );
        Self {
            child,
            stdin,
            stdout,
        }
    }

    fn ask(&mut self, cmd: &str) -> String {
        writeln!(self.stdin, "{cmd}").expect("write");
        self.stdin.flush().expect("flush");
        let mut line = String::new();
        self.stdout.read_line(&mut line).expect("read");
        line
    }

    fn timed(&mut self, method: &str) -> f64 {
        let l = self.ask(&format!("run {method}"));
        let mut it = l.split_whitespace();
        assert_eq!(it.next(), Some("t"), "bad reply: {l}");
        it.next().unwrap().parse().unwrap()
    }

    fn values(&mut self, method: &str) -> Vec<f64> {
        let l = self.ask(&format!("solve {method}"));
        let mut it = l.split_whitespace();
        assert_eq!(it.next(), Some("v"), "bad reply");
        let n: usize = it.next().unwrap().parse().unwrap();
        let v: Vec<f64> = it
            .map(|s| {
                if s == "nan" {
                    f64::NAN
                } else {
                    s.parse().unwrap()
                }
            })
            .collect();
        assert_eq!(v.len(), n);
        v
    }

    fn quit(mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.child.wait();
    }
}

fn main() {
    let mut a = std::env::args().skip(1);
    let rounds: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(31);
    let np: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(2000);
    let nq: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(5000);
    let method_name = a.next().unwrap_or_else(|| "linear".to_string());

    let method = match method_name.as_str() {
        "linear" => GriddataMethod::Linear,
        "nearest" => GriddataMethod::Nearest,
        "cubic" => GriddataMethod::Cubic,
        other => {
            eprintln!("unknown method {other:?}");
            std::process::exit(2);
        }
    };

    // Fixture. Deterministic, and written somewhere that exists.
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let pts: Vec<Vec<f64>> = (0..np).map(|_| vec![lcg(&mut s), lcg(&mut s)]).collect();
    let vals: Vec<f64> = pts
        .iter()
        .map(|p| (p[0] * 6.2).sin() + (p[1] * 4.7).cos())
        .collect();
    let xi: Vec<Vec<f64>> = (0..nq).map(|_| vec![lcg(&mut s), lcg(&mut s)]).collect();

    let fixture = std::env::var("FSCI_GRIDDATA_FIXTURE").unwrap_or_else(|_| {
        std::env::temp_dir()
            .join("fsci_griddata_in.bin")
            .display()
            .to_string()
    });
    let mut buf = Vec::new();
    buf.extend_from_slice(&(np as u64).to_le_bytes());
    buf.extend_from_slice(&(nq as u64).to_le_bytes());
    for p in &pts {
        buf.extend_from_slice(&p[0].to_le_bytes());
        buf.extend_from_slice(&p[1].to_le_bytes());
    }
    for v in &vals {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for q in &xi {
        buf.extend_from_slice(&q[0].to_le_bytes());
        buf.extend_from_slice(&q[1].to_le_bytes());
    }
    std::fs::write(&fixture, &buf).expect("write fixture");
    println!(
        "fixture: path={fixture} npoints={np} nqueries={nq} method={method_name} rounds={rounds}"
    );
    println!("FSCI_KEECK_CUBIC_LIVE_V1");
    let candidate_engine_sha = elf_sha256();
    println!("elf_sha256={candidate_engine_sha}");
    println!("frankenscipy_engine_sha256={candidate_engine_sha}");
    println!(
        "host_identity={} physical_cores={} logical_threads={} ram_bytes={} numa_count={} \
         requested_threads=1 actual_observed_worker_threads={} runtime_isa={} affinity={} \
         scaling_governor={}",
        std::fs::read_to_string("/etc/hostname")
            .map(|hostname| hostname.trim().to_owned())
            .unwrap_or_else(|_| "unknown".to_string()),
        physical_cores(),
        sysfs_count("cpu", "/sys/devices/system/cpu"),
        mem_total_bytes(),
        sysfs_count("node", "/sys/devices/system/node"),
        task_count(),
        runtime_isa(),
        status_value("Cpus_allowed_list").unwrap_or_else(|| "unknown".to_string()),
        std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            .map(|governor| governor.trim().to_owned())
            .unwrap_or_else(|_| "unknown".to_string()),
    );

    let script = std::env::var("FSCI_GRIDDATA_ARM")
        .unwrap_or_else(|_| "crates/fsci-interpolate/python/griddata_live_arm.py".to_string());
    let mut arm = ScipyArm::start(&script, &fixture);

    // PARITY BEFORE TIMING. A faster arm that computed something else is not a result.
    let ours = griddata(&pts, &vals, &xi, method).expect("fsci griddata");
    let theirs = arm.values(&method_name);
    assert_eq!(ours.len(), theirs.len(), "length mismatch");
    let mut worst = 0.0_f64;
    let mut nan_mismatch = 0usize;
    for (a, b) in ours.iter().zip(theirs.iter()) {
        match (a.is_nan(), b.is_nan()) {
            (true, true) => {}
            (false, false) => worst = worst.max((a - b).abs()),
            _ => nan_mismatch += 1,
        }
    }
    println!(
        "parity: worst_abs_diff={worst:.3e} nan_mismatches={nan_mismatch} n={}",
        ours.len()
    );
    assert!(
        nan_mismatch == 0 && worst <= 2e-6,
        "fsci and live SciPy disagree; no timing is admissible"
    );

    let pre_busy = host_mean_busy();
    println!(
        "host-wide quiescence pre=NOT_CERTIFIED (host_mean_busy={:.6})",
        pre_busy.unwrap_or(f64::NAN)
    );

    // Interleaved quartet, two samples per arm per round for the A/A nulls.
    //
    // THE QUARTET IS FLIPPED EVERY ROUND, and it has to be. A fixed `f1 s1 s2 f2` puts fsci's two
    // null samples in the OUTER slots and SciPy's in the two ADJACENT middle slots, so `s1` always
    // pays the cold-cache cost of following an fsci run and `s2` never does. That is a POSITION
    // effect, not drift, and ABBA does not cancel it: measured over 61 rounds the SciPy null sat
    // at 1.073 with a bootstrap CI of [1.049, 1.100] that excludes 1.0 outright, while the fsci
    // null in the outer slots was clean at 0.984. Alternating `f s s f` with `s f f s` gives each
    // arm half its null samples from each configuration, so neither arm owns the cold slot.
    //
    // Each arm also gets an UNTIMED warmup call at the top of every round, which attacks the same
    // asymmetry at its source rather than only cancelling it in aggregate.
    let (mut fs, mut sc) = (Vec::new(), Vec::new());
    let (mut fnull, mut snull) = (Vec::new(), Vec::new());
    let timed_fsci = || {
        let t = Instant::now();
        std::hint::black_box(griddata(&pts, &vals, &xi, method).unwrap());
        t.elapsed().as_secs_f64()
    };
    for round in 0..rounds {
        timed_fsci();
        arm.timed(&method_name);

        let (f1, f2, s1, s2) = if round % 2 == 0 {
            let f1 = timed_fsci();
            let s1 = arm.timed(&method_name);
            let s2 = arm.timed(&method_name);
            let f2 = timed_fsci();
            (f1, f2, s1, s2)
        } else {
            let s1 = arm.timed(&method_name);
            let f1 = timed_fsci();
            let f2 = timed_fsci();
            let s2 = arm.timed(&method_name);
            (f1, f2, s1, s2)
        };

        fs.push(f1.min(f2));
        sc.push(s1.min(s2));
        fnull.push(f1 / f2);
        snull.push(s1 / s2);
    }
    let post_busy = host_mean_busy();
    println!(
        "host-wide quiescence post=NOT_CERTIFIED (host_mean_busy={:.6})",
        post_busy.unwrap_or(f64::NAN)
    );

    let fm = median(fs.clone());
    let sm = median(sc.clone());
    let ratios: Vec<f64> = fs.iter().zip(sc.iter()).map(|(f, s)| s / f).collect();
    let rm = median(ratios.clone());
    let (lo, hi) = bootstrap_ci(&ratios, 2000);
    let fnm = median(fnull.clone());
    let snm = median(snull.clone());
    let (flo, fhi) = bootstrap_ci(&fnull, 2000);
    let (slo, shi) = bootstrap_ci(&snull, 2000);

    println!(
        "fsci_median_ms={:.6} scipy_median_ms={:.6}",
        fm * 1e3,
        sm * 1e3
    );
    println!("fsci_A/A: median={fnm:.6} ci95=[{flo:.6},{fhi:.6}]");
    println!("scipy_A/A: median={snm:.6} ci95=[{slo:.6},{shi:.6}]");
    let nulls_ok = (fnm - 1.0).abs() <= 0.02
        && (snm - 1.0).abs() <= 0.02
        && flo <= 1.0
        && fhi >= 1.0
        && slo <= 1.0
        && shi >= 1.0;
    println!("null_gate: medians_within_2pct_and_ci_span_unity={nulls_ok}");
    println!("competitive_ratio: scipy/fsci median={rm:.6} ci95=[{lo:.6},{hi:.6}]");
    let null_edge = [
        (fnm - 1.0).abs(),
        (snm - 1.0).abs(),
        (flo - 1.0).abs(),
        (fhi - 1.0).abs(),
        (slo - 1.0).abs(),
        (shi - 1.0).abs(),
    ]
    .into_iter()
    .fold(0.0_f64, f64::max);
    println!(
        "decision_gate: 2x A/A-null margin={:.6} (null_edge={null_edge:.6})",
        2.0 * null_edge
    );
    println!(
        "ratio_cv={:.6}; CV is provenance only; decisions use bootstrap median CI plus the null margin",
        coefficient_of_variation(&ratios)
    );
    println!(
        "GRIDDATA_LIVE verdict={}",
        decision(nulls_ok, lo, hi, null_edge)
    );
    arm.quit();
}

#[cfg(test)]
mod tests {
    use super::decision;

    #[test]
    fn null_margin_rejects_a_naively_positive_ratio() {
        // A naive `lo > 1.0` decision would call this faster. The 1% null edge
        // demands a 2% separation before a timing claim is admissible.
        assert_eq!(
            decision(true, 1.011, 1.030, 0.010),
            "INDISTINGUISHABLE(null margin)"
        );
    }

    #[test]
    fn null_margin_allows_a_separated_loss() {
        assert_eq!(decision(true, 0.80, 0.95, 0.010), "fsci SLOWER");
    }
}
