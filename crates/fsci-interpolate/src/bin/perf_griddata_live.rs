//! Live-SciPy `griddata` harness with A/A nulls, in ONE invocation.
//!
//! WHY THIS EXISTS. `perf_griddata_scipy.rs` times fsci only, has no SciPy arm in-process, has no
//! A/A null, and dumps its fixture to a hardcoded path in a scratchpad belonging to a session that
//! no longer exists. It cannot produce an admissible ratio. This can.
//!
//! ADMISSIBILITY IS ENFORCED, NOT ASSUMED. `Cubic` is REFUSED: `CloughTocher2DInterpolator`
//! diverges from SciPy's by up to 1.8e-1 because SciPy estimates vertex gradients with a global
//! iterative curvature-minimising solve while this crate estimates them locally
//! (frankenscipy-keeck). A cubic ratio would compare two different interpolants and two different
//! amounts of work. `Linear` and `Nearest` are permitted because their parity against SciPy 1.17.1
//! is pinned by `minkowski_tsearch_...`-style tests -- specifically
//! `nd_interpolators_against_scipy_1_17_1`, which measured `LinearNDInterpolator` agreeing to
//! 2.220e-16.
//!
//! PROTOCOL. One process. A persistent SciPy child holds the fixture; each round runs the two arms
//! in an interleaved A-B-B-A schedule so drift cancels rather than accumulating into one arm. Each
//! arm is ALSO sampled twice per round to give an A/A null: a self-ratio that must sit at 1.0. A
//! row whose null is off 1.0 is measuring the window, not the code.
//!
//! Usage: `perf_griddata_live [rounds] [npoints] [nqueries] [method]`
use fsci_interpolate::{GriddataMethod, griddata};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

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

struct ScipyArm {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl ScipyArm {
    fn start(script: &str, fixture: &str) -> Self {
        let mut child = Command::new("python3")
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
        "cubic" => {
            eprintln!(
                "REFUSED: cubic. CloughTocher2DInterpolator diverges from SciPy's by up to 1.8e-1 \
                 (frankenscipy-keeck): SciPy estimates vertex gradients with a global iterative \
                 solve, this crate estimates them locally. A cubic ratio would compare two \
                 different interpolants doing different amounts of work, which is not a \
                 measurement. Fix the parity first, then time it."
            );
            std::process::exit(2);
        }
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
        nan_mismatch == 0 && worst <= 1e-9,
        "fsci and live SciPy disagree; no timing is admissible"
    );

    // Interleaved A-B-B-A, two samples per arm per round for the A/A nulls.
    let (mut fs, mut sc) = (Vec::new(), Vec::new());
    let (mut fnull, mut snull) = (Vec::new(), Vec::new());
    for _ in 0..rounds {
        let t = Instant::now();
        std::hint::black_box(griddata(&pts, &vals, &xi, method).unwrap());
        let f1 = t.elapsed().as_secs_f64();
        let s1 = arm.timed(&method_name);
        let s2 = arm.timed(&method_name);
        let t = Instant::now();
        std::hint::black_box(griddata(&pts, &vals, &xi, method).unwrap());
        let f2 = t.elapsed().as_secs_f64();

        fs.push(f1.min(f2));
        sc.push(s1.min(s2));
        fnull.push(f1 / f2);
        snull.push(s1 / s2);
    }

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
    println!(
        "GRIDDATA_LIVE verdict={}",
        if !nulls_ok {
            "VOID(null gate failed)"
        } else if lo > 1.0 {
            "fsci FASTER"
        } else if hi < 1.0 {
            "fsci SLOWER"
        } else {
            "INDISTINGUISHABLE"
        }
    );
    arm.quit();
}
