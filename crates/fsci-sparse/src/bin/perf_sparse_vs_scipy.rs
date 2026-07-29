//! Sparse iterative solvers vs a LIVE SciPy arm.
//!
//! The 2026-07-23 row claims "sparse CG fsci-faster, 2.27x/1.59x vs
//! `scipy.sparse.linalg.cg`" — but that was measured ACROSS invocations, not with the
//! incumbent running side-by-side. Under the 2026-07-27 policy that is not a campaign
//! win, and frankensearch has just demonstrated the cost of building on an unmeasured
//! assumption (Quill measured 8.7x SLOWER than Tantivy after ~90 commits on gates that
//! all read `unmeasured`). So: re-measure it properly before anything is built on it.
//!
//! Same contract as the ODE arm — persistent `python3 -u` co-process, both arms timed
//! inside one invocation, interleaved with alternating order, dual A/A nulls, ELF
//! sha256 self-reported, results compared before any timing is admitted.
//!
//! Trap 6 (asymmetric component) does NOT arise here, and that is worth stating: the
//! incumbent gets a native `scipy.sparse` CSR and its C-level SpMV, so there is no
//! per-iteration Python callback on either side. The ODE comparison had to decompose
//! one; this one has none to decompose.
//!
//! Run: `cargo run --release --bin perf_sparse_vs_scipy --features sparse-incumbent-bench -- [side] [rounds] [reps] [method]`

#[cfg(feature = "sparse-incumbent-bench")]
mod bench {
    use fsci_runtime::RuntimeMode;
    use fsci_sparse::linalg::{IterativeSolveOptions, bicgstab, cg, gmres};
    use fsci_sparse::{CooMatrix, CsrMatrix, FormatConvertible, Shape2D};
    use sha2::{Digest, Sha256};
    use std::hint::black_box;
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
    use std::time::Instant;

    const TOL: f64 = 1e-8;

    /// 2-D 5-point Laplacian in CSR — SPD, the standard CG test problem. Assembly must
    /// match `scipy_sparse_arm.py::laplacian` exactly or the arms solve different
    /// systems (trap 2).
    fn laplacian(side: usize) -> CsrMatrix {
        let n = side * side;
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(4.0);
            if i % side != 0 {
                rows.push(i);
                cols.push(i - 1);
                vals.push(-1.0);
            }
            if (i + 1) % side != 0 {
                rows.push(i);
                cols.push(i + 1);
                vals.push(-1.0);
            }
            if i >= side {
                rows.push(i);
                cols.push(i - side);
                vals.push(-1.0);
            }
            if i + side < n {
                rows.push(i);
                cols.push(i + side);
                vals.push(-1.0);
            }
        }
        CooMatrix::from_triplets(Shape2D::new(n, n), vals, rows, cols, true)
            .expect("laplacian coo")
            .to_csr()
            .expect("laplacian csr")
    }

    fn rhs(n: usize) -> Vec<f64> {
        (0..n).map(|i| 1.0 + 0.25 * ((i % 7) as f64)).collect()
    }

    struct Scipy {
        child: Child,
        stdin: ChildStdin,
        stdout: BufReader<ChildStdout>,
    }

    struct Run {
        secs: f64,
        iters: usize,
        converged: bool,
        resid: f64,
        xsum: f64,
        xfirst: f64,
    }

    impl Scipy {
        fn start(script: &str) -> Result<(Self, String), String> {
            let mut child = Command::new("python3")
                .arg("-u")
                .arg(script)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
                .map_err(|e| format!("spawn python3: {e}"))?;
            let stdin = child.stdin.take().ok_or("no stdin")?;
            let mut stdout = BufReader::new(child.stdout.take().ok_or("no stdout")?);
            let mut ready = String::new();
            stdout.read_line(&mut ready).map_err(|e| e.to_string())?;
            Ok((
                Self {
                    child,
                    stdin,
                    stdout,
                },
                ready.trim().to_string(),
            ))
        }

        fn solve(
            &mut self,
            side: usize,
            maxiter: usize,
            reps: usize,
            method: &str,
        ) -> Result<Run, String> {
            writeln!(self.stdin, "SOLVE {side} {TOL} {maxiter} {reps} {method}")
                .map_err(|e| e.to_string())?;
            self.stdin.flush().map_err(|e| e.to_string())?;
            let mut out = String::new();
            self.stdout.read_line(&mut out).map_err(|e| e.to_string())?;
            let f: Vec<&str> = out.split_whitespace().collect();
            if f.first() != Some(&"TIME") || f.len() < 7 {
                return Err(format!("bad reply: {}", out.trim()));
            }
            Ok(Run {
                secs: f[1].parse().map_err(|_| "secs")?,
                iters: f[2].parse().unwrap_or(0),
                converged: f[3] == "True",
                resid: f[4].parse().map_err(|_| "resid")?,
                xsum: f[5].parse().map_err(|_| "xsum")?,
                xfirst: f[6].parse().map_err(|_| "xfirst")?,
            })
        }

        fn quit(mut self) {
            let _ = writeln!(self.stdin, "QUIT");
            let _ = self.stdin.flush();
            let _ = self.child.wait();
        }
    }

    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(f64::total_cmp);
        if v.is_empty() {
            return f64::NAN;
        }
        if v.len() % 2 == 1 {
            v[v.len() / 2]
        } else {
            0.5 * (v[v.len() / 2 - 1] + v[v.len() / 2])
        }
    }

    /// Deterministic percentile-bootstrap CI on the median — the campaign gate.
    fn boot_ci(v: &[f64]) -> (f64, f64) {
        if v.is_empty() {
            return (f64::NAN, f64::NAN);
        }
        let mut st = 0x6a09_e667_f3bc_c909u64;
        let mut meds = Vec::with_capacity(10_000);
        for _ in 0..10_000 {
            let mut s = Vec::with_capacity(v.len());
            for _ in 0..v.len() {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                s.push(v[(st as usize) % v.len()]);
            }
            meds.push(median(s));
        }
        meds.sort_by(f64::total_cmp);
        (meds[250], meds[9_750])
    }

    fn sha256_of_self() -> String {
        let exe = std::env::current_exe().expect("current_exe");
        let bytes = std::fs::read(exe).expect("read own ELF");
        format!("{:x}", Sha256::digest(bytes))
    }

    pub fn run() {
        println!("elf_sha256={}", sha256_of_self());
        let args: Vec<String> = std::env::args().collect();
        let side: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(80);
        let rounds: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(9);
        let reps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
        let method = args.get(4).cloned().unwrap_or_else(|| "cg".to_string());
        let script = args
            .get(5)
            .cloned()
            .unwrap_or_else(|| "crates/fsci-sparse/python/scipy_sparse_arm.py".to_string());
        let n = side * side;
        let maxiter = 10 * n;
        println!(
            "method={method} fixture=laplacian2d side={side} n={n} rounds={rounds} \
             reps={reps} tol={TOL} maxiter={maxiter}"
        );

        let (mut sp, ready) = match Scipy::start(&script) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("ABORT: cannot start SciPy arm: {e}");
                std::process::exit(3);
            }
        };
        println!("scipy_arm: {ready}");
        if !ready.contains("genuine=True") {
            eprintln!("ABORT: SciPy arm is not genuine (dispatch trap)");
            std::process::exit(4);
        }

        let a = laplacian(side);
        let b = rhs(n);
        let opts = IterativeSolveOptions {
            mode: RuntimeMode::Strict,
            check_finite: true,
            tol: TOL,
            max_iter: Some(maxiter),
        };
        let solve_ours = |method: &str| {
            match method {
                "bicgstab" => bicgstab(&a, &b, None, opts),
                "gmres" => gmres(&a, &b, None, opts),
                _ => cg(&a, &b, None, opts),
            }
            .expect("fsci iterative solve")
        };

        // ── TRAP 2: same system, same answer, checked before any timing.
        let ours = solve_ours(&method);
        let theirs = match sp.solve(side, maxiter, 1, &method) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("ABORT: scipy solve failed: {e}");
                std::process::exit(5);
            }
        };
        let our_sum: f64 = ours.solution.iter().sum();
        let rel = |x: f64, y: f64| (x - y).abs() / y.abs().max(1e-300);
        println!(
            "agreement: xsum_ours={our_sum:.12e} xsum_scipy={:.12e} rel={:.3e} | \
             x0_ours={:.12e} x0_scipy={:.12e} rel={:.3e}",
            theirs.xsum,
            rel(our_sum, theirs.xsum),
            ours.solution[0],
            theirs.xfirst,
            rel(ours.solution[0], theirs.xfirst)
        );
        println!(
            "counters: ours iters={} converged={} resid={:.3e} | scipy iters={} converged={} resid={:.3e}",
            ours.iterations,
            ours.converged,
            ours.residual_norm,
            theirs.iters,
            theirs.converged,
            theirs.resid
        );
        // Both must actually converge, and to the same solution. A solver that bailed
        // early is fast for the wrong reason.
        if !ours.converged || !theirs.converged {
            eprintln!("ABORT: an arm did not converge — timing is not comparable");
            std::process::exit(6);
        }
        if rel(our_sum, theirs.xsum) > 1e-6 || rel(ours.solution[0], theirs.xfirst) > 1e-6 {
            eprintln!("ABORT: arms disagree beyond 1e-6 — not the same system");
            std::process::exit(7);
        }

        // ── TRAPS 3 + 4: interleave, alternate, and null BOTH arms.
        let (mut ratio, mut null_o, mut null_s) = (vec![], vec![], vec![]);
        let (mut to, mut ts) = (vec![], vec![]);
        for round in 0..rounds {
            let (a_secs, b_secs) = if round % 2 == 0 {
                let st = Instant::now();
                for _ in 0..reps {
                    black_box(solve_ours(&method));
                }
                let o = st.elapsed().as_secs_f64();
                let s = sp
                    .solve(side, maxiter, reps, &method)
                    .map(|r| r.secs)
                    .unwrap_or(f64::NAN);
                (o, s)
            } else {
                let s = sp
                    .solve(side, maxiter, reps, &method)
                    .map(|r| r.secs)
                    .unwrap_or(f64::NAN);
                let st = Instant::now();
                for _ in 0..reps {
                    black_box(solve_ours(&method));
                }
                (st.elapsed().as_secs_f64(), s)
            };
            to.push(a_secs);
            ts.push(b_secs);
            ratio.push(b_secs / a_secs);
            let st2 = Instant::now();
            for _ in 0..reps {
                black_box(solve_ours(&method));
            }
            null_o.push(st2.elapsed().as_secs_f64() / a_secs);
            let s2 = sp
                .solve(side, maxiter, reps, &method)
                .map(|r| r.secs)
                .unwrap_or(f64::NAN);
            null_s.push(s2 / b_secs);
        }

        let (rlo, rhi) = boot_ci(&ratio);
        let (olo, ohi) = boot_ci(&null_o);
        let (slo, shi) = boot_ci(&null_s);
        println!(
            "OURS p50={:.6}ms/solve  SCIPY p50={:.6}ms/solve",
            median(to) * 1e3 / reps as f64,
            median(ts) * 1e3 / reps as f64
        );
        println!(
            "NULL-ours  median={:.6} ci95=[{olo:.6},{ohi:.6}]",
            median(null_o.clone())
        );
        println!(
            "NULL-scipy median={:.6} ci95=[{slo:.6},{shi:.6}]",
            median(null_s.clone())
        );
        let edge = ohi
            .max(shi)
            .max(1.0 / olo.max(1e-9))
            .max(1.0 / slo.max(1e-9))
            .max(1.0);
        let required = 1.0 + 2.0 * (edge - 1.0);
        let p50 = median(ratio.clone());
        println!(
            "Incumbent ratio: SciPy / FrankenSciPy = {p50:.4}x (bootstrap-median ci95=[{rlo:.4},{rhi:.4}])"
        );
        println!(
            "median-CI gate: worst_null_edge={edge:.4} required={required:.4} => {}",
            if rlo > required {
                "DECIDED FRANKENSCIPY WIN"
            } else if rhi < 1.0 / required {
                "DECIDED FRANKENSCIPY LOSS"
            } else {
                "NOT DECIDED"
            }
        );
        sp.quit();
    }
}

#[cfg(feature = "sparse-incumbent-bench")]
fn main() {
    bench::run();
}

#[cfg(not(feature = "sparse-incumbent-bench"))]
fn main() {
    eprintln!("perf_sparse_vs_scipy requires --features sparse-incumbent-bench");
    std::process::exit(2);
}
