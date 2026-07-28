//! ODE head-to-head against a LIVE SciPy arm.
//!
//! Campaign policy 2026-07-27: a self-speedup is maintenance; a campaign win needs a
//! measured ratio against the actual legacy incumbent, from a harness that runs the
//! incumbent side-by-side IN THE SAME INVOCATION. This binary measures whether the
//! structural BDF/Radau/LSODA and explicit-RK claims actually translate against
//! SciPy itself.
//!
//! SciPy runs in a persistent `python3 -u` co-process (`python/scipy_bdf_arm.py`).
//! Each arm times ITSELF — SciPy with `perf_counter` around its `solve_ivp` loop, we
//! with `Instant` — so the pipe round-trip is outside both measured regions.
//!
//! THE SIX TRAPS, each of which has already burned this fleet:
//!  1. DISPATCH — the Python side asserts genuine SciPy and that no `fsci`/`franken`
//!     module is loaded in that interpreter; this binary aborts unless it says so.
//!  2. UNMATCHED CONFIG — identical fixture, `t_span`, `y0`, `rtol`, `atol`, method;
//!     and the arms' RESULTS are compared before any timing is admitted.
//!  3. NON-INTERLEAVED ARMS — both arms run inside one round, order alternating.
//!  4. CORE CONTENTION — an A/A null is run for BOTH arms, not just ours, so
//!     asymmetric degradation is visible instead of being banked as a win.
//!  5. CLIENT-BOUND — timing is taken inside each arm; pipe I/O and result parsing are
//!     outside both measured regions.
//!  6. SHARED/ASYMMETRIC COMPONENT — SciPy's RHS is a Python callback, ours an inlined
//!     Rust closure, and a stiff solve makes thousands of calls. The callback cost is
//!     measured on its own and the ratio is decomposed, not banked.
//!
//! Run: `cargo run --release --bin perf_bdf_vs_scipy --features bdf-diag-bench -- [n] [rounds] [reps] [fixture] [method]`

#[cfg(feature = "bdf-diag-bench")]
mod bench {
    use fsci_integrate::bdf::{BDF_BAND_NEWTON_HITS, BDF_DIAG_NEWTON_HITS, BDF_FORCE_DENSE_NEWTON};
    use fsci_integrate::radau::RADAU_DIAG_NEWTON_HITS;
    use fsci_integrate::{SolveIvpOptions, SolveIvpResult, SolverKind, ToleranceValue, solve_ivp};
    use fsci_runtime::RuntimeMode;
    use sha2::{Digest, Sha256};
    use std::hint::black_box;
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
    use std::sync::atomic::Ordering;
    use std::time::Instant;

    const BDF_T_END: f64 = 1.0;
    const BDF_RTOL: f64 = 1e-8;
    const BDF_ATOL: f64 = 1e-10;

    /// Which problem the head-to-head solves. `Diagonal` is the exact fixture behind
    /// the self-speedup claim: `y'_i = -(1 + 10i) y_i`, decoupled, so `I - c*J` is
    /// exactly diagonal and our structural fast path fires.
    ///
    /// `Coupled` is the FALSIFYING EXPERIMENT that both ledger rows name as required
    /// before the diagonal number is quoted broadly. It adds nearest-neighbour terms
    /// — `y'_i = -(1 + 10i) y_i + 0.5(y_{i-1} - 2y_i + y_{i+1})` — so the Jacobian is
    /// TRIDIAGONAL, not diagonal. That single change decides an open question the
    /// diagonal row cannot answer on its own: how much of the measured ratio is the
    /// STRUCTURE (a fast path SciPy has no equivalent of) versus the IMPLEMENTATION
    /// (a compiled step loop against SciPy's Python one). If the ratio collapses
    /// toward the ~2-3x dense-linalg wall, the win is structural and bounded; if it
    /// stays large, the structural attribution in the diagonal row is WRONG even
    /// though its number is right, and that is a refutation worth landing.
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum Fixture {
        /// Exact historical scalar RK45 workload: `y' = -y`, `y(0)=1`,
        /// `t_span=[0,10]`, `rtol=1e-6`, `atol=1e-9`.
        Exponential,
        /// Exact historical three-component RK45 workload: the Lorenz system
        /// from `[1,1,1]` over `t_span=[0,1]`, `rtol=1e-6`, `atol=1e-9`.
        Lorenz,
        Diagonal,
        Coupled,
        Dense,
        /// Exact `radau-stiff64` workload behind the shipped diagonal-stage
        /// self-speedup: rates span 1..1000, y0 is all ones, t_end=0.2,
        /// rtol=1e-6, atol=1e-8.
        RadauStiff,
    }

    impl Fixture {
        fn parse(s: &str) -> Option<Self> {
            match s {
                "exponential" | "exp" => Some(Self::Exponential),
                "lorenz" => Some(Self::Lorenz),
                "diagonal" | "diag" => Some(Self::Diagonal),
                "coupled" | "tri" => Some(Self::Coupled),
                "dense" | "full" => Some(Self::Dense),
                "radau-stiff" | "radau64" => Some(Self::RadauStiff),
                _ => None,
            }
        }
        fn label(self) -> &'static str {
            match self {
                Self::Exponential => "rk-exponential-decay",
                Self::Lorenz => "rk-lorenz",
                Self::Diagonal => "exact-diagonal",
                Self::Coupled => "coupled-tridiagonal",
                Self::Dense => "dense-allpairs",
                Self::RadauStiff => "radau-stiff64-exact-diagonal",
            }
        }
        fn wire(self) -> &'static str {
            match self {
                Self::Exponential => "exponential",
                Self::Lorenz => "lorenz",
                Self::Diagonal => "diagonal",
                Self::Coupled => "coupled",
                Self::Dense => "dense",
                Self::RadauStiff => "radau-stiff",
            }
        }

        fn t_end(self) -> f64 {
            match self {
                Self::Exponential => 10.0,
                Self::Lorenz => 1.0,
                Self::RadauStiff => 0.2,
                _ => BDF_T_END,
            }
        }

        fn rtol(self) -> f64 {
            match self {
                Self::Exponential | Self::Lorenz | Self::RadauStiff => 1e-6,
                _ => BDF_RTOL,
            }
        }

        fn atol(self) -> f64 {
            match self {
                Self::Exponential | Self::Lorenz => 1e-9,
                Self::RadauStiff => 1e-8,
                _ => BDF_ATOL,
            }
        }

        fn rates(self, n: usize) -> Vec<f64> {
            match self {
                Self::Exponential => vec![1.0; n],
                Self::Lorenz => vec![0.0; n],
                Self::RadauStiff => {
                    let denom = n.saturating_sub(1).max(1) as f64;
                    (0..n).map(|i| 1.0 + 999.0 * (i as f64 / denom)).collect()
                }
                _ => (0..n).map(|i| 1.0 + 10.0 * i as f64).collect(),
            }
        }

        fn y0(self, n: usize) -> Vec<f64> {
            match self {
                Self::Exponential | Self::Lorenz => vec![1.0; n],
                Self::RadauStiff => vec![1.0; n],
                _ => (0..n).map(|i| 1.0 + 0.25 * ((i % 7) as f64)).collect(),
            }
        }

        fn accepts_dimension(self, n: usize) -> bool {
            match self {
                Self::Exponential => n == 1,
                Self::Lorenz => n == 3,
                _ => n >= 2,
            }
        }

        fn is_explicit_rk(self) -> bool {
            matches!(self, Self::Exponential | Self::Lorenz)
        }

        fn analytic_final(self, index: usize, y0: &[f64], rates: &[f64]) -> Option<f64> {
            match self {
                Self::Exponential | Self::Diagonal | Self::RadauStiff => {
                    Some(y0[index] * (-rates[index] * self.t_end()).exp())
                }
                Self::Lorenz | Self::Coupled | Self::Dense => None,
            }
        }
    }

    /// Which solver both arms run. SciPy's `method=` string and our
    /// `SolverKind` must name the SAME algorithm, or the comparison is trap 2
    /// (unmatched config) wearing a different hat.
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum Method {
        Rk23,
        Rk45,
        Dop853,
        Bdf,
        Radau,
        Lsoda,
    }

    impl Method {
        fn parse(s: &str) -> Option<Self> {
            match s.to_ascii_lowercase().as_str() {
                "rk23" => Some(Self::Rk23),
                "rk45" => Some(Self::Rk45),
                "dop853" => Some(Self::Dop853),
                "bdf" => Some(Self::Bdf),
                "radau" => Some(Self::Radau),
                "lsoda" => Some(Self::Lsoda),
                _ => None,
            }
        }
        fn scipy(self) -> &'static str {
            match self {
                Self::Rk23 => "RK23",
                Self::Rk45 => "RK45",
                Self::Dop853 => "DOP853",
                Self::Bdf => "BDF",
                Self::Radau => "Radau",
                Self::Lsoda => "LSODA",
            }
        }
        fn kind(self) -> SolverKind {
            match self {
                Self::Rk23 => SolverKind::Rk23,
                Self::Rk45 => SolverKind::Rk45,
                Self::Dop853 => SolverKind::Dop853,
                Self::Bdf => SolverKind::Bdf,
                Self::Radau => SolverKind::Radau,
                Self::Lsoda => SolverKind::Lsoda,
            }
        }

        fn is_explicit_rk(self) -> bool {
            matches!(self, Self::Rk23 | Self::Rk45 | Self::Dop853)
        }
    }

    thread_local! {
        static METHOD: std::cell::Cell<Method> = const { std::cell::Cell::new(Method::Bdf) };
    }

    /// Evaluate the fixture's RHS. Must stay bit-for-bit the same expression the
    /// Python arm evaluates, or the arms are solving different problems (trap 2).
    fn rhs_into(fixture: Fixture, r: &[f64], y: &[f64]) -> Vec<f64> {
        match fixture {
            Fixture::Exponential => vec![-y[0]],
            Fixture::Lorenz => {
                let (sigma, rho, beta) = (10.0, 28.0, 8.0 / 3.0);
                vec![
                    sigma * (y[1] - y[0]),
                    y[0] * (rho - y[2]) - y[1],
                    y[0] * y[1] - beta * y[2],
                ]
            }
            Fixture::Diagonal | Fixture::RadauStiff => (0..y.len()).map(|i| -r[i] * y[i]).collect(),
            // Every J_ij is non-zero (J_ij = 1e-3/n), so NEITHER structural path can
            // fire and both arms run a dense LU. This is the true implementation-only
            // control: it isolates "our compiled step loop vs SciPy's Python one"
            // from "we have a fast path SciPy lacks". The RHS stays O(n) so the
            // callback cost does not change character between fixtures.
            Fixture::Dense => {
                let n = y.len();
                let mean = y.iter().sum::<f64>() / n as f64;
                (0..n).map(|i| -r[i] * y[i] + 1e-3 * mean).collect()
            }
            Fixture::Coupled => {
                let n = y.len();
                (0..n)
                    .map(|i| {
                        let left = if i == 0 { 0.0 } else { y[i - 1] };
                        let right = if i + 1 == n { 0.0 } else { y[i + 1] };
                        -r[i] * y[i] + 0.5 * (left - 2.0 * y[i] + right)
                    })
                    .collect()
            }
        }
    }

    fn solve_ours(r: &[f64], y0: &[f64]) -> SolveIvpResult {
        solve_ours_fx(FIXTURE.with(|f| f.get()), r, y0)
    }

    thread_local! {
        /// Set once from argv before any timing; read by the diagonal-defaulting
        /// `solve_ours` shim so the existing call sites need no edit.
        static FIXTURE: std::cell::Cell<Fixture> = const { std::cell::Cell::new(Fixture::Diagonal) };
    }

    fn solve_ours_fx(fixture: Fixture, r: &[f64], y0: &[f64]) -> SolveIvpResult {
        BDF_FORCE_DENSE_NEWTON.store(false, Ordering::Relaxed);
        solve_ivp(
            &mut |_t: f64, y: &[f64]| rhs_into(fixture, r, y),
            &SolveIvpOptions {
                t_span: (0.0, fixture.t_end()),
                y0,
                method: METHOD.with(|m| m.get()).kind(),
                rtol: fixture.rtol(),
                atol: ToleranceValue::Scalar(fixture.atol()),
                mode: RuntimeMode::Strict,
                ..Default::default()
            },
        )
        .expect("FrankenSciPy stiff solve")
    }

    struct Scipy {
        child: Child,
        stdin: ChildStdin,
        stdout: BufReader<ChildStdout>,
    }

    struct ScipyRun {
        secs: f64,
        nfev: usize,
        njev: usize,
        nlu: usize,
        steps: usize,
        rhs_calls: usize,
        status: i32,
        success: bool,
        y: Vec<f64>,
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
            stdout
                .read_line(&mut ready)
                .map_err(|e| format!("read READY: {e}"))?;
            Ok((
                Self {
                    child,
                    stdin,
                    stdout,
                },
                ready.trim().to_string(),
            ))
        }

        fn line(&mut self, cmd: &str) -> Result<String, String> {
            writeln!(self.stdin, "{cmd}").map_err(|e| format!("write {cmd}: {e}"))?;
            self.stdin.flush().map_err(|e| format!("flush: {e}"))?;
            let mut out = String::new();
            self.stdout
                .read_line(&mut out)
                .map_err(|e| format!("read reply to {cmd}: {e}"))?;
            Ok(out.trim().to_string())
        }

        fn solve(&mut self, n: usize, reps: usize) -> Result<ScipyRun, String> {
            let fixture = FIXTURE.with(|f| f.get());
            let reply = self.line(&format!(
                "SOLVE {n} {} {} {} {reps} {} {}",
                fixture.t_end(),
                fixture.rtol(),
                fixture.atol(),
                fixture.wire(),
                METHOD.with(|m| m.get()).scipy()
            ))?;
            let f: Vec<&str> = reply.split_whitespace().collect();
            if f.first() != Some(&"TIME") || f.len() != 10 {
                return Err(format!("bad SOLVE reply: {reply}"));
            }
            let parse_usize = |i: usize| {
                f[i].parse::<usize>()
                    .map_err(|e| format!("parse {}: {e}", f[i]))
            };
            let y = f[9]
                .split(',')
                .map(|value| {
                    value
                        .parse::<f64>()
                        .map_err(|e| format!("parse result component {value}: {e}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ScipyRun {
                secs: f[1]
                    .parse::<f64>()
                    .map_err(|e| format!("parse elapsed time: {e}"))?,
                nfev: parse_usize(2)?,
                njev: parse_usize(3)?,
                nlu: parse_usize(4)?,
                steps: parse_usize(5)?,
                rhs_calls: parse_usize(6)?,
                status: f[7]
                    .parse::<i32>()
                    .map_err(|e| format!("parse status: {e}"))?,
                success: match f[8] {
                    "True" => true,
                    "False" => false,
                    value => return Err(format!("parse success: {value}")),
                },
                y,
            })
        }

        fn rhs_cost(&mut self, n: usize, calls: usize) -> Result<f64, String> {
            let reply = self.line(&format!(
                "RHSCOST {n} {calls} {}",
                FIXTURE.with(|f| f.get()).wire()
            ))?;
            reply
                .split_whitespace()
                .nth(1)
                .ok_or_else(|| format!("bad RHSCOST reply: {reply}"))?
                .parse::<f64>()
                .map_err(|e| format!("parse rhs cost: {e}"))
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

    /// Deterministic percentile-bootstrap CI on the median (the campaign gate).
    fn boot_ci(v: &[f64]) -> (f64, f64) {
        if v.is_empty() {
            return (f64::NAN, f64::NAN);
        }
        let mut state = 0x6a09_e667_f3bc_c909u64;
        let mut meds = Vec::with_capacity(10_000);
        for _ in 0..10_000 {
            let mut s = Vec::with_capacity(v.len());
            for _ in 0..v.len() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                s.push(v[(state as usize) % v.len()]);
            }
            meds.push(median(s));
        }
        meds.sort_by(f64::total_cmp);
        (
            meds[(10_000f64 * 0.025) as usize],
            meds[(10_000f64 * 0.975) as usize],
        )
    }

    fn cv(v: &[f64]) -> f64 {
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let variance = v.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / v.len() as f64;
        variance.sqrt() / mean
    }

    fn time_ours(r: &[f64], y0: &[f64], reps: usize) -> f64 {
        let mut result = None;
        let start = Instant::now();
        for _ in 0..reps {
            result = Some(black_box(solve_ours(black_box(r), black_box(y0))));
        }
        let elapsed = start.elapsed().as_secs_f64();
        // Match SciPy's timer: the final result remains live when its timer stops.
        // Earlier results are dropped on overwrite inside both loops.
        black_box(result);
        elapsed
    }

    fn time_scipy(sp: &mut Scipy, n: usize, reps: usize) -> f64 {
        let run = match sp.solve(n, reps) {
            Ok(run) => run,
            Err(error) => {
                eprintln!("ABORT: timed SciPy solve failed: {error}");
                std::process::exit(9);
            }
        };
        if !run.success || run.status != 0 || run.y.len() != n {
            eprintln!("ABORT: timed SciPy solve returned an invalid result");
            std::process::exit(9);
        }
        if !run.secs.is_finite() || run.secs <= 0.0 {
            eprintln!("ABORT: timed SciPy solve returned an invalid elapsed time");
            std::process::exit(9);
        }
        run.secs
    }

    /// One side-by-side incumbent pair. The conceptual arm identities remain fixed
    /// while execution order alternates, so every ratio is SciPy / FrankenSciPy.
    fn incumbent_pair(
        sp: &mut Scipy,
        n: usize,
        r: &[f64],
        y0: &[f64],
        reps: usize,
        round: usize,
    ) -> (f64, f64) {
        if round % 2 == 0 {
            let ours = time_ours(r, y0, reps);
            let scipy = time_scipy(sp, n, reps);
            (ours, scipy)
        } else {
            let scipy = time_scipy(sp, n, reps);
            let ours = time_ours(r, y0, reps);
            (ours, scipy)
        }
    }

    /// Identical-arm A/A control for FrankenSciPy. Labels A and B are stable while
    /// order alternates, preventing directional drift from being hidden in the null.
    fn ours_null_pair(r: &[f64], y0: &[f64], reps: usize, round: usize) -> f64 {
        let (a, b) = if round % 2 == 0 {
            (time_ours(r, y0, reps), time_ours(r, y0, reps))
        } else {
            let b = time_ours(r, y0, reps);
            let a = time_ours(r, y0, reps);
            (a, b)
        };
        a / b
    }

    /// Identical-arm A/A control for the live SciPy incumbent.
    fn scipy_null_pair(sp: &mut Scipy, n: usize, reps: usize, round: usize) -> f64 {
        let (a, b) = if round % 2 == 0 {
            (time_scipy(sp, n, reps), time_scipy(sp, n, reps))
        } else {
            let b = time_scipy(sp, n, reps);
            let a = time_scipy(sp, n, reps);
            (a, b)
        };
        a / b
    }

    fn cpu_affinity() -> String {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|status| {
                status
                    .lines()
                    .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
                    .map(str::trim)
                    .map(str::to_string)
            })
            .unwrap_or_else(|| "unknown".to_string())
    }

    pub fn run() {
        let exe = std::env::current_exe().expect("current_exe");
        let sha = {
            let mut h = Sha256::new();
            h.update(std::fs::read(&exe).expect("read own ELF"));
            format!("{:x}", h.finalize())
        };
        println!("elf_sha256={sha}");

        let args: Vec<String> = std::env::args().collect();
        let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(128);
        let rounds: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(11);
        let reps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
        let fixture = args
            .get(4)
            .and_then(|s| Fixture::parse(s))
            .unwrap_or(Fixture::Diagonal);
        FIXTURE.with(|f| f.set(fixture));
        let method = args
            .get(5)
            .and_then(|s| Method::parse(s))
            .unwrap_or(Method::Bdf);
        METHOD.with(|m| m.set(method));
        // argv[4] is the fixture; the optional script path moved to argv[5] when the
        // coupled fixture was added. Both reading argv[4] made the harness spawn
        // `python3 diagonal`, which surfaced as a confusing "not genuine" abort.
        let script = args
            .get(6)
            .cloned()
            .unwrap_or_else(|| "crates/fsci-integrate/python/scipy_bdf_arm.py".to_string());
        let affinity = cpu_affinity();
        println!("cpu_affinity={affinity}");
        if affinity == "unknown" || affinity.contains(',') || affinity.contains('-') {
            eprintln!("ABORT: pin this invocation to exactly one CPU with taskset");
            std::process::exit(2);
        }
        if !fixture.accepts_dimension(n) || rounds < 3 || reps == 0 {
            eprintln!(
                "ABORT: fixture {} rejects n={n}; require exponential n=1, \
                 lorenz n=3, all other fixtures n>=2, rounds>=3, and reps>=1",
                fixture.label()
            );
            std::process::exit(2);
        }
        if fixture.is_explicit_rk() != method.is_explicit_rk() {
            eprintln!(
                "ABORT: explicit-RK fixtures require rk23/rk45/dop853, while \
                 stiff fixtures require bdf/radau/lsoda"
            );
            std::process::exit(2);
        }
        if fixture == Fixture::RadauStiff && method != Method::Radau {
            eprintln!("ABORT: radau-stiff fixture requires method=radau");
            std::process::exit(2);
        }
        println!(
            "fixture={} n={n} rounds={rounds} reps={reps} method={} \
             t_span=[0,{}] rtol={} atol={} t_eval=None jac=None \
             scipy_rhs_counter_outside_timing=true",
            fixture.label(),
            method.scipy(),
            fixture.t_end(),
            fixture.rtol(),
            fixture.atol()
        );

        let (mut sp, ready) = match Scipy::start(&script) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("ABORT: cannot start SciPy arm: {e}");
                std::process::exit(3);
            }
        };
        println!("scipy_arm: {ready}");
        // TRAP 1: refuse to measure a non-genuine incumbent.
        if !ready.starts_with("READY scipy=")
            || !ready.contains("solve_ivp_mod=scipy.integrate._ivp.ivp")
            || !ready.contains("fsci_loaded=False")
            || !ready.contains("genuine=True")
        {
            eprintln!("ABORT: SciPy arm is not genuine (dispatch trap)");
            std::process::exit(4);
        }
        let scipy_version = ready
            .split_whitespace()
            .find_map(|field| field.strip_prefix("scipy="))
            .expect("READY line has scipy version");
        println!(
            "Legacy incumbent arm: SciPy {scipy_version}; side-by-side same-invocation; \
             child-side solve-only timing"
        );

        let r = fixture.rates(n);
        let y0 = fixture.y0(n);

        // ── TRAP 2: prove both arms solved the SAME problem, before any timing.
        // These discarded parity solves also warm both implementations once.
        BDF_DIAG_NEWTON_HITS.store(0, Ordering::Relaxed);
        BDF_BAND_NEWTON_HITS.store(0, Ordering::Relaxed);
        RADAU_DIAG_NEWTON_HITS.store(0, Ordering::Relaxed);
        let ours = solve_ours(&r, &y0);
        let diag_hits = BDF_DIAG_NEWTON_HITS.load(Ordering::Relaxed);
        let band_hits = BDF_BAND_NEWTON_HITS.load(Ordering::Relaxed);
        let radau_diag_hits = RADAU_DIAG_NEWTON_HITS.load(Ordering::Relaxed);
        let theirs = match sp.solve(n, 1) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("ABORT: scipy solve failed: {e}");
                std::process::exit(5);
            }
        };
        let our_y = ours
            .y
            .last()
            .expect("FrankenSciPy result has a final state");
        if !ours.success
            || ours.status != 0
            || !theirs.success
            || theirs.status != 0
            || our_y.len() != n
            || theirs.y.len() != n
            // EXECUTION PROOF, per fixture. Each fixture must route through the path
            // it is designed to exercise and NO other, otherwise the arm is not
            // measuring what the row will claim:
            //   diagonal → the diagonal path only
            //   coupled  → the banded path only (tridiagonal J; the diagonal
            //              predicate must decline)
            //   dense    → NEITHER path; both arms run a dense LU, which is what
            //              makes this the implementation-only control
            || match (method, fixture) {
                // BDF exposes per-path hit counters, so the proof is exact: the
                // fixture's intended path must fire and no other.
                (Method::Bdf, Fixture::Diagonal) => diag_hits == 0 || band_hits != 0,
                (Method::Bdf, Fixture::Coupled) => band_hits == 0 || diag_hits != 0,
                (Method::Bdf, Fixture::Dense) => diag_hits != 0 || band_hits != 0,
                (Method::Bdf, Fixture::RadauStiff) => true,
                // LSODA is a BDF-FAMILY method: it switches to BDF in stiff regions,
                // so the BDF counters legitimately fire and the per-fixture rule
                // applies to it exactly as it does to BDF. (Discovered by this proof
                // aborting on `diag_hits=30` — the assert was wrong, not the run.)
                (Method::Lsoda, Fixture::Diagonal) => diag_hits == 0 || band_hits != 0,
                (Method::Lsoda, Fixture::Coupled) => band_hits == 0 || diag_hits != 0,
                (Method::Lsoda, Fixture::Dense) => diag_hits != 0 || band_hits != 0,
                (Method::Lsoda, Fixture::RadauStiff) => true,
                // Radau's own counter proves the structural stage/error path fired.
                (Method::Radau, Fixture::Diagonal | Fixture::RadauStiff) => {
                    radau_diag_hits == 0 || diag_hits != 0 || band_hits != 0
                }
                (Method::Radau, Fixture::Coupled | Fixture::Dense) => {
                    radau_diag_hits != 0 || diag_hits != 0 || band_hits != 0
                }
                // Explicit Runge-Kutta methods never enter a BDF/Radau Newton
                // path. Zero counters are the counted dispatch proof.
                (
                    Method::Rk23 | Method::Rk45 | Method::Dop853,
                    Fixture::Exponential | Fixture::Lorenz,
                ) => diag_hits != 0 || band_hits != 0 || radau_diag_hits != 0,
                // Rejected above by the fixture/method compatibility gate.
                _ => true,
            }
        {
            eprintln!(
                "ABORT: invalid execution proof (ours success={} status={} len={}; \
                 scipy success={} status={} len={}; diag_hits={diag_hits} \
                 band_hits={band_hits} radau_diag_hits={radau_diag_hits})",
                ours.success,
                ours.status,
                our_y.len(),
                theirs.success,
                theirs.status,
                theirs.y.len()
            );
            std::process::exit(6);
        }

        let mut max_abs_diff = 0.0f64;
        let mut max_scaled_diff = 0.0f64;
        let mut max_scaled_ours_analytic = 0.0f64;
        let mut max_scaled_scipy_analytic = 0.0f64;
        let mut analytic_components = 0usize;
        for index in 0..n {
            let difference = (our_y[index] - theirs.y[index]).abs();
            let comparison_scale =
                fixture.atol() + fixture.rtol() * our_y[index].abs().max(theirs.y[index].abs());
            max_abs_diff = max_abs_diff.max(difference);
            max_scaled_diff = max_scaled_diff.max(difference / comparison_scale);
            if let Some(analytic) = fixture.analytic_final(index, &y0, &r) {
                analytic_components += 1;
                let analytic_scale = fixture.atol() + fixture.rtol() * analytic.abs();
                max_scaled_ours_analytic =
                    max_scaled_ours_analytic.max((our_y[index] - analytic).abs() / analytic_scale);
                max_scaled_scipy_analytic = max_scaled_scipy_analytic
                    .max((theirs.y[index] - analytic).abs() / analytic_scale);
            }
        }
        if analytic_components == n {
            println!(
                "agreement: components={n}/{n} max_abs_diff={max_abs_diff:.3e} \
                 max_scaled_diff={max_scaled_diff:.3} \
                 max_scaled_ours_vs_analytic={max_scaled_ours_analytic:.3} \
                 max_scaled_scipy_vs_analytic={max_scaled_scipy_analytic:.3}"
            );
        } else {
            println!(
                "agreement: components={n}/{n} max_abs_diff={max_abs_diff:.3e} \
                 max_scaled_diff={max_scaled_diff:.3} analytic_reference=not_available"
            );
        }
        println!(
            "counters: ours nfev={} njev={} nlu={} steps={} diag_hits={diag_hits} \
             band_hits={band_hits} radau_diag_hits={radau_diag_hits} | \
             scipy nfev={} njev={} nlu={} steps={} \
             actual_rhs_calls={}",
            ours.nfev,
            ours.njev,
            ours.nlu,
            ours.t.len(),
            theirs.nfev,
            theirs.njev,
            theirs.nlu,
            theirs.steps,
            theirs.rhs_calls
        );
        // Local-error controllers need not choose identical steps, but a 100x
        // tolerance-scaled componentwise disagreement is not admissible.
        if !max_scaled_diff.is_finite() || max_scaled_diff > 100.0 {
            eprintln!(
                "ABORT: arms disagree componentwise beyond 100 tolerance units — \
                 not an admissible comparison"
            );
            std::process::exit(7);
        }

        // ── TRAP 6: decompose the Python callback cost out of SciPy's total.
        let rhs_calls = theirs.rhs_calls.max(1);
        let callback_repeats = 32usize;
        let rhs_secs = sp
            .rhs_cost(n, rhs_calls.saturating_mul(callback_repeats))
            .unwrap_or(f64::NAN)
            / callback_repeats as f64;
        if !rhs_secs.is_finite() || rhs_secs < 0.0 {
            eprintln!("ABORT: invalid Python RHS decomposition timing");
            std::process::exit(8);
        }

        // ── TRAPS 3 + 4: interleave inside each round, alternate order, and run an
        // A/A null for BOTH arms.
        let (mut ours_t, mut theirs_t, mut ratio) = (vec![], vec![], vec![]);
        let (mut null_ours, mut null_theirs) = (vec![], vec![]);
        for round in 0..rounds {
            // Rotate the three pair types as well as alternating order within each
            // pair. No arm or null is systematically closest to a round boundary.
            let (ours_secs, scipy_secs, ours_null, scipy_null) = match round % 3 {
                0 => {
                    let incumbent = incumbent_pair(&mut sp, n, &r, &y0, reps, round);
                    let ours_null = ours_null_pair(&r, &y0, reps, round);
                    let scipy_null = scipy_null_pair(&mut sp, n, reps, round);
                    (incumbent.0, incumbent.1, ours_null, scipy_null)
                }
                1 => {
                    let scipy_null = scipy_null_pair(&mut sp, n, reps, round);
                    let incumbent = incumbent_pair(&mut sp, n, &r, &y0, reps, round);
                    let ours_null = ours_null_pair(&r, &y0, reps, round);
                    (incumbent.0, incumbent.1, ours_null, scipy_null)
                }
                _ => {
                    let ours_null = ours_null_pair(&r, &y0, reps, round);
                    let scipy_null = scipy_null_pair(&mut sp, n, reps, round);
                    let incumbent = incumbent_pair(&mut sp, n, &r, &y0, reps, round);
                    (incumbent.0, incumbent.1, ours_null, scipy_null)
                }
            };
            ours_t.push(ours_secs);
            theirs_t.push(scipy_secs);
            ratio.push(scipy_secs / ours_secs); // >1 => FrankenSciPy is faster
            null_ours.push(ours_null);
            null_theirs.push(scipy_null);
        }

        let (rlo, rhi) = boot_ci(&ratio);
        let (nolo, nohi) = boot_ci(&null_ours);
        let (ntlo, nthi) = boot_ci(&null_theirs);
        let p50_ours = median(ours_t.clone());
        let p50_theirs = median(theirs_t.clone());
        println!(
            "OURS   p50={:.6}ms/rep  SCIPY p50={:.6}ms/rep",
            p50_ours * 1e3 / reps as f64,
            p50_theirs * 1e3 / reps as f64
        );
        println!(
            "NULL-ours   median={:.6} ci95=[{:.6},{:.6}] cv={:.3}% (provenance only)",
            median(null_ours.clone()),
            nolo,
            nohi,
            cv(&null_ours) * 100.0
        );
        println!(
            "NULL-scipy  median={:.6} ci95=[{:.6},{:.6}] cv={:.3}% (provenance only)",
            median(null_theirs.clone()),
            ntlo,
            nthi,
            cv(&null_theirs) * 100.0
        );
        let ratio_p50 = median(ratio.clone());
        println!(
            "Incumbent ratio: SciPy / FrankenSciPy = {ratio_p50:.4}x \
             (bootstrap-median ci95=[{rlo:.4},{rhi:.4}], cv={:.3}% provenance only)",
            cv(&ratio) * 100.0
        );

        // Gate against the WORSE of the two arms' nulls — an asymmetric null is the
        // signature of trap 4, and using only our own would hide it.
        let edge = nohi
            .max(nthi)
            .max(1.0 / nolo.max(1e-9))
            .max(1.0 / ntlo.max(1e-9));
        let required = 1.0 + 2.0 * (edge - 1.0);
        let outcome = if rlo > required {
            "DECIDED FRANKENSCIPY WIN"
        } else if rhi < 1.0 / required {
            "DECIDED FRANKENSCIPY LOSS"
        } else {
            "NOT DECIDED"
        };
        println!(
            "median-CI gate: worst_null_edge={edge:.4} required={required:.4} \
             ratio_ci=[{rlo:.4},{rhi:.4}] => {outcome}"
        );

        // TRAP 6 decomposition, reported whatever it says.
        let scipy_per_rep = p50_theirs / reps as f64;
        let cb_frac = rhs_secs / scipy_per_rep;
        println!(
            "decomposition: scipy {:.4}ms/solve, of which Python RHS callbacks \
             ({rhs_calls} calls) = {:.4}ms = {:.1}% ; \
             callback-free sensitivity ratio = {:.4}x",
            scipy_per_rep * 1e3,
            rhs_secs * 1e3,
            cb_frac * 100.0,
            (scipy_per_rep - rhs_secs).max(0.0) / (p50_ours / reps as f64)
        );
        sp.quit();
    }
}

#[cfg(feature = "bdf-diag-bench")]
fn main() {
    bench::run();
}

#[cfg(not(feature = "bdf-diag-bench"))]
fn main() {
    eprintln!("perf_bdf_vs_scipy requires --features bdf-diag-bench");
    std::process::exit(2);
}
