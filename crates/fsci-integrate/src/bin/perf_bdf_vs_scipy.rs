//! ODE head-to-head against a LIVE SciPy arm.
//!
//! Campaign policy 2026-07-27: a self-speedup is maintenance; a campaign win needs a
//! measured ratio against the actual legacy incumbent, from a harness that runs the
//! incumbent side-by-side IN THE SAME INVOCATION. This binary measures whether the
//! structural BDF/Radau/LSODA and explicit-RK claims actually translate against
//! SciPy itself.
//!
//! SciPy runs in a persistent `python3 -u` co-process (`python/scipy_bdf_arm.py`).
//! Each arm times ITSELF — SciPy with `perf_counter` around its work, we with
//! `Instant` — so the pipe round-trip is outside both measured regions. The
//! `decay-screen` fixture times a complete parameter-screening job: deterministic
//! model/input construction, all solves, requested-output materialization, and
//! scientific postprocessing.
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
    use fsci_integrate::{
        SolveIvpOptions, SolveIvpResult, SolverKind, ToleranceValue, solve_ivp, solve_ivp_many,
    };
    use fsci_runtime::RuntimeMode;
    use sha2::{Digest, Sha256};
    use std::collections::{BTreeSet, HashSet};
    use std::hint::black_box;
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
    use std::sync::Mutex;
    use std::sync::atomic::Ordering;
    use std::time::Instant;

    const BDF_T_END: f64 = 1.0;
    const BDF_RTOL: f64 = 1e-8;
    const BDF_ATOL: f64 = 1e-10;
    const LOTKA_T_END: f64 = 10.0;
    const LOTKA_RTOL: f64 = 1e-8;
    const LOTKA_ATOL: f64 = 1e-10;
    const LOTKA_SAMPLES: usize = 150;
    const DECAY_JOB_SCENARIOS: usize = 16;
    const DECAY_JOB_SAMPLES: usize = 65;
    const DECAY_JOB_MAX_WORKERS: usize = 8;
    const DECAY_QUADRATIC: f64 = 0.125;
    const NULL_MEDIAN_BIAS_LIMIT: f64 = 0.02;

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
        /// Historical `solve_ivp_many` scientific workload: an ensemble of
        /// Lotka-Volterra trajectories integrated over `[0,10]` at 150 requested
        /// samples with `rtol=1e-8`, `atol=1e-10`. Here argv `n` is batch size.
        LotkaMany,
        /// Completion-only control for the same ensemble. Both arms retain their
        /// solver-chosen accepted-step histories (`t_eval=None`) and are compared
        /// at `t=10`. This remains admissible while the sampled RK45 dense-output
        /// path is correctness-blocked by frankenscipy-3m5ip.
        LotkaManyFinal,
        /// Whole scientific job: screen 16 initial-dose scenarios for a
        /// 512-species independent stiff kinetic model, retain 65 observation
        /// times, then compute per-species exposure (trapezoidal AUC) and
        /// per-scenario terminal mass. Here argv `n` is the species count.
        DecayScreen,
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
                "lotka-many" | "lotka" | "many" => Some(Self::LotkaMany),
                "lotka-final-many" | "lotka-final" | "many-final" => Some(Self::LotkaManyFinal),
                "decay-screen" | "decay-job" | "screen" => Some(Self::DecayScreen),
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
                Self::LotkaMany => "lotka-volterra-ensemble",
                Self::LotkaManyFinal => "lotka-volterra-completion-ensemble",
                Self::DecayScreen => "stiff-independent-reaction-screen",
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
                Self::LotkaMany => "lotka-many",
                Self::LotkaManyFinal => "lotka-final-many",
                Self::DecayScreen => "decay-screen",
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
                Self::LotkaMany | Self::LotkaManyFinal => LOTKA_T_END,
                Self::RadauStiff => 0.2,
                _ => BDF_T_END,
            }
        }

        fn rtol(self) -> f64 {
            match self {
                Self::Exponential | Self::Lorenz | Self::RadauStiff => 1e-6,
                Self::LotkaMany | Self::LotkaManyFinal => LOTKA_RTOL,
                _ => BDF_RTOL,
            }
        }

        fn atol(self) -> f64 {
            match self {
                Self::Exponential | Self::Lorenz => 1e-9,
                Self::LotkaMany | Self::LotkaManyFinal => LOTKA_ATOL,
                Self::RadauStiff => 1e-8,
                _ => BDF_ATOL,
            }
        }

        fn rates(self, n: usize) -> Vec<f64> {
            match self {
                Self::Exponential => vec![1.0; n],
                Self::Lorenz => vec![0.0; n],
                Self::LotkaMany | Self::LotkaManyFinal => Vec::new(),
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
                Self::LotkaMany | Self::LotkaManyFinal => Vec::new(),
                Self::RadauStiff => vec![1.0; n],
                _ => (0..n).map(|i| 1.0 + 0.25 * ((i % 7) as f64)).collect(),
            }
        }

        fn accepts_dimension(self, n: usize) -> bool {
            match self {
                Self::Exponential => n == 1,
                Self::Lorenz => n == 3,
                Self::LotkaMany | Self::LotkaManyFinal => n >= 1,
                _ => n >= 2,
            }
        }

        fn is_explicit_rk(self) -> bool {
            matches!(
                self,
                Self::Exponential | Self::Lorenz | Self::LotkaMany | Self::LotkaManyFinal
            )
        }

        fn is_lotka_many(self) -> bool {
            matches!(self, Self::LotkaMany | Self::LotkaManyFinal)
        }

        fn is_decay_screen(self) -> bool {
            self == Self::DecayScreen
        }

        fn is_batch_job(self) -> bool {
            self.is_lotka_many() || self.is_decay_screen()
        }

        fn lotka_sampled(self) -> bool {
            self == Self::LotkaMany
        }

        fn analytic_final(self, index: usize, y0: &[f64], rates: &[f64]) -> Option<f64> {
            match self {
                Self::Exponential | Self::DecayScreen | Self::Diagonal | Self::RadauStiff => {
                    Some(y0[index] * (-rates[index] * self.t_end()).exp())
                }
                Self::Lorenz
                | Self::LotkaMany
                | Self::LotkaManyFinal
                | Self::Coupled
                | Self::Dense => None,
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
            Fixture::LotkaMany | Fixture::LotkaManyFinal | Fixture::DecayScreen => {
                unreachable!("scientific batch fixtures use a dedicated batch path")
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

    struct ScipyManyCheck {
        successes: usize,
        nfev: usize,
        njev: usize,
        nlu: usize,
        stored_points: usize,
        rhs_calls: usize,
        samples: usize,
        input_sha256: String,
        min_component: f64,
        max_invariant_drift: f64,
        values: Vec<f64>,
    }

    struct ScipyDecayCheck {
        successes: usize,
        nfev: usize,
        njev: usize,
        nlu: usize,
        stored_points: usize,
        rhs_calls: usize,
        scenarios: usize,
        samples: usize,
        input_sha256: String,
        worker_processes: usize,
        worker_threads: usize,
        peak_rss_kib: usize,
        rhs_calls_by_scenario: Vec<usize>,
        values: Vec<f64>,
        exposures: Vec<f64>,
        terminal_masses: Vec<f64>,
    }

    impl Scipy {
        fn start(script: &str) -> Result<(Self, String), String> {
            let mut child = Command::new("python3")
                .arg("-u")
                .arg(script)
                .env("OPENBLAS_NUM_THREADS", "1")
                .env("OMP_NUM_THREADS", "1")
                .env("MKL_NUM_THREADS", "1")
                .env("BLIS_NUM_THREADS", "1")
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

        fn check_many(&mut self, batch: usize, sampled: bool) -> Result<ScipyManyCheck, String> {
            let mode = if sampled { "sampled" } else { "final" };
            let reply = self.line(&format!("MANY_CHECK {batch} {mode}"))?;
            let fields: Vec<&str> = reply.split_whitespace().collect();
            if fields.first() != Some(&"CHECK") || fields.len() != 12 {
                return Err(format!("bad MANY_CHECK reply: {reply}"));
            }
            let parse_usize = |index: usize| {
                fields[index]
                    .parse::<usize>()
                    .map_err(|error| format!("parse {}: {error}", fields[index]))
            };
            let parse_f64 = |index: usize| {
                fields[index]
                    .parse::<f64>()
                    .map_err(|error| format!("parse {}: {error}", fields[index]))
            };
            let values = fields[11]
                .split(',')
                .map(|value| {
                    value
                        .parse::<f64>()
                        .map_err(|error| format!("parse trajectory component {value}: {error}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ScipyManyCheck {
                successes: parse_usize(1)?,
                nfev: parse_usize(2)?,
                njev: parse_usize(3)?,
                nlu: parse_usize(4)?,
                stored_points: parse_usize(5)?,
                rhs_calls: parse_usize(6)?,
                samples: parse_usize(7)?,
                input_sha256: fields[8].to_string(),
                min_component: parse_f64(9)?,
                max_invariant_drift: parse_f64(10)?,
                values,
            })
        }

        fn time_many(&mut self, batch: usize, reps: usize, sampled: bool) -> Result<f64, String> {
            let mode = if sampled { "sampled" } else { "final" };
            let reply = self.line(&format!("MANY_TIME {batch} {reps} {mode}"))?;
            let fields: Vec<&str> = reply.split_whitespace().collect();
            if fields.first() != Some(&"TIME") || fields.len() != 3 {
                return Err(format!("bad MANY_TIME reply: {reply}"));
            }
            let elapsed = fields[1]
                .parse::<f64>()
                .map_err(|error| format!("parse many elapsed time: {error}"))?;
            let successes = fields[2]
                .parse::<usize>()
                .map_err(|error| format!("parse many success count: {error}"))?;
            if successes != batch || !elapsed.is_finite() || elapsed <= 0.0 {
                return Err(format!(
                    "invalid MANY_TIME result: successes={successes}/{batch} elapsed={elapsed}"
                ));
            }
            Ok(elapsed)
        }

        fn many_rhs_cost(&mut self, calls: usize) -> Result<f64, String> {
            let reply = self.line(&format!("MANY_RHSCOST {calls}"))?;
            reply
                .split_whitespace()
                .nth(1)
                .ok_or_else(|| format!("bad MANY_RHSCOST reply: {reply}"))?
                .parse::<f64>()
                .map_err(|error| format!("parse many rhs cost: {error}"))
        }

        fn check_decay(
            &mut self,
            n: usize,
            scenarios: usize,
            workers: usize,
            jacobian_mode: &str,
        ) -> Result<ScipyDecayCheck, String> {
            let reply = self.line(&format!(
                "DECAY_CHECK {n} {scenarios} {workers} {jacobian_mode}"
            ))?;
            let fields: Vec<&str> = reply.split_whitespace().collect();
            if fields.first() != Some(&"JOB_CHECK") || fields.len() != 17 {
                return Err(format!("bad DECAY_CHECK reply: {reply}"));
            }
            let parse_usize = |index: usize| {
                fields[index]
                    .parse::<usize>()
                    .map_err(|error| format!("parse {}: {error}", fields[index]))
            };
            let parse_usizes = |index: usize| {
                fields[index]
                    .split(',')
                    .map(|value| {
                        value
                            .parse::<usize>()
                            .map_err(|error| format!("parse count {value}: {error}"))
                    })
                    .collect::<Result<Vec<_>, _>>()
            };
            let parse_f64s = |index: usize| {
                fields[index]
                    .split(',')
                    .map(|value| {
                        value
                            .parse::<f64>()
                            .map_err(|error| format!("parse job value {value}: {error}"))
                    })
                    .collect::<Result<Vec<_>, _>>()
            };
            Ok(ScipyDecayCheck {
                successes: parse_usize(1)?,
                nfev: parse_usize(2)?,
                njev: parse_usize(3)?,
                nlu: parse_usize(4)?,
                stored_points: parse_usize(5)?,
                rhs_calls: parse_usize(6)?,
                scenarios: parse_usize(7)?,
                samples: parse_usize(8)?,
                input_sha256: fields[9].to_string(),
                worker_processes: parse_usize(10)?,
                worker_threads: parse_usize(11)?,
                peak_rss_kib: parse_usize(12)?,
                rhs_calls_by_scenario: parse_usizes(13)?,
                values: parse_f64s(14)?,
                exposures: parse_f64s(15)?,
                terminal_masses: parse_f64s(16)?,
            })
        }

        fn time_decay(
            &mut self,
            n: usize,
            scenarios: usize,
            workers: usize,
            reps: usize,
            jacobian_mode: &str,
        ) -> Result<f64, String> {
            let reply = self.line(&format!(
                "DECAY_TIME {n} {scenarios} {workers} {reps} {jacobian_mode}"
            ))?;
            let fields: Vec<&str> = reply.split_whitespace().collect();
            if fields.first() != Some(&"JOB_TIME") || fields.len() != 6 {
                return Err(format!("bad DECAY_TIME reply: {reply}"));
            }
            let elapsed = fields[1]
                .parse::<f64>()
                .map_err(|error| format!("parse decay-job elapsed time: {error}"))?;
            let successes = fields[2]
                .parse::<usize>()
                .map_err(|error| format!("parse decay-job successes: {error}"))?;
            let worker_processes = fields[3]
                .parse::<usize>()
                .map_err(|error| format!("parse decay-job worker processes: {error}"))?;
            let worker_threads = fields[4]
                .parse::<usize>()
                .map_err(|error| format!("parse decay-job worker threads: {error}"))?;
            let peak_rss_kib = fields[5]
                .parse::<usize>()
                .map_err(|error| format!("parse decay-job peak RSS: {error}"))?;
            if successes != scenarios
                || worker_processes != workers
                || worker_threads < workers
                || peak_rss_kib == 0
                || !elapsed.is_finite()
                || elapsed <= 0.0
            {
                return Err(format!(
                    "invalid DECAY_TIME result: successes={successes}/{scenarios} \
                     processes={worker_processes}/{workers} threads={worker_threads} \
                     peak_rss_kib={peak_rss_kib} elapsed={elapsed}"
                ));
            }
            Ok(elapsed)
        }

        fn decay_rhs_replay(
            &mut self,
            n: usize,
            workers: usize,
            calls: &[usize],
        ) -> Result<f64, String> {
            let call_list = calls
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let reply = self.line(&format!("DECAY_RHSCOST {n} {workers} {call_list}"))?;
            let fields: Vec<&str> = reply.split_whitespace().collect();
            if fields.first() != Some(&"JOB_RHS_TIME") || fields.len() != 2 {
                return Err(format!("bad DECAY_RHSCOST reply: {reply}"));
            }
            fields[1]
                .parse::<f64>()
                .map_err(|error| format!("parse decay RHS replay: {error}"))
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

    fn percentile(mut values: Vec<f64>, percentile: f64) -> f64 {
        values.sort_by(f64::total_cmp);
        if values.is_empty() {
            return f64::NAN;
        }
        let index = ((values.len() - 1) as f64 * percentile).ceil() as usize;
        values[index.min(values.len() - 1)]
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
        if round.is_multiple_of(2) {
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
        let (a, b) = if round.is_multiple_of(2) {
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
        let (a, b) = if round.is_multiple_of(2) {
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

    fn affinity_cpus(affinity: &str) -> Result<Vec<usize>, String> {
        let mut cpus = BTreeSet::new();
        for part in affinity.split(',') {
            if let Some((start, end)) = part.split_once('-') {
                let start = start
                    .parse::<usize>()
                    .map_err(|error| format!("parse affinity start {start}: {error}"))?;
                let end = end
                    .parse::<usize>()
                    .map_err(|error| format!("parse affinity end {end}: {error}"))?;
                if start > end {
                    return Err(format!("descending CPU affinity range: {part}"));
                }
                cpus.extend(start..=end);
            } else {
                cpus.insert(
                    part.parse::<usize>()
                        .map_err(|error| format!("parse affinity CPU {part}: {error}"))?,
                );
            }
        }
        if cpus.is_empty() {
            return Err("CPU affinity is empty".to_string());
        }
        Ok(cpus.into_iter().collect())
    }

    fn read_cpu_policy_field(cpu: usize, field: &str) -> Result<String, String> {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/{field}");
        std::fs::read_to_string(&path)
            .map(|value| value.trim().replace(char::is_whitespace, "_"))
            .map_err(|error| format!("read {path}: {error}"))
    }

    fn cpu_frequency_policy(affinity: &str) -> Result<String, String> {
        let cpus = affinity_cpus(affinity)?;
        let mut drivers = BTreeSet::new();
        let mut governors = BTreeSet::new();
        let mut min_frequencies = BTreeSet::new();
        let mut max_frequencies = BTreeSet::new();
        for &cpu in &cpus {
            drivers.insert(read_cpu_policy_field(cpu, "scaling_driver")?);
            governors.insert(read_cpu_policy_field(cpu, "scaling_governor")?);
            min_frequencies.insert(read_cpu_policy_field(cpu, "scaling_min_freq")?);
            max_frequencies.insert(read_cpu_policy_field(cpu, "scaling_max_freq")?);
        }
        Ok(format!(
            "cpus={} scaling_drivers={} scaling_governors={} \
             scaling_min_freq_khz={} scaling_max_freq_khz={}",
            cpus.iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(","),
            drivers.into_iter().collect::<Vec<_>>().join(","),
            governors.into_iter().collect::<Vec<_>>().join(","),
            min_frequencies.into_iter().collect::<Vec<_>>().join(","),
            max_frequencies.into_iter().collect::<Vec<_>>().join(",")
        ))
    }

    fn host_identity() -> String {
        std::fs::read_to_string("/proc/sys/kernel/hostname")
            .map(|hostname| hostname.trim().replace(char::is_whitespace, "_"))
            .unwrap_or_else(|_| "unknown".to_string())
    }

    fn cpu_topology() -> (usize, usize) {
        let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") else {
            return (0, 0);
        };
        let logical_threads = cpuinfo
            .lines()
            .filter(|line| line.starts_with("processor"))
            .count();
        let mut physical_cores = HashSet::new();
        for block in cpuinfo.split("\n\n") {
            let physical_id = block.lines().find_map(|line| {
                line.strip_prefix("physical id")
                    .and_then(|value| value.split_once(':'))
                    .map(|(_, value)| value.trim())
            });
            let core_id = block.lines().find_map(|line| {
                line.strip_prefix("core id")
                    .and_then(|value| value.split_once(':'))
                    .map(|(_, value)| value.trim())
            });
            if let (Some(physical_id), Some(core_id)) = (physical_id, core_id) {
                physical_cores.insert((physical_id.to_string(), core_id.to_string()));
            }
        }
        (physical_cores.len(), logical_threads)
    }

    fn ram_bytes() -> u64 {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|meminfo| {
                meminfo.lines().find_map(|line| {
                    let value = line.strip_prefix("MemTotal:")?;
                    value.split_whitespace().next()?.parse::<u64>().ok()
                })
            })
            .and_then(|kibibytes| kibibytes.checked_mul(1024))
            .unwrap_or(0)
    }

    fn numa_node_count() -> usize {
        std::fs::read_dir("/sys/devices/system/node")
            .ok()
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .filter(|entry| {
                entry.file_name().to_str().is_some_and(|name| {
                    name.strip_prefix("node").is_some_and(|suffix| {
                        !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
                    })
                })
            })
            .count()
    }

    fn runtime_isa_features() -> String {
        format!(
            "sse2={},sse4_2={},avx2={},fma={},bmi2={},vaes={},avx512f={}",
            std::is_x86_feature_detected!("sse2"),
            std::is_x86_feature_detected!("sse4.2"),
            std::is_x86_feature_detected!("avx2"),
            std::is_x86_feature_detected!("fma"),
            std::is_x86_feature_detected!("bmi2"),
            std::is_x86_feature_detected!("vaes"),
            std::is_x86_feature_detected!("avx512f")
        )
    }

    fn process_peak_rss_kib() -> usize {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|status| {
                status.lines().find_map(|line| {
                    let value = line.strip_prefix("VmHWM:")?;
                    value.split_whitespace().next()?.parse::<usize>().ok()
                })
            })
            .unwrap_or(0)
    }

    fn lotka_initial_states(batch: usize) -> Vec<Vec<f64>> {
        let mut state = 99u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            1.0 + 4.0 * ((state >> 11) as f64 / (1u64 << 53) as f64)
        };
        (0..batch).map(|_| vec![next(), next()]).collect()
    }

    fn lotka_t_eval() -> Vec<f64> {
        (0..LOTKA_SAMPLES)
            .map(|index| index as f64 * LOTKA_T_END / (LOTKA_SAMPLES - 1) as f64)
            .collect()
    }

    fn lotka_rhs(_t: f64, y: &[f64]) -> Vec<f64> {
        let (a, b, c, d) = (1.5_f64, 1.0, 3.0, 1.0);
        vec![a * y[0] - b * y[0] * y[1], -c * y[1] + d * y[0] * y[1]]
    }

    fn lotka_input_sha256(rows: &[Vec<f64>]) -> String {
        let mut hasher = Sha256::new();
        for row in rows {
            for value in row {
                hasher.update(value.to_le_bytes());
            }
        }
        format!("{:x}", hasher.finalize())
    }

    fn lotka_invariant(y: &[f64]) -> f64 {
        y[0] - 3.0 * y[0].ln() + y[1] - 1.5 * y[1].ln()
    }

    fn solve_ours_many(rows: &[Vec<f64>], t_eval: Option<&[f64]>) -> Vec<SolveIvpResult> {
        let template = SolveIvpOptions {
            t_span: (0.0, LOTKA_T_END),
            y0: &[0.0, 0.0],
            method: SolverKind::Rk45,
            t_eval,
            rtol: LOTKA_RTOL,
            atol: ToleranceValue::Scalar(LOTKA_ATOL),
            mode: RuntimeMode::Strict,
            ..Default::default()
        };
        solve_ivp_many(lotka_rhs, rows, &template)
            .into_iter()
            .map(|result| result.expect("FrankenSciPy Lotka-Volterra ensemble member"))
            .collect()
    }

    fn observe_lotka_worker_threads(rows: &[Vec<f64>], t_eval: Option<&[f64]>) -> usize {
        let template = SolveIvpOptions {
            t_span: (0.0, LOTKA_T_END),
            y0: &[0.0, 0.0],
            method: SolverKind::Rk45,
            t_eval,
            rtol: LOTKA_RTOL,
            atol: ToleranceValue::Scalar(LOTKA_ATOL),
            mode: RuntimeMode::Strict,
            ..Default::default()
        };
        let observed = Mutex::new(HashSet::new());
        let results = solve_ivp_many(
            |t, y| {
                observed
                    .lock()
                    .expect("Lotka worker observation mutex poisoned")
                    .insert(std::thread::current().id());
                lotka_rhs(t, y)
            },
            rows,
            &template,
        );
        if results.len() != rows.len() || results.iter().any(Result::is_err) {
            eprintln!("ABORT: worker-observation solve failed conformance");
            std::process::exit(5);
        }
        observed
            .into_inner()
            .expect("Lotka worker observation mutex poisoned")
            .len()
    }

    fn time_ours_many(rows: &[Vec<f64>], t_eval: Option<&[f64]>, reps: usize) -> f64 {
        let mut result = None;
        let start = Instant::now();
        for _ in 0..reps {
            result = Some(black_box(solve_ours_many(
                black_box(rows),
                black_box(t_eval),
            )));
        }
        let elapsed = start.elapsed().as_secs_f64();
        black_box(result);
        elapsed
    }

    fn time_scipy_many(sp: &mut Scipy, batch: usize, reps: usize, sampled: bool) -> f64 {
        match sp.time_many(batch, reps, sampled) {
            Ok(elapsed) => elapsed,
            Err(error) => {
                eprintln!("ABORT: timed SciPy ensemble failed: {error}");
                std::process::exit(9);
            }
        }
    }

    fn incumbent_many_pair(
        sp: &mut Scipy,
        batch: usize,
        rows: &[Vec<f64>],
        t_eval: Option<&[f64]>,
        sampled: bool,
        reps: usize,
        round: usize,
    ) -> (f64, f64) {
        if round.is_multiple_of(2) {
            let ours = time_ours_many(rows, t_eval, reps);
            let scipy = time_scipy_many(sp, batch, reps, sampled);
            (ours, scipy)
        } else {
            let scipy = time_scipy_many(sp, batch, reps, sampled);
            let ours = time_ours_many(rows, t_eval, reps);
            (ours, scipy)
        }
    }

    fn ours_many_null_pair(
        rows: &[Vec<f64>],
        t_eval: Option<&[f64]>,
        reps: usize,
        round: usize,
    ) -> f64 {
        let (a, b) = if round.is_multiple_of(2) {
            (
                time_ours_many(rows, t_eval, reps),
                time_ours_many(rows, t_eval, reps),
            )
        } else {
            let b = time_ours_many(rows, t_eval, reps);
            let a = time_ours_many(rows, t_eval, reps);
            (a, b)
        };
        a / b
    }

    fn scipy_many_null_pair(
        sp: &mut Scipy,
        batch: usize,
        reps: usize,
        sampled: bool,
        round: usize,
    ) -> f64 {
        let (a, b) = if round.is_multiple_of(2) {
            (
                time_scipy_many(sp, batch, reps, sampled),
                time_scipy_many(sp, batch, reps, sampled),
            )
        } else {
            let b = time_scipy_many(sp, batch, reps, sampled);
            let a = time_scipy_many(sp, batch, reps, sampled);
            (a, b)
        };
        a / b
    }

    struct DecayJobResult {
        solutions: Vec<SolveIvpResult>,
        exposures: Vec<f64>,
        terminal_masses: Vec<f64>,
    }

    struct DecayTimingConfig<'a> {
        n: usize,
        scenarios: usize,
        workers: usize,
        reps: usize,
        jacobian_mode: &'a str,
    }

    fn decay_rates(n: usize) -> Vec<f64> {
        (0..n).map(|index| 1.0 + 10.0 * index as f64).collect()
    }

    fn decay_initial_states(n: usize, scenarios: usize) -> Vec<Vec<f64>> {
        (0..scenarios)
            .map(|scenario| {
                let dose_scale = 1.0 + scenario as f64 / 32.0;
                (0..n)
                    .map(|index| (1.0 + 0.25 * (index % 7) as f64) * dose_scale)
                    .collect()
            })
            .collect()
    }

    fn decay_t_eval() -> Vec<f64> {
        (0..DECAY_JOB_SAMPLES)
            .map(|index| index as f64 / (DECAY_JOB_SAMPLES - 1) as f64)
            .collect()
    }

    fn decay_input_sha256(rates: &[f64], t_eval: &[f64], rows: &[Vec<f64>]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(DECAY_QUADRATIC.to_le_bytes());
        for value in rates.iter().chain(t_eval) {
            hasher.update(value.to_le_bytes());
        }
        for row in rows {
            for value in row {
                hasher.update(value.to_le_bytes());
            }
        }
        format!("{:x}", hasher.finalize())
    }

    fn decay_rhs(rates: &[f64], y: &[f64]) -> Vec<f64> {
        rates
            .iter()
            .zip(y)
            .map(|(rate, value)| -rate * value * (1.0 + DECAY_QUADRATIC * value))
            .collect()
    }

    fn decay_analytic(y0: f64, rate: f64, t: f64) -> f64 {
        let exponential = (-rate * t).exp();
        y0 * exponential / (1.0 + DECAY_QUADRATIC * y0 * (1.0 - exponential))
    }

    fn decay_postprocess(solutions: &[SolveIvpResult], n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut exposures = Vec::with_capacity(solutions.len() * n);
        let mut terminal_masses = Vec::with_capacity(solutions.len());
        for solution in solutions {
            for component in 0..n {
                let exposure = solution
                    .t
                    .windows(2)
                    .zip(solution.y.windows(2))
                    .map(|(times, states)| {
                        0.5 * (times[1] - times[0]) * (states[0][component] + states[1][component])
                    })
                    .sum();
                exposures.push(exposure);
            }
            terminal_masses.push(
                solution
                    .y
                    .last()
                    .expect("completed decay scenario has a final state")
                    .iter()
                    .sum(),
            );
        }
        (exposures, terminal_masses)
    }

    fn run_ours_decay_job(n: usize, scenarios: usize) -> DecayJobResult {
        let rates = decay_rates(n);
        let rows = decay_initial_states(n, scenarios);
        let t_eval = decay_t_eval();
        let template = SolveIvpOptions {
            t_span: (0.0, BDF_T_END),
            y0: &[],
            method: SolverKind::Bdf,
            t_eval: Some(&t_eval),
            rtol: BDF_RTOL,
            atol: ToleranceValue::Scalar(BDF_ATOL),
            mode: RuntimeMode::Strict,
            ..Default::default()
        };
        BDF_FORCE_DENSE_NEWTON.store(false, Ordering::Relaxed);
        let solutions = solve_ivp_many(|_t, y| decay_rhs(&rates, y), &rows, &template)
            .into_iter()
            .map(|result| result.expect("FrankenSciPy decay-screen scenario"))
            .collect::<Vec<_>>();
        let (exposures, terminal_masses) = decay_postprocess(&solutions, n);
        DecayJobResult {
            solutions,
            exposures,
            terminal_masses,
        }
    }

    fn valid_decay_job(result: &DecayJobResult, n: usize, scenarios: usize) -> bool {
        result.solutions.len() == scenarios
            && result.exposures.len() == scenarios * n
            && result.terminal_masses.len() == scenarios
            && result
                .exposures
                .iter()
                .chain(&result.terminal_masses)
                .all(|value| value.is_finite())
            && result.solutions.iter().all(|solution| {
                solution.success
                    && solution.status == 0
                    && solution.t.len() == DECAY_JOB_SAMPLES
                    && solution.y.len() == DECAY_JOB_SAMPLES
                    && solution
                        .t
                        .last()
                        .is_some_and(|value| value.to_bits() == BDF_T_END.to_bits())
                    && solution.y.iter().all(|state| {
                        state.len() == n && state.iter().all(|value| value.is_finite())
                    })
            })
    }

    fn flatten_decay_values(result: &DecayJobResult) -> Vec<f64> {
        result
            .solutions
            .iter()
            .flat_map(|solution| solution.y.iter().flatten().copied())
            .collect()
    }

    fn observe_decay_worker_threads(n: usize, scenarios: usize) -> usize {
        let rates = decay_rates(n);
        let rows = decay_initial_states(n, scenarios);
        let t_eval = decay_t_eval();
        let template = SolveIvpOptions {
            t_span: (0.0, BDF_T_END),
            y0: &[],
            method: SolverKind::Bdf,
            t_eval: Some(&t_eval),
            rtol: BDF_RTOL,
            atol: ToleranceValue::Scalar(BDF_ATOL),
            mode: RuntimeMode::Strict,
            ..Default::default()
        };
        let observed = Mutex::new(HashSet::new());
        let results = solve_ivp_many(
            |_t, y| {
                observed
                    .lock()
                    .expect("decay worker observation mutex poisoned")
                    .insert(std::thread::current().id());
                decay_rhs(&rates, y)
            },
            &rows,
            &template,
        );
        if results.len() != scenarios || results.iter().any(Result::is_err) {
            eprintln!("ABORT: decay worker-observation job failed");
            std::process::exit(5);
        }
        observed
            .into_inner()
            .expect("decay worker observation mutex poisoned")
            .len()
    }

    fn time_ours_decay_job(n: usize, scenarios: usize, reps: usize) -> f64 {
        let mut result = None;
        let start = Instant::now();
        for _ in 0..reps {
            result = Some(black_box(run_ours_decay_job(
                black_box(n),
                black_box(scenarios),
            )));
        }
        let elapsed = start.elapsed().as_secs_f64();
        let result = result.expect("at least one timed decay-screen repetition");
        if !valid_decay_job(&result, n, scenarios) {
            eprintln!("ABORT: timed FrankenSciPy decay-screen job was incomplete");
            std::process::exit(9);
        }
        black_box(result);
        elapsed
    }

    fn time_scipy_decay_job(
        sp: &mut Scipy,
        n: usize,
        scenarios: usize,
        workers: usize,
        reps: usize,
        jacobian_mode: &str,
    ) -> f64 {
        match sp.time_decay(n, scenarios, workers, reps, jacobian_mode) {
            Ok(elapsed) => elapsed,
            Err(error) => {
                eprintln!("ABORT: timed SciPy decay-screen job failed: {error}");
                std::process::exit(9);
            }
        }
    }

    fn incumbent_decay_pair(
        sp: &mut Scipy,
        config: &DecayTimingConfig<'_>,
        round: usize,
    ) -> (f64, f64) {
        if round.is_multiple_of(2) {
            let ours = time_ours_decay_job(config.n, config.scenarios, config.reps);
            let scipy = time_scipy_decay_job(
                sp,
                config.n,
                config.scenarios,
                config.workers,
                config.reps,
                config.jacobian_mode,
            );
            (ours, scipy)
        } else {
            let scipy = time_scipy_decay_job(
                sp,
                config.n,
                config.scenarios,
                config.workers,
                config.reps,
                config.jacobian_mode,
            );
            let ours = time_ours_decay_job(config.n, config.scenarios, config.reps);
            (ours, scipy)
        }
    }

    fn ours_decay_null_pair(config: &DecayTimingConfig<'_>, round: usize) -> f64 {
        let (a, b) = if round.is_multiple_of(2) {
            (
                time_ours_decay_job(config.n, config.scenarios, config.reps),
                time_ours_decay_job(config.n, config.scenarios, config.reps),
            )
        } else {
            let b = time_ours_decay_job(config.n, config.scenarios, config.reps);
            let a = time_ours_decay_job(config.n, config.scenarios, config.reps);
            (a, b)
        };
        a / b
    }

    fn scipy_decay_null_pair(sp: &mut Scipy, config: &DecayTimingConfig<'_>, round: usize) -> f64 {
        let (a, b) = if round.is_multiple_of(2) {
            (
                time_scipy_decay_job(
                    sp,
                    config.n,
                    config.scenarios,
                    config.workers,
                    config.reps,
                    config.jacobian_mode,
                ),
                time_scipy_decay_job(
                    sp,
                    config.n,
                    config.scenarios,
                    config.workers,
                    config.reps,
                    config.jacobian_mode,
                ),
            )
        } else {
            let b = time_scipy_decay_job(
                sp,
                config.n,
                config.scenarios,
                config.workers,
                config.reps,
                config.jacobian_mode,
            );
            let a = time_scipy_decay_job(
                sp,
                config.n,
                config.scenarios,
                config.workers,
                config.reps,
                config.jacobian_mode,
            );
            (a, b)
        };
        a / b
    }

    fn ready_value<'a>(ready: &'a str, key: &str) -> Option<&'a str> {
        ready
            .split_whitespace()
            .find_map(|field| field.strip_prefix(key))
    }

    fn start_genuine_scipy(script: &str) -> (Scipy, String, String) {
        let (sp, ready) = match Scipy::start(script) {
            Ok(value) => value,
            Err(error) => {
                eprintln!("ABORT: cannot start SciPy arm: {error}");
                std::process::exit(3);
            }
        };
        println!("scipy_arm: {ready}");
        if !ready.starts_with("READY scipy=")
            || !ready.contains("solve_ivp_mod=scipy.integrate._ivp.ivp")
            || !ready.contains("fsci_loaded=False")
            || !ready.contains("genuine=True")
            || !ready.contains("actual_observed_worker_threads=")
            || !ready.contains("scipy_engine_sha256=")
        {
            eprintln!("ABORT: SciPy arm is not genuine (dispatch trap)");
            std::process::exit(4);
        }
        let version = ready_value(&ready, "scipy=")
            .expect("READY line has scipy version")
            .to_string();
        (sp, ready, version)
    }

    fn run_lotka_many(
        script: &str,
        batch: usize,
        rounds: usize,
        reps: usize,
        affinity: &str,
        sampled: bool,
    ) {
        let parallelism = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1);
        let requested_fsci_threads = if batch < 4 { 1 } else { parallelism.min(batch) };
        let fixture = if sampled {
            "lotka-volterra-ensemble"
        } else {
            "lotka-volterra-completion-ensemble"
        };
        let t_eval_label = if sampled { "150" } else { "None" };
        println!(
            "fixture={fixture} batch={batch} rounds={rounds} reps={reps} \
             method=RK45 t_span=[0,{LOTKA_T_END}] rtol={LOTKA_RTOL} \
             atol={LOTKA_ATOL} t_eval={t_eval_label} affinity={affinity} \
             requested_threads={requested_fsci_threads} available_parallelism={parallelism} \
             scipy_rhs_counter_outside_timing=true"
        );

        let rows = lotka_initial_states(batch);
        let t_eval_values = lotka_t_eval();
        let t_eval = sampled.then_some(t_eval_values.as_slice());
        let actual_fsci_workers = observe_lotka_worker_threads(&rows, t_eval);

        let (mut sp, ready, scipy_version) = start_genuine_scipy(script);
        let actual_scipy_workers = ready_value(&ready, "actual_observed_worker_threads=")
            .and_then(|value| value.parse::<usize>().ok())
            .expect("READY line has numeric actual observed SciPy workers");
        let scipy_engine_sha256 = ready_value(&ready, "scipy_engine_sha256=")
            .expect("READY line has SciPy engine SHA-256");
        println!(
            "thread_provenance: requested_scipy_threads=1 \
             actual_observed_scipy_worker_threads={actual_scipy_workers} \
             requested_frankenscipy_threads={requested_fsci_threads} \
             actual_observed_frankenscipy_worker_threads={actual_fsci_workers} \
             cpuset_logical_cap={parallelism} python_blas_thread_cap=1 \
             worker_observation_outside_timing=true"
        );
        println!("scipy_engine_sha256={scipy_engine_sha256}");
        println!(
            "Legacy incumbent arm: SciPy {scipy_version}; side-by-side same-invocation; \
             child-side full-ensemble timing"
        );

        let input_sha256 = lotka_input_sha256(&rows);
        let ours = solve_ours_many(&rows, t_eval);
        let theirs = match sp.check_many(batch, sampled) {
            Ok(result) => result,
            Err(error) => {
                eprintln!("ABORT: SciPy ensemble parity check failed: {error}");
                std::process::exit(5);
            }
        };
        let compared_samples = if sampled { LOTKA_SAMPLES } else { 1 };
        let expected_values = batch * compared_samples * 2;
        if ours.len() != batch
            || ours.iter().any(|result| {
                !result.success
                    || result.status != 0
                    || result.t.is_empty()
                    || result.y.is_empty()
                    || result.t.len() != result.y.len()
                    || result
                        .t
                        .last()
                        .is_none_or(|value| value.to_bits() != LOTKA_T_END.to_bits())
                    || (sampled && result.t.len() != LOTKA_SAMPLES)
                    || result.y.iter().any(|state| state.len() != 2)
            })
            || theirs.successes != batch
            || theirs.samples != compared_samples
            || theirs.stored_points < batch * compared_samples
            || theirs.values.len() != expected_values
            || theirs.input_sha256 != input_sha256
            || !theirs.min_component.is_finite()
            || !theirs.max_invariant_drift.is_finite()
        {
            eprintln!(
                "ABORT: incomplete or mismatched ensemble proof \
                 (ours={}/{batch}, scipy={}/{batch}, scipy_samples={}, \
                 scipy_points={}, scipy_values={}/{expected_values}, \
                 input_sha_match={})",
                ours.len(),
                theirs.successes,
                theirs.samples,
                theirs.stored_points,
                theirs.values.len(),
                theirs.input_sha256 == input_sha256
            );
            std::process::exit(6);
        }

        let mut max_abs_diff = 0.0f64;
        let mut max_scaled_diff = 0.0f64;
        let mut max_final_scaled_diff = 0.0f64;
        let mut max_ours_invariant_drift = 0.0f64;
        let mut worst = (0usize, 0usize, 0usize, 0.0f64, 0.0f64);
        let mut positive_components = 0usize;
        let mut ours_full_history_positive = true;
        for (row_index, result) in ours.iter().enumerate() {
            let initial_invariant = lotka_invariant(&rows[row_index]);
            for state in &result.y {
                ours_full_history_positive &=
                    state.iter().all(|value| value.is_finite() && *value > 0.0);
                max_ours_invariant_drift = max_ours_invariant_drift
                    .max((lotka_invariant(state) - initial_invariant).abs());
            }
            let compared_ours: &[Vec<f64>] = if sampled {
                &result.y
            } else {
                std::slice::from_ref(
                    result
                        .y
                        .last()
                        .expect("completed ensemble member has final state"),
                )
            };
            for (sample_index, our_state) in compared_ours.iter().enumerate() {
                let offset = (row_index * compared_samples + sample_index) * 2;
                let scipy_state = &theirs.values[offset..offset + 2];
                if our_state
                    .iter()
                    .all(|value| value.is_finite() && *value > 0.0)
                    && scipy_state
                        .iter()
                        .all(|value| value.is_finite() && *value > 0.0)
                {
                    positive_components += 4;
                }
                for component in 0..2 {
                    let difference = (our_state[component] - scipy_state[component]).abs();
                    let scale = LOTKA_ATOL
                        + LOTKA_RTOL * our_state[component].abs().max(scipy_state[component].abs());
                    max_abs_diff = max_abs_diff.max(difference);
                    let scaled_difference = difference / scale;
                    if scaled_difference > max_scaled_diff {
                        max_scaled_diff = scaled_difference;
                        worst = (
                            row_index,
                            sample_index,
                            component,
                            our_state[component],
                            scipy_state[component],
                        );
                    }
                    if sample_index + 1 == compared_samples {
                        max_final_scaled_diff = max_final_scaled_diff.max(scaled_difference);
                    }
                }
            }
        }
        let expected_positive_components = expected_values * 2;
        println!(
            "conformance_diagnostic: max_abs_diff={max_abs_diff:.3e} \
             max_scaled_diff={max_scaled_diff:.3} \
             max_final_scaled_diff={max_final_scaled_diff:.3} \
             worst_row={} worst_sample={} worst_component={} \
             worst_ours={:.17e} worst_scipy={:.17e} \
             max_invariant_drift_ours={max_ours_invariant_drift:.3e} \
             max_invariant_drift_scipy={:.3e}",
            worst.0, worst.1, worst.2, worst.3, worst.4, theirs.max_invariant_drift
        );
        if positive_components != expected_positive_components
            || !ours_full_history_positive
            || theirs.min_component <= 0.0
            || !max_scaled_diff.is_finite()
            || max_scaled_diff > 100.0
        {
            eprintln!(
                "ABORT: ensemble violates positivity or differential tolerance \
                 (positive_components={positive_components}/{expected_positive_components}, \
                 max_scaled_diff={max_scaled_diff})"
            );
            std::process::exit(7);
        }

        let ours_nfev = ours.iter().map(|result| result.nfev).sum::<usize>();
        let ours_njev = ours.iter().map(|result| result.njev).sum::<usize>();
        let ours_nlu = ours.iter().map(|result| result.nlu).sum::<usize>();
        let ours_stored_points = ours.iter().map(|result| result.t.len()).sum::<usize>();
        println!(
            "agreement: trajectories={batch}/{batch} compared_samples={compared_samples} \
             compared_components={expected_values}/{expected_values} \
             input_sha256={input_sha256} max_abs_diff={max_abs_diff:.3e} \
             max_scaled_diff={max_scaled_diff:.3}"
        );
        println!(
            "scientific_job: all_trajectories_reached_t_end=true \
             full_histories_positive_finite=true \
             compared_positive_components={positive_components}/{expected_positive_components} \
             max_invariant_drift_ours={max_ours_invariant_drift:.3e} \
             max_invariant_drift_scipy={:.3e}",
            theirs.max_invariant_drift
        );
        println!(
            "counters: ours total_nfev={ours_nfev} total_njev={ours_njev} \
             total_nlu={ours_nlu} stored_points={} | scipy total_nfev={} \
             total_njev={} total_nlu={} stored_points={} actual_rhs_calls={}",
            ours_stored_points,
            theirs.nfev,
            theirs.njev,
            theirs.nlu,
            theirs.stored_points,
            theirs.rhs_calls
        );

        let callback_repeats = if batch >= 64 { 4usize } else { 32usize };
        let rhs_secs = sp
            .many_rhs_cost(theirs.rhs_calls.saturating_mul(callback_repeats))
            .unwrap_or(f64::NAN)
            / callback_repeats as f64;
        if !rhs_secs.is_finite() || rhs_secs < 0.0 {
            eprintln!("ABORT: invalid Python ensemble RHS decomposition timing");
            std::process::exit(8);
        }

        let (mut ours_t, mut scipy_t, mut ratios) = (Vec::new(), Vec::new(), Vec::new());
        let (mut null_ours, mut null_scipy) = (Vec::new(), Vec::new());
        for round in 0..rounds {
            let (ours_secs, scipy_secs, ours_null, scipy_null) = match round % 3 {
                0 => {
                    let incumbent =
                        incumbent_many_pair(&mut sp, batch, &rows, t_eval, sampled, reps, round);
                    let ours_null = ours_many_null_pair(&rows, t_eval, reps, round);
                    let scipy_null = scipy_many_null_pair(&mut sp, batch, reps, sampled, round);
                    (incumbent.0, incumbent.1, ours_null, scipy_null)
                }
                1 => {
                    let scipy_null = scipy_many_null_pair(&mut sp, batch, reps, sampled, round);
                    let incumbent =
                        incumbent_many_pair(&mut sp, batch, &rows, t_eval, sampled, reps, round);
                    let ours_null = ours_many_null_pair(&rows, t_eval, reps, round);
                    (incumbent.0, incumbent.1, ours_null, scipy_null)
                }
                _ => {
                    let ours_null = ours_many_null_pair(&rows, t_eval, reps, round);
                    let scipy_null = scipy_many_null_pair(&mut sp, batch, reps, sampled, round);
                    let incumbent =
                        incumbent_many_pair(&mut sp, batch, &rows, t_eval, sampled, reps, round);
                    (incumbent.0, incumbent.1, ours_null, scipy_null)
                }
            };
            ours_t.push(ours_secs);
            scipy_t.push(scipy_secs);
            ratios.push(scipy_secs / ours_secs);
            null_ours.push(ours_null);
            null_scipy.push(scipy_null);
        }

        let (ratio_low, ratio_high) = boot_ci(&ratios);
        let (ours_null_low, ours_null_high) = boot_ci(&null_ours);
        let (scipy_null_low, scipy_null_high) = boot_ci(&null_scipy);
        let p50_ours = median(ours_t.clone());
        let p50_scipy = median(scipy_t.clone());
        println!(
            "OURS   p50={:.6}ms/batch {:.6}us/trajectory  \
             SCIPY p50={:.6}ms/batch {:.6}ms/trajectory",
            p50_ours * 1e3 / reps as f64,
            p50_ours * 1e6 / (reps * batch) as f64,
            p50_scipy * 1e3 / reps as f64,
            p50_scipy * 1e3 / (reps * batch) as f64
        );
        println!(
            "NULL-ours   median={:.6} ci95=[{ours_null_low:.6},{ours_null_high:.6}] \
             cv={:.3}% (provenance only)",
            median(null_ours.clone()),
            cv(&null_ours) * 100.0
        );
        println!(
            "NULL-scipy  median={:.6} ci95=[{scipy_null_low:.6},{scipy_null_high:.6}] \
             cv={:.3}% (provenance only)",
            median(null_scipy.clone()),
            cv(&null_scipy) * 100.0
        );
        println!(
            "raw_samples_seconds: ours={ours_t:?} scipy={scipy_t:?} \
             ratios={ratios:?} null_ours={null_ours:?} null_scipy={null_scipy:?}"
        );
        let ratio_p50 = median(ratios.clone());
        println!(
            "Incumbent ratio: SciPy / FrankenSciPy = {ratio_p50:.4}x \
             (bootstrap-median ci95=[{ratio_low:.4},{ratio_high:.4}], \
             cv={:.3}% provenance only)",
            cv(&ratios) * 100.0
        );

        let ours_null_median = median(null_ours);
        let scipy_null_median = median(null_scipy);
        let null_half_width = ((ours_null_high - ours_null_low) / 2.0)
            .max((scipy_null_high - scipy_null_low) / 2.0)
            .max(0.0);
        let null_edge = ours_null_high
            .max(scipy_null_high)
            .max(1.0 / ours_null_low.max(1e-9))
            .max(1.0 / scipy_null_low.max(1e-9))
            .max(1.0);
        let c1_effect_ci_excludes_one = ratio_low > 1.0 || ratio_high < 1.0;
        let effect_deviation = if ratio_low > 1.0 {
            ratio_low - 1.0
        } else if ratio_high < 1.0 {
            1.0 - ratio_high
        } else {
            0.0
        };
        let c2_beats_half_width_margin = effect_deviation > 2.0 * null_half_width;
        let c2b_beats_endpoint_margin = effect_deviation > 2.0 * (null_edge - 1.0);
        let c3_null_medians_unbiased = (ours_null_median - 1.0).abs() <= NULL_MEDIAN_BIAS_LIMIT
            && (scipy_null_median - 1.0).abs() <= NULL_MEDIAN_BIAS_LIMIT;
        let decidable = c1_effect_ci_excludes_one
            && c2_beats_half_width_margin
            && c2b_beats_endpoint_margin
            && c3_null_medians_unbiased;
        let required = 1.0 + 2.0 * (null_edge - 1.0);
        let outcome = if decidable && ratio_low > 1.0 {
            "DECIDED FRANKENSCIPY WIN"
        } else if decidable && ratio_high < 1.0 {
            "DECIDED FRANKENSCIPY LOSS"
        } else {
            "NOT DECIDED"
        };
        println!(
            "corrected-null-gate: \
             c1_effect_ci_excludes_one={c1_effect_ci_excludes_one} \
             c2_beats_2x_half_width={c2_beats_half_width_margin} \
             c2b_beats_2x_endpoint={c2b_beats_endpoint_margin} \
             c3_null_medians_within_2pct={c3_null_medians_unbiased} \
             decidable={decidable} effect_deviation={effect_deviation:.6} \
             null_half_width={null_half_width:.6} \
             required_c2={:.6} required_c2b={:.6} \
             ours_null_median={ours_null_median:.6} \
             scipy_null_median={scipy_null_median:.6} \
             null_ci_veto=disabled_telemetry_only",
            2.0 * null_half_width,
            2.0 * (null_edge - 1.0)
        );
        println!(
            "median-CI gate: worst_null_edge={null_edge:.4} required={required:.4} \
             ratio_ci=[{ratio_low:.4},{ratio_high:.4}] null_margin=2x \
             cv_used_for_decision=false => {outcome}"
        );

        let scipy_per_batch = p50_scipy / reps as f64;
        let ours_per_batch = p50_ours / reps as f64;
        println!(
            "decomposition: scipy {:.4}ms/batch, of which Python RHS callbacks \
             ({} calls) = {:.4}ms = {:.1}% ; callback-free sensitivity ratio = {:.4}x",
            scipy_per_batch * 1e3,
            theirs.rhs_calls,
            rhs_secs * 1e3,
            rhs_secs / scipy_per_batch * 100.0,
            (scipy_per_batch - rhs_secs).max(0.0) / ours_per_batch
        );
        sp.quit();
    }

    fn run_decay_screen(script: &str, n: usize, rounds: usize, reps: usize, affinity: &str) {
        let parallelism = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1);
        let affinity_count = affinity_cpus(affinity).map(|cpus| cpus.len()).unwrap_or(0);
        let fsci_workers = DECAY_JOB_MAX_WORKERS
            .min(DECAY_JOB_SCENARIOS)
            .min(parallelism);
        if fsci_workers != DECAY_JOB_MAX_WORKERS
            || parallelism != DECAY_JOB_MAX_WORKERS
            || affinity_count != DECAY_JOB_MAX_WORKERS
        {
            eprintln!(
                "ABORT: decay-screen is an eight-CPU shape cell; require exactly \
                 {DECAY_JOB_MAX_WORKERS} affinity CPUs and available threads \
                 (affinity={affinity}, affinity_count={affinity_count}, \
                 available_parallelism={parallelism})"
            );
            std::process::exit(2);
        }
        println!(
            "fixture=stiff-independent-reaction-screen species={n} \
             scenarios={DECAY_JOB_SCENARIOS} rounds={rounds} reps={reps} \
             method=BDF t_span=[0,{BDF_T_END}] rtol={BDF_RTOL} atol={BDF_ATOL} \
             t_eval={DECAY_JOB_SAMPLES} jacobian_structure=exactly-diagonal \
             model=first-plus-second-order-self-decay quadratic={DECAY_QUADRATIC} \
             requested_frankenscipy_threads={fsci_workers} affinity={affinity}"
        );
        println!(
            "whole_job_boundary: INCLUDED=model_and_input_construction,pool_or_thread_lifecycle,\
             {DECAY_JOB_SCENARIOS}_solves,{DECAY_JOB_SAMPLES}-point_output_materialization,\
             trapezoidal_exposure_auc,terminal_mass; \
             EXCLUDED=python_interpreter_startup,scipy_import,pipe_transport,\
             parity_serialization,provenance_collection"
        );
        println!(
            "work_units: scenario_solves={DECAY_JOB_SCENARIOS} state_components_per_scenario={n} \
             requested_points_per_scenario={DECAY_JOB_SAMPLES} \
             materialized_state_values={} exposure_values={} terminal_summaries={DECAY_JOB_SCENARIOS}",
            DECAY_JOB_SCENARIOS * DECAY_JOB_SAMPLES * n,
            DECAY_JOB_SCENARIOS * n
        );

        let rates = decay_rates(n);
        let rows = decay_initial_states(n, DECAY_JOB_SCENARIOS);
        let t_eval = decay_t_eval();
        let input_sha256 = decay_input_sha256(&rates, &t_eval, &rows);
        let actual_fsci_workers = observe_decay_worker_threads(n, DECAY_JOB_SCENARIOS);
        if actual_fsci_workers != fsci_workers {
            eprintln!(
                "ABORT: FrankenSciPy used {actual_fsci_workers} compute threads; \
                 expected {fsci_workers}"
            );
            std::process::exit(5);
        }

        let (mut sp, ready, scipy_version) = start_genuine_scipy(script);
        if scipy_version != "1.17.1" {
            eprintln!("ABORT: this claim is version-pinned to SciPy 1.17.1, found {scipy_version}");
            std::process::exit(4);
        }
        let scipy_main_threads = ready_value(&ready, "actual_observed_worker_threads=")
            .and_then(|value| value.parse::<usize>().ok())
            .expect("READY line has numeric actual observed SciPy threads");
        let scipy_engine_sha256 = ready_value(&ready, "scipy_engine_sha256=")
            .expect("READY line has SciPy engine SHA-256");
        println!("scipy_engine_sha256={scipy_engine_sha256}");
        println!(
            "incumbent_backend_screen_contract: configurations=jac-none,\
             analytic-sparse-jacobian,sparsity-only; process_counts=1,2,4,8; \
             selection=lowest_live_whole_job_wall_time; screen_outside_headline_samples"
        );

        let mut screen = Vec::new();
        for jacobian_mode in ["none", "analytic-sparse", "sparsity-only"] {
            for workers in [1usize, 2, 4, DECAY_JOB_MAX_WORKERS] {
                let elapsed = match sp.time_decay(n, DECAY_JOB_SCENARIOS, workers, 1, jacobian_mode)
                {
                    Ok(value) => value,
                    Err(error) => {
                        eprintln!(
                            "ABORT: SciPy incumbent backend screen failed for \
                             jacobian_mode={jacobian_mode} workers={workers}: {error}"
                        );
                        std::process::exit(5);
                    }
                };
                println!(
                    "scipy_backend_screen: jacobian_mode={jacobian_mode} \
                     worker_processes={workers} whole_job_ms={:.6}",
                    elapsed * 1e3
                );
                screen.push((elapsed, jacobian_mode, workers));
            }
        }
        screen.sort_by(|left, right| left.0.total_cmp(&right.0));
        let (screen_elapsed, scipy_jacobian_mode, scipy_workers) = screen[0];
        println!(
            "selected_scipy_incumbent: jacobian_mode={scipy_jacobian_mode} \
             worker_processes={scipy_workers} screened_whole_job_ms={:.6} \
             reason=fastest_valid_live_SciPy_configuration",
            screen_elapsed * 1e3
        );

        BDF_DIAG_NEWTON_HITS.store(0, Ordering::Relaxed);
        BDF_BAND_NEWTON_HITS.store(0, Ordering::Relaxed);
        let ours = run_ours_decay_job(n, DECAY_JOB_SCENARIOS);
        let diag_hits = BDF_DIAG_NEWTON_HITS.load(Ordering::Relaxed);
        let band_hits = BDF_BAND_NEWTON_HITS.load(Ordering::Relaxed);
        let theirs =
            match sp.check_decay(n, DECAY_JOB_SCENARIOS, scipy_workers, scipy_jacobian_mode) {
                Ok(result) => result,
                Err(error) => {
                    eprintln!("ABORT: SciPy decay-screen parity check failed: {error}");
                    std::process::exit(5);
                }
            };
        let expected_values = DECAY_JOB_SCENARIOS * DECAY_JOB_SAMPLES * n;
        let expected_exposures = DECAY_JOB_SCENARIOS * n;
        let our_values = flatten_decay_values(&ours);
        if !valid_decay_job(&ours, n, DECAY_JOB_SCENARIOS)
            || diag_hits == 0
            || band_hits != 0
            || theirs.successes != DECAY_JOB_SCENARIOS
            || theirs.scenarios != DECAY_JOB_SCENARIOS
            || theirs.samples != DECAY_JOB_SAMPLES
            || theirs.stored_points != DECAY_JOB_SCENARIOS * DECAY_JOB_SAMPLES
            || theirs.input_sha256 != input_sha256
            || theirs.worker_processes != scipy_workers
            || theirs.worker_threads != scipy_workers
            || theirs.peak_rss_kib == 0
            || theirs.rhs_calls_by_scenario.len() != DECAY_JOB_SCENARIOS
            || theirs.rhs_calls_by_scenario.iter().sum::<usize>() != theirs.rhs_calls
            || our_values.len() != expected_values
            || theirs.values.len() != expected_values
            || theirs.exposures.len() != expected_exposures
            || theirs.terminal_masses.len() != DECAY_JOB_SCENARIOS
        {
            eprintln!(
                "ABORT: incomplete or mismatched decay-screen proof \
                 (ours_values={}/{expected_values}, scipy_values={}/{expected_values}, \
                 ours_scenarios={}, scipy_scenarios={}/{DECAY_JOB_SCENARIOS}, \
                 scipy_processes={}/{scipy_workers}, scipy_threads={}, \
                 diag_hits={diag_hits}, band_hits={band_hits}, input_sha_match={})",
                our_values.len(),
                theirs.values.len(),
                ours.solutions.len(),
                theirs.successes,
                theirs.worker_processes,
                theirs.worker_threads,
                theirs.input_sha256 == input_sha256
            );
            std::process::exit(6);
        }

        let mut max_abs_diff = 0.0_f64;
        let mut max_scaled_diff = 0.0_f64;
        let mut max_scaled_ours_analytic = 0.0_f64;
        let mut max_scaled_scipy_analytic = 0.0_f64;
        let mut worst = (0usize, 0usize, 0usize, 0.0_f64, 0.0_f64);
        for (index, (&our_value, &scipy_value)) in our_values.iter().zip(&theirs.values).enumerate()
        {
            let scenario_stride = DECAY_JOB_SAMPLES * n;
            let scenario = index / scenario_stride;
            let within_scenario = index % scenario_stride;
            let sample = within_scenario / n;
            let component = within_scenario % n;
            let difference = (our_value - scipy_value).abs();
            let scale = BDF_ATOL + BDF_RTOL * our_value.abs().max(scipy_value.abs());
            let scaled_difference = difference / scale;
            max_abs_diff = max_abs_diff.max(difference);
            if scaled_difference > max_scaled_diff {
                max_scaled_diff = scaled_difference;
                worst = (scenario, sample, component, our_value, scipy_value);
            }
            let analytic =
                decay_analytic(rows[scenario][component], rates[component], t_eval[sample]);
            let analytic_scale = BDF_ATOL + BDF_RTOL * analytic.abs();
            max_scaled_ours_analytic =
                max_scaled_ours_analytic.max((our_value - analytic).abs() / analytic_scale);
            max_scaled_scipy_analytic =
                max_scaled_scipy_analytic.max((scipy_value - analytic).abs() / analytic_scale);
        }

        let mut max_exposure_scaled_diff = 0.0_f64;
        for (&our_exposure, &scipy_exposure) in ours.exposures.iter().zip(&theirs.exposures) {
            let scale =
                BDF_ATOL * BDF_T_END + BDF_RTOL * our_exposure.abs().max(scipy_exposure.abs());
            max_exposure_scaled_diff =
                max_exposure_scaled_diff.max((our_exposure - scipy_exposure).abs() / scale);
        }
        let mut max_terminal_mass_scaled_diff = 0.0_f64;
        for (&our_mass, &scipy_mass) in ours.terminal_masses.iter().zip(&theirs.terminal_masses) {
            let scale = n as f64 * BDF_ATOL + BDF_RTOL * our_mass.abs().max(scipy_mass.abs());
            max_terminal_mass_scaled_diff =
                max_terminal_mass_scaled_diff.max((our_mass - scipy_mass).abs() / scale);
        }
        println!(
            "agreement: trajectories={DECAY_JOB_SCENARIOS}/{DECAY_JOB_SCENARIOS} \
             compared_state_values={expected_values}/{expected_values} \
             compared_exposures={expected_exposures}/{expected_exposures} \
             compared_terminal_masses={DECAY_JOB_SCENARIOS}/{DECAY_JOB_SCENARIOS} \
             input_sha256={input_sha256}"
        );
        println!(
            "conformance: max_abs_state_diff={max_abs_diff:.3e} \
             max_scaled_state_diff={max_scaled_diff:.3} \
             max_scaled_exposure_diff={max_exposure_scaled_diff:.3} \
             max_scaled_terminal_mass_diff={max_terminal_mass_scaled_diff:.3} \
             max_scaled_ours_vs_analytic={max_scaled_ours_analytic:.3} \
             max_scaled_scipy_vs_analytic={max_scaled_scipy_analytic:.3} \
             worst_scenario={} worst_sample={} worst_component={} \
             worst_ours={:.17e} worst_scipy={:.17e}",
            worst.0, worst.1, worst.2, worst.3, worst.4
        );
        if !max_scaled_diff.is_finite()
            || !max_exposure_scaled_diff.is_finite()
            || !max_terminal_mass_scaled_diff.is_finite()
            || !max_scaled_ours_analytic.is_finite()
            || !max_scaled_scipy_analytic.is_finite()
            || max_scaled_diff > 100.0
            || max_exposure_scaled_diff > 100.0
            || max_terminal_mass_scaled_diff > 100.0
            || max_scaled_ours_analytic > 100.0
            || max_scaled_scipy_analytic > 100.0
        {
            eprintln!("ABORT: decay-screen exceeds the 100-tolerance-unit conformance contract");
            std::process::exit(7);
        }

        let ours_nfev = ours
            .solutions
            .iter()
            .map(|solution| solution.nfev)
            .sum::<usize>();
        let ours_njev = ours
            .solutions
            .iter()
            .map(|solution| solution.njev)
            .sum::<usize>();
        let ours_nlu = ours
            .solutions
            .iter()
            .map(|solution| solution.nlu)
            .sum::<usize>();
        let ours_stored_points = ours
            .solutions
            .iter()
            .map(|solution| solution.t.len())
            .sum::<usize>();
        println!(
            "operation_counts: ours scenario_solves={DECAY_JOB_SCENARIOS} \
             total_nfev={ours_nfev} total_njev={ours_njev} total_nlu={ours_nlu} \
             stored_points={ours_stored_points} diag_hits={diag_hits} band_hits={band_hits} | \
             scipy scenario_solves={DECAY_JOB_SCENARIOS} total_nfev={} total_njev={} \
             total_nlu={} stored_points={} actual_rhs_calls={}",
            theirs.nfev, theirs.njev, theirs.nlu, theirs.stored_points, theirs.rhs_calls
        );
        println!(
            "thread_provenance: requested_frankenscipy_compute_threads={fsci_workers} \
             actual_observed_frankenscipy_compute_threads={actual_fsci_workers} \
             requested_scipy_worker_processes={scipy_workers} \
             actual_observed_scipy_worker_processes={} \
             actual_observed_scipy_compute_threads={} scipy_main_process_threads={scipy_main_threads} \
             threads_per_scipy_worker=1 python_blas_thread_cap=1 \
             cpuset_logical_cap={parallelism} worker_lifecycle_inside_timing=true",
            theirs.worker_processes, theirs.worker_threads
        );
        println!(
            "memory_provenance: frankenscipy_process_peak_rss_kib={} \
             scipy_process_tree_peak_rss_upper_bound_kib={} \
             scipy_metric=sum_of_per_process_high_water_marks",
            process_peak_rss_kib(),
            theirs.peak_rss_kib
        );

        let zero_calls = vec![0usize; DECAY_JOB_SCENARIOS];
        let mut rhs_replays = Vec::new();
        let mut setup_replays = Vec::new();
        for replay in 0usize..3 {
            let (rhs, setup) = if replay.is_multiple_of(2) {
                (
                    sp.decay_rhs_replay(n, scipy_workers, &theirs.rhs_calls_by_scenario),
                    sp.decay_rhs_replay(n, scipy_workers, &zero_calls),
                )
            } else {
                let setup = sp.decay_rhs_replay(n, scipy_workers, &zero_calls);
                let rhs = sp.decay_rhs_replay(n, scipy_workers, &theirs.rhs_calls_by_scenario);
                (rhs, setup)
            };
            let rhs = rhs.unwrap_or(f64::NAN);
            let setup = setup.unwrap_or(f64::NAN);
            if !rhs.is_finite() || !setup.is_finite() || rhs <= 0.0 || setup <= 0.0 {
                eprintln!("ABORT: invalid parallel Python RHS replay");
                std::process::exit(8);
            }
            rhs_replays.push(rhs);
            setup_replays.push(setup);
        }

        let timing = DecayTimingConfig {
            n,
            scenarios: DECAY_JOB_SCENARIOS,
            workers: scipy_workers,
            reps,
            jacobian_mode: scipy_jacobian_mode,
        };
        let (mut ours_t, mut scipy_t, mut ratios) = (Vec::new(), Vec::new(), Vec::new());
        let (mut null_ours, mut null_scipy) = (Vec::new(), Vec::new());
        for round in 0..rounds {
            let (ours_secs, scipy_secs, ours_null, scipy_null) = match round % 3 {
                0 => {
                    let incumbent = incumbent_decay_pair(&mut sp, &timing, round);
                    let ours_null = ours_decay_null_pair(&timing, round);
                    let scipy_null = scipy_decay_null_pair(&mut sp, &timing, round);
                    (incumbent.0, incumbent.1, ours_null, scipy_null)
                }
                1 => {
                    let scipy_null = scipy_decay_null_pair(&mut sp, &timing, round);
                    let incumbent = incumbent_decay_pair(&mut sp, &timing, round);
                    let ours_null = ours_decay_null_pair(&timing, round);
                    (incumbent.0, incumbent.1, ours_null, scipy_null)
                }
                _ => {
                    let ours_null = ours_decay_null_pair(&timing, round);
                    let scipy_null = scipy_decay_null_pair(&mut sp, &timing, round);
                    let incumbent = incumbent_decay_pair(&mut sp, &timing, round);
                    (incumbent.0, incumbent.1, ours_null, scipy_null)
                }
            };
            ours_t.push(ours_secs);
            scipy_t.push(scipy_secs);
            ratios.push(scipy_secs / ours_secs);
            null_ours.push(ours_null);
            null_scipy.push(scipy_null);
        }

        let (ratio_low, ratio_high) = boot_ci(&ratios);
        let (ours_null_low, ours_null_high) = boot_ci(&null_ours);
        let (scipy_null_low, scipy_null_high) = boot_ci(&null_scipy);
        let p50_ours = median(ours_t.clone());
        let p50_scipy = median(scipy_t.clone());
        let p95_ours = percentile(ours_t.clone(), 0.95);
        let p95_scipy = percentile(scipy_t.clone(), 0.95);
        let p99_ours = percentile(ours_t.clone(), 0.99);
        let p99_scipy = percentile(scipy_t.clone(), 0.99);
        println!(
            "whole_job_wall: FrankenSciPy p50={:.6}ms p95={:.6}ms p99={:.6}ms | \
             SciPy p50={:.6}ms p95={:.6}ms p99={:.6}ms",
            p50_ours * 1e3 / reps as f64,
            p95_ours * 1e3 / reps as f64,
            p99_ours * 1e3 / reps as f64,
            p50_scipy * 1e3 / reps as f64,
            p95_scipy * 1e3 / reps as f64,
            p99_scipy * 1e3 / reps as f64
        );
        println!(
            "raw_samples_seconds: ours={} scipy={} ratios={} \
             null_ours={} null_scipy={}",
            ours_t
                .iter()
                .map(|value| format!("{value:.9}"))
                .collect::<Vec<_>>()
                .join(","),
            scipy_t
                .iter()
                .map(|value| format!("{value:.9}"))
                .collect::<Vec<_>>()
                .join(","),
            ratios
                .iter()
                .map(|value| format!("{value:.9}"))
                .collect::<Vec<_>>()
                .join(","),
            null_ours
                .iter()
                .map(|value| format!("{value:.9}"))
                .collect::<Vec<_>>()
                .join(","),
            null_scipy
                .iter()
                .map(|value| format!("{value:.9}"))
                .collect::<Vec<_>>()
                .join(",")
        );
        println!(
            "NULL-ours   median={:.6} ci95=[{ours_null_low:.6},{ours_null_high:.6}] \
             cv={:.3}% (provenance only)",
            median(null_ours.clone()),
            cv(&null_ours) * 100.0
        );
        println!(
            "NULL-scipy  median={:.6} ci95=[{scipy_null_low:.6},{scipy_null_high:.6}] \
             cv={:.3}% (provenance only)",
            median(null_scipy.clone()),
            cv(&null_scipy) * 100.0
        );
        let ratio_p50 = median(ratios.clone());
        println!(
            "Incumbent ratio: SciPy / FrankenSciPy = {ratio_p50:.4}x \
             (bootstrap-median ci95=[{ratio_low:.4},{ratio_high:.4}], \
             cv={:.3}% provenance only)",
            cv(&ratios) * 100.0
        );
        let null_edge = ours_null_high
            .max(scipy_null_high)
            .max(1.0 / ours_null_low.max(1e-9))
            .max(1.0 / scipy_null_low.max(1e-9));
        let required = 1.0 + 2.0 * (null_edge - 1.0);
        let outcome = if ratio_low > required {
            "DECIDED FRANKENSCIPY WIN"
        } else if ratio_high < 1.0 / required {
            "DECIDED FRANKENSCIPY LOSS"
        } else {
            "NOT DECIDED"
        };
        println!(
            "median-CI gate: worst_null_edge={null_edge:.4} required={required:.4} \
             ratio_ci=[{ratio_low:.4},{ratio_high:.4}] => {outcome}"
        );

        let rhs_replay = median(rhs_replays);
        let setup_replay = median(setup_replays);
        let rhs_net = (rhs_replay - setup_replay).max(0.0);
        let scipy_per_job = p50_scipy / reps as f64;
        let ours_per_job = p50_ours / reps as f64;
        println!(
            "decomposition: scipy_parallel_rhs_replay_with_pool={:.4}ms \
             pool_setup_only={:.4}ms net_rhs_sensitivity={:.4}ms={:.1}%_of_scipy_job; \
             callback_free_sensitivity_ratio={:.4}x",
            rhs_replay * 1e3,
            setup_replay * 1e3,
            rhs_net * 1e3,
            rhs_net / scipy_per_job * 100.0,
            (scipy_per_job - rhs_net).max(0.0) / ours_per_job
        );
        println!(
            "scope_guard: this row applies only to batched independent/diagonal stiff \
             reaction screens with {DECAY_JOB_SCENARIOS} scenarios on eight CPUs; \
             it is not a general BDF or dense-Jacobian claim"
        );
        sp.quit();
        if outcome == "DECIDED FRANKENSCIPY WIN" {
            println!(
                "CHOOSER STATEMENT: Pick FrankenSciPy solve_ivp_many for this \
                 eight-CPU, multi-scenario independent stiff reaction-screen shape; \
                 pick SciPy 1.17.1 for dense-Jacobian BDF work, where the separate \
                 n=512 live-arm result favors SciPy by 2.12x. Outside those two proven \
                 shapes, benchmark the user's actual job; this result does not decide."
            );
        } else if outcome == "DECIDED FRANKENSCIPY LOSS" {
            println!(
                "CHOOSER STATEMENT: Pick SciPy 1.17.1 with the screened \
                 {scipy_jacobian_mode} Jacobian configuration for this eight-CPU \
                 reaction-screen job; this measurement does not support choosing \
                 FrankenSciPy on performance."
            );
        } else {
            println!(
                "CHOOSER STATEMENT: This run does not distinguish the implementations \
                 beyond its A/A noise floor; choose on API and deployment fit, and \
                 choose SciPy for the separately measured dense n=512 BDF shape."
            );
        }
    }

    pub fn run() {
        let exe = std::env::current_exe().expect("current_exe");
        let sha = {
            let mut h = Sha256::new();
            h.update(std::fs::read(&exe).expect("read own ELF"));
            format!("{:x}", h.finalize())
        };
        println!("elf_sha256={sha}");
        println!("frankenscipy_engine_sha256={sha}");

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
        let builder_identity =
            std::env::var("BINARY_BUILDER_IDENTITY").unwrap_or_else(|_| "unrecorded".to_string());
        let source_commit =
            std::env::var("BINARY_SOURCE_COMMIT").unwrap_or_else(|_| "unrecorded".to_string());
        let build_route =
            std::env::var("BINARY_BUILD_ROUTE").unwrap_or_else(|_| "unrecorded".to_string());
        println!(
            "binary_provenance: builder_identity={builder_identity} \
             source_commit={source_commit} build_route={build_route}"
        );
        if fixture.is_decay_screen()
            && (builder_identity == "unrecorded"
                || source_commit == "unrecorded"
                || build_route == "unrecorded")
        {
            eprintln!(
                "ABORT: decay-screen requires BINARY_BUILDER_IDENTITY, \
                 BINARY_SOURCE_COMMIT, and BINARY_BUILD_ROUTE"
            );
            std::process::exit(2);
        }
        // argv[4] is the fixture; the optional script path moved to argv[5] when the
        // coupled fixture was added. Both reading argv[4] made the harness spawn
        // `python3 diagonal`, which surfaced as a confusing "not genuine" abort.
        let script = args
            .get(6)
            .cloned()
            .unwrap_or_else(|| "crates/fsci-integrate/python/scipy_bdf_arm.py".to_string());
        let affinity = cpu_affinity();
        let host = host_identity();
        let (physical_cores, logical_threads) = cpu_topology();
        let ram_bytes = ram_bytes();
        let numa_nodes = numa_node_count();
        let cpuset_logical_cap = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1);
        println!(
            "hardware_provenance: host_identity={} physical_cores={physical_cores} \
             logical_threads={logical_threads} ram_bytes={ram_bytes} \
             numa_nodes={numa_nodes} runtime_detected_isa={} \
             affinity={affinity} cpuset_logical_cap={cpuset_logical_cap}",
            host,
            runtime_isa_features()
        );
        match cpu_frequency_policy(&affinity) {
            Ok(policy) => println!("cpu_frequency_policy: {policy}"),
            Err(error) if fixture.is_decay_screen() => {
                eprintln!("ABORT: CPU governor/frequency provenance unavailable: {error}");
                std::process::exit(2);
            }
            Err(error) => println!("cpu_frequency_policy: unavailable({error})"),
        }
        if host == "threadripperje" && fixture.is_batch_job() {
            let claim_message_id = std::env::var("TRJ_BOOKING_CLAIM_MESSAGE_ID")
                .ok()
                .filter(|value| {
                    !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit())
                })
                .unwrap_or_else(|| {
                    eprintln!(
                        "ABORT: a trj batch benchmark requires an exclusive \
                         Agent Mail booking; set TRJ_BOOKING_CLAIM_MESSAGE_ID"
                    );
                    std::process::exit(2);
                });
            println!("trj_booking_claim_message_id={claim_message_id}");
        }
        println!("cpu_affinity={affinity}");
        if affinity == "unknown"
            || (!fixture.is_batch_job() && (affinity.contains(',') || affinity.contains('-')))
        {
            eprintln!(
                "ABORT: pin ordinary solver cells to exactly one CPU; the \
                 batch shape cells require an explicit taskset affinity"
            );
            std::process::exit(2);
        }
        if !fixture.accepts_dimension(n) || rounds < 3 || reps == 0 {
            eprintln!(
                "ABORT: fixture {} rejects n={n}; require exponential n=1, \
                 lorenz n=3, lotka-many variants batch>=1, all stiff fixtures n>=2, \
                 rounds>=3, and reps>=1",
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
        if fixture.is_lotka_many() && method != Method::Rk45 {
            eprintln!("ABORT: lotka-many fixtures require method=rk45");
            std::process::exit(2);
        }
        if fixture == Fixture::RadauStiff && method != Method::Radau {
            eprintln!("ABORT: radau-stiff fixture requires method=radau");
            std::process::exit(2);
        }
        if fixture.is_decay_screen() && method != Method::Bdf {
            eprintln!("ABORT: decay-screen requires method=bdf");
            std::process::exit(2);
        }
        if fixture.is_decay_screen() && rounds < 7 {
            eprintln!("ABORT: decay-screen requires rounds>=7 for its median-CI gate");
            std::process::exit(2);
        }
        if fixture.is_lotka_many() {
            run_lotka_many(&script, n, rounds, reps, &affinity, fixture.lotka_sampled());
            return;
        }
        if fixture.is_decay_screen() {
            run_decay_screen(&script, n, rounds, reps, &affinity);
            return;
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
        println!(
            "thread_provenance: requested_frankenscipy_threads=1 \
             actual_observed_frankenscipy_worker_threads=1 \
             cpuset_logical_cap={cpuset_logical_cap}"
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
            || !ready.contains("actual_observed_worker_threads=")
            || !ready.contains("scipy_engine_sha256=")
        {
            eprintln!("ABORT: SciPy arm is not genuine (dispatch trap)");
            std::process::exit(4);
        }
        let actual_scipy_workers = ready_value(&ready, "actual_observed_worker_threads=")
            .and_then(|value| value.parse::<usize>().ok())
            .expect("READY line has numeric actual observed SciPy workers");
        let scipy_engine_sha256 = ready_value(&ready, "scipy_engine_sha256=")
            .expect("READY line has SciPy engine SHA-256");
        println!(
            "scipy_thread_provenance: requested_scipy_threads=1 \
             actual_observed_scipy_worker_threads={actual_scipy_workers} \
             python_blas_thread_cap=1"
        );
        println!("scipy_engine_sha256={scipy_engine_sha256}");
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
        for (index, (&our_value, &their_value)) in our_y.iter().zip(&theirs.y).enumerate() {
            let difference = (our_value - their_value).abs();
            let comparison_scale =
                fixture.atol() + fixture.rtol() * our_value.abs().max(their_value.abs());
            max_abs_diff = max_abs_diff.max(difference);
            max_scaled_diff = max_scaled_diff.max(difference / comparison_scale);
            if let Some(analytic) = fixture.analytic_final(index, &y0, &r) {
                analytic_components += 1;
                let analytic_scale = fixture.atol() + fixture.rtol() * analytic.abs();
                max_scaled_ours_analytic =
                    max_scaled_ours_analytic.max((our_value - analytic).abs() / analytic_scale);
                max_scaled_scipy_analytic =
                    max_scaled_scipy_analytic.max((their_value - analytic).abs() / analytic_scale);
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
