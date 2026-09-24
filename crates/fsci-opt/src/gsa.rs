//! SciPy's generalized simulated annealing (`dual_annealing`), transcribed from
//! `scipy/optimize/_dual_annealing.py` of SciPy 1.17.1: `VisitingDistribution`, `EnergyState`,
//! `StrategyChain`, `LocalSearchWrapper`'s acceptance test and the driver's temperature schedule
//! and restarts. std-only: the random source and the local search are injected.

/// A uniform draw on [0, 1).
pub(crate) trait Uniform {
    fn next_f64(&mut self) -> f64;
}

/// ln Γ(x) for x > 0 (Lanczos, g = 7, n = 9; ~1e-15 relative).
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + G + 0.5;
    for (i, c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// A standard normal draw (Box–Muller).
fn normal<R: Uniform>(rng: &mut R) -> f64 {
    loop {
        let u1 = rng.next_f64();
        if u1 > 0.0 {
            let u2 = rng.next_f64();
            return (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        }
    }
}

const TAIL_LIMIT: f64 = 1.0e8;
const MIN_VISIT_BOUND: f64 = 1.0e-10;

/// SciPy's `VisitingDistribution`: Tsallis–Stariolo visits with parameter `qv`.
struct Visiting {
    qv: f64,
    lower: Vec<f64>,
    range: Vec<f64>,
    factor4_p: f64,
    factor6: f64,
}

impl Visiting {
    fn new(lower: &[f64], upper: &[f64], qv: f64) -> Self {
        let factor2 = ((4.0 - qv) * (qv - 1.0).ln()).exp();
        let factor3 = ((2.0 - qv) * 2.0_f64.ln() / (qv - 1.0)).exp();
        let factor4_p = std::f64::consts::PI.sqrt() * factor2 / (factor3 * (3.0 - qv));
        let factor5 = 1.0 / (qv - 1.0) - 0.5;
        let d1 = 2.0 - factor5;
        let pi = std::f64::consts::PI;
        let factor6 = pi * (1.0 - factor5) / (pi * (1.0 - factor5)).sin() / ln_gamma(d1).exp();
        Self {
            qv,
            lower: lower.to_vec(),
            range: upper.iter().zip(lower).map(|(u, l)| u - l).collect(),
            factor4_p,
            factor6,
        }
    }

    /// SciPy's `visit_fn`: `dim` draws of x / |y|^((qv-1)/(3-qv)) with x, y standard normal.
    fn visit_fn<R: Uniform>(&self, temperature: f64, dim: usize, rng: &mut R) -> Vec<f64> {
        let qv = self.qv;
        let factor1 = (temperature.ln() / (qv - 1.0)).exp();
        let factor4 = self.factor4_p * factor1;
        let sigmax = (-(qv - 1.0) * (self.factor6 / factor4).ln() / (3.0 - qv)).exp();
        // numpy draws normal(size=(dim, 2)) row by row and splits the columns.
        let draws: Vec<(f64, f64)> = (0..dim).map(|_| (normal(rng), normal(rng))).collect();
        draws
            .into_iter()
            .map(|(x, y)| {
                let den = ((qv - 1.0) * y.abs().ln() / (3.0 - qv)).exp();
                x * sigmax / den
            })
            .collect()
    }

    fn wrap(&self, i: usize, v: f64) -> f64 {
        let a = v - self.lower[i];
        let b = a % self.range[i] + self.range[i];
        let mut w = b % self.range[i] + self.lower[i];
        if (w - self.lower[i]).abs() < MIN_VISIT_BOUND {
            w += MIN_VISIT_BOUND;
        }
        w
    }

    /// SciPy's `visiting`: all coordinates for the first `dim` steps of a chain, then one each.
    fn visiting<R: Uniform>(
        &self,
        x: &[f64],
        step: usize,
        temperature: f64,
        rng: &mut R,
    ) -> Vec<f64> {
        let dim = x.len();
        if step < dim {
            let mut visits = self.visit_fn(temperature, dim, rng);
            let upper_sample = rng.next_f64();
            let lower_sample = rng.next_f64();
            for v in &mut visits {
                if *v > TAIL_LIMIT {
                    *v = TAIL_LIMIT * upper_sample;
                } else if *v < -TAIL_LIMIT {
                    *v = -TAIL_LIMIT * lower_sample;
                }
            }
            (0..dim).map(|i| self.wrap(i, visits[i] + x[i])).collect()
        } else {
            let mut x_visit = x.to_vec();
            let mut visit = self.visit_fn(temperature, 1, rng)[0];
            if visit > TAIL_LIMIT {
                visit = TAIL_LIMIT * rng.next_f64();
            } else if visit < -TAIL_LIMIT {
                visit = -TAIL_LIMIT * rng.next_f64();
            }
            let index = step - dim;
            x_visit[index] = self.wrap(index, visit + x[index]);
            x_visit
        }
    }
}

pub(crate) struct GsaOutcome {
    pub x: Vec<f64>,
    pub fun: f64,
    pub nit: usize,
    pub nfev: usize,
    pub njev: usize,
    pub message: String,
    /// False only when the evaluation budget stopped the search.
    pub success: bool,
}

/// A local search from `x`: `Some((fun, x, nfev, njev))`, or `None` if it failed outright.
pub(crate) type LocalSearch<'a> = dyn FnMut(&[f64]) -> Option<(f64, Vec<f64>, usize, usize)> + 'a;

/// SciPy's `dual_annealing` with its defaults (initial_temp 5230, restart_temp_ratio 2e-5,
/// visit 2.62, accept -5, maxfun 1e7). `local` is `None` for `no_local_search`. `Err` is SciPy's
/// refusal of an objective that stays non-finite over 1000 random starts.
pub(crate) fn anneal<F: Fn(&[f64]) -> f64, R: Uniform>(
    func: &F,
    lower: &[f64],
    upper: &[f64],
    maxiter: usize,
    rng: &mut R,
    mut local: Option<&mut LocalSearch<'_>>,
) -> Result<GsaOutcome, String> {
    const INITIAL_TEMP: f64 = 5230.0;
    const RESTART_TEMP_RATIO: f64 = 2.0e-5;
    const VISIT: f64 = 2.62;
    const ACCEPT: f64 = -5.0;
    const MAXFUN: usize = 10_000_000;
    const MAX_REINIT_COUNT: usize = 1000;
    let n = lower.len();
    let mut nfev = 0usize;
    let mut njev = 0usize;
    let draw = |rng: &mut R| -> Vec<f64> {
        lower
            .iter()
            .zip(upper)
            .map(|(l, u)| l + (u - l) * rng.next_f64())
            .collect()
    };

    // EnergyState.reset: a random start, redrawn while its energy is not finite.
    let reset = |rng: &mut R, nfev: &mut usize| -> Result<(Vec<f64>, f64), String> {
        let mut location = draw(rng);
        let mut reinit = 0;
        loop {
            let energy = func(&location);
            *nfev += 1;
            if energy.is_finite() {
                return Ok((location, energy));
            }
            if reinit >= MAX_REINIT_COUNT {
                return Err(String::from(
                    "Stopping algorithm because function create NaN or (+/-) infinity values \
                     even with trying new random parameters",
                ));
            }
            location = draw(rng);
            reinit += 1;
        }
    };
    let (mut current_location, mut current_energy) = reset(rng, &mut nfev)?;
    let mut xbest = current_location.clone();
    let mut ebest = current_energy;

    let temperature_restart = INITIAL_TEMP * RESTART_TEMP_RATIO;
    let visit_dist = Visiting::new(lower, upper, VISIT);
    // StrategyChain state.
    let mut emin = current_energy;
    let mut xmin = current_location.clone();
    let mut not_improved_idx = 0usize;
    let mut not_improved_max_idx = 1000usize;
    let mut energy_state_improved = false;

    let no_local_search = local.is_none();
    // LocalSearchWrapper.local_search: keep the result only if finite, in bounds and lower.
    let mut local_search =
        |x: &[f64], e: f64, nfev: &mut usize, njev: &mut usize| -> (f64, Vec<f64>) {
            let Some(search) = local.as_deref_mut() else {
                return (e, x.to_vec());
            };
            match search(x) {
                Some((fun, xs, fe, je)) => {
                    *nfev += fe;
                    *njev += je;
                    let valid = xs.iter().all(|v| v.is_finite())
                        && fun.is_finite()
                        && xs.iter().zip(lower).all(|(v, l)| v >= l)
                        && xs.iter().zip(upper).all(|(v, u)| v <= u);
                    if valid && fun < e {
                        (fun, xs)
                    } else {
                        (e, x.to_vec())
                    }
                }
                None => (e, x.to_vec()),
            }
        };

    let t1 = ((VISIT - 1.0) * 2.0_f64.ln()).exp() - 1.0;
    let mut iteration = 0usize;
    let mut message = String::new();
    // status: SciPy starts `success = True` and clears it only when maxfun stops the search
    let mut success = true;
    let mut need_to_stop = false;
    while !need_to_stop {
        for i in 0..maxiter {
            let s = i as f64 + 2.0;
            let t2 = ((VISIT - 1.0) * s.ln()).exp() - 1.0;
            let temperature = INITIAL_TEMP * t1 / t2;
            if iteration >= maxiter {
                message = String::from("Maximum number of iteration reached");
                need_to_stop = true;
                break;
            }
            if temperature < temperature_restart {
                let (loc, energy) = reset(rng, &mut nfev)?;
                current_location = loc;
                current_energy = energy;
                break;
            }
            // StrategyChain.run. (SciPy also reads `temperature_step` in the probabilistic
            // local-search branch, which never fires; see below.)
            let temperature_step = temperature / (i as f64 + 1.0);
            not_improved_idx += 1;
            let mut stop = None;
            for j in 0..2 * n {
                if j == 0 {
                    energy_state_improved = i == 0;
                }
                let x_visit = visit_dist.visiting(&current_location, j, temperature, rng);
                let e = func(&x_visit);
                nfev += 1;
                if e < current_energy {
                    current_energy = e;
                    current_location.clone_from(&x_visit);
                    if e < ebest {
                        ebest = e;
                        xbest.clone_from(&x_visit);
                        energy_state_improved = true;
                        not_improved_idx = 0;
                    }
                } else {
                    // accept_reject
                    let r = rng.next_f64();
                    let pqv_temp = 1.0 - ((1.0 - ACCEPT) * (e - current_energy) / temperature_step);
                    let pqv = if pqv_temp <= 0.0 {
                        0.0
                    } else {
                        (pqv_temp.ln() / (1.0 - ACCEPT)).exp()
                    };
                    if r <= pqv {
                        current_energy = e;
                        current_location.clone_from(&x_visit);
                        xmin.clone_from(&current_location);
                    }
                    if not_improved_idx >= not_improved_max_idx && (j == 0 || current_energy < emin)
                    {
                        emin = current_energy;
                        xmin.clone_from(&current_location);
                    }
                }
                if nfev >= MAXFUN {
                    stop = Some("Maximum number of function call reached during annealing");
                    break;
                }
            }
            if let Some(msg) = stop {
                message = String::from(msg);
                need_to_stop = true;
                success = false;
                break;
            }
            if !no_local_search {
                // StrategyChain.local_search
                if energy_state_improved {
                    let (e, x) = local_search(&xbest.clone(), ebest, &mut nfev, &mut njev);
                    if e < ebest {
                        not_improved_idx = 0;
                        ebest = e;
                        xbest.clone_from(&x);
                        current_energy = e;
                        current_location = x;
                    }
                    if nfev >= MAXFUN {
                        message = String::from(
                            "Maximum number of function call reached during local search",
                        );
                        need_to_stop = true;
                        success = false;
                        break;
                    }
                }
                // SciPy's probabilistic branch needs K < 90 n, but K = 100 n and never changes,
                // so only the stagnation trigger remains.
                if not_improved_idx >= not_improved_max_idx {
                    let (e, x) = local_search(&xmin.clone(), emin, &mut nfev, &mut njev);
                    xmin.clone_from(&x);
                    emin = e;
                    not_improved_idx = 0;
                    not_improved_max_idx = n;
                    if e < ebest {
                        ebest = emin;
                        xbest.clone_from(&xmin);
                        current_energy = e;
                        current_location = x;
                    }
                    if nfev >= MAXFUN {
                        message = String::from(
                            "Maximum number of function call reached during dual annealing",
                        );
                        need_to_stop = true;
                        success = false;
                        break;
                    }
                }
            }
            iteration += 1;
        }
    }
    Ok(GsaOutcome {
        x: xbest,
        fun: ebest,
        nit: iteration,
        nfev,
        njev,
        message,
        success,
    })
}
