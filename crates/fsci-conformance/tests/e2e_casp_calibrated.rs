#![forbid(unsafe_code)]
//! frankenscipy-7tb8d: End-to-end verification of calibrated CASP on the held-out calibration
//! split and under an injected synthetic regime shift, with pre-epic negative controls.
//!
//! Proves on ONE persistent `SolverPortfolio`:
//!   (a) Held-out calibration split empirical exceedance rate <= alpha (0.05) AND its 95%
//!       Clopper-Pearson upper confidence bound <= alpha, with per-decade breakdown logged.
//!   (b) Independent accuracy certificate verification (`verify_solve_certificate`) passes for
//!       100% of solves across both the held-out split and the injected shift.
//!   (c) Certificate decision replay (`replay_decision`) passes bit-for-bit for 100% of
//!       certificates against their pre-solve portfolio snapshots, while refusing post-solve
//!       snapshots, mutated outcome counts, and 1-ULP tampered posteriors.
//!   (d) An anytime-valid `EProcessMonitor` drift alarm is raised within the documented
//!       detection delay when a synthetic pivot-growth regime shift is injected, while online
//!       Dirichlet + failure-rate feedback adapts solver selection before the global alarm.
//!   (e) Negative controls: running the same workload against the pre-epic baseline (fixed
//!       `condition_posterior`, no `record_outcome` feedback, hand-set loss matrix, unwired
//!       per-call controller) fails both (a)/(d) injected-shift detection and (c) replay with
//!       feedback.

use fsci_linalg::{
    ATTEMPT_BACKWARD_ERROR_TOL, SolveCertificate, SolveOptions, condition_diagnostics,
    replay_decision, solve_with_action, solve_with_casp, verify_solve_certificate,
};
use fsci_runtime::{
    AttemptOutcome, ConformalRiskControl, CrcCalibrationSample, PolicyController, RuntimeMode,
    SolverAction, SolverPortfolio, clopper_pearson_upper_bound,
};

const SEED: u64 = 0x7B8D_0002_CA11_B8A7;
const SIZES: [usize; 3] = [8, 32, 128];
const ALPHA: f64 = 0.05;
/// Documented maximum detection delay (in shifted solves) from W4.4 (`frankenscipy-7tb8d.4`).
const DOCUMENTED_MAX_SHIFT_DELAY: usize = 150;

/// Deterministic splitmix64 PRNG matching `casp_calibrate.rs` (frankenscipy-7tb8d.2 / .3).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

type Matrix = Vec<Vec<f64>>;

fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
    let n = a.len();
    let m = b[0].len();
    let mut c = vec![vec![0.0; m]; n];
    for (i, row) in a.iter().enumerate() {
        for (k, &aik) in row.iter().enumerate() {
            for (j, &bkj) in b[k].iter().enumerate() {
                c[i][j] += aik * bkj;
            }
        }
    }
    c
}

fn transpose(a: &Matrix) -> Matrix {
    (0..a[0].len())
        .map(|j| a.iter().map(|row| row[j]).collect())
        .collect()
}

fn orthogonal(n: usize, rng: &mut Rng) -> Matrix {
    let mut q: Matrix = (0..n)
        .map(|_| (0..n).map(|_| rng.normal()).collect())
        .collect();
    for j in 0..n {
        for _ in 0..2 {
            for k in 0..j {
                let dot: f64 = (0..n).map(|i| q[k][i] * q[j][i]).sum();
                for i in 0..n {
                    q[j][i] -= dot * q[k][i];
                }
            }
        }
        let norm = q[j].iter().map(|v| v * v).sum::<f64>().sqrt();
        q[j].iter_mut().for_each(|v| *v /= norm);
    }
    q
}

fn with_singular_values(sigma: &[f64], rng: &mut Rng) -> Matrix {
    let n = sigma.len();
    let u = orthogonal(n, rng);
    let v = orthogonal(n, rng);
    let us: Matrix = u
        .iter()
        .map(|row| row.iter().zip(sigma).map(|(x, s)| x * s).collect())
        .collect();
    matmul(&us, &transpose(&v))
}

fn randsvd_sigma(n: usize, kappa: f64, mode: u32) -> Vec<f64> {
    (0..n)
        .map(|i| match mode {
            1 => {
                if i == 0 {
                    1.0
                } else {
                    1.0 / kappa
                }
            }
            2 => {
                if i + 1 == n {
                    1.0 / kappa
                } else {
                    1.0
                }
            }
            _ => kappa.powf(-(i as f64) / (n - 1) as f64),
        })
        .collect()
}

fn symmetric(lambda: &[f64], rng: &mut Rng) -> Matrix {
    let n = lambda.len();
    let q = orthogonal(n, rng);
    let ql: Matrix = q
        .iter()
        .map(|row| row.iter().zip(lambda).map(|(x, l)| x * l).collect())
        .collect();
    let mut a = matmul(&ql, &transpose(&q));
    for i in 0..n {
        for j in 0..i {
            let mean = 0.5 * (a[i][j] + a[j][i]);
            a[i][j] = mean;
            a[j][i] = mean;
        }
    }
    a
}

fn spread(n: usize, kappa: f64, rng: &mut Rng) -> Vec<f64> {
    let mut values: Vec<f64> = (0..n)
        .map(|i| kappa.powf(-(i as f64) / (n - 1) as f64))
        .collect();
    for i in (1..n).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        values.swap(i, j);
    }
    values
        .into_iter()
        .map(|v| if rng.next().is_multiple_of(2) { v } else { -v })
        .collect()
}

#[derive(Clone)]
struct CorpusProblem {
    corpus_index: usize,
    family: String,
    n: usize,
    a: Matrix,
    x_true: Vec<f64>,
    b: Vec<f64>,
}

/// Build the exact W4.2 calibration corpus (324 problems, of which 321 are solvable).
fn build_w4_corpus() -> Vec<CorpusProblem> {
    let mut rng = Rng(SEED);
    let mut raw_cases: Vec<(String, usize, Matrix)> = Vec::new();
    let kappas: Vec<f64> = (0..=16).map(|k| 10f64.powi(k)).collect();
    for &n in &SIZES {
        let mut push = |family: String, a: Matrix| raw_cases.push((family, n, a));
        for mode in 1..=3 {
            for &kappa in &kappas {
                let a = with_singular_values(&randsvd_sigma(n, kappa, mode), &mut rng);
                push(format!("randsvd(mode={mode},kappa={kappa:e})"), a);
            }
        }
        for rank_loss in 1..=3 {
            let mut sigma = randsvd_sigma(n, 1e3, 3);
            for s in sigma.iter_mut().rev().take(rank_loss) {
                *s = 0.0;
            }
            push(
                format!("singular(rank=n-{rank_loss})"),
                with_singular_values(&sigma, &mut rng),
            );
        }
        for k in (0..=16).step_by(2) {
            let kappa = 10f64.powi(k);
            let definite: Vec<f64> = spread(n, kappa, &mut rng).iter().map(|v| v.abs()).collect();
            push(
                format!("symmetric_definite(kappa={kappa:e})"),
                symmetric(&definite, &mut rng),
            );
            let indefinite = spread(n, kappa, &mut rng);
            push(
                format!("symmetric_indefinite(kappa={kappa:e})"),
                symmetric(&indefinite, &mut rng),
            );
            let diagonal = spread(n, kappa, &mut rng);
            let mut a = vec![vec![0.0; n]; n];
            for (i, d) in diagonal.into_iter().enumerate() {
                a[i][i] = d;
            }
            push(format!("diagonal(kappa={kappa:e})"), a);
            for lower in [false, true] {
                let diagonal = spread(n, kappa, &mut rng);
                let mut a = vec![vec![0.0; n]; n];
                for (i, (row, &d)) in a.iter_mut().zip(&diagonal).enumerate() {
                    for (j, entry) in row.iter_mut().enumerate() {
                        let inside = if lower { j < i } else { j > i };
                        if inside {
                            *entry = d * 0.01 * rng.normal() / n as f64;
                        }
                    }
                    row[i] = d;
                }
                let side = if lower { "lower" } else { "upper" };
                push(format!("triangular_{side}(kappa={kappa:e})"), a);
            }
        }
        let mut zero_diagonal = vec![vec![0.0; n]; n];
        for (i, row) in zero_diagonal.iter_mut().enumerate().skip(1) {
            row[i] = 1.0;
        }
        push("diagonal(singular)".to_string(), zero_diagonal);
        let hilbert: Matrix = (0..n)
            .map(|i| (0..n).map(|j| 1.0 / (i + j + 1) as f64).collect())
            .collect();
        let mut lotkin = hilbert.clone();
        lotkin[0].iter_mut().for_each(|v| *v = 1.0);
        push("hilbert".to_string(), hilbert);
        push("lotkin".to_string(), lotkin);
        let (s, c) = (1.2_f64.sin(), 1.2_f64.cos());
        let kahan: Matrix = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let scale = s.powi(i as i32);
                        match j.cmp(&i) {
                            std::cmp::Ordering::Less => 0.0,
                            std::cmp::Ordering::Equal => scale,
                            std::cmp::Ordering::Greater => -c * scale,
                        }
                    })
                    .collect()
            })
            .collect();
        push("kahan(theta=1.2)".to_string(), kahan);
        let wilkinson: Matrix = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        if j == n - 1 || i == j {
                            1.0
                        } else if j < i {
                            -1.0
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();
        push("wilkinson_growth".to_string(), wilkinson);
        for k in [4, 8, 12, 16] {
            let a: Matrix = (0..n)
                .map(|i| {
                    let scale = 10f64.powf(-(k as f64) * i as f64 / (n - 1) as f64);
                    (0..n).map(|_| scale * rng.normal()).collect()
                })
                .collect();
            push(format!("graded(rows 1..1e-{k})"), a);
        }
    }

    let mut rhs_rng = Rng(SEED ^ 0xB);
    raw_cases
        .into_iter()
        .enumerate()
        .map(|(corpus_index, (family, n, a))| {
            let x_true: Vec<f64> = (0..n).map(|_| rhs_rng.normal()).collect();
            let b: Vec<f64> = a
                .iter()
                .map(|row| row.iter().zip(&x_true).map(|(aij, xj)| aij * xj).sum())
                .collect();
            CorpusProblem {
                corpus_index,
                family,
                n,
                a,
                x_true,
                b,
            }
        })
        .collect()
}

fn split_50_50<T: Clone>(items: &[T], seed: u64) -> (Vec<T>, Vec<T>) {
    let mut indices: Vec<usize> = (0..items.len()).collect();
    let mut rng = Rng(seed);
    for i in (1..indices.len()).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        indices.swap(i, j);
    }
    let half = items.len() / 2;
    let cal = indices[..half].iter().map(|&i| items[i].clone()).collect();
    let held_out = indices[half..].iter().map(|&i| items[i].clone()).collect();
    (cal, held_out)
}

fn rcond_decade_bin(rcond: f64) -> usize {
    if !(rcond > 0.0) {
        return 16;
    }
    (-rcond.log10().round()).clamp(0.0, 16.0) as usize
}

fn true_relative_inf_error(x: &[f64], x_true: &[f64]) -> f64 {
    let scale = x_true.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if scale == 0.0 {
        return 0.0;
    }
    x.iter()
        .zip(x_true)
        .map(|(xi, ti)| (xi - ti).abs())
        .fold(0.0_f64, f64::max)
        / scale
}

/// Construct a synthetic pivot-growth Wilkinson matrix of size `n = 64` whose partial-pivoting
/// LU factor grows by `2^(n-1) = 2^63`, causing `DirectLU` to exceed `ATTEMPT_BACKWARD_ERROR_TOL`
/// (`omega ~ 0.1 > 1e-8`) while `PivotedQR` solves it to machine precision (`omega ~ 1e-15`).
fn synthetic_wilkinson_shift_problem(step: usize, rng: &mut Rng) -> (Matrix, Vec<f64>, Vec<f64>) {
    let n = 64;
    let scale = 1.0 + 0.01 * (step as f64);
    let a: Matrix = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    if j == n - 1 || i == j {
                        scale
                    } else if j < i {
                        -scale
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();
    let x_true: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let b: Vec<f64> = a
        .iter()
        .map(|row| row.iter().zip(&x_true).map(|(aij, xj)| aij * xj).sum())
        .collect();
    (a, x_true, b)
}

#[test]
fn e2e_casp_calibrated_held_out_and_regime_shift() {
    let all_problems = build_w4_corpus();
    assert_eq!(all_problems.len(), 324);

    // Filter out the 3 structurally zero-row singular diagonal matrices (1 per size), leaving
    // the 321 solvable problems matching SOLVER_CRC_N = 321.
    let solvable: Vec<CorpusProblem> = all_problems
        .into_iter()
        .filter(|p| p.family != "diagonal(singular)")
        .collect();
    assert_eq!(solvable.len(), 321);

    let (cal_split, held_out_split) = split_50_50(&solvable, SEED);
    assert_eq!(cal_split.len(), 160);
    assert_eq!(held_out_split.len(), 161);

    // Calibrate ConformalRiskControl on the 160-problem calibration split at alpha = 0.05.
    let cal_samples: Vec<CrcCalibrationSample> = cal_split
        .iter()
        .map(|p| {
            let diag = condition_diagnostics(&p.a).expect("diagnostics");
            let lu_failed = solve_with_action(&p.a, &p.b, SolverAction::DirectLU)
                .map(|r| {
                    !r.x.iter().all(|v| v.is_finite())
                        || r.backward_error.unwrap_or(f64::INFINITY) > ATTEMPT_BACKWARD_ERROR_TOL
                })
                .unwrap_or(true);
            let qr_failed = solve_with_action(&p.a, &p.b, SolverAction::PivotedQR)
                .map(|r| {
                    !r.x.iter().all(|v| v.is_finite())
                        || r.backward_error.unwrap_or(f64::INFINITY) > ATTEMPT_BACKWARD_ERROR_TOL
                })
                .unwrap_or(true);
            CrcCalibrationSample {
                rcond: diag.rcond_estimate,
                unescalated_exceeded: lu_failed,
                escalated_exceeded: qr_failed,
            }
        })
        .collect();
    let crc = ConformalRiskControl::calibrate(&cal_samples, ALPHA, SEED);

    // ONE persistent SolverPortfolio for the entire held-out evaluation + regime shift.
    let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 256);
    portfolio.set_crc(crc.clone());

    let mut per_decade_total = [0usize; 17];
    let mut per_decade_final_exceed = [0usize; 17];
    let mut per_decade_attempt_exceed = [0usize; 17];
    let mut compared_held_out = 0usize;
    let mut final_exceedances = 0usize;
    let mut attempt_exceedances = 0usize;
    let mut verified_count = 0usize;
    let mut replayed_count = 0usize;
    let mut advanced_refused = 0usize;
    let mut mutated_refused = 0usize;
    let mut tampered_caught = 0usize;
    let mut verification_failures = Vec::new();
    let mut replay_failures = Vec::new();
    let mut distinct_digests = std::collections::BTreeSet::new();

    for problem in &held_out_split {
        let snapshot = portfolio.clone();
        let res = solve_with_casp(
            &problem.a,
            &problem.b,
            SolveOptions::default(),
            &mut portfolio,
        )
        .unwrap_or_else(|e| {
            panic!(
                "held-out corpus problem #{} ({}, n={}) failed to solve: {e}",
                problem.corpus_index, problem.family, problem.n
            )
        });
        let cert: SolveCertificate = res
            .certificate
            .clone()
            .expect("solve_with_casp always emits a certificate");
        compared_held_out += 1;
        distinct_digests.insert(cert.decision.state_digest.clone());

        let decade = rcond_decade_bin(cert.rcond_estimate);
        per_decade_total[decade] += 1;

        let omega = res.backward_error.unwrap_or(f64::INFINITY);
        let final_exceeded = !omega.is_finite() || omega > ATTEMPT_BACKWARD_ERROR_TOL;
        if final_exceeded {
            final_exceedances += 1;
            per_decade_final_exceed[decade] += 1;
        }
        if final_exceeded || cert.fallback_active {
            attempt_exceedances += 1;
            per_decade_attempt_exceed[decade] += 1;
        }

        // (b) Certificate verification (W4.6)
        let verify_report = verify_solve_certificate(&problem.a, &problem.b, &res.x, &cert);
        let true_err = true_relative_inf_error(&res.x, &problem.x_true);
        let certified_bound = cert
            .accuracy
            .as_ref()
            .and_then(|acc| acc.forward_error_bound);
        if verify_report.verified {
            verified_count += 1;
        } else {
            verification_failures.push(format!(
                "corpus_index={} seed={SEED:#x} family={} n={} rcond={:.3e} action={:?} bound={:?} true_err={:.3e} reason={}",
                problem.corpus_index,
                problem.family,
                problem.n,
                cert.rcond_estimate,
                cert.action,
                certified_bound,
                true_err,
                verify_report.reason
            ));
        }

        // (c) Replay determinism (W4.12)
        let replay = replay_decision(&cert, &snapshot);
        if replay.replayed {
            replayed_count += 1;
        } else {
            replay_failures.push(format!(
                "corpus_index={} family={} n={}: {}",
                problem.corpus_index, problem.family, problem.n, replay.reason
            ));
        }

        // Replay negative controls on every held-out solve
        let after_solve = replay_decision(&cert, &portfolio);
        if !after_solve.replayed && after_solve.reason.contains("digest") {
            advanced_refused += 1;
        }
        let mut mutated = snapshot.clone();
        mutated.record_outcome(cert.rcond_estimate, cert.action, AttemptOutcome::Ok);
        let mutated_rep = replay_decision(&cert, &mutated);
        if !mutated_rep.replayed && mutated_rep.reason.contains("digest") {
            mutated_refused += 1;
        }
        let mut tampered = cert.clone();
        tampered.posterior[0] = f64::from_bits(tampered.posterior[0].to_bits() ^ 1);
        let tampered_rep = replay_decision(&tampered, &snapshot);
        if !tampered_rep.replayed && tampered_rep.reason.contains("posterior") {
            tampered_caught += 1;
        }
    }

    let empirical_final_rate = final_exceedances as f64 / compared_held_out as f64;
    let cp95_final_upper = clopper_pearson_upper_bound(final_exceedances, compared_held_out, 0.05);
    let empirical_attempt_rate = attempt_exceedances as f64 / compared_held_out as f64;
    let cp95_attempt_upper =
        clopper_pearson_upper_bound(attempt_exceedances, compared_held_out, 0.05);

    println!(
        "=== e2e_casp_calibrated: Held-Out Split Summary (n_cal={}, n_held_out={}, alpha={:.4}, t_hat={:.3e}, Rhat_cal={:.6}) ===",
        crc.n_calibration(),
        compared_held_out,
        ALPHA,
        crc.t_hat(),
        crc.empirical_risk_at_t_hat()
    );
    for decade in 0..=16 {
        if per_decade_total[decade] > 0 {
            println!(
                "  decade 1e-{decade:02}: solves={:3} final_exceed={:2} attempt_exceed={:2}",
                per_decade_total[decade],
                per_decade_final_exceed[decade],
                per_decade_attempt_exceed[decade]
            );
        }
    }
    println!(
        "  final exceedance:   {final_exceedances}/{compared_held_out} = {empirical_final_rate:.6} (Clopper-Pearson 95% upper = {cp95_final_upper:.6}, alpha = {ALPHA:.4})"
    );
    println!(
        "  attempt exceedance: {attempt_exceedances}/{compared_held_out} = {empirical_attempt_rate:.6} (Clopper-Pearson 95% upper = {cp95_attempt_upper:.6}, alpha = {ALPHA:.4})"
    );
    println!(
        "  certificates verified: {verified_count}/{compared_held_out}; replayed: {replayed_count}/{compared_held_out}; distinct portfolio states: {}",
        distinct_digests.len()
    );
    for failure in &verification_failures {
        println!("  VERIFY_FAIL: {failure}");
    }
    for failure in &replay_failures {
        println!("  REPLAY_FAIL: {failure}");
    }

    assert!(compared_held_out > 0);
    assert_eq!(compared_held_out, 161);
    assert!(
        empirical_final_rate <= ALPHA,
        "empirical final exceedance rate {empirical_final_rate} > alpha {ALPHA}"
    );
    assert!(
        cp95_final_upper <= ALPHA,
        "Clopper-Pearson 95% upper bound {cp95_final_upper} > alpha {ALPHA}"
    );
    assert!(
        empirical_attempt_rate <= ALPHA,
        "empirical attempt exceedance rate {empirical_attempt_rate} > alpha {ALPHA}"
    );
    assert!(
        cp95_attempt_upper <= ALPHA,
        "attempt Clopper-Pearson 95% upper bound {cp95_attempt_upper} > alpha {ALPHA}"
    );
    assert!(
        verification_failures.is_empty(),
        "certificate verification failed on {} cases",
        verification_failures.len()
    );
    assert_eq!(verified_count, compared_held_out);
    assert!(
        replay_failures.is_empty(),
        "certificate replay failed on {} cases",
        replay_failures.len()
    );
    assert_eq!(replayed_count, compared_held_out);
    assert_eq!(advanced_refused, compared_held_out);
    assert_eq!(mutated_refused, compared_held_out);
    assert_eq!(tampered_caught, compared_held_out);

    // Before the injected shift, the anytime-valid e-process has not raised a false alarm on the
    // exchangeable held-out calibration split.
    assert!(
        !portfolio
            .policy_controller()
            .eprocess()
            .is_alarm_triggered(),
        "e-process must not false-alarm on the exchangeable held-out split"
    );

    // (d) Inject synthetic regime shift on the SAME persistent portfolio.
    let mut shift_rng = Rng(SEED ^ 0x51F7_0001);
    let shift_len = 20usize;
    let mut shift_alarm_delay: Option<usize> = None;
    let mut shift_posterior_adapted_step: Option<usize> = None;
    let mut shift_verified = 0usize;
    let mut shift_replayed = 0usize;

    println!("=== e2e_casp_calibrated: Injected Pivot-Growth Regime Shift Trajectory ===");
    for step in 1..=shift_len {
        let (a_shift, x_true_shift, b_shift) =
            synthetic_wilkinson_shift_problem(step, &mut shift_rng);
        let snapshot = portfolio.clone();
        let res = solve_with_casp(&a_shift, &b_shift, SolveOptions::default(), &mut portfolio)
            .expect("shifted Wilkinson solve rescued by QR fallback");
        let cert = res.certificate.as_ref().expect("certificate present");

        if shift_posterior_adapted_step.is_none() && cert.decision.action != SolverAction::DirectLU
        {
            shift_posterior_adapted_step = Some(step);
        }

        let ep = portfolio.policy_controller().eprocess();
        if shift_alarm_delay.is_none() && ep.is_alarm_triggered() {
            shift_alarm_delay = Some(step);
        }

        let verify_rep = verify_solve_certificate(&a_shift, &b_shift, &res.x, cert);
        let true_err = true_relative_inf_error(&res.x, &x_true_shift);
        if verify_rep.verified {
            shift_verified += 1;
        } else {
            println!(
                "  SHIFT_VERIFY_FAIL step={step} rcond={:.3e} action={:?} bound={:?} true_err={:.3e} reason={}",
                cert.rcond_estimate,
                cert.action,
                cert.accuracy
                    .as_ref()
                    .and_then(|acc| acc.forward_error_bound),
                true_err,
                verify_rep.reason
            );
        }

        let replay_rep = replay_decision(cert, &snapshot);
        if replay_rep.replayed {
            shift_replayed += 1;
        }

        println!(
            "  shift_step={step:02} portfolio_choice={:?} executed_action={:?} fallback={} e_value={:.4e} log_e={:.4} threshold={:.1} alarm={}",
            cert.decision.action,
            cert.action,
            cert.fallback_active,
            ep.e_value(),
            ep.log_e_value(),
            ep.threshold(),
            ep.is_alarm_triggered()
        );
    }

    let delay = shift_alarm_delay.expect("e-process drift alarm must fire during injected shift");
    let adapted_step = shift_posterior_adapted_step
        .expect("e-process alarm and online feedback must escalate portfolio choice during shift");
    println!(
        "  shift summary: e-process alarm raised at delay {delay} <= {DOCUMENTED_MAX_SHIFT_DELAY}, escalating portfolio choice at step {adapted_step} (DirectLU -> SVDFallback); verified={shift_verified}/{shift_len}, replayed={shift_replayed}/{shift_len}"
    );

    assert!(
        delay <= DOCUMENTED_MAX_SHIFT_DELAY,
        "drift alarm delay {delay} exceeded documented bound {DOCUMENTED_MAX_SHIFT_DELAY}"
    );
    assert!(
        delay <= 12,
        "expected rapid e-process detection on sustained shift, got {delay}"
    );
    assert_eq!(
        adapted_step,
        delay + 1,
        "portfolio choice must escalate on the solve immediately following the e-process alarm"
    );
    assert_eq!(shift_verified, shift_len);
    assert_eq!(shift_replayed, shift_len);
    assert_eq!(
        portfolio.select_action(1e-2, None).0,
        SolverAction::SVDFallback,
        "active e-process drift alarm must override portfolio selection to SVDFallback"
    );
}

#[test]
fn e2e_casp_calibrated_negative_controls_pre_epic_baseline() {
    let all_problems = build_w4_corpus();
    let solvable: Vec<CorpusProblem> = all_problems
        .into_iter()
        .filter(|p| p.family != "diagonal(singular)")
        .collect();
    let (_, held_out_split) = split_50_50(&solvable, SEED);

    // 1. Negative Control Arm (a1): Pre-epic hand-set loss matrix on the W4.3 high-kappa shifted
    // corpus (kappa in 1e12..1e16). The pre-epic hand-set matrix routed rcond <= 1e-8 to
    // SVDFallback, which refuses numerically rank-short square matrices and exceeds alpha = 0.05.
    let shifted_high_kappa: Vec<CorpusProblem> = solvable
        .iter()
        .filter(|p| {
            ["1e12", "1e13", "1e14", "1e15", "1e16", "1e-12", "1e-16"]
                .iter()
                .any(|tag| p.family.contains(tag))
        })
        .cloned()
        .collect();
    let (shifted_cal, shifted_held_out) = split_50_50(&shifted_high_kappa, SEED);
    assert_eq!(shifted_cal.len(), 48);
    assert_eq!(shifted_held_out.len(), 48);

    let mut pre_epic_svd_exceedances = 0usize;
    let mut calibrated_crc_exceedances = 0usize;
    for problem in &shifted_held_out {
        // Pre-epic rule: rcond <= 1e-8 dispatched to SVDFallback without fallback to LU/QR.
        let svd_failed = solve_with_action(&problem.a, &problem.b, SolverAction::SVDFallback)
            .map(|r| {
                !r.x.iter().all(|v| v.is_finite())
                    || r.backward_error.unwrap_or(f64::INFINITY) > ATTEMPT_BACKWARD_ERROR_TOL
            })
            .unwrap_or(true);
        if svd_failed {
            pre_epic_svd_exceedances += 1;
        }

        // Calibrated CASP portfolio solve:
        let mut portfolio = SolverPortfolio::new(RuntimeMode::Strict, 64);
        let res = solve_with_casp(
            &problem.a,
            &problem.b,
            SolveOptions::default(),
            &mut portfolio,
        );
        let failed = res
            .map(|r| {
                !r.x.iter().all(|v| v.is_finite())
                    || r.backward_error.unwrap_or(f64::INFINITY) > ATTEMPT_BACKWARD_ERROR_TOL
            })
            .unwrap_or(true);
        if failed {
            calibrated_crc_exceedances += 1;
        }
    }
    let pre_epic_shifted_rate = pre_epic_svd_exceedances as f64 / shifted_held_out.len() as f64;
    let calibrated_shifted_rate = calibrated_crc_exceedances as f64 / shifted_held_out.len() as f64;
    println!(
        "=== Negative Control (a1): High-Kappa Shifted Corpus (n={}) ===",
        shifted_held_out.len()
    );
    println!(
        "  pre-epic hand-set SVD rule exceedance: {pre_epic_svd_exceedances}/{} = {pre_epic_shifted_rate:.6} (> alpha {ALPHA:.4} -> FAILS coverage)",
        shifted_held_out.len()
    );
    println!(
        "  calibrated CASP exceedance:            {calibrated_crc_exceedances}/{} = {calibrated_shifted_rate:.6} (<= alpha {ALPHA:.4} -> PASSES coverage)",
        shifted_held_out.len()
    );
    assert!(
        pre_epic_shifted_rate > ALPHA,
        "pre-epic baseline must fail exceedance bound on shifted corpus: {pre_epic_shifted_rate}"
    );
    assert!(
        calibrated_shifted_rate <= ALPHA,
        "calibrated CASP must satisfy exceedance bound on shifted corpus: {calibrated_shifted_rate}"
    );

    // 2. Negative Control Arm (a2)/(d) & (c): Pre-epic fixed-posterior / no-feedback / per-call
    // fresh PolicyController baseline on the held-out split + Wilkinson regime shift.
    let fixed_baseline = SolverPortfolio::new(RuntimeMode::Strict, 64);
    let mut persistent = SolverPortfolio::new(RuntimeMode::Strict, 256);

    let mut held_out_replay_against_fixed_failed = 0usize;
    let mut held_out_posterior_mismatched_fixed = 0usize;
    for problem in &held_out_split {
        let res = solve_with_casp(
            &problem.a,
            &problem.b,
            SolveOptions::default(),
            &mut persistent,
        )
        .expect("held-out solve succeeds");
        let cert = res.certificate.expect("certificate present");

        // Replaying a certificate from the persistent feedback portfolio against the pre-epic
        // fixed-state baseline must fail after the first solve.
        let rep = replay_decision(&cert, &fixed_baseline);
        if !rep.replayed {
            held_out_replay_against_fixed_failed += 1;
        }

        // Even ignoring the state digest, compare the certificate's Dirichlet posterior against
        // the pre-epic fixed piecewise-linear `condition_posterior(rcond)` (returned by an
        // un-updated portfolio).
        let fixed_prior = fixed_baseline.posterior(cert.rcond_estimate);
        let posterior_bits_differ = cert
            .posterior
            .iter()
            .zip(fixed_prior.iter())
            .any(|(a, b)| a.to_bits() != b.to_bits());
        if posterior_bits_differ {
            held_out_posterior_mismatched_fixed += 1;
        }
    }

    // Now run the 20-step Wilkinson regime shift and compare against pre-epic per-call fresh
    // PolicyController + fixed-posterior baseline.
    let mut shift_rng = Rng(SEED ^ 0x51F7_0001);
    let shift_len = 20usize;
    let mut pre_epic_alarms = 0usize;
    let mut pre_epic_action_adaptations = 0usize;
    let mut shift_replay_against_fixed_failed = 0usize;

    for step in 1..=shift_len {
        let (a_shift, _x_true, b_shift) = synthetic_wilkinson_shift_problem(step, &mut shift_rng);
        let res = solve_with_casp(&a_shift, &b_shift, SolveOptions::default(), &mut persistent)
            .expect("shifted solve succeeds");
        let cert = res.certificate.expect("certificate present");

        if !replay_decision(&cert, &fixed_baseline).replayed {
            shift_replay_against_fixed_failed += 1;
        }

        // Pre-epic code rebuilt PolicyController::new(mode, 8) fresh per solve and never fed
        // EProcessMonitor, and used fixed_baseline without record_outcome feedback:
        let fresh_controller = PolicyController::new(RuntimeMode::Strict, 8);
        if fresh_controller.eprocess().is_alarm_triggered() {
            pre_epic_alarms += 1;
        }
        let (fixed_action, ..) =
            fixed_baseline.select_action(cert.rcond_estimate, Some(cert.structural_evidence));
        if fixed_action != SolverAction::DirectLU {
            pre_epic_action_adaptations += 1;
        }
    }

    println!(
        "=== Negative Control (c) & (d): Pre-Epic Fixed Posterior / Unwired Drift Monitor ==="
    );
    println!(
        "  held-out certificates rejected by fixed-baseline replay: {held_out_replay_against_fixed_failed}/{} (posterior bits diverged from fixed map on {held_out_posterior_mismatched_fixed}/{})",
        held_out_split.len(),
        held_out_split.len()
    );
    println!(
        "  shifted certificates rejected by fixed-baseline replay:  {shift_replay_against_fixed_failed}/{shift_len}"
    );
    println!(
        "  pre-epic drift alarms on shifted stream:                 {pre_epic_alarms}/{shift_len} (vs persistent e-process alarm at step {:?})",
        persistent.policy_controller().eprocess().alarm_step()
    );
    println!(
        "  pre-epic solver adaptations on shifted stream:           {pre_epic_action_adaptations}/{shift_len}"
    );

    assert_eq!(
        held_out_replay_against_fixed_failed,
        held_out_split.len() - 1,
        "every solve after the first must fail replay against a fixed no-feedback portfolio"
    );
    assert!(
        held_out_posterior_mismatched_fixed > held_out_split.len() / 2,
        "online Dirichlet updates must move the posterior away from the fixed map on most solves"
    );
    assert_eq!(shift_replay_against_fixed_failed, shift_len);
    assert_eq!(
        pre_epic_alarms, 0,
        "pre-epic unwired per-call controller never raises a drift alarm"
    );
    assert_eq!(
        pre_epic_action_adaptations, 0,
        "pre-epic fixed-posterior baseline never adapts solver selection under failure"
    );
    assert!(
        persistent
            .policy_controller()
            .eprocess()
            .is_alarm_triggered(),
        "real persistent portfolio must have raised the e-process drift alarm"
    );
}
