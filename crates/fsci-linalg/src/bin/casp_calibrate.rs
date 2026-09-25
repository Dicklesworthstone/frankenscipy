//! frankenscipy-7tb8d.2: calibrate the dense CASP portfolio's loss matrix from measurements.
//!
//! Every applicable action runs on every matrix of a seeded corpus (Higham's `randsvd` in three
//! singular-value modes at condition numbers 1e0..1e16, exactly singular matrices, Hilbert,
//! Lotkin, Kahan, Wilkinson's pivot-growth matrix, graded matrices, and diagonal, triangular
//! and symmetric definite / indefinite instances for the fast paths), at n = 8, 32 and 128.
//!
//! An action FAILS on a matrix when it returns an error, a non-finite answer, or a backward
//! error above [`ATTEMPT_BACKWARD_ERROR_TOL`]: the portfolio's own acceptance test, the one that
//! makes it fall back. A matrix counts toward each state with the weight the portfolio's prior
//! gives that state at the matrix's estimated rcond, which is what the portfolio decides on.
//!
//! `loss(a, s) = cost(a) + E[recovery cost of a failure of a | s]`: the expected cost of an
//! accepted answer when action `a` is tried first in state `s`. A failure is recovered by the
//! fallback, at no less than the cost of the cheapest action that succeeds on that matrix,
//! which is what it is charged. A matrix on which every action fails (an exactly singular one)
//! says nothing about which to try first, so it is counted and left out. `cost` is each action's
//! leading-order flop count relative to LU's `2n³/3` (Golub & Van Loan, 4th ed.): QR 2 (`4n³/3`),
//! SVD 31.5 (Golub–Reinsch with U and V, `21n³`), a symmetric factorization 0.5 (`n³/3`), a
//! triangular or diagonal solve 0 (`O(n²)`). Flops rather than wall time keep the constants
//! byte-identical from run to run. The measured wall-time ratios are in the report beside them.
//!
//! Output, on stdout between marker lines: the JSON report (`artifacts/casp-calibration-solver.json`)
//! and the generated module (`crates/fsci-runtime/src/calibrated_losses.rs`). `--no-timing` skips
//! the wall-time section (the constants do not depend on it).

use std::fmt::Write as _;
use std::time::Instant;

use fsci_linalg::{ATTEMPT_BACKWARD_ERROR_TOL, condition_diagnostics, solve_with_action};
use fsci_runtime::{Fingerprinter, RuntimeMode, SolverAction, SolverPortfolio, StructuralEvidence};

const SEED: u64 = 0x7B8D_0002_CA11_B8A7;
/// Bumped whenever the corpus or the rule changes; recorded in both outputs.
const CORPUS_VERSION: u32 = 1;
const SIZES: [usize; 3] = [8, 32, 128];
const STATES: [&str; 4] = [
    "WellConditioned",
    "Moderate",
    "IllConditioned",
    "NearSingular",
];
/// Leading-order flops relative to LU, in `SolverAction::ALL` order.
const COST: [f64; 6] = [1.0, 2.0, 31.5, 0.0, 0.0, 0.5];
const RECOVERY: usize = 2; // SVDFallback

/// splitmix64, for a deterministic corpus.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1).
    fn uniform(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Standard normal (Box–Muller).
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

/// A random orthogonal matrix: Gram–Schmidt, twice, on a Gaussian one.
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

/// `U·diag(sigma)·Vᵀ` with random orthogonal `U`, `V`.
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

/// Higham's `randsvd` singular values: mode 1, one large and the rest `1/kappa`; mode 2, all 1
/// but the last `1/kappa`; mode 3, geometric from 1 to `1/kappa`.
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

/// `Q·diag(lambda)·Qᵀ`, made exactly symmetric.
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

/// Values geometric from 1 to `1/kappa` in random order and signs.
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

struct Case {
    family: String,
    n: usize,
    a: Matrix,
}

fn corpus() -> Vec<Case> {
    let mut rng = Rng(SEED);
    let mut cases = Vec::new();
    let kappas: Vec<f64> = (0..=16).map(|k| 10f64.powi(k)).collect();
    for &n in &SIZES {
        let mut push = |family: String, a: Matrix| cases.push(Case { family, n, a });
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
            // `D·(I + E)` with `E` strictly triangular and small, so the condition number follows
            // `D`'s spread. A triangle of independent random entries is exponentially
            // ill-conditioned in n (Viswanath and Trefethen), whatever its diagonal.
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
        // Wilkinson's pivot-growth matrix: 1 on the diagonal and in the last column, -1 below
        // the diagonal. Well conditioned, but partial pivoting grows it by 2^(n-1), the classic
        // case where LU's backward error fails and QR's does not.
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
    cases
}

fn applicable(evidence: StructuralEvidence) -> Vec<SolverAction> {
    let mut actions = vec![
        SolverAction::DirectLU,
        SolverAction::PivotedQR,
        SolverAction::SVDFallback,
    ];
    match evidence {
        StructuralEvidence::Diagonal => actions.push(SolverAction::DiagonalFastPath),
        StructuralEvidence::Triangular => actions.push(SolverAction::TriangularFastPath),
        StructuralEvidence::Symmetric => actions.push(SolverAction::SymmetricFastPath),
        StructuralEvidence::General => {}
    }
    actions
}

fn json_f64(value: f64) -> String {
    if value.is_finite() {
        format!("{value:?}")
    } else {
        "null".to_string()
    }
}

/// The 95% Wilson score interval of a failure rate `p` observed over `n` effective samples.
fn wilson95(p: f64, n: f64) -> [f64; 2] {
    const Z: f64 = 1.959_963_984_540_054;
    let z2n = Z * Z / n;
    let centre = (p + z2n / 2.0) / (1.0 + z2n);
    let half = Z * (p * (1.0 - p) / n + z2n / (4.0 * n)).sqrt() / (1.0 + z2n);
    [(centre - half).max(0.0), (centre + half).min(1.0)]
}

fn median_ms(mut run: impl FnMut()) -> f64 {
    run();
    let mut times: Vec<f64> = (0..5)
        .map(|_| {
            let start = Instant::now();
            run();
            start.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[2]
}

fn main() {
    let timing = !std::env::args().any(|arg| arg == "--no-timing");
    let prior = SolverPortfolio::new(RuntimeMode::Strict, 1);
    let mut rng = Rng(SEED ^ 0xB);
    let cases = corpus();

    let mut rows = Vec::new();
    // Per matrix: its state weights and each applicable action's failure.
    let mut per_cell: Vec<([f64; 4], Vec<(SolverAction, bool)>)> = Vec::new();
    let mut errors = [0_usize; 6];
    for case in &cases {
        let n = case.n;
        let x_true: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let b: Vec<f64> = case
            .a
            .iter()
            .map(|row| row.iter().zip(&x_true).map(|(a, x)| a * x).sum())
            .collect();
        let report = condition_diagnostics(&case.a).expect("square finite corpus matrix");
        let rcond = report.rcond_estimate;
        let state_weights = prior.posterior(rcond);
        let x_scale = x_true.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let mut outcomes = Vec::new();
        let mut failed_by_action = Vec::new();
        for action in applicable(report.structural_evidence) {
            let failed = match solve_with_action(&case.a, &b, action) {
                Err(error) => {
                    errors[action.index()] += 1;
                    outcomes.push(format!(
                        "{{\"action\":\"{action:?}\",\"failed\":true,\"error\":{:?}}}",
                        error.to_string()
                    ));
                    true
                }
                Ok(result) => {
                    let omega = result.backward_error.unwrap_or(f64::INFINITY);
                    let finite = result.x.iter().all(|v| v.is_finite());
                    let forward = result
                        .x
                        .iter()
                        .zip(&x_true)
                        .map(|(x, t)| (x - t).abs())
                        .fold(0.0_f64, f64::max)
                        / x_scale;
                    let failed = !finite || !(omega <= ATTEMPT_BACKWARD_ERROR_TOL);
                    outcomes.push(format!(
                        "{{\"action\":\"{action:?}\",\"failed\":{failed},\"backward_error\":{},\"forward_error\":{}}}",
                        json_f64(omega),
                        json_f64(forward)
                    ));
                    failed
                }
            };
            failed_by_action.push((action, failed));
        }
        per_cell.push((state_weights, failed_by_action));
        rows.push(format!(
            "{{\"family\":{:?},\"n\":{n},\"rcond_estimate\":{},\"structural_evidence\":\"{:?}\",\"state_weights\":[{}],\"outcomes\":[{}]}}",
            case.family,
            json_f64(rcond),
            report.structural_evidence,
            state_weights.iter().map(|w| json_f64(*w)).collect::<Vec<_>>().join(","),
            outcomes.join(",")
        ));
    }

    // A matrix on which every applicable action fails (an exactly singular one) says nothing
    // about which to try first; it is counted and left out. On the others, a failure of `a`
    // costs at least the cheapest action that does succeed there, which the fallback reaches.
    // failures[a][s]: state-weighted failure count; penalties[a][s]: state-weighted recovery
    // cost; weights[a][s]: state weight of the matrices `a` applies to; square_weights[a][s]: the
    // sum of their squares, for Kish's effective sample size.
    let mut failures = [[0.0_f64; 4]; 6];
    let mut penalties = [[0.0_f64; 4]; 6];
    let mut weights = [[0.0_f64; 4]; 6];
    let mut square_weights = [[0.0_f64; 4]; 6];
    let mut unsolvable = 0_usize;
    for (state_weights, outcomes) in &per_cell {
        let recovery = outcomes
            .iter()
            .filter(|(_, failed)| !failed)
            .map(|(action, _)| COST[action.index()])
            .fold(f64::INFINITY, f64::min);
        if recovery.is_infinite() {
            unsolvable += 1;
            continue;
        }
        for &(action, failed) in outcomes {
            let a = action.index();
            for s in 0..4 {
                weights[a][s] += state_weights[s];
                square_weights[a][s] += state_weights[s] * state_weights[s];
                if failed {
                    failures[a][s] += state_weights[s];
                    penalties[a][s] += state_weights[s] * recovery;
                }
            }
        }
    }

    // Failure rates and losses; a (state, action) pair with no weight is flagged and priced as
    // an always-failing action recovered by the SVD.
    let mut rates = [[f64::NAN; 4]; 6];
    let mut losses = [[0.0_f64; 4]; 6];
    let mut effective = [[0.0_f64; 4]; 6];
    let mut intervals = [[[f64::NAN; 2]; 4]; 6];
    let mut unmeasured = Vec::new();
    for a in 0..6 {
        for s in 0..4 {
            if weights[a][s] > 0.0 {
                rates[a][s] = failures[a][s] / weights[a][s];
                losses[a][s] = COST[a] + penalties[a][s] / weights[a][s];
                effective[a][s] = weights[a][s] * weights[a][s] / square_weights[a][s];
                intervals[a][s] = wilson95(rates[a][s], effective[a][s]);
            } else {
                unmeasured.push(format!("\"{:?}/{}\"", SolverAction::ALL[a], STATES[s]));
                rates[a][s] = 1.0;
                losses[a][s] = COST[a] + COST[RECOVERY];
            }
        }
    }

    // Reachability: on how many corpus matrices each action has the least expected loss among
    // the actions that apply, and, on the solvable ones, what the fallback reaches when that
    // choice fails. The actions rank by expected loss under the prior, as the first attempt
    // does (the attempt loop re-ranks the rest under the updated posterior).
    let mut chosen = [0_usize; 6];
    let mut chosen_failed = [0_usize; 6];
    let mut fallback_success = [0_usize; 6];
    let mut sole_success = [0_usize; 6];
    for (state_weights, outcomes) in &per_cell {
        let expected = |action: SolverAction| -> f64 {
            (0..4)
                .map(|s| state_weights[s] * losses[action.index()][s])
                .sum()
        };
        let mut ranked = outcomes.clone();
        ranked.sort_by(|x, y| expected(x.0).total_cmp(&expected(y.0)));
        let (first, first_failed) = ranked[0];
        chosen[first.index()] += 1;
        let succeeded: Vec<SolverAction> = ranked
            .iter()
            .filter(|(_, failed)| !failed)
            .map(|(action, _)| *action)
            .collect();
        if let [only] = succeeded.as_slice() {
            sole_success[only.index()] += 1;
        }
        if first_failed && let Some(rescue) = succeeded.first() {
            chosen_failed[first.index()] += 1;
            fallback_success[rescue.index()] += 1;
        }
    }
    // An action that is never chosen and never the fallback that succeeds does no work on this
    // corpus. It is only retained as a later fallback, and the reachability test names each one.
    let unreached: Vec<String> = SolverAction::ALL
        .iter()
        .filter(|a| chosen[a.index()] == 0 && fallback_success[a.index()] == 0)
        .map(|a| format!("\"{a:?}\""))
        .collect();

    let mut timing_json = String::from("null");
    if timing {
        let mut entries = Vec::new();
        for n in [32_usize, 128, 512] {
            let mut trng = Rng(SEED ^ n as u64);
            let general = with_singular_values(&randsvd_sigma(n, 1e2, 3), &mut trng);
            let definite: Vec<f64> = spread(n, 1e2, &mut trng).iter().map(|v| v.abs()).collect();
            let spd = symmetric(&definite, &mut trng);
            let b: Vec<f64> = (0..n).map(|_| trng.normal()).collect();
            let lu = median_ms(|| {
                std::hint::black_box(solve_with_action(&general, &b, SolverAction::DirectLU).ok());
            });
            let mut ratios = Vec::new();
            for (action, matrix) in [
                (SolverAction::PivotedQR, &general),
                (SolverAction::SVDFallback, &general),
                (SolverAction::SymmetricFastPath, &spd),
            ] {
                let ms = median_ms(|| {
                    std::hint::black_box(solve_with_action(matrix, &b, action).ok());
                });
                ratios.push(format!(
                    "\"{action:?}\":{{\"median_ms\":{},\"ratio_to_lu\":{},\"flop_ratio\":{}}}",
                    json_f64(ms),
                    json_f64(ms / lu),
                    json_f64(COST[action.index()])
                ));
            }
            entries.push(format!(
                "\"{n}\":{{\"DirectLU\":{{\"median_ms\":{}}},{}}}",
                json_f64(lu),
                ratios.join(",")
            ));
        }
        timing_json = format!("{{{}}}", entries.join(","));
    }

    let exe = std::fs::read("/proc/self/exe").unwrap_or_default();
    let elf = Fingerprinter::new("casp_calibrate").bytes(&exe).finish();
    let hostname = std::fs::read_to_string("/etc/hostname").unwrap_or_default();
    let threads = std::thread::available_parallelism().map_or(0, |p| p.get());
    let matrix_json = |m: &[[f64; 4]; 6]| {
        m.iter()
            .map(|row| {
                format!(
                    "[{}]",
                    row.iter()
                        .map(|v| json_f64(*v))
                        .collect::<Vec<_>>()
                        .join(",")
                )
            })
            .collect::<Vec<_>>()
            .join(",")
    };
    let counts_json = |counts: &[usize; 6]| {
        SolverAction::ALL
            .iter()
            .map(|a| format!("\"{a:?}\":{}", counts[a.index()]))
            .collect::<Vec<_>>()
            .join(",")
    };
    let intervals_json = intervals
        .iter()
        .map(|row| {
            format!(
                "[{}]",
                row.iter()
                    .map(|[lo, hi]| format!("[{},{}]", json_f64(*lo), json_f64(*hi)))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let actions_json = SolverAction::ALL
        .iter()
        .map(|a| format!("\"{a:?}\""))
        .collect::<Vec<_>>()
        .join(",");
    let states_json = STATES
        .iter()
        .map(|s| format!("\"{s}\""))
        .collect::<Vec<_>>()
        .join(",");
    let mut report = String::new();
    let _ = write!(
        report,
        "{{\n\"bead\":\"frankenscipy-7tb8d.2\",\n\"portfolio\":\"solver\",\n\"corpus_version\":{CORPUS_VERSION},\n\"seed\":\"{SEED:#x}\",\n\"matrices\":{},\n\"sizes\":[8,32,128],\n\"rule\":\"loss(a,s) = cost(a) + E[recovery cost of a failure of a | s]; fail = error, non-finite x, or backward error > {} (ATTEMPT_BACKWARD_ERROR_TOL); a failure's recovery cost is the cost of the cheapest action that succeeds on that matrix; matrices on which every action fails are left out (unsolvable); matrices weigh into states by the portfolio prior at their estimated rcond; cost = leading-order flops relative to LU (QR 4n^3/3, SVD 21n^3, symmetric n^3/3, triangular and diagonal O(n^2))\",\n\"actions\":[{actions_json}],\n\"states\":[{states_json}],\n\"cost\":[{}],\n\"unsolvable\":{unsolvable},\n\"failure_weight\":[{}],\n\"recovery_weight\":[{}],\n\"total_weight\":[{}],\n\"failure_rate\":[{}],\n\"failure_rate_wilson95\":[{intervals_json}],\n\"effective_samples\":[{}],\n\"loss_matrix\":[{}],\n\"unmeasured\":[{}],\n\"chosen_on_corpus\":{{{}}},\n\"chosen_failed\":{{{}}},\n\"fallback_success\":{{{}}},\n\"sole_success\":{{{}}},\n\"error_count\":{{{}}},\n\"unreached\":[{}],\n\"wall_time\":{timing_json},\n\"provenance\":{{\"elf_blake3\":\"{elf}\",\"host\":{:?},\"threads_observed\":{threads},\"avx2\":{},\"fma\":{},\"os\":\"{}\",\"arch\":\"{}\"}},\n\"cells\":[\n{}\n]\n}}\n",
        cases.len(),
        json_f64(ATTEMPT_BACKWARD_ERROR_TOL),
        COST.iter()
            .map(|v| json_f64(*v))
            .collect::<Vec<_>>()
            .join(","),
        matrix_json(&failures),
        matrix_json(&penalties),
        matrix_json(&weights),
        matrix_json(&rates),
        matrix_json(&effective),
        matrix_json(&losses),
        unmeasured.join(","),
        counts_json(&chosen),
        counts_json(&chosen_failed),
        counts_json(&fallback_success),
        counts_json(&sole_success),
        counts_json(&errors),
        unreached.join(","),
        hostname.trim(),
        cfg!(target_feature = "avx2"),
        cfg!(target_feature = "fma"),
        std::env::consts::OS,
        std::env::consts::ARCH,
        rows.join(",\n"),
    );

    let mut module = String::new();
    let _ = write!(
        module,
        "//! GENERATED by `cargo run --release -p fsci-linalg --bin casp_calibrate` (frankenscipy-7tb8d.2).\n\
         //! Do not edit: regenerate it and commit it with `artifacts/casp-calibration-solver.json`,\n\
         //! whose `loss_matrix` a test compares with these values bit for bit.\n\
         //!\n\
         //! Corpus version {CORPUS_VERSION}, seed {SEED:#x}: {} matrices at n = 8, 32 and 128.\n\
         //! loss(a, s) = cost(a) + E[recovery cost of a failure of a | s]. A failure is what makes the\n\
         //! portfolio fall back (an error, a non-finite answer or a backward error above\n\
         //! `ATTEMPT_BACKWARD_ERROR_TOL`), recovered at the cost of the cheapest action that succeeds\n\
         //! on that matrix; cost is leading-order flops relative to LU.\n\
         \n\
         /// `SolverPortfolio`'s loss matrix: one row per action in `SolverAction::ALL` order, one\n\
         /// column per state (well conditioned, moderate, ill conditioned, near singular).\n\
         #[rustfmt::skip]\n\
         pub(crate) const SOLVER_LOSS_MATRIX: [[f64; 4]; 6] = [\n",
        cases.len()
    );
    for (a, row) in losses.iter().enumerate() {
        let _ = writeln!(
            module,
            "    [{}], // {:?}",
            row.iter()
                .map(|v| format!("{v:?}"))
                .collect::<Vec<_>>()
                .join(", "),
            SolverAction::ALL[a]
        );
    }
    module.push_str("];\n");

    println!("===BEGIN casp-calibration-solver.json===");
    print!("{report}");
    println!("===END casp-calibration-solver.json===");
    println!("===BEGIN calibrated_losses.rs===");
    print!("{module}");
    println!("===END calibrated_losses.rs===");
    // On stdout after the markers: rch merges the two streams, so stderr can land inside them.
    println!(
        "casp_calibrate: {} matrices, {unsolvable} unsolvable; failure rates {:?}; chosen {:?}; \
         fallback succeeds {:?}; unreached {:?}; unmeasured {:?}",
        cases.len(),
        rates,
        chosen,
        fallback_success,
        unreached,
        unmeasured
    );
}

#[cfg(test)]
mod tests {
    use super::{Case, corpus};

    fn entries(cases: &[Case]) -> Vec<(String, usize, Vec<u64>)> {
        cases
            .iter()
            .map(|case| {
                let bits = case.a.iter().flatten().map(|v| v.to_bits()).collect();
                (case.family.clone(), case.n, bits)
            })
            .collect()
    }

    /// The acceptance criterion rests on this: the same seed regenerates the same corpus, bit
    /// for bit, so the constants regenerate byte for byte.
    #[test]
    fn corpus_is_deterministic() {
        let first = entries(&corpus());
        let second = entries(&corpus());
        assert_eq!(first.len(), second.len());
        assert!(first == second, "two corpora from one seed differ");

        // The comparison is on bits: it sees a one-ULP change, and a lost sign of zero, which
        // `==` on f64 accepts.
        let mut ulp = second.clone();
        let (_, _, bits) = ulp
            .iter_mut()
            .find(|(family, _, _)| family.starts_with("randsvd"))
            .expect("a randsvd matrix");
        bits[0] ^= 1;
        assert!(first != ulp, "a one-ULP change went unseen");
        let mut sign = second;
        let zero = sign
            .iter_mut()
            .flat_map(|(_, _, bits)| bits.iter_mut())
            .find(|bits| **bits == 0.0_f64.to_bits())
            .expect("a zero entry");
        *zero = (-0.0_f64).to_bits();
        assert!(first != sign, "a signed zero went unseen");
    }
}
