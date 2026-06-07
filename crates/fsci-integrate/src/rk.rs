#![forbid(unsafe_code)]
#![allow(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    clippy::inconsistent_digit_grouping
)]

//! Explicit Runge-Kutta ODE solvers (RK23, RK45, DOP853).
//!
//! Implements the Dormand-Prince RK5(4) and Bogacki-Shampine RK3(2) methods
//! with adaptive step-size control matching SciPy's `_ivp/rk.py`.

use fsci_runtime::RuntimeMode;

use crate::solver::{OdeSolver, OdeSolverState, StepFailure, StepOutcome};
use crate::step_size::InitialStepRequest;
use crate::validation::{ToleranceValue, validate_tol};
use crate::{
    IntegrateValidationError, select_initial_step, validate_first_step, validate_max_step,
};

// Step-size control constants (from SciPy's rk.py).
const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 10.0;

type OdeFn = Box<dyn FnMut(f64, &[f64]) -> Vec<f64>>;

/// Butcher tableau for an explicit Runge-Kutta method.
pub struct ButcherTableau {
    /// A coefficients (lower-triangular, row-major, n_stages × n_stages).
    pub a: &'static [&'static [f64]],
    /// B coefficients (weights for combining stages, length n_stages).
    pub b: &'static [f64],
    /// C coefficients (time increments, length n_stages).
    pub c: &'static [f64],
    /// E coefficients (error estimation, length n_stages + 1).
    pub e: &'static [f64],
    /// Number of stages.
    pub n_stages: usize,
    /// Order of the main method.
    pub order: usize,
    /// Order of the error estimator.
    pub error_estimator_order: usize,
}

// ═══════════════════════════════════════════════════════════════
// RK45: Dormand-Prince 5(4) Butcher tableau
// ═══════════════════════════════════════════════════════════════

static RK45_C: &[f64] = &[0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];

static RK45_A0: &[f64] = &[];
static RK45_A1: &[f64] = &[1.0 / 5.0];
static RK45_A2: &[f64] = &[3.0 / 40.0, 9.0 / 40.0];
static RK45_A3: &[f64] = &[44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0];
static RK45_A4: &[f64] = &[
    19372.0 / 6561.0,
    -25360.0 / 2187.0,
    64448.0 / 6561.0,
    -212.0 / 729.0,
];
static RK45_A5: &[f64] = &[
    9017.0 / 3168.0,
    -355.0 / 33.0,
    46732.0 / 5247.0,
    49.0 / 176.0,
    -5103.0 / 18656.0,
];
static RK45_A6: &[f64] = &[
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
];

static RK45_A: &[&[f64]] = &[
    RK45_A0, RK45_A1, RK45_A2, RK45_A3, RK45_A4, RK45_A5, RK45_A6,
];

static RK45_B: &[f64] = &[
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
];

static RK45_E: &[f64] = &[
    -71.0 / 57600.0,
    0.0,
    71.0 / 16695.0,
    -71.0 / 1920.0,
    17253.0 / 339200.0,
    -22.0 / 525.0,
    1.0 / 40.0,
];

pub static RK45_TABLEAU: ButcherTableau = ButcherTableau {
    a: RK45_A,
    b: RK45_B,
    c: RK45_C,
    e: RK45_E,
    n_stages: 7,
    order: 5,
    error_estimator_order: 4,
};

// ═══════════════════════════════════════════════════════════════
// RK23: Bogacki-Shampine 3(2) Butcher tableau
// ═══════════════════════════════════════════════════════════════

static RK23_C: &[f64] = &[0.0, 1.0 / 2.0, 3.0 / 4.0, 1.0];

static RK23_A0: &[f64] = &[];
static RK23_A1: &[f64] = &[1.0 / 2.0];
static RK23_A2: &[f64] = &[0.0, 3.0 / 4.0];
static RK23_A3: &[f64] = &[2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0];

static RK23_A: &[&[f64]] = &[RK23_A0, RK23_A1, RK23_A2, RK23_A3];

static RK23_B: &[f64] = &[2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0];

static RK23_E: &[f64] = &[5.0 / 72.0, -1.0 / 12.0, -1.0 / 9.0, 1.0 / 8.0];

pub static RK23_TABLEAU: ButcherTableau = ButcherTableau {
    a: RK23_A,
    b: RK23_B,
    c: RK23_C,
    e: RK23_E,
    n_stages: 4,
    order: 3,
    error_estimator_order: 2,
};

// ═══════════════════════════════════════════════════════════════
// DOP853: Dormand-Prince 8(5,3) Butcher tableau
// ═══════════════════════════════════════════════════════════════

// DOP853 has 12 stages. Coefficients from Hairer, Nørsett, Wanner
// "Solving Ordinary Differential Equations I", 2nd ed.

static DOP853_C: &[f64] = &[
    0.0,
    0.526_001_519_587_677_3e-1,
    0.789_002_279_381_515_9e-1,
    0.118_350_341_907_227_4,
    0.281_649_658_092_772_6,
    0.333_333_333_333_333_3,
    0.25,
    0.307_692_307_692_307_7,
    0.651_282_051_282_051_3,
    0.6,
    0.857_142_857_142_857_1,
    1.0,
];

// A matrix: exact coefficients from scipy/integrate/_ivp/dop853_coefficients.py
static DOP853_A0: &[f64] = &[];
static DOP853_A1: &[f64] = &[5.26001519587677318785587544488e-2];
static DOP853_A2: &[f64] = &[
    1.97250569845378994544595329183e-2,
    5.91751709536136983633785987549e-2,
];
static DOP853_A3: &[f64] = &[
    2.95875854768068491816892993775e-2,
    0.0,
    8.87627564304205475450678981324e-2,
];
static DOP853_A4: &[f64] = &[
    2.41365134159266685502369798665e-1,
    0.0,
    -8.84549479328286085344864962717e-1,
    9.24834003261792003115737966543e-1,
];
static DOP853_A5: &[f64] = &[
    3.7037037037037037037037037037e-2,
    0.0,
    0.0,
    1.70828608729473871279604482173e-1,
    1.25467687566822425016691814123e-1,
];
static DOP853_A6: &[f64] = &[
    3.7109375e-2,
    0.0,
    0.0,
    1.70252211019544039314978060272e-1,
    6.02165389804559606850219397283e-2,
    -1.7578125e-2,
];
static DOP853_A7: &[f64] = &[
    3.70920001185047927108779319836e-2,
    0.0,
    0.0,
    1.70383925712239993810214054705e-1,
    1.07262030446373284651809199168e-1,
    -1.53194377486244017527936158236e-2,
    8.27378916381402288758473766002e-3,
];
static DOP853_A8: &[f64] = &[
    6.24110958716075717114429577812e-1,
    0.0,
    0.0,
    -3.36089262944694129406857109825e0,
    -8.68219346841726006818189891453e-1,
    2.75920996994467083049415600797e1,
    2.01540675504778934086186788979e1,
    -4.34898841810699588477366255144e1,
];
static DOP853_A9: &[f64] = &[
    4.77662536438264365890433908527e-1,
    0.0,
    0.0,
    -2.48811461997166764192642586468e0,
    -5.90290826836842996371446475743e-1,
    2.12300514481811942347288949897e1,
    1.52792336328824235832596922938e1,
    -3.32882109689848629194453265587e1,
    -2.03312017085086261358222928593e-2,
];
static DOP853_A10: &[f64] = &[
    -9.3714243008598732571704021658e-1,
    0.0,
    0.0,
    5.18637242884406370830023853209e0,
    1.09143734899672957818500254654e0,
    -8.14978701074692612513997267357e0,
    -1.85200656599969598641566180701e1,
    2.27394870993505042818970056734e1,
    2.49360555267965238987089396762e0,
    -3.0467644718982195003823669022e0,
];
static DOP853_A11: &[f64] = &[
    2.27331014751653820792359768449e0,
    0.0,
    0.0,
    -1.05344954667372501984066689879e1,
    -2.00087205822486249909675718444e0,
    -1.79589318631187989172765950534e1,
    2.79488845294199600508499808837e1,
    -2.85899827713502369474065508674e0,
    -8.87285693353062954433549289258e0,
    1.23605671757943030647266201528e1,
    6.43392746015763530355970484046e-1,
];

static DOP853_A: &[&[f64]] = &[
    DOP853_A0, DOP853_A1, DOP853_A2, DOP853_A3, DOP853_A4, DOP853_A5, DOP853_A6, DOP853_A7,
    DOP853_A8, DOP853_A9, DOP853_A10, DOP853_A11,
];

// 8th-order solution weights: B = A[12, 0:12] from the extended 16-stage tableau.
// Source: scipy/integrate/_ivp/dop853_coefficients.py
static DOP853_B: &[f64] = &[
    5.42937341165687622380535766363e-2,
    0.0,
    0.0,
    0.0,
    0.0,
    4.45031289275240888144113950566e0,
    1.89151789931450038304281599044e0,
    -5.8012039600105847814672114227e0,
    3.1116436695781989440891606237e-1,
    -1.52160949662516078556178806805e-1,
    2.01365400804030348374776537501e-1,
    4.47106157277725905176885569043e-2,
];

// Error estimation weights (E5): 5th-order embedded error from SciPy.
// E[12] = 0.0 since k[12] = f_new (FSAL).
// Source: scipy/integrate/_ivp/dop853_coefficients.py
static DOP853_E: &[f64] = &[
    0.1312004499419488073250102996e-1,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.1225156446376204440720569753e1,
    -0.4957589496572501915214079952e0,
    0.1664377182454986536961530415e1,
    -0.3503288487499736816886487290e0,
    0.3341791187130174790297318841e0,
    0.8192320648511571246570742613e-1,
    -0.2235530786388629525884427845e-1,
    0.0,
];

pub static DOP853_TABLEAU: ButcherTableau = ButcherTableau {
    a: DOP853_A,
    b: DOP853_B,
    c: DOP853_C,
    e: DOP853_E,
    n_stages: 12,
    order: 8,
    error_estimator_order: 7,
};

// ═══════════════════════════════════════════════════════════════
// Core RK step function
// ═══════════════════════════════════════════════════════════════

/// Perform a single explicit Runge-Kutta step.
///
/// Returns `(y_new, f_new)` where `f_new = fun(t + h, y_new)`.
/// `k` is filled with the stage derivatives (k[0] = f, k[n_stages] = f_new).
fn rk_step<F>(
    fun: &mut F,
    t: f64,
    y: &[f64],
    f: &[f64],
    h: f64,
    tableau: &ButcherTableau,
    k: &mut [Vec<f64>],
) -> (Vec<f64>, Vec<f64>)
where
    F: FnMut(f64, &[f64]) -> Vec<f64> + ?Sized,
{
    let n = y.len();
    k[0] = f.to_vec();

    for s in 1..tableau.n_stages {
        let a_row = tableau.a[s];
        let c_s = tableau.c[s];
        let mut dy = vec![0.0; n];
        for (j, &a_sj) in a_row.iter().enumerate() {
            if a_sj != 0.0 {
                for i in 0..n {
                    dy[i] += a_sj * k[j][i];
                }
            }
        }
        let y_stage: Vec<f64> = y
            .iter()
            .zip(dy.iter())
            .map(|(yi, di)| yi + h * di)
            .collect();
        k[s] = fun(t + c_s * h, &y_stage);
    }

    // y_new = y + h * sum(B[s] * K[s])
    let mut y_new = y.to_vec();
    for (s, &b_s) in tableau.b.iter().enumerate() {
        if b_s != 0.0 {
            for i in 0..n {
                y_new[i] += h * b_s * k[s][i];
            }
        }
    }

    // FSAL (First Same As Last) property: for methods like Dormand-Prince RK45
    // and Bogacki-Shampine RK23, the last stage k[n_stages-1] is the derivative
    // at (t+h, y_new).
    let f_new = k[tableau.n_stages - 1].clone();
    k[tableau.n_stages] = f_new.clone(); // store for next step if needed

    (y_new, f_new)
}

fn error_scale(atol: f64, rtol: f64, y: f64, y_new: f64) -> f64 {
    let a = y.abs();
    let b = y_new.abs();
    let max_val = if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    };
    atol + max_val * rtol
}

fn estimate_error_component(k: &[Vec<f64>], e: &[f64], h: f64, component: usize) -> f64 {
    let mut err = 0.0;
    for (s, &e_s) in e.iter().enumerate() {
        if e_s != 0.0 {
            err += e_s * k[s][component];
        }
    }
    err * h
}

// ═══════════════════════════════════════════════════════════════
// Runge-Kutta solver state machine
// ═══════════════════════════════════════════════════════════════

/// Configuration for constructing an RK solver.
pub struct RkSolverConfig<'a> {
    pub t0: f64,
    pub y0: &'a [f64],
    pub t_bound: f64,
    pub rtol: f64,
    pub atol: ToleranceValue,
    pub max_step: f64,
    pub first_step: Option<f64>,
    pub mode: RuntimeMode,
    pub tableau: &'static ButcherTableau,
}

/// An explicit Runge-Kutta ODE solver with adaptive step-size control.
///
/// Supports RK45 (Dormand-Prince) and RK23 (Bogacki-Shampine) via
/// interchangeable Butcher tableaux.
pub struct RkSolver {
    mode: RuntimeMode,
    state: OdeSolverState,
    tableau: &'static ButcherTableau,
    // Problem definition
    n: usize,
    t: f64,
    y: Vec<f64>,
    t_old: Option<f64>,
    y_old: Option<Vec<f64>>,
    t_bound: f64,
    direction: f64,
    // Tolerances
    rtol: f64,
    atol: ToleranceValue,
    max_step: f64,
    // Solver state
    f: Vec<f64>,
    f_old: Option<Vec<f64>>,
    h_abs: f64,
    error_exponent: f64,
    k: Vec<Vec<f64>>,
    // Statistics
    nfev: usize,
    // Stored ODE function for trait-object stepping (Some if created with new_owned)
    fun: Option<OdeFn>,
}

impl RkSolver {
    /// Create a new RK solver from configuration (reference-based).
    ///
    /// Use this constructor when you want to use `step_with(fun)` to advance
    /// the solver. For trait-object stepping via `OdeSolver::step()`, use
    /// `new_owned` instead.
    ///
    /// # Contract (P2C-001-D2)
    /// - Validates tolerances and step parameters.
    /// - Calls `select_initial_step` if no `first_step` is provided.
    /// - Threads `RuntimeMode` through all validation paths.
    pub fn new<F>(fun: &mut F, config: RkSolverConfig<'_>) -> Result<Self, IntegrateValidationError>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        let n = config.y0.len();

        // Validate tolerances
        let validated_tol = validate_tol(
            ToleranceValue::Scalar(config.rtol),
            config.atol.clone(),
            n,
            config.mode,
        )?;
        let rtol = validated_tol
            .rtol
            .into_scalar()
            .expect("RkSolverConfig always carries scalar rtol");
        let atol = validated_tol.atol;

        // Validate max_step (per frankenscipy-ljmg / i9vw): NaN fails
        // closed via validate_max_step; Inf is accepted as "no cap" per
        // scipy convention (default max_step = Inf).
        if config.max_step.is_finite() || config.max_step.is_nan() {
            validate_max_step(config.max_step)?;
        }

        let direction = if config.t_bound != config.t0 {
            (config.t_bound - config.t0).signum()
        } else {
            1.0
        };

        // Evaluate f0
        let f0 = fun(config.t0, config.y0);
        let nfev = 1;

        // Determine initial step size
        let h_abs = if let Some(first_step) = config.first_step {
            validate_first_step(first_step, config.t0, config.t_bound)?
        } else {
            let step_request = InitialStepRequest {
                t0: config.t0,
                y0: config.y0,
                t_bound: config.t_bound,
                max_step: config.max_step,
                f0: &f0,
                direction,
                order: config.tableau.error_estimator_order as f64,
                rtol,
                atol: atol.clone(),
                mode: config.mode,
            };
            select_initial_step(fun, &step_request)?
        };

        let error_exponent = -1.0 / (config.tableau.error_estimator_order as f64 + 1.0);

        // Allocate K storage: n_stages + 1 vectors of length n
        let k = vec![vec![0.0; n]; config.tableau.n_stages + 1];

        Ok(Self {
            mode: config.mode,
            state: OdeSolverState::Running,
            tableau: config.tableau,
            n,
            t: config.t0,
            y: config.y0.to_vec(),
            t_old: None,
            y_old: None,
            t_bound: config.t_bound,
            direction,
            rtol,
            atol,
            max_step: config.max_step,
            f: f0,
            f_old: None,
            h_abs,
            error_exponent,
            k,
            nfev,
            fun: None,
        })
    }

    /// Create a new RK solver that owns its ODE function.
    ///
    /// This constructor enables the `OdeSolver::step()` trait method for
    /// trait-object stepping. The solver stores the function internally.
    ///
    /// # Contract (P2C-001-D2)
    /// - Validates tolerances and step parameters.
    /// - Calls `select_initial_step` if no `first_step` is provided.
    /// - Threads `RuntimeMode` through all validation paths.
    /// - Stores the ODE function for use by `OdeSolver::step()`.
    pub fn new_owned<F>(
        fun: F,
        config: RkSolverConfig<'_>,
    ) -> Result<Self, IntegrateValidationError>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64> + 'static,
    {
        let n = config.y0.len();

        // Validate tolerances
        let validated_tol = validate_tol(
            ToleranceValue::Scalar(config.rtol),
            config.atol.clone(),
            n,
            config.mode,
        )?;
        let rtol = validated_tol
            .rtol
            .into_scalar()
            .expect("RkSolverConfig always carries scalar rtol");
        let atol = validated_tol.atol;

        // Validate max_step (per frankenscipy-ljmg / i9vw): NaN fails
        // closed via validate_max_step; Inf is accepted as "no cap" per
        // scipy convention (default max_step = Inf).
        if config.max_step.is_finite() || config.max_step.is_nan() {
            validate_max_step(config.max_step)?;
        }

        let direction = if config.t_bound != config.t0 {
            (config.t_bound - config.t0).signum()
        } else {
            1.0
        };

        // Box the function for storage and initial evaluation
        let mut fun_box: OdeFn = Box::new(fun);

        // Evaluate f0
        let f0 = fun_box(config.t0, config.y0);
        let nfev = 1;

        // Determine initial step size
        let h_abs = if let Some(first_step) = config.first_step {
            validate_first_step(first_step, config.t0, config.t_bound)?
        } else {
            let step_request = InitialStepRequest {
                t0: config.t0,
                y0: config.y0,
                t_bound: config.t_bound,
                max_step: config.max_step,
                f0: &f0,
                direction,
                order: config.tableau.error_estimator_order as f64,
                rtol,
                atol: atol.clone(),
                mode: config.mode,
            };
            select_initial_step(&mut *fun_box, &step_request)?
        };

        let error_exponent = -1.0 / (config.tableau.error_estimator_order as f64 + 1.0);

        // Allocate K storage: n_stages + 1 vectors of length n
        let k = vec![vec![0.0; n]; config.tableau.n_stages + 1];

        Ok(Self {
            mode: config.mode,
            state: OdeSolverState::Running,
            tableau: config.tableau,
            n,
            t: config.t0,
            y: config.y0.to_vec(),
            t_old: None,
            y_old: None,
            t_bound: config.t_bound,
            direction,
            rtol,
            atol,
            max_step: config.max_step,
            f: f0,
            f_old: None,
            h_abs,
            error_exponent,
            k,
            nfev,
            fun: Some(fun_box),
        })
    }

    /// Current derivative.
    pub fn f(&self) -> &[f64] {
        &self.f
    }

    /// Previous derivative (after at least one successful step).
    pub fn f_old(&self) -> Option<&[f64]> {
        self.f_old.as_deref()
    }

    /// Current time.
    pub fn t(&self) -> f64 {
        self.t
    }

    /// Current state vector.
    pub fn y(&self) -> &[f64] {
        &self.y
    }

    /// Previous time (after at least one successful step).
    pub fn t_old(&self) -> Option<f64> {
        self.t_old
    }

    /// Previous state vector (after at least one successful step).
    pub fn y_old(&self) -> Option<&[f64]> {
        self.y_old.as_deref()
    }

    /// Number of function evaluations.
    pub fn nfev(&self) -> usize {
        self.nfev
    }

    /// Returns true if this solver supports standalone `OdeSolver::step()`.
    ///
    /// Solvers created with `new_owned()` return true; those created with
    /// `new()` return false (use `step_with()` instead).
    #[must_use]
    pub fn can_step(&self) -> bool {
        self.fun.is_some()
    }

    /// Perform one adaptive step, advancing the solver.
    ///
    /// This is the core step implementation matching SciPy's `RungeKutta._step_impl`.
    pub fn step_with<F>(&mut self, fun: &mut F) -> Result<StepOutcome, StepFailure>
    where
        F: FnMut(f64, &[f64]) -> Vec<f64>,
    {
        if self.state != OdeSolverState::Running {
            return Err(StepFailure::RuntimeError(
                "Attempt to step on a finished or failed solver.",
            ));
        }

        // Handle empty system or already at boundary
        if self.n == 0 || self.t == self.t_bound {
            self.t_old = Some(self.t);
            self.y_old = Some(self.y.clone());
            self.f_old = Some(self.f.clone());
            self.t = self.t_bound;
            self.state = OdeSolverState::Finished;
            return Ok(StepOutcome {
                message: None,
                state: OdeSolverState::Finished,
            });
        }

        let t = self.t;

        // Minimum step: 10 * machine epsilon at current t
        let min_step = 10.0 * (next_after(t, self.direction * f64::INFINITY) - t).abs();

        let mut h_abs = self.h_abs.clamp(0.0, self.max_step);

        let mut step_accepted = false;
        let mut step_rejected = false;

        while !step_accepted {
            if h_abs < min_step {
                // If we are extremely close to the boundary, we can allow one final tiny step
                let remaining = (self.t_bound - t).abs();
                if remaining < min_step {
                    h_abs = remaining;
                } else {
                    self.state = OdeSolverState::Failed;
                    return Err(StepFailure::StepSizeTooSmall);
                }
            }

            let h = h_abs * self.direction;
            let mut t_new = t + h;

            // Clamp to boundary
            if self.direction * (t_new - self.t_bound) > 0.0 {
                t_new = self.t_bound;
            }

            let h = t_new - t;
            h_abs = h.abs();

            // Perform the RK step
            let (y_new, f_new) = rk_step(fun, t, &self.y, &self.f, h, self.tableau, &mut self.k);
            self.nfev += self.tableau.n_stages - 1; // leverage FSAL: exactly n_stages - 1 new evals per step

            let err_norm = self.estimate_error_norm(h, &y_new);

            if !err_norm.is_finite() {
                self.state = OdeSolverState::Failed;
                return Err(StepFailure::NonFiniteState);
            }

            if err_norm < 1.0 {
                // Step accepted
                let factor = if err_norm == 0.0 {
                    MAX_FACTOR
                } else {
                    MAX_FACTOR.min(SAFETY * err_norm.powf(self.error_exponent))
                };

                let factor = if step_rejected {
                    factor.min(1.0)
                } else {
                    factor
                };

                h_abs *= factor;
                step_accepted = true;

                self.y_old = Some(self.y.clone());
                self.t_old = Some(t);
                self.f_old = Some(self.f.clone());
                self.t = t_new;
                self.y = y_new;
                self.h_abs = h_abs;
                self.f = f_new;
            } else {
                // Step rejected: decrease step size
                h_abs *= MIN_FACTOR.max(SAFETY * err_norm.powf(self.error_exponent));
                step_rejected = true;
            }
        }

        if self.direction * (self.t - self.t_bound) >= 0.0 {
            self.state = OdeSolverState::Finished;
        }

        Ok(StepOutcome {
            message: None,
            state: self.state,
        })
    }

    /// Compute the RMS error norm without materializing the per-component error and scale vectors.
    fn estimate_error_norm(&self, h: f64, y_new: &[f64]) -> f64 {
        if self.n == 0 {
            return 0.0;
        }

        let mut sum_sq = 0.0;
        match &self.atol {
            ToleranceValue::Scalar(atol) => {
                let scale_len = self.n.min(self.y.len()).min(y_new.len());
                for (i, (&y_i, &y_new_i)) in
                    self.y.iter().zip(y_new.iter()).enumerate().take(scale_len)
                {
                    let err = estimate_error_component(&self.k, self.tableau.e, h, i);
                    let scale = error_scale(*atol, self.rtol, y_i, y_new_i);
                    sum_sq += (err / scale) * (err / scale);
                }
            }
            ToleranceValue::Vector(atol_vec) => {
                let scale_len = self
                    .n
                    .min(self.y.len())
                    .min(y_new.len())
                    .min(atol_vec.len());
                for (i, ((&y_i, &y_new_i), &atol_i)) in self
                    .y
                    .iter()
                    .zip(y_new.iter())
                    .zip(atol_vec.iter())
                    .enumerate()
                    .take(scale_len)
                {
                    let err = estimate_error_component(&self.k, self.tableau.e, h, i);
                    let scale = error_scale(atol_i, self.rtol, y_i, y_new_i);
                    sum_sq += (err / scale) * (err / scale);
                }
            }
        }

        (sum_sq / self.n as f64).sqrt()
    }

    /// Internal step implementation using the stored ODE function.
    ///
    /// This enables the `OdeSolver::step()` trait method to work for trait-object usage.
    /// Requires the solver to be created with `new_owned`.
    fn step_impl(&mut self) -> Result<StepOutcome, StepFailure> {
        // Check early that we have a stored function
        if self.fun.is_none() {
            return Err(StepFailure::NotYetImplemented(
                "OdeSolver::step requires solver created with new_owned; use step_with instead",
            ));
        }

        if self.state != OdeSolverState::Running {
            return Err(StepFailure::RuntimeError(
                "Attempt to step on a finished or failed solver.",
            ));
        }

        // Handle empty system or already at boundary
        if self.n == 0 || self.t == self.t_bound {
            self.t_old = Some(self.t);
            self.y_old = Some(self.y.clone());
            self.f_old = Some(self.f.clone());
            self.t = self.t_bound;
            self.state = OdeSolverState::Finished;
            return Ok(StepOutcome {
                message: None,
                state: OdeSolverState::Finished,
            });
        }

        let t = self.t;

        // Minimum step: 10 * machine epsilon at current t
        let min_step = 10.0 * (next_after(t, self.direction * f64::INFINITY) - t).abs();

        let mut h_abs = self.h_abs.clamp(0.0, self.max_step);

        let mut step_accepted = false;
        let mut step_rejected = false;

        while !step_accepted {
            if h_abs < min_step {
                // If we are extremely close to the boundary, we can allow one final tiny step
                let remaining = (self.t_bound - t).abs();
                if remaining < min_step {
                    h_abs = remaining;
                } else {
                    self.state = OdeSolverState::Failed;
                    return Err(StepFailure::StepSizeTooSmall);
                }
            }

            let h = h_abs * self.direction;
            let mut t_new = t + h;

            // Clamp to boundary
            if self.direction * (t_new - self.t_bound) > 0.0 {
                t_new = self.t_bound;
            }

            let h = t_new - t;
            h_abs = h.abs();

            // Perform the RK step using stored function (borrow only for this call)
            let (y_new, f_new) = {
                let fun = self.fun.as_mut().unwrap();
                rk_step(
                    &mut **fun,
                    t,
                    &self.y,
                    &self.f,
                    h,
                    self.tableau,
                    &mut self.k,
                )
            };
            self.nfev += self.tableau.n_stages - 1;

            let err_norm = self.estimate_error_norm(h, &y_new);

            if !err_norm.is_finite() {
                self.state = OdeSolverState::Failed;
                return Err(StepFailure::NonFiniteState);
            }

            if err_norm < 1.0 {
                // Step accepted
                let factor = if err_norm == 0.0 {
                    MAX_FACTOR
                } else {
                    MAX_FACTOR.min(SAFETY * err_norm.powf(self.error_exponent))
                };

                let factor = if step_rejected {
                    factor.min(1.0)
                } else {
                    factor
                };

                h_abs *= factor;
                step_accepted = true;

                self.y_old = Some(self.y.clone());
                self.t_old = Some(t);
                self.f_old = Some(self.f.clone());
                self.t = t_new;
                self.y = y_new;
                self.h_abs = h_abs;
                self.f = f_new;
            } else {
                // Step rejected: decrease step size
                h_abs *= MIN_FACTOR.max(SAFETY * err_norm.powf(self.error_exponent));
                step_rejected = true;
            }
        }

        if self.direction * (self.t - self.t_bound) >= 0.0 {
            self.state = OdeSolverState::Finished;
        }

        Ok(StepOutcome {
            message: None,
            state: self.state,
        })
    }
}

/// `next_after` equivalent: the next representable f64 toward `toward`.
fn next_after(from: f64, toward: f64) -> f64 {
    if from == toward {
        return from;
    }
    if from.is_nan() || toward.is_nan() {
        return f64::NAN;
    }
    if from == 0.0 {
        if toward > 0.0 {
            return f64::from_bits(1);
        }
        return -f64::from_bits(1);
    }
    let bits = from.to_bits();
    let next_bits = if (toward > from) == (from > 0.0) {
        bits + 1
    } else {
        bits - 1
    };
    f64::from_bits(next_bits)
}

impl OdeSolver for RkSolver {
    fn mode(&self) -> RuntimeMode {
        self.mode
    }

    fn state(&self) -> OdeSolverState {
        self.state
    }

    fn step(&mut self) -> Result<StepOutcome, StepFailure> {
        self.step_impl()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rk_step_single_euler_like() {
        // Trivial test: single stage with A=[], B=[1], C=[0] would be Euler
        // But we test with the actual RK45 tableau on a simple system
        let n = 1;
        let mut k = vec![vec![0.0; n]; RK45_TABLEAU.n_stages + 1];
        let (y_new, _f_new) = rk_step(
            &mut |_t, y| vec![-0.5 * y[0]],
            0.0,
            &[2.0],
            &[-1.0], // f(0, 2) = -1
            0.1,
            &RK45_TABLEAU,
            &mut k,
        );
        // y_new should be close to 2 * exp(-0.5 * 0.1) ≈ 1.90245
        assert!(
            (y_new[0] - 2.0 * (-0.05_f64).exp()).abs() < 1e-6,
            "RK45 step should be accurate for exp decay, got {}",
            y_new[0]
        );
    }

    #[test]
    fn rk23_step_exponential() {
        let n = 1;
        let mut k = vec![vec![0.0; n]; RK23_TABLEAU.n_stages + 1];
        let (y_new, _f_new) = rk_step(
            &mut |_t, y| vec![-0.5 * y[0]],
            0.0,
            &[2.0],
            &[-1.0],
            0.1,
            &RK23_TABLEAU,
            &mut k,
        );
        // RK23 is order 3, so the error should be O(h^4) ≈ 1e-4
        assert!(
            (y_new[0] - 2.0 * (-0.05_f64).exp()).abs() < 1e-4,
            "RK23 step should be reasonably accurate, got {}",
            y_new[0]
        );
    }

    #[test]
    fn solver_exponential_decay() {
        // y' = -y, y(0) = 1, integrate to t=1
        let mut fun = |_t: f64, y: &[f64]| -> Vec<f64> { vec![-y[0]] };
        let config = RkSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 1.0,
            rtol: 1e-8,
            atol: ToleranceValue::Scalar(1e-10),
            max_step: f64::INFINITY,
            first_step: None,
            mode: RuntimeMode::Strict,
            tableau: &RK45_TABLEAU,
        };
        let mut solver = RkSolver::new(&mut fun, config).expect("solver creation should succeed");
        assert_eq!(solver.state(), OdeSolverState::Running);

        let mut steps = 0;
        while solver.state() == OdeSolverState::Running {
            solver
                .step_with(&mut |_t, y| vec![-y[0]])
                .expect("step should succeed");
            steps += 1;
            assert!(steps <= 1000, "too many steps");
        }

        assert_eq!(solver.state(), OdeSolverState::Finished);
        let expected = (-1.0_f64).exp();
        assert!(
            (solver.y()[0] - expected).abs() < 1e-6,
            "y(1) should be close to e^-1 ≈ {expected}, got {}",
            solver.y()[0]
        );
    }

    #[test]
    fn solver_rejects_nan_max_step() {
        let mut fun = |_t: f64, y: &[f64]| -> Vec<f64> { vec![-y[0]] };
        let config = RkSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 1.0,
            rtol: 1e-8,
            atol: ToleranceValue::Scalar(1e-10),
            max_step: f64::NAN,
            first_step: None,
            mode: RuntimeMode::Strict,
            tableau: &RK45_TABLEAU,
        };
        let result = RkSolver::new(&mut fun, config);
        assert!(matches!(
            result,
            Err(IntegrateValidationError::NonFiniteMaxStep)
        ));
    }

    #[test]
    fn solver_two_component_system() {
        // y' = [y[1], -y[0]] (simple harmonic oscillator)
        // y(0) = [1, 0], exact: y(t) = [cos(t), -sin(t)]
        let mut fun = |_t: f64, y: &[f64]| -> Vec<f64> { vec![y[1], -y[0]] };
        let config = RkSolverConfig {
            t0: 0.0,
            y0: &[1.0, 0.0],
            t_bound: std::f64::consts::PI,
            rtol: 1e-8,
            atol: ToleranceValue::Scalar(1e-10),
            max_step: f64::INFINITY,
            first_step: None,
            mode: RuntimeMode::Strict,
            tableau: &RK45_TABLEAU,
        };
        let mut solver = RkSolver::new(&mut fun, config).expect("solver creation should succeed");

        let mut steps = 0;
        while solver.state() == OdeSolverState::Running {
            solver
                .step_with(&mut |_t, y| vec![y[1], -y[0]])
                .expect("step should succeed");
            steps += 1;
            assert!(steps <= 1000, "too many steps");
        }

        // At t=pi: y ≈ [-1, 0]
        assert!(
            (solver.y()[0] - (-1.0)).abs() < 1e-6,
            "y[0](pi) should be close to -1, got {}",
            solver.y()[0]
        );
        assert!(
            solver.y()[1].abs() < 1e-5,
            "y[1](pi) should be close to 0, got {}",
            solver.y()[1]
        );
    }

    #[test]
    fn solver_empty_system() {
        let mut fun = |_t: f64, _y: &[f64]| -> Vec<f64> { vec![] };
        let config = RkSolverConfig {
            t0: 0.0,
            y0: &[],
            t_bound: 1.0,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            max_step: f64::INFINITY,
            first_step: None,
            mode: RuntimeMode::Strict,
            tableau: &RK45_TABLEAU,
        };
        let mut solver = RkSolver::new(&mut fun, config).expect("solver creation should succeed");
        let outcome = solver
            .step_with(&mut |_t, _y| vec![])
            .expect("step should succeed");
        assert_eq!(outcome.state, OdeSolverState::Finished);
    }

    #[test]
    fn solver_with_first_step() {
        let mut fun = |_t: f64, y: &[f64]| -> Vec<f64> { vec![-y[0]] };
        let config = RkSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 1.0,
            rtol: 1e-6,
            atol: ToleranceValue::Scalar(1e-8),
            max_step: f64::INFINITY,
            first_step: Some(0.01),
            mode: RuntimeMode::Strict,
            tableau: &RK45_TABLEAU,
        };
        let solver = RkSolver::new(&mut fun, config).expect("solver creation should succeed");
        assert_eq!(solver.state(), OdeSolverState::Running);
    }

    #[test]
    fn solver_rk23_exponential() {
        let mut fun = |_t: f64, y: &[f64]| -> Vec<f64> { vec![-y[0]] };
        let config = RkSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 1.0,
            rtol: 1e-6,
            atol: ToleranceValue::Scalar(1e-8),
            max_step: f64::INFINITY,
            first_step: None,
            mode: RuntimeMode::Strict,
            tableau: &RK23_TABLEAU,
        };
        let mut solver = RkSolver::new(&mut fun, config).expect("solver creation should succeed");

        let mut steps = 0;
        while solver.state() == OdeSolverState::Running {
            solver
                .step_with(&mut |_t, y| vec![-y[0]])
                .expect("step should succeed");
            steps += 1;
            assert!(steps <= 5000, "too many steps");
        }

        let expected = (-1.0_f64).exp();
        assert!(
            (solver.y()[0] - expected).abs() < 1e-4,
            "RK23: y(1) should be close to e^-1 ≈ {expected}, got {}",
            solver.y()[0]
        );
    }

    #[test]
    fn next_after_basic() {
        let x = next_after(1.0, 2.0);
        assert!(x > 1.0);
        assert!(x < 1.0 + 1e-15);
        let y = next_after(1.0, 0.0);
        assert!(y < 1.0);
    }

    #[test]
    fn solver_rk45_fsal_efficiency() {
        let mut fun = |_t: f64, y: &[f64]| -> Vec<f64> { vec![y[0]] };
        let config = RkSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 1.0,
            rtol: 1e-3,
            atol: ToleranceValue::Scalar(1e-6),
            max_step: f64::INFINITY,
            first_step: Some(0.1), // Fix step size to make nfev predictable
            mode: RuntimeMode::Strict,
            tableau: &RK45_TABLEAU,
        };
        let mut solver = RkSolver::new(&mut fun, config).expect("solver creation");

        let mut steps = 0;
        while solver.state() == OdeSolverState::Running {
            solver.step_with(&mut |_t, y| vec![y[0]]).unwrap();
            steps += 1;
        }

        // RK45 has 6 stages. Without FSAL, each step would take 7 evaluations (6 stages + 1 final).
        // With FSAL, each step takes 6 evaluations (the 7th is reused as the 1st of the next).
        // Total nfev should be: 1 (initial) + 6 * steps.
        let expected_nfev = 1 + 6 * steps;
        assert_eq!(
            solver.nfev(),
            expected_nfev,
            "FSAL: expected {expected_nfev} evaluations for {steps} steps, got {}",
            solver.nfev()
        );
    }

    #[test]
    fn odesolver_step_trait_method_works() {
        let config = RkSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 1.0,
            rtol: 1e-8,
            atol: ToleranceValue::Scalar(1e-10),
            max_step: f64::INFINITY,
            first_step: None,
            mode: RuntimeMode::Strict,
            tableau: &RK45_TABLEAU,
        };
        let mut solver: Box<dyn OdeSolver> =
            Box::new(RkSolver::new_owned(|_t, y| vec![-y[0]], config).expect("solver creation"));

        assert_eq!(solver.state(), OdeSolverState::Running);

        let mut steps = 0;
        while solver.state() == OdeSolverState::Running {
            solver.step().expect("trait step should succeed");
            steps += 1;
            assert!(steps <= 1000, "too many steps");
        }

        assert_eq!(solver.state(), OdeSolverState::Finished);
    }

    #[test]
    fn odesolver_step_exponential_decay_accuracy() {
        let config = RkSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 1.0,
            rtol: 1e-8,
            atol: ToleranceValue::Scalar(1e-10),
            max_step: f64::INFINITY,
            first_step: None,
            mode: RuntimeMode::Strict,
            tableau: &RK45_TABLEAU,
        };
        let mut solver = RkSolver::new_owned(|_t, y| vec![-y[0]], config).expect("solver creation");

        while solver.state() == OdeSolverState::Running {
            solver.step().expect("step should succeed");
        }

        let expected = (-1.0_f64).exp();
        assert!(
            (solver.y()[0] - expected).abs() < 1e-6,
            "OdeSolver::step: y(1) should be close to e^-1 ≈ {expected}, got {}",
            solver.y()[0]
        );
    }

    #[test]
    fn ref_created_solver_requires_step_with() {
        // Solvers created with new() must use step_with() instead of step()
        let mut fun = |_t: f64, y: &[f64]| -> Vec<f64> { vec![-y[0]] };
        let config = RkSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 1.0,
            rtol: 1e-8,
            atol: ToleranceValue::Scalar(1e-10),
            max_step: f64::INFINITY,
            first_step: None,
            mode: RuntimeMode::Strict,
            tableau: &RK45_TABLEAU,
        };
        let mut solver = RkSolver::new(&mut fun, config).expect("solver creation");

        // can_step() should return false for ref-created solvers
        assert!(!solver.can_step());

        // step() returns NotYetImplemented - use step_with() instead
        let result = solver.step();
        assert!(matches!(result, Err(StepFailure::NotYetImplemented(_))));

        // step_with() works correctly
        let step_result = solver.step_with(&mut fun);
        assert!(step_result.is_ok());
    }

    #[test]
    fn owned_solver_supports_step() {
        let config = RkSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 1.0,
            rtol: 1e-8,
            atol: ToleranceValue::Scalar(1e-10),
            max_step: f64::INFINITY,
            first_step: None,
            mode: RuntimeMode::Strict,
            tableau: &RK45_TABLEAU,
        };
        let solver = RkSolver::new_owned(|_t, y| vec![-y[0]], config).expect("solver creation");

        // can_step() should return true for owned solvers
        assert!(solver.can_step());
    }
}
