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
use crate::validation::{ToleranceValue, validate_rhs_shape, validate_tol};
use crate::{
    IntegrateValidationError, select_initial_step, validate_first_step, validate_max_step,
};

// Step-size control constants (from SciPy's rk.py).
const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 10.0;
const RHS_WRONG_SHAPE_RUNTIME_ERROR: &str =
    "right-hand side returned wrong number of derivative values";

type OdeFn = Box<dyn FnMut(f64, &[f64]) -> Vec<f64>>;

fn validate_hardened_initial_rhs(
    mode: RuntimeMode,
    f0: &[f64],
) -> Result<(), IntegrateValidationError> {
    if mode == RuntimeMode::Hardened && !f0.iter().all(|value| value.is_finite()) {
        return Err(IntegrateValidationError::NonFiniteF0);
    }
    Ok(())
}

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

/// Dense-output interpolation matrix for RK45, as exact rationals.
///
/// This is SciPy's `RK45.P` (`scipy/integrate/_ivp/rk.py`), a 7x4 matrix
/// mapping the seven stage derivatives onto the quartic interpolating
/// polynomial. SciPy builds `Q = K.T @ P` and evaluates
/// `y(t) = y_old + h * Q @ [x, x², x³, x⁴]`, which is 4th-order accurate in
/// `h` and is the *only* interpolant that reproduces SciPy's `t_eval` samples.
/// Cubic Hermite through the endpoints is one order lower and is what caused
/// frankenscipy-3m5ip.
///
/// Written as ratios rather than pasted decimals so the constants are auditable
/// against the reference; a unit test checks each entry against the decimal
/// SciPy actually reports.
static RK45_P: [[f64; 4]; 7] = [
    [
        1.0,
        -8_048_581_381.0 / 2_820_520_608.0,
        8_663_915_743.0 / 2_820_520_608.0,
        -12_715_105_075.0 / 11_282_082_432.0,
    ],
    [0.0, 0.0, 0.0, 0.0],
    [
        0.0,
        131_558_114_200.0 / 32_700_410_799.0,
        -68_118_460_800.0 / 10_900_136_933.0,
        87_487_479_700.0 / 32_700_410_799.0,
    ],
    [
        0.0,
        -1_754_552_775.0 / 470_086_768.0,
        14_199_869_525.0 / 1_410_260_304.0,
        -10_690_763_975.0 / 1_880_347_072.0,
    ],
    [
        0.0,
        127_303_824_393.0 / 49_829_197_408.0,
        -318_862_633_887.0 / 49_829_197_408.0,
        701_980_252_875.0 / 199_316_789_632.0,
    ],
    [
        0.0,
        -282_668_133.0 / 205_662_961.0,
        2_019_193_451.0 / 616_988_883.0,
        -1_453_857_185.0 / 822_651_844.0,
    ],
    [
        0.0,
        40_617_522.0 / 29_380_423.0,
        -110_615_467.0 / 29_380_423.0,
        69_997_945.0 / 29_380_423.0,
    ],
];

/// Number of stage derivatives RK45's dense output consumes (SciPy's `K` has
/// `n_stages + 1` rows; our tableau counts the FSAL stage inside `n_stages`).
const RK45_DENSE_STAGES: usize = 7;

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
/// Writes `y_new` and `f_new` where `f_new = fun(t + h, y_new)`.
/// `k` is filled with the stage derivatives (k[0] = f, k[n_stages] = f_new).
fn rk_step<F>(
    fun: &mut F,
    t: f64,
    y: &[f64],
    f: &[f64],
    h: f64,
    tableau: &ButcherTableau,
    k: &mut [Vec<f64>],
    y_new: &mut [f64],
    f_new: &mut [f64],
    dy: &mut [f64],
    y_stage: &mut [f64],
) -> Result<(), StepFailure>
where
    F: FnMut(f64, &[f64]) -> Vec<f64> + ?Sized,
{
    let n = y.len();
    validate_stage_rhs_shape(f.len(), n)?;
    debug_assert_eq!(k.len(), tableau.n_stages + 1);
    debug_assert_eq!(y_new.len(), n);
    debug_assert_eq!(f_new.len(), n);
    debug_assert_eq!(dy.len(), n);
    debug_assert_eq!(y_stage.len(), n);
    k[0].copy_from_slice(f);

    for s in 1..tableau.n_stages {
        let a_row = tableau.a[s];
        let c_s = tableau.c[s];
        dy.fill(0.0);
        for (j, &a_sj) in a_row.iter().enumerate() {
            if a_sj != 0.0 {
                for i in 0..n {
                    dy[i] += a_sj * k[j][i];
                }
            }
        }
        for i in 0..n {
            y_stage[i] = y[i] + h * dy[i];
        }
        let stage = fun(t + c_s * h, y_stage);
        validate_stage_rhs_shape(stage.len(), n)?;
        k[s].copy_from_slice(&stage);
    }

    // y_new = y + h * sum(B[s] * K[s])
    y_new.copy_from_slice(y);
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
    f_new.copy_from_slice(&k[tableau.n_stages - 1]);
    k[tableau.n_stages].copy_from_slice(f_new); // store for next step if needed

    Ok(())
}

fn validate_stage_rhs_shape(actual: usize, expected: usize) -> Result<(), StepFailure> {
    validate_rhs_shape(actual, expected)
        .map_err(|_| StepFailure::RuntimeError(RHS_WRONG_SHAPE_RUNTIME_ERROR))
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
    y_new: Vec<f64>,
    f_new: Vec<f64>,
    dy: Vec<f64>,
    y_stage: Vec<f64>,
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
        validate_rhs_shape(f0.len(), n)?;
        validate_hardened_initial_rhs(config.mode, &f0)?;
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
        let y_new = vec![0.0; n];
        let f_new = vec![0.0; n];
        let dy = vec![0.0; n];
        let y_stage = vec![0.0; n];

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
            y_new,
            f_new,
            dy,
            y_stage,
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
        validate_rhs_shape(f0.len(), n)?;
        validate_hardened_initial_rhs(config.mode, &f0)?;
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
        let y_new = vec![0.0; n];
        let f_new = vec![0.0; n];
        let dy = vec![0.0; n];
        let y_stage = vec![0.0; n];

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
            y_new,
            f_new,
            dy,
            y_stage,
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

    /// Solver-specific dense output at `t`, matching SciPy's `RkDenseOutput`.
    ///
    /// Returns `None` when this solver has no solver-specific interpolant
    /// implemented (RK23 and DOP853 today) or when no step has been taken yet,
    /// so callers fall back to cubic Hermite exactly as before.
    ///
    /// For RK45 this reproduces SciPy's construction: with `Q = Kᵀ P` and
    /// `x = (t - t_old) / h`,
    ///
    /// ```text
    /// y(t) = y_old + h · Q · [x, x², x³, x⁴]ᵀ
    /// ```
    ///
    /// evaluated as `y_old + h · Σⱼ (Σₛ K[s]·P[s][j]) · x^{j+1}`. Cubic Hermite
    /// through the endpoints is one order lower and drifts from SciPy's samples
    /// mid-step even when the step endpoints agree — frankenscipy-3m5ip.
    #[must_use]
    pub fn dense_output_at(&self, t: f64) -> Option<Vec<f64>> {
        if !std::ptr::eq(self.tableau, &RK45_TABLEAU) {
            return None;
        }
        let (t_old, y_old) = (self.t_old?, self.y_old.as_deref()?);
        let h = self.t - t_old;
        if h == 0.0 {
            return Some(y_old.to_vec());
        }
        let x = (t - t_old) / h;

        let mut out = y_old.to_vec();
        for (i, out_i) in out.iter_mut().enumerate() {
            // Horner in x over the quartic, accumulating Q[i][j] on the fly so
            // no n×4 scratch matrix is allocated per sample.
            let mut acc = 0.0;
            for j in (0..4).rev() {
                let q_ij: f64 = (0..RK45_DENSE_STAGES)
                    .map(|s| self.k[s][i] * RK45_P[s][j])
                    .sum();
                acc = acc * x + q_ij;
            }
            *out_i += h * acc * x;
        }
        Some(out)
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

            // Perform the RK step into reusable buffers owned by the solver.
            rk_step(
                fun,
                t,
                &self.y,
                &self.f,
                h,
                self.tableau,
                &mut self.k,
                &mut self.y_new,
                &mut self.f_new,
                &mut self.dy,
                &mut self.y_stage,
            )?;
            self.nfev += self.tableau.n_stages - 1; // leverage FSAL: exactly n_stages - 1 new evals per step

            let err_norm = self.estimate_error_norm(h, &self.y_new);

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
                std::mem::swap(&mut self.y, &mut self.y_new);
                self.h_abs = h_abs;
                std::mem::swap(&mut self.f, &mut self.f_new);
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

            // Perform the RK step using stored function (borrow only for this call).
            {
                let Some(fun) = self.fun.as_mut() else {
                    return Err(StepFailure::NotYetImplemented(
                        "OdeSolver::step requires solver created with new_owned; use step_with instead",
                    ));
                };
                rk_step(
                    &mut **fun,
                    t,
                    &self.y,
                    &self.f,
                    h,
                    self.tableau,
                    &mut self.k,
                    &mut self.y_new,
                    &mut self.f_new,
                    &mut self.dy,
                    &mut self.y_stage,
                )?;
            };
            self.nfev += self.tableau.n_stages - 1;

            let err_norm = self.estimate_error_norm(h, &self.y_new);

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
                std::mem::swap(&mut self.y, &mut self.y_new);
                self.h_abs = h_abs;
                std::mem::swap(&mut self.f, &mut self.f_new);
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

    fn rk_scratch(
        n: usize,
        tableau: &ButcherTableau,
    ) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        (
            vec![vec![0.0; n]; tableau.n_stages + 1],
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
            vec![0.0; n],
        )
    }

    /// `RK45_P` is written as exact rationals for auditability; this pins every
    /// entry to the decimal `scipy.integrate._ivp.rk.RK45.P` actually reports,
    /// so a typo in a numerator cannot slip through. Values captured from
    /// SciPy 1.17.1. frankenscipy-3m5ip.
    #[test]
    fn rk45_dense_output_matrix_matches_scipy() {
        let scipy: [[f64; 4]; 7] = [
            [
                1.0,
                -2.853_580_065_386_283_5,
                3.071_743_464_105_900_5,
                -1.127_017_565_386_283_5,
            ],
            [0.0, 0.0, 0.0, 0.0],
            [
                0.0,
                4.023_133_379_230_305,
                -6.249_321_565_289,
                2.675_424_484_351_598,
            ],
            [
                0.0,
                -3.732_401_961_588_504_2,
                10.068_970_589_843_675,
                -5.685_526_961_588_504,
            ],
            [
                0.0,
                2.554_803_830_184_942_3,
                -6.399_112_377_351_017,
                3.521_932_367_920_791_2,
            ],
            [
                0.0,
                -1.374_424_114_218_602_4,
                3.272_657_752_246_729,
                -1.767_281_257_075_745_5,
            ],
            [
                0.0,
                1.382_468_931_778_143_6,
                -3.764_937_863_556_287,
                2.382_468_931_778_144,
            ],
        ];
        for (s, (mine, theirs)) in RK45_P.iter().zip(scipy.iter()).enumerate() {
            for (j, (a, b)) in mine.iter().zip(theirs.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= f64::EPSILON * b.abs().max(1.0),
                    "RK45_P[{s}][{j}] = {a:?} but SciPy reports {b:?}"
                );
            }
        }
    }

    /// The quartic dense output must reproduce the step endpoints exactly:
    /// `x = 0` gives `y_old` and `x = 1` gives `y_new`. This is a property the
    /// interpolant satisfies by construction, so it catches an indexing or
    /// Horner-order slip without needing a SciPy oracle in-process.
    #[test]
    fn rk45_dense_output_reproduces_step_endpoints() {
        // y' = y with y(0) = 1, so the stage derivatives are non-trivial.
        let mut fun = |_t: f64, y: &[f64]| vec![y[0], -0.5 * y[1]];
        let y0 = [1.0, 2.0];
        let mut solver = RkSolver::new(
            &mut fun,
            RkSolverConfig {
                mode: RuntimeMode::Strict,
                t0: 0.0,
                y0: &y0,
                t_bound: 1.0,
                rtol: 1e-8,
                atol: ToleranceValue::Scalar(1e-10),
                max_step: f64::INFINITY,
                first_step: None,
                tableau: &RK45_TABLEAU,
            },
        )
        .expect("solver init");
        solver.step_with(&mut fun).expect("first step");

        let t_old = solver.t_old().expect("t_old after a step");
        let t_new = solver.t();
        let at_start = solver
            .dense_output_at(t_old)
            .expect("RK45 has dense output");
        let at_end = solver
            .dense_output_at(t_new)
            .expect("RK45 has dense output");

        for (got, want) in at_start.iter().zip(solver.y_old().expect("y_old")) {
            assert!(
                (got - want).abs() < 1e-13,
                "x=0 must give y_old: {got} vs {want}"
            );
        }
        for (got, want) in at_end.iter().zip(solver.y()) {
            assert!(
                (got - want).abs() < 1e-12,
                "x=1 must give y_new: {got} vs {want}"
            );
        }
    }

    /// RK23 and DOP853 have their own `P` matrices that are not implemented
    /// yet, so they must return `None` and keep the cubic-Hermite fallback
    /// rather than silently borrowing RK45's interpolant.
    #[test]
    fn dense_output_is_none_for_solvers_without_one() {
        let mut fun = |_t: f64, y: &[f64]| vec![-y[0]];
        let y0 = [1.0];
        for tableau in [&RK23_TABLEAU, &DOP853_TABLEAU] {
            let mut solver = RkSolver::new(
                &mut fun,
                RkSolverConfig {
                    mode: RuntimeMode::Strict,
                    t0: 0.0,
                    y0: &y0,
                    t_bound: 1.0,
                    rtol: 1e-6,
                    atol: ToleranceValue::Scalar(1e-9),
                    max_step: f64::INFINITY,
                    first_step: None,
                    tableau,
                },
            )
            .expect("solver init");
            solver.step_with(&mut fun).expect("step");
            assert!(
                solver.dense_output_at(solver.t()).is_none(),
                "only RK45 provides a solver-specific dense output today"
            );
        }
    }

    #[test]
    fn rk_step_single_euler_like() {
        // Trivial test: single stage with A=[], B=[1], C=[0] would be Euler
        // But we test with the actual RK45 tableau on a simple system
        let n = 1;
        let (mut k, mut y_new, mut f_new, mut dy, mut y_stage) = rk_scratch(n, &RK45_TABLEAU);
        rk_step(
            &mut |_t, y| vec![-0.5 * y[0]],
            0.0,
            &[2.0],
            &[-1.0], // f(0, 2) = -1
            0.1,
            &RK45_TABLEAU,
            &mut k,
            &mut y_new,
            &mut f_new,
            &mut dy,
            &mut y_stage,
        )
        .expect("rk step");
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
        let (mut k, mut y_new, mut f_new, mut dy, mut y_stage) = rk_scratch(n, &RK23_TABLEAU);
        rk_step(
            &mut |_t, y| vec![-0.5 * y[0]],
            0.0,
            &[2.0],
            &[-1.0],
            0.1,
            &RK23_TABLEAU,
            &mut k,
            &mut y_new,
            &mut f_new,
            &mut dy,
            &mut y_stage,
        )
        .expect("rk step");
        // RK23 is order 3, so the error should be O(h^4) ≈ 1e-4
        assert!(
            (y_new[0] - 2.0 * (-0.05_f64).exp()).abs() < 1e-4,
            "RK23 step should be reasonably accurate, got {}",
            y_new[0]
        );
    }

    #[test]
    fn rk_step_rejects_wrong_size_stage_rhs() {
        let n = 1;
        let (mut k, mut y_new, mut f_new, mut dy, mut y_stage) = rk_scratch(n, &RK45_TABLEAU);
        let err = rk_step(
            &mut |_t, _y| Vec::new(),
            0.0,
            &[2.0],
            &[-1.0],
            0.1,
            &RK45_TABLEAU,
            &mut k,
            &mut y_new,
            &mut f_new,
            &mut dy,
            &mut y_stage,
        )
        .expect_err("wrong-size stage derivative");
        assert_eq!(
            err,
            StepFailure::RuntimeError(RHS_WRONG_SHAPE_RUNTIME_ERROR)
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
    fn solver_with_first_step_hardened_rejects_non_finite_f0() {
        let mut fun = |_t: f64, _y: &[f64]| -> Vec<f64> { vec![f64::NAN] };
        let config = RkSolverConfig {
            t0: 0.0,
            y0: &[1.0],
            t_bound: 1.0,
            rtol: 1e-6,
            atol: ToleranceValue::Scalar(1e-8),
            max_step: f64::INFINITY,
            first_step: Some(0.01),
            mode: RuntimeMode::Hardened,
            tableau: &RK45_TABLEAU,
        };
        let err = match RkSolver::new(&mut fun, config) {
            Ok(_) => panic!("non-finite f0 should fail"),
            Err(err) => err,
        };
        assert_eq!(err, IntegrateValidationError::NonFiniteF0);
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
