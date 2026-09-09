#![forbid(unsafe_code)]

use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use fsci_runtime::RuntimeMode;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    #[must_use]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[must_use]
    pub const fn from_real(value: f64) -> Self {
        Self { re: value, im: 0.0 }
    }

    #[must_use]
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }

    #[must_use]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    #[must_use]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    #[must_use]
    pub const fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[must_use]
    pub fn exp(self) -> Self {
        let scale = self.re.exp();
        if scale == 0.0 {
            return Self::new(0.0, 0.0);
        }
        Self {
            re: scale * self.im.cos(),
            im: scale * self.im.sin(),
        }
    }

    #[must_use]
    pub fn recip(self) -> Self {
        if self.re.is_infinite() || self.im.is_infinite() {
            return Self::new(0.0, 0.0);
        }
        let norm = self.norm_sqr();
        Self {
            re: self.re / norm,
            im: -self.im / norm,
        }
    }

    #[must_use]
    pub fn ln(self) -> Self {
        Self {
            re: self.abs().ln(),
            im: self.im.atan2(self.re),
        }
    }

    #[must_use]
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    #[must_use]
    pub fn powf(self, n: f64) -> Self {
        let r = self.abs();
        let theta = self.arg();
        let r_n = r.powf(n);
        Self {
            re: r_n * (n * theta).cos(),
            im: r_n * (n * theta).sin(),
        }
    }

    #[must_use]
    pub fn powc(self, n: Self) -> Self {
        if self.re == 0.0 && self.im == 0.0 {
            if n.re > 0.0 {
                return Self::new(0.0, 0.0);
            } else {
                return Self::new(f64::INFINITY, 0.0);
            }
        }
        (n * self.ln()).exp()
    }

    #[must_use]
    pub fn sin(self) -> Self {
        // sin(a + bi) = sin(a)cosh(b) + i cos(a)sinh(b)
        Self::new(
            self.re.sin() * self.im.cosh(),
            self.re.cos() * self.im.sinh(),
        )
    }

    #[must_use]
    pub fn cos(self) -> Self {
        // cos(a + bi) = cos(a)cosh(b) - i sin(a)sinh(b)
        Self::new(
            self.re.cos() * self.im.cosh(),
            -self.re.sin() * self.im.sinh(),
        )
    }

    #[must_use]
    pub fn sinh(self) -> Self {
        // sinh(a + bi) = sinh(a)cos(b) + i cosh(a)sin(b)
        Self::new(
            self.re.sinh() * self.im.cos(),
            self.re.cosh() * self.im.sin(),
        )
    }

    #[must_use]
    pub fn cosh(self) -> Self {
        // cosh(a + bi) = cosh(a)cos(b) + i sinh(a)sin(b)
        Self::new(
            self.re.cosh() * self.im.cos(),
            self.re.sinh() * self.im.sin(),
        )
    }
}

impl From<f64> for Complex64 {
    fn from(value: f64) -> Self {
        Self::from_real(value)
    }
}

impl Add for Complex64 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl Sub for Complex64 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl Mul for Complex64 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl Div for Complex64 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let norm = rhs.norm_sqr();
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / norm,
            im: (self.im * rhs.re - self.re * rhs.im) / norm,
        }
    }
}

impl Mul<f64> for Complex64 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

impl Div<f64> for Complex64 {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self {
            re: self.re / rhs,
            im: self.im / rhs,
        }
    }
}

impl Neg for Complex64 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum SpecialTensor {
    #[default]
    Empty,
    RealScalar(f64),
    ComplexScalar(Complex64),
    RealVec(Vec<f64>),
    ComplexVec(Vec<Complex64>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelRegime {
    Series,
    Asymptotic,
    Recurrence,
    ContinuedFraction,
    Reflection,
    BackendDelegate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchStep {
    pub regime: KernelRegime,
    pub when: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchPlan {
    pub function: &'static str,
    pub steps: &'static [DispatchStep],
    pub notes: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecialErrorKind {
    DomainError,
    FixtureSchemaError,
    PoleInput,
    NonFiniteInput,
    CancellationRisk,
    OverflowRisk,
    SingularityRisk,
    NotYetImplemented,
    ShapeMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecialError {
    pub function: &'static str,
    pub kind: SpecialErrorKind,
    pub mode: RuntimeMode,
    pub detail: &'static str,
}

impl fmt::Display for SpecialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "special function `{}` failed in {:?} mode: {} ({:?})",
            self.function, self.mode, self.detail, self.kind
        )
    }
}

impl std::error::Error for SpecialError {}

pub type SpecialResult = Result<SpecialTensor, SpecialError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecialTraceEntry {
    pub timestamp_ms: u128,
    pub function: &'static str,
    pub mode: RuntimeMode,
    pub category: &'static str,
    pub input_summary: String,
    pub action_taken: &'static str,
    pub result_summary: String,
    pub clamped: bool,
}

impl SpecialTraceEntry {
    #[must_use]
    pub fn to_json_line(&self) -> String {
        format!(
            "{{\"timestamp_ms\":{},\"function\":\"{}\",\"mode\":\"{:?}\",\"category\":\"{}\",\"input_summary\":\"{}\",\"action_taken\":\"{}\",\"result_summary\":\"{}\",\"clamped\":{}}}",
            self.timestamp_ms,
            self.function,
            self.mode,
            self.category,
            escape_json(&self.input_summary),
            self.action_taken,
            escape_json(&self.result_summary),
            self.clamped
        )
    }
}

static TRACE_LOG: OnceLock<Mutex<Vec<SpecialTraceEntry>>> = OnceLock::new();

fn trace_log() -> &'static Mutex<Vec<SpecialTraceEntry>> {
    TRACE_LOG.get_or_init(|| Mutex::new(Vec::new()))
}

/// Acquire the trace-log guard, recovering from a poisoned mutex so
/// trace entries are preserved across panics.
/// Resolves [frankenscipy-wtxpg].
fn lock_trace_log_or_recover() -> std::sync::MutexGuard<'static, Vec<SpecialTraceEntry>> {
    let log = trace_log();
    match log.lock() {
        Ok(g) => g,
        Err(poisoned) => {
            log.clear_poison();
            poisoned.into_inner()
        }
    }
}

#[must_use]
pub fn take_special_traces() -> Vec<SpecialTraceEntry> {
    let mut log = lock_trace_log_or_recover();
    let mut out = Vec::with_capacity(log.len());
    std::mem::swap(&mut *log, &mut out);
    out
}

const MAX_SPECIAL_TRACES: usize = 10_000;

pub fn record_special_trace(
    function: &'static str,
    mode: RuntimeMode,
    category: &'static str,
    input_summary: impl Into<String>,
    action_taken: &'static str,
    result_summary: impl Into<String>,
    clamped: bool,
) {
    let mut log = lock_trace_log_or_recover();
    if log.len() >= MAX_SPECIAL_TRACES {
        log.drain(0..MAX_SPECIAL_TRACES / 2);
    }
    log.push(SpecialTraceEntry {
        timestamp_ms: now_unix_ms(),
        function,
        mode,
        category,
        input_summary: input_summary.into(),
        action_taken,
        result_summary: result_summary.into(),
        clamped,
    });
}

pub fn not_yet_implemented(
    function: &'static str,
    mode: RuntimeMode,
    detail: &'static str,
) -> SpecialResult {
    Err(SpecialError {
        function,
        kind: SpecialErrorKind::NotYetImplemented,
        mode,
        detail,
    })
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn escape_json(input: &str) -> String {
    input
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}
