#![forbid(unsafe_code)]

//! Orthogonal distance regression support for FrankenSciPy.
//!
//! The public surface mirrors the durable pieces of `scipy.odr`: data
//! containers, model containers, an `ODR` runner, an `Output` result, a
//! low-level `odr` helper, and the standard model factories. The solver is a
//! conservative explicit-model implementation with a local damped
//! Gauss-Newton/Levenberg-Marquardt loop: it estimates both fit parameters and
//! input corrections, so weighted errors in `x` and `y` participate in the same
//! objective.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

/// Callable shape used by `Model`: `f(beta, x) -> y`.
pub type ModelFn = Arc<dyn Fn(&[f64], &[f64]) -> Vec<f64> + Send + Sync + 'static>;

/// Analytic Jacobian callback, called as `jac(beta, x + delta)` with one ROW PER OBSERVATION.
///
/// * `fjacb` returns `∂f_i/∂β_j`: `n_obs × len(beta)`.
/// * `fjacd` returns `∂f_i/∂x` for observation `i`'s OWN inputs: `n_obs × m`, where the `m =
///   len(x)/n_obs` inputs of observation `i` are `x[i·m .. (i+1)·m]`. That is ODRPACK's model:
///   `f_i` depends on no other observation's input.
///
/// SciPy's `fjacb`/`fjacd` return the transposes, shapes `(p, n)` and `(m, n)`. When a model
/// supplies one, the fit uses it in place of finite differences.
pub type JacobianFn = Arc<dyn Fn(&[f64], &[f64]) -> Vec<Vec<f64>> + Send + Sync + 'static>;

/// Parameter-estimate callback: `estimate(data) -> beta0`, used by
/// [`ODR::from_model_estimate`] when no `beta0` is given (SciPy's `ODR(data, model)`).
pub type EstimateFn = Arc<dyn Fn(&Data) -> Vec<f64> + Send + Sync + 'static>;

/// Warning marker matching SciPy's `OdrWarning` symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OdrWarning {
    pub detail: String,
}

/// Stop marker matching SciPy's `OdrStop` symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OdrStop {
    pub detail: String,
}

/// Error type matching SciPy's `OdrError` role.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OdrError {
    InvalidArgument { detail: String },
    NonFiniteInput { detail: String },
    SolverFailure { detail: String },
    Unsupported { detail: String },
}

impl fmt::Display for OdrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArgument { detail }
            | Self::NonFiniteInput { detail }
            | Self::SolverFailure { detail }
            | Self::Unsupported { detail } => f.write_str(detail),
        }
    }
}

impl std::error::Error for OdrError {}

/// Data to fit, equivalent to `scipy.odr.Data` for explicit one-response data.
#[derive(Debug, Clone, PartialEq)]
pub struct Data {
    pub x: Vec<f64>,
    pub y: Option<Vec<f64>>,
    pub we: Vec<f64>,
    pub wd: Vec<f64>,
    pub fix: Option<Vec<bool>>,
    pub meta: BTreeMap<String, String>,
}

impl Data {
    /// Construct explicit response data with unit weights.
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Result<Self, OdrError> {
        validate_finite_slice("x", &x)?;
        validate_finite_slice("y", &y)?;
        if x.is_empty() {
            return Err(OdrError::InvalidArgument {
                detail: String::from("x must contain at least one observation"),
            });
        }
        if y.is_empty() {
            return Err(OdrError::InvalidArgument {
                detail: String::from("y must contain at least one observation"),
            });
        }
        Ok(Self {
            we: vec![1.0; y.len()],
            wd: vec![1.0; x.len()],
            x,
            y: Some(y),
            fix: None,
            meta: BTreeMap::new(),
        })
    }

    /// Construct implicit data. The current runner rejects implicit models, but
    /// the container is exposed so callers can represent the SciPy shape.
    pub fn implicit(x: Vec<f64>) -> Result<Self, OdrError> {
        validate_finite_slice("x", &x)?;
        if x.is_empty() {
            return Err(OdrError::InvalidArgument {
                detail: String::from("x must contain at least one observation"),
            });
        }
        Ok(Self {
            wd: vec![1.0; x.len()],
            x,
            y: None,
            we: Vec::new(),
            fix: None,
            meta: BTreeMap::new(),
        })
    }

    /// Set response weights `we`.
    pub fn with_response_weights(mut self, we: Vec<f64>) -> Result<Self, OdrError> {
        let y_len = self.response()?.len();
        validate_weight_vec("we", &we, y_len)?;
        self.we = we;
        Ok(self)
    }

    /// Set input weights `wd`.
    pub fn with_input_weights(mut self, wd: Vec<f64>) -> Result<Self, OdrError> {
        validate_weight_vec("wd", &wd, self.x.len())?;
        self.wd = wd;
        Ok(self)
    }

    /// Set input correction freedom flags. `true` means free, `false` means
    /// fixed; this follows the semantics of SciPy's positive/free `ifixx`.
    pub fn with_input_free(mut self, free: Vec<bool>) -> Result<Self, OdrError> {
        if free.len() != self.x.len() {
            return Err(OdrError::InvalidArgument {
                detail: format!(
                    "fix/free mask length must match x length (got {} and {})",
                    free.len(),
                    self.x.len()
                ),
            });
        }
        self.fix = Some(free);
        Ok(self)
    }

    pub fn response(&self) -> Result<&[f64], OdrError> {
        self.y.as_deref().ok_or_else(|| OdrError::Unsupported {
            detail: String::from("implicit ODR data has no explicit response vector"),
        })
    }

    pub fn set_meta(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.meta.insert(key.into(), value.into());
    }
}

/// Data with actual standard deviations, equivalent to `scipy.odr.RealData`.
#[derive(Debug, Clone, PartialEq)]
pub struct RealData {
    pub data: Data,
    pub sx: Option<Vec<f64>>,
    pub sy: Option<Vec<f64>>,
}

impl RealData {
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Result<Self, OdrError> {
        Ok(Self {
            data: Data::new(x, y)?,
            sx: None,
            sy: None,
        })
    }

    /// Construct `RealData` from standard deviations. As in SciPy, standard
    /// deviations are converted to weights with `1 / sigma^2`.
    pub fn from_stddev(
        x: Vec<f64>,
        y: Vec<f64>,
        sx: Option<Vec<f64>>,
        sy: Option<Vec<f64>>,
    ) -> Result<Self, OdrError> {
        let mut data = Data::new(x, y)?;
        if let Some(values) = sx.as_ref() {
            let x_len = data.x.len();
            let wd = stddev_to_weights("sx", values, x_len)?;
            data = data.with_input_weights(wd)?;
        }
        if let Some(values) = sy.as_ref() {
            let y_len = data.response()?.len();
            let we = stddev_to_weights("sy", values, y_len)?;
            data = data.with_response_weights(we)?;
        }
        Ok(Self { data, sx, sy })
    }
}

impl From<RealData> for Data {
    fn from(value: RealData) -> Self {
        value.data
    }
}

/// Model container matching `scipy.odr.Model`.
#[derive(Clone)]
pub struct Model {
    pub name: String,
    pub fcn: ModelFn,
    pub fjacb: Option<JacobianFn>,
    pub fjacd: Option<JacobianFn>,
    pub estimate: Option<EstimateFn>,
    pub implicit: bool,
    pub scalar_separable: bool,
    pub parameter_count: Option<usize>,
    pub meta: BTreeMap<String, String>,
}

impl fmt::Debug for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Model")
            .field("name", &self.name)
            .field("has_fjacb", &self.fjacb.is_some())
            .field("has_fjacd", &self.fjacd.is_some())
            .field("has_estimate", &self.estimate.is_some())
            .field("implicit", &self.implicit)
            .field("scalar_separable", &self.scalar_separable)
            .field("parameter_count", &self.parameter_count)
            .field("meta", &self.meta)
            .finish_non_exhaustive()
    }
}

impl Model {
    pub fn new<F>(fcn: F) -> Self
    where
        F: Fn(&[f64], &[f64]) -> Vec<f64> + Send + Sync + 'static,
    {
        Self {
            name: String::from("custom"),
            fcn: Arc::new(fcn),
            fjacb: None,
            fjacd: None,
            estimate: None,
            implicit: false,
            scalar_separable: false,
            parameter_count: None,
            meta: BTreeMap::new(),
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn with_parameter_count(mut self, parameter_count: usize) -> Self {
        self.parameter_count = Some(parameter_count);
        self
    }

    /// Declare that the model evaluates each observation independently: output
    /// point `i` depends only on input point `i` and `beta`.
    pub fn with_scalar_separable(mut self, scalar_separable: bool) -> Self {
        self.scalar_separable = scalar_separable;
        self
    }

    /// Whether the model was declared [`Self::with_scalar_separable`].
    pub fn is_scalar_separable(&self) -> bool {
        self.scalar_separable
    }

    pub fn with_estimate<F>(mut self, estimate: F) -> Self
    where
        F: Fn(&Data) -> Vec<f64> + Send + Sync + 'static,
    {
        self.estimate = Some(Arc::new(estimate));
        self
    }

    pub fn with_fjacb<F>(mut self, fjacb: F) -> Self
    where
        F: Fn(&[f64], &[f64]) -> Vec<Vec<f64>> + Send + Sync + 'static,
    {
        self.fjacb = Some(Arc::new(fjacb));
        self
    }

    pub fn with_fjacd<F>(mut self, fjacd: F) -> Self
    where
        F: Fn(&[f64], &[f64]) -> Vec<Vec<f64>> + Send + Sync + 'static,
    {
        self.fjacd = Some(Arc::new(fjacd));
        self
    }

    pub fn implicit(mut self, implicit: bool) -> Self {
        self.implicit = implicit;
        self
    }

    pub fn set_meta(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.meta.insert(key.into(), value.into());
    }

    pub fn evaluate(&self, beta: &[f64], x: &[f64]) -> Vec<f64> {
        (self.fcn)(beta, x)
    }
}

/// Solver options for `ODR::run`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OdrOptions {
    pub maxit: usize,
    pub sstol: f64,
    pub partol: f64,
    pub diff_step: f64,
    pub fit_type: FitType,
}

impl Default for OdrOptions {
    fn default() -> Self {
        Self {
            maxit: 50,
            sstol: f64::EPSILON.sqrt(),
            partol: f64::EPSILON.cbrt(),
            diff_step: 1.490_116_119_384_765_6e-8,
            fit_type: FitType::Odr,
        }
    }
}

/// Fit mode. `Ols` fixes input corrections at zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitType {
    Odr,
    Ols,
}

/// Coordinates an ODR fit, equivalent to `scipy.odr.ODR`.
#[derive(Debug, Clone)]
pub struct ODR {
    pub data: Data,
    pub model: Model,
    pub beta0: Vec<f64>,
    pub delta0: Option<Vec<f64>>,
    pub ifixb: Option<Vec<bool>>,
    pub ifixx: Option<Vec<bool>>,
    pub options: OdrOptions,
}

impl ODR {
    pub fn new(data: Data, model: Model, beta0: Vec<f64>) -> Result<Self, OdrError> {
        validate_beta0(&beta0, model.parameter_count)?;
        Ok(Self {
            data,
            model,
            beta0,
            delta0: None,
            ifixb: None,
            ifixx: None,
            options: OdrOptions::default(),
        })
    }

    /// `scipy.odr.ODR(data, model)` without `beta0`: the starting parameters come from the
    /// model's `estimate(data)`. Fails, as SciPy does, when the model has no estimator.
    pub fn from_model_estimate(data: Data, model: Model) -> Result<Self, OdrError> {
        let Some(estimate) = model.estimate.clone() else {
            return Err(OdrError::InvalidArgument {
                detail: String::from("must specify beta0 or provide an estimator with the model"),
            });
        };
        let beta0 = estimate(&data);
        Self::new(data, model, beta0)
    }

    pub fn with_options(mut self, options: OdrOptions) -> Result<Self, OdrError> {
        validate_options(options)?;
        self.options = options;
        Ok(self)
    }

    pub fn with_delta0(mut self, delta0: Vec<f64>) -> Result<Self, OdrError> {
        validate_finite_slice("delta0", &delta0)?;
        if delta0.len() != self.data.x.len() {
            return Err(OdrError::InvalidArgument {
                detail: format!(
                    "delta0 length must match x length (got {} and {})",
                    delta0.len(),
                    self.data.x.len()
                ),
            });
        }
        self.delta0 = Some(delta0);
        Ok(self)
    }

    /// Set beta freedom flags. `true` means free, `false` means fixed.
    pub fn with_beta_free(mut self, free: Vec<bool>) -> Result<Self, OdrError> {
        if free.len() != self.beta0.len() {
            return Err(OdrError::InvalidArgument {
                detail: format!(
                    "ifixb/free mask length must match beta0 length (got {} and {})",
                    free.len(),
                    self.beta0.len()
                ),
            });
        }
        self.ifixb = Some(free);
        Ok(self)
    }

    /// Set input correction freedom flags. `true` means free, `false` means fixed.
    pub fn with_input_free(mut self, free: Vec<bool>) -> Result<Self, OdrError> {
        if free.len() != self.data.x.len() {
            return Err(OdrError::InvalidArgument {
                detail: format!(
                    "ifixx/free mask length must match x length (got {} and {})",
                    free.len(),
                    self.data.x.len()
                ),
            });
        }
        self.ifixx = Some(free);
        Ok(self)
    }

    pub fn set_job(&mut self, fit_type: FitType) {
        self.options.fit_type = fit_type;
    }

    pub fn run(&self) -> Result<Output, OdrError> {
        self.run_impl(true)
    }

    /// Reference solver that forms the full `(n_beta + n_free_delta)`-dimensional
    /// finite-difference Jacobian and normal equations (the pre-structural
    /// `O(n^3)`-per-iteration path). Retained as an in-tree oracle: the structured
    /// solver reproduces its `beta`/`delta`/covariance to convergence tolerance.
    #[doc(hidden)]
    pub fn run_dense_reference(&self) -> Result<Output, OdrError> {
        self.run_impl(false)
    }

    fn run_impl(&self, use_structured: bool) -> Result<Output, OdrError> {
        if self.model.implicit {
            return Err(OdrError::Unsupported {
                detail: String::from("implicit ODR models are represented but not solved yet"),
            });
        }
        validate_options(self.options)?;
        let y = self.data.response()?;
        if y.len() > self.data.x.len() {
            return Err(OdrError::InvalidArgument {
                detail: format!(
                    "explicit ODR requires at least as many x values as y observations (got {} and {})",
                    self.data.x.len(),
                    y.len()
                ),
            });
        }

        let beta_free = freedom_mask(self.ifixb.as_deref(), self.beta0.len());
        let data_free = self
            .ifixx
            .as_deref()
            .or(self.data.fix.as_deref())
            .map_or_else(|| vec![true; self.data.x.len()], ToOwned::to_owned);
        let delta_free = match self.options.fit_type {
            FitType::Odr => data_free,
            FitType::Ols => vec![false; self.data.x.len()],
        };

        let free_beta_indices = free_indices(&beta_free);
        let free_delta_indices = free_indices(&delta_free);
        if free_beta_indices.is_empty() && free_delta_indices.is_empty() {
            return Err(OdrError::InvalidArgument {
                detail: String::from("at least one beta or delta variable must be free"),
            });
        }

        let delta0 = self
            .delta0
            .clone()
            .unwrap_or_else(|| vec![0.0; self.data.x.len()]);
        let variable0 = pack_variables(
            &self.beta0,
            &delta0,
            &free_beta_indices,
            &free_delta_indices,
        );
        let model = self.model.clone();
        let data = self.data.clone();
        let beta_template = self.beta0.clone();
        let delta_template = delta0.clone();
        let residual_beta_indices = free_beta_indices.clone();
        let residual_delta_indices = free_delta_indices.clone();
        let residuals = move |variables: &[f64]| {
            let (beta, delta) = unpack_variables(
                variables,
                &beta_template,
                &delta_template,
                &residual_beta_indices,
                &residual_delta_indices,
            );
            weighted_residuals(&data, &model, &beta, &delta).unwrap_or_else(|_| {
                vec![f64::INFINITY; data.response().map_or(data.x.len(), |resp| resp.len())]
            })
        };
        let ctx = OdrStruct {
            model: &self.model,
            data: &self.data,
            y,
            beta_template: &self.beta0,
            delta_template: &delta0,
            free_beta: &free_beta_indices,
            free_delta: &free_delta_indices,
            diff_step: self.options.diff_step,
        };
        // The structured (Schur-eliminated) path is valid only for scalar
        // pointwise models: response `i` depends on `x[i]` and not on any other
        // input coordinate. Arbitrary custom closures can legally couple output
        // rows, so they stay on the dense reference path unless the model is
        // explicitly marked scalar-separable.
        let can_structured =
            use_structured && self.data.x.len() == y.len() && self.model.is_scalar_separable();
        let solved = if can_structured {
            let sr = solve_odr_structured(&ctx, &residuals, &variable0, self.options)?;
            SolvedFit {
                x: sr.x,
                cost: sr.cost,
                success: sr.success,
                message: sr.message,
                info: sr.info,
                nfev: sr.nfev,
                njev: sr.njev,
                nit: sr.nit,
                cov: CovSource::Struct(sr.jac),
            }
        } else {
            let lr = solve_least_squares(
                &residuals,
                |x: &[f64], r: &[f64]| ctx.dense_jac(&residuals, x, r),
                &variable0,
                self.options,
            )?;
            SolvedFit {
                x: lr.x,
                cost: lr.cost,
                success: lr.success,
                message: lr.message,
                info: lr.info,
                nfev: lr.nfev,
                njev: lr.njev,
                nit: lr.nit,
                cov: CovSource::Dense(lr.jac),
            }
        };
        let result = solved;
        let (beta, delta) = unpack_variables(
            &result.x,
            &self.beta0,
            &delta0,
            &free_beta_indices,
            &free_delta_indices,
        );
        let xplus = add_slices(&self.data.x, &delta);
        let yfit = self.model.evaluate(&beta, &xplus);
        if yfit.len() != y.len() {
            return Err(OdrError::InvalidArgument {
                detail: format!(
                    "model output length must match y length (got {} and {})",
                    yfit.len(),
                    y.len()
                ),
            });
        }
        validate_finite_slice("model output", &yfit)?;
        // SciPy's `eps` is ODRPACK's F = f(x + δ; β) − y, fitted minus observed
        // (frankenscipy-i49i2); the solver's internal residual has the opposite sign.
        let eps = y
            .iter()
            .zip(yfit.iter())
            .map(|(observed, fitted)| fitted - observed)
            .collect::<Vec<_>>();
        let sum_square_eps = weighted_sum_square(&eps, &self.data.we);
        let sum_square_delta = weighted_sum_square(&delta, &self.data.wd);
        let sum_square = sum_square_eps + sum_square_delta;
        let mut info = result.info;
        let mut stopreason = vec![result.message];
        // ODRPACK never reaches its covariance step (DODVCV) after a numerical error, so SciPy
        // reports res_var, cov_beta and sd_beta as zeros there.
        let covariance = if result.info == ODRPACK_NUMERICAL_ERROR {
            OdrpackCovariance::skipped(beta.len())
        } else {
            let (columns, observed_rows) = match &result.cov {
                CovSource::Dense(jac) => (
                    dense_profiled_beta_columns(jac, free_beta_indices.len()),
                    dense_observed_rows(jac, y.len()),
                ),
                CovSource::Struct(jac) => (ctx.profiled_beta_columns(jac), ctx.observed_rows(jac)),
            };
            OdrpackCovariance::new(
                columns,
                observed_rows,
                sum_square,
                &self.beta0,
                &free_beta_indices,
            )
        };
        // ODRPACK's tens digit of `info`: 1 when it dropped some of the free parameters from the
        // covariance, 2 when it dropped all of them; SciPy reports both with this stopreason.
        if covariance.rank_deficiency > 0 {
            info += if covariance.rank_deficiency < free_beta_indices.len() {
                10
            } else {
                20
            };
            stopreason.insert(0, String::from("Problem is not full rank at solution"));
        }
        let OdrpackCovariance {
            res_var,
            cov_beta,
            sd_beta,
            rcond: inv_condnum,
            ..
        } = covariance;
        Ok(Output {
            beta,
            sd_beta,
            cov_beta,
            delta,
            eps,
            xplus,
            y: yfit,
            res_var,
            sum_square,
            sum_square_delta,
            sum_square_eps,
            inv_condnum,
            rel_error: result.cost.abs() * f64::EPSILON,
            info,
            stopreason,
            nfev: result.nfev,
            njev: result.njev,
            nit: result.nit,
            success: result.success,
        })
    }

    pub fn restart(&self, additional_iterations: usize) -> Result<Output, OdrError> {
        let mut restarted = self.clone();
        restarted.options.maxit = self.options.maxit.saturating_add(additional_iterations);
        restarted.run()
    }
}

/// Output from an ODR fit, equivalent to `scipy.odr.Output`.
#[derive(Debug, Clone, PartialEq)]
pub struct Output {
    pub beta: Vec<f64>,
    /// `√(res_var · var_k)`, 0 for a fixed or dropped (rank-deficient) parameter, and NaN where
    /// SciPy's is: an overflowed variance times a zero `res_var`.
    pub sd_beta: Vec<f64>,
    /// SciPy's `cov_beta` multiplied by `res_var` (SciPy leaves it unscaled).
    pub cov_beta: Vec<Vec<f64>>,
    /// Input corrections: `xplus = x + delta`.
    pub delta: Vec<f64>,
    /// Response errors as SciPy reports them: fitted minus observed (`Output::y − Data::y`).
    pub eps: Vec<f64>,
    pub xplus: Vec<f64>,
    pub y: Vec<f64>,
    pub res_var: f64,
    pub sum_square: f64,
    pub sum_square_delta: f64,
    pub sum_square_eps: f64,
    /// ODRPACK's reciprocal condition estimate of the scaled β Jacobian kept for the covariance.
    pub inv_condnum: f64,
    pub rel_error: f64,
    pub info: i32,
    pub stopreason: Vec<String>,
    pub nfev: usize,
    pub njev: usize,
    pub nit: usize,
    pub success: bool,
}

impl Output {
    pub fn pprint(&self) -> String {
        format!(
            "beta={:?}\nsd_beta={:?}\nres_var={:.6e}\nsum_square={:.6e}\ninfo={}\nstopreason={:?}",
            self.beta, self.sd_beta, self.res_var, self.sum_square, self.info, self.stopreason
        )
    }
}

/// Low-level helper matching `scipy.odr.odr`.
pub fn odr<F>(fcn: F, beta0: Vec<f64>, y: Vec<f64>, x: Vec<f64>) -> Result<Output, OdrError>
where
    F: Fn(&[f64], &[f64]) -> Vec<f64> + Send + Sync + 'static,
{
    ODR::new(Data::new(x, y)?, Model::new(fcn), beta0)?.run()
}

/// Names of the `scipy.odr` public surface tracked by the docs census.
pub fn public_api_symbols() -> &'static [&'static str] {
    &[
        "Data",
        "RealData",
        "Model",
        "ODR",
        "Output",
        "odr",
        "OdrWarning",
        "OdrError",
        "OdrStop",
        "polynomial",
        "exponential",
        "multilinear",
        "unilinear",
        "quadratic",
        "models",
        "odrpack",
    ]
}

// The standard models below are `scipy.odr.models`: the same function, parameter order,
// analytic `fjacb`/`fjacd` and `estimate` (all ones: ODRPACK's scaling dislikes zeros).

/// `scipy.odr.unilinear`: `y = β0·x + β1`.
pub fn unilinear() -> Model {
    Model::new(|beta, x| {
        let slope = beta.first().copied().unwrap_or(0.0);
        let intercept = beta.get(1).copied().unwrap_or(0.0);
        x.iter().map(|value| slope * value + intercept).collect()
    })
    .with_fjacb(|_beta, x| x.iter().map(|&value| vec![value, 1.0]).collect())
    .with_fjacd(|beta, x| {
        let slope = beta.first().copied().unwrap_or(0.0);
        x.iter().map(|_| vec![slope]).collect()
    })
    .with_estimate(|_data| vec![1.0, 1.0])
    .with_name("unilinear")
    .with_parameter_count(2)
    .with_scalar_separable(true)
}

/// `scipy.odr.quadratic`: `y = β0·x² + β1·x + β2` -- HIGHEST degree first, unlike
/// [`polynomial`]. This used to be `polynomial(2)` (lowest degree first), so a SciPy `beta0`
/// fitted a different model.
pub fn quadratic() -> Model {
    Model::new(|beta, x| {
        let (a, b, c) = (
            beta.first().copied().unwrap_or(0.0),
            beta.get(1).copied().unwrap_or(0.0),
            beta.get(2).copied().unwrap_or(0.0),
        );
        x.iter().map(|&value| value * (value * a + b) + c).collect()
    })
    .with_fjacb(|_beta, x| {
        x.iter()
            .map(|&value| vec![value * value, value, 1.0])
            .collect()
    })
    .with_fjacd(|beta, x| {
        let (a, b) = (
            beta.first().copied().unwrap_or(0.0),
            beta.get(1).copied().unwrap_or(0.0),
        );
        x.iter().map(|&value| vec![2.0 * value * a + b]).collect()
    })
    .with_estimate(|_data| vec![1.0, 1.0, 1.0])
    .with_name("quadratic")
    .with_parameter_count(3)
    .with_scalar_separable(true)
}

/// `scipy.odr.polynomial(order)`: `y = β0 + β1·x + … + β_order·x^order` (lowest degree first).
pub fn polynomial(order: usize) -> Model {
    Model::new(move |beta, x| {
        x.iter()
            .map(|value| {
                beta.iter()
                    .take(order + 1)
                    .rev()
                    .fold(0.0, |acc, coeff| acc * value + coeff)
            })
            .collect()
    })
    .with_fjacb(move |_beta, x| {
        x.iter()
            .map(|&value| {
                let mut power = 1.0;
                (0..=order)
                    .map(|_| {
                        let term = power;
                        power *= value;
                        term
                    })
                    .collect()
            })
            .collect()
    })
    .with_fjacd(move |beta, x| {
        x.iter()
            .map(|&value| {
                // Σ k·β_k·x^(k−1), by Horner on the derivative's coefficients.
                let derivative = (1..=order).rev().fold(0.0, |acc, k| {
                    acc * value + k as f64 * beta.get(k).copied().unwrap_or(0.0)
                });
                vec![derivative]
            })
            .collect()
    })
    .with_estimate(move |_data| vec![1.0; order + 1])
    .with_name(format!("polynomial({order})"))
    .with_parameter_count(order + 1)
    .with_scalar_separable(true)
}

/// `scipy.odr.exponential`: `y = β0 + exp(β1·x)`. This used to be a three-parameter
/// `β0·exp(β1·x) + β2`, so a SciPy `beta0` did not even have the right length.
pub fn exponential() -> Model {
    Model::new(|beta, x| {
        let offset = beta.first().copied().unwrap_or(0.0);
        let rate = beta.get(1).copied().unwrap_or(0.0);
        x.iter()
            .map(|&value| offset + (rate * value).exp())
            .collect()
    })
    .with_fjacb(|beta, x| {
        let rate = beta.get(1).copied().unwrap_or(0.0);
        x.iter()
            .map(|&value| vec![1.0, value * (rate * value).exp()])
            .collect()
    })
    .with_fjacd(|beta, x| {
        let rate = beta.get(1).copied().unwrap_or(0.0);
        x.iter()
            .map(|&value| vec![rate * (rate * value).exp()])
            .collect()
    })
    .with_estimate(|_data| vec![1.0, 1.0])
    .with_name("exponential")
    .with_parameter_count(2)
    .with_scalar_separable(true)
}

/// `scipy.odr.multilinear` with `input_dim` inputs per observation, laid out contiguously in
/// `x` (`x[i·input_dim + k]` is input `k` of observation `i`): `y_i = β0 + Σ_k β_{k+1}·x_ik`.
pub fn multilinear(input_dim: usize) -> Model {
    Model::new(move |beta, x| {
        if input_dim == 0 {
            return Vec::new();
        }
        x.chunks_exact(input_dim)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .fold(beta.first().copied().unwrap_or(0.0), |acc, (idx, value)| {
                        acc + beta.get(idx + 1).copied().unwrap_or(0.0) * value
                    })
            })
            .collect()
    })
    .with_fjacb(move |_beta, x| {
        if input_dim == 0 {
            return Vec::new();
        }
        x.chunks_exact(input_dim)
            .map(|row| std::iter::once(1.0).chain(row.iter().copied()).collect())
            .collect()
    })
    .with_fjacd(move |beta, x| {
        if input_dim == 0 {
            return Vec::new();
        }
        let slopes: Vec<f64> = (1..=input_dim)
            .map(|k| beta.get(k).copied().unwrap_or(0.0))
            .collect();
        x.chunks_exact(input_dim).map(|_| slopes.clone()).collect()
    })
    .with_estimate(move |_data| vec![1.0; input_dim + 1])
    .with_name(format!("multilinear({input_dim})"))
    .with_parameter_count(input_dim + 1)
    .with_scalar_separable(true)
}

fn validate_beta0(beta0: &[f64], expected: Option<usize>) -> Result<(), OdrError> {
    validate_finite_slice("beta0", beta0)?;
    if beta0.is_empty() {
        return Err(OdrError::InvalidArgument {
            detail: String::from("beta0 must contain at least one parameter"),
        });
    }
    if let Some(expected) = expected
        && beta0.len() != expected
    {
        return Err(OdrError::InvalidArgument {
            detail: format!(
                "beta0 length must match model parameter count (got {} and {expected})",
                beta0.len()
            ),
        });
    }
    Ok(())
}

fn validate_options(options: OdrOptions) -> Result<(), OdrError> {
    if options.maxit == 0 {
        return Err(OdrError::InvalidArgument {
            detail: String::from("maxit must be at least 1"),
        });
    }
    for (name, value) in [
        ("sstol", options.sstol),
        ("partol", options.partol),
        ("diff_step", options.diff_step),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(OdrError::InvalidArgument {
                detail: format!("{name} must be positive and finite"),
            });
        }
    }
    Ok(())
}

fn validate_finite_slice(name: &str, values: &[f64]) -> Result<(), OdrError> {
    if let Some((idx, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(OdrError::NonFiniteInput {
            detail: format!("{name}[{idx}] must be finite, got {value}"),
        });
    }
    Ok(())
}

fn validate_weight_vec(name: &str, values: &[f64], expected_len: usize) -> Result<(), OdrError> {
    if values.len() != expected_len {
        return Err(OdrError::InvalidArgument {
            detail: format!(
                "{name} length must match expected length (got {} and {expected_len})",
                values.len()
            ),
        });
    }
    for (idx, value) in values.iter().copied().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err(OdrError::InvalidArgument {
                detail: format!("{name}[{idx}] must be finite and non-negative, got {value}"),
            });
        }
    }
    Ok(())
}

fn stddev_to_weights(
    name: &str,
    values: &[f64],
    expected_len: usize,
) -> Result<Vec<f64>, OdrError> {
    if values.len() != expected_len {
        return Err(OdrError::InvalidArgument {
            detail: format!(
                "{name} length must match expected length (got {} and {expected_len})",
                values.len()
            ),
        });
    }
    values
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, value)| {
            if !value.is_finite() || value <= 0.0 {
                Err(OdrError::InvalidArgument {
                    detail: format!("{name}[{idx}] must be finite and positive, got {value}"),
                })
            } else {
                Ok(1.0 / (value * value))
            }
        })
        .collect()
}

fn freedom_mask(mask: Option<&[bool]>, len: usize) -> Vec<bool> {
    mask.map_or_else(|| vec![true; len], ToOwned::to_owned)
}

fn free_indices(mask: &[bool]) -> Vec<usize> {
    mask.iter()
        .copied()
        .enumerate()
        .filter_map(|(idx, free)| free.then_some(idx))
        .collect()
}

fn pack_variables(
    beta: &[f64],
    delta: &[f64],
    beta_indices: &[usize],
    delta_indices: &[usize],
) -> Vec<f64> {
    beta_indices
        .iter()
        .map(|&idx| beta[idx])
        .chain(delta_indices.iter().map(|&idx| delta[idx]))
        .collect()
}

fn unpack_variables(
    variables: &[f64],
    beta_template: &[f64],
    delta_template: &[f64],
    beta_indices: &[usize],
    delta_indices: &[usize],
) -> (Vec<f64>, Vec<f64>) {
    let mut beta = beta_template.to_vec();
    let mut delta = delta_template.to_vec();
    for (position, &idx) in beta_indices.iter().enumerate() {
        beta[idx] = variables[position];
    }
    let delta_offset = beta_indices.len();
    for (position, &idx) in delta_indices.iter().enumerate() {
        delta[idx] = variables[delta_offset + position];
    }
    (beta, delta)
}

fn weighted_residuals(
    data: &Data,
    model: &Model,
    beta: &[f64],
    delta: &[f64],
) -> Result<Vec<f64>, OdrError> {
    let y = data.response()?;
    let xplus = add_slices(&data.x, delta);
    let prediction = model.evaluate(beta, &xplus);
    if prediction.len() != y.len() {
        return Err(OdrError::InvalidArgument {
            detail: format!(
                "model output length must match y length (got {} and {})",
                prediction.len(),
                y.len()
            ),
        });
    }
    if prediction.iter().any(|value| !value.is_finite()) {
        return Err(OdrError::NonFiniteInput {
            detail: String::from("model returned a non-finite prediction"),
        });
    }
    Ok(y.iter()
        .zip(prediction.iter())
        .zip(data.we.iter())
        .map(|((observed, fitted), weight)| weight.sqrt() * (observed - fitted))
        .chain(
            delta
                .iter()
                .zip(data.wd.iter())
                .map(|(correction, weight)| weight.sqrt() * correction),
        )
        .collect())
}

fn add_slices(left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| lhs + rhs)
        .collect()
}

fn weighted_sum_square(values: &[f64], weights: &[f64]) -> f64 {
    values
        .iter()
        .zip(weights.iter())
        .map(|(value, weight)| weight * value * value)
        .sum()
}

#[derive(Debug, Clone, PartialEq)]
struct LocalLeastSquaresResult {
    x: Vec<f64>,
    cost: f64,
    success: bool,
    message: String,
    /// ODRPACK's `info`: 1 converged, 4 iteration limit, [`ODRPACK_NUMERICAL_ERROR`].
    info: i32,
    nfev: usize,
    njev: usize,
    nit: usize,
    jac: Vec<Vec<f64>>,
}

/// Dense Levenberg–Marquardt on `residuals`. `jacobian(x, r)` returns the residual Jacobian at
/// `x` (base residual `r`) and the number of residual evaluations it spent.
fn solve_least_squares<F, J>(
    residuals: F,
    jacobian: J,
    x0: &[f64],
    options: OdrOptions,
) -> Result<LocalLeastSquaresResult, OdrError>
where
    F: Fn(&[f64]) -> Vec<f64>,
    J: Fn(&[f64], &[f64]) -> Result<(Vec<Vec<f64>>, usize), OdrError>,
{
    if x0.is_empty() {
        return Err(OdrError::InvalidArgument {
            detail: String::from("least-squares variable vector must not be empty"),
        });
    }
    let mut x = x0.to_vec();
    let mut r = residuals(&x);
    let mut nfev = 1usize;
    if r.len() < x.len() {
        return Err(OdrError::InvalidArgument {
            detail: format!(
                "number of residuals ({}) must be >= number of variables ({})",
                r.len(),
                x.len()
            ),
        });
    }
    validate_finite_slice("initial residuals", &r)?;
    let mut cost = 0.5 * dot(&r, &r);
    let mut damping = 1.0e-3;
    let (mut jac, jac_evals) = jacobian(&x, &r)?;
    nfev += jac_evals;
    let mut njev = 1usize;
    for nit in 0..options.maxit {
        let gradient = jt_residual(&jac, &r);
        let cosine = gradient_cosine(&gradient, &column_norms(&jac), &r);
        if cosine.is_nan() {
            return Ok(LocalLeastSquaresResult {
                x,
                cost,
                // status: ‖r‖, a column norm or Jᵀr overflowed (ODRPACK's info 60000)
                success: false,
                message: String::from(NUMERICAL_ERROR_STOPREASON),
                info: ODRPACK_NUMERICAL_ERROR,
                nfev,
                njev,
                nit,
                jac,
            });
        }
        if cosine <= options.sstol {
            return Ok(LocalLeastSquaresResult {
                x,
                cost,
                // status: max_j |cos(r, J_j)| <= sstol (MINPACK's scale-free gtol test)
                success: true,
                message: String::from("gradient tolerance reached"),
                info: 1,
                nfev,
                njev,
                nit,
                jac,
            });
        }

        // JᵀJ/Jᵀr are damping-independent, so build them ONCE per Jacobian and reuse them
        // across the ≤8 damping retries below (each retry only re-adds `damping` to a copy
        // of the diagonal). `jac`/`r` are fixed for the whole inner loop — an accepted step
        // recomputes them and `break`s to the next outer iteration. `ODR_LMSTEP_HOIST_DISABLE`
        // restores the per-retry rebuild (the A/B baseline). Byte-identical.
        let cached_normal = if ODR_LMSTEP_HOIST_DISABLE.load(std::sync::atomic::Ordering::Relaxed) {
            None
        } else {
            Some(build_normal_equations(&jac, &r))
        };

        let mut accepted = false;
        for _ in 0..8 {
            let step = match &cached_normal {
                Some((normal, rhs)) => solve_damped_normal(normal, rhs, damping)?,
                None => solve_lm_step(&jac, &r, damping)?,
            };
            let candidate = x
                .iter()
                .zip(step.iter())
                .map(|(value, delta)| value + delta)
                .collect::<Vec<_>>();
            let candidate_r = residuals(&candidate);
            nfev += 1;
            if step_within_partol(&step, &x, options.partol) {
                // Take the converged step, as ODRPACK does (see the structured solver).
                let candidate_cost = 0.5 * dot(&candidate_r, &candidate_r);
                if candidate_cost <= cost {
                    x = candidate;
                    r = candidate_r;
                    cost = candidate_cost;
                    let (new_jac, jac_evals) = jacobian(&x, &r)?;
                    jac = new_jac;
                    nfev += jac_evals;
                    njev += 1;
                }
                return Ok(LocalLeastSquaresResult {
                    x,
                    cost,
                    // status: max|step| <= partol·max|x| (ODRPACK's relative parameter test)
                    success: true,
                    message: String::from("parameter tolerance reached"),
                    info: 1,
                    nfev,
                    njev,
                    nit,
                    jac,
                });
            }
            if candidate_r.iter().any(|value| !value.is_finite()) {
                damping *= 10.0;
                continue;
            }
            let candidate_cost = 0.5 * dot(&candidate_r, &candidate_r);
            if candidate_cost < cost {
                let rel_change = relative_ss_reduction(cost, candidate_cost);
                x = candidate;
                r = candidate_r;
                cost = candidate_cost;
                let (new_jac, jac_evals) = jacobian(&x, &r)?;
                jac = new_jac;
                nfev += jac_evals;
                njev += 1;
                damping = (damping * 0.3).max(1.0e-12);
                accepted = true;
                if rel_change <= options.sstol {
                    return Ok(LocalLeastSquaresResult {
                        x,
                        cost,
                        // status: accepted step with (SS_old - SS_new)/SS_old <= sstol
                        success: true,
                        message: String::from("sum-of-squares tolerance reached"),
                        info: 1,
                        nfev,
                        njev,
                        nit: nit + 1,
                        jac,
                    });
                }
                break;
            }
            damping *= 10.0;
        }
        if !accepted {
            damping *= 10.0;
        }
    }
    Ok(LocalLeastSquaresResult {
        x,
        cost,
        success: false,
        message: String::from("maximum iterations reached"),
        info: 4,
        nfev,
        njev,
        nit: options.maxit,
        jac,
    })
}

// ---------------------------------------------------------------------------
// Structured ODR solver.
//
// The explicit-ODR least-squares problem packs the model parameters `beta`
// (p free entries) and one x-error `delta` per data point (m free entries) into
// a single variable vector; the residual is `[ sqrt(we)·(y - f(beta, x+delta)) ;
// sqrt(wd)·delta ]` (length 2n). The dense reference path forms the full
// (p+m)-column finite-difference Jacobian and (p+m)² normal equations, so each
// iteration costs O(n·(p+m)²) ~ O(n³) with m ~ n.
//
// Because observation i's prediction depends only on x_i + delta_i, the entire
// delta block of the Jacobian is a single diagonal (response rows) plus the
// analytic diagonal sqrt(wd) (penalty rows). That lets us
//   * build the whole delta diagonal from ONE batched model evaluation, and
//   * eliminate the delta variables from the LM step via the Schur complement of
//     the diagonal delta–delta block, leaving a p×p system in beta,
// dropping the per-iteration cost to O(n·p²) — linear in the data size. The step
// and covariance are mathematically identical to the dense path (equal up to
// floating-point roundoff), so the fit converges to the same solution.
// ---------------------------------------------------------------------------

/// Structured finite-difference Jacobian for an explicit ODR problem.
/// Response residual rows are `0..n`; delta-penalty rows are `n..2n`.
struct StructJac {
    /// `a[i][k]` = ∂r_resp[i] / ∂(free β_k), the dense response-block β columns.
    a: Vec<Vec<f64>>,
    /// `d_diag[j]` = ∂r_resp[dj] / ∂δ_dj, the response-block δ diagonal
    /// (`dj = free_delta[j]`).
    d_diag: Vec<f64>,
    /// `b[j]` = sqrt(wd[dj]), the analytic δ-penalty diagonal.
    b: Vec<f64>,
}

struct OdrStruct<'a> {
    model: &'a Model,
    data: &'a Data,
    y: &'a [f64],
    beta_template: &'a [f64],
    delta_template: &'a [f64],
    free_beta: &'a [usize],
    free_delta: &'a [usize],
    diff_step: f64,
}

struct StructResult {
    x: Vec<f64>,
    cost: f64,
    success: bool,
    message: String,
    info: i32,
    nfev: usize,
    njev: usize,
    nit: usize,
    jac: StructJac,
}

impl OdrStruct<'_> {
    fn n(&self) -> usize {
        self.data.x.len()
    }
    fn p(&self) -> usize {
        self.free_beta.len()
    }
    fn m(&self) -> usize {
        self.free_delta.len()
    }

    /// Structured Jacobian at packed variables `x` with base residual `r`: from the
    /// model's `fjacb`/`fjacd` when it has them, else by finite differences. The FD β
    /// columns reproduce the dense path bit-for-bit; the FD δ diagonal comes from a
    /// single all-δ-perturbed model evaluation (each output point depends only on
    /// its own input, so the batched value equals the one-at-a-time perturbation).
    fn jac(&self, x: &[f64], r: &[f64]) -> Result<StructJac, OdrError> {
        let (n, p, m) = (self.n(), self.p(), self.m());
        let (beta, delta) = unpack_variables(
            x,
            self.beta_template,
            self.delta_template,
            self.free_beta,
            self.free_delta,
        );
        let xplus = add_slices(&self.data.x, &delta);

        let mut a = vec![vec![0.0; p]; n];
        if let Some(fjacb) = &self.model.fjacb {
            // r_i = √we_i·(y_i − f_i), so ∂r_i/∂β = −√we_i·∂f_i/∂β.
            let jb = analytic_model_jacobian(fjacb, "fjacb", &beta, &xplus, n, beta.len())?;
            for (i, row) in a.iter_mut().enumerate() {
                let scale = -self.data.we[i].sqrt();
                for (k, value) in row.iter_mut().enumerate() {
                    *value = scale * jb[i][self.free_beta[k]];
                }
            }
        } else {
            for k in 0..p {
                let h = self.diff_step * x[k].abs().max(1.0);
                let mut beta_pert = beta.clone();
                beta_pert[self.free_beta[k]] += h;
                let pred = self.model.evaluate(&beta_pert, &xplus);
                if pred.len() != n {
                    return Err(jac_length_error(pred.len(), n));
                }
                for i in 0..n {
                    let r_pert = self.data.we[i].sqrt() * (self.y[i] - pred[i]);
                    if !r_pert.is_finite() {
                        return Err(jac_nonfinite_error());
                    }
                    a[i][k] = (r_pert - r[i]) / h;
                }
            }
        }

        let mut d_diag = vec![0.0; m];
        let mut b = vec![0.0; m];
        if m > 0
            && let Some(fjacd) = &self.model.fjacd
        {
            // The structured path has one input per observation (x.len() == n).
            let jd = analytic_model_jacobian(fjacd, "fjacd", &beta, &xplus, n, 1)?;
            for (j, &dj) in self.free_delta.iter().enumerate() {
                b[j] = self.data.wd[dj].sqrt();
                d_diag[j] = -self.data.we[dj].sqrt() * jd[dj][0];
            }
        } else if m > 0 {
            let mut xpert = xplus.clone();
            let mut hs = vec![0.0; m];
            for (j, &dj) in self.free_delta.iter().enumerate() {
                let h = self.diff_step * x[p + j].abs().max(1.0);
                hs[j] = h;
                xpert[dj] += h;
                b[j] = self.data.wd[dj].sqrt();
            }
            let pred = self.model.evaluate(&beta, &xpert);
            if pred.len() != n {
                return Err(jac_length_error(pred.len(), n));
            }
            for (j, &dj) in self.free_delta.iter().enumerate() {
                let r_pert = self.data.we[dj].sqrt() * (self.y[dj] - pred[dj]);
                if !r_pert.is_finite() {
                    return Err(jac_nonfinite_error());
                }
                d_diag[j] = (r_pert - r[dj]) / hs[j];
            }
        }
        Ok(StructJac { a, d_diag, b })
    }

    /// The full residual Jacobian for the dense solver at packed variables `x` (base residual
    /// `r`), with the number of residual evaluations it spent. Columns come from the model's
    /// `fjacb`/`fjacd` where it has them and from finite differences otherwise; with neither
    /// this is exactly [`finite_diff_jacobian`].
    fn dense_jac<F>(
        &self,
        residuals: &F,
        x: &[f64],
        r: &[f64],
    ) -> Result<(Vec<Vec<f64>>, usize), OdrError>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        if self.model.fjacb.is_none() && self.model.fjacd.is_none() {
            return Ok((
                finite_diff_jacobian(residuals, x, r, self.diff_step)?,
                x.len(),
            ));
        }
        let (p, m) = (self.p(), self.m());
        let n_obs = self.y.len();
        let (beta, delta) = unpack_variables(
            x,
            self.beta_template,
            self.delta_template,
            self.free_beta,
            self.free_delta,
        );
        let xplus = add_slices(&self.data.x, &delta);
        let mut jac = vec![vec![0.0; x.len()]; r.len()];
        let mut evals = 0;

        // Response rows are r_i = √we_i·(y_i − f_i); δ-penalty rows (n_obs + k) are √wd_k·δ_k.
        if let Some(fjacb) = &self.model.fjacb {
            let jb = analytic_model_jacobian(fjacb, "fjacb", &beta, &xplus, n_obs, beta.len())?;
            for (i, row) in jac.iter_mut().take(n_obs).enumerate() {
                let scale = -self.data.we[i].sqrt();
                for (k, &beta_index) in self.free_beta.iter().enumerate() {
                    row[k] = scale * jb[i][beta_index];
                }
            }
        } else {
            finite_diff_columns(residuals, x, r, self.diff_step, 0..p, &mut jac)?;
            evals += p;
        }
        if let Some(fjacd) = &self.model.fjacd {
            let inputs = self.data.x.len();
            if !inputs.is_multiple_of(n_obs) {
                return Err(OdrError::InvalidArgument {
                    detail: format!(
                        "fjacd needs the same number of inputs per observation ({inputs} inputs, {n_obs} observations)"
                    ),
                });
            }
            let per_obs = inputs / n_obs;
            let jd = analytic_model_jacobian(fjacd, "fjacd", &beta, &xplus, n_obs, per_obs)?;
            for (j, &dj) in self.free_delta.iter().enumerate() {
                let (obs, input) = (dj / per_obs, dj % per_obs);
                jac[obs][p + j] = -self.data.we[obs].sqrt() * jd[obs][input];
                jac[n_obs + dj][p + j] = self.data.wd[dj].sqrt();
            }
        } else {
            finite_diff_columns(residuals, x, r, self.diff_step, p..p + m, &mut jac)?;
            evals += m;
        }
        Ok((jac, evals))
    }

    /// Number of model evaluations the structured Jacobian consumes: one full
    /// evaluation per free β column, plus one batched evaluation for all δ, each
    /// skipped when the model supplies that Jacobian analytically.
    fn jac_evals(&self) -> usize {
        let beta_evals = if self.model.fjacb.is_some() {
            0
        } else {
            self.p()
        };
        beta_evals + usize::from(self.m() > 0 && self.model.fjacd.is_none())
    }

    /// `max_abs(Jᵀr)`, matching the dense gradient-tolerance convergence test.
    /// The structured form of [`gradient_cosine`]: β column k holds `a[·][k]` on the response
    /// rows; free-δ column j holds `d_diag[j]` (response row) and `b[j]` (its δ row).
    fn gradient_cosine(&self, sj: &StructJac, r: &[f64]) -> f64 {
        let (n, p) = (self.n(), self.p());
        let mut gradient = Vec::with_capacity(p + self.free_delta.len());
        let mut norms = Vec::with_capacity(p + self.free_delta.len());
        for k in 0..p {
            let (mut g, mut sq) = (0.0, 0.0);
            for (row, &ri) in sj.a.iter().zip(r).take(n) {
                g += row[k] * ri;
                sq += row[k] * row[k];
            }
            gradient.push(g);
            norms.push(sq.sqrt());
        }
        for (j, &dj) in self.free_delta.iter().enumerate() {
            gradient.push(sj.d_diag[j] * r[dj] + sj.b[j] * r[n + dj]);
            norms.push(sj.d_diag[j].hypot(sj.b[j]));
        }
        gradient_cosine(&gradient, &norms, r)
    }

    /// LM step `(JᵀJ + μI) s = -Jᵀr` solved by eliminating the diagonal δ block:
    /// reduce to a p×p system in the β step, then back-substitute each δ step.
    /// Equal to the dense `solve_lm_step` up to roundoff.
    fn lm_step(&self, sj: &StructJac, r: &[f64], damping: f64) -> Option<Vec<f64>> {
        let (n, p, m) = (self.n(), self.p(), self.m());

        // g_beta = Aᵀ r_resp (the β part of Jᵀr); rhs_beta = -g_beta.
        let mut g_beta = vec![0.0; p];
        for (i, &ri) in r.iter().take(n).enumerate() {
            for (k, g) in g_beta.iter_mut().enumerate() {
                *g += sj.a[i][k] * ri;
            }
        }

        // Per-row Schur coefficient c_i (1 for fixed-δ rows, (b²+μ)/g for free-δ
        // rows) and the δ-elimination contribution to the reduced right-hand side.
        let mut cvec = vec![1.0; n];
        let mut red_rhs: Vec<f64> = g_beta.iter().map(|g| -g).collect();
        for (j, &dj) in self.free_delta.iter().enumerate() {
            let a_j = sj.d_diag[j];
            let b_j = sj.b[j];
            let g_j = a_j * a_j + b_j * b_j + damping;
            if g_j.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
                return None;
            }
            cvec[dj] = (b_j * b_j + damping) / g_j;
            let rhs_delta_j = -(a_j * r[dj] + b_j * r[n + dj]);
            let coeff = (a_j / g_j) * rhs_delta_j;
            for (k, value) in red_rhs.iter_mut().enumerate() {
                *value -= coeff * sj.a[dj][k];
            }
        }

        // Reduced β system R = μI + Σ_i c_i a_i a_iᵀ.
        let mut rmat = vec![vec![0.0; p]; p];
        for (k, row) in rmat.iter_mut().enumerate() {
            row[k] = damping;
        }
        for (i, &ci) in cvec.iter().enumerate().take(n) {
            if ci == 0.0 {
                continue;
            }
            let ai = &sj.a[i];
            for lhs in 0..p {
                let scaled = ai[lhs] * ci;
                if scaled == 0.0 {
                    continue;
                }
                let row = &mut rmat[lhs];
                for rhs in 0..p {
                    row[rhs] += scaled * ai[rhs];
                }
            }
        }

        let s_beta = if p > 0 {
            gaussian_solve(rmat, red_rhs)?
        } else {
            Vec::new()
        };

        let mut step = Vec::with_capacity(p + m);
        step.extend_from_slice(&s_beta);
        for (j, &dj) in self.free_delta.iter().enumerate() {
            let a_j = sj.d_diag[j];
            let b_j = sj.b[j];
            let g_j = a_j * a_j + b_j * b_j + damping;
            let rhs_delta_j = -(a_j * r[dj] + b_j * r[n + dj]);
            let a_dot_s: f64 = sj.a[dj].iter().zip(&s_beta).map(|(a, s)| a * s).sum();
            step.push((rhs_delta_j - a_j * a_dot_s) / g_j);
        }
        Some(step)
    }

    /// The free-β columns of the weighted Jacobian with the δ corrections profiled out, in
    /// ODRPACK's sign (`FJACB = √we·∂f/∂β`): DODSTP's `TFJACB = Ω⁻ᵀ·FJACB` for one input per
    /// observation, row `i` divided by `Ω_i = √(1 + we_i·(∂f_i/∂x_i)²/wd_i)` when `δ_i` is free.
    /// Their Gram matrix is the Schur complement of JᵀJ's δ–δ block, built in O(n·p) without
    /// forming it. See [`dense_profiled_beta_columns`] for the dense path.
    fn profiled_beta_columns(&self, sj: &StructJac) -> Vec<Vec<f64>> {
        let (n, p) = (self.n(), self.p());
        let mut omega = vec![1.0; n];
        for (j, &dj) in self.free_delta.iter().enumerate() {
            // d/b = −√we·f′/√wd. A δ with no effect on its response profiles out nothing; a zero
            // wd lets δ absorb the response error entirely (Ω = ∞, the row drops out).
            let ratio = if sj.d_diag[j] == 0.0 {
                0.0
            } else {
                sj.d_diag[j] / sj.b[j]
            };
            omega[dj] = (1.0 + ratio * ratio).sqrt();
        }
        (0..p)
            .map(|k| (0..n).map(|i| -sj.a[i][k] / omega[i]).collect())
            .collect()
    }

    /// Observations with a nonzero weighted derivative with respect to a free β or, when its δ
    /// is free, to its own input: ODRPACK's count behind the degrees of freedom (DODVCV's IDF).
    fn observed_rows(&self, sj: &StructJac) -> usize {
        let mut observed: Vec<bool> =
            sj.a.iter()
                .map(|row| row.iter().any(|&value| value != 0.0))
                .collect();
        for (j, &dj) in self.free_delta.iter().enumerate() {
            if sj.d_diag[j] != 0.0 {
                observed[dj] = true;
            }
        }
        observed.into_iter().filter(|&hit| hit).count()
    }
}

/// Evaluates a model's analytic Jacobian at `(beta, xplus)` and checks that it is `rows × cols`
/// and finite.
fn analytic_model_jacobian(
    jacobian: &JacobianFn,
    name: &str,
    beta: &[f64],
    xplus: &[f64],
    rows: usize,
    cols: usize,
) -> Result<Vec<Vec<f64>>, OdrError> {
    let value = jacobian(beta, xplus);
    if value.len() != rows || value.iter().any(|row| row.len() != cols) {
        return Err(OdrError::InvalidArgument {
            detail: format!(
                "{name} must return {rows} rows (one per observation) of {cols} entries"
            ),
        });
    }
    if value.iter().flatten().any(|entry| !entry.is_finite()) {
        return Err(OdrError::NonFiniteInput {
            detail: format!("{name} returned a non-finite entry"),
        });
    }
    Ok(value)
}

fn jac_length_error(got: usize, expected: usize) -> OdrError {
    OdrError::InvalidArgument {
        detail: format!(
            "residual length changed during finite differences (got {got} and {expected})"
        ),
    }
}

fn jac_nonfinite_error() -> OdrError {
    OdrError::NonFiniteInput {
        detail: String::from("finite-difference residuals[..] must be finite"),
    }
}

/// Structured Levenberg–Marquardt loop for explicit ODR. Mirrors the control
/// flow of [`solve_least_squares`] (identical damping schedule, tolerances, and
/// accept/reject logic) but uses the structured Jacobian and Schur-eliminated
/// step, so it costs O(n·p²) per iteration instead of O(n³).
fn solve_odr_structured<F>(
    ctx: &OdrStruct,
    residuals: &F,
    x0: &[f64],
    options: OdrOptions,
) -> Result<StructResult, OdrError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    if x0.is_empty() {
        return Err(OdrError::InvalidArgument {
            detail: String::from("least-squares variable vector must not be empty"),
        });
    }
    let mut x = x0.to_vec();
    let mut r = residuals(&x);
    let mut nfev = 1usize;
    if r.len() < x.len() {
        return Err(OdrError::InvalidArgument {
            detail: format!(
                "number of residuals ({}) must be >= number of variables ({})",
                r.len(),
                x.len()
            ),
        });
    }
    validate_finite_slice("initial residuals", &r)?;
    let mut cost = 0.5 * dot(&r, &r);
    let mut damping = 1.0e-3;
    let mut jac = ctx.jac(&x, &r)?;
    nfev += ctx.jac_evals();
    let mut njev = 1usize;
    for nit in 0..options.maxit {
        let cosine = ctx.gradient_cosine(&jac, &r);
        if cosine.is_nan() {
            return Ok(StructResult {
                x,
                cost,
                // status: ‖r‖, a column norm or Jᵀr overflowed (ODRPACK's info 60000)
                success: false,
                message: String::from(NUMERICAL_ERROR_STOPREASON),
                info: ODRPACK_NUMERICAL_ERROR,
                nfev,
                njev,
                nit,
                jac,
            });
        }
        if cosine <= options.sstol {
            return Ok(StructResult {
                x,
                cost,
                // status: max_j |cos(r, J_j)| <= sstol (structured MINPACK gtol test)
                success: true,
                message: String::from("gradient tolerance reached"),
                info: 1,
                nfev,
                njev,
                nit,
                jac,
            });
        }

        let mut accepted = false;
        for _ in 0..8 {
            let step = ctx
                .lm_step(&jac, &r, damping)
                .ok_or_else(|| OdrError::SolverFailure {
                    detail: String::from("normal equations are singular"),
                })?;
            let candidate = x
                .iter()
                .zip(step.iter())
                .map(|(value, delta)| value + delta)
                .collect::<Vec<_>>();
            let candidate_r = residuals(&candidate);
            nfev += 1;
            if step_within_partol(&step, &x, options.partol) {
                // ODRPACK tests parameter convergence on the step it has just TAKEN, so the
                // returned β includes it. Stopping before it left the fit one Gauss–Newton
                // step short: 3 − 5.3e-6 on exact data where SciPy returns 3.0
                // (frankenscipy-szq1n.12).
                let candidate_cost = 0.5 * dot(&candidate_r, &candidate_r);
                if candidate_cost <= cost {
                    x = candidate;
                    r = candidate_r;
                    cost = candidate_cost;
                    jac = ctx.jac(&x, &r)?;
                    nfev += ctx.jac_evals();
                    njev += 1;
                }
                return Ok(StructResult {
                    x,
                    cost,
                    // status: max|step| <= partol·max|x| (ODRPACK's relative parameter test)
                    success: true,
                    message: String::from("parameter tolerance reached"),
                    info: 1,
                    nfev,
                    njev,
                    nit,
                    jac,
                });
            }
            if candidate_r.iter().any(|value| !value.is_finite()) {
                damping *= 10.0;
                continue;
            }
            let candidate_cost = 0.5 * dot(&candidate_r, &candidate_r);
            if candidate_cost < cost {
                let rel_change = relative_ss_reduction(cost, candidate_cost);
                x = candidate;
                r = candidate_r;
                cost = candidate_cost;
                jac = ctx.jac(&x, &r)?;
                nfev += ctx.jac_evals();
                njev += 1;
                damping = (damping * 0.3).max(1.0e-12);
                accepted = true;
                if rel_change <= options.sstol {
                    return Ok(StructResult {
                        x,
                        cost,
                        // status: accepted step with (SS_old - SS_new)/SS_old <= sstol
                        success: true,
                        message: String::from("sum-of-squares tolerance reached"),
                        info: 1,
                        nfev,
                        njev,
                        nit: nit + 1,
                        jac,
                    });
                }
                break;
            }
            damping *= 10.0;
        }
        if !accepted {
            damping *= 10.0;
        }
    }
    Ok(StructResult {
        x,
        cost,
        success: false,
        message: String::from("maximum iterations reached"),
        info: 4,
        nfev,
        njev,
        nit: options.maxit,
        jac,
    })
}

/// Result of an LM solve, carrying whichever Jacobian representation the chosen
/// solver produced so the covariance can be assembled the matching way.
struct SolvedFit {
    x: Vec<f64>,
    cost: f64,
    success: bool,
    message: String,
    info: i32,
    nfev: usize,
    njev: usize,
    nit: usize,
    cov: CovSource,
}

enum CovSource {
    Dense(Vec<Vec<f64>>),
    Struct(StructJac),
}

fn finite_diff_jacobian<F>(
    residuals: &F,
    x: &[f64],
    r0: &[f64],
    step: f64,
) -> Result<Vec<Vec<f64>>, OdrError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let mut jac = vec![vec![0.0; x.len()]; r0.len()];
    finite_diff_columns(residuals, x, r0, step, 0..x.len(), &mut jac)?;
    Ok(jac)
}

/// Forward-difference columns `cols` of the residual Jacobian into `jac`.
fn finite_diff_columns<F>(
    residuals: &F,
    x: &[f64],
    r0: &[f64],
    step: f64,
    cols: std::ops::Range<usize>,
    jac: &mut [Vec<f64>],
) -> Result<(), OdrError>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    for col in cols {
        let mut x_plus = x.to_vec();
        let h = step * x[col].abs().max(1.0);
        x_plus[col] += h;
        let r_plus = residuals(&x_plus);
        if r_plus.len() != r0.len() {
            return Err(OdrError::InvalidArgument {
                detail: format!(
                    "residual length changed during finite differences (got {} and {})",
                    r_plus.len(),
                    r0.len()
                ),
            });
        }
        validate_finite_slice("finite-difference residuals", &r_plus)?;
        for row in 0..r0.len() {
            jac[row][col] = (r_plus[row] - r0[row]) / h;
        }
    }
    Ok(())
}

/// When `true`, [`solve_lm_step`] builds its `JᵀJ` normal matrix and `Jᵀr` vector serially (the ORIG
/// row-outer accumulation); default `false` fans the build across output rows. Byte-identical.
/// `#[doc(hidden)]` — internal, exposed only for the same-binary A/B benchmark.
#[doc(hidden)]
pub static ODR_LMSTEP_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// When `true`, the dense LM loop rebuilds the `JᵀJ`/`Jᵀr` normal equations on EVERY
/// damping retry (the ORIG behaviour); default `false` builds them ONCE per Jacobian and
/// only re-adds `damping` to a copy of the diagonal for each retry — the normal equations
/// are damping-independent, so within a Levenberg-Marquardt iteration (`jac`/`r` fixed
/// across the ≤8 damping tries) the rebuild was redundant. `JᵀJ` is the O(n²·nrows)
/// per-iteration hot spot; the copy is O(n²). Byte-identical. `#[doc(hidden)]` — A/B knob.
#[doc(hidden)]
pub static ODR_LMSTEP_HOIST_DISABLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Build the damping-independent LM normal equations `(JᵀJ, −Jᵀr)` from the Jacobian and
/// residuals. Split out of [`solve_lm_step`] so the dense LM loop can build them ONCE per
/// Jacobian and reuse them across the damping retries (see [`solve_damped_normal`]).
fn build_normal_equations(jacobian: &[Vec<f64>], residuals: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = jacobian.first().map_or(0, Vec::len);
    let nrows = jacobian.len();
    let mut normal = vec![vec![0.0; n]; n];
    let mut rhs = vec![0.0; n];

    // Each output row `lhs` of the normal matrix and its `rhs[lhs]` are a pure reduction over the
    // Jacobian rows: `normal[lhs][j] = Σ_row J[row][lhs]·J[row][j]` and `rhs[lhs] = −Σ_row
    // J[row][lhs]·r[row]`, both summed in ascending row order. The rows of the output are mutually
    // independent (each owns a disjoint `normal[lhs]` Vec and `rhs[lhs]` slot), so fanning across
    // output-row chunks accumulates every cell in the SAME row order as the serial build → BYTE-
    // IDENTICAL. For ODR the residual count exceeds `n` (delta penalties), so this JᵀJ build is the
    // per-iteration hot spot. Gated on the build work `n²·nrows`.
    let parallel = !ODR_LMSTEP_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed)
        && n >= 32
        && (n as u64) * (n as u64) * (nrows as u64) >= (1 << 20);
    if !parallel {
        for (row_idx, row) in jacobian.iter().enumerate() {
            for lhs in 0..n {
                rhs[lhs] -= row[lhs] * residuals[row_idx];
                for rhs_idx in 0..n {
                    normal[lhs][rhs_idx] += row[lhs] * row[rhs_idx];
                }
            }
        }
    } else {
        let nthreads = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(n);
        let chunk = n.div_ceil(nthreads);
        std::thread::scope(|scope| {
            for (ci, (norm_rows, rhs_slots)) in normal
                .chunks_mut(chunk)
                .zip(rhs.chunks_mut(chunk))
                .enumerate()
            {
                let base = ci * chunk;
                scope.spawn(move || {
                    for (k, (norm_row, rhs_slot)) in
                        norm_rows.iter_mut().zip(rhs_slots.iter_mut()).enumerate()
                    {
                        let lhs = base + k;
                        let mut acc_rhs = 0.0;
                        for (row_idx, row) in jacobian.iter().enumerate() {
                            let jl = row[lhs];
                            acc_rhs -= jl * residuals[row_idx];
                            for (rhs_idx, cell) in norm_row.iter_mut().enumerate() {
                                *cell += jl * row[rhs_idx];
                            }
                        }
                        *rhs_slot = acc_rhs;
                    }
                });
            }
        });
    }
    (normal, rhs)
}

/// Solve `(JᵀJ + damping·I)·step = −Jᵀr` from the cached normal equations, re-adding
/// `damping` to a COPY of `normal`'s diagonal (so the cached `normal`/`rhs` stay reusable
/// across the damping retries). Byte-identical to [`solve_lm_step`]'s in-place add + solve
/// — same matrix `JᵀJ + damping·I`, same `rhs`, same [`gaussian_solve`].
fn solve_damped_normal(
    normal: &[Vec<f64>],
    rhs: &[f64],
    damping: f64,
) -> Result<Vec<f64>, OdrError> {
    let mut m: Vec<Vec<f64>> = normal.iter().map(Clone::clone).collect();
    for (idx, row) in m.iter_mut().enumerate() {
        row[idx] += damping;
    }
    gaussian_solve(m, rhs.to_vec()).ok_or_else(|| OdrError::SolverFailure {
        detail: String::from("normal equations are singular"),
    })
}

/// Build `JᵀJ + damping·I` and solve `·step = −Jᵀr` in one shot (rebuilding the normal
/// equations each call — the ORIG per-damping-retry path, kept as the A/B baseline).
/// A direct Gauss-Jordan solve does ~half the O(n³) work of inverting-then-multiplying.
fn solve_lm_step(
    jacobian: &[Vec<f64>],
    residuals: &[f64],
    damping: f64,
) -> Result<Vec<f64>, OdrError> {
    let (mut normal, rhs) = build_normal_equations(jacobian, residuals);
    for (idx, row) in normal.iter_mut().enumerate() {
        row[idx] += damping;
    }
    gaussian_solve(normal, rhs).ok_or_else(|| OdrError::SolverFailure {
        detail: String::from("normal equations are singular"),
    })
}

/// Solve `matrix · x = rhs` via Gauss-Jordan elimination with partial pivoting, transforming
/// only the single RHS column. Used for the damped LM step, whose diagonal carries the damping.
fn gaussian_solve(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let n = matrix.len();
    if n == 0 || rhs.len() != n || matrix.iter().any(|row| row.len() != n) {
        return None;
    }
    for pivot in 0..n {
        let best = (pivot..n).max_by(|&lhs, &rhs_row| {
            matrix[lhs][pivot]
                .abs()
                .total_cmp(&matrix[rhs_row][pivot].abs())
        })?;
        if matrix[best][pivot].abs() <= 1.0e-14 {
            return None;
        }
        matrix.swap(pivot, best);
        rhs.swap(pivot, best);
        let scale = matrix[pivot][pivot];
        for value in &mut matrix[pivot] {
            *value /= scale;
        }
        rhs[pivot] /= scale;
        let pivot_row = matrix[pivot].clone();
        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            if factor == 0.0 {
                continue;
            }
            for (value, pivot_value) in matrix[row].iter_mut().zip(&pivot_row) {
                *value -= factor * pivot_value;
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    Some(rhs)
}

// Convergence tests (frankenscipy-szq1n.7). All three used to be ABSOLUTE, so the verdict
// depended on the units of the data: `max|Jᵀr| <= sstol` reported success at nit = 0 with β0
// unchanged for small-magnitude data, `|step| <= partol*(1 + |x|)` put a floor of about 6e-6 on
// the step so parameters near 1e-7 "converged" at once, and the SS change divided by
// max(cost, 1) was absolute whenever cost < 1. Each is now invariant to rescaling the data.

/// ODRPACK's `info` after a numerical error, which SciPy reports as "Numerical error detected".
/// ODRPACK adds a rank digit (60010, 60020) that this solver does not compute.
const ODRPACK_NUMERICAL_ERROR: i32 = 60_000;
const NUMERICAL_ERROR_STOPREASON: &str = "Numerical error detected";

/// MINPACK's scale-free `gtol` measure: the largest `|cos|` of the angle between the residual
/// and a Jacobian column, `max_j |(Jᵀr)_j| / (‖r‖·‖J_j‖)`. Zero-norm columns are skipped; an
/// exact fit (`r = 0`) is optimal.
///
/// NaN when the measure cannot be formed: once `‖r‖²`, a column's `‖J_j‖²` or `(Jᵀr)_j`
/// overflows, every cosine is `g/∞ = 0` or `∞/∞ = NaN` (which `f64::max` drops), and the test
/// read as converged at β0. SciPy's ODRPACK stops those fits with info 60000/60010
/// (frankenscipy-vfs3g); the callers report the same status on a NaN.
fn gradient_cosine(gradient: &[f64], column_norms: &[f64], residuals: &[f64]) -> f64 {
    let r_norm = dot(residuals, residuals).sqrt();
    if r_norm == 0.0 {
        return 0.0;
    }
    if !r_norm.is_finite() || gradient.iter().chain(column_norms).any(|v| !v.is_finite()) {
        return f64::NAN;
    }
    gradient
        .iter()
        .zip(column_norms)
        .filter(|&(_, &norm)| norm > 0.0)
        .map(|(&g, &norm)| g.abs() / (r_norm * norm))
        .fold(0.0_f64, f64::max)
}

fn column_norms(jacobian: &[Vec<f64>]) -> Vec<f64> {
    let n = jacobian.first().map_or(0, Vec::len);
    let mut sq = vec![0.0; n];
    for row in jacobian {
        for (acc, &value) in sq.iter_mut().zip(row) {
            *acc += value * value;
        }
    }
    sq.into_iter().map(f64::sqrt).collect()
}

/// ODRPACK's relative parameter-change test, `‖step‖ <= partol·‖x‖` (no additive floor).
fn step_within_partol(step: &[f64], x: &[f64], partol: f64) -> bool {
    max_abs(step) <= partol * max_abs(x)
}

/// ODRPACK's relative sum-of-squares reduction, `(SS_old - SS_new) / SS_old`; only called
/// for an accepted step, so `old > new >= 0`.
fn relative_ss_reduction(old: f64, new: f64) -> f64 {
    (old - new) / old
}

fn jt_residual(jacobian: &[Vec<f64>], residuals: &[f64]) -> Vec<f64> {
    let n = jacobian.first().map_or(0, Vec::len);
    let mut gradient = vec![0.0; n];
    for (row_idx, row) in jacobian.iter().enumerate() {
        for col in 0..n {
            gradient[col] += row[col] * residuals[row_idx];
        }
    }
    gradient
}

fn dot(values: &[f64], rhs: &[f64]) -> f64 {
    values
        .iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| lhs * rhs)
        .sum()
}

fn max_abs(values: &[f64]) -> f64 {
    values
        .iter()
        .copied()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

// ---------------------------------------------------------------------------
// Covariance of the estimated β: ODRPACK's DODVCV (frankenscipy-i49i2).
//
// ODRPACK profiles the δ corrections out of the weighted β Jacobian (DODSTP's TFJACB), divides
// column k by the scale of β_k (DSCLB, set from β0) and QR-factors the result with column
// pivoting (DQRDC). While the reciprocal condition estimate of R (DTRCO) is at most EPSFCN it
// drops the column that estimate's approximate null vector points at (DCHEX), so the rank test
// is RELATIVE: a Jacobian passes or fails it the same way at any magnitude. (RᵀR)⁻¹ of the kept
// columns (DPODI), unscaled, is the covariance before the residual variance; fixed and dropped
// parameters get zero variance, and the degrees of freedom are the observations with a nonzero
// weighted derivative minus the rank.
//
// This replaced an inverse of the full normal matrix JᵀJ behind an ABSOLUTE 1e-14 pivot test: a
// β·x fit with x ~ 1e-100 failed it (sd_beta 0 where SciPy has 0.0148), and a parameter the
// model ignores made every variance NaN where SciPy drops that one column.
// ---------------------------------------------------------------------------

/// ODRPACK's EPSFCN in the covariance rank test: η, the relative noise in the model's values.
/// ODRPACK estimates it from five model evaluations at β0 (DETAF), floored at DMPREC = 2⁻⁵²; on
/// every fit measured for frankenscipy-i49i2 it returned exactly 2⁻⁵². fsci does not spend those
/// evaluations, so for a model noisier than machine precision it keeps a column whose reciprocal
/// condition lies in (2⁻⁵², η] that ODRPACK would drop.
const ODRPACK_RANK_TOLERANCE: f64 = f64::EPSILON;

/// ODRPACK's covariance step (DODVCV), expanded to every parameter.
struct OdrpackCovariance {
    res_var: f64,
    /// SciPy's `cov_beta` times `res_var`.
    cov_beta: Vec<Vec<f64>>,
    sd_beta: Vec<f64>,
    /// DTRCO's estimate for the R that was kept (0 with no free parameter).
    rcond: f64,
    /// Free parameters dropped from R (ODRPACK's IRANK).
    rank_deficiency: usize,
}

impl OdrpackCovariance {
    /// The zeros SciPy reports when ODRPACK skipped its covariance step.
    fn skipped(np: usize) -> Self {
        Self {
            res_var: 0.0,
            cov_beta: vec![vec![0.0; np]; np],
            sd_beta: vec![0.0; np],
            rcond: 0.0,
            rank_deficiency: 0,
        }
    }

    /// DODVCV from the profiled Jacobian `columns` (one per entry of `free_beta`, in ODRPACK's
    /// sign), the number of observations with a nonzero weighted derivative, the weighted sum of
    /// squares, and β0, which sets the column scales.
    fn new(
        mut columns: Vec<Vec<f64>>,
        observed_rows: usize,
        sum_square: f64,
        beta0: &[f64],
        free_beta: &[usize],
    ) -> Self {
        let npp = free_beta.len();
        let ssf = odrpack_beta_scale(beta0);
        for (column, &k) in columns.iter_mut().zip(free_beta) {
            for value in column.iter_mut() {
                *value /= ssf[k];
            }
        }
        let mut jpvt = qr_column_pivoted(&mut columns);
        let rows = columns.first().map_or(0, Vec::len);
        let mut r: Vec<Vec<f64>> = (0..npp)
            .map(|i| {
                (0..npp)
                    .map(|j| {
                        if i <= j && i < rows {
                            columns[j][i]
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();

        // DODSTP's elimination loop: drop the column the null vector points at (IDAMAX: its
        // first entry of largest magnitude) while RCOND <= EPSFCN. A NaN estimate drops nothing.
        let mut kept = npp;
        let mut rcond = 0.0;
        while kept > 0 {
            let (estimate, null_vector) = triangular_rcond(&r, kept);
            rcond = estimate;
            if !(estimate <= ODRPACK_RANK_TOLERANCE) {
                break;
            }
            let drop = (0..kept).fold(0, |best, i| {
                if null_vector[i].abs() > null_vector[best].abs() {
                    i
                } else {
                    best
                }
            });
            if drop + 1 != kept {
                triangle_move_column_last(&mut r, drop, kept);
                jpvt[drop..kept].rotate_left(1);
            }
            kept -= 1;
        }
        invert_gram_from_triangle(&mut r, kept);

        let res_var = if observed_rows > kept {
            sum_square / (observed_rows - kept) as f64
        } else {
            sum_square
        };
        // (RᵀR)⁻¹ back in parameter order; fixed and dropped parameters keep zero.
        let np = beta0.len();
        let mut variance = vec![vec![0.0; np]; np];
        for i in 0..kept {
            for j in i..kept {
                let (a, b) = (free_beta[jpvt[i]], free_beta[jpvt[j]]);
                variance[a][b] = r[i][j];
                variance[b][a] = r[i][j];
            }
        }
        // SD = √(RVAR·var)/SSF with no clamp: an overflowed variance times a zero res_var is NaN,
        // as SciPy reports it. VCV = var/(SSF_i·SSF_j), then scaled by res_var.
        let sd_beta = (0..np)
            .map(|k| (res_var * variance[k][k]).sqrt() / ssf[k])
            .collect();
        let cov_beta = (0..np)
            .map(|i| {
                (0..np)
                    .map(|j| variance[i][j] / (ssf[i] * ssf[j]) * res_var)
                    .collect()
            })
            .collect();
        Self {
            res_var,
            cov_beta,
            sd_beta,
            rcond,
            rank_deficiency: npp - kept,
        }
    }
}

/// ODRPACK's DSCLB: the scale of each β from β0 (every parameter, fixed ones included).
fn odrpack_beta_scale(beta0: &[f64]) -> Vec<f64> {
    let bmax = beta0.iter().fold(0.0_f64, |acc, b| acc.max(b.abs()));
    if bmax == 0.0 {
        return vec![1.0; beta0.len()];
    }
    let bmin = beta0
        .iter()
        .filter(|b| **b != 0.0)
        .fold(bmax, |acc, b| acc.min(b.abs()));
    let spread = bmax.log10() - bmin.log10() >= 1.0;
    beta0
        .iter()
        .map(|&b| {
            if b == 0.0 {
                10.0 / bmin
            } else if spread {
                1.0 / b.abs()
            } else {
                1.0 / bmax
            }
        })
        .collect()
}

/// The dense Jacobian's free-β columns (`..p`) with its free-δ columns (`p..`) profiled out, in
/// ODRPACK's sign: a Householder QR of the δ columns applied to the β columns, keeping the rows
/// below the δ triangle. Their Gram matrix is the Schur complement of JᵀJ's δ–δ block, the matrix
/// DODSTP builds row by row (TFJACB), here also for models whose responses share inputs.
fn dense_profiled_beta_columns(jacobian: &[Vec<f64>], p: usize) -> Vec<Vec<f64>> {
    let width = jacobian.first().map_or(0, Vec::len);
    // Column-major and negated: the residual is y − f, ODRPACK's FJACB is ∂f/∂β.
    let mut cols: Vec<Vec<f64>> = (0..width)
        .map(|j| jacobian.iter().map(|row| -row[j]).collect())
        .collect();
    let mut top = 0;
    for d in p..width {
        let (before, from_d) = cols.split_at_mut(d);
        let Some((delta_column, later)) = from_d.split_first_mut() else {
            break;
        };
        let reflector = &mut delta_column[top..];
        // A δ column that is zero below the triangle has nothing left to profile out.
        if householder_reflector(reflector).is_none() {
            continue;
        }
        for column in before[..p].iter_mut().chain(later.iter_mut()) {
            apply_householder(reflector, &mut column[top..]);
        }
        top += 1;
    }
    cols.truncate(p);
    for column in &mut cols {
        column.drain(..top);
    }
    cols
}

/// Response rows (`..n_obs`) of the dense Jacobian with a nonzero weighted derivative for some
/// free β or free δ: ODRPACK's count behind the degrees of freedom (DODVCV's IDF).
fn dense_observed_rows(jacobian: &[Vec<f64>], n_obs: usize) -> usize {
    jacobian
        .iter()
        .take(n_obs)
        .filter(|row| row.iter().any(|&value| value != 0.0))
        .count()
}

/// Euclidean norm with no overflow or underflow in the squares (LAPACK's scaled `dnrm2`):
/// columns of order 1e-160 have subnormal squares.
fn nrm2(values: &[f64]) -> f64 {
    let (mut scale, mut ssq) = (0.0_f64, 1.0_f64);
    for &value in values {
        if value != 0.0 {
            let magnitude = value.abs();
            if scale < magnitude {
                ssq = 1.0 + ssq * (scale / magnitude).powi(2);
                scale = magnitude;
            } else {
                ssq += (magnitude / scale).powi(2);
            }
        }
    }
    scale * ssq.sqrt()
}

/// Turn `x` into a LINPACK (DQRDC) Householder reflector in place and return the signed norm it
/// reflects `x` onto, `x ↦ (−norm, 0, …)`. `None`, with `x` untouched, for a zero vector.
fn householder_reflector(x: &mut [f64]) -> Option<f64> {
    let norm = nrm2(x);
    if norm == 0.0 {
        return None;
    }
    let norm = if x[0] == 0.0 {
        norm
    } else {
        norm.copysign(x[0])
    };
    let inverse = 1.0 / norm;
    for value in x.iter_mut() {
        *value *= inverse;
    }
    x[0] += 1.0;
    Some(norm)
}

/// Apply a reflector from [`householder_reflector`] to `column`, both starting at the same row.
fn apply_householder(reflector: &[f64], column: &mut [f64]) {
    let t = -dot(reflector, column) / reflector[0];
    for (value, &v) in column.iter_mut().zip(reflector) {
        *value += t * v;
    }
}

/// LINPACK's DQRDC pivoting over every column (ODRPACK's call, all JPVT = 0): Householder QR of
/// the matrix held column-major in `cols`, bringing forward the remaining column of largest
/// (downdated) norm at each step. Leaves R's upper triangle in `cols[j][..=j]` and returns the
/// original index of each column.
fn qr_column_pivoted(cols: &mut [Vec<f64>]) -> Vec<usize> {
    let p = cols.len();
    let n = cols.first().map_or(0, Vec::len);
    let mut jpvt: Vec<usize> = (0..p).collect();
    let mut norms: Vec<f64> = cols.iter().map(|column| nrm2(column)).collect();
    let mut reference = norms.clone();
    for l in 0..n.min(p) {
        if l + 1 < p {
            let mut largest = 0.0;
            let mut pick = l;
            for j in l..p {
                if norms[j] > largest {
                    largest = norms[j];
                    pick = j;
                }
            }
            if pick != l {
                cols.swap(l, pick);
                norms[pick] = norms[l];
                reference[pick] = reference[l];
                jpvt.swap(l, pick);
            }
        }
        if l + 1 == n {
            break;
        }
        let (done, rest) = cols.split_at_mut(l + 1);
        let reflector = &mut done[l][l..];
        let Some(norm) = householder_reflector(reflector) else {
            continue;
        };
        for (offset, column) in rest.iter_mut().enumerate() {
            let j = l + 1 + offset;
            apply_householder(reflector, &mut column[l..]);
            if norms[j] == 0.0 {
                continue;
            }
            // Downdate the column norm; recompute it once cancellation would make that inexact.
            let tt = (1.0 - (column[l].abs() / norms[j]).powi(2)).max(0.0);
            if 1.0 + 0.05 * tt * (norms[j] / reference[j]).powi(2) == 1.0 {
                norms[j] = nrm2(&column[l + 1..]);
                reference[j] = norms[j];
            } else {
                norms[j] *= tt.sqrt();
            }
        }
        reflector[0] = -norm;
    }
    jpvt
}

/// LINPACK's DTRCO for the upper triangle of the `k × k` leading block of `r`: the reciprocal
/// condition estimate `ynorm/‖R‖₁` and the approximate null vector it was built from.
fn triangular_rcond(r: &[Vec<f64>], k: usize) -> (f64, Vec<f64>) {
    let tnorm = (0..k)
        .map(|j| (0..=j).map(|i| r[i][j].abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    // Solve Rᵀy = e, choosing e = ±1 to make y large.
    let mut ek = 1.0_f64;
    let mut z = vec![0.0; k];
    for kk in 0..k {
        let diagonal = r[kk][kk];
        if z[kk] != 0.0 {
            ek = ek.abs().copysign(-z[kk]);
        }
        if (ek - z[kk]).abs() > diagonal.abs() {
            let s = diagonal.abs() / (ek - z[kk]).abs();
            for value in &mut z {
                *value *= s;
            }
            ek *= s;
        }
        let mut wk = ek - z[kk];
        let mut wkm = -ek - z[kk];
        let mut s = wk.abs();
        let mut sm = wkm.abs();
        if diagonal == 0.0 {
            wk = 1.0;
            wkm = 1.0;
        } else {
            wk /= diagonal;
            wkm /= diagonal;
        }
        if kk + 1 < k {
            for j in kk + 1..k {
                sm += (z[j] + wkm * r[kk][j]).abs();
                z[j] += wk * r[kk][j];
                s += z[j].abs();
            }
            if s < sm {
                let w = wkm - wk;
                wk = wkm;
                for j in kk + 1..k {
                    z[j] += w * r[kk][j];
                }
            }
        }
        z[kk] = wk;
    }
    let s = 1.0 / z.iter().map(|value| value.abs()).sum::<f64>();
    for value in &mut z {
        *value *= s;
    }
    // Solve Rz = y.
    let mut ynorm = 1.0;
    for kk in (0..k).rev() {
        let diagonal = r[kk][kk];
        if z[kk].abs() > diagonal.abs() {
            let s = diagonal.abs() / z[kk].abs();
            for value in &mut z {
                *value *= s;
            }
            ynorm *= s;
        }
        if diagonal == 0.0 {
            z[kk] = 1.0;
        } else {
            z[kk] /= diagonal;
        }
        let w = -z[kk];
        for i in 0..kk {
            z[i] += w * r[i][kk];
        }
    }
    let s = 1.0 / z.iter().map(|value| value.abs()).sum::<f64>();
    for value in &mut z {
        *value *= s;
    }
    ynorm *= s;
    let rcond = if tnorm == 0.0 { 0.0 } else { ynorm / tnorm };
    (rcond, z)
}

/// BLAS DROTG: the rotation with `[c s; −s c]·[a; b] = [r; 0]`, as `(r, c, s)`.
fn givens_rotation(a: f64, b: f64) -> (f64, f64, f64) {
    let scale = a.abs() + b.abs();
    if scale == 0.0 {
        return (0.0, 1.0, 0.0);
    }
    let larger = if a.abs() > b.abs() { a } else { b };
    let radius = (scale * ((a / scale).powi(2) + (b / scale).powi(2)).sqrt()).copysign(larger);
    (radius, a / radius, b / radius)
}

/// LINPACK's DCHEX, job 2, with `P = L = l`: move column `k` (0-based) of the `l × l` upper
/// triangle in `r` to the last position, shift columns `k+1..l` left, and restore the triangle
/// with Givens rotations. ODRPACK uses it to put the column it drops behind the kept ones.
fn triangle_move_column_last(r: &mut [Vec<f64>], k: usize, l: usize) {
    // 1-based from here on, as in LINPACK: column `kk` moves to position `l`.
    let kk = k + 1;
    let lmk = l - kk;
    let mut c = vec![0.0; l + 1];
    let mut s = vec![0.0; l + 1];
    for i in 1..=kk {
        s[lmk + i] = r[i - 1][kk - 1];
    }
    for j in kk..l {
        for i in 1..=j {
            r[i - 1][j - 1] = r[i - 1][j];
        }
        s[j - kk + 1] = r[j][j];
    }
    for i in 1..=kk {
        r[i - 1][l - 1] = s[lmk + i];
    }
    for i in kk + 1..=l {
        r[i - 1][l - 1] = 0.0;
    }
    for j in kk..=l {
        if j != kk {
            for i in kk..=(j - 1).min(l - 1) {
                let ii = i - kk + 1;
                let t = c[ii] * r[i - 1][j - 1] + s[ii] * r[i][j - 1];
                r[i][j - 1] = c[ii] * r[i][j - 1] - s[ii] * r[i - 1][j - 1];
                r[i - 1][j - 1] = t;
            }
        }
        if j < l {
            let jj = j - kk + 1;
            let (radius, cosine, sine) = givens_rotation(r[j - 1][j - 1], s[jj]);
            r[j - 1][j - 1] = radius;
            c[jj] = cosine;
            s[jj] = sine;
        }
    }
}

/// LINPACK's DPODI, job 1: overwrite the upper triangle of the `k × k` leading block of `r`, an
/// upper-triangular R, with the upper triangle of (RᵀR)⁻¹.
fn invert_gram_from_triangle(r: &mut [Vec<f64>], k: usize) {
    // R⁻¹ in place.
    for kk in 0..k {
        r[kk][kk] = 1.0 / r[kk][kk];
        let t = -r[kk][kk];
        for i in 0..kk {
            r[i][kk] *= t;
        }
        for j in kk + 1..k {
            let t = r[kk][j];
            r[kk][j] = 0.0;
            for i in 0..=kk {
                r[i][j] += t * r[i][kk];
            }
        }
    }
    // R⁻¹·R⁻ᵀ.
    for j in 0..k {
        for kk in 0..j {
            let t = r[kk][j];
            for i in 0..=kk {
                r[i][kk] += t * r[i][j];
            }
        }
        let t = r[j][j];
        for i in 0..=j {
            r[i][j] *= t;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(lhs: f64, rhs: f64, tol: f64) {
        assert!(
            (lhs - rhs).abs() <= tol,
            "expected {lhs} ~= {rhs} within {tol}"
        );
    }

    fn assert_vec_close(lhs: &[f64], rhs: &[f64], tol: f64) {
        assert_eq!(lhs.len(), rhs.len());
        for (left, right) in lhs.iter().zip(rhs) {
            let scale = right.abs().max(1.0);
            assert!(
                (left - right).abs() / scale <= tol,
                "expected {left} ~= {right} within relative tol {tol}"
            );
        }
    }

    fn assert_output_close(lhs: &Output, rhs: &Output, tol: f64) {
        assert_vec_close(&lhs.beta, &rhs.beta, tol);
        assert_vec_close(&lhs.delta, &rhs.delta, tol);
        assert_vec_close(&lhs.sd_beta, &rhs.sd_beta, tol);
        assert_close(
            lhs.sum_square,
            rhs.sum_square,
            tol * rhs.sum_square.abs().max(1.0),
        );
        assert_eq!(lhs.cov_beta.len(), rhs.cov_beta.len());
        for (left, right) in lhs.cov_beta.iter().zip(&rhs.cov_beta) {
            assert_vec_close(left, right, tol);
        }
    }

    #[test]
    fn public_api_matches_documented_scipy_odr_symbols() {
        assert_eq!(public_api_symbols().len(), 16);
        for symbol in [
            "Data",
            "RealData",
            "Model",
            "ODR",
            "Output",
            "odr",
            "OdrWarning",
            "OdrError",
            "OdrStop",
            "polynomial",
            "exponential",
            "multilinear",
            "unilinear",
            "quadratic",
            "models",
            "odrpack",
        ] {
            assert!(public_api_symbols().contains(&symbol));
        }
    }

    #[test]
    fn realdata_stddevs_convert_to_inverse_variance_weights() -> Result<(), OdrError> {
        let real = RealData::from_stddev(
            vec![0.0, 1.0],
            vec![1.0, 3.0],
            Some(vec![2.0, 4.0]),
            Some(vec![0.5, 0.25]),
        )?;
        assert_eq!(real.data.wd, vec![0.25, 0.0625]);
        assert_eq!(real.data.we, vec![4.0, 16.0]);
        Ok(())
    }

    #[test]
    fn odr_recovers_exact_unilinear_parameters() -> Result<(), OdrError> {
        let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let y = x.iter().map(|value| 2.5 * value - 1.25).collect();
        let output = ODR::new(Data::new(x, y)?, unilinear(), vec![0.0, 0.0])?.run()?;
        assert!(output.success);
        assert_close(output.beta[0], 2.5, 1.0e-6);
        assert_close(output.beta[1], -1.25, 1.0e-6);
        assert!(output.sum_square < 1.0e-10);
        assert!(output.delta.iter().all(|value| value.abs() < 1.0e-6));
        Ok(())
    }

    // frankenscipy-szq1n.7: the tolerance tests were absolute. With every parameter near 1e-7,
    // the first LM step (~1e-7) fell under `partol*(1 + |x|)` ~ 6e-6 and the solver reported
    // "parameter tolerance reached" at beta0 = [0, 0] without moving. SciPy 1.17.1:
    // odr(unilinear, beta0=[0,0]) on y = 2e-7 x + 1e-7 -> beta = [2e-7, 1e-7], info 2.
    #[test]
    fn odr_convergence_tests_are_invariant_to_data_scale() -> Result<(), OdrError> {
        let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let y = x.iter().map(|value| 2.0e-7 * value + 1.0e-7).collect();
        let output = ODR::new(Data::new(x, y)?, unilinear(), vec![0.0, 0.0])?.run()?;
        println!(
            "small-scale ODR: beta={:?} nit={} success={} stop={:?}",
            output.beta, output.nit, output.success, output.stopreason
        );
        assert!(output.success);
        assert!(output.nit > 0, "stopped before taking a step");
        assert_close(output.beta[0], 2.0e-7, 1.0e-12);
        assert_close(output.beta[1], 1.0e-7, 1.0e-12);
        Ok(())
    }

    #[test]
    fn odr_convergence_measures_are_scale_free() {
        // cos(r, J_j): scaling J or r leaves it unchanged; an exact fit is optimal.
        let g = [3.0, 0.0];
        let norms = [2.0, 5.0];
        let r = [1.0, 1.0];
        let c = gradient_cosine(&g, &norms, &r);
        let scaled = gradient_cosine(&[3.0e-9, 0.0], &[2.0e-9, 5.0e-9], &r);
        assert!((c - scaled).abs() < 1e-15 && c > 0.0);
        assert_eq!(gradient_cosine(&g, &norms, &[0.0, 0.0]), 0.0);
        assert!(step_within_partol(&[1e-13], &[1e-7], 1e-5));
        assert!(
            !step_within_partol(&[1e-9], &[1e-7], 1e-5),
            "the old +1 floor accepted this"
        );
        assert!((relative_ss_reduction(1e-10, 0.5e-10) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn odr_matches_scipy_odr_on_noisy_data() -> Result<(), OdrError> {
        // Golden values from scipy.odr.ODR(Data(x,y), unilinear, beta0).run()
        // (SciPy 1.17.1, ODRPACK) on data where x and y both carry error, so the
        // orthogonal-distance fit genuinely differs from OLS. Independent Rust
        // and Fortran ODRPACK implementations agree to convergence tolerance.
        struct Case {
            x: Vec<f64>,
            y: Vec<f64>,
            beta0: Vec<f64>,
            beta: [f64; 2],
            sd_beta: [f64; 2],
            res_var: f64,
            sum_square: f64,
        }
        let cases = [
            Case {
                x: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                y: vec![1.1, 2.9, 5.2, 6.8, 9.1, 10.9],
                beta0: vec![1.0, 0.0],
                beta: [1.979694827136702e0, 1.050762934064172e0],
                sd_beta: [3.983626321501051e-2, 1.205905103654928e-1],
                res_var: 5.639702685253586e-3,
                sum_square: 2.255881074101434e-2,
            },
            Case {
                x: vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
                y: vec![-8.5, -5.1, -3.2, 0.3, 2.7, 5.4, 8.9],
                beta0: vec![1.0, 0.0],
                beta: [2.833504940851501e0, 7.142907832097929e-2],
                sd_beta: [7.371963337556588e-2, 1.472423725008963e-1],
                res_var: 1.680877299337602e-2,
                sum_square: 8.404386496688011e-2,
            },
        ];
        for c in cases {
            let out = ODR::new(Data::new(c.x, c.y)?, unilinear(), c.beta0)?.run()?;
            assert!(out.success);
            // Parameters agree to ODR convergence tolerance (~1e-5).
            assert_close(out.beta[0], c.beta[0], 1.0e-5);
            assert_close(out.beta[1], c.beta[1], 1.0e-5);
            assert_close(out.sd_beta[0], c.sd_beta[0], 1.0e-5);
            assert_close(out.sd_beta[1], c.sd_beta[1], 1.0e-5);
            // The achieved objective matches much more tightly (both find the
            // same minimum even where the flat objective loosely pins beta).
            assert_close(out.res_var, c.res_var, 1.0e-8);
            assert_close(out.sum_square, c.sum_square, 1.0e-8);
        }
        Ok(())
    }

    #[test]
    fn odr_ols_mode_reduces_to_response_residual_fit() -> Result<(), OdrError> {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 3.0, 5.0, 7.0];
        let mut odr = ODR::new(Data::new(x, y)?, unilinear(), vec![1.0, 0.0])?;
        odr.set_job(FitType::Ols);
        let output = odr.run()?;
        assert_close(output.beta[0], 2.0, 1.0e-6);
        assert_close(output.beta[1], 1.0, 1.0e-6);
        assert!(output.delta.iter().all(|value| *value == 0.0));
        Ok(())
    }

    #[test]
    fn fixed_beta_flag_holds_parameter_constant() -> Result<(), OdrError> {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 4.0, 7.0, 10.0];
        let output = ODR::new(Data::new(x, y)?, unilinear(), vec![3.0, 0.0])?
            .with_beta_free(vec![false, true])?
            .run()?;
        assert_close(output.beta[0], 3.0, 0.0);
        assert_close(output.beta[1], 1.0, 1.0e-5);
        assert_close(output.cov_beta[0][0], 0.0, 0.0);
        assert_close(output.sd_beta[0], 0.0, 0.0);
        Ok(())
    }

    #[test]
    fn structured_scalar_odr_matches_dense_reference() -> Result<(), OdrError> {
        let x = (0..24)
            .map(|idx| idx as f64 * 0.05 + (idx % 5) as f64 * 0.002)
            .collect::<Vec<_>>();
        let y = x
            .iter()
            .map(|value| 0.7 + 1.3 * value + 0.4 * (0.9 * value).sin())
            .collect::<Vec<_>>();
        let model = Model::new(|beta: &[f64], x: &[f64]| {
            x.iter()
                .map(|value| beta[0] + beta[1] * value + beta[2] * (beta[3] * value).sin())
                .collect()
        })
        .with_parameter_count(4)
        .with_scalar_separable(true);
        let odr = ODR::new(Data::new(x, y)?, model, vec![0.5, 1.5, 0.2, 0.8])?;

        let structured = odr.run()?;
        let dense = odr.run_dense_reference()?;

        assert_output_close(&structured, &dense, 1.0e-6);
        Ok(())
    }

    #[test]
    fn custom_coupled_model_uses_dense_reference_path() -> Result<(), OdrError> {
        let x = vec![0.2, 0.4, 0.8, 1.6, 3.2];
        let y = vec![0.44, 0.88, 1.76, 3.52, 7.04];
        let model = Model::new(|beta: &[f64], x: &[f64]| {
            x.iter()
                .enumerate()
                .map(|(idx, value)| {
                    let neighbor = x[(idx + 1) % x.len()];
                    beta[0] * value + beta[1] * neighbor
                })
                .collect()
        })
        .with_parameter_count(2);
        let odr = ODR::new(Data::new(x, y)?, model, vec![1.0, 0.1])?;

        let default = odr.run()?;
        let dense = odr.run_dense_reference()?;

        assert_eq!(default, dense);
        Ok(())
    }

    #[test]
    fn structured_covariance_is_zero_when_all_beta_fixed() -> Result<(), OdrError> {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 3.0, 5.0, 7.0];
        let output = ODR::new(Data::new(x, y)?, unilinear(), vec![2.0, 1.0])?
            .with_beta_free(vec![false, false])?
            .run()?;

        assert_eq!(output.cov_beta, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
        assert_eq!(output.sd_beta, vec![0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn covariance_from_jacobian_uses_full_normal_beta_block() {
        let jacobian = vec![vec![1.0, 0.0], vec![1.0, 1.0]];
        // Column 0 is β, column 1 is δ; two observed rows and rank 1 make res_var = 3/1.
        let covariance = OdrpackCovariance::new(
            dense_profiled_beta_columns(&jacobian, 1),
            2,
            3.0,
            &[1.0],
            &[0],
        );

        // The full normal matrix is [[2, 1], [1, 1]]. Its inverse has
        // beta-block [[1]], while the old beta-only inverse was [[0.5]].
        assert_close(covariance.cov_beta[0][0], 3.0, 1.0e-12);
    }

    #[test]
    fn polynomial_model_recovers_quadratic_coefficients() -> Result<(), OdrError> {
        let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let y: Vec<f64> = x
            .iter()
            .map(|value| 1.0 - 2.0 * value + 0.5 * value * value)
            .collect();
        // `quadratic` is SciPy's: highest degree first. SciPy 1.17.1 OLS: [0.5, -2.0, 1.0].
        let mut odr = ODR::new(
            Data::new(x.clone(), y.clone())?,
            quadratic(),
            vec![0.0, 0.0, 0.0],
        )?;
        odr.set_job(FitType::Ols);
        let output = odr.run()?;
        assert_close(output.beta[0], 0.5, 1.0e-6);
        assert_close(output.beta[1], -2.0, 1.0e-6);
        assert_close(output.beta[2], 1.0, 1.0e-6);
        // `polynomial(2)` is lowest degree first.
        let mut odr = ODR::new(Data::new(x, y)?, polynomial(2), vec![0.0, 0.0, 0.0])?;
        odr.set_job(FitType::Ols);
        let output = odr.run()?;
        assert_close(output.beta[0], 1.0, 1.0e-6);
        assert_close(output.beta[1], -2.0, 1.0e-6);
        assert_close(output.beta[2], 0.5, 1.0e-6);
        Ok(())
    }

    #[test]
    fn multilinear_model_uses_chunked_input_rows() -> Result<(), OdrError> {
        let x = vec![1.0, 2.0, 2.0, 1.0, -1.0, 3.0, 0.0, -2.0];
        let (rows, remainder) = x.as_chunks::<2>();
        assert!(remainder.is_empty());
        let y = rows
            .iter()
            .map(|row| 0.5 + 2.0 * row[0] - 3.0 * row[1])
            .collect();
        let mut odr = ODR::new(Data::new(x, y)?, multilinear(2), vec![0.0, 0.0, 0.0])?;
        odr.set_job(FitType::Ols);
        let output = odr.run()?;
        assert_close(output.beta[0], 0.5, 1.0e-6);
        assert_close(output.beta[1], 2.0, 1.0e-6);
        assert_close(output.beta[2], -3.0, 1.0e-6);
        Ok(())
    }

    #[test]
    fn invalid_shapes_and_nonfinite_inputs_fail_closed() {
        assert!(Data::new(vec![0.0], vec![f64::NAN]).is_err());
        assert!(RealData::from_stddev(vec![0.0], vec![1.0], Some(vec![0.0]), None).is_err());
        let data = match Data::new(vec![0.0], vec![1.0]) {
            Ok(data) => data,
            Err(error) => return assert!(matches!(error, OdrError::InvalidArgument { .. })),
        };
        let err = ODR::new(data, unilinear(), vec![0.0]);
        assert!(err.is_err());
    }

    #[test]
    fn odr_linear_matches_scipy_reference_values() -> Result<(), OdrError> {
        // scipy.odr with linear model on y = 2*x + noise
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.1, 3.9, 6.1, 7.9, 10.1];
        let data = Data::new(x, y)?;
        let odr = ODR::new(data, unilinear(), vec![1.0, 0.0])?;
        let output = odr.run()?;
        // scipy.odr 1.17.1, same data and beta0: [2.00192013931748, 0.014239795736218601].
        assert_close(output.beta[0], 2.001_920_139_317_48, 1.0e-6);
        assert_close(output.beta[1], 0.014_239_795_736_218_601, 1.0e-5);
        Ok(())
    }

    #[test]
    fn odr_quadratic_matches_scipy_reference_values() -> Result<(), OdrError> {
        // scipy.odr 1.17.1: ODR(Data(x, y), quadratic, beta0=[1, 0, 0]).run().beta. SciPy's
        // quadratic is β0·x² + β1·x + β2.
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.1, 4.0, 9.1, 15.9, 25.0];
        let data = Data::new(x, y)?;
        let odr = ODR::new(data, quadratic(), vec![1.0, 0.0, 0.0])?;
        let output = odr.run()?;
        let scipy = [
            1.009_873_295_116_383_1,
            -0.087_567_362_326_858_98,
            0.174_187_411_993_578_4,
        ];
        assert_vec_close(&output.beta, &scipy, 1.0e-6);
        Ok(())
    }

    #[test]
    fn odr_polynomial_degree2_matches_scipy_reference_values() -> Result<(), OdrError> {
        // scipy.odr with polynomial model
        // Polynomial params are [c, b, a] for ax^2 + bx + c
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 4.0, 9.0, 16.0]; // y = x^2
        let data = Data::new(x, y)?;
        // Initial guess: c=0, b=0, a=1
        let odr = ODR::new(data, polynomial(2), vec![0.0, 0.0, 1.0])?;
        let output = odr.run()?;
        // Should fit x^2: beta[2]=1, beta[1]=0, beta[0]=0
        assert_close(output.beta[2], 1.0, 0.1);
        assert!(output.beta[1].abs() < 0.2, "linear term should be near 0");
        assert!(output.beta[0].abs() < 0.2, "constant term should be near 0");
        Ok(())
    }

    #[test]
    fn odr_exponential_matches_scipy_reference_values() -> Result<(), OdrError> {
        // scipy.odr's exponential is y = β0 + exp(β1·x) (two parameters). SciPy 1.17.1 on
        // y = 3 + exp(0.5x) from beta0 = [1, 1] -- and from the model's own estimate -- returns
        // [3.0, 0.5] to rounding.
        let x: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&xi| 3.0 + (0.5_f64 * xi).exp()).collect();
        let output = ODR::new(
            Data::new(x.clone(), y.clone())?,
            exponential(),
            vec![1.0, 1.0],
        )?
        .run()?;
        assert_vec_close(&output.beta, &[3.0, 0.5], 1.0e-8);
        let estimated = ODR::from_model_estimate(Data::new(x, y)?, exponential())?;
        assert_eq!(estimated.beta0, vec![1.0, 1.0]);
        assert_vec_close(&estimated.run()?.beta, &[3.0, 0.5], 1.0e-8);
        Ok(())
    }

    /// frankenscipy-szq1n.12: `fjacb`, `fjacd` and `estimate` were stored and never read, so
    /// every fit ran on finite differences whatever the model supplied. SciPy 1.17.1 fits this
    /// model and data to β = [2.4988338487313224, 1.3001941322482524] with or without the
    /// analytic Jacobians.
    #[test]
    fn analytic_jacobians_are_used_on_both_solver_paths() -> Result<(), OdrError> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let x: Vec<f64> = (0..12).map(|i| 0.1 + 2.9 * f64::from(i) / 11.0).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&v| 2.5 * (1.3 * v).sin() + 0.01 * (7.0 * v).cos())
            .collect();
        let scipy = [2.498_833_848_731_322_4, 1.300_194_132_248_252_4];
        let fcn = |beta: &[f64], x: &[f64]| -> Vec<f64> {
            x.iter().map(|&v| beta[0] * (beta[1] * v).sin()).collect()
        };
        let calls = Arc::new((AtomicUsize::new(0), AtomicUsize::new(0)));
        let (cb, cd) = (Arc::clone(&calls), Arc::clone(&calls));
        let analytic = Model::new(fcn)
            .with_fjacb(move |beta, x| {
                cb.0.fetch_add(1, Ordering::Relaxed);
                x.iter()
                    .map(|&v| vec![(beta[1] * v).sin(), beta[0] * v * (beta[1] * v).cos()])
                    .collect()
            })
            .with_fjacd(move |beta, x| {
                cd.1.fetch_add(1, Ordering::Relaxed);
                x.iter()
                    .map(|&v| vec![beta[0] * beta[1] * (beta[1] * v).cos()])
                    .collect()
            });
        for model in [analytic.clone(), analytic.with_scalar_separable(true)] {
            let path = if model.is_scalar_separable() {
                "structured"
            } else {
                "dense"
            };
            let before = (
                calls.0.load(Ordering::Relaxed),
                calls.1.load(Ordering::Relaxed),
            );
            let fit = ODR::new(Data::new(x.clone(), y.clone())?, model, vec![2.0, 1.0])?.run()?;
            let after = (
                calls.0.load(Ordering::Relaxed),
                calls.1.load(Ordering::Relaxed),
            );
            assert!(
                after.0 > before.0 && after.1 > before.1,
                "{path}: fjacb/fjacd were not called ({before:?} -> {after:?})"
            );
            assert!(fit.success, "{path}: {:?}", fit.stopreason);
            assert_vec_close(&fit.beta, &scipy, 1.0e-6);

            // Finite differences reach the same fit with more model evaluations.
            let fd = ODR::new(
                Data::new(x.clone(), y.clone())?,
                Model::new(fcn),
                vec![2.0, 1.0],
            )?
            .run()?;
            assert_vec_close(&fd.beta, &scipy, 1.0e-6);
            assert!(fit.nfev < fd.nfev, "{path}: {} vs {}", fit.nfev, fd.nfev);
        }
        Ok(())
    }

    #[test]
    fn analytic_jacobian_shape_and_estimate_are_checked() -> Result<(), OdrError> {
        let data = Data::new(vec![1.0, 2.0, 3.0], vec![2.0, 4.1, 5.9])?;
        // fjacb must have one row per observation and one column per parameter.
        let bad = unilinear().with_fjacb(|_beta, x| x.iter().map(|&v| vec![v]).collect());
        let err = ODR::new(data.clone(), bad, vec![1.0, 1.0])?
            .run()
            .expect_err("bad fjacb");
        assert!(matches!(err, OdrError::InvalidArgument { .. }), "{err:?}");
        // SciPy: "must specify beta0 or provide an estimator with the model".
        let no_estimate =
            Model::new(|beta: &[f64], x: &[f64]| x.iter().map(|&v| beta[0] * v).collect());
        assert!(ODR::from_model_estimate(data.clone(), no_estimate).is_err());
        let estimated = ODR::from_model_estimate(data, unilinear())?;
        assert_eq!(estimated.beta0, vec![1.0, 1.0]);
        Ok(())
    }

    #[test]
    fn odr_unilinear_matches_scipy_reference_values() -> Result<(), OdrError> {
        // scipy.odr with unilinear model y = a * x + b
        // Test data: y = 2x + 1
        let x: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let data = Data::new(x, y)?;
        // Initial guess: a=1, b=0
        let odr = ODR::new(data, unilinear(), vec![1.0, 0.0])?;
        let output = odr.run()?;
        // Should fit: beta[0]=a≈2, beta[1]=b≈1
        assert_close(output.beta[0], 2.0, 0.1);
        assert_close(output.beta[1], 1.0, 0.1);
        Ok(())
    }

    /// frankenscipy-vfs3g: once `‖r‖²` (or a column's `‖J_j‖²`) overflowed, every gradient cosine
    /// was `g/∞ = 0` or `∞/∞ = NaN` (dropped by `f64::max`), so the fit reported "gradient
    /// tolerance reached" (info 1) at β0. SciPy 1.17.1, unilinear, β0 = [1, 1]:
    ///
    /// * x = [1..5]·1e160, y = 2x + 1: info 60010, stopreason ["Numerical error detected"],
    ///   beta [2.0000000224401786, 1.0], sum_square inf, res_var 0, sd_beta [0, 0], cov_beta 0.
    /// * x = [1..5], y = [1..5]·1e155 (no NaN anywhere; only `‖r‖²` overflows): info 60000, the
    ///   same stopreason, beta [9.999999999999998e154, 1.4557439427618822e140], sum_square inf,
    ///   res_var 0, sd_beta [0, 0], cov_beta 0.
    ///
    /// ODRPACK skips its covariance step after the error, hence the zeros. fsci stops before its
    /// first step, so beta stays β0 (ODRPACK's partial trust-region iterate is not reproduced),
    /// and it computes no rank digit, so 60010 reads 60000.
    #[test]
    fn odr_overflowed_residual_norm_is_a_numerical_error_like_scipy() -> Result<(), OdrError> {
        let base = [1.0, 2.0, 3.0, 4.0, 5.0];
        let huge_x: Vec<f64> = base.iter().map(|v| v * 1.0e160).collect();
        let huge_x_y = huge_x.iter().map(|v| 2.0 * v + 1.0).collect();
        let huge_y = base.iter().map(|v| v * 1.0e155).collect();
        for (case, x, y) in [
            ("x ~ 1e160", huge_x, huge_x_y),
            ("y ~ 1e155", base.to_vec(), huge_y),
        ] {
            let odr = ODR::new(Data::new(x, y)?, unilinear(), vec![1.0, 1.0])?;
            for (path, out) in [
                ("structured", odr.run()?),
                ("dense", odr.run_dense_reference()?),
            ] {
                assert_eq!(out.info, 60_000, "{case}, {path}: {:?}", out.stopreason);
                assert_eq!(
                    out.stopreason,
                    ["Numerical error detected"],
                    "{case}, {path}"
                );
                assert!(!out.success, "{case}, {path}");
                assert_eq!(out.beta, [1.0, 1.0], "{case}, {path}");
                assert_eq!(out.sum_square, f64::INFINITY, "{case}, {path}");
                assert_eq!(out.res_var, 0.0, "{case}, {path}");
                assert_eq!(out.sd_beta, [0.0, 0.0], "{case}, {path}");
                assert_eq!(out.cov_beta, [[0.0, 0.0], [0.0, 0.0]], "{case}, {path}");
            }
        }

        // Must not change: `‖r‖²` finite. SciPy 1.17.1 converges on this data (info 1, beta
        // [2.00192013931748, 0.014239795736218601]) and on it scaled by 1e150 (info 11: a rank
        // note plus sum-of-squares convergence), never with a numerical error.
        let noisy = [2.1, 3.9, 6.1, 7.9, 10.1];
        let unit = ODR::new(
            Data::new(base.to_vec(), noisy.to_vec())?,
            unilinear(),
            vec![1.0, 0.0],
        )?;
        let large = ODR::new(
            Data::new(
                base.iter().map(|v| v * 1.0e150).collect(),
                noisy.iter().map(|v| v * 1.0e150).collect(),
            )?,
            unilinear(),
            vec![1.0, 0.0],
        )?;
        for (path, out) in [
            ("structured", unit.run()?),
            ("dense", unit.run_dense_reference()?),
        ] {
            assert!(out.success && out.info == 1, "{path}: {out:?}");
            assert_close(out.beta[0], 2.001_920_139_317_48, 1.0e-6);
            assert_close(out.beta[1], 0.014_239_795_736_218_601, 1.0e-5);
        }
        // At 1e150 the intercept column, scaled by β0 as ODRPACK scales it, is 1e-151 of the slope
        // column, so the covariance's relative rank test drops it: SciPy's info 11 and "Problem is
        // not full rank at solution" (frankenscipy-i49i2; fsci reported info 1 before it ported
        // that test).
        for (path, out) in [
            ("structured 1e150", large.run()?),
            ("dense 1e150", large.run_dense_reference()?),
        ] {
            assert!(out.success && out.info == 11, "{path}: {out:?}");
            assert_eq!(
                out.stopreason[0], "Problem is not full rank at solution",
                "{path}"
            );
        }
        Ok(())
    }

    fn assert_rel(actual: f64, expected: f64, rel: f64, context: &str) {
        assert!(
            (actual - expected).abs() <= rel * expected.abs(),
            "{context}: {actual:e} vs {expected:e} (relative tolerance {rel:e})"
        );
    }

    /// `Model(β0·x)` with `parameters` parameters (β1.. unused), scalar-separable so that `run`
    /// takes the structured path and `run_dense_reference` the dense one.
    fn proportional_model(parameters: usize) -> Model {
        Model::new(|b: &[f64], x: &[f64]| x.iter().map(|&v| b[0] * v).collect())
            .with_parameter_count(parameters)
            .with_scalar_separable(true)
    }

    /// frankenscipy-i49i2: the covariance inverted JᵀJ behind an ABSOLUTE 1e-14 pivot test, so a
    /// Jacobian of order 1e-100 lost it (cov_beta NaN, sd_beta clamped to 0). ODRPACK rank-tests
    /// the β-scaled R relative to its own norm, and SciPy reports the same sd_beta at any scale.
    /// SciPy 1.17.1, `Model(β0·x)`, OLS, x = [1..5]·s, y = [2.1, 3.9, 6.1, 7.9, 10.1]·s, from the
    /// optimum β0 = 110.3/55 (so fsci's solver takes no step at either scale):
    ///
    /// * s = 1e-100: sd_beta [0.01482682366364836], res_var 1.2090909090909083e-202,
    ///   cov_beta·res_var [[0.00021983469995292298]], inv_condnum 1.0.
    /// * s = 1, must not change: sd_beta [0.014826824266328165], res_var 0.012090909090909053,
    ///   cov_beta·res_var [[0.0002198347178245777]].
    ///
    /// And the bead's exact fit, y = 2x, β0 = [2], s = 1e-100: cov_beta [[1.818182690913911e198]]
    /// with res_var 0, so cov_beta·res_var [[0.0]] (fsci had NaN) and sd_beta [0.0].
    #[test]
    fn odr_covariance_rank_test_is_scale_relative_like_scipy() -> Result<(), OdrError> {
        let base = [1.0, 2.0, 3.0, 4.0, 5.0];
        let noisy = [2.1, 3.9, 6.1, 7.9, 10.1];
        for (scale, sd, res_var, cov) in [
            (
                1.0e-100,
                0.014_826_823_663_648_36,
                1.209_090_909_090_908_3e-202,
                0.000_219_834_699_952_922_98,
            ),
            (
                1.0,
                0.014_826_824_266_328_165,
                0.012_090_909_090_909_053,
                0.000_219_834_717_824_577_7,
            ),
        ] {
            let mut odr = ODR::new(
                Data::new(
                    base.iter().map(|v| v * scale).collect(),
                    noisy.iter().map(|v| v * scale).collect(),
                )?,
                proportional_model(1),
                vec![110.3 / 55.0],
            )?;
            odr.set_job(FitType::Ols);
            for (path, out) in [
                ("structured", odr.run()?),
                ("dense", odr.run_dense_reference()?),
            ] {
                let context = format!("s = {scale:e}, {path}: {out:?}");
                assert_rel(out.sd_beta[0], sd, 1.0e-7, &context);
                assert_rel(out.res_var, res_var, 1.0e-12, &context);
                assert_rel(out.cov_beta[0][0], cov, 1.0e-7, &context);
                assert_rel(out.inv_condnum, 1.0, 1.0e-12, &context);
            }
        }

        let x: Vec<f64> = base.iter().map(|v| v * 1.0e-100).collect();
        let y = x.iter().map(|v| 2.0 * v).collect();
        let mut exact = ODR::new(Data::new(x, y)?, proportional_model(1), vec![2.0])?;
        exact.set_job(FitType::Ols);
        for (path, out) in [
            ("structured", exact.run()?),
            ("dense", exact.run_dense_reference()?),
        ] {
            assert_eq!(out.res_var, 0.0, "exact 1e-100, {path}");
            assert_eq!(out.cov_beta, [[0.0]], "exact 1e-100, {path}");
            assert_eq!(out.sd_beta, [0.0], "exact 1e-100, {path}");
        }
        Ok(())
    }

    /// frankenscipy-i49i2: a model that ignores β1 has an exactly zero Jacobian column. fsci
    /// inverted the singular JᵀJ (NaN everywhere, sd_beta clamped to [0, 0]); ODRPACK drops the
    /// column from R (rank deficiency 1), reports the rest, and counts the degrees of freedom as
    /// observations with a nonzero weighted derivative minus the rank. SciPy 1.17.1,
    /// `Model(β0·x)`, x = [0, 1, 2, 3], y = 2x + 0.1·[1, −1, 1, −1], β0 = [1, 1]:
    ///
    /// * ODR: sd_beta [0.02975082061988635, 0.0], res_var 0.0025036415077379776 = sum_square/3,
    ///   cov_beta·res_var [[0.0008851113275566549, 0], [0, 0]], info 11, stopreason
    ///   ["Problem is not full rank at solution", "Sum of squares convergence"].
    /// * OLS: sd_beta [0.03642156969967579, 0.0], res_var 0.018571428571428513 (x = 0 has no
    ///   derivative, so 3 − 1 degrees of freedom), info 11.
    ///
    /// A fixed parameter leaves the count the same way: unilinear with β0 fixed at 3, x = [0..3],
    /// y = [1.1, 3.9, 7.2, 9.9], β0 = [3, 0]: SciPy res_var 0.0022500000000339484 (sum_square/3;
    /// fsci divided by n − 2) and sd_beta [0.0, 0.0749997788384343].
    ///
    /// Must not change, a full-rank fit: unilinear, x = [1..5], y = [2.1, 3.9, 6.1, 7.9, 10.1],
    /// β0 = [1, 0], info 1 and SciPy's sd_beta with the analytic Jacobian fsci's unilinear uses,
    /// [0.04003073470991557, 0.1327576599279166].
    #[test]
    fn odr_rank_deficient_parameter_is_dropped_like_scipy() -> Result<(), OdrError> {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y: Vec<f64> = x
            .iter()
            .zip([1.0, -1.0, 1.0, -1.0])
            .map(|(v, sign)| 2.0 * v + 0.1 * sign)
            .collect();
        for (fit_type, sd, res_var, cov) in [
            (
                FitType::Odr,
                0.029_750_820_619_886_35,
                0.002_503_641_507_737_977_6,
                0.000_885_111_327_556_654_9,
            ),
            (
                FitType::Ols,
                0.036_421_569_699_675_79,
                0.018_571_428_571_428_513,
                0.001_326_530_739_388_341_6,
            ),
        ] {
            let mut odr = ODR::new(
                Data::new(x.clone(), y.clone())?,
                proportional_model(2),
                vec![1.0, 1.0],
            )?;
            odr.set_job(fit_type);
            for (path, out) in [
                ("structured", odr.run()?),
                ("dense", odr.run_dense_reference()?),
            ] {
                let context = format!("{fit_type:?}, {path}: {out:?}");
                assert_rel(out.sd_beta[0], sd, 1.0e-6, &context);
                assert_eq!(out.sd_beta[1].to_bits(), 0.0_f64.to_bits(), "{context}");
                assert_rel(out.res_var, res_var, 1.0e-12, &context);
                assert_rel(out.cov_beta[0][0], cov, 1.0e-6, &context);
                assert_eq!(
                    [out.cov_beta[0][1], out.cov_beta[1][0], out.cov_beta[1][1]],
                    [0.0; 3],
                    "{context}"
                );
                assert_eq!(out.info, 11, "{context}");
                assert_eq!(
                    out.stopreason[0], "Problem is not full rank at solution",
                    "{context}"
                );
            }
        }

        let fixed = ODR::new(
            Data::new(x.clone(), vec![1.1, 3.9, 7.2, 9.9])?,
            unilinear(),
            vec![3.0, 0.0],
        )?
        .with_beta_free(vec![false, true])?;
        for (path, out) in [
            ("structured", fixed.run()?),
            ("dense", fixed.run_dense_reference()?),
        ] {
            let context = format!("fixed β0, {path}: {out:?}");
            assert_rel(out.res_var, 0.002_250_000_000_033_948_4, 1.0e-9, &context);
            assert_eq!(out.sd_beta[0], 0.0, "{context}");
            assert_rel(out.sd_beta[1], 0.074_999_778_838_434_3, 1.0e-5, &context);
            assert_eq!(out.info, 1, "{context}");
        }

        let full_rank = ODR::new(
            Data::new(
                vec![1.0, 2.0, 3.0, 4.0, 5.0],
                vec![2.1, 3.9, 6.1, 7.9, 10.1],
            )?,
            unilinear(),
            vec![1.0, 0.0],
        )?;
        for (path, out) in [
            ("structured", full_rank.run()?),
            ("dense", full_rank.run_dense_reference()?),
        ] {
            let context = format!("full rank, {path}: {out:?}");
            assert_eq!(out.info, 1, "{context}");
            assert_eq!(out.stopreason.len(), 1, "{context}");
            assert_rel(out.sd_beta[0], 0.040_030_734_709_915_57, 1.0e-8, &context);
            assert_rel(out.sd_beta[1], 0.132_757_659_927_916_6, 1.0e-8, &context);
        }
        Ok(())
    }

    /// frankenscipy-i49i2: sd_beta was `max(var, 0).sqrt()`, which turned a NaN variance into 0.
    /// SciPy 1.17.1, `Model(β0·x)`, OLS, y = 2x exactly, β0 = [2], x = [1..5]·s:
    ///
    /// * s = 1e-160: (RᵀR)⁻¹ overflows, cov_beta [[inf]] with res_var 0, so sd_beta [nan] and
    ///   cov_beta·res_var [[nan]]; inv_condnum 1.0.
    /// * s = 1e-100: cov_beta [[1.818182690913911e198]], sd_beta [0.0].
    /// * s = 1, must not change: cov_beta [[0.018181818181818184]], sd_beta [0.0].
    ///
    /// And the rank-deficient fit of the test above keeps sd_beta[1] = 0.0, where keeping the NaN
    /// alone gave [NaN, NaN]. That change was tried and backed out in vfs3g because the absolute
    /// pivot then also turned s = 1e-100 into NaN; with the relative rank test all four match.
    #[test]
    fn odr_sd_beta_keeps_a_nan_variance_like_scipy() -> Result<(), OdrError> {
        let base = [1.0, 2.0, 3.0, 4.0, 5.0];
        for scale in [1.0e-160, 1.0e-100, 1.0] {
            let x: Vec<f64> = base.iter().map(|v| v * scale).collect();
            let y = x.iter().map(|v| 2.0 * v).collect();
            let mut odr = ODR::new(Data::new(x, y)?, proportional_model(1), vec![2.0])?;
            odr.set_job(FitType::Ols);
            for (path, out) in [
                ("structured", odr.run()?),
                ("dense", odr.run_dense_reference()?),
            ] {
                let context = format!("s = {scale:e}, {path}: {out:?}");
                assert_eq!(out.res_var, 0.0, "{context}");
                if scale == 1.0e-160 {
                    assert!(out.sd_beta[0].is_nan(), "{context}");
                    assert!(out.cov_beta[0][0].is_nan(), "{context}");
                    assert_rel(out.inv_condnum, 1.0, 1.0e-12, &context);
                } else {
                    assert_eq!(out.sd_beta, [0.0], "{context}");
                    assert_eq!(out.cov_beta, [[0.0]], "{context}");
                }
            }
        }

        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.1, 1.9, 4.1, 5.9];
        let odr = ODR::new(Data::new(x, y)?, proportional_model(2), vec![1.0, 1.0])?;
        for (path, out) in [
            ("structured", odr.run()?),
            ("dense", odr.run_dense_reference()?),
        ] {
            assert!(
                out.sd_beta[0].is_finite() && out.sd_beta[0] > 0.0,
                "{path}: {out:?}"
            );
            assert_eq!(out.sd_beta[1], 0.0, "{path}: {out:?}");
        }
        Ok(())
    }

    /// frankenscipy-i49i2: Output.eps was observed − fitted. SciPy's `eps` is ODRPACK's
    /// F = f(x + δ; β) − y, fitted minus observed, and `delta` is `xplus − x`. SciPy 1.17.1,
    /// unilinear, x = [1..5], y = [2.1, 3.9, 6.1, 7.9, 10.1], β0 = [1, 0], with the analytic
    /// Jacobian fsci's unilinear uses: eps[0] = −0.016742455718879334 (fitted 2.08326 against
    /// 2.1), delta[0] = 0.03351704770275878. With finite differences SciPy has eps[0] =
    /// −0.016742296199194318. Must not change: beta and sum_square_eps
    /// (SciPy 0.0019155833882863924).
    #[test]
    fn odr_eps_is_fitted_minus_observed_like_scipy() -> Result<(), OdrError> {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.1, 3.9, 6.1, 7.9, 10.1];
        let odr = ODR::new(
            Data::new(x.clone(), y.clone())?,
            unilinear(),
            vec![1.0, 0.0],
        )?;
        for (path, out) in [
            ("structured", odr.run()?),
            ("dense", odr.run_dense_reference()?),
        ] {
            let context = format!("{path}: {out:?}");
            assert_rel(out.eps[0], -0.016_742_455_718_879_334, 1.0e-6, &context);
            assert_rel(out.delta[0], 0.033_517_047_702_758_78, 1.0e-6, &context);
            for i in 0..y.len() {
                assert_eq!(out.eps[i], out.y[i] - y[i], "{context}");
                assert_eq!(out.xplus[i], x[i] + out.delta[i], "{context}");
            }
            assert_rel(out.beta[0], 2.001_920_459_630_725_6, 1.0e-8, &context);
            assert_rel(out.beta[1], 0.014_238_621_107_823_236, 1.0e-5, &context);
            assert_rel(
                out.sum_square_eps,
                0.001_915_583_388_286_392_4,
                1.0e-6,
                &context,
            );
        }
        Ok(())
    }
}
