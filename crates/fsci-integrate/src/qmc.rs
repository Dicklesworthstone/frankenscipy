//! Randomized quasi-Monte Carlo integration with reproducible Halton replicates.
//!
//! Each replicate uses independently drawn permutations for every base digit,
//! including the trailing zero digits. This is the digit-permutation Halton
//! randomization described by Owen, "A randomized Halton algorithm in R"
//! (2017), https://arxiv.org/abs/1706.02808. The standard error comes from the
//! variation between randomized integral estimates, not consecutive blocks
//! of one deterministic sequence.

use crate::validation::IntegrateValidationError;

/// The integral estimate and the estimated standard error of its mean.
///
/// The standard error uses the sample variance of independent randomized
/// replicates, divided by their count. It is an uncertainty estimate, not a
/// deterministic bound on the integration error. With only one replicate it
/// is NaN because the between-replicate variance cannot be estimated.
/// A domain with any zero-width axis has exact integral and standard error zero.
#[derive(Debug, Clone, PartialEq)]
pub struct QmcQuadResult {
    pub integral: f64,
    pub standard_error: f64,
}

/// Sampling and reproducibility controls for [qmc_quad_with_options].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QmcQuadOptions {
    /// Number of independently scrambled estimates; defaults to 8.
    /// One is allowed, but its standard error is NaN.
    pub n_estimates: usize,
    /// Points in each estimate; defaults to 1024.
    pub n_points: usize,
    /// Reproducible scrambling seed; defaults to zero.
    ///
    /// The permutations use SplitMix64. This seed does not select the same
    /// point set as a NumPy/SciPy random generator with the same integer seed.
    pub seed: u64,
    /// Maximum replicate workers. None chooses automatically; Some(1) is serial.
    /// Some(0) is invalid. Changing this never changes samples or reduction order.
    pub workers: Option<usize>,
}

impl Default for QmcQuadOptions {
    fn default() -> Self {
        Self {
            n_estimates: 8,
            n_points: 1024,
            seed: 0,
            workers: None,
        }
    }
}

/// Integrate a scalar function on a finite hyper-rectangle with randomized Halton points.
///
/// This retains the original five-argument interface and uses the reproducible
/// default seed. Use [qmc_quad_with_options] to select a seed or worker count.
/// Bounds may be reversed; only the integral changes sign. At most 32 dimensions
/// are supported. Exactly n_estimates * n_points calls are made unless a
/// zero-width domain or an error ends the computation.
///
/// # Errors
/// Invalid dimensions, bounds, counts or worker settings; non-finite integrand
/// values; resource exhaustion; or an integral/standard error outside f64 range.
pub fn qmc_quad<F>(
    f: F,
    lb: &[f64],
    ub: &[f64],
    n_estimates: usize,
    n_points: usize,
) -> Result<QmcQuadResult, IntegrateValidationError>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    qmc_quad_with_options(
        f,
        lb,
        ub,
        QmcQuadOptions {
            n_estimates,
            n_points,
            ..QmcQuadOptions::default()
        },
    )
}

/// Integrate with independently scrambled, reproducibly seeded Halton replicates.
///
/// Every replicate starts at index zero with a fresh set of random digit
/// permutations. Its points are correlated within that replicate, as intended
/// for QMC; separate replicates provide the variance estimate. Scrambling also
/// permutes trailing zero digits, so index zero is randomized rather than fixed
/// at the corner of the integration domain.
///
/// The integral is the mean of replicate estimates. For R > 1, standard_error
/// is sqrt(sum((estimate_r - mean)^2) / (R * (R - 1))). Compensated replicate
/// sums stay in two components through centering; their common point-count
/// denominator is applied afterwards. Binary exponents are retained through
/// reduction and volume scaling, preserving small differences between large
/// sums and finite results for mixed tiny/huge widths.
///
/// The same options and integrand produce the same results for any worker
/// count. The callback should depend only on its point; its execution order
/// across different replicates is unspecified when workers are enabled.
///
/// # Errors
/// As for [qmc_quad]. A single replicate returns an estimate with NaN
/// standard_error; it does not claim zero uncertainty.
pub fn qmc_quad_with_options<F>(
    f: F,
    lb: &[f64],
    ub: &[f64],
    options: QmcQuadOptions,
) -> Result<QmcQuadResult, IntegrateValidationError>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    let domain = QmcDomain::new(lb, ub, options)?;
    if domain.volume.mantissa == 0.0 {
        return Ok(QmcQuadResult {
            integral: 0.0,
            standard_error: 0.0,
        });
    }

    let mut estimates = Vec::new();
    estimates
        .try_reserve_exact(options.n_estimates)
        .map_err(|error| IntegrateValidationError::ResourceExhausted {
            message: format!("qmc_quad: cannot allocate replicate estimates: {error}"),
        })?;
    estimates.resize_with(options.n_estimates, || Ok(ScaledSum::default()));

    let estimate = |replicate: usize| -> Result<ScaledSum, IntegrateValidationError> {
        let scramble = HaltonScramble::new(lb.len(), options.seed, replicate);
        let mut point = vec![0.0; lb.len()];
        let mut sum = ScaledSum::default();
        for index in 0..options.n_points {
            domain.point(&scramble, index, &mut point);
            let value = checked_value(f(&point), replicate, index)?;
            sum.add(Scaled::from_f64(value));
        }
        // Keep both components and the common point-count denominator separate.
        // Collapsing here could erase a small remainder before another replicate
        // cancels its large part, or erase sub-ulp variation between large means.
        Ok(sum)
    };

    let workers = match options.workers {
        Some(workers) => workers.min(options.n_estimates),
        None if domain.work < (1 << 16) => 1,
        None => std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .min(16)
            .min(options.n_estimates),
    };
    if workers == 1 {
        for (replicate, slot) in estimates.iter_mut().enumerate() {
            *slot = estimate(replicate);
            if slot.is_err() {
                break;
            }
        }
    } else {
        let chunk = options.n_estimates.div_ceil(workers);
        let estimate = &estimate;
        std::thread::scope(|scope| -> Result<(), IntegrateValidationError> {
            for (worker, slots) in estimates.chunks_mut(chunk).enumerate() {
                let first = worker * chunk;
                std::thread::Builder::new()
                    .spawn_scoped(scope, move || {
                        for (index, slot) in slots.iter_mut().enumerate() {
                            *slot = estimate(first + index);
                            if slot.is_err() {
                                break;
                            }
                        }
                    })
                    .map_err(|error| IntegrateValidationError::ResourceExhausted {
                        message: format!("qmc_quad: cannot start replicate worker: {error}"),
                    })?;
            }
            Ok(())
        })?;
    }

    let count = Scaled::from_f64(options.n_estimates as f64);
    let points = Scaled::from_f64(options.n_points as f64);
    let mut sum = ScaledSum::default();
    for estimate in &estimates {
        sum.merge(estimate.as_ref().map_err(Clone::clone)?);
    }
    let integral = finite_output(
        sum.total().div(count).div(points).mul(domain.volume),
        "integral",
    )?;
    let standard_error = if options.n_estimates == 1 {
        f64::NAN
    } else {
        // Center on a complete replicate sum, not an individual function value.
        // Large components cancel before the small differences are rounded.
        let origin = estimates[0].as_ref().map_err(Clone::clone)?;
        let mut deviations = ScaledSum::default();
        for estimate in &estimates {
            deviations.add(estimate.as_ref().map_err(Clone::clone)?.difference(origin));
        }
        let mean_deviation = deviations.total().div(count);
        let mut squares = ScaledSum::default();
        for estimate in &estimates {
            let difference = estimate
                .as_ref()
                .map_err(Clone::clone)?
                .difference(origin)
                .sub(mean_deviation);
            squares.add(difference.mul(difference));
        }
        let variance_of_sum_mean = squares
            .total()
            .div(count)
            .div(Scaled::from_f64((options.n_estimates - 1) as f64));
        finite_output(
            variance_of_sum_mean
                .sqrt()
                .div(points)
                .mul(domain.volume.abs()),
            "standard error",
        )?
    };
    Ok(QmcQuadResult {
        integral,
        standard_error,
    })
}

const PRIMES: [usize; 32] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131,
];

struct QmcDomain {
    lower: Vec<f64>,
    upper: Vec<f64>,
    volume: Scaled,
    work: usize,
}

impl QmcDomain {
    fn new(
        lb: &[f64],
        ub: &[f64],
        options: QmcQuadOptions,
    ) -> Result<Self, IntegrateValidationError> {
        if lb.len() != ub.len() || lb.is_empty() || lb.len() > PRIMES.len() {
            return Err(IntegrateValidationError::QuadInvalidBounds {
                detail: "qmc_quad: bounds must have the same length in 1..=32".to_string(),
            });
        }
        if options.n_estimates == 0 || options.n_points == 0 || options.workers == Some(0) {
            return Err(IntegrateValidationError::QuadInvalidTolerance {
                detail: "qmc_quad: estimate, point and explicit worker counts must be positive"
                    .to_string(),
            });
        }
        let work = options
            .n_estimates
            .checked_mul(options.n_points)
            .and_then(|count| count.checked_mul(lb.len()))
            .ok_or_else(|| IntegrateValidationError::ResourceExhausted {
                message: "qmc_quad: requested sample coordinate count overflows usize".to_string(),
            })?;
        if options.n_points as u128 > (1_u128 << 53) {
            return Err(IntegrateValidationError::ResourceExhausted {
                message: "qmc_quad: more than 2^53 points per replicate exceeds f64 resolution"
                    .to_string(),
            });
        }
        let mut lower = Vec::with_capacity(lb.len());
        let mut upper = Vec::with_capacity(lb.len());
        let mut volume = Scaled::from_f64(1.0);
        for (&left, &right) in lb.iter().zip(ub) {
            if !left.is_finite() || !right.is_finite() {
                return Err(IntegrateValidationError::QuadInvalidBounds {
                    detail: "qmc_quad: bounds must be finite".to_string(),
                });
            }
            lower.push(left.min(right));
            upper.push(left.max(right));
            volume = volume.mul(Scaled::from_f64(right).sub(Scaled::from_f64(left)));
        }
        Ok(Self {
            lower,
            upper,
            volume,
            work,
        })
    }

    fn point(&self, scramble: &HaltonScramble, index: usize, point: &mut [f64]) {
        scramble.point(index, point);
        for ((value, &lower), &upper) in point.iter_mut().zip(&self.lower).zip(&self.upper) {
            let u = *value;
            let mapped = if lower < 0.0 && upper > 0.0 {
                // The width may exceed f64 even though both weighted terms fit.
                (1.0 - u) * lower + u * upper
            } else {
                lower + u * (upper - lower)
            };
            // Rounding a convex interpolation must never leave its finite bounds.
            *value = mapped.clamp(lower, upper);
        }
    }
}

fn checked_value(
    value: f64,
    replicate: usize,
    point: usize,
) -> Result<f64, IntegrateValidationError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(IntegrateValidationError::IntegrationFailed {
            message: format!(
                "qmc_quad: non-finite integrand value at replicate {replicate}, point {point}"
            ),
        })
    }
}

fn finite_output(value: Scaled, name: &str) -> Result<f64, IntegrateValidationError> {
    value
        .to_f64()
        .ok_or_else(|| IntegrateValidationError::IntegrationFailed {
            message: format!("qmc_quad: {name} exceeds the finite f64 range"),
        })
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

struct ScrambleRng(u64);

impl ScrambleRng {
    fn new(seed: u64, replicate: usize) -> Self {
        // Distinct replicate keys get distinct mixed starting states. Each
        // replicate consumes only its own stream, independent of scheduling.
        Self(mix64(seed.wrapping_add(mix64(
            (replicate as u64) ^ 0xd1b5_4a32_d192_ed03,
        ))))
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix64(self.0)
    }

    fn below(&mut self, upper: usize) -> usize {
        let upper = upper as u64;
        let threshold = upper.wrapping_neg() % upper;
        loop {
            let value = self.next();
            if value >= threshold {
                return (value % upper) as usize;
            }
        }
    }
}

struct ScrambledCoordinate {
    base: usize,
    permutations: Vec<Vec<usize>>,
    weights: Vec<f64>,
    zero_suffix: Vec<f64>,
}

struct HaltonScramble {
    coordinates: Vec<ScrambledCoordinate>,
}

impl HaltonScramble {
    fn new(dimension: usize, seed: u64, replicate: usize) -> Self {
        let mut rng = ScrambleRng::new(seed, replicate);
        let mut coordinates = Vec::with_capacity(dimension);
        for &base in &PRIMES[..dimension] {
            let mut permutations = Vec::new();
            let mut weights = Vec::new();
            let mut weight = 1.0 / base as f64;
            loop {
                let mut permutation: Vec<usize> = (0..base).collect();
                for index in (1..base).rev() {
                    let other = rng.below(index + 1);
                    permutation.swap(index, other);
                }
                permutations.push(permutation);
                weights.push(weight);
                // The unrepresented tail is at most base^-digits. Include
                // enough *zero* input digits to make that tail <= 2^-54.
                if weight <= f64::EPSILON * 0.25 {
                    break;
                }
                weight /= base as f64;
            }
            let mut zero_suffix = vec![0.0; permutations.len() + 1];
            for digit in (0..permutations.len()).rev() {
                zero_suffix[digit] =
                    permutations[digit][0] as f64 * weights[digit] + zero_suffix[digit + 1];
            }
            coordinates.push(ScrambledCoordinate {
                base,
                permutations,
                weights,
                zero_suffix,
            });
        }
        Self { coordinates }
    }

    fn point(&self, index: usize, out: &mut [f64]) {
        for (coordinate, slot) in self.coordinates.iter().zip(out) {
            let mut digits = index;
            let mut value = 0.0;
            let mut digit = 0;
            while digits > 0 && digit < coordinate.permutations.len() {
                value += coordinate.permutations[digit][digits % coordinate.base] as f64
                    * coordinate.weights[digit];
                digits /= coordinate.base;
                digit += 1;
            }
            // All remaining input digits are zero. Their scrambled contribution
            // is fixed within this replicate, so it was summed once at setup.
            value += coordinate.zero_suffix[digit];
            // A finite digit sum can round upward to one.
            *slot = value.min(1.0_f64.next_down());
        }
    }
}

/// A finite floating-point significand with an unbounded-for-this-problem
/// binary exponent. All operations below have bounded finite intermediates.
#[derive(Clone, Copy, Default)]
struct Scaled {
    mantissa: f64,
    exponent: i64,
}

impl Scaled {
    const SIGN_AND_FRACTION: u64 = 0x800f_ffff_ffff_ffff;

    fn from_f64(value: f64) -> Self {
        debug_assert!(value.is_finite());
        if value == 0.0 {
            return Self {
                mantissa: value,
                exponent: 0,
            };
        }
        let mut bits = value.to_bits();
        let mut exponent = ((bits >> 52) & 0x7ff) as i64 - 1023;
        if exponent == -1023 {
            bits = (value * 4_503_599_627_370_496.0).to_bits();
            exponent = ((bits >> 52) & 0x7ff) as i64 - 1023 - 52;
        }
        Self {
            mantissa: f64::from_bits((bits & Self::SIGN_AND_FRACTION) | (1023_u64 << 52)),
            exponent,
        }
    }

    fn normalized(mantissa: f64, exponent: i64) -> Self {
        let mut value = Self::from_f64(mantissa);
        if value.mantissa != 0.0 {
            value.exponent += exponent;
        }
        value
    }

    fn add(self, other: Self) -> Self {
        if self.mantissa == 0.0 {
            return other;
        }
        if other.mantissa == 0.0 {
            return self;
        }
        let (large, small) = if self.exponent >= other.exponent {
            (self, other)
        } else {
            (other, self)
        };
        let difference = large.exponent - small.exponent;
        if difference > 54 {
            return large;
        }
        let ratio = f64::from_bits(((1023 - difference) as u64) << 52);
        Self::normalized(large.mantissa + small.mantissa * ratio, large.exponent)
    }

    fn sub(self, other: Self) -> Self {
        self.add(Self {
            mantissa: -other.mantissa,
            exponent: other.exponent,
        })
    }

    fn mul(self, other: Self) -> Self {
        Self::normalized(
            self.mantissa * other.mantissa,
            self.exponent + other.exponent,
        )
    }

    fn div(self, other: Self) -> Self {
        Self::normalized(
            self.mantissa / other.mantissa,
            self.exponent - other.exponent,
        )
    }

    fn abs(self) -> Self {
        Self {
            mantissa: self.mantissa.abs(),
            exponent: self.exponent,
        }
    }

    fn abs_le(self, other: Self) -> bool {
        if self.mantissa == 0.0 {
            return true;
        }
        other.mantissa != 0.0
            && (self.exponent < other.exponent
                || (self.exponent == other.exponent && self.mantissa.abs() <= other.mantissa.abs()))
    }

    fn sqrt(self) -> Self {
        debug_assert!(self.mantissa >= 0.0);
        let mantissa = if self.exponent.rem_euclid(2) == 0 {
            self.mantissa
        } else {
            2.0 * self.mantissa
        };
        Self::normalized(mantissa.sqrt(), self.exponent.div_euclid(2))
    }

    fn to_f64(self) -> Option<f64> {
        if self.mantissa == 0.0 {
            return Some(self.mantissa);
        }
        if self.exponent > 1023 {
            return None;
        }
        if self.exponent >= -1022 {
            return Some(f64::from_bits(
                (self.mantissa.to_bits() & Self::SIGN_AND_FRACTION)
                    | (((self.exponent + 1023) as u64) << 52),
            ));
        }
        if self.exponent < -1075 {
            return Some(0.0_f64.copysign(self.mantissa));
        }
        let scale = f64::from_bits(((self.exponent + 2045) as u64) << 52);
        Some((self.mantissa * f64::MIN_POSITIVE) * scale)
    }
}

/// Neumaier summation after exponent alignment. The two components remain
/// separate until large parts cancel across replicates or during centering.
#[derive(Clone, Copy, Default)]
struct ScaledSum {
    sum: Scaled,
    correction: Scaled,
}

impl ScaledSum {
    fn add(&mut self, value: Scaled) {
        let total = self.sum.add(value);
        let remainder = if value.abs_le(self.sum) {
            self.sum.sub(total).add(value)
        } else {
            value.sub(total).add(self.sum)
        };
        self.correction = self.correction.add(remainder);
        self.sum = total;
    }

    fn merge(&mut self, other: &Self) {
        self.add(other.sum);
        self.add(other.correction);
    }

    fn difference(&self, other: &Self) -> Scaled {
        let mut difference = Self::default();
        difference.add(self.sum);
        difference.add(Scaled {
            mantissa: -other.sum.mantissa,
            exponent: other.sum.exponent,
        });
        difference.add(self.correction);
        difference.add(Scaled {
            mantissa: -other.correction.mantissa,
            exponent: other.correction.exponent,
        });
        difference.total()
    }

    fn total(&self) -> Scaled {
        self.sum.add(self.correction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn small_options() -> QmcQuadOptions {
        QmcQuadOptions {
            n_estimates: 4,
            n_points: 16,
            workers: Some(1),
            ..QmcQuadOptions::default()
        }
    }

    fn close_relative(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual / expected - 1.0).abs() <= tolerance,
            "actual={actual:e}, expected={expected:e}"
        );
    }

    #[test]
    fn qmc_seed_is_reproducible_across_worker_counts() {
        let f = |point: &[f64]| point[0].exp() * point[1].cos() + point[2] * point[2];
        let options = QmcQuadOptions {
            n_estimates: 7,
            n_points: 257,
            seed: 0x1234_5678,
            workers: Some(1),
        };
        let serial = qmc_quad_with_options(f, &[0.0; 3], &[1.0; 3], options).unwrap();
        for workers in [Some(1), Some(2), Some(4), None] {
            let result = qmc_quad_with_options(
                f,
                &[0.0; 3],
                &[1.0; 3],
                QmcQuadOptions { workers, ..options },
            )
            .unwrap();
            assert_eq!(serial.integral.to_bits(), result.integral.to_bits());
            assert_eq!(
                serial.standard_error.to_bits(),
                result.standard_error.to_bits()
            );
        }
        let different = qmc_quad_with_options(
            f,
            &[0.0; 3],
            &[1.0; 3],
            QmcQuadOptions {
                seed: options.seed + 1,
                ..options
            },
        )
        .unwrap();
        assert_ne!(serial, different);
        let legacy = qmc_quad(f, &[0.0; 3], &[1.0; 3], 7, 257).unwrap();
        let explicit_default_seed = qmc_quad_with_options(
            f,
            &[0.0; 3],
            &[1.0; 3],
            QmcQuadOptions { seed: 0, ..options },
        )
        .unwrap();
        assert_eq!(legacy, explicit_default_seed);
    }

    #[test]
    fn qmc_zero_index_is_randomized_independently_in_every_replicate() {
        let sampled = Mutex::new(Vec::new());
        let result = qmc_quad_with_options(
            |point| {
                sampled.lock().unwrap().push(point.to_vec());
                point[0] + point[1]
            },
            &[0.0; 2],
            &[1.0; 2],
            QmcQuadOptions {
                n_estimates: 8,
                n_points: 1,
                ..small_options()
            },
        )
        .unwrap();
        let sampled = sampled.into_inner().unwrap();
        assert_eq!(sampled.len(), 8);
        for (i, point) in sampled.iter().enumerate() {
            assert!(point.iter().all(|&x| x > 0.0 && x < 1.0));
            assert_ne!(point[0], point[1]);
            assert!(sampled[..i].iter().all(|previous| previous != point));
        }
        assert!(result.standard_error > 0.0);
    }

    #[test]
    fn qmc_scrambling_keeps_prime_power_stratification() {
        for replicate in 0..4 {
            let sampler = HaltonScramble::new(3, 42, replicate);
            for (dimension, points) in [(0, 64), (1, 81), (2, 125)] {
                let mut counts = vec![0; points];
                for index in 0..points {
                    let mut point = [0.0; 3];
                    sampler.point(index, &mut point);
                    let cell = (point[dimension] * points as f64).floor() as usize;
                    assert!(cell < points);
                    counts[cell] += 1;
                }
                assert!(counts.iter().all(|&count| count == 1));
            }
        }
    }

    #[test]
    fn qmc_zero_digit_suffix_matches_full_digit_scrambling_to_roundoff() {
        for seed in [0, 1, u64::MAX] {
            let sampler = HaltonScramble::new(32, seed, 3);
            for index in [0, 1, 2, 7, 64, 1023, 65_535, 1 << 20] {
                let mut point = [0.0; 32];
                sampler.point(index, &mut point);
                for (coordinate, &actual) in sampler.coordinates.iter().zip(&point) {
                    let mut digits = index;
                    let mut reference = 0.0;
                    for (permutation, &weight) in
                        coordinate.permutations.iter().zip(&coordinate.weights)
                    {
                        reference += permutation[digits % coordinate.base] as f64 * weight;
                        digits /= coordinate.base;
                    }
                    reference = reference.min(1.0_f64.next_down());
                    assert!(
                        actual.to_bits().abs_diff(reference.to_bits()) <= 16,
                        "suffix changed a scrambled point beyond summation roundoff"
                    );
                }
            }
        }
    }

    #[test]
    fn qmc_randomized_replicates_expose_deterministic_aliasing() {
        // Every old index 1..=127 aliases to +1 at frequency 128; index 128
        // gives -1. The old result was 0.984375 +/- 0.015625 for an exact
        // integral of zero, more than 60 reported standard errors away.
        let result = qmc_quad_with_options(
            |point| (std::f64::consts::TAU * 128.0 * point[0]).cos(),
            &[0.0],
            &[1.0],
            QmcQuadOptions {
                n_estimates: 8,
                n_points: 16,
                ..small_options()
            },
        )
        .unwrap();
        assert!(result.standard_error > 0.1);
        assert!(result.integral.abs() < 4.0 * result.standard_error);
    }

    #[test]
    fn qmc_reported_variance_tracks_actual_error_over_fixed_seed_ensemble() {
        let mut squared_error = 0.0;
        let mut reported_variance = 0.0;
        let mut mean = 0.0;
        for seed in 0..128 {
            let result = qmc_quad_with_options(
                |point| (std::f64::consts::TAU * 128.0 * point[0]).cos(),
                &[0.0],
                &[1.0],
                QmcQuadOptions {
                    n_estimates: 8,
                    n_points: 16,
                    seed,
                    ..small_options()
                },
            )
            .unwrap();
            mean += result.integral;
            squared_error += result.integral * result.integral;
            reported_variance += result.standard_error * result.standard_error;
        }
        // A deterministic ensemble of seeds checks the uncertainty, not just
        // integration accuracy at one favorable randomization. For this
        // integrand each replicate has variance 1/2, so the mean has 1/16.
        let ratio = reported_variance / squared_error;
        assert!((0.5..1.5).contains(&ratio), "variance ratio={ratio}");
        assert!((mean / 128.0).abs() < 0.1);
        assert!((reported_variance / 128.0 - 1.0 / 16.0).abs() < 0.02);
    }

    #[test]
    fn qmc_standard_error_is_replicate_variance_and_counts_every_sample_once() {
        let calls = AtomicUsize::new(0);
        let result = qmc_quad_with_options(
            |_| (calls.fetch_add(1, Ordering::Relaxed) + 1) as f64,
            &[0.0],
            &[2.0],
            QmcQuadOptions {
                n_estimates: 4,
                n_points: 3,
                ..small_options()
            },
        )
        .unwrap();
        // Replicate means are 2, 5, 8, 11; signed volume is 2.
        assert_eq!(calls.load(Ordering::Relaxed), 12);
        assert_eq!(result.integral, 13.0);
        close_relative(result.standard_error, 15.0_f64.sqrt(), 2e-15);
    }

    #[test]
    fn qmc_keeps_variation_smaller_than_an_ulp_of_large_replicate_means() {
        let increments = [
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ];
        for exponent in [100, 900] {
            let offset = 2.0_f64.powi(exponent);
            let unit = 2.0_f64.powi(exponent - 52);
            let calls = AtomicUsize::new(0);
            let result = qmc_quad_with_options(
                |_| offset + increments[calls.fetch_add(1, Ordering::Relaxed)] * unit,
                &[0.0],
                &[1.0],
                QmcQuadOptions {
                    n_estimates: 4,
                    n_points: 4,
                    ..small_options()
                },
            )
            .unwrap();
            // Centered replicate means: .25, .5, .75, 1 ulps. Forming the
            // uncentered means first destroys the variance we need to report.
            assert_eq!(calls.load(Ordering::Relaxed), 16);
            assert_eq!(result.integral, offset + unit);
            close_relative(
                result.standard_error,
                unit * (5.0_f64 / 192.0).sqrt(),
                2e-15,
            );
        }
    }

    #[test]
    fn qmc_volume_scaling_preserves_finite_mixed_range_integrals() {
        let options = QmcQuadOptions {
            n_estimates: 3,
            n_points: 4,
            ..small_options()
        };
        let mixed =
            qmc_quad_with_options(|_| 2.0, &[0.0; 4], &[1e200, 1e200, 1e-200, 1e-200], options)
                .unwrap();
        close_relative(mixed.integral, 2.0, 3e-15);
        assert_eq!(mixed.standard_error, 0.0);
        for (width, value, expected) in [(1e-200, 1e300, 1e-100), (1e200, 1e-300, 1e100)] {
            let result = qmc_quad_with_options(|_| value, &[0.0; 2], &[width; 2], options).unwrap();
            close_relative(result.integral, expected, 3e-15);
            assert_eq!(result.standard_error, 0.0);
        }
        let opposite = qmc_quad_with_options(
            |point| {
                assert!(point[0].is_finite());
                0.25
            },
            &[-f64::MAX],
            &[f64::MAX],
            options,
        )
        .unwrap();
        assert_eq!(opposite.integral, f64::MAX * 0.5);
        let subnormal =
            qmc_quad_with_options(|_| f64::MAX, &[0.0], &[f64::from_bits(1)], options).unwrap();
        assert_eq!(subnormal.integral, f64::from_bits(1) * f64::MAX);
    }

    #[test]
    fn qmc_opposite_large_samples_cancel_without_overflowing_the_centered_difference() {
        let calls = AtomicUsize::new(0);
        let result = qmc_quad_with_options(
            |_| {
                if calls.fetch_add(1, Ordering::Relaxed).is_multiple_of(2) {
                    f64::MAX
                } else {
                    -f64::MAX
                }
            },
            &[0.0],
            &[1.0],
            QmcQuadOptions {
                n_estimates: 2,
                n_points: 2,
                ..small_options()
            },
        )
        .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 4);
        assert_eq!(result.integral, 0.0);
        assert_eq!(result.standard_error, 0.0);
    }

    #[test]
    fn qmc_cancellation_keeps_small_remainders_and_their_standard_error() {
        for tail in [1.0, 1e-200, 1e200] {
            let calls = AtomicUsize::new(0);
            let values = [f64::MAX, -f64::MAX, tail];
            let one = qmc_quad_with_options(
                |_| values[calls.fetch_add(1, Ordering::Relaxed)],
                &[0.0],
                &[1.0],
                QmcQuadOptions {
                    n_estimates: 1,
                    n_points: 3,
                    ..small_options()
                },
            )
            .unwrap();
            close_relative(one.integral, tail / 3.0, 3e-15);
            assert!(one.standard_error.is_nan());

            let calls = AtomicUsize::new(0);
            let values = [f64::MAX, -f64::MAX, tail, f64::MAX, -f64::MAX, 2.0 * tail];
            let two = qmc_quad_with_options(
                |_| values[calls.fetch_add(1, Ordering::Relaxed)],
                &[0.0],
                &[1.0],
                QmcQuadOptions {
                    n_estimates: 2,
                    n_points: 3,
                    ..small_options()
                },
            )
            .unwrap();
            close_relative(two.integral, tail * 0.5, 3e-15);
            close_relative(two.standard_error, tail / 6.0, 3e-15);
        }
    }

    #[test]
    fn qmc_cancellation_across_replicates_retains_both_sum_components() {
        let calls = AtomicUsize::new(0);
        let values = [f64::MAX, 1.0, -f64::MAX, 1.0];
        let result = qmc_quad_with_options(
            |_| values[calls.fetch_add(1, Ordering::Relaxed)],
            &[0.0],
            &[1.0],
            QmcQuadOptions {
                n_estimates: 2,
                n_points: 2,
                ..small_options()
            },
        )
        .unwrap();
        assert_eq!(result.integral, 0.5);
        close_relative(result.standard_error, f64::MAX * 0.5, 3e-15);
    }

    #[test]
    fn qmc_reversed_axes_change_only_the_integral_sign() {
        let f = |point: &[f64]| (point[0] * point[1]).exp();
        let options = QmcQuadOptions {
            n_points: 128,
            ..small_options()
        };
        let forward = qmc_quad_with_options(f, &[-2.0, -1.0], &[1.0, 3.0], options).unwrap();
        let one = qmc_quad_with_options(f, &[1.0, -1.0], &[-2.0, 3.0], options).unwrap();
        let two = qmc_quad_with_options(f, &[1.0, 3.0], &[-2.0, -1.0], options).unwrap();
        assert_eq!(forward.integral, -one.integral);
        assert_eq!(forward.integral, two.integral);
        assert_eq!(forward.standard_error, one.standard_error);
        assert_eq!(forward.standard_error, two.standard_error);
        assert!(forward.standard_error > 0.0);
    }

    #[test]
    fn qmc_single_replicate_has_undefined_error_and_zero_volume_needs_no_callback() {
        let single = qmc_quad_with_options(
            |_| 3.0,
            &[0.0],
            &[2.0],
            QmcQuadOptions {
                n_estimates: 1,
                ..small_options()
            },
        )
        .unwrap();
        assert_eq!(single.integral, 6.0);
        assert!(single.standard_error.is_nan());
        let zero = qmc_quad_with_options(
            |_| panic!("zero-volume domain must not evaluate the integrand"),
            &[0.0, -f64::MAX],
            &[0.0, f64::MAX],
            QmcQuadOptions {
                n_estimates: 1,
                ..small_options()
            },
        )
        .unwrap();
        assert_eq!(zero.integral, 0.0);
        assert_eq!(zero.standard_error, 0.0);
    }

    #[test]
    fn qmc_invalid_requests_fail_before_calling_the_integrand() {
        let never = |_: &[f64]| -> f64 { panic!("invalid request evaluated the integrand") };
        for (lower, upper) in [
            (vec![], vec![]),
            (vec![0.0], vec![1.0, 2.0]),
            (vec![0.0; 33], vec![1.0; 33]),
            (vec![f64::NAN], vec![1.0]),
            (vec![0.0], vec![f64::INFINITY]),
        ] {
            assert!(matches!(
                qmc_quad_with_options(never, &lower, &upper, small_options()),
                Err(IntegrateValidationError::QuadInvalidBounds { .. })
            ));
        }
        for options in [
            QmcQuadOptions {
                n_estimates: 0,
                ..small_options()
            },
            QmcQuadOptions {
                n_points: 0,
                ..small_options()
            },
            QmcQuadOptions {
                workers: Some(0),
                ..small_options()
            },
        ] {
            assert!(matches!(
                qmc_quad_with_options(never, &[0.0], &[1.0], options),
                Err(IntegrateValidationError::QuadInvalidTolerance { .. })
            ));
        }
        assert!(matches!(
            qmc_quad_with_options(
                never,
                &[0.0],
                &[1.0],
                QmcQuadOptions {
                    n_estimates: usize::MAX,
                    n_points: 2,
                    ..small_options()
                }
            ),
            Err(IntegrateValidationError::ResourceExhausted { .. })
        ));
    }

    #[test]
    fn qmc_nonfinite_samples_and_unrepresentable_results_fail_explicitly() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                qmc_quad_with_options(|_| bad, &[0.0], &[1.0], small_options()),
                Err(IntegrateValidationError::IntegrationFailed { .. })
            ));
        }
        let calls = AtomicUsize::new(0);
        let error = qmc_quad_with_options(
            |_| {
                if calls.fetch_add(1, Ordering::Relaxed) == 2 {
                    f64::NAN
                } else {
                    1.0
                }
            },
            &[0.0],
            &[1.0],
            small_options(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            IntegrateValidationError::IntegrationFailed { .. }
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 3);
        assert!(matches!(
            qmc_quad_with_options(|_| 1.0, &[0.0; 2], &[1e200; 2], small_options()),
            Err(IntegrateValidationError::IntegrationFailed { .. })
        ));
        let calls = AtomicUsize::new(0);
        let error = qmc_quad_with_options(
            |_| {
                if calls.fetch_add(1, Ordering::Relaxed) == 0 {
                    f64::MAX
                } else {
                    -f64::MAX
                }
            },
            &[0.0],
            &[2.0],
            QmcQuadOptions {
                n_estimates: 2,
                n_points: 1,
                ..small_options()
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            IntegrateValidationError::IntegrationFailed { message } if message.contains("standard error")
        ));
    }

    #[test]
    fn qmc_default_preserves_existing_analytical_accuracy_cases() {
        type Case = (fn(&[f64]) -> f64, Vec<f64>, Vec<f64>, f64);
        let cases: Vec<Case> = vec![
            (|p| p[0] * p[0], vec![0.0], vec![1.0], 1.0 / 3.0),
            (|p| p[0].sin(), vec![0.0], vec![std::f64::consts::PI], 2.0),
            (|p| p[0] + p[1], vec![0.0; 2], vec![1.0; 2], 1.0),
            (
                |p| (-p[0] * p[1]).exp(),
                vec![0.0; 2],
                vec![1.0; 2],
                0.796_599_599_297_053_2,
            ),
        ];
        for (f, lower, upper, expected) in cases {
            let result = qmc_quad(f, &lower, &upper, 8, 1024).unwrap();
            assert!((result.integral - expected).abs() <= 1e-3);
            assert!(result.standard_error.is_finite());
            assert!(result.standard_error > 0.0);
        }
    }
}
