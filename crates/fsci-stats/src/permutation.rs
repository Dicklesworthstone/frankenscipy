//! Scalar permutation tests for independent samples, paired sample labels,
//! and within-sample pairings. Exact tests enumerate the combinatorial null;
//! randomized tests retain every draw and use the conservative +1 correction.

use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::{
    RuntimeMode, StatsError, hardened_max_pairwise_len, hardened_max_resamples,
    hardened_max_sample_len, with_stats_runtime_mode,
};

/// What is exchangeable under the null hypothesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermutationType {
    /// Reassign observations between independent samples, preserving sample sizes.
    Independent,
    /// Permute sample labels separately within each paired observation.
    /// With one sample, independently reverse the sign of each observation.
    Samples,
    /// Independently permute the observation order within every sample.
    /// A single sample can be permuted against a fixed reference in the callback.
    Pairings,
}

/// Which tail of the permutation distribution defines the alternative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermutationAlternative {
    /// Twice the smaller one-sided p-value, clipped to one.
    TwoSided,
    Less,
    Greater,
}

/// Controls for a scalar permutation test.
///
/// The number of resamples is an explicit work and output-size budget. If the
/// number of distinct permutations fits in that budget, the test enumerates
/// them exactly; otherwise it draws exactly this many random permutations.
/// Counts are checked without constructing the permutation space.
///
/// Hardened mode checks the existing sample and resample resource caps before
/// invoking the statistic or allocating resampling buffers. Each sample is
/// bounded by hardened_max_pairwise_len and by hardened_max_sample_len, the way
/// the other two-sample hardened tests bound each of their inputs; the work is
/// bounded by n_resamples, which hardened_max_resamples caps. Allocation-size
/// overflow and allocation failure produce typed errors in either mode.
///
/// The random seed is reproducible within this implementation; it does not
/// select NumPy's random-number stream. Scalar callbacks are evaluated in
/// deterministic order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PermutationTestOptions {
    pub permutation_type: PermutationType,
    pub alternative: PermutationAlternative,
    pub n_resamples: usize,
    pub rng: u64,
    pub mode: RuntimeMode,
}

impl Default for PermutationTestOptions {
    fn default() -> Self {
        Self {
            permutation_type: PermutationType::Independent,
            alternative: PermutationAlternative::TwoSided,
            n_resamples: 9_999,
            rng: 0,
            mode: RuntimeMode::Strict,
        }
    }
}

/// Observed statistic, p-value, and the complete scalar null distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct PermutationTestResult {
    pub statistic: f64,
    pub pvalue: f64,
    /// Statistics in deterministic enumeration or random-draw order.
    pub null_distribution: Vec<f64>,
    /// Whether all distinct permutations were enumerated.
    pub exact: bool,
}

fn resource_error() -> StatsError {
    StatsError::ResourceExhausted {
        detail: "permutation test allocation size or capacity is not representable",
    }
}

fn check_allocation<T>(length: usize) -> Result<(), StatsError> {
    let bytes = length
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(resource_error)?;
    if bytes > isize::MAX as usize {
        return Err(resource_error());
    }
    Ok(())
}

fn filled<T: Clone>(length: usize, value: T) -> Result<Vec<T>, StatsError> {
    check_allocation::<T>(length)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| resource_error())?;
    values.resize(length, value);
    Ok(values)
}

fn indices(length: usize) -> Result<Vec<usize>, StatsError> {
    let mut result = filled(length, 0)?;
    for (index, value) in result.iter_mut().enumerate() {
        *value = index;
    }
    Ok(result)
}

/// Return C(n,k) only when it fits within the supplied budget. Each intermediate
/// is at most usize::MAX squared before exact division, so u128 suffices on
/// supported 32-/64-bit targets.
fn binomial_bounded(n: usize, k: usize, limit: usize) -> Option<usize> {
    let k = k.min(n - k);
    let mut value = 1_u128;
    for i in 1..=k {
        value = value * (n - k + i) as u128 / i as u128;
        if value > limit as u128 {
            return None;
        }
    }
    (value <= limit as u128).then_some(value as usize)
}

fn factorial_bounded(n: usize, limit: usize) -> Option<usize> {
    let mut value = 1_usize;
    for factor in 2..=n {
        if value > limit / factor {
            return None;
        }
        value *= factor;
    }
    (value <= limit).then_some(value)
}

fn power_bounded(base: usize, exponent: usize, limit: usize) -> Option<usize> {
    let mut value = 1_usize;
    for _ in 0..exponent {
        if value > limit / base {
            return None;
        }
        value *= base;
    }
    Some(value)
}

fn exact_count(
    data: &[&[f64]],
    total: usize,
    kind: PermutationType,
    limit: usize,
) -> Option<usize> {
    match kind {
        PermutationType::Independent => {
            let mut remaining = total;
            let mut count = 1_usize;
            for sample in &data[..data.len() - 1] {
                let ways = binomial_bounded(remaining, sample.len(), limit / count)?;
                count *= ways;
                remaining -= sample.len();
            }
            Some(count)
        }
        PermutationType::Samples if data.len() == 1 => power_bounded(2, data[0].len(), limit),
        PermutationType::Samples => {
            let labels = factorial_bounded(data.len(), limit)?;
            power_bounded(labels, data[0].len(), limit)
        }
        PermutationType::Pairings => {
            let orders = factorial_bounded(data[0].len(), limit)?;
            power_bounded(orders, data.len(), limit)
        }
    }
}

fn validate(data: &[&[f64]], options: PermutationTestOptions) -> Result<usize, StatsError> {
    if options.n_resamples == 0 {
        return Err(StatsError::InvalidArgument(
            "n_resamples must be a positive integer".into(),
        ));
    }
    let minimum_samples = if options.permutation_type == PermutationType::Independent {
        2
    } else {
        1
    };
    if data.len() < minimum_samples {
        return Err(StatsError::InvalidArgument(format!(
            "this permutation type requires at least {minimum_samples} samples"
        )));
    }
    if options.mode == RuntimeMode::Hardened && options.n_resamples > hardened_max_resamples() {
        return Err(StatsError::ResourceExhausted {
            detail: "Hardened mode n_resamples exceeds HARDENED_MAX_RESAMPLES",
        });
    }
    let mut total = 0_usize;
    for sample in data {
        if sample.len() < 2 {
            return Err(StatsError::DataTooSmall {
                required: 2,
                got: sample.len(),
            });
        }
        if options.permutation_type != PermutationType::Independent && sample.len() != data[0].len()
        {
            return Err(StatsError::ShapeMismatch {
                expected: data[0].len(),
                actual: sample.len(),
            });
        }
        total = total.checked_add(sample.len()).ok_or_else(resource_error)?;
        // Per sample, as in ttest_ind_with_options, pearsonr_with_options and
        // spearmanr_with_options. The combinatorial work needs no length cap:
        // at most n_resamples statistic calls are made, and an exact
        // Independent enumeration needs C(total, 2) <= n_resamples, so
        // total <= 447 at the default HARDENED_MAX_RESAMPLES.
        if options.mode == RuntimeMode::Hardened
            && (sample.len() > hardened_max_pairwise_len()
                || sample.len() > hardened_max_sample_len())
        {
            return Err(StatsError::ResourceExhausted {
                detail: "Hardened mode permutation sample lengths exceed stats caps",
            });
        }
    }
    check_allocation::<f64>(total)?;
    check_allocation::<usize>(total)?;
    check_allocation::<Vec<f64>>(data.len())?;
    check_allocation::<&[f64]>(data.len())?;
    if data
        .iter()
        .any(|sample| sample.iter().any(|x| !x.is_finite()))
    {
        return Err(StatsError::NonFiniteInput { argument: "data" });
    }
    Ok(total)
}

fn sample_buffers(data: &[&[f64]]) -> Result<Vec<Vec<f64>>, StatsError> {
    let mut rows = Vec::new();
    rows.try_reserve_exact(data.len())
        .map_err(|_| resource_error())?;
    for sample in data {
        let mut row = filled(sample.len(), 0.0)?;
        row.copy_from_slice(sample);
        rows.push(row);
    }
    Ok(rows)
}

enum Enumeration {
    Independent {
        remaining: Vec<Vec<usize>>,
        selections: Vec<Vec<usize>>,
    },
    Samples(Vec<Vec<usize>>),
    Pairings(Vec<Vec<usize>>),
    Signs,
    Random,
}

fn enumeration(
    data: &[&[f64]],
    total: usize,
    kind: PermutationType,
    exact: bool,
) -> Result<Enumeration, StatsError> {
    if !exact {
        return Ok(Enumeration::Random);
    }
    match kind {
        PermutationType::Independent => {
            let mut remaining = Vec::new();
            let mut selections = Vec::new();
            remaining
                .try_reserve_exact(data.len())
                .map_err(|_| resource_error())?;
            selections
                .try_reserve_exact(data.len())
                .map_err(|_| resource_error())?;
            let mut length = total;
            for sample in data {
                remaining.push(filled(length, 0)?);
                selections.push(indices(sample.len())?);
                length -= sample.len();
            }
            for (index, value) in remaining[0].iter_mut().enumerate() {
                *value = index;
            }
            Ok(Enumeration::Independent {
                remaining,
                selections,
            })
        }
        PermutationType::Samples if data.len() == 1 => Ok(Enumeration::Signs),
        PermutationType::Samples => {
            let mut orders = Vec::new();
            orders
                .try_reserve_exact(data[0].len())
                .map_err(|_| resource_error())?;
            for _ in 0..data[0].len() {
                orders.push(indices(data.len())?);
            }
            Ok(Enumeration::Samples(orders))
        }
        PermutationType::Pairings => {
            let mut orders = Vec::new();
            orders
                .try_reserve_exact(data.len())
                .map_err(|_| resource_error())?;
            for sample in data {
                orders.push(indices(sample.len())?);
            }
            Ok(Enumeration::Pairings(orders))
        }
    }
}

struct NullDistribution<'a, F> {
    statistic: &'a F,
    values: Vec<f64>,
    expected: usize,
}

impl<F: Fn(&[&[f64]]) -> f64> NullDistribution<'_, F> {
    fn record(&mut self, rows: &[Vec<f64>]) -> Result<(), StatsError> {
        let mut views = Vec::new();
        views
            .try_reserve_exact(rows.len())
            .map_err(|_| resource_error())?;
        views.extend(rows.iter().map(Vec::as_slice));
        let value = (self.statistic)(&views);
        if !value.is_finite() {
            return Err(StatsError::Arithmetic {
                detail: "permutation statistic returned a nonfinite value on a resample".into(),
            });
        }
        if self.values.len() >= self.expected {
            return Err(StatsError::Arithmetic {
                detail: "permutation enumeration exceeded its computed size".into(),
            });
        }
        self.values.push(value);
        Ok(())
    }
}

fn next_combination(chosen: &mut [usize], n: usize) -> bool {
    let k = chosen.len();
    for i in (0..k).rev() {
        if chosen[i] < n - k + i {
            chosen[i] += 1;
            for j in i + 1..k {
                chosen[j] = chosen[j - 1] + 1;
            }
            return true;
        }
    }
    false
}

fn next_permutation(order: &mut [usize]) -> bool {
    let Some(pivot) = (0..order.len() - 1)
        .rev()
        .find(|&i| order[i] < order[i + 1])
    else {
        return false;
    };
    let successor = (pivot + 1..order.len())
        .rev()
        .find(|&i| order[i] > order[pivot])
        .expect("an increasing suffix has a successor");
    order.swap(pivot, successor);
    order[pivot + 1..].reverse();
    true
}

fn enumerate_independent<F: Fn(&[&[f64]]) -> f64>(
    group: usize,
    pooled: &[f64],
    rows: &mut [Vec<f64>],
    remaining: &mut [Vec<usize>],
    selections: &mut [Vec<usize>],
    output: &mut NullDistribution<'_, F>,
) -> Result<(), StatsError> {
    if group + 1 == rows.len() {
        for (value, &index) in rows[group].iter_mut().zip(&remaining[group]) {
            *value = pooled[index];
        }
        return output.record(rows);
    }
    for (index, selection) in selections[group].iter_mut().enumerate() {
        *selection = index;
    }
    loop {
        for (position, &chosen) in selections[group].iter().enumerate() {
            rows[group][position] = pooled[remaining[group][chosen]];
        }
        let mut selected = 0;
        let mut next = 0;
        for position in 0..remaining[group].len() {
            if selected < selections[group].len() && selections[group][selected] == position {
                selected += 1;
            } else {
                let index = remaining[group][position];
                remaining[group + 1][next] = index;
                next += 1;
            }
        }
        enumerate_independent(group + 1, pooled, rows, remaining, selections, output)?;
        if !next_combination(&mut selections[group], remaining[group].len()) {
            return Ok(());
        }
    }
}

fn enumerate_samples<F: Fn(&[&[f64]]) -> f64>(
    observation: usize,
    data: &[&[f64]],
    rows: &mut [Vec<f64>],
    orders: &mut [Vec<usize>],
    output: &mut NullDistribution<'_, F>,
) -> Result<(), StatsError> {
    if observation == data[0].len() {
        return output.record(rows);
    }
    for (index, value) in orders[observation].iter_mut().enumerate() {
        *value = index;
    }
    loop {
        for sample in 0..data.len() {
            rows[sample][observation] = data[orders[observation][sample]][observation];
        }
        enumerate_samples(observation + 1, data, rows, orders, output)?;
        if !next_permutation(&mut orders[observation]) {
            return Ok(());
        }
    }
}

fn enumerate_pairings<F: Fn(&[&[f64]]) -> f64>(
    sample: usize,
    data: &[&[f64]],
    rows: &mut [Vec<f64>],
    orders: &mut [Vec<usize>],
    output: &mut NullDistribution<'_, F>,
) -> Result<(), StatsError> {
    if sample == data.len() {
        return output.record(rows);
    }
    for (index, value) in orders[sample].iter_mut().enumerate() {
        *value = index;
    }
    loop {
        for (position, &index) in orders[sample].iter().enumerate() {
            rows[sample][position] = data[sample][index];
        }
        enumerate_pairings(sample + 1, data, rows, orders, output)?;
        if !next_permutation(&mut orders[sample]) {
            return Ok(());
        }
    }
}

fn shuffle<T>(values: &mut [T], rng: &mut StdRng) {
    for i in (1..values.len()).rev() {
        let j = rng.random_range(0..=i);
        values.swap(i, j);
    }
}

pub(crate) fn resampling_pvalue(
    observed: f64,
    null: &[f64],
    exact: bool,
    alternative: PermutationAlternative,
) -> f64 {
    // SciPy regards values within 100 machine epsilons relative to the
    // observed statistic as equal. The tolerance is zero when it is zero.
    let tolerance = (100.0 * f64::EPSILON) * observed.abs();
    let less = null.iter().filter(|&&x| x <= observed + tolerance).count();
    let greater = null.iter().filter(|&&x| x >= observed - tolerance).count();
    let adjustment = if exact { 0.0 } else { 1.0 };
    let denominator = null.len() as f64 + adjustment;
    let lower = (less as f64 + adjustment) / denominator;
    let upper = (greater as f64 + adjustment) / denominator;
    match alternative {
        PermutationAlternative::Less => lower,
        PermutationAlternative::Greater => upper,
        PermutationAlternative::TwoSided => (2.0 * lower.min(upper)).min(1.0),
    }
}

/// Test a scalar statistic of one or more samples by permutation.
///
/// Independent samples may have different lengths, and require at least two
/// samples. Samples and Pairings require equal lengths and permit one sample.
/// Every sample must contain at least two finite observations; the statistic
/// must return a finite scalar for the original data and every resample.
///
/// Exact Independent tests enumerate sample memberships, not redundant orders
/// within each sample. Their statistic must therefore be invariant to order
/// within a sample, as in SciPy's independent permutation test. Samples keeps
/// observation pairings intact; Pairings preserves sample membership while
/// changing order. Statistics should be deterministic functions of their inputs.
///
/// Exact tests use count/N. Randomized tests use (count+1)/(N+1). TwoSided is
/// twice the smaller inclusive tail probability, not an absolute-statistic
/// test; adding a constant to an ordinary statistic does not change its test.
pub fn permutation_test_samples<F>(
    data: &[&[f64]],
    statistic: F,
    options: PermutationTestOptions,
) -> Result<PermutationTestResult, StatsError>
where
    F: Fn(&[&[f64]]) -> f64 + Sync,
{
    let total = validate(data, options)?;
    let count = exact_count(data, total, options.permutation_type, options.n_resamples);
    let exact = count.is_some();
    let draws = count.unwrap_or(options.n_resamples);
    check_allocation::<f64>(draws)?;

    // Resource checks precede both user callbacks and resampling allocation.
    let mut null = Vec::new();
    null.try_reserve_exact(draws)
        .map_err(|_| resource_error())?;
    let mut rows = sample_buffers(data)?;
    let mut plan = enumeration(data, total, options.permutation_type, exact)?;
    let mut pooled = if options.permutation_type == PermutationType::Independent {
        filled(total, 0.0)?
    } else {
        Vec::new()
    };
    let mut offset = 0;
    for sample in data {
        if !pooled.is_empty() {
            pooled[offset..offset + sample.len()].copy_from_slice(sample);
        }
        offset += sample.len();
    }
    let mut shuffled = if !exact && !pooled.is_empty() {
        filled(total, 0.0)?
    } else {
        Vec::new()
    };
    let mut labels = if !exact && options.permutation_type == PermutationType::Samples {
        indices(data.len())?
    } else {
        Vec::new()
    };

    with_stats_runtime_mode(options.mode, || {
        let observed = statistic(data);
        if !observed.is_finite() {
            return Err(StatsError::Arithmetic {
                detail: "permutation statistic returned a nonfinite observed value".into(),
            });
        }
        let mut output = NullDistribution {
            statistic: &statistic,
            values: null,
            expected: draws,
        };
        match &mut plan {
            Enumeration::Independent {
                remaining,
                selections,
            } => {
                enumerate_independent(0, &pooled, &mut rows, remaining, selections, &mut output)?;
            }
            Enumeration::Samples(orders) => {
                enumerate_samples(0, data, &mut rows, orders, &mut output)?;
            }
            Enumeration::Pairings(orders) => {
                enumerate_pairings(0, data, &mut rows, orders, &mut output)?;
            }
            Enumeration::Signs => {
                for mask in 0..draws {
                    for (index, &value) in data[0].iter().enumerate() {
                        rows[0][index] = if (mask >> index) & 1 == 0 {
                            value
                        } else {
                            -value
                        };
                    }
                    output.record(&rows)?;
                }
            }
            Enumeration::Random => {
                for draw in 0..draws {
                    // Each draw has its own stream: its values never depend on
                    // callback behavior or how draws might later be partitioned.
                    let seed = options
                        .rng
                        .wrapping_add((draw as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
                    let mut rng = StdRng::seed_from_u64(seed);
                    match options.permutation_type {
                        PermutationType::Independent => {
                            shuffled.copy_from_slice(&pooled);
                            shuffle(&mut shuffled, &mut rng);
                            let mut start = 0;
                            for row in &mut rows {
                                let end = start + row.len();
                                row.copy_from_slice(&shuffled[start..end]);
                                start = end;
                            }
                        }
                        PermutationType::Samples if data.len() == 1 => {
                            for (value, &original) in rows[0].iter_mut().zip(data[0]) {
                                *value = if rng.random_range(0..2) == 0 {
                                    original
                                } else {
                                    -original
                                };
                            }
                        }
                        PermutationType::Samples => {
                            for observation in 0..data[0].len() {
                                for (index, value) in labels.iter_mut().enumerate() {
                                    *value = index;
                                }
                                shuffle(&mut labels, &mut rng);
                                for sample in 0..data.len() {
                                    rows[sample][observation] = data[labels[sample]][observation];
                                }
                            }
                        }
                        PermutationType::Pairings => {
                            for (row, sample) in rows.iter_mut().zip(data) {
                                row.copy_from_slice(sample);
                                shuffle(row, &mut rng);
                            }
                        }
                    }
                    output.record(&rows)?;
                }
            }
        }
        if output.values.len() != draws {
            return Err(StatsError::Arithmetic {
                detail: "permutation enumeration did not reach its computed size".into(),
            });
        }
        Ok(PermutationTestResult {
            statistic: observed,
            pvalue: resampling_pvalue(observed, &output.values, exact, options.alternative),
            null_distribution: output.values,
            exact,
        })
    })
}
