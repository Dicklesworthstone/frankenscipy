#![no_main]

mod p2c007_stats_common;

use arbitrary::Arbitrary;
use fsci_stats::{
    describe, iqr, median_abs_deviation, percentileofscore, scoreatpercentile, trim_mean,
    variation, zscore,
};
use libfuzzer_sys::fuzz_target;
use p2c007_stats_common::EdgeF64;

const MAX_LEN: usize = 128;
const MAX_PERCENTILES: usize = 16;

#[derive(Debug, Arbitrary)]
struct PercentileInput {
    data: Vec<EdgeF64>,
    percentiles: Vec<EdgeF64>,
    lower: EdgeF64,
    upper: EdgeF64,
    score: EdgeF64,
    trim: EdgeF64,
    kind: u8,
    method: u8,
    use_limit: bool,
}

fn data_values(input: &[EdgeF64]) -> Vec<f64> {
    input.iter().take(MAX_LEN).map(|edge| edge.raw()).collect()
}

fn finite_data(input: &[EdgeF64]) -> Vec<f64> {
    let mut values: Vec<f64> = input
        .iter()
        .take(MAX_LEN)
        .map(|edge| edge.finite(-1.0e6, 1.0e6, 0.0))
        .collect();
    if values.is_empty() {
        values.push(0.0);
    }
    values
}

fn percentile_values(input: &[EdgeF64]) -> Vec<f64> {
    let mut values: Vec<f64> = input
        .iter()
        .take(MAX_PERCENTILES)
        .map(|edge| edge.finite(-100.0, 200.0, 50.0))
        .collect();
    if values.is_empty() {
        values.push(50.0);
    }
    values
}

fn kind(selector: u8) -> Option<&'static str> {
    match selector % 5 {
        0 => None,
        1 => Some("rank"),
        2 => Some("weak"),
        3 => Some("strict"),
        _ => Some("mean"),
    }
}

fn method(selector: u8) -> Option<&'static str> {
    match selector % 5 {
        0 => None,
        1 => Some("fraction"),
        2 => Some("lower"),
        3 => Some("higher"),
        _ => Some("bogus"),
    }
}

fn assert_probability_like(name: &str, value: f64) {
    if value.is_finite() {
        assert!(
            (-1.0e-9..=100.0 + 1.0e-9).contains(&value),
            "{name}: expected percentage in [0, 100], got {value}"
        );
    }
}

fuzz_target!(|input: PercentileInput| {
    let raw_data = data_values(&input.data);
    let finite_data = finite_data(&input.data);
    let percentiles = percentile_values(&input.percentiles);
    let lower = input.lower.finite(-1.0e6, 1.0e6, -1.0e6);
    let upper = input.upper.finite(-1.0e6, 1.0e6, 1.0e6);
    let limit = if input.use_limit {
        Some((lower.min(upper), lower.max(upper)))
    } else {
        None
    };

    let score = input.score.raw();
    assert_probability_like(
        "percentileofscore",
        percentileofscore(&raw_data, score, kind(input.kind)),
    );

    if let Ok(values) = scoreatpercentile(&raw_data, &percentiles, limit, method(input.method)) {
        assert_eq!(
            values.len(),
            percentiles.len(),
            "scoreatpercentile output length mismatch"
        );
    }

    let trim = input.trim.finite(0.0, 0.5, 0.0);
    let trimmed = trim_mean(&raw_data, trim);
    if let Ok(val) = trimmed
        && val.is_finite()
    {
        assert!(
            raw_data.iter().any(|value| !value.is_nan()),
            "trim_mean returned finite value for all-NaN data"
        );
    }

    let z = zscore(&finite_data);
    assert_eq!(z.len(), finite_data.len(), "zscore length mismatch");
    for (idx, value) in z.iter().enumerate() {
        assert!(
            value.is_nan() || value.is_finite(),
            "zscore invalid at {idx}"
        );
    }

    let _ = describe(&raw_data);
    let _ = median_abs_deviation(&raw_data, 1.0);
    let iqr_value = iqr(&finite_data);
    assert!(
        iqr_value.is_nan() || iqr_value >= 0.0,
        "iqr must be non-negative"
    );
    let variation_value = variation(&finite_data);
    assert!(
        variation_value.is_nan() || variation_value.is_finite(),
        "variation should be NaN or finite"
    );
});
