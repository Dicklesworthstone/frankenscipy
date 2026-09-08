#![no_main]

use arbitrary::Arbitrary;
use fsci_stats::{
    circmean, circstd, circvar, gmean, gstd, hmean, pmean, quantile, weighted_mean, weighted_var,
};
use libfuzzer_sys::fuzz_target;

const MAX_LEN: usize = 128;
const TOL: f64 = 1e-10;

#[derive(Debug, Arbitrary)]
struct DescriptiveInput {
    raw_data: Vec<f64>,
    raw_weights: Vec<f64>,
    raw_quantiles: Vec<f64>,
    p_exp: i8,
}

fn sanitize(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(-1e6, 1e6)
    } else {
        0.0
    }
}

fn sanitize_positive(x: f64) -> f64 {
    if x.is_finite() && x > 0.0 {
        x.clamp(1e-10, 1e6)
    } else {
        1.0
    }
}

fn reference_weighted_mean(values: &[f64], weights: &[f64]) -> f64 {
    let sum_w: f64 = weights.iter().sum();
    if sum_w == 0.0 {
        return f64::NAN;
    }
    values
        .iter()
        .zip(weights.iter())
        .map(|(&v, &w)| v * w)
        .sum::<f64>()
        / sum_w
}

fn reference_weighted_var(values: &[f64], weights: &[f64]) -> f64 {
    let mean = reference_weighted_mean(values, weights);
    if !mean.is_finite() {
        return f64::NAN;
    }
    let sum_w: f64 = weights.iter().sum();
    if sum_w == 0.0 {
        return f64::NAN;
    }
    values
        .iter()
        .zip(weights.iter())
        .map(|(&v, &w)| w * (v - mean).powi(2))
        .sum::<f64>()
        / sum_w
}

fn reference_gmean(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) {
        return f64::NAN;
    }
    let log_sum: f64 = data.iter().map(|&x| x.ln()).sum();
    (log_sum / data.len() as f64).exp()
}

fn reference_hmean(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|&x| x <= 0.0) {
        return f64::NAN;
    }
    let inv_sum: f64 = data.iter().map(|&x| 1.0 / x).sum();
    data.len() as f64 / inv_sum
}

fn reference_pmean(data: &[f64], p: f64) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    if p == 0.0 {
        return reference_gmean(data);
    }
    let pow_sum: f64 = data.iter().map(|&x| x.abs().powf(p)).sum();
    (pow_sum / data.len() as f64).powf(1.0 / p)
}

fn reference_circmean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let sin_sum: f64 = data.iter().map(|&x| x.sin()).sum();
    let cos_sum: f64 = data.iter().map(|&x| x.cos()).sum();
    sin_sum.atan2(cos_sum)
}

fn reference_circvar(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let n = data.len() as f64;
    let sin_sum: f64 = data.iter().map(|&x| x.sin()).sum();
    let cos_sum: f64 = data.iter().map(|&x| x.cos()).sum();
    let r = (sin_sum.powi(2) + cos_sum.powi(2)).sqrt() / n;
    1.0 - r
}

fuzz_target!(|input: DescriptiveInput| {
    if input.raw_data.is_empty() {
        return;
    }

    let data: Vec<f64> = input
        .raw_data
        .iter()
        .take(MAX_LEN)
        .map(|&v| sanitize(v))
        .collect();

    if data.is_empty() {
        return;
    }

    // Oracle 1: weighted_mean with uniform weights equals arithmetic mean
    let uniform_weights: Vec<f64> = vec![1.0; data.len()];
    let wm = weighted_mean(&data, &uniform_weights);
    let arith_mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
    if wm.is_finite() && arith_mean.is_finite() {
        assert!(
            (wm - arith_mean).abs() < TOL,
            "weighted_mean with uniform weights {} != arithmetic mean {}",
            wm,
            arith_mean
        );
    }

    // Oracle 2: weighted_mean matches reference implementation
    let weights: Vec<f64> = input
        .raw_weights
        .iter()
        .take(data.len())
        .map(|&w| sanitize_positive(w))
        .collect();
    if weights.len() == data.len() {
        let wm = weighted_mean(&data, &weights);
        let ref_wm = reference_weighted_mean(&data, &weights);
        if wm.is_finite() && ref_wm.is_finite() {
            assert!(
                (wm - ref_wm).abs() < TOL,
                "weighted_mean {} != reference {}",
                wm,
                ref_wm
            );
        }
    }

    // Oracle 3: weighted_var >= 0 and matches reference
    if weights.len() == data.len() {
        let wv = weighted_var(&data, &weights);
        if wv.is_finite() {
            assert!(wv >= -TOL, "weighted_var {} must be >= 0", wv);
            let ref_wv = reference_weighted_var(&data, &weights);
            if ref_wv.is_finite() {
                assert!(
                    (wv - ref_wv).abs() < TOL,
                    "weighted_var {} != reference {}",
                    wv,
                    ref_wv
                );
            }
        }
    }

    // Oracle 4: gmean with all positive data
    let positive_data: Vec<f64> = data
        .iter()
        .map(|&x| if x > 0.0 { x } else { 1.0 })
        .collect();
    let gm = gmean(&positive_data);
    let ref_gm = reference_gmean(&positive_data);
    if gm.is_finite() && ref_gm.is_finite() {
        let rel_err = (gm - ref_gm).abs() / ref_gm.abs().max(1e-10);
        assert!(
            rel_err < 1e-8,
            "gmean {} != reference {} (rel_err {})",
            gm,
            ref_gm,
            rel_err
        );
    }

    // Oracle 5: gstd >= 1 for positive data (geometric std is multiplicative)
    let gs = gstd(&positive_data);
    if gs.is_finite() {
        assert!(
            gs >= 1.0 - TOL,
            "gstd {} should be >= 1 for positive data",
            gs
        );
    }

    // Oracle 6: hmean <= gmean <= arithmetic mean (for positive data) and matches reference
    let hm = hmean(&positive_data);
    let am: f64 = positive_data.iter().sum::<f64>() / positive_data.len() as f64;
    if hm.is_finite() && gm.is_finite() && am.is_finite() {
        assert!(
            hm <= gm + TOL,
            "hmean {} should be <= gmean {}",
            hm,
            gm
        );
        assert!(
            gm <= am + TOL,
            "gmean {} should be <= arithmetic mean {}",
            gm,
            am
        );
        let ref_hm = reference_hmean(&positive_data);
        if ref_hm.is_finite() {
            let rel_err = (hm - ref_hm).abs() / ref_hm.abs().max(1e-10);
            assert!(
                rel_err < 1e-8,
                "hmean {} != reference {} (rel_err {})",
                hm,
                ref_hm,
                rel_err
            );
        }
    }

    // Oracle 7: pmean with p=1 equals arithmetic mean
    let pm1 = pmean(&data.iter().map(|&x| x.abs().max(1e-10)).collect::<Vec<_>>(), 1.0);
    let abs_mean: f64 = data.iter().map(|&x| x.abs().max(1e-10)).sum::<f64>() / data.len() as f64;
    if pm1.is_finite() && abs_mean.is_finite() {
        assert!(
            (pm1 - abs_mean).abs() < TOL,
            "pmean(p=1) {} != arithmetic mean of abs {}",
            pm1,
            abs_mean
        );
    }

    // Oracle 8: pmean with p=2 equals RMS and matches reference
    let abs_data: Vec<f64> = data.iter().map(|&x| x.abs().max(1e-10)).collect();
    let pm2 = pmean(&abs_data, 2.0);
    let rms = (data.iter().map(|&x| x.powi(2)).sum::<f64>() / data.len() as f64).sqrt();
    if pm2.is_finite() && rms.is_finite() {
        assert!(
            (pm2 - rms).abs() < TOL,
            "pmean(p=2) {} != RMS {}",
            pm2,
            rms
        );
        let ref_pm2 = reference_pmean(&abs_data, 2.0);
        if ref_pm2.is_finite() {
            assert!(
                (pm2 - ref_pm2).abs() < TOL,
                "pmean(p=2) {} != reference {}",
                pm2,
                ref_pm2
            );
        }
    }

    // Oracle 9: circmean matches reference
    let cm = circmean(&data);
    let ref_cm = reference_circmean(&data);
    if cm.is_finite() && ref_cm.is_finite() {
        let diff = (cm - ref_cm).abs();
        let wrapped_diff = diff.min((2.0 * std::f64::consts::PI - diff).abs());
        assert!(
            wrapped_diff < TOL,
            "circmean {} != reference {}",
            cm,
            ref_cm
        );
    }

    // Oracle 10: circvar in [0, 1] and matches reference
    let cv = circvar(&data);
    if cv.is_finite() {
        assert!(
            (-TOL..=1.0 + TOL).contains(&cv),
            "circvar {} must be in [0, 1]",
            cv
        );
        let ref_cv = reference_circvar(&data);
        if ref_cv.is_finite() {
            assert!(
                (cv - ref_cv).abs() < TOL,
                "circvar {} != reference {}",
                cv,
                ref_cv
            );
        }
    }

    // Oracle 11: circstd >= 0
    let cs = circstd(&data);
    if cs.is_finite() {
        assert!(cs >= -TOL, "circstd {} must be >= 0", cs);
    }

    // Oracle 12: quantile at 0.5 is median
    let q = quantile(&data, &[0.5]);
    if !q.is_empty() && q[0].is_finite() {
        let mut sorted = data.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        let expected_median = if n % 2 == 1 {
            sorted[n / 2]
        } else {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        };
        assert!(
            (q[0] - expected_median).abs() < TOL,
            "quantile(0.5) {} != median {}",
            q[0],
            expected_median
        );
    }

    // Oracle 13: quantile(0) = min, quantile(1) = max
    let q_bounds = quantile(&data, &[0.0, 1.0]);
    if q_bounds.len() == 2 {
        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if q_bounds[0].is_finite() {
            assert!(
                (q_bounds[0] - min_val).abs() < TOL,
                "quantile(0) {} != min {}",
                q_bounds[0],
                min_val
            );
        }
        if q_bounds[1].is_finite() {
            assert!(
                (q_bounds[1] - max_val).abs() < TOL,
                "quantile(1) {} != max {}",
                q_bounds[1],
                max_val
            );
        }
    }

    // Oracle 14: quantiles are monotonically increasing
    let qs: Vec<f64> = (0..=10).map(|i| i as f64 / 10.0).collect();
    let q_vals = quantile(&data, &qs);
    for i in 1..q_vals.len() {
        if q_vals[i].is_finite() && q_vals[i - 1].is_finite() {
            assert!(
                q_vals[i] >= q_vals[i - 1] - TOL,
                "quantile not monotonic: q[{}]={} < q[{}]={}",
                i,
                q_vals[i],
                i - 1,
                q_vals[i - 1]
            );
        }
    }

    // Oracle 15: pmean with arbitrary p matches reference
    let p = input.p_exp as f64;
    if p.abs() > 0.1 && p.abs() < 10.0 {
        let pm = pmean(&abs_data, p);
        let ref_pm = reference_pmean(&abs_data, p);
        if pm.is_finite() && ref_pm.is_finite() {
            let rel_err = (pm - ref_pm).abs() / ref_pm.abs().max(1e-10);
            assert!(
                rel_err < 1e-6,
                "pmean(p={}) {} != reference {} (rel_err {})",
                p,
                pm,
                ref_pm,
                rel_err
            );
        }
    }

    // Oracle 16: quantiles from raw_quantiles
    let user_qs: Vec<f64> = input
        .raw_quantiles
        .iter()
        .take(20)
        .filter(|&&q| q.is_finite() && (0.0..=1.0).contains(&q))
        .copied()
        .collect();
    if !user_qs.is_empty() {
        let user_q_vals = quantile(&data, &user_qs);
        for (i, &qv) in user_q_vals.iter().enumerate() {
            if qv.is_finite() {
                let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                assert!(
                    (min_val - TOL..=max_val + TOL).contains(&qv),
                    "quantile({}) {} outside data range [{}, {}]",
                    user_qs[i],
                    qv,
                    min_val,
                    max_val
                );
            }
        }
    }
});
