#![no_main]

use arbitrary::Arbitrary;
use fsci_signal::gauspuls;
use libfuzzer_sys::fuzz_target;

// gauspuls property oracle for [frankenscipy-a65ml].
//
// Verifies invariants on random valid input (fc, bw, bwr) and a small
// t-grid:
//
//   1. envelope ≥ 0 everywhere.
//   2. envelope at t=0 is exactly 1 regardless of fc/bw/bwr.
//   3. yI² + yQ² = envelope² (analytic-signal modulus).
//   4. envelope is monotonically non-increasing in |t| (Gaussian).
//   5. all outputs are finite for finite valid input.

const TOL: f64 = 1e-12;

#[derive(Debug, Arbitrary)]
struct GauspulsInput {
    fc: f64,
    bw: f64,
    bwr: f64,
    t0: f64,
    t1: f64,
    t2: f64,
    t3: f64,
}

fn sanitize_fc(value: f64) -> f64 {
    if value.is_finite() {
        value.abs().clamp(1.0, 1.0e6)
    } else {
        1000.0
    }
}

fn sanitize_bw(value: f64) -> f64 {
    if value.is_finite() {
        value.abs().clamp(1e-3, 2.0)
    } else {
        0.5
    }
}

fn sanitize_bwr(value: f64) -> f64 {
    if value.is_finite() {
        // Always negative dB; clamp to a sensible range.
        -(value.abs().clamp(0.5, 60.0))
    } else {
        -6.0
    }
}

fn sanitize_t(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

fuzz_target!(|input: GauspulsInput| {
    let fc = sanitize_fc(input.fc);
    let bw = sanitize_bw(input.bw);
    let bwr = sanitize_bwr(input.bwr);

    // Build an ascending t-grid that includes 0 and four other points.
    let mut ts: Vec<f64> = vec![
        sanitize_t(input.t0),
        sanitize_t(input.t1),
        sanitize_t(input.t2),
        sanitize_t(input.t3),
        0.0,
    ];
    ts.sort_by(|a, b| a.total_cmp(b));
    ts.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    let r = match gauspuls(&ts, fc, bw, bwr) {
        Ok(r) => r,
        Err(e) => panic!("gauspuls rejected sanitized valid input ({fc}, {bw}, {bwr}): {e}"),
    };

    assert_eq!(r.i.len(), ts.len(), "i length");
    assert_eq!(r.q.len(), ts.len(), "q length");
    assert_eq!(r.envelope.len(), ts.len(), "env length");

    // Property 1+5: envelope is finite and ≥ 0; everything finite.
    for (k, &e) in r.envelope.iter().enumerate() {
        if !e.is_finite() {
            panic!("envelope non-finite at k={k}: {e} (t={}, fc={fc}, bw={bw}, bwr={bwr})", ts[k]);
        }
        if e < 0.0 {
            panic!("envelope negative at k={k}: {e}");
        }
        if !r.i[k].is_finite() || !r.q[k].is_finite() {
            panic!("yI/yQ non-finite at k={k}");
        }
    }

    // Property 3: yI² + yQ² = env².
    for k in 0..ts.len() {
        let mod_sq = r.i[k] * r.i[k] + r.q[k] * r.q[k];
        let env_sq = r.envelope[k] * r.envelope[k];
        if (mod_sq - env_sq).abs() > TOL {
            panic!(
                "modulus invariant broken at t={}: yI²+yQ² = {mod_sq}, env² = {env_sq}",
                ts[k]
            );
        }
    }

    // Property 2: envelope at t=0 is exactly 1.
    let zero_idx = ts.iter().position(|&t| t == 0.0).expect("0 is in grid");
    let env0 = r.envelope[zero_idx];
    if (env0 - 1.0).abs() > TOL {
        panic!("envelope at t=0 must be 1; got {env0}");
    }

    // Property 4: envelope monotonically non-increasing in |t|.
    // Walk left half (negative t, sorted ascending — env should rise toward 0)
    // and right half (positive t, sorted ascending — env should fall away from 0).
    let mut prev_env: Option<f64> = None;
    for k in 0..=zero_idx {
        if let Some(p) = prev_env {
            // ts ascending and ≤ 0: |t_k| ≤ |t_{k-1}| ⇒ env_k ≥ env_{k-1}.
            if r.envelope[k] + TOL < p {
                panic!(
                    "envelope not non-increasing in |t| (left half): env[{k}]={} < prev={p}",
                    r.envelope[k]
                );
            }
        }
        prev_env = Some(r.envelope[k]);
    }
    let mut prev_env = r.envelope[zero_idx];
    for k in (zero_idx + 1)..ts.len() {
        // ts ascending and ≥ 0: |t_k| ≥ |t_{k-1}| ⇒ env_k ≤ env_{k-1}.
        if r.envelope[k] > prev_env + TOL {
            panic!(
                "envelope not non-increasing in |t| (right half): env[{k}]={} > prev={prev_env}",
                r.envelope[k]
            );
        }
        prev_env = r.envelope[k];
    }
});
