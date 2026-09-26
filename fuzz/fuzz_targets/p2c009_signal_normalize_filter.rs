#![no_main]

use arbitrary::Arbitrary;
use fsci_signal::normalize;
use libfuzzer_sys::fuzz_target;

// normalize (scipy.signal.normalize) property oracle for [frankenscipy-u0hsm]. This target
// fuzzed `normalize_filter`, a second normalizer that has been folded into `normalize`.
//
// Verifies:
//   1. On Ok, a_norm[0] == 1.0 exactly (monic).
//   2. Idempotence — normalize twice == once.
//   3. b_norm and a_norm are finite when b/a are finite.
//   4. All-zero a is rejected (Err).
//   5. Length invariants — len(a_norm) ≤ len(a); 1 ≤ len(b_norm) ≤ max(len(b), 1), since
//      leading numerator coefficients within 1e-14 of zero are trimmed, keeping one.

const BOUND: f64 = 1.0e6;

#[derive(Debug, Arbitrary)]
struct NormalizeFilterInput {
    b: Vec<f64>,
    a: Vec<f64>,
}

fn sanitize(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(-BOUND, BOUND)
    } else {
        0.0
    }
}

fn approx_eq(a: f64, b: f64) -> bool {
    let scale = a.abs().max(b.abs()).max(1.0);
    (a - b).abs() < 1e-9 * scale
}

fuzz_target!(|input: NormalizeFilterInput| {
    let b: Vec<f64> = input.b.iter().take(64).copied().map(sanitize).collect();
    let a: Vec<f64> = input.a.iter().take(64).copied().map(sanitize).collect();

    match normalize(&b, &a) {
        Ok((b_norm, a_norm)) => {
            // Property 1: monic.
            assert!(
                !a_norm.is_empty() && a_norm[0] == 1.0,
                "normalize must produce monic a; got {:?}",
                a_norm
            );

            // Property 5: length invariants.
            assert!(
                !b_norm.is_empty() && b_norm.len() <= b.len().max(1),
                "b may only lose leading near-zero coefficients"
            );
            assert!(a_norm.len() <= a.len(), "a length must not grow on trim");

            // Property 3: finiteness.
            for &v in &b_norm {
                assert!(v.is_finite(), "b_norm contains non-finite: {v}");
            }
            for &v in &a_norm {
                assert!(v.is_finite(), "a_norm contains non-finite: {v}");
            }

            // Property 2: idempotence.
            let (b_again, a_again) =
                normalize(&b_norm, &a_norm).expect("idempotent normalize must succeed");
            assert_eq!(b_again.len(), b_norm.len());
            assert_eq!(a_again.len(), a_norm.len());
            for (i, (&u, &v)) in b_norm.iter().zip(b_again.iter()).enumerate() {
                if !approx_eq(u, v) {
                    panic!("b idempotence broken at {i}: {u} vs {v}");
                }
            }
            for (i, (&u, &v)) in a_norm.iter().zip(a_again.iter()).enumerate() {
                if !approx_eq(u, v) {
                    panic!("a idempotence broken at {i}: {u} vs {v}");
                }
            }
        }
        Err(_) => {
            // Property 4: all-zero or empty a should yield Err. Verify the
            // converse — if Err was returned, a is empty or all zeros.
            assert!(
                a.is_empty() || a.iter().all(|&v| v == 0.0),
                "Err path should only fire on empty/all-zero a; got a = {:?}",
                a
            );
        }
    }
});
