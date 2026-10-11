//! `exp` and `log10` from +, -, *, / and bit moves only, so every platform
//! computes the same bits.
//!
//! `f64::exp` and `f64::log10` call the platform libm, whose last bit differs
//! between glibc, macOS and the rest. Where a result feeds a decision (the solve
//! policy's condition signal and softmax, the solver portfolio's rcond map and
//! decade), that last bit can flip the decision at a threshold, so the same
//! input could fail closed on one platform and solve on another. These two
//! functions are the fixed replacements.
//!
//! * [`deterministic_exp`] is FDLIBM `e_exp`: Cody-Waite `ln 2` hi/lo
//!   reduction, the P1..P5 minimax polynomial, `2^k` through the exponent
//!   bits.
//! * [`deterministic_log10`] is musl's `log10` (FreeBSD `e_log10`): the
//!   `e_log` reduction into `[sqrt(2)/2, sqrt(2))`, the Lg1..Lg7 polynomial,
//!   and a base-10 reconstruction with hi/lo splits of `1/ln 10` and
//!   `log10(2)`.
//!
//! Both are within 1 ulp of the correctly rounded result, and
//! `deterministic_log10(1ek) == k` exactly for every normal decade literal,
//! `1e-307` through `1e308`. Rust never contracts `a * b + c` into a fused
//! multiply-add, so the operation order below is the arithmetic.
//!
//! The algorithms and coefficients carry this notice:
//!
//! > Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//! >
//! > Developed at SunSoft, a Sun Microsystems, Inc. business.
//! > Permission to use, copy, modify, and distribute this
//! > software is freely granted, provided that this notice
//! > is preserved.

const HUGE: f64 = 1.0e300;
/// 2^-1000: squared, a clean underflow to zero.
const TWOM1000: f64 = f64::from_bits(0x0170_0000_0000_0000);
/// 7.09782712893383973096e+02, the largest x whose exp is finite.
const O_THRESHOLD: f64 = f64::from_bits(0x4086_2E42_FEFA_39EF);
/// -7.45133219101941108420e+02, below which exp underflows to zero.
const U_THRESHOLD: f64 = f64::from_bits(0xC087_4910_D52D_3051);
/// ln 2 high part, 6.93147180369123816490e-01 (low 20 bits zero).
const LN2_HI: f64 = f64::from_bits(0x3FE6_2E42_FEE0_0000);
/// ln 2 low part, 1.90821492927058770002e-10.
const LN2_LO: f64 = f64::from_bits(0x3DEA_39EF_3579_3C76);
/// 1/ln 2, 1.44269504088896338700e+00.
const INV_LN2: f64 = f64::from_bits(0x3FF7_1547_652B_82FE);
/// 2^54, the subnormal pre-scale for the logarithm.
const TWO54: f64 = f64::from_bits(0x4350_0000_0000_0000);
/// 2^1023, the overflow-side step of [`scale_by_power_of_two`].
const TWO1023: f64 = f64::from_bits(0x7FE0_0000_0000_0000);
/// 2^-969 = 2^-1022 * 2^53, the underflow-side step: the last multiply is the
/// only rounding, so a subnormal result is not rounded twice.
const TWOM969: f64 = f64::from_bits(0x0360_0000_0000_0000);

// e_exp's minimax polynomial.
const P1: f64 = f64::from_bits(0x3FC5_5555_5555_553E); //  1.66666666666666019037e-01
const P2: f64 = f64::from_bits(0xBF66_C16C_16BE_BD93); // -2.77777777770155933842e-03
const P3: f64 = f64::from_bits(0x3F11_566A_AF25_DE2C); //  6.61375632143793436117e-05
const P4: f64 = f64::from_bits(0xBEBB_BD41_C5D2_6BF1); // -1.65339022054652515390e-06
const P5: f64 = f64::from_bits(0x3E66_3769_72BE_A4D0); //  4.13813679705723846039e-08

// e_log's minimax polynomial.
const LG1: f64 = f64::from_bits(0x3FE5_5555_5555_5593); // 6.666666666666735130e-01
const LG2: f64 = f64::from_bits(0x3FD9_9999_9997_FA04); // 3.999999999940941908e-01
const LG3: f64 = f64::from_bits(0x3FD2_4924_9422_9359); // 2.857142874366239149e-01
const LG4: f64 = f64::from_bits(0x3FCC_71C5_1D8E_78AF); // 2.222219843214978396e-01
const LG5: f64 = f64::from_bits(0x3FC7_4664_96CB_03DE); // 1.818357216161805012e-01
const LG6: f64 = f64::from_bits(0x3FC3_9A09_D078_C69F); // 1.531383769920937332e-01
const LG7: f64 = f64::from_bits(0x3FC2_F112_DF3E_5244); // 1.479819860511658591e-01

// log10's base-10 reconstruction.
/// 1/ln 10 high part, 4.34294481878168880939e-01 (low 32 bits zero).
const IVLN10_HI: f64 = f64::from_bits(0x3FDB_CB7B_1520_0000);
/// 1/ln 10 low part, 2.50829467116452752298e-11.
const IVLN10_LO: f64 = f64::from_bits(0x3DBB_9438_CA9A_ADD5);
/// log10(2) high part, 3.01029995663611771306e-01.
const LOG10_2_HI: f64 = f64::from_bits(0x3FD3_4413_509F_6000);
/// log10(2) low part, 3.69423907715893078616e-13.
const LOG10_2_LO: f64 = f64::from_bits(0x3D59_FEF3_11F1_2B36);

fn high_word(x: f64) -> u32 {
    (x.to_bits() >> 32) as u32
}

fn low_word(x: f64) -> u32 {
    x.to_bits() as u32
}

/// Horner evaluation in a fixed order, coefficients highest degree first.
fn horner(x: f64, coefficients: &[f64]) -> f64 {
    let mut acc = 0.0;
    for &c in coefficients {
        acc = acc * x + c;
    }
    acc
}

/// `x * 2^n` with one rounding, through the exponent bits (no platform `ldexp`).
fn scale_by_power_of_two(x: f64, n: i32) -> f64 {
    let mut y = x;
    let mut n = n;
    if n > 1023 {
        y *= TWO1023;
        n -= 1023;
        if n > 1023 {
            y *= TWO1023;
            n -= 1023;
            if n > 1023 {
                n = 1023;
            }
        }
    } else if n < -1022 {
        y *= TWOM969;
        n += 969;
        if n < -1022 {
            y *= TWOM969;
            n += 969;
            if n < -1022 {
                n = -1022;
            }
        }
    }
    // 2^n is exact for n in [-1022, 1023].
    y * f64::from_bits(u64::from((0x3ff + n) as u32) << 52)
}

/// e^x, FDLIBM `e_exp`, within 1 ulp and the same bits on every platform.
///
/// `NaN` gives `NaN`, `+inf` gives `+inf`, `-inf` gives `0`, `±0` gives
/// exactly `1`; above ~709.78 the result overflows to `+inf`, below ~-745.13
/// it underflows to `+0`.
#[must_use]
pub fn deterministic_exp(x: f64) -> f64 {
    let hx0 = high_word(x);
    let negative = (hx0 >> 31) != 0;
    let hx = hx0 & 0x7fff_ffff;

    if hx >= 0x4086_2e42 {
        // |x| >= 709.78...
        if hx >= 0x7ff0_0000 {
            if ((hx & 0x000f_ffff) | low_word(x)) != 0 {
                return x + x; // NaN
            }
            return if negative { 0.0 } else { x };
        }
        if x > O_THRESHOLD {
            return HUGE * HUGE;
        }
        if x < U_THRESHOLD {
            return TWOM1000 * TWOM1000;
        }
    }

    // x = k ln 2 + r, |r| <= ln 2 / 2.
    let mut x = x;
    let mut hi = 0.0;
    let mut lo = 0.0;
    let mut k: i32 = 0;
    if hx > 0x3fd6_2e42 {
        // |x| > ln 2 / 2
        if hx < 0x3ff0_a2b2 {
            // and |x| < 1.5 ln 2
            if negative {
                hi = x + LN2_HI;
                lo = -LN2_LO;
                k = -1;
            } else {
                hi = x - LN2_HI;
                lo = LN2_LO;
                k = 1;
            }
        } else {
            let half = if negative { -0.5 } else { 0.5 };
            // Truncation toward zero after adding ±0.5: round half away.
            k = (INV_LN2 * x + half) as i32;
            let t = f64::from(k);
            hi = x - t * LN2_HI; // exact: LN2_HI's low 20 bits are zero
            lo = t * LN2_LO;
        }
        x = hi - lo;
    } else if hx < 0x3e30_0000 {
        // |x| < 2^-28
        return 1.0 + x;
    }

    let t = x * x;
    let c = x - t * horner(t, &[P5, P4, P3, P2, P1]);
    if k == 0 {
        return 1.0 - ((x * c / (c - 2.0)) - x);
    }
    let y = 1.0 - ((lo - (x * c / (2.0 - c))) - hi);
    scale_by_power_of_two(y, k)
}

/// Base-10 logarithm, musl `log10`, within 1 ulp and the same bits on every
/// platform; exactly `k` at every normal decade literal `1ek`.
///
/// `1` gives exactly `+0`, `±0` gives `-inf`, a negative number gives `NaN`,
/// `+inf` gives `+inf`, `NaN` gives `NaN`.
#[must_use]
pub fn deterministic_log10(x: f64) -> f64 {
    let mut x = x;
    let mut hx = high_word(x);
    let mut k: i32 = 0;

    if hx < 0x0010_0000 || (hx >> 31) != 0 {
        if x.to_bits() << 1 == 0 {
            return f64::NEG_INFINITY;
        }
        if (hx >> 31) != 0 {
            return f64::NAN;
        }
        // Subnormal: scale into the normal range.
        k -= 54;
        x *= TWO54;
        hx = high_word(x);
    } else if hx >= 0x7ff0_0000 {
        return x; // +inf or NaN
    } else if hx == 0x3ff0_0000 && low_word(x) == 0 {
        return 0.0;
    }

    // Reduce x into [sqrt(2)/2, sqrt(2)).
    hx = hx.wrapping_add(0x3ff0_0000 - 0x3fe6_a09e);
    k += ((hx >> 20) as i32) - 0x3ff;
    hx = (hx & 0x000f_ffff) + 0x3fe6_a09e;
    x = f64::from_bits((u64::from(hx) << 32) | u64::from(low_word(x)));

    let f = x - 1.0;
    let hfsq = 0.5 * f * f;
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let t1 = w * horner(w, &[LG6, LG4, LG2]);
    let t2 = z * horner(w, &[LG7, LG5, LG3, LG1]);
    let r = t2 + t1;

    // hi + lo = f - hfsq + s (hfsq + R) ~ ln(1 + f), f - hfsq split so the
    // cancellation near sqrt(2)^±1 loses nothing.
    let hi = f64::from_bits((f - hfsq).to_bits() & 0xffff_ffff_0000_0000);
    let lo = f - hi - hfsq + s * (hfsq + r);

    // val_hi + val_lo ~ log10(1 + f) + k log10(2).
    let mut val_hi = hi * IVLN10_HI;
    let dk = f64::from(k);
    let y = dk * LOG10_2_HI;
    let mut val_lo = dk * LOG10_2_LO + (lo + hi) * IVLN10_LO + lo * IVLN10_HI;
    let sum = y + val_hi;
    val_lo += (y - sum) + val_hi;
    val_hi = sum;
    val_lo + val_hi
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ulps_apart(a: f64, b: f64) -> u64 {
        if a == b {
            return 0;
        }
        if a.is_sign_negative() != b.is_sign_negative() {
            return u64::MAX;
        }
        a.to_bits().abs_diff(b.to_bits())
    }

    // The bits are the specification: a port that drifts by one operation
    // changes some of them. Computed by an independent transcription of the
    // same FDLIBM and musl algorithms over IEEE doubles.
    #[test]
    fn golden_bits() {
        let log10_cases: [(f64, u64); 9] = [
            (2.0, 0x3FD3_4413_509F_79FF),
            (3.0, 0x3FDE_8927_964F_D5FD),
            (0.5, 0xBFD3_4413_509F_79FF),
            (1e-12, 0xC028_0000_0000_0000),
            (7.25e-9, 0xC020_4781_C6DC_8B23),
            (0.1, 0xBFF0_0000_0000_0000),
            (123_456.789, 0x4014_5DB6_1A28_2512),
            (5e-324, 0xC074_34E6_420F_4374),
            (f64::MAX, 0x4073_4413_509F_79FF),
        ];
        for (x, bits) in log10_cases {
            assert_eq!(deterministic_log10(x).to_bits(), bits, "log10({x:e})");
        }
        let exp_cases: [(f64, u64); 10] = [
            (1.0, 0x4005_BF0A_8B14_576A),
            (-1.0, 0x3FD7_8B56_362C_EF38),
            (0.5, 0x3FFA_6129_8E1E_069C),
            (-3.75, 0x3F98_1509_354F_0D2A),
            (10.0, 0x40D5_829D_CF95_0560),
            (-20.0, 0x3E21_B486_55F3_7267),
            (1e-10, 0x3FF0_0000_0006_DF38),
            (700.0, 0x7F0D_945D_F4F8_EC8E),
            (-740.0, 0x0000_0000_0000_0055),
            (-0.0, 0x3FF0_0000_0000_0000),
        ];
        for (x, bits) in exp_cases {
            assert_eq!(deterministic_exp(x).to_bits(), bits, "exp({x:e})");
        }
    }

    // Normal decades only: a subnormal literal such as 1e-323 parses to ~9.88e-324, whose
    // log10 is not an integer.
    #[test]
    fn every_normal_decade_literal_is_exact() {
        for k in -307..=308 {
            let x: f64 = format!("1e{k}").parse().expect("decade literal");
            assert_eq!(deterministic_log10(x), f64::from(k), "log10(1e{k})");
        }
    }

    #[test]
    fn special_values() {
        assert!(deterministic_log10(f64::NAN).is_nan());
        assert!(deterministic_log10(-1.0).is_nan());
        assert!(deterministic_log10(-0.0).is_infinite());
        assert_eq!(deterministic_log10(0.0), f64::NEG_INFINITY);
        assert_eq!(deterministic_log10(f64::INFINITY), f64::INFINITY);
        assert_eq!(deterministic_log10(1.0).to_bits(), 0.0_f64.to_bits());

        assert!(deterministic_exp(f64::NAN).is_nan());
        assert_eq!(deterministic_exp(f64::INFINITY), f64::INFINITY);
        assert_eq!(deterministic_exp(f64::NEG_INFINITY).to_bits(), 0);
        assert_eq!(deterministic_exp(0.0), 1.0);
        assert_eq!(deterministic_exp(710.0), f64::INFINITY);
        assert_eq!(deterministic_exp(-746.0).to_bits(), 0);
    }

    // Against the platform libm, itself within an ulp of the true value: two
    // ulps apart at most, over a deterministic sweep of the whole range.
    #[test]
    fn agrees_with_the_platform_libm_to_two_ulps() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1_u64 << 53) as f64
        };
        for _ in 0..100_000 {
            let decade = -300.0 + 600.0 * next();
            let x = 10_f64.powf(decade);
            let platform = x.log10();
            assert!(
                ulps_apart(deterministic_log10(x), platform) <= 2,
                "log10({x:e}): {} vs {platform}",
                deterministic_log10(x)
            );
            let y = -745.0 + 1454.7 * next();
            let platform = y.exp();
            assert!(
                ulps_apart(deterministic_exp(y), platform) <= 2,
                "exp({y:e}): {} vs {platform}",
                deterministic_exp(y)
            );
        }
    }
}
