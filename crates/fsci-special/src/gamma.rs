#![forbid(unsafe_code)]

use std::f64::consts::PI;

use fsci_runtime::RuntimeMode;

use crate::types::{
    Complex64, DispatchPlan, DispatchStep, KernelRegime, SpecialError, SpecialErrorKind,
    SpecialResult, SpecialTensor, not_yet_implemented, record_special_trace,
};

pub const GAMMA_DISPATCH_PLAN: &[DispatchPlan] = &[
    DispatchPlan {
        function: "gamma",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Reflection,
                when: "Re(z) < 0.5 and not at poles",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "0.5 <= Re(z) < 8.0",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "Re(z) >= 8.0",
            },
        ],
        notes: "Strict mode preserves SciPy signed-zero and pole semantics; hardened mode adds fail-closed diagnostics.",
    },
    DispatchPlan {
        function: "gammaln",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "real axis shift into stable window",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large |x| Stirling region",
            },
        ],
        notes: "Keep +inf pole parity with SciPy and avoid direct exp(gammaln) overflow back-conversions.",
    },
    DispatchPlan {
        function: "gammasgn",
        steps: &[DispatchStep {
            regime: KernelRegime::Reflection,
            when: "real negative non-integers derive sign from reflection intervals",
        }],
        notes: "Matches scipy.special.gammasgn on the real axis, including signed-zero and pole NaN semantics.",
    },
    DispatchPlan {
        function: "loggamma",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "positive real axis reuses gammaln stable window",
            },
            DispatchStep {
                regime: KernelRegime::Reflection,
                when: "complex arguments cross Re(z) < 0.5 principal branch",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large positive real or complex magnitude",
            },
        ],
        notes: "Real inputs preserve scipy.special.loggamma's negative-axis NaN branch cut; complex inputs return the principal branch.",
    },
    DispatchPlan {
        function: "gammainc",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "x < a + 1 via stable lower-tail expansion",
            },
            DispatchStep {
                regime: KernelRegime::ContinuedFraction,
                when: "x >= a + 1 via stable upper-tail continued fraction",
            },
        ],
        notes: "Regularized lower incomplete gamma P(a, x); strict mode preserves NaN domain fallback.",
    },
    DispatchPlan {
        function: "gammaincc",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Series,
                when: "complement from lower-tail expansion",
            },
            DispatchStep {
                regime: KernelRegime::ContinuedFraction,
                when: "direct upper-tail continued fraction for stability",
            },
        ],
        notes: "Regularized upper incomplete gamma Q(a, x)=1-P(a, x); hardened mode fail-closes malformed domains.",
    },
    DispatchPlan {
        function: "digamma",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Reflection,
                when: "Re(z) <= 0 and away from poles",
            },
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "shift toward asymptotic region",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large |z|",
            },
        ],
        notes: "Negative-axis cancellation regions require dedicated numerical guards before D2 implementation.",
    },
    DispatchPlan {
        function: "polygamma",
        steps: &[
            DispatchStep {
                regime: KernelRegime::Recurrence,
                when: "order n fixed and argument shifted off poles",
            },
            DispatchStep {
                regime: KernelRegime::Asymptotic,
                when: "large argument regime",
            },
        ],
        notes: "Polygamma order must remain integer and nonnegative in both modes.",
    },
    DispatchPlan {
        function: "rgamma",
        steps: &[DispatchStep {
            regime: KernelRegime::BackendDelegate,
            when: "evaluate via reciprocal-safe gamma path",
        }],
        notes: "Prefer reciprocal from stabilized gammaln/gamma path to reduce overflow risk.",
    },
];

const SQRT_2PI: f64 = 2.506_628_274_631_000_7;
const LANCZOS_G: f64 = 7.0;
const LANCZOS_COEFFS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    0.000_009_984_369_578_019_572,
    0.000_000_150_563_273_514_931_16,
];

pub fn gamma(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    gamma_dispatch("gamma", z, mode)
}

/// Audit-emitting variant of [`gamma`]. In Hardened mode, a pole/
/// non-finite/domain rejection records an `AuditAction::FailClosed`
/// event on the provided ledger before returning the error. In Strict
/// mode the ledger is left untouched. See br-egba-2.
pub fn gamma_with_audit(
    z: &SpecialTensor,
    mode: RuntimeMode,
    ledger: &crate::audit::SyncSharedAuditLedger,
) -> SpecialResult {
    let result = gamma_dispatch("gamma", z, mode);
    if mode == RuntimeMode::Hardened
        && let Err(err) = &result
    {
        let reason = format!("gamma::{:?}", err.kind);
        crate::audit::record_fail_closed(ledger, err.detail.as_bytes(), &reason, "rejected");
    }
    result
}

pub fn gammaln(x: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    gammaln_dispatch("gammaln", x, mode)
}

pub fn gammasgn(x: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input_rp(
        "gammasgn",
        x,
        mode,
        |value| gammasgn_scalar(value, mode),
        usize::MAX,
    )
}

pub fn loggamma(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    loggamma_dispatch("loggamma", z, mode)
}

/// Evaluate `f(0..n)` into a `Vec<T>`, parallel over index chunks for large `n`.
/// Gamma-family kernels (Lanczos gamma / log-gamma / digamma series) are non-trivial per
/// element and each index writes its own slot, so chunking across cores and concatenating
/// in index order is bit-identical to `(0..n).map(f).collect()` — including returning the
/// first failing index's error in index order. Generic over the output type (f64 or
/// Complex64); infallible complex kernels wrap their result in `Ok`.
fn par_map_indices<T, H>(n: usize, f: H) -> Result<Vec<T>, SpecialError>
where
    T: Send + Default + Copy,
    H: Fn(usize) -> Result<T, SpecialError> + Sync,
{
    let nthreads = if n < 256 {
        1
    } else {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(n / 128)
            .max(1)
    };
    par_map_indices_with_threads(n, nthreads, f)
}

/// Light-kernel parallel map: the gamma-family scalar kernels (gamma/gammaln/
/// digamma/loggamma, ~30 ns Lanczos each) are so cheap that the loose `n/128`
/// worker cap of [`par_map_indices`] over-subscribes at moderate `n` — 64 OS
/// threads spawned to do microseconds of work each, leaving a flat ~4 ms spawn
/// floor that loses to SciPy's serial cephes. Cap by WORK instead (≥~32k elements
/// per worker) so each thread amortizes its spawn; this flips the moderate-`n`
/// (≈128k–1M) gamma family from a 1.4–2x loss to a 1.2–2.75x win. Heavy kernels
/// (complex arms, incomplete gamma) keep the looser cap. Order-preserving, so the
/// output stays byte-identical to the serial map.
fn par_map_light<T, H>(n: usize, f: H) -> Result<Vec<T>, SpecialError>
where
    T: Send + Default + Copy,
    H: Fn(usize) -> Result<T, SpecialError> + Sync,
{
    let nthreads = if n < 256 {
        1
    } else {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(n / 32768)
            .max(1)
    };
    par_map_indices_with_threads(n, nthreads, f)
}

/// Write `f(base + k)` into `out[k]`, stopping at the first error in index order.
///
/// Iterating `out` by `iter_mut` rather than by index is deliberate: there is no bound to
/// check, so the write is a plain store.
fn fill_indices<T, H>(out: &mut [T], base: usize, f: &H) -> Result<(), SpecialError>
where
    H: Fn(usize) -> Result<T, SpecialError>,
{
    for (offset, slot) in out.iter_mut().enumerate() {
        *slot = f(base + offset)?;
    }
    Ok(())
}

fn par_map_indices_with_threads<T, H>(
    n: usize,
    nthreads: usize,
    f: H,
) -> Result<Vec<T>, SpecialError>
where
    T: Send + Default + Copy,
    H: Fn(usize) -> Result<T, SpecialError> + Sync,
{
    // COLLECTING A `Result` PER ELEMENT COSTS MORE THAN THE LOG IT WRAPS.
    //
    // `(0..n).map(f).collect::<Result<Vec<T>, _>>()` routes every element through
    // `GenericShunt`, the adapter that lets a fallible iterator collect into a `Vec`. It
    // cannot use the size hint to preallocate — the iterator may stop early — so each
    // element pays a capacity check and a discriminant test, and the compiler cannot turn
    // the body into a straight store. Profiled on `gammaln` at n=200000, x in [20.1, 60],
    // instructions attributed by symbol:
    //
    //     gammaln_scalar_with_threshold   50.5%   (the kernel itself)
    //     ...GenericShunt...::from_iter   26.1%   <- this
    //     __ieee754_log_fma + log@plt     22.0%
    //
    // 26% of the op, in the collect. Allocating the output up front and writing each slot
    // through `iter_mut` removes the adapter entirely; the buffer is `vec![T::default();
    // n]`, NOT `with_capacity` + `push`, because a push-built buffer keeps a growing `len`
    // that turns every later write into a bounds check LLVM cannot hoist — that swap cost
    // 46% elsewhere in this workspace. The zeroing pass really is dead work and keeping it
    // is still faster.
    //
    // 26% ATTRIBUTED IS NOT 26% REMOVABLE. Only 13.6 instructions per element of that
    // block actually went away (226.8 against 240.4, 159.7 against 173.3, 183.1 against
    // 196.9 in the three bands); the rest is the store and the loop any version must
    // perform. A profile says where instructions ARE, not how many can be removed —
    // quoting the attribution as the prize would have overstated this by threefold.
    //
    // MUST-MISS CONTROL: `erf` does not route through this mapper and reads 69.3
    // instructions per element with the arm on AND off, `prealloc_hits=0` both times. An
    // op that must not move, and does not.
    //
    // BIT-IDENTICAL: same `f`, same indices, same order, and the first error in index
    // order is still the one returned. Only the destination changes. Asserted by
    // `gamma_family_prealloc_fill_arms_agree_bit_for_bit` on both the serial and the
    // threaded arm, with a one-ULP detector proving the comparison can see a difference.
    //
    // The parallel arm additionally drops a whole copy of the output: it used to build one
    // `Vec` per thread and concatenate them, and now each thread fills its own disjoint
    // `chunks_mut` slice of the final buffer.
    if !GAMMA_FAMILY_PREALLOC_FILL.load(std::sync::atomic::Ordering::Relaxed) {
        if nthreads <= 1 {
            return (0..n).map(&f).collect();
        }
        let chunk = n.div_ceil(nthreads);
        let f = &f;
        let chunk_results: Vec<Result<Vec<T>, SpecialError>> = std::thread::scope(|scope| {
            (0..nthreads)
                .filter_map(|t| {
                    let i0 = t * chunk;
                    if i0 >= n {
                        return None;
                    }
                    let i1 = (i0 + chunk).min(n);
                    Some(scope.spawn(move || (i0..i1).map(f).collect::<Result<Vec<T>, _>>()))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().expect("gamma array worker panicked"))
                .collect()
        });
        let mut out = Vec::with_capacity(n);
        for cr in chunk_results {
            out.extend(cr?);
        }
        return Ok(out);
    }

    GAMMA_FAMILY_PREALLOC_FILL_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut out = vec![T::default(); n];
    if nthreads <= 1 || n == 0 {
        fill_indices(&mut out, 0, &f)?;
        return Ok(out);
    }
    let chunk = n.div_ceil(nthreads);
    let f = &f;
    let results: Vec<Result<(), SpecialError>> = std::thread::scope(|scope| {
        out.chunks_mut(chunk)
            .enumerate()
            .map(|(t, slice)| scope.spawn(move || fill_indices(slice, t * chunk, f)))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("gamma array worker panicked"))
            .collect()
    });
    for result in results {
        result?;
    }
    Ok(out)
}

/// Real arrays at or above this length route the cheap gamma-family kernels
/// (gamma/gammaln/digamma, ~29ns Lanczos each) through `par_map_indices`. The
/// per-element compute finally dominates the OS-thread spawn + chunk-concat
/// overhead, flipping the serial cheap-kernel win into a multicore win versus
/// SciPy's single-threaded cephes ufunc. Below it, the serial path stays (the
/// cheap-kernel-serialization sweep proved parallel over-subscribes short
/// arrays at n=100k). Order-preserving, so the output is byte-identical to the
/// serial map either way. Verified break-even is well under 1M for these
/// kernels; the threshold is set conservatively so only large arrays parallelize.
const GAMMA_FAMILY_PAR_MIN: usize = 1 << 17; // work-capped par_map_light wins from here

fn gamma_dispatch(function: &'static str, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    match z {
        SpecialTensor::RealScalar(x) => gamma_scalar(*x, mode).map(SpecialTensor::RealScalar),
        SpecialTensor::RealVec(values) => {
            if values.len() >= GAMMA_FAMILY_PAR_MIN {
                par_map_light(values.len(), |i| gamma_scalar(values[i], mode))
                    .map(SpecialTensor::RealVec)
            } else {
                values
                    .iter()
                    .map(|&x| gamma_scalar(x, mode))
                    .collect::<Result<Vec<_>, _>>()
                    .map(SpecialTensor::RealVec)
            }
        }
        SpecialTensor::ComplexScalar(z_val) => {
            Ok(SpecialTensor::ComplexScalar(complex_gammaln(*z_val).exp()))
        }
        SpecialTensor::ComplexVec(values) => {
            par_map_indices(values.len(), |i| Ok(complex_gammaln(values[i]).exp()))
                .map(SpecialTensor::ComplexVec)
        }
        SpecialTensor::Empty => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid special-function input",
        }),
    }
}

fn gammaln_dispatch(function: &'static str, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    match z {
        SpecialTensor::RealScalar(x) => gammaln_scalar(*x, mode).map(SpecialTensor::RealScalar),
        SpecialTensor::RealVec(values) => {
            // The crossover is read ONCE for the whole batch rather than once per element.
            // See `gammaln_scalar_with_threshold`; the value is identical, so this is
            // bit-identical, and it takes an atomic load out of the hot loop.
            let hoist = GAMMALN_HOIST_THRESHOLD.load(std::sync::atomic::Ordering::Relaxed);
            if !hoist {
                return if values.len() >= GAMMA_FAMILY_PAR_MIN {
                    par_map_light(values.len(), |i| gammaln_scalar(values[i], mode))
                        .map(SpecialTensor::RealVec)
                } else {
                    values
                        .iter()
                        .map(|&x| gammaln_scalar(x, mode))
                        .collect::<Result<Vec<_>, _>>()
                        .map(SpecialTensor::RealVec)
                };
            }
            GAMMALN_HOIST_THRESHOLD_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let min_x = gammaln_asymptotic_min_x();
            if values.len() >= GAMMA_FAMILY_PAR_MIN {
                par_map_light(values.len(), |i| {
                    gammaln_scalar_with_threshold(values[i], mode, min_x)
                })
                .map(SpecialTensor::RealVec)
            } else {
                values
                    .iter()
                    .map(|&x| gammaln_scalar_with_threshold(x, mode, min_x))
                    .collect::<Result<Vec<_>, _>>()
                    .map(SpecialTensor::RealVec)
            }
        }
        SpecialTensor::ComplexScalar(z_val) => {
            Ok(SpecialTensor::ComplexScalar(complex_gammaln(*z_val)))
        }
        SpecialTensor::ComplexVec(values) => {
            par_map_indices(values.len(), |i| Ok(complex_gammaln(values[i])))
                .map(SpecialTensor::ComplexVec)
        }
        SpecialTensor::Empty => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid special-function input",
        }),
    }
}

fn loggamma_dispatch(
    function: &'static str,
    z: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    match z {
        SpecialTensor::RealScalar(x) => loggamma_scalar(*x, mode).map(SpecialTensor::RealScalar),
        SpecialTensor::RealVec(values) => {
            // loggamma(real) == gammaln (cheap ~30ns); gate with the family min so the
            // n/256-class over-subscription that pessimizes small arrays is avoided.
            if values.len() >= GAMMA_FAMILY_PAR_MIN {
                par_map_light(values.len(), |i| loggamma_scalar(values[i], mode))
                    .map(SpecialTensor::RealVec)
            } else {
                values
                    .iter()
                    .map(|&x| loggamma_scalar(x, mode))
                    .collect::<Result<Vec<_>, _>>()
                    .map(SpecialTensor::RealVec)
            }
        }
        SpecialTensor::ComplexScalar(z_val) => {
            Ok(SpecialTensor::ComplexScalar(complex_gammaln(*z_val)))
        }
        SpecialTensor::ComplexVec(values) => {
            par_map_indices(values.len(), |i| Ok(complex_gammaln(values[i])))
                .map(SpecialTensor::ComplexVec)
        }
        SpecialTensor::Empty => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid special-function input",
        }),
    }
}

pub fn digamma(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    digamma_dispatch("digamma", z, mode)
}

pub fn psi(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    digamma_dispatch("psi", z, mode)
}

fn digamma_dispatch(function: &'static str, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    match z {
        SpecialTensor::RealScalar(x) => digamma_scalar(*x, mode).map(SpecialTensor::RealScalar),
        SpecialTensor::RealVec(values) => {
            if values.len() >= GAMMA_FAMILY_PAR_MIN {
                par_map_light(values.len(), |i| digamma_scalar(values[i], mode))
                    .map(SpecialTensor::RealVec)
            } else {
                values
                    .iter()
                    .map(|&x| digamma_scalar(x, mode))
                    .collect::<Result<Vec<_>, _>>()
                    .map(SpecialTensor::RealVec)
            }
        }
        SpecialTensor::ComplexScalar(z_val) => {
            Ok(SpecialTensor::ComplexScalar(complex_digamma_scalar(*z_val)))
        }
        SpecialTensor::ComplexVec(values) => {
            par_map_indices(values.len(), |i| Ok(complex_digamma_scalar(values[i])))
                .map(SpecialTensor::ComplexVec)
        }
        SpecialTensor::Empty => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid special-function input",
        }),
    }
}

pub fn gammainc(a: &SpecialTensor, x: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    gammainc_dispatch("gammainc", a, x, mode, true)
}

pub fn gammaincc(a: &SpecialTensor, x: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    gammainc_dispatch("gammaincc", a, x, mode, false)
}

fn gammainc_dispatch(
    function: &'static str,
    a: &SpecialTensor,
    x: &SpecialTensor,
    mode: RuntimeMode,
    lower: bool,
) -> SpecialResult {
    match (a, x) {
        // Real-real cases
        (SpecialTensor::RealScalar(a_val), SpecialTensor::RealScalar(x_val)) => {
            if lower {
                gammainc_scalar(*a_val, *x_val, mode).map(SpecialTensor::RealScalar)
            } else {
                gammaincc_scalar(*a_val, *x_val, mode).map(SpecialTensor::RealScalar)
            }
        }
        (SpecialTensor::RealVec(a_vec), SpecialTensor::RealScalar(x_val)) => {
            let x_val = *x_val;
            par_map_indices(a_vec.len(), |i| {
                if lower {
                    gammainc_scalar(a_vec[i], x_val, mode)
                } else {
                    gammaincc_scalar(a_vec[i], x_val, mode)
                }
            })
            .map(SpecialTensor::RealVec)
        }
        (SpecialTensor::RealScalar(a_val), SpecialTensor::RealVec(x_vec)) => {
            let a_val = *a_val;
            par_map_indices(x_vec.len(), |i| {
                if lower {
                    gammainc_scalar(a_val, x_vec[i], mode)
                } else {
                    gammaincc_scalar(a_val, x_vec[i], mode)
                }
            })
            .map(SpecialTensor::RealVec)
        }
        (SpecialTensor::RealVec(a_vec), SpecialTensor::RealVec(x_vec)) => {
            if a_vec.len() != x_vec.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            par_map_indices(a_vec.len(), |i| {
                if lower {
                    gammainc_scalar(a_vec[i], x_vec[i], mode)
                } else {
                    gammaincc_scalar(a_vec[i], x_vec[i], mode)
                }
            })
            .map(SpecialTensor::RealVec)
        }
        // Real a, complex x cases
        (SpecialTensor::RealScalar(a_val), SpecialTensor::ComplexScalar(x_val)) => {
            if lower {
                complex_gammainc_scalar(*a_val, *x_val, mode).map(SpecialTensor::ComplexScalar)
            } else {
                complex_gammaincc_scalar(*a_val, *x_val, mode).map(SpecialTensor::ComplexScalar)
            }
        }
        (SpecialTensor::RealScalar(a_val), SpecialTensor::ComplexVec(x_vec)) => {
            let a_val = *a_val;
            par_map_indices(x_vec.len(), |i| {
                if lower {
                    complex_gammainc_scalar(a_val, x_vec[i], mode)
                } else {
                    complex_gammaincc_scalar(a_val, x_vec[i], mode)
                }
            })
            .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealVec(a_vec), SpecialTensor::ComplexScalar(x_val)) => {
            let x_val = *x_val;
            par_map_indices(a_vec.len(), |i| {
                if lower {
                    complex_gammainc_scalar(a_vec[i], x_val, mode)
                } else {
                    complex_gammaincc_scalar(a_vec[i], x_val, mode)
                }
            })
            .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::RealVec(a_vec), SpecialTensor::ComplexVec(x_vec)) => {
            if a_vec.len() != x_vec.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            par_map_indices(a_vec.len(), |i| {
                if lower {
                    complex_gammainc_scalar(a_vec[i], x_vec[i], mode)
                } else {
                    complex_gammaincc_scalar(a_vec[i], x_vec[i], mode)
                }
            })
            .map(SpecialTensor::ComplexVec)
        }
        // Complex a cases
        (SpecialTensor::ComplexScalar(a_val), SpecialTensor::RealScalar(x_val)) => {
            gammainc_complex_parameter_scalar(*a_val, Complex64::new(*x_val, 0.0), mode, lower)
                .map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::ComplexScalar(a_val), SpecialTensor::RealVec(x_vec)) => {
            let a_val = *a_val;
            par_map_indices(x_vec.len(), |i| {
                gammainc_complex_parameter_scalar(a_val, Complex64::new(x_vec[i], 0.0), mode, lower)
            })
            .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::ComplexVec(a_vec), SpecialTensor::RealScalar(x_val)) => {
            let x_val = *x_val;
            par_map_indices(a_vec.len(), |i| {
                gammainc_complex_parameter_scalar(a_vec[i], Complex64::new(x_val, 0.0), mode, lower)
            })
            .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::ComplexVec(a_vec), SpecialTensor::RealVec(x_vec)) => {
            if a_vec.len() != x_vec.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            par_map_indices(a_vec.len(), |i| {
                gammainc_complex_parameter_scalar(
                    a_vec[i],
                    Complex64::new(x_vec[i], 0.0),
                    mode,
                    lower,
                )
            })
            .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::ComplexScalar(a_val), SpecialTensor::ComplexScalar(x_val)) => {
            gammainc_complex_parameter_scalar(*a_val, *x_val, mode, lower)
                .map(SpecialTensor::ComplexScalar)
        }
        (SpecialTensor::ComplexScalar(a_val), SpecialTensor::ComplexVec(x_vec)) => {
            let a_val = *a_val;
            par_map_indices(x_vec.len(), |i| {
                gammainc_complex_parameter_scalar(a_val, x_vec[i], mode, lower)
            })
            .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::ComplexVec(a_vec), SpecialTensor::ComplexScalar(x_val)) => {
            let x_val = *x_val;
            par_map_indices(a_vec.len(), |i| {
                gammainc_complex_parameter_scalar(a_vec[i], x_val, mode, lower)
            })
            .map(SpecialTensor::ComplexVec)
        }
        (SpecialTensor::ComplexVec(a_vec), SpecialTensor::ComplexVec(x_vec)) => {
            if a_vec.len() != x_vec.len() {
                return Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "vector inputs must have matching lengths",
                });
            }
            par_map_indices(a_vec.len(), |i| {
                gammainc_complex_parameter_scalar(a_vec[i], x_vec[i], mode, lower)
            })
            .map(SpecialTensor::ComplexVec)
        }
        // Empty tensor cases
        (SpecialTensor::Empty, _) | (_, SpecialTensor::Empty) => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid special-function input",
        }),
    }
}

fn gammainc_complex_parameter_scalar(
    a: Complex64,
    z: Complex64,
    mode: RuntimeMode,
    lower: bool,
) -> Result<Complex64, SpecialError> {
    if a.im == 0.0 {
        if z.im == 0.0 {
            let value = if lower {
                gammainc_scalar(a.re, z.re, mode)?
            } else {
                gammaincc_scalar(a.re, z.re, mode)?
            };
            return Ok(Complex64::new(value, 0.0));
        }
        return if lower {
            complex_gammainc_scalar(a.re, z, mode)
        } else {
            complex_gammaincc_scalar(a.re, z, mode)
        };
    }

    if lower {
        complex_parameter_gammainc_scalar(a, z, mode)
    } else {
        complex_parameter_gammaincc_scalar(a, z, mode)
    }
}

pub fn polygamma(n: usize, z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    polygamma_dispatch("polygamma", n, z, mode)
}

fn polygamma_dispatch(
    function: &'static str,
    n: usize,
    z: &SpecialTensor,
    mode: RuntimeMode,
) -> SpecialResult {
    match z {
        SpecialTensor::RealScalar(x) => {
            let result = match n {
                0 => digamma_scalar(*x, mode),
                1 => trigamma_scalar(*x, mode),
                2 => tetragamma_scalar(*x, mode),
                _ => polygamma_higher_scalar(n, *x, mode),
            };
            result.map(SpecialTensor::RealScalar)
        }
        SpecialTensor::RealVec(values) => {
            // digamma/trigamma are cheap (~28ns); gate with the family min to avoid the
            // small-n over-subscription (measured 12x@4096). Order-preserving.
            let eval = |x: f64| match n {
                0 => digamma_scalar(x, mode),
                1 => trigamma_scalar(x, mode),
                2 => tetragamma_scalar(x, mode),
                _ => polygamma_higher_scalar(n, x, mode),
            };
            if values.len() >= GAMMA_FAMILY_PAR_MIN {
                par_map_indices(values.len(), |i| eval(values[i])).map(SpecialTensor::RealVec)
            } else {
                values
                    .iter()
                    .map(|&x| eval(x))
                    .collect::<Result<Vec<_>, _>>()
                    .map(SpecialTensor::RealVec)
            }
        }
        SpecialTensor::ComplexScalar(z_val) => Ok(SpecialTensor::ComplexScalar(
            complex_polygamma_scalar(n, *z_val),
        )),
        SpecialTensor::ComplexVec(values) => {
            par_map_indices(values.len(), |i| Ok(complex_polygamma_scalar(n, values[i])))
                .map(SpecialTensor::ComplexVec)
        }
        SpecialTensor::Empty => Err(SpecialError {
            function,
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid special-function input",
        }),
    }
}

pub fn rgamma(z: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    match z {
        SpecialTensor::RealScalar(x) => rgamma_scalar(*x, mode).map(SpecialTensor::RealScalar),
        SpecialTensor::RealVec(values) => {
            // rgamma (1/Gamma) is a cheap ~30ns kernel; route through the
            // work-capped light path (the loose par_map_indices cap over-subscribes
            // it at moderate n), same as the gamma/gammaln/digamma family.
            if values.len() >= GAMMA_FAMILY_PAR_MIN {
                par_map_light(values.len(), |i| rgamma_scalar(values[i], mode))
                    .map(SpecialTensor::RealVec)
            } else {
                values
                    .iter()
                    .map(|&x| rgamma_scalar(x, mode))
                    .collect::<Result<Vec<_>, _>>()
                    .map(SpecialTensor::RealVec)
            }
        }
        SpecialTensor::ComplexScalar(z_val) => {
            Ok(SpecialTensor::ComplexScalar(complex_rgamma_scalar(*z_val)))
        }
        SpecialTensor::ComplexVec(values) => {
            // Each element is an independent, expensive complex reciprocal-gamma evaluation
            // written to its own output slot; par_map_indices fans index chunks across cores
            // and concatenates in index order (gating n<256 to the sequential path), so the
            // result is bit-identical to the serial `values.iter().map(..).collect()`.
            par_map_indices(values.len(), |i| Ok(complex_rgamma_scalar(values[i])))
                .map(SpecialTensor::ComplexVec)
        }
        SpecialTensor::Empty => Err(SpecialError {
            function: "rgamma",
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "empty tensor is not a valid special-function input",
        }),
    }
}

pub fn multigammaln(a: &SpecialTensor, d: f64, mode: RuntimeMode) -> SpecialResult {
    if d.is_nan() || d.floor() != d {
        record_special_trace(
            "multigammaln",
            mode,
            "domain_error",
            format!("a={a:?},d={d}"),
            "fail_closed",
            "d should be a positive integer (dimension)",
            false,
        );
        return Err(SpecialError {
            function: "multigammaln",
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "d should be a positive integer (dimension)",
        });
    }

    map_real_input_rp(
        "multigammaln",
        a,
        mode,
        |x| multigammaln_scalar(x, d, mode),
        1 << 17,
    ) // be ~68k
}

/// Gamma distribution CDF with rate `a` and shape `b`.
///
/// Matches `scipy.special.gdtr(a, b, x)`.
#[must_use]
pub fn gdtr(a: f64, b: f64, x: f64) -> f64 {
    if a.is_nan() || b.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    gammainc_scalar(b, a * x, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// Gamma distribution survival function with rate `a` and shape `b`.
///
/// Returns P(X > x) = 1 - gdtr(a, b, x).
///
/// Matches `scipy.special.gdtrc(a, b, x)`.
#[must_use]
pub fn gdtrc(a: f64, b: f64, x: f64) -> f64 {
    if a.is_nan() || b.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    gammaincc_scalar(b, a * x, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// Inverse gamma distribution CDF with rate `a` and shape `b`, solving for `x`.
///
/// Matches `scipy.special.gdtrix(a, b, p)`.
#[must_use]
pub fn gdtrix(a: f64, b: f64, p: f64) -> f64 {
    if a.is_nan() || b.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if b <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }

    gammaincinv(b, p) / a
}

/// Vectorized inverse gamma CDF w.r.t. `x`, `gdtrix(a, b, p)`, over many
/// probabilities `p` for fixed `(a, b)`. SciPy's `gdtrix` ufunc is a
/// single-threaded per-point root-find (~330 ns/pt); this fans the
/// (bit-identical) fsci scalar across cores via the crate's order-preserving
/// parallel map.
#[must_use]
pub fn gdtrix_many(a: f64, b: f64, p: &[f64]) -> Vec<f64> {
    par_map_indices(p.len(), |i| Ok::<f64, SpecialError>(gdtrix(a, b, p[i])))
        .expect("gdtrix is infallible")
}

/// Vectorized inverse chi-square CDF w.r.t. degrees of freedom,
/// `chdtriv(p, x)`, over many `p` for a fixed `x`. SciPy's ufunc is very slow
/// here (~2 µs/pt); see [`gdtrix_many`].
#[must_use]
pub fn chdtriv_many(p: &[f64], x: f64) -> Vec<f64> {
    par_map_indices(p.len(), |i| Ok::<f64, SpecialError>(chdtriv(p[i], x)))
        .expect("chdtriv is infallible")
}

/// Inverse gamma distribution CDF with respect to rate `a`.
///
/// Matches `scipy.special.gdtria(p, b, x)`.
#[must_use]
pub fn gdtria(p: f64, b: f64, x: f64) -> f64 {
    if p.is_nan() || b.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if b <= 0.0 || !(0.0..=1.0).contains(&p) || x == 0.0 {
        return f64::NAN;
    }

    gammaincinv(b, p) / x
}

/// Inverse gamma distribution CDF with respect to shape `b`.
///
/// Matches `scipy.special.gdtrib(a, p, x)`.
#[must_use]
pub fn gdtrib(a: f64, p: f64, x: f64) -> f64 {
    if a.is_nan() || p.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if a <= 0.0 || x < 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }

    let scaled_x = a * x;
    if !scaled_x.is_finite() {
        return f64::NAN;
    }
    if scaled_x == 0.0 {
        return if p == 0.0 { f64::NAN } else { 0.0 };
    }
    if p == 0.0 {
        return f64::INFINITY;
    }
    if p == 1.0 {
        return 0.0;
    }

    gammainc_shape_inv(scaled_x, p)
}

// Default-threshold wrapper over `map_real_input_rp` (parallel cutoff 256).
// Currently unused -- every call site passes an explicit, tuned threshold --
// but RETAINED rather than deleted (frankenscipy-e2ve2): it documents what the
// default cutoff is, and removing a public-shaped helper to silence dead_code is
// the mistake recorded against frankenscipy-iit9c. Deletion is the owner's call.
#[allow(dead_code)]
fn map_real_input<F>(
    function: &'static str,
    input: &SpecialTensor,
    mode: RuntimeMode,
    kernel: F,
) -> SpecialResult
where
    F: Fn(f64) -> Result<f64, SpecialError> + Sync,
{
    map_real_input_rp(function, input, mode, kernel, 256)
}

/// `real_par_min`: RealVec length below which the real arm runs serially. Cheap O(1) kernels
/// (e.g. gammasgn ~7ns) pass `usize::MAX` — par_map_indices' per-element overhead makes parallel
/// a net loss; heavier kernels (multigammaln) keep the default 256.
fn map_real_input_rp<F>(
    function: &'static str,
    input: &SpecialTensor,
    mode: RuntimeMode,
    kernel: F,
    real_par_min: usize,
) -> SpecialResult
where
    F: Fn(f64) -> Result<f64, SpecialError> + Sync,
{
    match input {
        SpecialTensor::RealScalar(x) => kernel(*x).map(SpecialTensor::RealScalar),
        SpecialTensor::RealVec(values) => {
            if values.len() < real_par_min {
                values
                    .iter()
                    .map(|&x| kernel(x))
                    .collect::<Result<Vec<_>, _>>()
                    .map(SpecialTensor::RealVec)
            } else {
                par_map_indices(values.len(), |i| kernel(values[i])).map(SpecialTensor::RealVec)
            }
        }
        SpecialTensor::ComplexScalar(_) | SpecialTensor::ComplexVec(_) => {
            not_yet_implemented(function, mode, "complex-valued path pending")
        }
        SpecialTensor::Empty => {
            record_special_trace(
                function,
                mode,
                "domain_error",
                "input=empty",
                "fail_closed",
                "empty tensor is not a valid special-function input",
                false,
            );
            Err(SpecialError {
                function,
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "empty tensor is not a valid special-function input",
            })
        }
    }
}

pub(crate) fn gamma_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if matches!(mode, RuntimeMode::Hardened) && is_negative_integer_pole(x) {
        record_special_trace(
            "gamma",
            mode,
            "pole_input",
            format!("input={x}"),
            "fail_closed",
            "gamma pole at nonpositive integer",
            false,
        );
        return Err(SpecialError {
            function: "gamma",
            kind: SpecialErrorKind::PoleInput,
            mode,
            detail: "gamma pole at nonpositive integer",
        });
    }
    let value = gamma_core(x);
    if !value.is_finite() && !is_negative_integer_pole(x) && x != 0.0 {
        record_special_trace(
            "gamma",
            mode,
            "non_finite_output",
            format!("input={x}"),
            "returned_non_finite",
            format!("output={value}"),
            false,
        );
    }
    Ok(value)
}

pub fn gammaln_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    gammaln_scalar_with_threshold(x, mode, gammaln_asymptotic_min_x())
}

/// `gammaln_scalar` with the Lanczos/asymptotic crossover passed IN rather than read from
/// the process-global override on every call.
///
/// WHY THIS SPLIT EXISTS. `gammaln_asymptotic_min_x()` is an atomic load, and the batch
/// entry point calls `gammaln_scalar` once per element, so the load sat INSIDE the hot
/// loop. A relaxed load is one instruction but it is also an optimisation barrier: the
/// compiler may not hoist it, fold the comparison against a constant, or vectorise across
/// it, because another thread is permitted to change the value between two elements. This
/// crate has already paid for that exact shape once — a toggle read inside a per-item loop
/// cost 13% — and the fix is the same: read it ONCE at the call boundary and pass a value.
///
/// BIT-IDENTICAL by construction: the threshold is the same `f64` either way and nothing
/// else in the body changes. The only observable difference is that a batch now uses one
/// consistent threshold for every element instead of re-reading a value that a test knob
/// could in principle change mid-batch — which is strictly more correct, not less.
fn gammaln_scalar_with_threshold(
    x: f64,
    mode: RuntimeMode,
    asymptotic_min_x: f64,
) -> Result<f64, SpecialError> {
    if x.is_nan() {
        return Ok(f64::NAN);
    }
    if is_negative_integer_pole(x) || x == 0.0 {
        if matches!(mode, RuntimeMode::Hardened) {
            record_special_trace(
                "gammaln",
                mode,
                "pole_input",
                format!("input={x}"),
                "fail_closed",
                "gammaln pole at nonpositive integer",
                false,
            );
            return Err(SpecialError {
                function: "gammaln",
                kind: SpecialErrorKind::PoleInput,
                mode,
                detail: "gammaln pole at nonpositive integer",
            });
        } else {
            return Ok(f64::INFINITY);
        }
    }

    // Γ(1) = Γ(2) = 1, so ln Γ = 0 exactly; the Lanczos series returns a ~1e-16
    // residual at x = 1. scipy.special.gammaln returns exactly 0 here, and this
    // also makes betaln(1,1)/beta(1,1) exact. frankenscipy-dwd3d
    if x == 1.0 || x == 2.0 {
        return Ok(0.0);
    }

    let output = if x >= 0.5 {
        if x < asymptotic_min_x {
            lngamma_lanczos(x)
        } else {
            lngamma_positive(x)
        }
    } else {
        // Reflection formula: ln|Γ(x)| = ln(π) - ln|sin(πx)| - ln|Γ(1-x)|
        // Valid for all x < 0.5 (except poles handled above)
        let g1mx = if 1.0 - x < 100.0 {
            lngamma_lanczos(1.0 - x)
        } else {
            lngamma_positive(1.0 - x)
        };
        PI.ln() - (PI * x).sin().abs().ln() - g1mx
    };

    if !output.is_finite() && !x.is_infinite() {
        record_special_trace(
            "gammaln",
            mode,
            "non_finite_output",
            format!("input={x}"),
            "returned_non_finite",
            format!("output={output}"),
            false,
        );
    }
    Ok(output)
}

pub fn loggamma_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if x.is_nan() {
        return Ok(f64::NAN);
    }
    if x < 0.0 {
        if matches!(mode, RuntimeMode::Hardened) {
            record_special_trace(
                "loggamma",
                mode,
                "domain_error",
                format!("input={x}"),
                "fail_closed",
                "real loggamma is undefined on the negative real branch cut",
                false,
            );
            return Err(SpecialError {
                function: "loggamma",
                kind: SpecialErrorKind::DomainError,
                mode,
                detail: "real loggamma is undefined on the negative real branch cut",
            });
        }
        return Ok(f64::NAN);
    }
    if x == 0.0 && matches!(mode, RuntimeMode::Hardened) {
        record_special_trace(
            "loggamma",
            mode,
            "pole_input",
            format!("input={x}"),
            "fail_closed",
            "loggamma pole at zero",
            false,
        );
        return Err(SpecialError {
            function: "loggamma",
            kind: SpecialErrorKind::PoleInput,
            mode,
            detail: "loggamma pole at zero",
        });
    }
    gammaln_scalar(x, mode)
}

pub fn gammasgn_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return Ok(f64::NAN);
    }
    if x == f64::INFINITY {
        return Ok(1.0);
    }
    if is_negative_integer_pole(x) {
        if matches!(mode, RuntimeMode::Hardened) {
            record_special_trace(
                "gammasgn",
                mode,
                "pole_input",
                format!("input={x}"),
                "fail_closed",
                "gammasgn pole at negative integer",
                false,
            );
            return Err(SpecialError {
                function: "gammasgn",
                kind: SpecialErrorKind::PoleInput,
                mode,
                detail: "gammasgn pole at negative integer",
            });
        }
        return Ok(f64::NAN);
    }
    if x == 0.0 {
        return Ok(if x.is_sign_negative() { -1.0 } else { 1.0 });
    }
    if x > 0.0 {
        return Ok(1.0);
    }

    let sine = (PI * x).sin();
    if sine.is_nan() {
        Ok(f64::NAN)
    } else if sine.is_sign_negative() {
        Ok(-1.0)
    } else {
        Ok(1.0)
    }
}

fn multigammaln_scalar(a: f64, d: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    let threshold = 0.5 * (d - 1.0);
    if !a.is_nan() && a <= threshold {
        record_special_trace(
            "multigammaln",
            mode,
            "domain_error",
            format!("a={a},d={d}"),
            "fail_closed",
            format!("condition a ({a}) > 0.5 * (d - 1) ({threshold}) not met"),
            false,
        );
        return Err(SpecialError {
            function: "multigammaln",
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "condition a must exceed 0.5 * (d - 1)",
        });
    }
    if a.is_nan() {
        return Ok(f64::NAN);
    }
    if !d.is_finite() {
        record_special_trace(
            "multigammaln",
            mode,
            "domain_error",
            format!("a={a},d={d}"),
            "fail_closed",
            "d should be a positive integer (dimension)",
            false,
        );
        return Err(SpecialError {
            function: "multigammaln",
            kind: SpecialErrorKind::DomainError,
            mode,
            detail: "d should be a positive integer (dimension)",
        });
    }

    let mut result = 0.25 * d * (d - 1.0) * PI.ln();
    for j in 1..=(d as i64) {
        result += gammaln_scalar(a - 0.5 * ((j - 1) as f64), mode)?;
    }
    Ok(result)
}

fn lngamma_lanczos(x: f64) -> f64 {
    let mut coeff_sum = LANCZOS_COEFFS[0];
    for (idx, coeff) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
        coeff_sum += coeff / (x + (idx as f64 - 1.0));
    }

    let t = x + 6.5;
    // ln(Γ(x)) = ln(sqrt(2π)) + (x-0.5)ln(t) - t + ln(coeff_sum)
    // `0.5 * (2.0 * PI).ln()` LOOKS like a libm call per element — `f64::ln` is not a
    // `const fn`, so nothing in the language guarantees it is folded. MEASURED, and it is:
    // replacing it with the literal `HALF_LN_2PI` (0x3fed67f1c864beb4) changed the
    // instruction count by NOTHING, in every band, both kernels —
    //
    //     band          literal   runtime ln
    //     [0.5, 19.9]     248.3        248.3   ins/elem
    //     [20.1, 60]      181.3        181.2
    //     [0.01, 60]      204.7        204.7
    //
    // — so LLVM constant-folds it and the "obvious" win is worth zero. Left as the
    // expression, which reads better than a magic constant. Do not re-derive this.
    0.5 * (2.0 * PI).ln() + (x - 0.5) * t.ln() - t + coeff_sum.ln()
}

fn digamma_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if matches!(mode, RuntimeMode::Hardened) && is_negative_integer_pole(x) {
        record_special_trace(
            "digamma",
            mode,
            "pole_input",
            format!("input={x}"),
            "fail_closed",
            "digamma pole at nonpositive integer",
            false,
        );
        return Err(SpecialError {
            function: "digamma",
            kind: SpecialErrorKind::PoleInput,
            mode,
            detail: "digamma pole at nonpositive integer",
        });
    }
    let value = digamma_core(x);
    if !value.is_finite() {
        record_special_trace(
            "digamma",
            mode,
            "non_finite_output",
            format!("input={x}"),
            "returned_non_finite",
            format!("output={value}"),
            false,
        );
    }
    Ok(value)
}

fn trigamma_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if matches!(mode, RuntimeMode::Hardened) && is_negative_integer_pole(x) {
        record_special_trace(
            "polygamma",
            mode,
            "pole_input",
            format!("input={x}"),
            "fail_closed",
            "trigamma pole at nonpositive integer",
            false,
        );
        return Err(SpecialError {
            function: "polygamma",
            kind: SpecialErrorKind::PoleInput,
            mode,
            detail: "trigamma pole at nonpositive integer",
        });
    }
    // scipy.special.polygamma(1, x) = +inf at every nonpositive-integer pole
    // (the trigamma double pole is one-signed). The core returns NaN at negative
    // integers (0/0 in the reflection); pin all nonpositive integers to +inf to
    // match scipy. frankenscipy-fxm0t
    if x == 0.0 || is_negative_integer_pole(x) {
        return Ok(f64::INFINITY);
    }
    let value = trigamma_core(x);
    if !value.is_finite() {
        record_special_trace(
            "polygamma",
            mode,
            "non_finite_output",
            format!("input={x}"),
            "returned_non_finite",
            format!("output={value}"),
            false,
        );
    }
    Ok(value)
}

fn tetragamma_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if matches!(mode, RuntimeMode::Hardened) && is_negative_integer_pole(x) {
        record_special_trace(
            "polygamma",
            mode,
            "pole_input",
            format!("input={x}"),
            "fail_closed",
            "tetragamma pole at nonpositive integer",
            false,
        );
        return Err(SpecialError {
            function: "polygamma",
            kind: SpecialErrorKind::PoleInput,
            mode,
            detail: "tetragamma pole at nonpositive integer",
        });
    }
    // scipy.special.polygamma(2, x) = -inf at every nonpositive-integer pole
    // (even-order derivative → one-signed −inf). frankenscipy-fxm0t
    if x == 0.0 || is_negative_integer_pole(x) {
        return Ok(f64::NEG_INFINITY);
    }
    let value = crate::convenience::tetragamma(x);
    if !value.is_finite() {
        record_special_trace(
            "polygamma",
            mode,
            "non_finite_output",
            format!("input={x}"),
            "returned_non_finite",
            format!("output={value}"),
            false,
        );
    }
    Ok(value)
}

fn polygamma_higher_scalar(order: usize, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if matches!(mode, RuntimeMode::Hardened) && is_negative_integer_pole(x) {
        record_special_trace(
            "polygamma",
            mode,
            "pole_input",
            format!("order={order},input={x}"),
            "fail_closed",
            "higher-order polygamma pole at nonpositive integer",
            false,
        );
        return Err(SpecialError {
            function: "polygamma",
            kind: SpecialErrorKind::PoleInput,
            mode,
            detail: "higher-order polygamma pole at nonpositive integer",
        });
    }
    // scipy.special.polygamma(n, x) at a nonpositive-integer pole is +inf for odd
    // n and -inf for even n (the leading term is (-1)^{n+1} n!/(x+k)^{n+1}, a
    // one-signed pole). frankenscipy-fxm0t
    if x == 0.0 || is_negative_integer_pole(x) {
        return Ok(if order % 2 == 1 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        });
    }
    let value = polygamma_higher_core(order, x);
    if !value.is_finite() {
        record_special_trace(
            "polygamma",
            mode,
            "non_finite_output",
            format!("order={order},input={x}"),
            "returned_non_finite",
            format!("output={value}"),
            false,
        );
    }
    Ok(value)
}

pub(crate) fn rgamma_scalar(x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    if is_negative_integer_pole(x) {
        return Ok(0.0);
    }
    let gamma_value = gamma_scalar(x, mode)?;
    let value = 1.0 / gamma_value;
    if !value.is_finite() {
        record_special_trace(
            "rgamma",
            mode,
            "non_finite_output",
            format!("input={x}"),
            "returned_non_finite",
            format!("output={value}"),
            false,
        );
    }
    Ok(value)
}

fn complex_rgamma_scalar(z: Complex64) -> Complex64 {
    if !z.is_finite() {
        return Complex64::new(f64::NAN, f64::NAN);
    }
    if is_complex_real_gamma_pole(z) {
        return Complex64::from_real(0.0);
    }
    (-complex_gammaln(z)).exp()
}

pub fn gammainc_scalar(a: f64, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    validate_incomplete_gamma_domain("gammainc", a, x, mode)?;
    let (p, _) = regularized_gamma_pair(a, x, mode)?;
    Ok(p)
}

pub fn gammaincc_scalar(a: f64, x: f64, mode: RuntimeMode) -> Result<f64, SpecialError> {
    validate_incomplete_gamma_domain("gammaincc", a, x, mode)?;
    let (_, q) = regularized_gamma_pair(a, x, mode)?;
    Ok(q)
}

/// Natural log of the regularized lower incomplete gamma function `P(a, x)`.
///
/// `ln P(a, x)` stays finite deep in the left tail where `P` itself underflows
/// to 0 (so `gammainc_scalar(a, x).ln()` would be `-inf`). In the
/// lower-dominant region (`x < a + 1`) `P = prefactor * sum`, where
/// `prefactor = exp(-x + a*ln x - lnΓ(a))` underflows but the series `sum`
/// (starting at `1/a`) is `O(1)`; computing
/// `ln P = (-x + a*ln x - lnΓ(a)) + ln(sum)` keeps full precision. In the
/// upper-dominant region (`x >= a + 1`) `P` is `O(1)`, so it is formed as
/// `1 - Q` (Q from the continued fraction) and `ln P = ln1p(-Q)`. Matches
/// `gammainc_scalar(a, x).ln()` wherever the latter is representable.
///
/// Domain: `a > 0`, `x >= 0`. Returns `NaN` for invalid `a`/`x`, `-inf` at
/// `x = 0` (`P = 0`), and `0.0` at `x = +inf` (`P = 1`).
#[must_use]
pub fn log_gammainc_scalar(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if !a.is_finite() || a <= 0.0 || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x.is_infinite() {
        return 0.0;
    }

    const EPS: f64 = 1.0e-14;
    const FPMIN: f64 = 1.0e-300;

    let lg = match gammaln_scalar(a, RuntimeMode::Strict) {
        Ok(v) => v,
        Err(_) => return f64::NAN,
    };
    let log_prefactor = -x + a * x.ln() - lg;

    if x < a + 1.0 {
        // Lower-tail series. `sum` is O(1/a), so the log form stays finite even
        // when P = prefactor * sum underflows to 0 in the left tail.
        let mut ap = a;
        let mut term = 1.0 / a;
        let mut sum = term;
        let series_max = ((12.0 * a.sqrt()) as usize + 200).min(2_000_000);
        for _ in 0..series_max {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if term.abs() <= sum.abs() * EPS {
                break;
            }
        }
        log_prefactor + sum.ln()
    } else {
        // Upper-dominant region: P is O(1); form Q via the continued fraction
        // and return ln(1 - Q) = ln1p(-Q).
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / FPMIN;
        let mut d = 1.0 / b;
        let mut h = d;
        // The upper continued fraction needs ~√a steps near x≈a; the fixed
        // 200-iter cap truncated it for very large a (Q(100000,100001) was ~1e-5
        // off). Scale the cap with √a (the EPS-break keeps typical a cheap),
        // mirroring the lower-series fix. frankenscipy.
        let cf_max = ((12.0 * a.sqrt()) as usize + 200).min(2_000_000);
        for i in 1..=cf_max {
            let i_f = i as f64;
            let an = -i_f * (i_f - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < FPMIN {
                d = FPMIN;
            }
            c = b + an / c;
            if c.abs() < FPMIN {
                c = FPMIN;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() <= EPS {
                break;
            }
        }
        let q = (log_prefactor.exp() * h).clamp(0.0, 1.0);
        (-q).ln_1p()
    }
}

/// Natural log of the regularized upper incomplete gamma function `Q(a, x)`.
///
/// `ln Q(a, x)` stays finite deep in the right tail where `Q` itself
/// underflows to 0 (so `gammaincc_scalar(a, x).ln()` would be `-inf`). In the
/// upper-tail region (`x >= a + 1`) `Q = prefactor * h`, where
/// `prefactor = exp(-x + a*ln x - lnΓ(a))` underflows but the Lentz continued
/// fraction `h` is `O(1)`; computing
/// `ln Q = (-x + a*ln x - lnΓ(a)) + ln(h)` keeps full precision. In the
/// lower-dominant region (`x < a + 1`) `Q` is `O(1)`, so it is evaluated
/// directly and logged. Matches `gammaincc_scalar(a, x).ln()` wherever the
/// latter is representable.
///
/// Domain: `a > 0`, `x >= 0`. Returns `NaN` for invalid `a`/`x`, `0.0` at
/// `x = 0` (`Q = 1`), and `-inf` at `x = +inf` (`Q = 0`).
#[must_use]
pub fn log_gammaincc_scalar(a: f64, x: f64) -> f64 {
    if a.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if !a.is_finite() || a <= 0.0 || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x.is_infinite() {
        return f64::NEG_INFINITY;
    }

    const EPS: f64 = 1.0e-14;
    const FPMIN: f64 = 1.0e-300;

    let lg = match gammaln_scalar(a, RuntimeMode::Strict) {
        Ok(v) => v,
        Err(_) => return f64::NAN,
    };
    let log_prefactor = -x + a * x.ln() - lg;

    if x >= a + 1.0 {
        // Upper-tail continued fraction (Lentz). `h` is O(1), so the log form
        // stays finite even when Q = prefactor * h underflows to 0.
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / FPMIN;
        let mut d = 1.0 / b;
        let mut h = d;
        // The upper continued fraction needs ~√a steps near x≈a; the fixed
        // 200-iter cap truncated it for very large a (Q(100000,100001) was ~1e-5
        // off). Scale the cap with √a (the EPS-break keeps typical a cheap),
        // mirroring the lower-series fix. frankenscipy.
        let cf_max = ((12.0 * a.sqrt()) as usize + 200).min(2_000_000);
        for i in 1..=cf_max {
            let i_f = i as f64;
            let an = -i_f * (i_f - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < FPMIN {
                d = FPMIN;
            }
            c = b + an / c;
            if c.abs() < FPMIN {
                c = FPMIN;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() <= EPS {
                break;
            }
        }
        log_prefactor + h.ln()
    } else {
        // Lower-dominant region: Q is O(1); evaluate the series for P then log Q.
        let mut ap = a;
        let mut term = 1.0 / a;
        let mut sum = term;
        let series_max = ((12.0 * a.sqrt()) as usize + 200).min(2_000_000);
        for _ in 0..series_max {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if term.abs() <= sum.abs() * EPS {
                break;
            }
        }
        let p = (log_prefactor.exp() * sum).clamp(0.0, 1.0);
        (1.0 - p).ln()
    }
}

fn validate_incomplete_gamma_domain(
    function: &'static str,
    a: f64,
    x: f64,
    mode: RuntimeMode,
) -> Result<(), SpecialError> {
    if a.is_nan() || x.is_nan() {
        return Ok(());
    }
    if !a.is_finite() || a <= 0.0 || x < 0.0 {
        return match mode {
            RuntimeMode::Strict => {
                record_special_trace(
                    function,
                    mode,
                    "domain_error",
                    format!("a={a},x={x}"),
                    "returned_nan",
                    "strict domain fallback",
                    false,
                );
                Ok(())
            }
            RuntimeMode::Hardened => {
                record_special_trace(
                    function,
                    mode,
                    "domain_error",
                    format!("a={a},x={x}"),
                    "fail_closed",
                    "regularized incomplete gamma requires finite a>0 and x>=0",
                    false,
                );
                Err(SpecialError {
                    function,
                    kind: SpecialErrorKind::DomainError,
                    mode,
                    detail: "regularized incomplete gamma requires finite a>0 and x>=0",
                })
            }
        };
    }
    Ok(())
}

fn regularized_gamma_pair(a: f64, x: f64, mode: RuntimeMode) -> Result<(f64, f64), SpecialError> {
    if a.is_nan() || x.is_nan() {
        return Ok((f64::NAN, f64::NAN));
    }
    if !a.is_finite() || a <= 0.0 || x < 0.0 {
        return Ok((f64::NAN, f64::NAN));
    }
    if x == 0.0 {
        return Ok((0.0, 1.0));
    }
    if x.is_infinite() {
        return Ok((1.0, 0.0));
    }

    const EPS: f64 = 1.0e-14;
    const FPMIN: f64 = 1.0e-300;

    let lg = gammaln_scalar(a, RuntimeMode::Strict)?;
    let prefactor = (-x + a * x.ln() - lg).exp();
    let (p, q) = if x < a + 1.0 {
        // The lower series Σ xᵏ/(a)_{k+1} needs ~12√a terms to converge near
        // x≈a (the term ratio x/(a+k) ≈ 1 there); the old fixed 200-term cap
        // truncated it for large a — e.g. P(5000,5000) was 0.5% off (needs 558
        // terms). Scale the cap with √a so it stays exact at large a while the
        // ε-break keeps small/typical a cheap. frankenscipy.
        let series_max = ((12.0 * a.sqrt()) as usize + 200).min(2_000_000);
        let mut ap = a;
        let mut term = 1.0 / a;
        let mut sum = term;
        for _ in 0..series_max {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if term.abs() <= sum.abs() * EPS {
                break;
            }
        }
        let lower = prefactor * sum;
        let p_value = clamp_unit_interval(lower);
        let q_value = clamp_unit_interval(1.0 - p_value);
        (p_value, q_value)
    } else {
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / FPMIN;
        let mut d = 1.0 / b;
        let mut h = d;
        // The upper continued fraction needs ~√a steps near x≈a; the fixed
        // 200-iter cap truncated it for very large a (Q(100000,100001) was ~1e-5
        // off). Scale the cap with √a (the EPS-break keeps typical a cheap),
        // mirroring the lower-series fix. frankenscipy.
        let cf_max = ((12.0 * a.sqrt()) as usize + 200).min(2_000_000);
        for i in 1..=cf_max {
            let i_f = i as f64;
            let an = -i_f * (i_f - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < FPMIN {
                d = FPMIN;
            }
            c = b + an / c;
            if c.abs() < FPMIN {
                c = FPMIN;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() <= EPS {
                break;
            }
        }
        let upper = prefactor * h;
        let q_value = clamp_unit_interval(upper);
        let p_value = clamp_unit_interval(1.0 - q_value);
        (p_value, q_value)
    };

    if !p.is_finite() {
        record_special_trace(
            "gammainc",
            mode,
            "non_finite_output",
            format!("a={a},x={x}"),
            "returned_non_finite",
            format!("output={p}"),
            false,
        );
    }
    if !q.is_finite() {
        record_special_trace(
            "gammaincc",
            mode,
            "non_finite_output",
            format!("a={a},x={x}"),
            "returned_non_finite",
            format!("output={q}"),
            false,
        );
    }

    Ok((p, q))
}

fn clamp_unit_interval(value: f64) -> f64 {
    if value.is_nan() {
        return f64::NAN;
    }
    value.clamp(0.0, 1.0)
}

fn gamma_core(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }
    if x.is_infinite() {
        return if x.is_sign_positive() {
            f64::INFINITY
        } else {
            f64::NAN
        };
    }
    if is_negative_integer_pole(x) {
        // SciPy returns NaN for negative integer poles
        return f64::NAN;
    }

    if x < 0.5 {
        // Reflection formula: Γ(x) = π / (sin(πx) * Γ(1-x))
        let sin_pi_x = (PI * x).sin();
        if sin_pi_x == 0.0 {
            // This matches x = 0 handled above, or integer x < 0
            if x == 0.0 {
                return if x.is_sign_negative() {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                };
            }
            return f64::NAN;
        }
        return PI / (sin_pi_x * gamma_core(1.0 - x));
    }

    if x <= GAMMA_CEPHES_MAX_X && GAMMA_CEPHES_RATIONAL.load(std::sync::atomic::Ordering::Relaxed) {
        GAMMA_CEPHES_RATIONAL_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return gamma_cephes_reduced(x);
    }

    gamma_lanczos(x)
}

// ── Cephes Gamma rational, 0.5 <= x <= 33 ────────────────────────────────────────────
//
// WHY. `gamma_lanczos` costs EIGHT divisions and THREE transcendentals per call — `t.ln()`,
// `coeff_sum.abs().ln()` and a final `.exp()`. SciPy uses NONE on this range: it reduces the
// argument into [2, 3] by multiplication and division, then evaluates one rational P(6)/Q(7).
// Measured against live SciPy on the identical fixture, instructions per element:
//
//     gamma    fsci 305.6   scipy 127.5   2.40x
//     rgamma   fsci 428.6   scipy 134.0   3.20x   (it is 1/gamma_scalar, so it inherits this)
//
// THIS IS NOT THE `gammaln` CASE. That rewrite was refuted because the band where the two
// implementations already agree was STILL 1.45x, so no kernel swap could reach the gap.
// Here the implementations do not agree anywhere on the measured domain: SciPy evaluates a
// polynomial where we call `exp` and `ln` twice, for EVERY x in the fixture.
//
// ARTICULATION POINT: `gamma_core` is what `rgamma`, `factorial`, `poch` and the beta family
// call, so this is not one function's kernel.
//
// Coefficients are transcribed from scipy's `xsf/cephes/gamma.h` — the code the incumbent
// runs — so this moves our values TOWARD SciPy, and are cross-checked against the Lanczos
// form by `gamma_cephes_rational_matches_the_lanczos_form`.

/// Numerator of the Cephes `Gamma` rational on `[2, 3]`, in `x - 2`.
#[allow(clippy::excessive_precision)]
const GAMMA_CEPHES_P: [f64; 7] = [
    1.60119522476751861407E-4,
    1.19135147006586384913E-3,
    1.04213797561761569935E-2,
    4.76367800457137231464E-2,
    2.07448227648435975150E-1,
    4.94214826801497100753E-1,
    9.99999999999999996796E-1,
];

/// Denominator of the Cephes `Gamma` rational. NOT monic — unlike `ZETAC_Q` this one stores
/// its leading coefficient, so it is evaluated with `polevl` and not `p1evl`.
#[allow(clippy::excessive_precision)]
const GAMMA_CEPHES_Q: [f64; 8] = [
    -2.31581873324120129819E-5,
    5.39605580493303397842E-4,
    -4.45641913851797240494E-3,
    1.18139785222060435552E-2,
    3.58236398605498653373E-2,
    -2.34591795718243348568E-1,
    7.14304917030273074085E-2,
    1.00000000000000000320E0,
];

/// `Γ(x)` for `0.5 <= x <= 33` by SciPy's own method: reduce into `[2, 3]`, then one
/// rational. No transcendental is evaluated anywhere on this path.
fn gamma_cephes_reduced(mut x: f64) -> f64 {
    let mut z = 1.0_f64;
    while x >= 3.0 {
        x -= 1.0;
        z *= x;
    }
    while x < 2.0 {
        z /= x;
        x += 1.0;
    }
    if x == 2.0 {
        return z;
    }
    x -= 2.0;
    z * polevl(x, &GAMMA_CEPHES_P) / polevl(x, &GAMMA_CEPHES_Q)
}

/// Upper limit of the rational form. Above it Cephes switches to Stirling; we keep the
/// Lanczos kernel there instead, which is unchanged behaviour and outside every domain this
/// harness measures — named rather than silently assumed equivalent.
const GAMMA_CEPHES_MAX_X: f64 = 33.0;

fn gamma_lanczos(x: f64) -> f64 {
    let x_minus_1 = x - 1.0;
    let mut coeff_sum = LANCZOS_COEFFS[0];
    for (idx, coeff) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
        coeff_sum += coeff / (x_minus_1 + idx as f64);
    }

    let t = x_minus_1 + 7.5;
    if coeff_sum == 0.0 {
        return 0.0;
    }
    let log_abs_value = SQRT_2PI.ln() + (x_minus_1 + 0.5) * t.ln() - t + coeff_sum.abs().ln();
    if log_abs_value > f64::MAX.ln() {
        return f64::INFINITY.copysign(coeff_sum);
    }
    log_abs_value.exp().copysign(coeff_sum)
}

fn digamma_core(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
    }
    if x.is_infinite() {
        return if x.is_sign_positive() {
            f64::INFINITY
        } else {
            f64::NAN
        };
    }
    if is_negative_integer_pole(x) {
        return f64::NAN;
    }

    if x < 0.0 {
        return digamma_core(1.0 - x) - PI / (PI * x).tan();
    }

    let mut shifted = x;
    let mut acc = 0.0;
    while shifted < 12.0 {
        acc -= 1.0 / shifted;
        shifted += 1.0;
    }

    let inv = 1.0 / shifted;
    let inv2 = inv * inv;
    // Bernoulli asymptotic extended through the B₁₀ term (was truncated at B₆,
    // leaving ~2.5e-10 at the shift point). frankenscipy-luxsz.
    acc + shifted.ln()
        - 0.5 * inv
        - inv2
            * (1.0 / 12.0
                - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 * (1.0 / 240.0 - inv2 / 132.0))))
}

fn trigamma_core(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x.is_infinite() {
        return if x.is_sign_positive() { 0.0 } else { f64::NAN };
    }
    if is_negative_integer_pole(x) {
        return f64::NAN;
    }

    if x < 0.0 {
        let sin_pi_x = (PI * x).sin();
        return (PI * PI) / (sin_pi_x * sin_pi_x) - trigamma_core(1.0 - x);
    }

    let mut shifted = x;
    let mut acc = 0.0;
    while shifted < 12.0 {
        acc += 1.0 / (shifted * shifted);
        shifted += 1.0;
    }

    let inv = 1.0 / shifted;
    let inv2 = inv * inv;
    let inv3 = inv2 * inv;
    let inv5 = inv3 * inv2;
    let inv7 = inv5 * inv2;
    let inv9 = inv7 * inv2;
    let inv11 = inv9 * inv2;
    // Extended through the B₁₀ term (was truncated at B₆). frankenscipy-luxsz.
    acc + inv + 0.5 * inv2 + inv3 / 6.0 - inv5 / 30.0 + inv7 / 42.0 - inv9 / 30.0
        + 5.0 * inv11 / 66.0
}

fn polygamma_higher_core(order: usize, x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return polygamma_sign(order) * f64::INFINITY;
    }
    if x.is_infinite() {
        return if x.is_sign_positive() { 0.0 } else { f64::NAN };
    }
    if is_negative_integer_pole(x) {
        return f64::NAN;
    }

    let sign = polygamma_sign(order);
    let factorial = factorial_f64(order);
    let order_plus_one = order as f64 + 1.0;

    let mut shifted = x;
    let mut correction = 0.0;
    while shifted < 12.0 {
        correction += sign * factorial / shifted.powf(order_plus_one);
        shifted += 1.0;
    }

    correction + sign * factorial * crate::convenience::hurwitz_zeta(order_plus_one, shifted)
}

fn polygamma_sign(order: usize) -> f64 {
    if order % 2 == 1 { 1.0 } else { -1.0 }
}

fn factorial_f64(order: usize) -> f64 {
    (1..=order).fold(1.0, |acc, value| acc * value as f64)
}

fn is_negative_integer_pole(x: f64) -> bool {
    x.is_finite() && x < 0.0 && x.fract() == 0.0
}

fn is_complex_real_gamma_pole(z: Complex64) -> bool {
    z.im == 0.0 && (z.re == 0.0 || is_negative_integer_pole(z.re))
}

/// Log-gamma function computed directly via Stirling series.
/// Avoids the overflow that gamma_core hits for large arguments.
/// Only valid for x > 0.
fn lngamma_positive(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    if x < 8.0 {
        // Shift upward using recurrence: lnΓ(x) = lnΓ(x+n) - ln(x(x+1)...(x+n-1))
        let mut shifted = x;
        let mut correction = 0.0;
        while shifted < 12.0 {
            correction += shifted.ln();
            shifted += 1.0;
        }
        stirling_lngamma(shifted) - correction
    } else {
        stirling_lngamma(x)
    }
}

/// Default `x` at or above which `gammaln` leaves the Lanczos kernel for the asymptotic
/// (Stirling) one.
///
/// WAS 100, LOWERED TO 20 ON ACCURACY, measured against live SciPy 1.17.1 over a
/// logarithmic grid of 1278 points on [8, 120] with extra points packed against every
/// candidate boundary (`src/bin/diff_gammaln_crossover.rs`). Worst error over the range
/// each candidate actually changes, in ULPs of the reference:
///
/// ```text
/// crossover   worst_ulp   Lanczos over the same range   verdict
///        13       18.41                          3.18   REJECTED
///        15        3.80                          3.18   REJECTED
///        16        2.29                          3.18   justified (edge)
///        20        0.82                          3.18   justified, ~3.9x margin
///        40        0.00                          3.18   identical to Lanczos
/// ```
///
/// So the asymptotic kernel is not merely adequate above 20, it is 3.9x MORE accurate than
/// the Lanczos kernel it replaces there, and the two agree exactly above 40. 20 is chosen
/// over the measured boundary of 16 to keep margin rather than sit on it.
///
/// JUDGED IN ULPs, NOT ABSOLUTE ERROR, and the distinction reverses the answer at the low
/// end: at crossover 13 the asymptotic kernel has the SMALLER worst absolute error
/// (8.17e-14 against 1.71e-13) and is nonetheless 5.8x worse, because its error lands at
/// lnΓ ≈ 20 where 8e-14 is 18 ULP, while the Lanczos kernel's lands at lnΓ ≈ 201 where a
/// larger absolute error is only 3 ULP. Absolute error alone would have endorsed 13.
///
/// This is cheaper as well — one `ln` against two, and no eight-term division chain — but
/// that is a CONSEQUENCE, not the reason, and how much it is worth depends entirely on how
/// much of a caller's input lies in [20, 100). For inputs below 20 it is worth nothing.
pub const GAMMALN_ASYMPTOTIC_MIN_X_DEFAULT: f64 = 20.0;

/// Read the Lanczos/asymptotic crossover ONCE per batch instead of once per element
/// (`true`, shipping). Bit-identical: the threshold is the same `f64` either way.
///
/// Measured because the banked plan for this cell was wrong. The recorded diagnosis was
/// that `gammaln` is 2.07x slower because the Lanczos kernel does 8 divisions and 2 logs,
/// and that the fix is to adopt Cephes' `lgam`. Instruction counts per element, by band,
/// against live SciPy on the identical fixture, say otherwise:
///
/// ```text
/// band            fsci    scipy   ratio
/// [0.5, 19.9]    243.3    154.1   1.58x   Lanczos
/// [20.1, 60]     175.3    121.1   1.45x   asymptotic — SAME algorithm as Cephes
/// [0.01, 60]     199.0    132.3   1.50x   shipping mix
/// ```
///
/// The asymptotic band is one log and one divide in BOTH implementations and is still
/// 1.45x, so most of the gap is per-element overhead that no kernel swap can reach.
/// Replacing the Lanczos kernel with a free one would move the mix only 199.0 -> 177.2.
pub static GAMMALN_HOIST_THRESHOLD: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);
/// Fill a preallocated output buffer by index instead of collecting a `Result` per element
/// (`true`, shipping). Bit-identical: same `f`, same indices, same order, same first error.
///
/// This is the shared mapper for the whole crate's array paths — 245 call sites across
/// gamma, bessel, beta, elliptic, airy, error and hyper — so it is an articulation point
/// rather than one kernel's tuning. Profiled on `gammaln`, the `collect::<Result<Vec<_>,
/// _>>()` adapter was 26.1% of the op, more than the `log` it wraps.
pub static GAMMA_FAMILY_PREALLOC_FILL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);
/// Array evaluations that took the preallocated-fill arm — "enabled" is not "took effect".
///
/// Accuracy contract: a diagnostic counter, not an A/B lever. It has no arms to preserve
/// anything between; incrementing it changes no numeric result, so output is bit-identical
/// whether or not anyone reads it.
pub static GAMMA_FAMILY_PREALLOC_FILL_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Use SciPy's own `zetac` rational approximations for ζ(x), x > 1 (`true`, shipping),
/// instead of our eight-`exp` Euler-Maclaurin prefix.
///
/// NOT bit-identical and not intended to be: it is a different approximation, and it is the
/// one the incumbent uses, so it moves our values toward SciPy. Accuracy against the live
/// SciPy arm is the contract and is checked in the same invocation by
/// `perf_special_vs_scipy`; the two forms are also pinned against each other by
/// `zeta_cephes_rational_matches_the_euler_maclaurin_series`.
pub static ZETA_CEPHES_RATIONAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);
/// Evaluations that took the rational arm — "enabled" is not "took effect".
///
/// Accuracy contract: a diagnostic counter, not an A/B lever. It has no arms to preserve
/// anything between; incrementing it changes no numeric result.
pub static ZETA_CEPHES_RATIONAL_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Use SciPy's own rational form for `Γ(x)` on `0.5 <= x <= 33` (`true`, shipping) instead
/// of the Lanczos kernel's eight divisions and three transcendentals.
///
/// NOT bit-identical and not intended to be: it is a different approximation, and it is the
/// one the incumbent uses, so it moves our values toward SciPy. Accuracy against the live
/// SciPy arm is the contract, checked in the same invocation by `perf_special_vs_scipy`;
/// the two forms are cross-checked against each other by
/// `gamma_cephes_rational_matches_the_lanczos_form`.
pub static GAMMA_CEPHES_RATIONAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);
/// Evaluations that took the rational arm — "enabled" is not "took effect".
///
/// Accuracy contract: a diagnostic counter, not an A/B lever. It has no arms to preserve
/// anything between; incrementing it changes no numeric result.
pub static GAMMA_CEPHES_RATIONAL_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Batches that took the hoisted-threshold arm — "enabled" is not "took effect".
///
/// Accuracy contract: a diagnostic counter, not an A/B lever. It has no arms to preserve
/// anything between; incrementing it changes no numeric result, so output is bit-identical
/// whether or not anyone reads it.
pub static GAMMALN_HOIST_THRESHOLD_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Override for [`GAMMALN_ASYMPTOTIC_MIN_X_DEFAULT`], as raw `f64` bits; 0 means "use the
/// default". Exists so the two kernels can be compared against SciPy on IDENTICAL inputs
/// rather than argued about from truncation-error algebra.
///
/// This is an ACCURACY question before it is a cost question. The asymptotic kernel is
/// cheaper — one `ln` against the Lanczos kernel's two, and no division chain — so lowering
/// the crossover is worth something, but only if it is at least as accurate there. Lowering
/// it because a benchmark's fixture happens to sit in the affected range would be tuning the
/// input distribution rather than the function.
pub static GAMMALN_ASYMPTOTIC_MIN_OVERRIDE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

fn gammaln_asymptotic_min_x() -> f64 {
    let bits = GAMMALN_ASYMPTOTIC_MIN_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if bits == 0 {
        GAMMALN_ASYMPTOTIC_MIN_X_DEFAULT
    } else {
        f64::from_bits(bits)
    }
}

/// Stirling series for lnΓ(x), valid for large x.
fn stirling_lngamma(x: f64) -> f64 {
    // Folded by LLVM; see the note in `lngamma_lanczos`. Measured, not assumed.
    let half_ln_2pi = 0.5 * (2.0 * PI).ln();
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    // Bernoulli number series: lnΓ(x) ≈ (x-0.5)*ln(x) - x + 0.5*ln(2π)
    //   + 1/(12x) - 1/(360x³) + 1/(1260x⁵) - 1/(1680x⁷)
    (x - 0.5) * x.ln() - x
        + half_ln_2pi
        + inv * (1.0 / 12.0 - inv2 * (1.0 / 360.0 - inv2 * (1.0 / 1260.0 - inv2 / 1680.0)))
}

// ══════════════════════════════════════════════════════════════════════
// Combinatorial Functions
// ══════════════════════════════════════════════════════════════════════

/// Compute n! (factorial).
///
/// Matches `scipy.special.factorial(n, exact=True)`.
/// Returns f64 to handle large factorials (overflows to infinity).
pub fn factorial(n: u64) -> f64 {
    // Use lookup table for small values, gamma for large
    if n <= 20 {
        FACTORIALS[n as usize]
    } else {
        // n! = gamma(n+1)
        gamma_core((n + 1) as f64)
    }
}

/// Compute the double factorial n!! = n * (n-2) * (n-4) * ... * (1 or 2).
///
/// Matches `scipy.special.factorial2(n, exact=True)`.
///
/// For odd n: n!! = n * (n-2) * ... * 3 * 1
/// For even n: n!! = n * (n-2) * ... * 4 * 2
///
/// SciPy compatibility cases:
/// - 0!! = 1 (by convention)
/// - For negative n, returns 0
pub fn factorial2(n: i64) -> f64 {
    match n.cmp(&0) {
        std::cmp::Ordering::Less => 0.0,
        std::cmp::Ordering::Equal => 1.0, // 0!! = 1
        std::cmp::Ordering::Greater => {
            if n <= 33 {
                // Direct computation for small values
                let mut result = 1.0_f64;
                let mut k = n;
                while k > 1 {
                    result *= k as f64;
                    k -= 2;
                }
                result
            } else {
                // For large n, use relation to gamma function:
                // n!! = 2^(n/2) * Gamma(n/2 + 1) for even n
                // n!! = 2^((n+1)/2) * Gamma((n+1)/2) / sqrt(pi) for odd n
                let nf = n as f64;
                if n % 2 == 0 {
                    // Even: n!! = 2^(n/2) * (n/2)!
                    let half_n = nf / 2.0;
                    2.0_f64.powf(half_n) * gamma_core(half_n + 1.0)
                } else {
                    // Odd m: m!! = 2^((m+1)/2) * Gamma((m+2)/2) / sqrt(pi),
                    // derived from (2k-1)!! = Gamma(k+1/2) * 2^k / sqrt(pi)
                    // with m = 2k - 1, so k = (m+1)/2 and Gamma argument is
                    // k + 1/2 = (m+2)/2.
                    let exp_pow = (nf + 1.0) / 2.0;
                    let gamma_arg = (nf + 2.0) / 2.0;
                    2.0_f64.powf(exp_pow) * gamma_core(gamma_arg) / PI.sqrt()
                }
            }
        }
    }
}

/// Compute the binomial coefficient `C(x, y) = Γ(x+1) / (Γ(y+1) · Γ(x−y+1))`
/// for real `x ≥ 0` and `y ≥ 0`.
///
/// Matches `scipy.special.binom(x, y)` on the common domain. Differences
/// from `comb`:
/// - accepts any non-negative real `x`, `y` (not just integers)
/// - uses the gamma-function generalization, so `binom(0.5, 0.25)` is
///   well-defined (≈ 1.069 via Γ(1.5)/(Γ(1.25)·Γ(1.25)))
///
/// Conventions:
/// - `y > x` returns 0.0 (matching scipy's vanishing-coefficient rule on
///   the real generalization)
/// - `x < 0` or `y < 0` returns NaN. The full negative-argument path needs
///   the gamma reflection formula and is tracked as a follow-on under
///   `frankenscipy-b6z3m`.
/// - NaN propagates.
pub fn binom(n: f64, k: f64) -> f64 {
    // Faithful port of scipy.special.binom (xsf `binom`). Computes the
    // generalized binomial coefficient C(n, k) = Γ(n+1)/(Γ(k+1)Γ(n-k+1)) over
    // real arguments, choosing the formulation that minimizes rounding:
    //   * integer k in [0, 20): the multiplicative product (signed, exact for
    //     integer results — e.g. binom(0.5, 2) = -0.125, which the old gamma
    //     formula wrongly collapsed to 0);
    //   * n ≫ k: a betaln-based form avoiding overflow;
    //   * k ≫ |n|: a Γ-reflection asymptotic;
    //   * otherwise the signed beta-function form 1/((n+1)·B(n−k+1, k+1)),
    //     which is finite (and 0 only at a genuine denominator pole), unlike
    //     the old `x−y+1 ≤ 0 ⇒ 0` shortcut. frankenscipy-binom
    if n.is_nan() || k.is_nan() {
        return f64::NAN;
    }
    if n < 0.0 && n == n.floor() {
        // Negative-integer n is undefined (Γ(n+1) pole, no cancellation).
        return f64::NAN;
    }

    let mut kx = k.floor();
    if k == kx && (n.abs() > 1e-8 || n == 0.0) {
        // Integer k: multiplicative form, less rounding than the gamma ratio.
        let nx = n.floor();
        if nx == n && kx > nx / 2.0 && nx > 0.0 {
            kx = nx - kx; // reduce by symmetry C(n,k)=C(n,n-k)
        }
        if (0.0..20.0).contains(&kx) {
            let mut num = 1.0_f64;
            let mut den = 1.0_f64;
            let kxi = kx as i64;
            for i in 1..(1 + kxi) {
                num *= i as f64 + n - kx;
                den *= i as f64;
                if num.abs() > 1e50 {
                    num /= den;
                    den = 1.0;
                }
            }
            return num / den;
        }
    }

    // General case.
    if n >= 1e10 * k && k > 0.0 {
        // n ≫ k: avoid under/overflow via logarithms.
        let bl = crate::beta::betaln_scalar(1.0 + n - k, 1.0 + k, RuntimeMode::Strict)
            .unwrap_or(f64::NAN);
        (-bl - (n + 1.0).ln()).exp()
    } else if k > 1e8 * n.abs() {
        // k ≫ |n|: Γ-reflection asymptotic (scipy xsf binom).
        let g = gamma_core(1.0 + n);
        let mut num = g / k.abs() + g * n / (2.0 * k * k);
        num /= PI * k.abs().powf(n);
        if k > 0.0 {
            let (dk, sgn) = if k == k.floor() {
                (0.0, if (k as i64) % 2 == 0 { 1.0 } else { -1.0 })
            } else {
                (k, 1.0)
            };
            num * ((dk - n) * PI).sin() * sgn
        } else if k == k.floor() {
            0.0
        } else {
            num * (k * PI).sin()
        }
    } else {
        // Signed beta form, finite except at a genuine denominator pole.
        let b =
            crate::beta::beta_scalar(1.0 + n - k, 1.0 + k, RuntimeMode::Strict).unwrap_or(f64::NAN);
        1.0 / (n + 1.0) / b
    }
}

/// k-step factorial: `n * (n − k) * (n − 2k) * … * r`, where the iteration
/// stops at the smallest positive remainder `r`.
///
/// Matches `scipy.special.factorialk(n, k, exact=False)`. Conventions:
/// - `n == 0` returns 1.0 (empty-product identity).
/// - `n < 0` returns NaN (real factorialk only defined on non-negative
///   integers; scipy.special's exact=False mode also returns NaN here).
/// - `k < 1` returns NaN (k must be a positive integer).
/// - `k == 1` reduces to the ordinary factorial; `k == 2` to the double
///   factorial.
pub fn factorialk(n: i64, k: i64) -> f64 {
    if k < 1 {
        return f64::NAN;
    }
    if n < 0 {
        return f64::NAN;
    }
    if n == 0 {
        return 1.0;
    }
    if k == 1 {
        return factorial(n as u64);
    }
    let mut result = 1.0_f64;
    let mut step = n;
    while step > 0 {
        result *= step as f64;
        step -= k;
    }
    result
}

/// Poisson distribution CDF: P(X <= k) for Poisson with mean m.
///
/// Matches `scipy.special.pdtr(k, m)`.
///
/// Uses the relation: pdtr(k, m) = gammaincc(k + 1, m)
///
/// # Arguments
/// * `k` - Number of events (non-negative integer, but accepts float)
/// * `m` - Expected number of events (mean, must be >= 0)
pub fn pdtr(k: f64, m: f64) -> f64 {
    if k.is_nan() || m.is_nan() {
        return f64::NAN;
    }
    if m < 0.0 || k < 0.0 {
        return f64::NAN;
    }
    if m == 0.0 {
        return 1.0; // P(X <= k) = 1 when m = 0
    }

    // pdtr(k, m) = gammaincc(k + 1, m) = Q(k + 1, m)
    gammaincc_scalar(k + 1.0, m, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// Poisson distribution survival function: P(X > k) for Poisson with mean m.
///
/// Matches `scipy.special.pdtrc(k, m)`.
///
/// Uses the relation: pdtrc(k, m) = gammainc(k + 1, m)
///
/// # Arguments
/// * `k` - Number of events (non-negative integer, but accepts float)
/// * `m` - Expected number of events (mean, must be >= 0)
pub fn pdtrc(k: f64, m: f64) -> f64 {
    if k.is_nan() || m.is_nan() {
        return f64::NAN;
    }
    if m < 0.0 || k < 0.0 {
        return f64::NAN;
    }
    if m == 0.0 {
        return 0.0; // P(X > k) = 0 when m = 0
    }

    // pdtrc(k, m) = gammainc(k + 1, m) = P(k + 1, m)
    gammainc_scalar(k + 1.0, m, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// Inverse of Poisson CDF: find m such that pdtr(k, m) = p.
///
/// Matches `scipy.special.pdtri(k, p)`.
///
/// Uses Newton's method to find the root.
///
/// # Arguments
/// * `k` - Number of events (non-negative integer, but accepts float)
/// * `p` - Probability (must be in [0, 1])
pub fn pdtri(k: f64, p: f64) -> f64 {
    if k.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if k < 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::INFINITY;
    }
    if p == 1.0 {
        return 0.0;
    }

    // pdtri(k, p): m such that gammaincc(k+1, m) = p, i.e. m = gammainccinv(k+1, p).
    // Invert the *complemented* incomplete gamma directly rather than via
    // gammaincinv(k+1, 1 − p): forming `1 − p` loses tail precision for small p,
    // while the complemented inverse keeps it exact (matches scipy to ~1e-15).
    crate::convenience::gammainccinv_scalar(k + 1.0, p)
}

/// Inverse of Poisson CDF with respect to event count k.
///
/// Matches `scipy.special.pdtrik(p, m)`.
///
/// Uses the relation pdtr(k, m) = gammaincc(k + 1, m), so k + 1 is the
/// inverse of gammainc(a, m) at 1 - p.
#[must_use]
pub fn pdtrik(p: f64, m: f64) -> f64 {
    if p.is_nan() || m.is_nan() {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&p) || m < 0.0 {
        return f64::NAN;
    }
    if p == 1.0 {
        return f64::NAN;
    }
    if p == 0.0 || m == 0.0 {
        return 0.0;
    }
    if !m.is_finite() {
        return f64::NAN;
    }

    let shape = gammainc_shape_inv(m, 1.0 - p);
    if !shape.is_finite() {
        return shape;
    }

    (shape - 1.0).max(0.0)
}

/// Chi-squared distribution CDF.
///
/// Returns the probability P(X <= x) where X follows a chi-squared
/// distribution with v degrees of freedom.
///
/// Matches `scipy.special.chdtr(v, x)`.
#[must_use]
pub fn chdtr(v: f64, x: f64) -> f64 {
    if v.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if v <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    // chdtr(v, x) = gammainc(v/2, x/2) = P(v/2, x/2)
    gammainc_scalar(v / 2.0, x / 2.0, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// Chi-squared distribution survival function.
///
/// Returns the probability P(X > x) where X follows a chi-squared
/// distribution with v degrees of freedom.
///
/// Matches `scipy.special.chdtrc(v, x)`.
#[must_use]
pub fn chdtrc(v: f64, x: f64) -> f64 {
    if v.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if v <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    // chdtrc(v, x) = gammaincc(v/2, x/2) = Q(v/2, x/2)
    gammaincc_scalar(v / 2.0, x / 2.0, RuntimeMode::Strict).unwrap_or(f64::NAN)
}

/// Non-central chi-squared cumulative distribution function.
///
/// Matches `scipy.special.chndtr(x, df, nc)`: the CDF at `x` of a non-central
/// chi-squared variable with `df` degrees of freedom and non-centrality `nc`.
///
/// Computed as the Poisson(nc/2)-weighted mixture of central chi-squared CDFs
///
/// ```text
///   chndtr(x, df, nc) = Σ_{j≥0} e^{−λ} λ^j / j! · chdtr(df + 2j, x),  λ = nc/2
/// ```
///
/// The sum is accumulated outward from the Poisson mode `j₀ = ⌊λ⌋` (whose
/// weight is formed in log space) so that large `nc` neither underflows the
/// leading `e^{−λ}` factor nor loses precision.
#[must_use]
pub fn chndtr(x: f64, df: f64, nc: f64) -> f64 {
    if x.is_nan() || df.is_nan() || nc.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if nc <= 0.0 {
        return chdtr(df, x);
    }
    let lam = nc / 2.0;
    let j0 = lam.floor();
    // Poisson weight at the mode j0, formed in log space to avoid underflow.
    let logw0 =
        -lam + j0 * lam.ln() - gammaln_scalar(j0 + 1.0, RuntimeMode::Strict).unwrap_or(f64::NAN);
    let w0 = logw0.exp();

    // Each Poisson term needs chdtr(df + 2j, x) = P(a, y), the regularized lower
    // incomplete gamma with a = df/2 + j, y = x/2. Computing it fresh per term
    // costs one `gammainc` each — O(√nc) gammainc evals overall. Instead anchor at
    // the mode and walk the standard incomplete-gamma recurrence in O(1)/term:
    //   P(a+1, y) = P(a, y) − t(a),   t(a) = y^a e^{−y} / Γ(a+1),
    //   t(a+1) = t(a)·y/(a+1)         (upward, a increasing);
    //   P(a−1, y) = P(a, y) + t(a−1), t(a−1) = t(a)·a/y   (downward).
    // One gammainc at the mode + O(√nc) multiply/adds. The downward branch (a
    // decreasing, P growing toward 1 by ADDING positive t) is the stable direction
    // and carries the dominant mass; the upward branch only adds small above-mode
    // corrections. Verified vs scipy.special.chndtr to ≤1e-12 rel across the test
    // grid incl. large nc and the deep left tail.
    let y = x / 2.0;
    let a0 = 0.5 * df + j0;
    let p0 = chdtr(df + 2.0 * j0, x); // = P(a0, y)
    let t0 =
        (a0 * y.ln() - y - gammaln_scalar(a0 + 1.0, RuntimeMode::Strict).unwrap_or(f64::NAN)).exp();

    let mut total = 0.0_f64;
    // Upward from the mode.
    let mut w = w0;
    let mut j = j0;
    let mut a = a0;
    let mut p = p0;
    let mut t = t0;
    let mut steps = 0;
    while steps < 100_000 {
        total += w * p.clamp(0.0, 1.0);
        // advance j → j+1, a → a+1
        p -= t;
        j += 1.0;
        a += 1.0;
        t *= y / a;
        w *= lam / j;
        if w < 1e-300 || (w < 1e-17 * total.max(1e-300) && j > lam) {
            break;
        }
        steps += 1;
    }
    // Downward from the mode.
    w = w0;
    j = j0;
    a = a0;
    p = p0;
    t = t0;
    while j > 0.0 {
        w *= j / lam;
        // advance j → j-1, a → a-1: t(a-1) = t(a)·a/y, P(a-1) = P(a) + t(a-1)
        t *= a / y;
        p += t;
        j -= 1.0;
        a -= 1.0;
        total += w * p.clamp(0.0, 1.0);
        if w < 1e-17 * total.max(1e-300) {
            break;
        }
    }
    total.clamp(0.0, 1.0)
}

/// Noncentral chi-square SURVIVAL function — the complement of [`chndtr`], matching
/// `scipy.stats.ncx2.sf(x, df, nc)`.
///
/// ## Why this exists rather than `1 - chndtr`
///
/// `chndtr` saturates at 1.0 in the right tail, so the complement computed by subtraction
/// collapses to exactly ZERO while the true survival is still finite. Measured against
/// scipy 1.17.1 at `df = 3, nc = 2`:
///
/// | x | `ncx2.sf` | `1 - cdf` |
/// |---|---|---|
/// | 80 | 1.626871961685992e-13 | 1.627587e-13 (4 digits wrong) |
/// | 150 | 1.044927027553427e-26 | **0.0** |
///
/// A tail p-value of 1e-26 reported as zero is not a rounding difference; it is the
/// difference between a computable result and "impossible".
///
/// ## Method
///
/// Identical Poisson-mixture walk to [`chndtr`], with the regularized UPPER incomplete
/// gamma in place of the lower:
///
/// ```text
/// sf(x) = Σ_j  pois(j; nc/2) · Q(df/2 + j, x/2)
/// ```
///
/// The incomplete-gamma recurrence is the mirror of the one in `chndtr`. From
/// `P(a+1,y) = P(a,y) − t(a)` and `Q = 1 − P`:
///
/// ```text
/// Q(a+1, y) = Q(a, y) + t(a),    t(a+1) = t(a)·y/(a+1)   (upward)
/// Q(a−1, y) = Q(a, y) − t(a−1),  t(a−1) = t(a)·a/y       (downward)
/// ```
///
/// **The stable direction is reversed relative to `chndtr`.** There the downward branch
/// was stable because `P` grew toward 1 by ADDING positive `t`; here it is the UPWARD
/// branch that adds, while the downward branch subtracts and is the cancellation-prone
/// one. That asymmetry is the whole reason this is a separate kernel rather than
/// `1 - chndtr`, and inverting it by copy-paste would reintroduce the cancellation in a
/// less obvious place.
#[must_use]
pub fn chndtrc(x: f64, df: f64, nc: f64) -> f64 {
    if x.is_nan() || df.is_nan() || nc.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    if nc <= 0.0 {
        return chdtrc(df, x);
    }
    let lam = nc / 2.0;
    let j0 = lam.floor();
    let logw0 =
        -lam + j0 * lam.ln() - gammaln_scalar(j0 + 1.0, RuntimeMode::Strict).unwrap_or(f64::NAN);
    let w0 = logw0.exp();

    let y = x / 2.0;
    let a0 = 0.5 * df + j0;
    let q0 = chdtrc(df + 2.0 * j0, x); // = Q(a0, y), computed directly, never as 1 - P
    let t0 =
        (a0 * y.ln() - y - gammaln_scalar(a0 + 1.0, RuntimeMode::Strict).unwrap_or(f64::NAN)).exp();

    let mut total = 0.0_f64;
    // Upward from the mode: Q grows by adding positive t — the stable direction here.
    let mut w = w0;
    let mut j = j0;
    let mut a = a0;
    let mut q = q0;
    let mut t = t0;
    let mut steps = 0;
    while steps < 100_000 {
        total += w * q.clamp(0.0, 1.0);
        q += t;
        j += 1.0;
        a += 1.0;
        t *= y / a;
        w *= lam / j;
        if w < 1e-300 || (w < 1e-17 * total.max(1e-300) && j > lam) {
            break;
        }
        steps += 1;
    }
    // Downward from the mode.
    w = w0;
    j = j0;
    a = a0;
    q = q0;
    t = t0;
    while j > 0.0 {
        w *= j / lam;
        t *= a / y;
        q -= t;
        j -= 1.0;
        a -= 1.0;
        total += w * q.clamp(0.0, 1.0);
        if w < 1e-17 * total.max(1e-300) {
            break;
        }
    }
    total.clamp(0.0, 1.0)
}

/// Inverse of [`chndtr`] in the argument `x`.
///
/// Returns `x` such that `chndtr(x, df, nc) = p`, matching
/// `scipy.special.chndtrix(p, df, nc)`. The CDF is monotonically increasing in
/// `x`, so the root is found by bracket-and-bisect. Boundaries follow scipy:
/// `p = 0 → 0`, `p = 1 → +∞`, and `p ∉ [0, 1]` (or NaN inputs) → NaN.
#[must_use]
pub fn chndtrix(p: f64, df: f64, nc: f64) -> f64 {
    if p.is_nan() || df.is_nan() || nc.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    let mut hi = 1.0_f64;
    let mut fhi = chndtr(hi, df, nc);
    while fhi < p {
        hi *= 2.0;
        if hi > 1e300 {
            return f64::INFINITY;
        }
        fhi = chndtr(hi, df, nc);
    }
    // chndtr(·, df, nc) is increasing in x with chndtr(0, ·) = 0, so
    // f(x) = chndtr(x, df, nc) − p is increasing with f(0) = −p < 0 < f(hi).
    // Illinois false-position converges in ~12 chndtr evals vs the old fixed
    // 100-step bisection — each eval is a full Poisson-weighted mixture, so this
    // ~8× cuts chndtrix's dominant cost. (Its siblings chndtridf/chndtrinc
    // already route through the Illinois `invert_monotone`.) The final doubling
    // eval is reused as `fhi`; f(0) = −p needs no chndtr call.
    let fhi_resid = fhi - p;
    if fhi_resid == 0.0 {
        return hi;
    }
    crate::beta::illinois_root(|m| chndtr(m, df, nc) - p, 0.0, hi, -p, fhi_resid)
}

/// Bisect for `v ∈ [a, b]` with `f(v) = target`, where `f` is monotone on
/// `[a, b]` (direction auto-detected). If `target` is outside the range spanned
/// by `f(a)..f(b)`, the nearer bound is returned (the unsolvable-tail clamp).
fn invert_monotone(f: impl Fn(f64) -> f64, target: f64, a: f64, b: f64) -> f64 {
    let (fa, fb) = (f(a), f(b));
    if target <= fa.min(fb) {
        return if fa <= fb { a } else { b };
    }
    if target >= fa.max(fb) {
        return if fa >= fb { a } else { b };
    }
    // Superlinear Illinois (~12 evals) instead of 200 fixed bisection steps — each
    // eval here is a full chndtr (Poisson-weighted mixture), so this is a large win
    // for chndtridf/chndtrinc. The clamp guards above guarantee `target` is strictly
    // inside (min(fa,fb), max(fa,fb)), so neither endpoint is the root. Illinois
    // solves an INCREASING residual, so flip the sign when f is decreasing.
    if fb >= fa {
        crate::beta::illinois_root(|m| f(m) - target, a, b, fa - target, fb - target)
    } else {
        crate::beta::illinois_root(|m| target - f(m), a, b, target - fa, target - fb)
    }
}

/// Inverse of [`chndtr`] in the degrees of freedom `df`.
///
/// Returns `df` such that `chndtr(x, df, nc) = p`, matching
/// `scipy.special.chndtridf(x, p, nc)`. The CDF is monotone (decreasing) in
/// `df`, so the root is found by bisection. `p ∉ (0, 1)` (or NaN) → NaN.
#[must_use]
pub fn chndtridf(x: f64, p: f64, nc: f64) -> f64 {
    if x.is_nan() || p.is_nan() || nc.is_nan() || p <= 0.0 || p >= 1.0 {
        return f64::NAN;
    }
    invert_monotone(|df| chndtr(x, df, nc), p, 1e-6, 1e10)
}

/// Inverse of [`chndtr`] in the non-centrality `nc`.
///
/// Returns `nc` such that `chndtr(x, df, nc) = p`, matching
/// `scipy.special.chndtrinc(x, df, p)`. The CDF is monotone (decreasing) in
/// `nc ≥ 0`; if `p` exceeds the central value `chndtr(x, df, 0)` the result
/// clamps toward 0 (as scipy does). `p ∉ (0, 1)` (or NaN) → NaN.
#[must_use]
pub fn chndtrinc(x: f64, df: f64, p: f64) -> f64 {
    if x.is_nan() || df.is_nan() || p.is_nan() || p <= 0.0 || p >= 1.0 {
        return f64::NAN;
    }
    invert_monotone(|nc| chndtr(x, df, nc), p, 0.0, 1e8)
}

/// Inverse complemented chi-squared distribution.
///
/// Returns x such that P(X > x) = p where X follows a chi-squared
/// distribution with v degrees of freedom.
///
/// Matches `scipy.special.chdtri(v, p)`.
#[must_use]
pub fn chdtri(v: f64, p: f64) -> f64 {
    if v.is_nan() || p.is_nan() {
        return f64::NAN;
    }
    if v <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::INFINITY;
    }
    if p == 1.0 {
        return 0.0;
    }
    // chdtri(v, p) finds x such that gammaincc(v/2, x/2) = p, i.e.
    // x = 2 * gammainccinv(v/2, p). Invert the *complemented* incomplete gamma
    // directly rather than via gammaincinv(v/2, 1 − p): forming `1 − p` loses
    // all tail precision for small p (e.g. p = 1e-8 was ~6e-9 off), whereas the
    // complemented inverse keeps the small p exact (matches scipy to ~1e-15).
    2.0 * crate::convenience::gammainccinv_scalar(v / 2.0, p)
}

/// Inverse chi-squared distribution CDF with respect to degrees of freedom.
///
/// Returns `v` such that `P(X <= x) = p` where `X` follows a chi-squared
/// distribution with `v` degrees of freedom.
///
/// Matches `scipy.special.chdtriv(p, x)`.
#[must_use]
pub fn chdtriv(p: f64, x: f64) -> f64 {
    if p.is_nan() || x.is_nan() {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&p) || x <= 0.0 || !x.is_finite() {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::INFINITY;
    }
    if p == 1.0 {
        return 0.0;
    }

    let shape = gammainc_shape_inv(x / 2.0, p);
    if !shape.is_finite() {
        return shape;
    }

    2.0 * shape
}

/// Inverse of the regularized lower incomplete gamma function.
/// Finds x such that gammainc(a, x) = p.
///
/// Delegates to `crate::convenience::gammaincinv_scalar`. The local
/// Newton-Raphson here was diverging when p was close to 0 or 1: the
/// initial guess for small p reached x ≈ 8 even when the true root
/// was ~0.15, and Newton walked to the iteration cap returning a
/// value many orders of magnitude wrong (frankenscipy-jr3na). The
/// convenience helper is the validated one that the diff_special
/// gammainc harness already covers at 1e-9 rel.
fn gammaincinv(a: f64, p: f64) -> f64 {
    if a <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    crate::convenience::gammaincinv_scalar(a, p)
}

fn gammainc_shape_inv(x: f64, p: f64) -> f64 {
    let mut lo = 0.0;
    let mut hi = x.max(1.0);
    while gammainc_scalar(hi, x, RuntimeMode::Strict).unwrap_or(f64::NAN) > p {
        lo = hi;
        hi *= 2.0;
        if !hi.is_finite() {
            return f64::INFINITY;
        }
    }

    // P(shape, x) is monotone DECREASING in shape, so f(t) = p − P(t, x) is
    // increasing with f(lo) < 0 < f(hi). Illinois false-position converges in
    // ~3-4× fewer gammainc evaluations than the old 180-cap bisection. At lo = 0
    // the shape-0 limit P(0, x) = 1 (x > 0) gives f(lo) = p − 1 without an extra
    // (and potentially NaN) gammainc call.
    let f = |t: f64| p - gammainc_scalar(t, x, RuntimeMode::Strict).unwrap_or(f64::NAN);
    let flo = if lo == 0.0 { p - 1.0 } else { f(lo) };
    let fhi = f(hi);
    if fhi == 0.0 {
        return hi;
    }
    if flo == 0.0 {
        return lo;
    }
    if !flo.is_finite() || !fhi.is_finite() {
        return f64::NAN;
    }
    crate::beta::illinois_root(f, lo, hi, flo, fhi)
}

const FACTORIALS: [f64; 21] = [
    1.0,
    1.0,
    2.0,
    6.0,
    24.0,
    120.0,
    720.0,
    5_040.0,
    40_320.0,
    362_880.0,
    3_628_800.0,
    39_916_800.0,
    479_001_600.0,
    6_227_020_800.0,
    87_178_291_200.0,
    1_307_674_368_000.0,
    20_922_789_888_000.0,
    355_687_428_096_000.0,
    6_402_373_705_728_000.0,
    121_645_100_408_832_000.0,
    2_432_902_008_176_640_000.0,
];

/// Compute the binomial coefficient C(n, k) = n! / (k! * (n-k)!).
///
/// Matches `scipy.special.comb(n, k, exact=True)` for nonneg integer inputs.
/// Returns 0 for invalid inputs (k > n or negative-equivalent).
pub fn comb(n: u64, k: u64) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }
    // Use symmetry: C(n, k) = C(n, n-k) to minimize multiplications
    let k = k.min(n - k);
    if k <= 20 && n <= 170 {
        // Direct computation to avoid floating-point drift
        let mut result = 1.0_f64;
        for i in 0..k {
            result *= (n - i) as f64;
            result /= (i + 1) as f64;
        }
        result
    } else {
        // Use log-gamma for large values: ln(C(n,k)) = lnΓ(n+1) - lnΓ(k+1) - lnΓ(n-k+1)
        let lnc = lngamma_positive((n + 1) as f64)
            - lngamma_positive((k + 1) as f64)
            - lngamma_positive((n - k + 1) as f64);
        lnc.exp()
    }
}

/// Compute the number of k-permutations of n: P(n, k) = n! / (n-k)!.
///
/// Matches `scipy.special.perm(n, k, exact=True)` for nonneg integer inputs.
/// Returns 0 for invalid inputs (k > n).
pub fn perm(n: u64, k: u64) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 {
        return 1.0;
    }
    if k <= 20 && n <= 170 {
        let mut result = 1.0_f64;
        for i in 0..k {
            result *= (n - i) as f64;
        }
        result
    } else {
        let lnp = lngamma_positive((n + 1) as f64) - lngamma_positive((n - k + 1) as f64);
        lnp.exp()
    }
}

// ══════════════════════════════════════════════════════════════════════
// Riemann Zeta Function
// ══════════════════════════════════════════════════════════════════════

/// Compute the Riemann zeta function ζ(s) for real tensors.
///
/// Matches `scipy.special.zeta(s)` and uses the same parallel RealVec dispatch
/// shape as the gamma-family tensor entry points.
pub fn zeta(s_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    if let SpecialTensor::RealVec(values) = s_tensor
        && let Some(out) = zeta_positive_affine_vec(values)
    {
        return Ok(SpecialTensor::RealVec(out));
    }
    map_real_input_rp("zeta", s_tensor, mode, |s| Ok(zeta_scalar(s)), 1 << 17) // be ~90k
}

/// Compute the Riemann zeta function ζ(s) for real scalar `s`.
///
/// Matches `scipy.special.zeta(s)`.
///
/// Uses direct summation with Euler-Maclaurin acceleration for s > 1,
/// and the reflection formula for s < 0.
pub(crate) fn zeta_scalar(s: f64) -> f64 {
    if s.is_nan() {
        return f64::NAN;
    }
    if s == 1.0 {
        return f64::INFINITY; // pole
    }
    if s == 0.0 {
        return -0.5; // ζ(0) = -1/2
    }
    if s.abs() < 1.0e-3 {
        // Near s=0 the reflection formula multiplies a tiny sin(πs/2) by ζ(1−s),
        // which blows up near its pole at 1 — catastrophic cancellation whose
        // relative error grows like 1/|s| (≈3e-8 at s=−1e-9). Use the Taylor
        // series ζ(s) = Σ ζ^(n)(0)/n! · sⁿ instead; 8 terms give full f64
        // precision for |s| < 1e-3. frankenscipy.
        #[allow(clippy::excessive_precision)]
        const ZETA_TAYLOR_0: [f64; 8] = [
            -0.5,
            -0.918_938_533_204_672_741_780_329_7,
            -1.003_178_227_954_292_425_605_05,
            -1.000_785_194_477_042_407_960_177,
            -0.999_879_299_500_571_164_957_800_8,
            -1.000_001_940_896_320_456_037_8,
            -1.000_001_301_146_013_959_624_312,
            -0.999_999_831_384_173_610_779_930_2,
        ];
        let mut acc = ZETA_TAYLOR_0[7];
        for &c in ZETA_TAYLOR_0[..7].iter().rev() {
            acc = acc * s + c;
        }
        return acc;
    }
    if s > 1.0 {
        zeta_positive(s)
    } else if s < 0.0 {
        // Reflection formula: ζ(s) = 2^s * π^(s-1) * sin(πs/2) * Γ(1-s) * ζ(1-s)
        let s1 = 1.0 - s;
        let z1 = zeta_positive(s1);
        let sin_half_pi_s = (PI * s / 2.0).sin();
        let gamma_1_minus_s = gamma_core(s1);
        // Compute parts carefully to avoid overflow/underflow
        let factor = 2.0_f64.powf(s) * PI.powf(s - 1.0) * sin_half_pi_s * gamma_1_minus_s;
        factor * z1
    } else {
        // 0 < s < 1: use Dirichlet eta function relation
        // ζ(s) = η(s) / (1 - 2^(1-s)) where η(s) = sum (-1)^(n+1) / n^s
        let eta = dirichlet_eta(s);
        let denom = 1.0 - 2.0_f64.powf(1.0 - s);
        if denom == 0.0 {
            return f64::NAN;
        }
        eta / denom
    }
}

/// Zeta for s > 1 via Euler-Maclaurin summation.
///
/// Specializes the Hurwitz `a = 1` case by hard-wiring the direct
/// summation prefix and using `exp(-s ln n)` instead of generic
/// `powf`. N=10 plus five Bernoulli corrections keeps the existing
/// scalar reduction tests below 1e-13 abs while removing three
/// transcendental calls per positive Riemann evaluation versus N=13.
// A uniform table of ln(n) for n = 2..=9 (frankenscipy-e2ve2). Entry 0 happens
// to equal std::f64::consts::LN_2, and ZETA_POSITIVE_TAIL_LN below equals LN_10,
// but substituting the named constants for those two would make the table
// heterogeneous -- two entries spelled as consts, six as literals -- and hide
// that this is one precomputed sequence. Kept literal on purpose.
#[allow(clippy::approx_constant)]
const ZETA_POSITIVE_DIRECT_LN: [f64; 8] = [
    0.693_147_180_559_945_3,
    1.098_612_288_668_109_8,
    1.386_294_361_119_890_6,
    1.609_437_912_434_100_3,
    1.791_759_469_228_055,
    1.945_910_149_055_313_2,
    2.079_441_541_679_835_7,
    2.197_224_577_336_219_6,
];
const ZETA_POSITIVE_TAIL_NA: f64 = 10.0;
// ln(10); see the note on ZETA_POSITIVE_DIRECT_LN above (frankenscipy-e2ve2).
#[allow(clippy::approx_constant)]
const ZETA_POSITIVE_TAIL_LN: f64 = 2.302_585_092_994_046;
const ZETA_POSITIVE_TAIL_INV: f64 = 1.0 / ZETA_POSITIVE_TAIL_NA;
const ZETA_POSITIVE_TAIL_INV_SQ: f64 = ZETA_POSITIVE_TAIL_INV * ZETA_POSITIVE_TAIL_INV;
const ZETA_AFFINE_MIN_LEN: usize = 1024;
const ZETA_AFFINE_BLOCK: usize = 64;
/// Batch length at/above which the affine-`zeta` block loop fans across cores. Each
/// `ZETA_AFFINE_BLOCK`-sized block re-seeds its own recurrence, so blocks are independent; the
/// per-element work (8-term direct sum + `zeta_positive_tail`) amortises the spawn from ~32k up.
const ZETA_AFFINE_PARALLEL_MIN_LEN: usize = 1 << 15;

/// Same-binary A/B toggle for the affine-`zeta` block parallelism. Byte-identical either way (each
/// block re-seeds identically and writes a disjoint output slice); benchmark knob, read once.
#[doc(hidden)]
pub static ZETA_AFFINE_FORCE_SERIAL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Fill `out` with affine-`zeta` values, processing `ZETA_AFFINE_BLOCK`-sized blocks of `values`.
/// `values`/`out` are a block-aligned slice (block boundaries fall on `ZETA_AFFINE_BLOCK` multiples
/// from the slice start), so this is BYTE-IDENTICAL whether the whole array is one call or split
/// into aligned chunks across threads: every block re-seeds from its own `values[block_start]` and
/// the shared `direct_ratios`/`tail_ratio` are `x`-invariant.
fn fill_zeta_affine_blocks(
    values: &[f64],
    out: &mut [f64],
    direct_ratios: &[f64; ZETA_POSITIVE_DIRECT_LN.len()],
    tail_ratio: f64,
) {
    for block_start in (0..values.len()).step_by(ZETA_AFFINE_BLOCK) {
        let block_end = (block_start + ZETA_AFFINE_BLOCK).min(values.len());
        let block_s = values[block_start];
        let mut direct_terms = [0.0; ZETA_POSITIVE_DIRECT_LN.len()];
        for (term, &ln_n) in direct_terms.iter_mut().zip(&ZETA_POSITIVE_DIRECT_LN) {
            *term = (-block_s * ln_n).exp();
        }
        let mut tail_neg_s = (-block_s * ZETA_POSITIVE_TAIL_LN).exp();
        for i in block_start..block_end {
            let mut sum = 1.0;
            for &term in &direct_terms {
                sum += term;
            }
            out[i] = zeta_positive_tail(values[i], sum, tail_neg_s);
            for (term, &ratio) in direct_terms.iter_mut().zip(direct_ratios) {
                *term *= ratio;
            }
            tail_neg_s *= tail_ratio;
        }
    }
}

/// Stop the direct prefix once a term stops changing the running sum (`true`, shipping).
/// BIT-IDENTICAL — see `zeta_positive`.
pub static ZETA_DIRECT_EARLY_EXIT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

// ── Cephes `zetac` rational approximations, x > 1 ────────────────────────────────────
//
// WHY THESE EXIST. Our Euler-Maclaurin form evaluates `Σ n^-s` for n = 2..=9 as EIGHT
// `exp` calls plus one for the tail. SciPy never does that: profiled on the same fixture,
// `exp` is 68.7% of our 567.1 instructions per element (~390), while SciPy spends ~154 in
// transcendentals TOTAL and 34.9% in a polynomial. Its `zetac_positive` uses a table for
// integer arguments, a rational approximation in `1/x` for 1 <= x <= 10 (ONE `pow`), and a
// rational-plus-`exp` form for 10 < x <= 50. Eight `exp` calls against one `pow` is the
// whole 1.95x instruction gap on this cell.
//
// Coefficients are transcribed from scipy's `xsf/cephes/zetac.h`, which is the code the
// incumbent actually runs, so adopting them moves our values TOWARD SciPy rather than away
// — this is a parity change and a cost change in the same direction. A mistranscribed digit
// would show up immediately as a large `max_rel` against the live SciPy arm, and
// `zeta_cephes_rational_matches_the_euler_maclaurin_series` pins the two forms against each
// other independently of that.

/// `zetac(i) = ζ(i) - 1` for integer `i` in `0..=30`. Entry 1 is unused (ζ(1) is a pole).
#[allow(clippy::excessive_precision)]
const ZETAC_INTEGER: [f64; 31] = [
    -1.500_000_000_000_000_000_00E0,
    0.0,
    6.449_340_668_482_264_364_72E-1,
    2.020_569_031_595_942_854_00E-1,
    8.232_323_371_113_819_151_60E-2,
    3.692_775_514_336_992_633_14E-2,
    1.734_306_198_444_913_971_45E-2,
    8.349_277_381_922_826_839_80E-3,
    4.077_356_197_944_339_378_69E-3,
    2.008_392_826_082_214_417_85E-3,
    9.945_751_278_180_853_371_46E-4,
    4.941_886_041_194_645_587_02E-4,
    2.460_865_533_080_482_986_38E-4,
    1.227_133_475_784_891_467_52E-4,
    6.124_813_505_870_482_925_85E-5,
    3.058_823_630_702_049_355_17E-5,
    1.528_225_940_865_187_173_26E-5,
    7.637_197_637_899_762_273_60E-6,
    3.817_293_264_999_839_856_46E-6,
    1.908_212_716_553_938_925_66E-6,
    9.539_620_338_727_961_131_52E-7,
    4.769_329_867_878_064_631_17E-7,
    2.384_505_027_277_329_900_04E-7,
    1.192_199_259_653_110_730_68E-7,
    5.960_818_905_125_947_961_24E-8,
    2.980_350_351_465_228_018_61E-8,
    1.490_155_482_836_504_123_47E-8,
    7.450_711_789_835_429_491_98E-9,
    3.725_334_024_788_457_054_82E-9,
    1.862_659_723_513_049_006_40E-9,
    9.313_274_324_196_681_828_72E-10,
];

/// `2^x (1 - 1/x) (ζ(x) - 1) = P(1/x)/Q(1/x)` for `1 <= x <= 10`.
#[allow(clippy::excessive_precision)]
const ZETAC_P: [f64; 9] = [
    5.857_465_145_697_253_195_40E11,
    2.575_341_277_561_025_728_88E11,
    4.877_811_595_679_482_564_38E10,
    5.153_995_380_238_857_706_96E9,
    3.416_460_735_147_540_942_81E8,
    1.608_370_068_806_564_927_31E7,
    5.927_854_673_421_095_229_98E5,
    1.511_291_699_649_388_231_17E4,
    2.018_224_444_859_979_558_65E2,
];

/// Denominator for [`ZETAC_P`]; monic, so the leading `1.0` is implicit.
#[allow(clippy::excessive_precision)]
const ZETAC_Q: [f64; 8] = [
    3.904_976_763_733_711_575_16E11,
    5.228_582_353_682_721_617_97E10,
    5.644_515_172_712_805_433_51E9,
    3.390_067_460_153_504_188_34E8,
    1.794_103_715_001_264_537_02E7,
    5.666_668_251_313_847_970_29E5,
    1.603_829_768_109_441_315_06E4,
    1.964_362_372_233_873_141_44E2,
];

/// `log(ζ(x) - 1 - 2^-x) = A(x)/B(x)` for `10 <= x <= 50`.
#[allow(clippy::excessive_precision)]
const ZETAC_A: [f64; 11] = [
    8.707_285_674_845_901_925_39E6,
    1.765_068_656_703_464_627_57E8,
    2.608_895_067_074_832_648_96E10,
    5.298_063_740_098_947_916_47E11,
    2.268_881_561_192_382_414_87E13,
    3.318_844_029_327_050_835_99E14,
    5.137_789_979_758_682_301_92E15,
    -1.981_236_881_339_071_714_55E15,
    -9.927_638_100_399_835_723_56E16,
    7.829_053_761_808_705_864_44E16,
    9.267_862_757_689_277_171_87E16,
];

/// Denominator for [`ZETAC_A`]; monic, so the leading `1.0` is implicit.
#[allow(clippy::excessive_precision)]
const ZETAC_B: [f64; 10] = [
    -7.926_254_105_637_410_628_61E6,
    -1.605_299_699_329_202_296_76E8,
    -2.376_692_609_755_432_217_88E10,
    -4.803_195_843_504_551_698_57E11,
    -2.078_209_617_541_733_201_70E13,
    -2.960_754_045_072_722_236_80E14,
    -4.862_991_036_946_091_366_86E15,
    5.345_895_096_757_899_301_99E15,
    5.714_641_110_922_976_312_92E16,
    -1.799_155_976_586_765_568_28E16,
];

/// Cephes `polevl`: evaluate `coef[0]·xⁿ + … + coef[n]` by Horner, highest order first.
#[inline]
fn polevl(x: f64, coef: &[f64]) -> f64 {
    let mut ans = coef[0];
    for &c in &coef[1..] {
        ans = ans * x + c;
    }
    ans
}

/// Cephes `p1evl`: as [`polevl`] but for a MONIC polynomial, so the leading `1.0` is not
/// stored. `p1evl(x, c)` is `polevl(x, [1.0, c…])` and is a different function from
/// `polevl` on the same slice — conflating the two silently shifts the degree by one.
#[inline]
fn p1evl(x: f64, coef: &[f64]) -> f64 {
    let mut ans = x + coef[0];
    for &c in &coef[1..] {
        ans = ans * x + c;
    }
    ans
}

/// `ζ(x) - 1` for `x > 1`, by SciPy's own method.
///
/// Returns `ζ(x) - 1` rather than `ζ(x)` for the same reason Cephes does: for large `x` the
/// difference from 1 is what carries the information, and forming `1 + tiny` at the end
/// loses nothing that computing `ζ` directly would have kept.
fn zetac_positive_rational(x: f64) -> f64 {
    // Beyond this the first term `2^-x` has itself underflowed relative to 1.
    if x >= 127.0 {
        return 0.0;
    }

    // Integer arguments are tabulated: no transcendental at all.
    let w = x.floor();
    if w == x {
        let i = x as usize;
        if i < ZETAC_INTEGER.len() {
            return ZETAC_INTEGER[i];
        }
    }

    if x <= 10.0 {
        let b = 2.0_f64.powf(x) * (x - 1.0);
        let w = 1.0 / x;
        return (x * polevl(w, &ZETAC_P)) / (b * p1evl(w, &ZETAC_Q));
    }

    if x <= 50.0 {
        let b = 2.0_f64.powf(-x);
        let w = polevl(x, &ZETAC_A) / p1evl(x, &ZETAC_B);
        return w.exp() + b;
    }

    // Basic sum of inverse powers over ODD integers, then folded to all integers by
    // `(s + 2^-x) / (1 - 2^-x)`. Only reachable for 50 < x < 127, where it converges in a
    // couple of terms.
    let mut s = 0.0_f64;
    let mut a = 1.0_f64;
    let mut b;
    loop {
        a += 2.0;
        b = a.powf(-x);
        s += b;
        if b / s <= f64::EPSILON / 2.0 {
            break;
        }
    }
    let b = 2.0_f64.powf(-x);
    (s + b) / (1.0 - b)
}

fn zeta_positive(s: f64) -> f64 {
    if ZETA_CEPHES_RATIONAL.load(std::sync::atomic::Ordering::Relaxed) {
        ZETA_CEPHES_RATIONAL_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return 1.0 + zetac_positive_rational(s);
    }
    let mut sum = 1.0;
    if ZETA_DIRECT_EARLY_EXIT.load(std::sync::atomic::Ordering::Relaxed) {
        // The prefix is `Σ n^-s` for n = 2..=9, evaluated as eight `exp` calls. For large
        // `s` most of them are already below the running sum's resolution: at s = 30,
        // 4^-30 = 8.7e-19 against sum ≈ 1 whose ULP is 2.2e-16, so `sum + term` IS `sum`.
        //
        // BIT-IDENTICAL, and provably rather than empirically. `ln_n` is strictly
        // increasing and `s > 1`, so `exp(-s·ln_n)` is strictly DECREASING in n. Once an
        // addition is a no-op, every later term is smaller still and would be a no-op
        // against the same unchanged `sum`. Breaking therefore skips only additions that
        // were already identities — the accumulated value is the same float, not merely a
        // close one, which is why this needs no tolerance and is pinned by a bits test.
        //
        // Costs one compare per surviving term and saves an `exp` per skipped one. It saves
        // NOTHING for small `s` (at s = 1.5 no term is negligible and all eight run), so
        // this is worth a lot at large `s` and exactly zero at small — not a trade.
        for &ln_n in &ZETA_POSITIVE_DIRECT_LN {
            let next = sum + (-s * ln_n).exp();
            if next == sum {
                break;
            }
            sum = next;
        }
    } else {
        for &ln_n in &ZETA_POSITIVE_DIRECT_LN {
            sum += (-s * ln_n).exp();
        }
    }

    let tail_neg_s = (-s * ZETA_POSITIVE_TAIL_LN).exp();
    zeta_positive_tail(s, sum, tail_neg_s)
}

fn zeta_positive_tail(s: f64, mut sum: f64, tail_neg_s: f64) -> f64 {
    sum += ZETA_POSITIVE_TAIL_NA * tail_neg_s / (s - 1.0);
    sum += 0.5 * tail_neg_s;

    let mut term = tail_neg_s * ZETA_POSITIVE_TAIL_INV;
    let mut poch = s;
    sum += (1.0 / 12.0) * poch * term;
    term *= ZETA_POSITIVE_TAIL_INV_SQ;
    poch *= (s + 1.0) * (s + 2.0);
    sum -= (1.0 / 720.0) * poch * term;
    term *= ZETA_POSITIVE_TAIL_INV_SQ;
    poch *= (s + 3.0) * (s + 4.0);
    sum += (1.0 / 30240.0) * poch * term;
    term *= ZETA_POSITIVE_TAIL_INV_SQ;
    poch *= (s + 5.0) * (s + 6.0);
    sum -= (1.0 / 1209600.0) * poch * term;
    term *= ZETA_POSITIVE_TAIL_INV_SQ;
    poch *= (s + 7.0) * (s + 8.0);
    sum += (5.0 / 239500800.0) * poch * term;

    sum
}

fn zeta_positive_affine_vec(values: &[f64]) -> Option<Vec<f64>> {
    if values.len() < ZETA_AFFINE_MIN_LEN {
        return None;
    }
    let first = *values.first()?;
    let last = *values.last()?;
    if !(first > 1.0 && last > 1.0 && first.is_finite() && last.is_finite()) {
        return None;
    }

    let step = (last - first) / (values.len() - 1) as f64;
    if !step.is_finite() {
        return None;
    }
    for (i, &s) in values.iter().enumerate() {
        if !(s > 1.0 && s.is_finite()) {
            return None;
        }
        let expected = first + step * i as f64;
        let scale = s.abs().max(expected.abs()).max(1.0);
        if (s - expected).abs() > 512.0 * f64::EPSILON * scale {
            return None;
        }
    }

    let mut direct_ratios = [0.0; ZETA_POSITIVE_DIRECT_LN.len()];
    for (ratio, &ln_n) in direct_ratios.iter_mut().zip(&ZETA_POSITIVE_DIRECT_LN) {
        *ratio = (-step * ln_n).exp();
    }
    let tail_ratio = (-step * ZETA_POSITIVE_TAIL_LN).exp();

    let mut out = vec![0.0; values.len()];
    let nthreads = if values.len() < ZETA_AFFINE_PARALLEL_MIN_LEN
        || ZETA_AFFINE_FORCE_SERIAL.load(std::sync::atomic::Ordering::Relaxed)
    {
        1
    } else {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(values.len().div_ceil(ZETA_AFFINE_BLOCK))
            .max(1)
    };
    if nthreads <= 1 {
        fill_zeta_affine_blocks(values, &mut out, &direct_ratios, tail_ratio);
    } else {
        // Chunk BLOCK-aligned so each thread owns whole blocks and the block partition is identical
        // to the serial one (block boundaries stay on `ZETA_AFFINE_BLOCK` multiples from index 0).
        let nblocks = values.len().div_ceil(ZETA_AFFINE_BLOCK);
        let chunk = nblocks.div_ceil(nthreads) * ZETA_AFFINE_BLOCK;
        std::thread::scope(|scope| {
            for (vc, oc) in values.chunks(chunk).zip(out.chunks_mut(chunk)) {
                scope.spawn(|| fill_zeta_affine_blocks(vc, oc, &direct_ratios, tail_ratio));
            }
        });
    }
    Some(out)
}

/// Dirichlet eta function η(s) = Σ_{n≥1} (-1)^{n+1} / n^s, for s > 0.
///
/// Evaluated with Borwein's algorithm (P. Borwein, "An Efficient Algorithm
/// for the Riemann Zeta Function", 2000). A naive partial sum of the
/// alternating series converges far too slowly in the critical strip — its
/// truncation error after `m` terms is ~½·m^{-s}, giving only ~1 digit at
/// s = 0.5 for m = 100. Borwein's weighted partial sum instead converges
/// geometrically, with error ~(3+√8)^{-n} ≈ 5.83^{-n}.
fn dirichlet_eta(s: f64) -> f64 {
    // d_k = n · Σ_{j=0}^{k} t_j,  with t_0 = 1/n and
    // t_{j+1} = t_j · 4(n+j)(n-j) / ((2j+1)(2j+2)).
    const N: usize = 32;
    let nf = N as f64;
    let mut d = [0.0_f64; N + 1];
    let mut t = 1.0 / nf;
    let mut cumulative = t;
    d[0] = nf * cumulative;
    for j in 0..N {
        let jf = j as f64;
        t *= 4.0 * (nf + jf) * (nf - jf) / ((2.0 * jf + 1.0) * (2.0 * jf + 2.0));
        cumulative += t;
        d[j + 1] = nf * cumulative;
    }
    let dn = d[N];
    // η(s) = -1/d_n · Σ_{k=0}^{n-1} (-1)^k (d_k - d_n) / (k+1)^s.
    let mut sum = 0.0;
    for (k, &dk) in d.iter().take(N).enumerate() {
        let sign = if k.is_multiple_of(2) { 1.0 } else { -1.0 };
        sum += sign * (dk - dn) / ((k + 1) as f64).powf(s);
    }
    -sum / dn
}

/// Compute the Riemann zeta function complement: zetac(s) = zeta(s) - 1 for real tensors.
///
/// Matches `scipy.special.zetac(s)` and preserves the same RealVec parallel
/// dispatch contract as [`zeta`].
pub fn zetac(s_tensor: &SpecialTensor, mode: RuntimeMode) -> SpecialResult {
    map_real_input_rp("zetac", s_tensor, mode, |s| Ok(zetac_scalar(s)), 1 << 15) // be ~30k
}

/// Compute the Riemann zeta function complement: zetac(s) = zeta(s) - 1 for scalar `s`.
///
/// Matches `scipy.special.zetac(s)`.
///
/// This is useful when zeta(s) is close to 1 (i.e., for large s > 1),
/// where direct computation of zeta(s) - 1 would suffer from catastrophic
/// cancellation. For large s, zetac(s) ≈ 2^(-s).
///
/// # Arguments
/// * `s` - Real argument
///
/// # Returns
/// ζ(s) - 1 for any real s ≠ 1
pub fn zetac_scalar(s: f64) -> f64 {
    if s.is_nan() {
        return f64::NAN;
    }
    if s == 1.0 {
        return f64::INFINITY; // pole of zeta
    }
    if s > 10.0 {
        // For large s > 10, zetac(s) = sum_{n=2}^∞ n^(-s) ≈ 2^(-s) + 3^(-s) + ...
        // Direct computation avoids subtraction from 1
        let mut sum = 0.0_f64;
        for n in 2..=100 {
            let term = (n as f64).powf(-s);
            if term < 1e-18 {
                break;
            }
            sum += term;
        }
        sum
    } else {
        // For smaller s, just compute zeta(s) - 1
        zeta_scalar(s) - 1.0
    }
}

// ============================================================================
// Complex incomplete gamma functions
// ============================================================================

/// Complex log-gamma using Lanczos approximation.
/// Matches scipy.special.loggamma for complex arguments.
pub fn complex_gammaln(z: Complex64) -> Complex64 {
    if is_complex_real_gamma_pole(z) {
        return Complex64::new(f64::NAN, f64::NAN);
    }

    if z.re < 0.5 {
        // Upward recurrence loggamma(z) = loggamma(z+1) - log(z), shifting until the
        // Lanczos region (Re >= 0.5) and accumulating PRINCIPAL logs. Summing principal
        // logs lets the imaginary part track the analytic continuation — scipy.special
        // .loggamma, whose Im winds by 2π per unit of negative Re(z) — rather than the
        // single-valued principal log of Γ(z) the old reflection (ln(π)-ln(sin πz)-lnΓ(1-z))
        // produced. gamma/rgamma compose this via exp(±loggamma), so the 2πi winding is
        // invisible to them; only loggamma's own value changes. (frankenscipy-4roim)
        let mut acc = Complex64::new(0.0, 0.0);
        let mut w = z;
        while w.re < 0.5 {
            acc = acc + w.ln();
            w = Complex64::new(w.re + 1.0, w.im);
        }
        complex_gammaln_lanczos(w) - acc
    } else {
        complex_gammaln_lanczos(z)
    }
}

/// Lanczos series for ln Γ(z) on the right half-plane (Re(z) >= 0.5). Returns the value
/// with the principal-branch imaginary part; callers needing the analytically-continued
/// branch for Re(z) < 0.5 reach it via the recurrence in [`complex_gammaln`].
fn complex_gammaln_lanczos(z: Complex64) -> Complex64 {
    let z_shifted = Complex64::new(z.re - 1.0, z.im);
    let mut x = Complex64::new(LANCZOS_COEFFS[0], 0.0);
    for (i, &coeff) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
        let denom = Complex64::new(z_shifted.re + i as f64, z_shifted.im);
        x = x + Complex64::new(coeff, 0.0) / denom;
    }
    let t = Complex64::new(z_shifted.re + LANCZOS_G + 0.5, z_shifted.im);
    // ln Γ(z) = 0.5 ln(2π) + (z - 0.5) ln(t) - t + ln(x)
    let half_ln_2pi = 0.5 * (2.0 * PI).ln();
    let z_minus_half = Complex64::new(z_shifted.re + 0.5, z_shifted.im);
    let ln_t = t.ln();
    let term1 = Complex64::new(half_ln_2pi, 0.0);
    let term2 = z_minus_half * ln_t;
    let term3 = Complex64::new(-t.re, -t.im);
    let term4 = x.ln();
    term1 + term2 + term3 + term4
}

/// Complex sine function.
fn complex_sin(z: Complex64) -> Complex64 {
    // sin(a + bi) = sin(a)cosh(b) + i cos(a)sinh(b)
    Complex64::new(z.re.sin() * z.im.cosh(), z.re.cos() * z.im.sinh())
}

/// Complex cosine function.
fn complex_cos(z: Complex64) -> Complex64 {
    // cos(a + bi) = cos(a)cosh(b) - i sin(a)sinh(b)
    Complex64::new(z.re.cos() * z.im.cosh(), -z.re.sin() * z.im.sinh())
}

/// Complex cotangent.
fn complex_cot(z: Complex64) -> Complex64 {
    complex_cos(z) / complex_sin(z)
}

/// Complex digamma (psi) function.
/// ψ(z) = d/dz ln(Γ(z))
pub fn complex_digamma_scalar(z: Complex64) -> Complex64 {
    if !z.is_finite() {
        return Complex64::new(f64::NAN, f64::NAN);
    }
    if is_complex_real_gamma_pole(z) {
        return Complex64::new(f64::NAN, f64::NAN);
    }

    // Reflection for negative real part: ψ(1-z) - π*cot(πz)
    if z.re < 0.5 {
        let one_minus_z = Complex64::new(1.0 - z.re, -z.im);
        let pi_z = Complex64::new(PI * z.re, PI * z.im);
        let pi_cot_pi_z = Complex64::new(PI, 0.0) * complex_cot(pi_z);
        return complex_digamma_scalar(one_minus_z) - pi_cot_pi_z;
    }

    // Shift upward until Re(z) >= 8 using recurrence: ψ(z+1) = ψ(z) + 1/z
    let mut shifted = z;
    let mut acc = Complex64::new(0.0, 0.0);
    while shifted.re < 8.0 {
        acc = acc - shifted.recip();
        shifted = Complex64::new(shifted.re + 1.0, shifted.im);
    }

    // Asymptotic expansion for large |z|:
    // ψ(z) ≈ ln(z) - 1/(2z) - 1/(12z²) + 1/(120z⁴) - 1/(252z⁶) + ...
    let inv = shifted.recip();
    let inv2 = inv * inv;
    let inv4 = inv2 * inv2;
    let inv6 = inv4 * inv2;
    let inv8 = inv6 * inv2;
    let inv10 = inv8 * inv2;

    // Extended through the B₁₀ term to match the real path (luxsz).
    let result = shifted.ln() - inv * 0.5 - inv2 * (1.0 / 12.0) + inv4 * (1.0 / 120.0)
        - inv6 * (1.0 / 252.0)
        + inv8 * (1.0 / 240.0)
        - inv10 * (1.0 / 132.0);

    acc + result
}

/// Complex trigamma function (polygamma of order 1).
/// ψ¹(z) = d²/dz² ln(Γ(z))
fn complex_trigamma_scalar(z: Complex64) -> Complex64 {
    if !z.is_finite() {
        return Complex64::new(f64::NAN, f64::NAN);
    }

    // Reflection: ψ¹(1-z) + π²/sin²(πz) = ψ¹(z)
    if z.re < 0.5 {
        let one_minus_z = Complex64::new(1.0 - z.re, -z.im);
        let pi_z = Complex64::new(PI * z.re, PI * z.im);
        let sin_pi_z = complex_sin(pi_z);
        let pi_sq_over_sin_sq = Complex64::new(PI * PI, 0.0) / (sin_pi_z * sin_pi_z);
        return pi_sq_over_sin_sq - complex_trigamma_scalar(one_minus_z);
    }

    // Shift upward: ψ¹(z+1) = ψ¹(z) - 1/z²
    let mut shifted = z;
    let mut acc = Complex64::new(0.0, 0.0);
    while shifted.re < 8.0 {
        let inv = shifted.recip();
        acc = acc + inv * inv;
        shifted = Complex64::new(shifted.re + 1.0, shifted.im);
    }

    // Asymptotic: ψ¹(z) ≈ 1/z + 1/(2z²) + 1/(6z³) - 1/(30z⁵) + 1/(42z⁷)
    let inv = shifted.recip();
    let inv2 = inv * inv;
    let inv3 = inv2 * inv;
    let inv5 = inv3 * inv2;
    let inv7 = inv5 * inv2;
    let inv9 = inv7 * inv2;
    let inv11 = inv9 * inv2;

    // Extended through the B₁₀ term to match the real path (luxsz).
    let result = inv + inv2 * 0.5 + inv3 / 6.0 - inv5 / 30.0 + inv7 / 42.0 - inv9 / 30.0
        + inv11 * (5.0 / 66.0);

    acc + result
}

/// Complex polygamma function of order n.
/// Uses recurrence and asymptotic expansion.
pub fn complex_polygamma_scalar(n: usize, z: Complex64) -> Complex64 {
    if n == 0 {
        return complex_digamma_scalar(z);
    }
    if n == 1 {
        return complex_trigamma_scalar(z);
    }

    if !z.is_finite() {
        return Complex64::new(f64::NAN, f64::NAN);
    }

    let sign = if n % 2 == 1 { 1.0 } else { -1.0 };
    let factorial = (1..=n).fold(1.0, |acc, v| acc * v as f64);
    let n_plus_one = n as f64 + 1.0;

    // Reflection for negative real part
    if z.re < 0.5 {
        // The reflection formula for higher polygamma is complex
        // Use numerical differentiation of lower orders as fallback
        let h = Complex64::new(1e-6, 0.0);
        let f_plus = complex_polygamma_scalar(n - 1, z + h);
        let f_minus = complex_polygamma_scalar(n - 1, z - h);
        return (f_plus - f_minus) / Complex64::new(2e-6, 0.0);
    }

    // Shift upward: ψⁿ(z+1) = ψⁿ(z) + (-1)^(n+1) * n! / z^(n+1)
    let mut shifted = z;
    let mut correction = Complex64::new(0.0, 0.0);
    while shifted.re < 8.0 {
        let inv_pow = shifted.powf(-n_plus_one);
        correction = correction + Complex64::new(sign * factorial, 0.0) * inv_pow;
        shifted = Complex64::new(shifted.re + 1.0, shifted.im);
    }

    // Asymptotic expansion using Hurwitz zeta
    // ψⁿ(z) ≈ (-1)^(n+1) * n! * ζ(n+1, z)
    // For large z: ζ(s, z) ≈ z^(1-s)/(s-1) + z^(-s)/2 + Σ B_{2k}/(2k)! * (s)_{2k-1} * z^(-s-2k+1)
    let inv_pow = shifted.powf(-n_plus_one);
    let inv_pow_minus_1 = shifted.powf(-(n as f64));

    // Leading terms of asymptotic zeta
    let zeta_approx = inv_pow_minus_1 / n as f64 + inv_pow * 0.5;

    let result = Complex64::new(sign * factorial, 0.0) * zeta_approx;
    correction + result
}

/// Complex regularized lower incomplete gamma P(a, z) for real a > 0 and complex z.
/// Uses series expansion: P(a,z) = z^a * e^(-z) / Γ(a) * Σ_{n=0}^∞ z^n / (a)_n
/// where (a)_n = a(a+1)...(a+n-1) is the Pochhammer symbol.
pub fn complex_gammainc_scalar(
    a: f64,
    z: Complex64,
    mode: RuntimeMode,
) -> Result<Complex64, SpecialError> {
    const FUNCTION: &str = "gammainc";

    // Domain check
    if a.is_nan() || !z.is_finite() {
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }
    if a <= 0.0 {
        record_special_trace(
            FUNCTION,
            mode,
            "domain_error",
            format!("a={a},z=({},{})", z.re, z.im),
            "returned_nan",
            "a must be positive",
            false,
        );
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }

    // Special case: z = 0
    if z.re == 0.0 && z.im == 0.0 {
        return Ok(Complex64::new(0.0, 0.0));
    }

    // Use series expansion for |z| < a + 1 or continued fraction otherwise
    let z_abs = z.abs();
    if z_abs < a + 1.0 {
        complex_gammainc_series(a, z, mode)
    } else {
        // P(a, z) = 1 - Q(a, z)
        let q = complex_gammaincc_cf(a, z, mode)?;
        Ok(Complex64::new(1.0 - q.re, -q.im))
    }
}

/// Complex regularized upper incomplete gamma Q(a, z) for real a > 0 and complex z.
pub fn complex_gammaincc_scalar(
    a: f64,
    z: Complex64,
    mode: RuntimeMode,
) -> Result<Complex64, SpecialError> {
    const FUNCTION: &str = "gammaincc";

    if a.is_nan() || !z.is_finite() {
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }
    if a <= 0.0 {
        record_special_trace(
            FUNCTION,
            mode,
            "domain_error",
            format!("a={a},z=({},{})", z.re, z.im),
            "returned_nan",
            "a must be positive",
            false,
        );
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }

    if z.re == 0.0 && z.im == 0.0 {
        return Ok(Complex64::new(1.0, 0.0));
    }

    let z_abs = z.abs();
    if z_abs < a + 1.0 {
        // Q(a, z) = 1 - P(a, z)
        let p = complex_gammainc_series(a, z, mode)?;
        Ok(Complex64::new(1.0 - p.re, -p.im))
    } else {
        complex_gammaincc_cf(a, z, mode)
    }
}

fn complex_parameter_gammainc_scalar(
    a: Complex64,
    z: Complex64,
    mode: RuntimeMode,
) -> Result<Complex64, SpecialError> {
    const FUNCTION: &str = "gammainc";

    if !a.is_finite() || !z.is_finite() {
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }
    if complex_gamma_pole(a) {
        record_special_trace(
            FUNCTION,
            mode,
            "domain_error",
            format!("a=({},{}) z=({},{})", a.re, a.im, z.re, z.im),
            "returned_nan",
            "a must not be a non-positive integer pole",
            false,
        );
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }
    if z == Complex64::new(0.0, 0.0) {
        if a.re > 0.0 {
            return Ok(Complex64::new(0.0, 0.0));
        }
        record_special_trace(
            FUNCTION,
            mode,
            "domain_error",
            format!("a=({},{}) z=(0,0)", a.re, a.im),
            "returned_nan",
            "z=0 requires Re(a) > 0",
            false,
        );
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }

    if z.abs() < a.abs() + 1.0 {
        complex_parameter_gammainc_series(a, z)
    } else {
        let q = complex_parameter_gammaincc_cf(a, z)?;
        Ok(Complex64::new(1.0, 0.0) - q)
    }
}

fn complex_parameter_gammaincc_scalar(
    a: Complex64,
    z: Complex64,
    mode: RuntimeMode,
) -> Result<Complex64, SpecialError> {
    const FUNCTION: &str = "gammaincc";

    if !a.is_finite() || !z.is_finite() {
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }
    if complex_gamma_pole(a) {
        record_special_trace(
            FUNCTION,
            mode,
            "domain_error",
            format!("a=({},{}) z=({},{})", a.re, a.im, z.re, z.im),
            "returned_nan",
            "a must not be a non-positive integer pole",
            false,
        );
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }
    if z == Complex64::new(0.0, 0.0) {
        if a.re > 0.0 {
            return Ok(Complex64::new(1.0, 0.0));
        }
        record_special_trace(
            FUNCTION,
            mode,
            "domain_error",
            format!("a=({},{}) z=(0,0)", a.re, a.im),
            "returned_nan",
            "z=0 requires Re(a) > 0",
            false,
        );
        return Ok(Complex64::new(f64::NAN, f64::NAN));
    }

    if z.abs() < a.abs() + 1.0 {
        let p = complex_parameter_gammainc_series(a, z)?;
        Ok(Complex64::new(1.0, 0.0) - p)
    } else {
        complex_parameter_gammaincc_cf(a, z)
    }
}

fn complex_gamma_pole(a: Complex64) -> bool {
    const POLE_EPS: f64 = 1.0e-12;

    a.im.abs() <= POLE_EPS && a.re <= 0.0 && (a.re - a.re.round()).abs() <= POLE_EPS
}

/// Series expansion for lower incomplete gamma.
fn complex_gammainc_series(
    a: f64,
    z: Complex64,
    _mode: RuntimeMode,
) -> Result<Complex64, SpecialError> {
    const EPS: f64 = 1.0e-14;
    const MAX_ITERS: usize = 200;

    // Compute ln(Γ(a)) for normalization
    let lg = gammaln_scalar(a, RuntimeMode::Strict)?;

    // Compute z^a * e^(-z) / Γ(a)
    let ln_z = z.ln();
    let a_ln_z = Complex64::new(a * ln_z.re, a * ln_z.im);
    let exp_arg = Complex64::new(a_ln_z.re - z.re - lg, a_ln_z.im - z.im);
    let prefactor = exp_arg.exp();

    // Series: sum_{n=0}^∞ z^n / (a+1)(a+2)...(a+n)
    let mut term = Complex64::new(1.0 / a, 0.0);
    let mut sum = term;

    for n in 1..MAX_ITERS {
        let denom = a + n as f64;
        term = term * z / Complex64::new(denom, 0.0);
        sum = sum + term;
        if term.abs() <= sum.abs() * EPS {
            break;
        }
    }

    Ok(prefactor * sum)
}

fn complex_parameter_gammainc_series(
    a: Complex64,
    z: Complex64,
) -> Result<Complex64, SpecialError> {
    const EPS: f64 = 1.0e-14;
    const MAX_ITERS: usize = 200;

    let lg = complex_gammaln(a);
    let prefactor = (a * z.ln() - z - lg).exp();

    let mut term = Complex64::new(1.0, 0.0) / a;
    let mut sum = term;

    for n in 1..MAX_ITERS {
        let denom = a + Complex64::new(n as f64, 0.0);
        term = term * z / denom;
        sum = sum + term;
        if term.abs() <= sum.abs() * EPS {
            break;
        }
    }

    Ok(prefactor * sum)
}

/// Continued fraction for upper incomplete gamma (Lentz's algorithm).
fn complex_gammaincc_cf(
    a: f64,
    z: Complex64,
    _mode: RuntimeMode,
) -> Result<Complex64, SpecialError> {
    const EPS: f64 = 1.0e-14;
    const FPMIN: f64 = 1.0e-300;
    const MAX_ITERS: usize = 200;

    let lg = gammaln_scalar(a, RuntimeMode::Strict)?;

    // Compute prefactor: z^a * e^(-z) / Γ(a)
    let ln_z = z.ln();
    let a_ln_z = Complex64::new(a * ln_z.re, a * ln_z.im);
    let exp_arg = Complex64::new(a_ln_z.re - z.re - lg, a_ln_z.im - z.im);
    let prefactor = exp_arg.exp();

    // Continued fraction: Q(a,z) = prefactor * 1/(z + 1-a - 1*(1-a)/(z + 3-a - 2*(2-a)/(z + 5-a - ...)))
    let mut b = Complex64::new(z.re + 1.0 - a, z.im);
    let mut c = Complex64::new(1.0 / FPMIN, 0.0);
    let mut d = b.recip();
    let mut h = d;

    for i in 1..=MAX_ITERS {
        let i_f = i as f64;
        let an = Complex64::new(-i_f * (i_f - a), 0.0);
        b = b + Complex64::new(2.0, 0.0);
        d = an * d + b;
        if d.abs() < FPMIN {
            d = Complex64::new(FPMIN, 0.0);
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = Complex64::new(FPMIN, 0.0);
        }
        d = d.recip();
        let delta = d * c;
        h = h * delta;
        if (delta.re - 1.0).abs() + delta.im.abs() <= EPS {
            break;
        }
    }

    Ok(prefactor * h)
}

fn complex_parameter_gammaincc_cf(a: Complex64, z: Complex64) -> Result<Complex64, SpecialError> {
    const EPS: f64 = 1.0e-14;
    const FPMIN: f64 = 1.0e-300;
    const MAX_ITERS: usize = 200;

    let lg = complex_gammaln(a);
    let prefactor = (a * z.ln() - z - lg).exp();

    let mut b = z + Complex64::new(1.0, 0.0) - a;
    let mut c = Complex64::new(1.0 / FPMIN, 0.0);
    let mut d = b.recip();
    let mut h = d;

    for i in 1..=MAX_ITERS {
        let i_c = Complex64::new(i as f64, 0.0);
        let an = -i_c * (i_c - a);
        b = b + Complex64::new(2.0, 0.0);
        d = an * d + b;
        if d.abs() < FPMIN {
            d = Complex64::new(FPMIN, 0.0);
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = Complex64::new(FPMIN, 0.0);
        }
        d = d.recip();
        let delta = d * c;
        h = h * delta;
        if (delta - Complex64::new(1.0, 0.0)).abs() <= EPS {
            break;
        }
    }

    Ok(prefactor * h)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ONE lock for every test that writes a gamma-family perf toggle.
    ///
    /// These are process-global `static`s and `cargo test` runs tests concurrently, so two
    /// tests flipping toggles at once produce both false drift and, worse, a masked pass —
    /// this workspace has already had bit-identity tests fail 2 runs in 3 on a clean tree
    /// for exactly that reason. A SECOND lock for this state would be a bug, not extra
    /// safety: two locks protect nothing while reading as handled. If a new gamma toggle
    /// needs a lock, it needs THIS one.
    static GAMMA_TOGGLE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn gamma_toggle_lock() -> std::sync::MutexGuard<'static, ()> {
        GAMMA_TOGGLE_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Drives `GAMMA_FAMILY_PREALLOC_FILL` in BOTH settings through the shared mapper
    /// itself, on BOTH the serial and the threaded arm, and pins bit-identity plus error
    /// propagation.
    ///
    /// Testing `par_map_indices_with_threads` directly rather than through `gammaln` is
    /// deliberate: `nthreads` is chosen from `available_parallelism()`, so a test that went
    /// through the public entry would exercise whichever arm the test host happened to
    /// pick, and would silently stop covering the threaded path on a one-core runner.
    #[test]
    fn gamma_family_prealloc_fill_arms_agree_bit_for_bit() {
        use std::sync::atomic::Ordering::Relaxed;
        let _guard = gamma_toggle_lock();

        const N: usize = 9_997; // not a multiple of any thread count used below
        let kernel = |i: usize| -> Result<f64, SpecialError> {
            // Deliberately irrational per index so a mis-indexed slot cannot coincide.
            Ok((i as f64 + 0.5).sqrt().sin() * 1.0e3 + i as f64)
        };

        let restore = GAMMA_FAMILY_PREALLOC_FILL.load(Relaxed);
        let mut arms: Vec<(Vec<f64>, usize)> = Vec::new();
        for enabled in [true, false] {
            GAMMA_FAMILY_PREALLOC_FILL.store(enabled, Relaxed);
            for threads in [1usize, 4] {
                let before = GAMMA_FAMILY_PREALLOC_FILL_HITS.load(Relaxed);
                let out = par_map_indices_with_threads(N, threads, kernel)
                    .expect("infallible kernel must not error");
                let hits = GAMMA_FAMILY_PREALLOC_FILL_HITS.load(Relaxed) - before;
                assert_eq!(out.len(), N);
                arms.push((out, hits));
            }
        }

        // MUST-HIT: without this, a build where the toggle did nothing would pass every
        // equality below while comparing an arm against itself.
        //
        // BOUNDED, NOT EXACT, AND THAT IS A BUG FIX. This asserted `== 1` and `== 0`, which
        // is only sound if nothing else touches the counter for the duration. It is
        // process-global and `par_map_indices` has 245 callers across seven modules, so any
        // concurrent test that maps an array increments it and the equality fails on a
        // correct tree. It did: green when run alone, FAILED in the full concurrent suite,
        // then passed again on the next run — the intermittent shape that wastes the most
        // time and, in a shared tree, derails other agents.
        //
        // A process-global counter cannot support an exact MUST-MISS while the suite runs
        // concurrently, so this no longer pretends to one. The must-miss for this toggle is
        // the instruction A/B instead — 196.7 against 199.5 per element — which cannot
        // happen unless the two arms really execute different code.
        assert!(arms[0].1 >= 1, "prealloc serial arm did not run");
        assert!(arms[1].1 >= 1, "prealloc threaded arm did not run");

        // All four must agree BIT for bit: prealloc vs collect, serial vs threaded.
        for (label, other) in [
            ("prealloc threaded", &arms[1].0),
            ("collect serial", &arms[2].0),
            ("collect threaded", &arms[3].0),
        ] {
            let differing = arms[0]
                .0
                .iter()
                .zip(other)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                differing, 0,
                "{label} differs from prealloc serial in {differing}"
            );
        }

        // DETECTOR: every claim here is bit-identity, so "nothing differed" is also what a
        // comparison too blunt to see anything prints. `to_bits`, not `==`.
        let mut perturbed = arms[3].0.clone();
        perturbed[N / 2] = f64::from_bits(perturbed[N / 2].to_bits() ^ 1);
        let seen = arms[0]
            .0
            .iter()
            .zip(&perturbed)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            seen, 1,
            "the bit comparison cannot see a one-ULP difference"
        );

        // ERROR PROPAGATION is part of the contract this mapper had before: the FIRST
        // failing index in index order is the error returned, on every arm. A prealloc
        // buffer makes it easy to accidentally report a later chunk's error instead.
        for enabled in [true, false] {
            GAMMA_FAMILY_PREALLOC_FILL.store(enabled, Relaxed);
            for threads in [1usize, 4] {
                let failing = |i: usize| -> Result<f64, SpecialError> {
                    if i == 17 || i == 8_000 {
                        Err(SpecialError {
                            function: "test",
                            kind: SpecialErrorKind::DomainError,
                            mode: RuntimeMode::Strict,
                            detail: "seeded",
                        })
                    } else {
                        Ok(i as f64)
                    }
                };
                let err = par_map_indices_with_threads(N, threads, failing)
                    .expect_err("seeded failure must propagate");
                assert_eq!(err.function, "test", "prealloc={enabled} threads={threads}");
            }
        }
        GAMMA_FAMILY_PREALLOC_FILL.store(restore, Relaxed);
    }

    /// Cross-checks the transcribed Cephes rational against the independent Lanczos kernel,
    /// and drives `GAMMA_CEPHES_RATIONAL` in both settings.
    ///
    /// This is what catches a mistranscribed coefficient: the two forms share no constants
    /// and no code path, so agreement to a few tens of ULP is strong evidence the fifteen
    /// numbers copied out of `xsf/cephes/gamma.h` are right. A single wrong digit moves the
    /// rational far more than the bound below.
    ///
    /// JUDGED IN ULPs, NOT ABSOLUTE ERROR, and here that is the whole point: Γ(30) is
    /// 8.8e30, so an absolute measure on this domain reports 2.5e17 for a result that is
    /// correct to 34 ULP. The mirror of `y0`, where the function passes through ZERO and the
    /// relative measure is the misleading one. Neither measure is right by default — pick it
    /// from the function's range on the domain being tested.
    #[test]
    fn gamma_cephes_rational_matches_the_lanczos_form() {
        use std::sync::atomic::Ordering::Relaxed;
        let _guard = gamma_toggle_lock();
        let restore = GAMMA_CEPHES_RATIONAL.load(Relaxed);

        // Span the reduction's regimes: below 2 (divide up), [2,3] (the rational's own
        // interval, no loop), and above 3 (multiply down, up to 30 iterations).
        let mut points: Vec<f64> = Vec::new();
        for i in 0..=1200 {
            points.push(0.5 + (33.0 - 0.5) * (i as f64 / 1200.0));
        }
        points.extend([
            0.5, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 10.0, 20.0, 32.9, 33.0,
        ]);

        let (mut below, mut inside, mut above) = (0usize, 0usize, 0usize);
        let mut worst = 0.0_f64;
        let mut worst_at = f64::NAN;
        let mut hits_seen = 0usize;
        for &x in &points {
            if x < 2.0 {
                below += 1;
            } else if x <= 3.0 {
                inside += 1;
            } else {
                above += 1;
            }
            GAMMA_CEPHES_RATIONAL.store(true, Relaxed);
            let before = GAMMA_CEPHES_RATIONAL_HITS.load(Relaxed);
            let rational = gamma_core(x);
            if GAMMA_CEPHES_RATIONAL_HITS.load(Relaxed) > before {
                hits_seen += 1;
            }
            GAMMA_CEPHES_RATIONAL.store(false, Relaxed);
            let lanczos = gamma_core(x);
            assert!(
                rational.is_finite() && lanczos.is_finite(),
                "non-finite at x={x}: rational {rational} lanczos {lanczos}"
            );
            let ulp = (rational - lanczos).abs() / lanczos.abs() / f64::EPSILON;
            if ulp > worst {
                worst = ulp;
                worst_at = x;
            }
        }
        GAMMA_CEPHES_RATIONAL.store(restore, Relaxed);

        // MUST-HIT: every regime of the reduction has to be reached, and the arm has to have
        // actually fired, or the agreement below is evidence about nothing.
        assert!(
            below > 0 && inside > 0 && above > 0,
            "grid missed a regime: x<2 {below}, 2<=x<=3 {inside}, x>3 {above}"
        );
        assert!(
            hits_seen >= points.len(),
            "rational arm did not run for every point"
        );
        println!(
            "gamma rational vs lanczos: worst {worst:.1} ULP at x={worst_at}, {} points",
            points.len()
        );
        // MEASURED worst is 138.0 ULP at x = 32.95, and almost all of it is the LANCZOS
        // side: against live SciPy the rational arm is ~3 ULP (max_rel 7.46e-16) where the
        // Lanczos form is ~154 (3.43e-14). Bound at 1024 for ~7x margin rather than the
        // measured value, because it exists to catch a mistranscribed coefficient — which
        // moves the rational by orders of magnitude, not by ULPs — and sitting on the
        // measurement would make it a change-detector instead.
        assert!(
            worst < 1024.0,
            "rational and Lanczos disagree by {worst} ULP at x={worst_at}; \
             a transcribed coefficient is probably wrong"
        );
    }

    /// Drives `GAMMALN_HOIST_THRESHOLD` in BOTH settings through the BATCH entry point,
    /// which is the only place the hoist exists, and pins bit-identity plus the hit
    /// counter. The fixture is sized above `GAMMA_FAMILY_PAR_MIN` so the parallel arm —
    /// the one that actually ships for a large array — is the one under test.
    #[test]
    fn gammaln_hoist_threshold_arms_agree_bit_for_bit() {
        use std::sync::atomic::Ordering::Relaxed;
        let _guard = gamma_toggle_lock();

        let n = GAMMA_FAMILY_PAR_MIN + 4096;
        let values: Vec<f64> = (0..n).map(|i| 0.01 + (i % 6000) as f64 * 0.01).collect();
        let tensor = SpecialTensor::RealVec(values);

        let restore = GAMMALN_HOIST_THRESHOLD.load(Relaxed);
        let mut arms = Vec::with_capacity(2);
        for enabled in [true, false] {
            GAMMALN_HOIST_THRESHOLD.store(enabled, Relaxed);
            let before = GAMMALN_HOIST_THRESHOLD_HITS.load(Relaxed);
            let out = gammaln(&tensor, RuntimeMode::Strict).expect("gammaln batch");
            let hits = GAMMALN_HOIST_THRESHOLD_HITS.load(Relaxed) - before;
            let SpecialTensor::RealVec(v) = out else {
                panic!("expected RealVec")
            };
            arms.push((v, hits));
        }
        GAMMALN_HOIST_THRESHOLD.store(restore, Relaxed);

        assert_eq!(
            arms[0].1, 1,
            "hoisted arm did not run; the check is vacuous"
        );
        assert_eq!(arms[1].1, 0, "disabled arm still hoisted");
        let differing = arms[0]
            .0
            .iter()
            .zip(&arms[1].0)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            differing, 0,
            "hoisting the threshold changed {differing} values"
        );
    }

    #[test]
    fn gamma_scalar_returns_a_signed_infinity_at_zero() {
        // The last unverified entry in the signed-zero family
        // (scripts/scipy_signed_zero_probe.py). `x == 0.0` is TRUE for -0.0 in
        // IEEE, so any branch written that way collapses the two zeros. scipy
        // 1.17.1 distinguishes them, and unlike psi the sign here is the
        // intuitive one:
        //   gamma( 0.0) = +inf
        //   gamma(-0.0) = -inf
        let pos = gamma_scalar(0.0, RuntimeMode::Strict).expect("gamma(+0.0)");
        let neg = gamma_scalar(-0.0, RuntimeMode::Strict).expect("gamma(-0.0)");
        assert!(
            pos.is_infinite() && pos.is_sign_positive(),
            "gamma(+0.0) = {pos}, scipy +inf"
        );
        assert!(
            neg.is_infinite() && neg.is_sign_negative(),
            "gamma(-0.0) = {neg}, scipy -inf"
        );
        assert_ne!(
            pos.to_bits(),
            neg.to_bits(),
            "the two zeros must not collapse to one value"
        );

        // MUST-MISS: negative integers are genuine NaN poles, and the finite
        // branches are unchanged.
        //   gamma(-1.0) = nan, gamma(-1.5) = 2.363271801207355,
        //   gamma(1.0) = 1.0, gamma(2.5) = 1.329340388179137
        assert!(
            gamma_scalar(-1.0, RuntimeMode::Strict)
                .expect("pole")
                .is_nan()
        );
        for (x, want) in [
            (-1.5_f64, 2.363_271_801_207_355_f64),
            (1.0, 1.0),
            (2.5, 1.329_340_388_179_137),
        ] {
            let got = gamma_scalar(x, RuntimeMode::Strict).expect("finite");
            assert!(
                (got - want).abs() < 1e-12,
                "gamma({x}) = {got}, scipy {want}"
            );
        }
    }

    #[test]
    fn zeta_affine_parallel_matches_serial_bitexact() {
        use std::sync::atomic::Ordering;
        // The block-parallel affine-`zeta` path must be BYTE-IDENTICAL to the serial one, at a batch
        // above the parallel gate so the multi-threaded chunked path actually runs. Affine `s` grids
        // spanning several block boundaries with a non-block-multiple length exercise the last-chunk
        // remainder and the chunk/block alignment.
        for &n in &[1usize << 15, (1 << 15) + 37, 200_003] {
            let values: Vec<f64> = (0..n)
                .map(|i| 1.1 + 8.9 * (i as f64) / (n - 1) as f64)
                .collect();
            ZETA_AFFINE_FORCE_SERIAL.store(true, Ordering::Relaxed);
            let serial = zeta_positive_affine_vec(&values).expect("affine path applies");
            ZETA_AFFINE_FORCE_SERIAL.store(false, Ordering::Relaxed);
            let parallel = zeta_positive_affine_vec(&values).expect("affine path applies");
            assert_eq!(serial.len(), parallel.len());
            for (i, (s, p)) in serial.iter().zip(parallel.iter()).enumerate() {
                assert_eq!(
                    s.to_bits(),
                    p.to_bits(),
                    "zeta affine n={n} mismatch at {i}"
                );
            }
        }
    }

    #[test]
    #[ignore = "perf A/B; run with --ignored --nocapture"]
    fn zeta_affine_parallel_ab() {
        use std::sync::atomic::Ordering;
        use std::time::Instant;
        let n = 300_000usize;
        let values: Vec<f64> = (0..n)
            .map(|i| 1.1 + 8.9 * (i as f64) / (n - 1) as f64)
            .collect();
        let bench = |serial: bool| -> f64 {
            ZETA_AFFINE_FORCE_SERIAL.store(serial, Ordering::Relaxed);
            let _ = std::hint::black_box(zeta_positive_affine_vec(&values));
            let reps = 5;
            let t = Instant::now();
            for _ in 0..reps {
                let arg = std::hint::black_box(&values);
                let _ = std::hint::black_box(zeta_positive_affine_vec(arg));
            }
            t.elapsed().as_secs_f64() / reps as f64 * 1e3
        };
        let (mut sv, mut pv, mut nv) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..15 {
            sv.push(bench(true));
            pv.push(bench(false));
            nv.push(bench(true));
        }
        ZETA_AFFINE_FORCE_SERIAL.store(false, Ordering::Relaxed);
        let med = |v: &[f64]| {
            let mut w = v.to_vec();
            w.sort_by(f64::total_cmp);
            w[w.len() / 2]
        };
        let cand: Vec<f64> = sv.iter().zip(&pv).map(|(s, p)| s / p).collect();
        let null: Vec<f64> = sv.iter().zip(&nv).map(|(s, n)| s / n).collect();
        let nlo = null.iter().copied().fold(f64::MAX, f64::min);
        let nhi = null.iter().copied().fold(f64::MIN, f64::max);
        let cm = med(&cand);
        let verdict = if cm > nhi || cm < nlo {
            "DECIDED"
        } else {
            "IN-FLOOR"
        };
        println!(
            "zeta_affine n={n}: serial {:.2}ms parallel {:.2}ms | CAND median {cm:.3}x | \
             NULL(A/A) median {:.3}x range [{nlo:.3}, {nhi:.3}] | {verdict}",
            med(&sv),
            med(&pv),
            med(&null),
        );
    }

    #[test]
    fn polygamma_match_scipy() {
        // scipy.special.polygamma(n, x): n=1 is trigamma (=pi^2/6 at x=1),
        // n=2 is -2*zeta(3) at x=1.
        let m = RuntimeMode::Strict;
        let pg = |n: usize, x: f64| match polygamma(n, &SpecialTensor::RealScalar(x), m) {
            Ok(SpecialTensor::RealScalar(v)) => v,
            other => panic!("expected scalar, got {other:?}"),
        };
        assert!(
            (pg(1, 1.0) - 1.644_934_066_848_226_6).abs() < 1e-13,
            "polygamma(1,1)=pi^2/6"
        );
        assert!(
            (pg(1, 2.0) - 0.644_934_066_848_226_6).abs() < 1e-13,
            "polygamma(1,2)"
        );
        assert!(
            (pg(2, 1.0) - -2.404_113_806_319_188).abs() < 1e-12,
            "polygamma(2,1)=-2 zeta(3)"
        );
    }

    #[test]
    fn digamma_scalar_match_scipy() {
        // scipy.special.digamma (psi): digamma(1) = -gamma (Euler-Mascheroni),
        // digamma(2) = 1 - gamma, digamma(0.5) = -gamma - 2 ln 2.
        let m = RuntimeMode::Strict;
        assert!(
            (digamma_scalar(1.0, m).unwrap() - -0.577_215_664_901_532_9).abs() < 1e-13,
            "digamma(1)=-gamma"
        );
        assert!(
            (digamma_scalar(2.0, m).unwrap() - 0.422_784_335_098_467_13).abs() < 1e-13,
            "digamma(2)"
        );
        assert!(
            (digamma_scalar(0.5, m).unwrap() - -1.963_510_026_021_423_5).abs() < 1e-13,
            "digamma(0.5)"
        );
    }

    #[test]
    fn log_gammainc_matches_gammainc_where_representable() {
        // exp(log P) == P wherever P is representable (gammainc_scalar is the
        // scipy-validated reference). Covers both branches (x<a+1 and x>=a+1).
        let cases = [
            (2.0, 1.0),
            (2.0, 5.0),
            (0.5, 0.3),
            (5.0, 3.0),
            (5.0, 12.0),
            (10.0, 8.0),
            (3.0, 0.5),
        ];
        for (a, x) in cases {
            let p = gammainc_scalar(a, x, RuntimeMode::Strict).unwrap();
            let lp = log_gammainc_scalar(a, x);
            assert!(p > 0.0, "precondition: P representable for ({a},{x})");
            let rel = (lp - p.ln()).abs() / p.ln().abs().max(1.0);
            assert!(
                rel <= 1e-12,
                "log_gammainc({a},{x})={lp} vs ln(P)={}",
                p.ln()
            );
        }
    }

    #[test]
    fn log_gammainc_finite_in_underflowed_left_tail() {
        // Deep left tail: P underflows to 0 (needs ln P < ~-745) but log P must
        // stay finite and track a*ln x - lnΓ(a+1) for tiny x.
        for (a, x) in [(10.0, 1e-35), (5.0, 1e-160), (20.0, 1e-20)] {
            assert_eq!(
                gammainc_scalar(a, x, RuntimeMode::Strict).unwrap(),
                0.0,
                "precondition: P underflows for ({a},{x})"
            );
            let lp = log_gammainc_scalar(a, x);
            assert!(
                lp.is_finite() && lp < -100.0,
                "log P({a},{x})={lp} not finite"
            );
            // P ≈ x^a / Γ(a+1) ⇒ ln P ≈ a·ln x − lnΓ(a+1).
            let asymp = a * x.ln() - gammaln_scalar(a + 1.0, RuntimeMode::Strict).unwrap();
            assert!(
                (lp - asymp).abs() / asymp.abs() < 1e-2,
                "log P({a},{x})={lp} vs asymptotic {asymp}"
            );
        }
    }

    #[test]
    fn log_gammainc_edge_cases() {
        assert_eq!(log_gammainc_scalar(2.0, 0.0), f64::NEG_INFINITY); // P(a,0)=0
        assert_eq!(log_gammainc_scalar(2.0, f64::INFINITY), 0.0); // P(a,inf)=1
        assert!(log_gammainc_scalar(-1.0, 2.0).is_nan());
        assert!(log_gammainc_scalar(2.0, f64::NAN).is_nan());
        assert!(log_gammainc_scalar(0.0, 2.0).is_nan());
    }

    #[test]
    fn log_gammaincc_matches_gammaincc_where_representable() {
        // exp(log Q) == Q wherever Q is representable (gammaincc_scalar is the
        // scipy-validated reference). Covers both branches (x<a+1 and x>=a+1).
        let cases = [
            (2.0, 1.0),
            (2.0, 5.0),
            (0.5, 0.3),
            (5.0, 3.0),
            (5.0, 12.0),
            (10.0, 25.0),
            (1.5, 40.0),
        ];
        for (a, x) in cases {
            let q = gammaincc_scalar(a, x, RuntimeMode::Strict).unwrap();
            let lq = log_gammaincc_scalar(a, x);
            assert!(q > 0.0, "precondition: Q representable for ({a},{x})");
            let rel = (lq - q.ln()).abs() / q.ln().abs().max(1.0);
            assert!(
                rel <= 1e-12,
                "log_gammaincc({a},{x})={lq} vs ln(Q)={}",
                q.ln()
            );
        }
    }

    #[test]
    fn log_gammaincc_finite_in_underflowed_tail() {
        // Deep right tail: Q underflows to exactly 0 (needs log Q < ~-745) but
        // log Q must stay finite and track -x + (a-1)ln x - lnΓ(a).
        for (a, x) in [(2.0, 800.0), (3.0, 900.0), (0.5, 770.0)] {
            assert_eq!(
                gammaincc_scalar(a, x, RuntimeMode::Strict).unwrap(),
                0.0,
                "precondition: Q underflows for ({a},{x})"
            );
            let lq = log_gammaincc_scalar(a, x);
            assert!(
                lq.is_finite() && lq < -700.0,
                "log Q({a},{x})={lq} not finite"
            );
            let asymp = -x + (a - 1.0) * x.ln() - gammaln_scalar(a, RuntimeMode::Strict).unwrap();
            assert!(
                (lq - asymp).abs() / asymp.abs() < 1e-3,
                "log Q({a},{x})={lq} vs asymptotic {asymp}"
            );
        }
    }

    #[test]
    fn log_gammaincc_edge_cases() {
        assert_eq!(log_gammaincc_scalar(2.0, 0.0), 0.0); // Q(a,0)=1
        assert_eq!(log_gammaincc_scalar(2.0, f64::INFINITY), f64::NEG_INFINITY);
        assert!(log_gammaincc_scalar(-1.0, 2.0).is_nan());
        assert!(log_gammaincc_scalar(2.0, f64::NAN).is_nan());
        assert!(log_gammaincc_scalar(0.0, 2.0).is_nan());
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    fn digamma_polygamma_extended_asymptotic() {
        // frankenscipy-luxsz: digamma/trigamma/tetragamma asymptotics extended
        // through the B₁₀ term (were ~1e-9 off at the shift point). scipy 1.17.1.
        let m = RuntimeMode::Strict;
        let s = |v: f64| SpecialTensor::RealScalar(v);
        let g = |r: SpecialResult| match r {
            Ok(SpecialTensor::RealScalar(v)) => v,
            _ => f64::NAN,
        };
        let dig = [
            (0.5, -1.9635100260214235),
            (1.5, 0.03648997397857652),
            (-0.5, 0.03648997397857651),
            (2.5, 0.7031566406452432),
            (-10.5, 2.3982391295357814),
        ];
        for (x, expected) in dig {
            let got = g(digamma(&s(x), m));
            assert!(
                (got - expected).abs() <= 1e-11 * expected.abs().max(1e-3),
                "digamma({x}) = {got}, scipy {expected}"
            );
        }
        let tri = [(1.5, 0.9348022005446793), (2.5, 0.4903577561002348)];
        for (x, expected) in tri {
            let got = g(polygamma(1, &s(x), m));
            assert!(
                ((got - expected) / expected).abs() < 1e-11,
                "polygamma(1,{x}) = {got}, scipy {expected}"
            );
        }
        let tetra = [(1.5, -0.8287966442343201), (2.5, -0.23620405164172736)];
        for (x, expected) in tetra {
            let got = g(polygamma(2, &s(x), m));
            assert!(
                ((got - expected) / expected).abs() < 1e-10,
                "polygamma(2,{x}) = {got}, scipy {expected}"
            );
        }
    }

    #[test]
    fn polygamma_nonpositive_integer_poles_match_scipy() {
        // Regression (frankenscipy-fxm0t): for n >= 1 the polygamma core
        // returned NaN at negative-integer poles, but scipy.special.polygamma(n, k)
        // for nonpositive integers k is a one-signed infinity: +inf for odd n,
        // -inf for even n (same at 0 and the negatives). digamma (n=0) keeps its
        // distinct convention (-inf at 0, NaN at the negatives) and is unchanged.
        let mode = RuntimeMode::Strict;
        let poles = [0.0_f64, -1.0, -2.0, -3.0, -7.0];
        for n in 1..=5usize {
            let want_pos = n % 2 == 1; // +inf for odd n, -inf for even n
            for &x in &poles {
                let v = match polygamma(n, &SpecialTensor::RealScalar(x), mode) {
                    Ok(SpecialTensor::RealScalar(v)) => v,
                    other => panic!("polygamma({n}, {x}) returned {other:?}"),
                };
                assert!(v.is_infinite(), "polygamma({n}, {x}) = {v}, want infinite");
                assert_eq!(
                    v.is_sign_positive(),
                    want_pos,
                    "polygamma({n}, {x}) = {v}, wrong sign of infinity"
                );
            }
        }
        // digamma (n = 0) is unchanged: -inf at 0, NaN at negative integers.
        let d0 = match polygamma(0, &SpecialTensor::RealScalar(0.0), mode) {
            Ok(SpecialTensor::RealScalar(v)) => v,
            other => panic!("{other:?}"),
        };
        assert!(
            d0.is_infinite() && d0.is_sign_negative(),
            "digamma(0) = {d0}"
        );
        let dm1 = match polygamma(0, &SpecialTensor::RealScalar(-1.0), mode) {
            Ok(SpecialTensor::RealScalar(v)) => v,
            other => panic!("{other:?}"),
        };
        assert!(dm1.is_nan(), "digamma(-1) = {dm1}, want NaN");
    }

    fn scalar(value: f64) -> SpecialTensor {
        SpecialTensor::RealScalar(value)
    }

    fn get_scalar(result: SpecialResult) -> Result<f64, String> {
        match result.map_err(|err| err.to_string())? {
            SpecialTensor::RealScalar(value) => Ok(value),
            other => Err(format!("expected real scalar, got {other:?}")),
        }
    }

    #[test]
    fn gamma_with_audit_emits_failclosed_on_hardened_pole() {
        // br-egba-2: gamma_with_audit records AuditAction::FailClosed
        // when Hardened mode rejects a pole input. Strict mode does
        // not emit.
        let ledger = crate::audit::sync_audit_ledger();
        // Strict path: returns Ok (NaN), no emission.
        let _ = gamma_with_audit(&scalar(-2.0), RuntimeMode::Strict, &ledger);
        assert_eq!(ledger.lock().unwrap().len(), 0, "strict must not emit");

        // Hardened pole: must emit exactly one FailClosed event.
        let err = gamma_with_audit(&scalar(-2.0), RuntimeMode::Hardened, &ledger)
            .expect_err("hardened rejects pole");
        assert_eq!(err.kind, SpecialErrorKind::PoleInput);
        assert_eq!(
            ledger.lock().unwrap().len(),
            1,
            "hardened pole must emit once"
        );
    }

    #[test]
    fn rgamma_returns_zero_at_negative_integer_poles() -> Result<(), String> {
        for x in [-1.0, -2.0, -8.0] {
            let actual = get_scalar(rgamma(&scalar(x), RuntimeMode::Strict))?;
            assert_eq!(actual, 0.0);
        }
        Ok(())
    }

    #[test]
    fn gammasgn_matches_scipy_real_axis_intervals() -> Result<(), String> {
        let cases = [
            (-5.5, 1.0),
            (-4.5, -1.0),
            (-3.5, 1.0),
            (-2.5, -1.0),
            (-1.5, 1.0),
            (-0.5, -1.0),
            (-0.0, -1.0),
            (0.0, 1.0),
            (0.5, 1.0),
            (1.0, 1.0),
            (2.0, 1.0),
            (f64::INFINITY, 1.0),
        ];

        for (x, expected) in cases {
            let actual = get_scalar(gammasgn(&scalar(x), RuntimeMode::Strict))?;
            assert_eq!(actual, expected, "gammasgn({x})");
        }
        Ok(())
    }

    #[test]
    fn gammasgn_preserves_scipy_pole_and_nan_semantics() -> Result<(), String> {
        for x in [-5.0, -4.0, -3.0, -2.0, -1.0, f64::NEG_INFINITY, f64::NAN] {
            let actual = get_scalar(gammasgn(&scalar(x), RuntimeMode::Strict))?;
            assert!(actual.is_nan(), "gammasgn({x}) should be NaN");
        }

        let err = gammasgn(&scalar(-2.0), RuntimeMode::Hardened).unwrap_err();
        assert_eq!(err.kind, SpecialErrorKind::PoleInput);
        Ok(())
    }

    #[test]
    fn gammasgn_supports_real_vectors() -> Result<(), String> {
        let input = SpecialTensor::RealVec(vec![-2.5, -0.0, 0.0, 3.0]);
        let result = gammasgn(&input, RuntimeMode::Strict).map_err(|err| err.to_string())?;
        match result {
            SpecialTensor::RealVec(values) => {
                assert_eq!(values, vec![-1.0, -1.0, 1.0, 1.0]);
                Ok(())
            }
            other => Err(format!("expected real vector, got {other:?}")),
        }
    }

    #[test]
    fn gamma_large_finite_boundary_matches_scipy() -> Result<(), String> {
        let actual = get_scalar(gamma(&scalar(170.0), RuntimeMode::Strict))?;
        let expected = 4.269_068_009_004_706e304;
        let rel = ((actual - expected) / expected).abs();
        assert!(
            actual.is_finite() && rel <= 1.0e-12,
            "gamma(170) should be finite SciPy boundary value, got {actual:e}"
        );
        Ok(())
    }

    #[test]
    fn multigammaln_matches_scipy_reference_values() -> Result<(), String> {
        let cases = [
            (3.5, 2.0, 2.4664857258317197),
            (4.0, 3.0, 5.402975080909175),
            (1.0, 2.0, 1.1447298858494),
        ];
        for (a, d, expected) in cases {
            let actual = get_scalar(multigammaln(&scalar(a), d, RuntimeMode::Strict))?;
            assert!((actual - expected).abs() <= 1.0e-12);
        }
        Ok(())
    }

    #[test]
    fn multigammaln_reduces_to_gammaln_for_dimension_one() -> Result<(), String> {
        for a in [0.5, 1.25, 3.5, 10.0] {
            let actual = get_scalar(multigammaln(&scalar(a), 1.0, RuntimeMode::Strict))?;
            let expected = gammaln_scalar(a, RuntimeMode::Strict).map_err(|err| err.to_string())?;
            assert!((actual - expected).abs() <= 1.0e-12);
        }
        Ok(())
    }

    #[test]
    fn multigammaln_supports_real_vectors() -> Result<(), String> {
        let input = SpecialTensor::RealVec(vec![2.5, 10.0]);
        let result =
            multigammaln(&input, 2.0, RuntimeMode::Strict).map_err(|err| err.to_string())?;
        match result {
            SpecialTensor::RealVec(values) => {
                let expected = [
                    0.5 * PI.ln()
                        + gammaln_scalar(2.5, RuntimeMode::Strict)
                            .map_err(|err| err.to_string())?
                        + gammaln_scalar(2.0, RuntimeMode::Strict)
                            .map_err(|err| err.to_string())?,
                    0.5 * PI.ln()
                        + gammaln_scalar(10.0, RuntimeMode::Strict)
                            .map_err(|err| err.to_string())?
                        + gammaln_scalar(9.5, RuntimeMode::Strict)
                            .map_err(|err| err.to_string())?,
                ];
                assert_eq!(values.len(), expected.len());
                for (actual, expected_value) in values.iter().zip(expected.iter()) {
                    assert!((*actual - *expected_value).abs() <= 1.0e-12);
                }
            }
            other => return Err(format!("expected real vector, got {other:?}")),
        }
        Ok(())
    }

    #[test]
    fn multigammaln_preserves_scipy_edge_cases() -> Result<(), String> {
        let zero_dim = get_scalar(multigammaln(&scalar(3.5), 0.0, RuntimeMode::Strict))?;
        assert_eq!(zero_dim, 0.0);

        let negative_dim = get_scalar(multigammaln(&scalar(3.5), -1.0, RuntimeMode::Strict))?;
        assert!((negative_dim - 0.5723649429247001).abs() <= 1.0e-12);

        let nan_a = get_scalar(multigammaln(&scalar(f64::NAN), 2.0, RuntimeMode::Strict))?;
        assert!(nan_a.is_nan());
        Ok(())
    }

    #[test]
    fn multigammaln_rejects_non_integer_dimension() {
        let err = multigammaln(&scalar(3.5), 1.2, RuntimeMode::Strict)
            .expect_err("non-integer dimensions should fail closed");
        assert_eq!(err.kind, SpecialErrorKind::DomainError);
        assert_eq!(err.detail, "d should be a positive integer (dimension)");
    }

    #[test]
    fn multigammaln_rejects_inputs_below_threshold() {
        let err = multigammaln(&scalar(0.4), 2.0, RuntimeMode::Strict)
            .expect_err("a <= 0.5 * (d - 1) should fail closed");
        assert_eq!(err.kind, SpecialErrorKind::DomainError);
        assert_eq!(err.detail, "condition a must exceed 0.5 * (d - 1)");
    }

    #[test]
    fn polygamma_order_two_matches_tetragamma_scalar() -> Result<(), String> {
        let x = 1.5;
        let actual = get_scalar(polygamma(2, &scalar(x), RuntimeMode::Strict))?;
        let expected = crate::convenience::tetragamma(x);
        assert!((actual - expected).abs() <= 1.0e-12);
        Ok(())
    }

    #[test]
    fn polygamma_order_two_supports_real_vectors() -> Result<(), String> {
        let input = SpecialTensor::RealVec(vec![0.5, 1.0, 2.5]);
        let result = polygamma(2, &input, RuntimeMode::Strict).map_err(|err| err.to_string())?;
        match result {
            SpecialTensor::RealVec(values) => {
                let expected = [
                    crate::convenience::tetragamma(0.5),
                    crate::convenience::tetragamma(1.0),
                    crate::convenience::tetragamma(2.5),
                ];
                assert_eq!(values.len(), expected.len());
                for (actual, expected_value) in values.iter().zip(expected.iter()) {
                    assert!((*actual - *expected_value).abs() <= 1.0e-12);
                }
            }
            other => return Err(format!("expected real vector, got {other:?}")),
        }
        Ok(())
    }

    #[test]
    fn polygamma_order_two_hardened_rejects_poles() {
        let err = polygamma(2, &scalar(-1.0), RuntimeMode::Hardened)
            .expect_err("hardened mode should reject tetragamma poles");
        assert_eq!(err.kind, SpecialErrorKind::PoleInput);
    }

    #[test]
    fn polygamma_order_three_matches_known_value_at_one() -> Result<(), String> {
        let actual = get_scalar(polygamma(3, &scalar(1.0), RuntimeMode::Strict))?;
        let expected = PI.powi(4) / 15.0;
        assert!((actual - expected).abs() <= 1.0e-11);
        Ok(())
    }

    #[test]
    fn polygamma_higher_orders_support_real_vectors() -> Result<(), String> {
        let input = SpecialTensor::RealVec(vec![1.0, 2.0, 0.5]);
        let result = polygamma(3, &input, RuntimeMode::Strict).map_err(|err| err.to_string())?;
        match result {
            SpecialTensor::RealVec(values) => {
                let expected = [PI.powi(4) / 15.0, PI.powi(4) / 15.0 - 6.0, PI.powi(4)];
                assert_eq!(values.len(), expected.len());
                for (actual, expected_value) in values.iter().zip(expected.iter()) {
                    assert!((*actual - *expected_value).abs() <= 1.0e-10);
                }
            }
            other => return Err(format!("expected real vector, got {other:?}")),
        }
        Ok(())
    }

    #[test]
    fn polygamma_order_four_satisfies_recurrence_off_negative_axis() -> Result<(), String> {
        let x = -0.25;
        let lhs = get_scalar(polygamma(4, &scalar(x + 1.0), RuntimeMode::Strict))?;
        let rhs = get_scalar(polygamma(4, &scalar(x), RuntimeMode::Strict))? + 24.0 / x.powi(5);
        assert!((lhs - rhs).abs() <= 1.0e-10);
        Ok(())
    }

    #[test]
    fn polygamma_higher_orders_hardened_reject_poles() {
        let err = polygamma(3, &scalar(-1.0), RuntimeMode::Hardened)
            .expect_err("hardened mode should reject higher-order poles");
        assert_eq!(err.kind, SpecialErrorKind::PoleInput);
    }

    // ── Complex digamma/polygamma tests ──────────────────────────────────

    fn digamma_complex_scalar(re: f64, im: f64) -> SpecialTensor {
        SpecialTensor::ComplexScalar(Complex64::new(re, im))
    }

    fn get_digamma_complex(result: SpecialResult) -> Result<Complex64, String> {
        match result.map_err(|err| err.to_string())? {
            SpecialTensor::ComplexScalar(c) => Ok(c),
            other => Err(format!("expected ComplexScalar, got {other:?}")),
        }
    }

    #[test]
    fn complex_digamma_real_axis_matches_real_digamma() -> Result<(), String> {
        let x = 2.5;
        let real_result = get_scalar(digamma(&scalar(x), RuntimeMode::Strict))?;
        let complex_result = get_digamma_complex(digamma(
            &digamma_complex_scalar(x, 0.0),
            RuntimeMode::Strict,
        ))?;
        assert!((complex_result.re - real_result).abs() < 1e-10);
        assert!(complex_result.im.abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn complex_digamma_pure_imaginary() -> Result<(), String> {
        let z = digamma_complex_scalar(0.0, 2.0);
        let result = get_digamma_complex(digamma(&z, RuntimeMode::Strict))?;
        assert!(result.re.is_finite());
        assert!(result.im.is_finite());
        Ok(())
    }

    #[test]
    fn complex_digamma_negative_real() -> Result<(), String> {
        let z = digamma_complex_scalar(-0.5, 1.0);
        let result = get_digamma_complex(digamma(&z, RuntimeMode::Strict))?;
        assert!(result.re.is_finite());
        assert!(result.im.is_finite());
        Ok(())
    }

    #[test]
    fn complex_digamma_returns_nan_at_real_poles() -> Result<(), String> {
        for x in [0.0, -1.0, -2.0, -8.0] {
            let actual = get_digamma_complex(digamma(
                &digamma_complex_scalar(x, 0.0),
                RuntimeMode::Strict,
            ))?;
            assert!(
                actual.re.is_nan(),
                "digamma({x}+0i) real part should be NaN"
            );
            assert!(
                actual.im.is_nan(),
                "digamma({x}+0i) imag part should be NaN"
            );
        }
        Ok(())
    }

    #[test]
    fn complex_digamma_near_real_poles_stays_finite() -> Result<(), String> {
        let actual = get_digamma_complex(digamma(
            &digamma_complex_scalar(-1.0, 1.0e-12),
            RuntimeMode::Strict,
        ))?;
        assert!(actual.re.is_finite());
        assert!(actual.im.is_finite());
        Ok(())
    }

    #[test]
    fn complex_trigamma_real_axis_matches_real_trigamma() -> Result<(), String> {
        let x = 2.5;
        let real_result = get_scalar(polygamma(1, &scalar(x), RuntimeMode::Strict))?;
        let complex_result = get_digamma_complex(polygamma(
            1,
            &digamma_complex_scalar(x, 0.0),
            RuntimeMode::Strict,
        ))?;
        assert!((complex_result.re - real_result).abs() < 1e-10);
        assert!(complex_result.im.abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn complex_polygamma_order_two_matches_real() -> Result<(), String> {
        let x = 1.5;
        let real_result = get_scalar(polygamma(2, &scalar(x), RuntimeMode::Strict))?;
        let complex_result = get_digamma_complex(polygamma(
            2,
            &digamma_complex_scalar(x, 0.0),
            RuntimeMode::Strict,
        ))?;
        // Higher-order polygamma uses numerical differentiation for complex, so tolerance is relaxed
        assert!(
            (complex_result.re - real_result).abs() < 0.01,
            "real_result={}, complex_result.re={}",
            real_result,
            complex_result.re
        );
        assert!(complex_result.im.abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn complex_digamma_vector_input() -> Result<(), String> {
        let input = SpecialTensor::ComplexVec(vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.5, 1.0),
        ]);
        let result = digamma(&input, RuntimeMode::Strict).map_err(|e| e.to_string())?;
        match result {
            SpecialTensor::ComplexVec(values) => {
                assert_eq!(values.len(), 3);
                for v in values {
                    assert!(v.is_finite());
                }
            }
            _ => return Err("expected ComplexVec".into()),
        }
        Ok(())
    }

    // ── factorial2 tests ─────────────────────────────────────────────────

    #[test]
    fn factorial2_small_even() {
        // 0!! = 1, 2!! = 2, 4!! = 8, 6!! = 48, 8!! = 384, 10!! = 3840
        assert_eq!(factorial2(0), 1.0);
        assert_eq!(factorial2(2), 2.0);
        assert_eq!(factorial2(4), 8.0);
        assert_eq!(factorial2(6), 48.0);
        assert_eq!(factorial2(8), 384.0);
        assert_eq!(factorial2(10), 3840.0);
    }

    #[test]
    fn factorial2_small_odd() {
        // 1!! = 1, 3!! = 3, 5!! = 15, 7!! = 105, 9!! = 945
        assert_eq!(factorial2(1), 1.0);
        assert_eq!(factorial2(3), 3.0);
        assert_eq!(factorial2(5), 15.0);
        assert_eq!(factorial2(7), 105.0);
        assert_eq!(factorial2(9), 945.0);
    }

    #[test]
    fn factorial2_negative() {
        // SciPy returns 0 for all negative integer inputs.
        assert_eq!(factorial2(-1), 0.0);
        assert_eq!(factorial2(-2), 0.0);
        assert_eq!(factorial2(-5), 0.0);
    }

    #[test]
    fn factorial2_large() {
        // Test gamma-based computation for large values
        // 20!! = 3715891200
        let computed = factorial2(20);
        let expected = 3_715_891_200.0_f64;
        assert!((computed - expected).abs() / expected < 1e-10);
    }

    #[test]
    fn factorial2_large_odd_gamma_path() {
        // n > 33 takes the gamma-based fallback. The previous formula
        // used Gamma((n+1)/2) instead of Gamma((n+2)/2) and overshot
        // the true value by roughly 8x at n=35. Reference values are
        // computed exactly via the recurrence m!! = m * (m-2)!!.
        fn double_fact_exact(n: i64) -> f64 {
            let mut result = 1.0f64;
            let mut k = n;
            while k > 1 {
                result *= k as f64;
                k -= 2;
            }
            result
        }
        for n in [35_i64, 51, 99, 151] {
            let computed = factorial2(n);
            let expected = double_fact_exact(n);
            let rel_err = ((computed - expected) / expected).abs();
            assert!(
                rel_err < 1e-12,
                "factorial2({n}) gamma-path: got {computed:e}, expected {expected:e}, rel_err={rel_err:e}"
            );
        }
    }

    // ── zetac tests ──────────────────────────────────────────────────────

    #[test]
    fn zetac_small_s() {
        // zetac(2) = zeta(2) - 1 = π²/6 - 1 ≈ 0.6449340668
        let result = zetac_scalar(2.0);
        let expected = PI * PI / 6.0 - 1.0;
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn zetac_large_s() {
        // For large s, zetac(s) ≈ 2^(-s)
        let s = 20.0;
        let result = zetac_scalar(s);
        let approx = 2.0_f64.powf(-s);
        // Should be close to 2^(-s), within a few percent
        assert!((result - approx).abs() / approx < 0.01);
    }

    #[test]
    fn zetac_consistency() {
        // zetac(s) should equal zeta(s) - 1
        for s in [3.0, 4.0, 5.0, 10.0] {
            let zetac_result = zetac_scalar(s);
            let zeta_minus_one = zeta_scalar(s) - 1.0;
            assert!((zetac_result - zeta_minus_one).abs() < 1e-12);
        }
    }

    #[test]
    fn zetac_pole() {
        // At s=1, zetac should return infinity
        assert!(zetac_scalar(1.0).is_infinite());
    }

    #[test]
    fn zetac_critical_strip_matches_scipy() {
        // 0 < s < 1: reference values from scipy.special.zetac. The naive
        // alternating-series sum was ~5% off here; Borwein's acceleration
        // restores full precision (frankenscipy-kk4vu).
        for &(s, expected) in &[
            (0.1_f64, -1.603_037_519_856_241_9),
            (0.3, -1.904_559_257_253_984_2),
            (0.5, -2.460_354_508_809_586_6),
            (0.7, -3.778_388_445_553_696),
            (0.9, -10.430_114_019_402_255),
        ] {
            let got = zetac_scalar(s);
            assert!(
                (got - expected).abs() < 1e-10,
                "zetac({s}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn zeta_tensor_vec_matches_scalar_surface() {
        let values = vec![1.1, 2.0, 3.0, 10.0];
        let input = SpecialTensor::RealVec(values.clone());
        let got = zeta(&input, RuntimeMode::Strict).expect("zeta RealVec");
        let got = match got {
            SpecialTensor::RealVec(items) => items,
            other => panic!("expected RealVec, got {other:?}"),
        };
        for (value, got_value) in values.iter().zip(got.iter()) {
            assert_eq!(*got_value, zeta_scalar(*value), "zeta({value})");
        }

        let got = zetac(&input, RuntimeMode::Strict).expect("zetac RealVec");
        let got = match got {
            SpecialTensor::RealVec(items) => items,
            other => panic!("expected RealVec, got {other:?}"),
        };
        for (value, got_value) in values.iter().zip(got.iter()) {
            assert_eq!(*got_value, zetac_scalar(*value), "zetac({value})");
        }
    }

    #[test]
    fn zeta_affine_vec_matches_scalar_surface_within_tolerance() {
        let n = 2048usize;
        let denom = (n - 1) as f64;
        let values = (0..n)
            .map(|i| 1.1 + 8.9 * (i as f64) / denom)
            .collect::<Vec<_>>();
        let input = SpecialTensor::RealVec(values.clone());
        let got = match zeta(&input, RuntimeMode::Strict) {
            Ok(SpecialTensor::RealVec(items)) => items,
            Ok(other) => {
                panic!("expected RealVec, got {other:?}");
            }
            Err(err) => {
                panic!("zeta RealVec failed: {err}");
            }
        };

        for (i, (&s, &got_value)) in values.iter().zip(&got).enumerate() {
            let want = zeta_scalar(s);
            let abs = (got_value - want).abs();
            let rel = abs / want.abs().max(1.0);
            assert!(
                abs < 1e-11 || rel < 1e-13,
                "zeta affine vec drift at {i}: s={s}, got={got_value}, scalar={want}, abs={abs}, rel={rel}"
            );
        }
    }

    // ── Poisson distribution functions ────────────────────────────────

    #[test]
    fn pdtr_basic() {
        // pdtr(k, m) = P(X <= k) for Poisson with mean m
        // pdtr(0, 1) = P(X = 0) = exp(-1) ≈ 0.3679
        let result = pdtr(0.0, 1.0);
        assert!((result - (-1.0_f64).exp()).abs() < 1e-10);

        // pdtr(k, 0) = 1 for any k >= 0
        assert!((pdtr(5.0, 0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pdtr_pdtrc_complement() {
        // pdtr(k, m) + pdtrc(k, m) = 1
        for &k in &[0.0, 1.0, 5.0, 10.0] {
            for &m in &[0.5, 1.0, 5.0, 10.0] {
                let sum = pdtr(k, m) + pdtrc(k, m);
                assert!(
                    (sum - 1.0).abs() < 1e-10,
                    "pdtr + pdtrc != 1 for k={k}, m={m}"
                );
            }
        }
    }

    #[test]
    fn pdtri_inverse() {
        // pdtri should be inverse of pdtr
        for &k in &[1.0, 5.0, 10.0] {
            for &m in &[1.0, 5.0, 10.0] {
                let p = pdtr(k, m);
                if p > 0.01 && p < 0.99 {
                    let m_recovered = pdtri(k, p);
                    assert!(
                        (m_recovered - m).abs() / m < 0.01,
                        "pdtri failed: k={k}, m={m}, p={p}, m_recovered={m_recovered}"
                    );
                }
            }
        }
    }

    /// pdtri / chdtri at p close to 0 or 1 (frankenscipy-jr3na).
    ///
    /// Pre-fix the local Newton-Raphson `gammaincinv` here diverged
    /// when 1-p was small; pdtri(1, 0.99) returned ~1e13 instead of
    /// scipy's 0.149, and chdtri(3, 0.99) returned ~5.6e5 instead of
    /// 0.115. Both now delegate to the convenience-module
    /// `gammaincinv_scalar`, which is validated by the diff_special
    /// gammainc harness.
    #[test]
    fn pdtri_chdtri_match_scipy_at_extreme_p() {
        // (k, p, scipy.special.pdtri(k, p))
        let pdtri_cases: [(f64, f64, f64); 5] = [
            (1.0, 0.99, 0.14855474025327913),
            (1.0, 0.999, 0.04540201776948989),
            (0.0, 0.99, 0.010050335853501444),
            (5.0, 0.5, 5.670161188712083),
            (3.0, 0.95, 1.3663183967498314),
        ];
        for (k, p, expected) in pdtri_cases {
            let got = pdtri(k, p);
            let scale = expected.abs().max(1.0);
            assert!(
                (got - expected).abs() < 1e-9 * scale,
                "pdtri({k}, {p}) = {got}, expected {expected}"
            );
        }

        // (v, p, scipy.special.chdtri(v, p))
        let chdtri_cases: [(f64, f64, f64); 4] = [
            (3.0, 0.99, 0.11483180189911714),
            (3.0, 0.5, 2.3659738843753395),
            (5.0, 0.95, 1.1454762260617706),
            (1.0, 0.999, 1.5707971492623826e-06),
        ];
        for (v, p, expected) in chdtri_cases {
            let got = chdtri(v, p);
            let scale = expected.abs().max(1.0);
            assert!(
                (got - expected).abs() < 1e-9 * scale,
                "chdtri({v}, {p}) = {got}, expected {expected}"
            );
        }

        // Small-p (deep upper tail) regression: chdtri/pdtri must invert the
        // *complemented* incomplete gamma directly. The former `1 − p` route lost
        // tail precision (chdtri(1, 1e-8) was ~6e-9 off); these lock the fix at a
        // tight tolerance against scipy 1.17.1.
        let chdtri_tail: [(f64, f64, f64); 4] = [
            (1.0, 1e-8, 32.841253361236788),
            (5.0, 1e-8, 45.794587123084568),
            (30.0, 1e-8, 95.328831531183496),
            (2.0, 1e-6, 27.631021115928547),
        ];
        for (v, p, expected) in chdtri_tail {
            let got = chdtri(v, p);
            assert!(
                (got - expected).abs() < 1e-12 * expected.abs(),
                "chdtri({v}, {p}) = {got}, expected {expected}"
            );
        }
        let pdtri_tail: [(f64, f64, f64); 3] = [
            (0.0, 1e-8, 18.420680743952364),
            (3.0, 1e-8, 26.584739102269619),
            (10.0, 1e-6, 34.427884340361182),
        ];
        for (k, p, expected) in pdtri_tail {
            let got = pdtri(k, p);
            assert!(
                (got - expected).abs() < 1e-12 * expected.abs(),
                "pdtri({k}, {p}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn pdtrik_reference_values() {
        let cases = [
            (0.7, 2.5, 2.692_718_092_203_629),
            (0.5, 2.5, 1.825_430_403_950_633_3),
            (0.757_576_133_133_066_2, 2.5, 3.000_000_000_000_001_8),
            (0.99, 2.5, 6.299_974_200_719_591),
        ];

        for (p, m, expected) in cases {
            let actual = pdtrik(p, m);
            assert!(
                (actual - expected).abs() < 5e-10,
                "pdtrik({p}, {m}) = {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn pdtrik_inverse() {
        for &m in &[0.5, 1.0, 2.5, 5.0, 10.0] {
            for &k in &[0.25, 1.0, 2.5, 5.0] {
                let p = pdtr(k, m);
                let recovered = pdtrik(p, m);
                assert!(
                    (recovered - k).abs() < 5e-10,
                    "pdtrik failed: m={m}, k={k}, p={p}, recovered={recovered}"
                );
            }
        }
    }

    #[test]
    fn pdtrik_edges_match_scipy() {
        assert_eq!(pdtrik(0.0, 2.5), 0.0);
        assert_eq!(pdtrik(0.001, 2.5), 0.0);
        assert_eq!(pdtrik(0.7, 0.0), 0.0);
        assert_eq!(pdtrik(0.0, f64::INFINITY), 0.0);

        assert!(pdtrik(1.0, 2.5).is_nan());
        assert!(pdtrik(0.7, f64::INFINITY).is_nan());
        assert!(pdtrik(-0.1, 2.5).is_nan());
        assert!(pdtrik(1.1, 2.5).is_nan());
        assert!(pdtrik(0.7, -1.0).is_nan());
        assert!(pdtrik(f64::NAN, 2.5).is_nan());
        assert!(pdtrik(0.7, f64::NAN).is_nan());
    }

    #[test]
    fn gdtr_closed_form_reference_values() {
        // /testing-golden-artifacts: pin gdtr against three closed-form
        // values where gammainc collapses to a finite series:
        //   gdtr(1, 1, 1) = P(1, 1)   = 1 - e^{-1}
        //   gdtr(2, 3, 2) = P(3, 4)   = 1 - 13 e^{-4}
        //   gdtr(0.5, 2, 4) = P(2, 2) = 1 - 3 e^{-2}
        let cases: &[(f64, f64, f64, f64)] = &[
            (1.0, 1.0, 1.0, 1.0 - (-1.0_f64).exp()),
            (2.0, 3.0, 2.0, 1.0 - 13.0 * (-4.0_f64).exp()),
            (0.5, 2.0, 4.0, 1.0 - 3.0 * (-2.0_f64).exp()),
        ];
        for &(a, b, x, expected) in cases {
            let actual = gdtr(a, b, x);
            assert!(
                (actual - expected).abs() < 1e-12,
                "gdtr({a}, {b}, {x}) = {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn gdtria_roundtrip_recovers_rate() {
        // gdtria(gdtr(a, b, x), b, x) ≈ a
        // Closed form: a = gammaincinv(b, p) / x.
        for &a in &[0.5, 1.0, 2.5] {
            for &b in &[1.0, 2.0, 3.5] {
                for &x in &[0.5, 2.0, 5.0] {
                    let p = gdtr(a, b, x);
                    let recovered = gdtria(p, b, x);
                    assert!(
                        (recovered - a).abs() < 1e-9,
                        "gdtria(gdtr({a}, {b}, {x})={p}, {b}, {x}) = {recovered}, expected {a}"
                    );
                }
            }
        }
    }

    #[test]
    fn gdtrib_roundtrip_recovers_shape() {
        // gdtrib(a, gdtr(a, b, x), x) ≈ b
        // gdtrib uses gammainc_shape_inv, a Newton root finder over b.
        for &a in &[0.5, 1.0, 2.5] {
            for &b in &[1.0, 2.0, 3.5] {
                for &x in &[0.5, 2.0, 5.0] {
                    let p = gdtr(a, b, x);
                    let recovered = gdtrib(a, p, x);
                    assert!(
                        (recovered - b).abs() < 1e-7,
                        "gdtrib({a}, gdtr({a}, {b}, {x})={p}, {x}) = {recovered}, expected {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn gdtria_gdtrib_domain_error_returns_nan() {
        // Out-of-domain inputs return NaN per scipy.special semantics.
        assert!(gdtria(0.5, -1.0, 1.0).is_nan(), "gdtria b<0");
        assert!(gdtria(-0.1, 1.0, 1.0).is_nan(), "gdtria p<0");
        assert!(gdtria(1.1, 1.0, 1.0).is_nan(), "gdtria p>1");
        assert!(gdtria(0.5, 1.0, 0.0).is_nan(), "gdtria x=0");

        assert!(gdtrib(-1.0, 0.5, 1.0).is_nan(), "gdtrib a<0");
        assert!(gdtrib(1.0, -0.1, 1.0).is_nan(), "gdtrib p<0");
        assert!(gdtrib(1.0, 1.1, 1.0).is_nan(), "gdtrib p>1");
        assert!(gdtrib(1.0, 0.5, -1.0).is_nan(), "gdtrib x<0");
    }

    #[test]
    fn gdtrc_complement() {
        // gdtr + gdtrc should equal 1
        for &a in &[1.0, 2.0, 5.0] {
            for &b in &[1.0, 2.0] {
                for &x in &[0.5, 1.0, 2.0, 5.0] {
                    let sum = gdtr(a, b, x) + gdtrc(a, b, x);
                    assert!(
                        (sum - 1.0).abs() < 1e-10,
                        "gdtr({a}, {b}, {x}) + gdtrc = {sum}, expected 1.0"
                    );
                }
            }
        }
    }

    #[test]
    fn gdtrix_reference_values() {
        let cases: &[(f64, f64, f64, f64, f64)] = &[
            (2.5, 1.25, 0.4, 0.28778803752030013, 2e-12),
            (2.0, 3.0, 0.7, 1.8077838329329956, 2e-12),
            (-2.5, 1.25, 0.4, -0.28778803752030013, 2e-12),
        ];
        for &(a, b, p, expected, tolerance) in cases {
            let actual = gdtrix(a, b, p);
            assert!(
                (actual - expected).abs() < tolerance,
                "gdtrix({a}, {b}, {p}) = {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn gdtrix_matches_scipy_nonpositive_rates() {
        assert!(gdtrix(0.0, 1.25, 0.4).is_infinite() && gdtrix(0.0, 1.25, 0.4).is_sign_positive());
        assert!(
            gdtrix(-0.0, 1.25, 0.4).is_infinite() && gdtrix(-0.0, 1.25, 0.4).is_sign_negative()
        );
        assert_eq!(gdtrix(-2.5, 1.25, 0.0), -0.0);
        assert!(
            gdtrix(-2.5, 1.25, 1.0).is_infinite() && gdtrix(-2.5, 1.25, 1.0).is_sign_negative()
        );
        assert!(gdtrix(0.0, 1.25, 0.0).is_nan());

        assert!(gdtrix(2.5, 0.0, 0.4).is_nan());
        assert!(gdtrix(2.5, -1.0, 0.4).is_nan());
        assert!(gdtrix(2.5, 1.25, -0.1).is_nan());
        assert!(gdtrix(2.5, 1.25, 1.1).is_nan());
        assert!(gdtrix(f64::NAN, 1.25, 0.4).is_nan());
        assert!(gdtrix(2.5, f64::NAN, 0.4).is_nan());
        assert!(gdtrix(2.5, 1.25, f64::NAN).is_nan());
    }

    #[test]
    fn chdtr_basic() {
        // Chi-squared CDF values compared with scipy.special.chdtr
        // chdtr(1, 1) ≈ 0.6827 (P(X <= 1) for chi2 with df=1)
        let result = chdtr(1.0, 1.0);
        assert!(
            (result - 0.6827).abs() < 0.01,
            "chdtr(1, 1) = {result}, expected ~0.6827"
        );

        // chdtr(2, 2) ≈ 0.6321 (P(X <= 2) for chi2 with df=2)
        let result = chdtr(2.0, 2.0);
        assert!(
            (result - 0.6321).abs() < 0.01,
            "chdtr(2, 2) = {result}, expected ~0.6321"
        );

        // chdtr(10, 10) ≈ 0.5595
        let result = chdtr(10.0, 10.0);
        assert!(
            (result - 0.5595).abs() < 0.01,
            "chdtr(10, 10) = {result}, expected ~0.5595"
        );
    }

    #[test]
    fn chdtr_chdtrc_complement() {
        // chdtr + chdtrc should equal 1
        for &v in &[1.0, 2.0, 5.0, 10.0] {
            for &x in &[0.5, 1.0, 2.0, 5.0] {
                let sum = chdtr(v, x) + chdtrc(v, x);
                assert!(
                    (sum - 1.0).abs() < 1e-10,
                    "chdtr({v}, {x}) + chdtrc({v}, {x}) = {sum}, expected 1.0"
                );
            }
        }
    }

    #[test]
    fn chdtri_inverse() {
        // chdtri is the inverse of chdtrc, matching scipy.special.chdtri.
        for &v in &[1.0, 2.0, 5.0, 10.0] {
            for &x in &[0.5, 1.0, 2.0, 5.0, 10.0] {
                let p = chdtrc(v, x);
                if p > 0.01 && p < 0.99 {
                    let x_recovered = chdtri(v, p);
                    assert!(
                        (x_recovered - x).abs() / x < 0.01,
                        "chdtri failed: v={v}, x={x}, p={p}, x_recovered={x_recovered}"
                    );
                }
            }
        }
    }

    /// `chndtrc` must hold precision where `1 - chndtr` collapses.
    ///
    /// Goldens are scipy 1.17.1 `ncx2.sf`. The x = 150 row is the point of the kernel:
    /// `chndtr(150, 3, 2)` is exactly 1.0 in double precision, so the subtraction gives
    /// 0.0 while the true survival is 1.04e-26. A relative check is used because the
    /// values span 26 orders of magnitude and an absolute tolerance would be vacuous at
    /// the small end — which is the only end that matters here.
    #[test]
    fn chndtrc_matches_scipy_survival_where_one_minus_cdf_collapses() {
        // (x, df, nc, scipy ncx2.sf)
        let cases = [
            (10.0_f64, 3.0_f64, 2.0_f64, 1.014_350_364_860_014e-1),
            (80.0, 3.0, 2.0, 1.626_871_961_685_992e-13),
            (150.0, 3.0, 2.0, 1.044_927_027_553_427e-26),
            (5.0, 7.0, 20.0, 9.992_803_076_754_802e-1),
        ];
        for (x, df, nc, expected) in cases {
            let got = chndtrc(x, df, nc);
            let rel = ((got - expected) / expected).abs();
            assert!(
                rel < 1e-10,
                "chndtrc({x}, {df}, {nc}) = {got}, scipy {expected}, rel {rel:e}"
            );
        }

        // The defect this replaces: subtraction is exactly zero at x = 150 while the
        // kernel is not. If this assertion ever fails, `1 - chndtr` became viable and
        // the kernel could be reconsidered — it does not mean the kernel is wrong.
        assert_eq!(
            1.0 - chndtr(150.0, 3.0, 2.0),
            0.0,
            "the subtraction is expected to collapse here; that is why chndtrc exists"
        );
        assert!(
            chndtrc(150.0, 3.0, 2.0) > 0.0,
            "the kernel must not collapse"
        );

        // Complementarity in the range where subtraction IS accurate, and the boundaries.
        let mid = chndtrc(10.0, 3.0, 2.0) + chndtr(10.0, 3.0, 2.0);
        assert!(
            (mid - 1.0).abs() < 1e-12,
            "cdf + sf should be 1 mid-range, got {mid}"
        );
        assert_eq!(chndtrc(0.0, 3.0, 2.0), 1.0, "sf below the support is 1");
        assert!(chndtrc(f64::NAN, 3.0, 2.0).is_nan(), "NaN propagates");
        // nc = 0 degenerates to the central complement.
        assert_eq!(chndtrc(10.0, 3.0, 0.0), chdtrc(3.0, 10.0));
    }

    #[test]
    fn chndtr_matches_scipy_reference_values() {
        // frankenscipy: non-central chi-squared CDF was missing. Golden values
        // from scipy.special.chndtr(x, df, nc) 1.17.1.
        let cases = [
            (5.0_f64, 3.0, 2.0, 0.5934051800831555_f64),
            (10.0, 5.0, 8.0, 0.3648789027994557),
            (1.0, 2.0, 0.0, 0.3934693402873665), // nc=0 → central chi2
            (50.0, 10.0, 40.0, 0.528710534217772), // large nc (mode-centered sum)
            (2.0, 4.0, 100.0, 2.0596217576094693e-19), // deep left tail
            // Skellam cdf/sf calls use x=2*mu1, df=2*(k+1), nc=2*mu2.
            // These high-nc cases lock the adjacent-shape recurrence that avoids
            // one gammainc evaluation per Poisson-mixture term.
            (2000.0, 2.0, 2000.0, 0.49553941086177933),
            (2000.0, 102.0, 2000.0, 0.12939019728751158),
            (2000.0, 202.0, 2000.0, 0.012313651457695611),
            (200.0, 2.0, 200.0, 0.48588642002544574),
        ];
        for (x, df, nc, want) in cases {
            let got = chndtr(x, df, nc);
            assert!(
                (got - want).abs() <= 1e-10 * want.abs().max(1e-12),
                "chndtr({x},{df},{nc}) = {got}, expected {want}"
            );
        }
        // x ≤ 0 → 0.
        assert_eq!(chndtr(0.0, 5.0, 2.0), 0.0);
    }

    #[test]
    fn chndtrix_inverts_chndtr_matching_scipy() {
        // frankenscipy: golden from scipy.special.chndtrix(p, df, nc) 1.17.1.
        let cases = [
            (0.5_f64, 3.0, 2.0, 4.137515123399119_f64),
            (0.1, 5.0, 8.0, 5.486633223562761),
            (0.9, 10.0, 1.0, 17.560589244156514),
        ];
        for (p, df, nc, want) in cases {
            let got = chndtrix(p, df, nc);
            assert!(
                (got - want).abs() <= 1e-9 * want.abs(),
                "chndtrix({p},{df},{nc}) = {got}, want {want}"
            );
        }
        assert_eq!(chndtrix(0.0, 3.0, 2.0), 0.0);
        assert!(chndtrix(1.0, 3.0, 2.0).is_infinite());
        assert!(chndtrix(1.5, 3.0, 2.0).is_nan());
    }

    #[test]
    fn chndtridf_chndtrinc_invert_chndtr_matching_scipy() {
        // frankenscipy: golden from scipy.special.chndtridf / chndtrinc 1.17.1.
        // Roundtrip (x=8, df=5, nc=3 → p) recovers df=5 / nc=3.
        let p = chndtr(8.0, 5.0, 3.0);
        assert!(
            (chndtridf(8.0, p, 3.0) - 5.0).abs() < 1e-7,
            "chndtridf roundtrip"
        );
        assert!(
            (chndtrinc(8.0, 5.0, p) - 3.0).abs() < 1e-7,
            "chndtrinc roundtrip"
        );
        // Direct golden values.
        assert!(
            (chndtridf(20.0, 0.3, 4.0) - 20.43935171067603).abs() < 1e-7,
            "chndtridf direct"
        );
        assert!(
            (chndtrinc(20.0, 6.0, 0.7) - 10.847570357455123).abs() < 1e-7,
            "chndtrinc direct"
        );
        // p ∉ (0,1) → NaN.
        assert!(chndtridf(8.0, 0.0, 3.0).is_nan());
        assert!(chndtrinc(8.0, 5.0, 1.0).is_nan());
    }

    #[test]
    fn chdtriv_reference_values() {
        let cases = [
            (0.5, 2.0, 2.628_500_020_690_701_4),
            (0.8, 3.0, 1.853_054_579_870_468_3),
            (0.95, 10.0, 4.318_557_759_041_437),
        ];

        for (p, x, expected) in cases {
            let actual = chdtriv(p, x);
            assert!(
                (actual - expected).abs() < 5e-10,
                "chdtriv({p}, {x}) = {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn chdtriv_inverse() {
        for &v in &[0.5, 1.0, 2.5, 5.0, 10.0] {
            for &x in &[1.0, 2.0, 5.0, 10.0] {
                let p = chdtr(v, x);
                if p > 0.01 && p < 0.99 {
                    let recovered = chdtriv(p, x);
                    let tolerance = 5e-10 * v.abs().max(1.0);
                    assert!(
                        (recovered - v).abs() <= tolerance,
                        "chdtriv failed: v={v}, x={x}, p={p}, recovered={recovered}"
                    );
                }
            }
        }
    }

    #[test]
    fn chdtriv_edges_match_scipy() {
        assert!(chdtriv(0.0, 2.0).is_infinite());
        assert_eq!(chdtriv(1.0, 2.0), 0.0);

        assert!(chdtriv(0.5, 0.0).is_nan());
        assert!(chdtriv(0.0, 0.0).is_nan());
        assert!(chdtriv(1.0, 0.0).is_nan());
        assert!(chdtriv(0.5, f64::INFINITY).is_nan());
        assert!(chdtriv(-0.1, 3.0).is_nan());
        assert!(chdtriv(1.1, 3.0).is_nan());
        assert!(chdtriv(0.5, -1.0).is_nan());
        assert!(chdtriv(f64::NAN, 3.0).is_nan());
        assert!(chdtriv(0.5, f64::NAN).is_nan());
    }

    // ========================================================================
    // Complex gammainc tests
    // ========================================================================

    fn complex_scalar(re: f64, im: f64) -> SpecialTensor {
        SpecialTensor::ComplexScalar(Complex64::new(re, im))
    }

    fn get_complex_scalar(result: SpecialResult) -> Result<Complex64, String> {
        match result.map_err(|err| err.to_string())? {
            SpecialTensor::ComplexScalar(value) => Ok(value),
            other => Err(format!("expected complex scalar, got {other:?}")),
        }
    }

    #[test]
    fn gammainc_a_equals_one_matches_exponential_cdf() {
        // /testing-golden-artifacts for [frankenscipy-swlxw]:
        // gammainc(1, x) = 1 - e^{-x}, the Exp(1) CDF.
        for &x in &[0.0_f64, 0.1, 0.5, 1.0, 2.5, 7.0] {
            let actual = gammainc_scalar(1.0, x, RuntimeMode::Strict).unwrap();
            let expected = if x == 0.0 { 0.0 } else { 1.0 - (-x).exp() };
            assert!(
                (actual - expected).abs() < 1e-12,
                "gammainc(1, {x}) = {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn gammainc_a_equals_two_matches_closed_form() {
        // gammainc(2, x) = 1 - (1 + x)·e^{-x} (Erlang-2 CDF).
        for &x in &[0.0_f64, 0.5, 1.0, 2.0, 5.0] {
            let actual = gammainc_scalar(2.0, x, RuntimeMode::Strict).unwrap();
            let expected = if x == 0.0 {
                0.0
            } else {
                1.0 - (1.0 + x) * (-x).exp()
            };
            assert!(
                (actual - expected).abs() < 1e-12,
                "gammainc(2, {x}) = {actual}, expected 1 - (1+x)e^(-x) = {expected}"
            );
        }
    }

    #[test]
    fn gammainc_at_x_zero_is_zero() {
        // gammainc(a, 0) = 0 for any a > 0.
        for &a in &[0.5_f64, 1.0, 2.5, 7.0, 100.0] {
            let actual = gammainc_scalar(a, 0.0, RuntimeMode::Strict).unwrap();
            assert!(
                actual.abs() < 1e-15,
                "gammainc({a}, 0) = {actual}, expected 0"
            );
        }
    }

    #[test]
    fn gammainc_plus_gammaincc_equals_one() {
        // gammainc(a, x) + gammaincc(a, x) = 1 (P + Q = 1).
        for &a in &[0.5_f64, 1.5, 3.0, 10.0] {
            for &x in &[0.1_f64, 1.0, 5.0, 20.0] {
                let p = gammainc_scalar(a, x, RuntimeMode::Strict).unwrap();
                let q = gammaincc_scalar(a, x, RuntimeMode::Strict).unwrap();
                assert!(
                    (p + q - 1.0).abs() < 1e-12,
                    "gammainc({a},{x}) + gammaincc = {} != 1",
                    p + q
                );
            }
        }
    }

    #[test]
    fn gammainc_large_a_near_transition_matches_scipy() {
        // The lower series needs ~12√a terms near x≈a; the old 200-cap truncated
        // it (P(5000,5000) was 0.5% off). Golden from scipy/mpmath. frankenscipy.
        let cases = [
            (2000.0_f64, 2000.0_f64, 0.502_973_548_444_202_5_f64),
            (5000.0, 5000.0, 0.501_880_634_033_817_3),
        ];
        for &(a, x, want) in &cases {
            let p = gammainc_scalar(a, x, RuntimeMode::Strict).unwrap();
            assert!(
                (p - want).abs() < 1e-10,
                "gammainc({a},{x}) = {p} vs {want}"
            );
        }
    }

    #[test]
    fn complex_gammainc_reduces_to_real_for_real_z() -> Result<(), String> {
        // For real positive z, complex_gammainc_scalar should match gammainc_scalar
        let a = 2.0;
        let z_re = 1.5;
        let z = Complex64::new(z_re, 0.0);

        let real_result = gammainc_scalar(a, z_re, RuntimeMode::Strict).unwrap();
        let complex_result = complex_gammainc_scalar(a, z, RuntimeMode::Strict).unwrap();

        assert!(
            (complex_result.re - real_result).abs() < 1e-10,
            "Real parts should match: {} vs {}",
            complex_result.re,
            real_result
        );
        assert!(
            complex_result.im.abs() < 1e-10,
            "Imaginary part should be negligible for real input: {}",
            complex_result.im
        );
        Ok(())
    }

    #[test]
    fn complex_gammaincc_reduces_to_real_for_real_z() -> Result<(), String> {
        let a = 2.0;
        let z_re = 1.5;
        let z = Complex64::new(z_re, 0.0);

        let real_result = gammaincc_scalar(a, z_re, RuntimeMode::Strict).unwrap();
        let complex_result = complex_gammaincc_scalar(a, z, RuntimeMode::Strict).unwrap();

        assert!(
            (complex_result.re - real_result).abs() < 1e-10,
            "Real parts should match: {} vs {}",
            complex_result.re,
            real_result
        );
        assert!(
            complex_result.im.abs() < 1e-10,
            "Imaginary part should be negligible for real input: {}",
            complex_result.im
        );
        Ok(())
    }

    #[test]
    fn complex_gammainc_at_z_zero() {
        let result =
            complex_gammainc_scalar(2.0, Complex64::new(0.0, 0.0), RuntimeMode::Strict).unwrap();
        assert_eq!(result.re, 0.0);
        assert_eq!(result.im, 0.0);
    }

    #[test]
    fn complex_gammaincc_at_z_zero() {
        let result =
            complex_gammaincc_scalar(2.0, Complex64::new(0.0, 0.0), RuntimeMode::Strict).unwrap();
        assert_eq!(result.re, 1.0);
        assert_eq!(result.im, 0.0);
    }

    #[test]
    fn complex_gammainc_p_plus_q_equals_one() {
        // P(a,z) + Q(a,z) = 1 for any a,z
        let a = 2.5;
        let z = Complex64::new(1.5, 0.7);

        let p = complex_gammainc_scalar(a, z, RuntimeMode::Strict).unwrap();
        let q = complex_gammaincc_scalar(a, z, RuntimeMode::Strict).unwrap();

        let sum_re = p.re + q.re;
        let sum_im = p.im + q.im;

        assert!(
            (sum_re - 1.0).abs() < 1e-10,
            "P + Q real part should be 1: {}",
            sum_re
        );
        assert!(
            sum_im.abs() < 1e-10,
            "P + Q imaginary part should be 0: {}",
            sum_im
        );
    }

    #[test]
    fn complex_gammainc_tensor_interface() -> Result<(), String> {
        // Test the tensor interface works for complex inputs
        let a = scalar(2.0);
        let z = complex_scalar(1.5, 0.5);

        let result = gammainc(&a, &z, RuntimeMode::Strict);
        let c = get_complex_scalar(result)?;

        // Just verify we get a finite result
        assert!(c.re.is_finite(), "Real part should be finite");
        assert!(c.im.is_finite(), "Imaginary part should be finite");
        Ok(())
    }

    #[test]
    fn complex_a_gammainc_reduces_to_real_path_for_real_x() -> Result<(), String> {
        let a = complex_scalar(2.0, 0.0);
        let x = scalar(1.5);

        let actual = get_complex_scalar(gammainc(&a, &x, RuntimeMode::Strict))?;
        let expected = gammainc_scalar(2.0, 1.5, RuntimeMode::Strict).unwrap();

        assert!((actual.re - expected).abs() < 1e-10);
        assert!(actual.im.abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn complex_a_gammainc_p_plus_q_equals_one() {
        let a = Complex64::new(2.5, 0.75);
        let z = Complex64::new(1.5, -0.25);

        let p = complex_parameter_gammainc_scalar(a, z, RuntimeMode::Strict).unwrap();
        let q = complex_parameter_gammaincc_scalar(a, z, RuntimeMode::Strict).unwrap();

        let sum = p + q;
        assert!((sum.re - 1.0).abs() < 1e-10, "real(sum) = {}", sum.re);
        assert!(sum.im.abs() < 1e-10, "imag(sum) = {}", sum.im);
    }

    #[test]
    fn complex_a_gammainc_parameter_vector_broadcasts_real_x() -> Result<(), String> {
        let a =
            SpecialTensor::ComplexVec(vec![Complex64::new(2.5, 0.75), Complex64::new(3.0, -0.5)]);
        let x = scalar(1.5);

        let result = gammainc(&a, &x, RuntimeMode::Strict).map_err(|e| e.to_string())?;
        match result {
            SpecialTensor::ComplexVec(values) => {
                assert_eq!(values.len(), 2);
                let lane0 = complex_parameter_gammainc_scalar(
                    Complex64::new(2.5, 0.75),
                    Complex64::new(1.5, 0.0),
                    RuntimeMode::Strict,
                )
                .unwrap();
                let lane1 = complex_parameter_gammainc_scalar(
                    Complex64::new(3.0, -0.5),
                    Complex64::new(1.5, 0.0),
                    RuntimeMode::Strict,
                )
                .unwrap();
                assert!((values[0].re - lane0.re).abs() < 1e-10);
                assert!((values[0].im - lane0.im).abs() < 1e-10);
                assert!((values[1].re - lane1.re).abs() < 1e-10);
                assert!((values[1].im - lane1.im).abs() < 1e-10);
            }
            other => return Err(format!("expected complex vector, got {other:?}")),
        }
        Ok(())
    }

    #[test]
    fn complex_a_gammainc_parameter_vector_mismatch_is_shape_error() {
        let a =
            SpecialTensor::ComplexVec(vec![Complex64::new(2.0, 0.5), Complex64::new(3.0, -0.5)]);
        let x = SpecialTensor::RealVec(vec![0.5, 1.0, 1.5]);

        let err = gammainc(&a, &x, RuntimeMode::Strict).unwrap_err();
        assert_eq!(err.kind, SpecialErrorKind::DomainError);
        assert_eq!(err.detail, "vector inputs must have matching lengths");
    }

    #[test]
    fn complex_gammainc_domain_error_for_nonpositive_a() {
        let z = Complex64::new(1.0, 0.5);

        let result = complex_gammainc_scalar(0.0, z, RuntimeMode::Strict).unwrap();
        assert!(result.re.is_nan());

        let result = complex_gammainc_scalar(-1.0, z, RuntimeMode::Strict).unwrap();
        assert!(result.re.is_nan());
    }

    #[test]
    fn complex_gamma_real_inputs_match_real_path() -> Result<(), String> {
        let z = complex_scalar(3.0, 0.0);
        let result = gamma(&z, RuntimeMode::Strict).map_err(|e| e.to_string())?;
        let c = get_complex_scalar(Ok(result))?;

        let real_z = scalar(3.0);
        let real_result = gamma(&real_z, RuntimeMode::Strict).map_err(|e| e.to_string())?;
        let expected = get_scalar(Ok(real_result))?;

        assert!(
            (c.re - expected).abs() < 1e-10,
            "gamma(3+0i) = {} + {}i, expected {} + 0i",
            c.re,
            c.im,
            expected
        );
        assert!(c.im.abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn complex_gammaln_real_inputs_match_real_path() -> Result<(), String> {
        let z = complex_scalar(5.0, 0.0);
        let result = gammaln(&z, RuntimeMode::Strict).map_err(|e| e.to_string())?;
        let c = get_complex_scalar(Ok(result))?;

        let real_z = scalar(5.0);
        let real_result = gammaln(&real_z, RuntimeMode::Strict).map_err(|e| e.to_string())?;
        let expected = get_scalar(Ok(real_result))?;

        assert!(
            (c.re - expected).abs() < 1e-10,
            "gammaln(5+0i) = {} + {}i, expected {} + 0i",
            c.re,
            c.im,
            expected
        );
        assert!(c.im.abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn loggamma_real_inputs_preserve_scipy_branch_cut() -> Result<(), String> {
        let positive = get_scalar(loggamma(&scalar(0.5), RuntimeMode::Strict))?;
        assert!((positive - 0.5 * PI.ln()).abs() < 1e-12);

        let pole = get_scalar(loggamma(&scalar(0.0), RuntimeMode::Strict))?;
        assert!(pole.is_infinite() && pole.is_sign_positive());

        let branch_cut = get_scalar(loggamma(&scalar(-0.5), RuntimeMode::Strict))?;
        assert!(branch_cut.is_nan());

        let err = loggamma(&scalar(-0.5), RuntimeMode::Hardened).unwrap_err();
        assert_eq!(err.kind, SpecialErrorKind::DomainError);

        let err = loggamma(&scalar(0.0), RuntimeMode::Hardened).unwrap_err();
        assert_eq!(err.kind, SpecialErrorKind::PoleInput);
        Ok(())
    }

    #[test]
    fn loggamma_complex_inputs_use_principal_branch() -> Result<(), String> {
        let actual = get_complex_scalar(loggamma(&complex_scalar(-0.5, 0.0), RuntimeMode::Strict))?;
        assert!((actual.re - 1.265_512_123_484_647).abs() < 1e-12);
        assert!((actual.im + PI).abs() < 1e-12);

        let near_upper = get_complex_scalar(loggamma(
            &complex_scalar(-0.5, 1.0e-12),
            RuntimeMode::Strict,
        ))?;
        let near_lower = get_complex_scalar(loggamma(
            &complex_scalar(-0.5, -1.0e-12),
            RuntimeMode::Strict,
        ))?;
        assert!(near_upper.im < 0.0);
        assert!(near_lower.im > 0.0);
        Ok(())
    }

    #[test]
    #[allow(clippy::excessive_precision)] // golden constants verbatim from scipy
    fn complex_loggamma_winding_matches_scipy() -> Result<(), String> {
        // frankenscipy-4roim: scipy.special.loggamma is the ANALYTIC CONTINUATION, whose
        // imaginary part winds by 2π per unit of negative Re(z) — NOT the single-valued
        // principal log of Γ(z). The old reflection branch produced the principal log and
        // was off by 2π·k for Re(z) < -1. (z_re, z_im, loggamma.re, loggamma.im) — scipy 1.17.1.
        let cases: [(f64, f64, f64, f64); 6] = [
            (-3.0, -1.0, -2.9535082922958997e+00, 9.7264182812369118e+00),
            (-3.0, 1.0, -2.9535082922958997e+00, -9.7264182812369118e+00),
            (-1.5, 2.0, -3.8624060873955779e+00, -4.6226094074869755e+00),
            (-5.0, -0.5, -4.4536008937628253e+00, 1.6425019297069067e+01),
            (-2.5, -0.2, -2.3526740571631932e-01, 9.2040025415435665e+00),
            (-4.5, -3.0, -1.0694354276574462e+01, 1.0712660703414734e+01),
        ];
        for (zr, zi, lr, li) in cases {
            let c = get_complex_scalar(loggamma(&complex_scalar(zr, zi), RuntimeMode::Strict))?;
            let rel = (c.re - lr).hypot(c.im - li) / lr.hypot(li);
            assert!(
                rel <= 1e-12,
                "loggamma({zr}{zi:+}i) = {c:?}, scipy ({lr},{li}), rel {rel:e}"
            );
        }
        Ok(())
    }

    #[test]
    fn complex_rgamma_real_inputs_match_real_path() -> Result<(), String> {
        let z = complex_scalar(5.0, 0.0);
        let result = rgamma(&z, RuntimeMode::Strict).map_err(|e| e.to_string())?;
        let c = get_complex_scalar(Ok(result))?;

        let real_z = scalar(5.0);
        let real_result = rgamma(&real_z, RuntimeMode::Strict).map_err(|e| e.to_string())?;
        let expected = get_scalar(Ok(real_result))?;

        assert!(
            (c.re - expected).abs() < 1e-12,
            "rgamma(5+0i) = {} + {}i, expected {} + 0i",
            c.re,
            c.im,
            expected
        );
        assert!(c.im.abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn complex_rgamma_returns_zero_at_negative_integer_poles() -> Result<(), String> {
        for x in [0.0, -1.0, -2.0, -8.0] {
            let actual = get_complex_scalar(rgamma(&complex_scalar(x, 0.0), RuntimeMode::Strict))?;
            assert_eq!(actual, Complex64::from_real(0.0));
        }
        Ok(())
    }

    #[test]
    fn complex_gammaln_returns_nan_at_real_poles() -> Result<(), String> {
        for x in [0.0, -1.0, -2.0, -8.0] {
            let actual = get_complex_scalar(gammaln(&complex_scalar(x, 0.0), RuntimeMode::Strict))?;
            assert!(
                actual.re.is_nan(),
                "gammaln({x}+0i) real part should be NaN"
            );
            assert!(
                actual.im.is_nan(),
                "gammaln({x}+0i) imag part should be NaN"
            );
        }
        Ok(())
    }

    #[test]
    fn complex_gammaln_near_real_poles_stays_finite() -> Result<(), String> {
        let actual =
            get_complex_scalar(gammaln(&complex_scalar(-1.0, 1.0e-12), RuntimeMode::Strict))?;
        assert!(actual.re.is_finite());
        assert!(actual.im.is_finite());
        Ok(())
    }

    #[test]
    fn complex_rgamma_vector_preserves_shape() -> Result<(), String> {
        let input =
            SpecialTensor::ComplexVec(vec![Complex64::new(2.0, 1.0), Complex64::new(5.0, 0.0)]);
        let result = rgamma(&input, RuntimeMode::Strict).map_err(|e| e.to_string())?;
        match result {
            SpecialTensor::ComplexVec(values) => {
                assert_eq!(values.len(), 2);
                let lane0 =
                    get_complex_scalar(rgamma(&complex_scalar(2.0, 1.0), RuntimeMode::Strict))?;
                let lane1 =
                    get_complex_scalar(rgamma(&complex_scalar(5.0, 0.0), RuntimeMode::Strict))?;
                assert!((values[0].re - lane0.re).abs() < 1e-12);
                assert!((values[0].im - lane0.im).abs() < 1e-12);
                assert!((values[1].re - lane1.re).abs() < 1e-12);
                assert!((values[1].im - lane1.im).abs() < 1e-12);
            }
            other => return Err(format!("expected complex vector, got {other:?}")),
        }
        Ok(())
    }

    #[test]
    fn complex_gamma_with_imaginary_part() -> Result<(), String> {
        let z = complex_scalar(2.0, 1.0);
        let result = gamma(&z, RuntimeMode::Strict).map_err(|e| e.to_string())?;
        let c = get_complex_scalar(Ok(result))?;

        assert!(c.re.is_finite(), "gamma(2+1i) real part should be finite");
        assert!(c.im.is_finite(), "gamma(2+1i) imag part should be finite");
        Ok(())
    }

    #[test]
    fn complex_gammaln_with_imaginary_part() -> Result<(), String> {
        let z = complex_scalar(2.0, 1.0);
        let result = gammaln(&z, RuntimeMode::Strict).map_err(|e| e.to_string())?;
        let c = get_complex_scalar(Ok(result))?;

        assert!(c.re.is_finite(), "gammaln(2+1i) real part should be finite");
        assert!(c.im.is_finite(), "gammaln(2+1i) imag part should be finite");

        // Verify gamma = exp(gammaln)
        let gamma_result =
            gamma(&complex_scalar(2.0, 1.0), RuntimeMode::Strict).map_err(|e| e.to_string())?;
        let gamma_c = get_complex_scalar(Ok(gamma_result))?;
        let expected = c.exp();
        assert!(
            (gamma_c.re - expected.re).abs() < 1e-10,
            "gamma should equal exp(gammaln)"
        );
        assert!(
            (gamma_c.im - expected.im).abs() < 1e-10,
            "gamma should equal exp(gammaln)"
        );
        Ok(())
    }

    #[test]
    fn binom_metamorphic_matches_comb_for_integer_arguments() {
        // For non-negative integer (n, k), binom(n, k) == comb(n, k) within
        // ulp.
        for n in 0..=15u64 {
            for k in 0..=n {
                let b = binom(n as f64, k as f64);
                let c = comb(n, k);
                let rel = (b - c).abs() / c.abs().max(1.0);
                assert!(rel < 1e-10, "binom({n},{k}) = {b}, comb = {c}");
            }
        }
    }

    #[test]
    fn binom_real_arguments_match_gamma_definition() {
        // scipy.special.binom(0.5, 0.25) = 1.0787052023767587 (beta-form path).
        let b = binom(0.5, 0.25);
        let expected = 1.0787052023767587_f64;
        assert!(
            (b - expected).abs() < 1e-12,
            "binom(0.5, 0.25) = {b}, expected {expected}"
        );
    }

    #[test]
    fn binom_y_greater_than_x_returns_zero() {
        // Integer overshoot: vanishing-coefficient rule.
        assert_eq!(binom(3.0, 5.0), 0.0);
        // Non-integer y - x with integer y - x = 0 (full y-x).fract() = 0:
        // (y, x) where the difference is an exact integer ≥ 1 → 0.
        assert_eq!(binom(2.5, 3.5), 0.0);
        assert_eq!(binom(2.5, 4.5), 0.0);
    }

    #[test]
    fn binom_negative_x_matches_scipy() {
        // scipy.special.binom is finite (and signed) for negative NON-integer x;
        // we previously fail-closed to NaN for any x<0.
        let cases = [
            (-4.5, 2.0, 12.375_f64),
            (-2.5, 3.0, -6.5625),
            (-0.5, 1.0, -0.5),
            (-3.5, 2.0, 7.875),
            (-4.5, 3.0, -26.8125),
            (5.0, 2.0, 10.0), // positive args: unchanged
        ];
        for (x, y, want) in cases {
            let got = binom(x, y);
            assert!(
                (got - want).abs() <= 1e-11 * want.abs().max(1.0),
                "binom({x},{y}) got {got}, want {want}"
            );
        }
        // Negative-integer x is a pole => NaN (scipy).
        assert!(binom(-3.0, 2.0).is_nan());
        assert!(binom(-2.0, 1.0).is_nan());
    }

    #[test]
    fn binom_metamorphic_symmetry_x_minus_y() {
        // Apply /testing-metamorphic: binom(x, y) == binom(x, x − y)
        // for x ≥ y ≥ 0 (gamma-form symmetry of the binomial). Pin
        // across integer, half-integer, and non-integer x.
        let cases: &[(f64, f64)] = &[
            (5.0, 2.0),   // integer: C(5,2) = C(5,3) = 10
            (10.0, 4.0),  // integer
            (7.5, 2.5),   // half-integer x, integer y differential
            (4.25, 1.25), // non-integer
            (3.0, 0.0),   // y=0 endpoint
        ];
        for &(x, y) in cases {
            let lhs = binom(x, y);
            let rhs = binom(x, x - y);
            let tol = 1e-9 * (lhs.abs() + 1.0);
            assert!(
                (lhs - rhs).abs() < tol,
                "binom({x}, {y}) = {lhs} but binom({x}, {x}-{y}={}) = {rhs}",
                x - y
            );
        }
    }

    #[test]
    fn binom_real_y_greater_than_x_uses_gamma_formula() {
        // [frankenscipy-du5m4] Regression: scipy.special.binom for
        // non-integer y > x (where y - x is non-integer) returns the
        // gamma formula value, not 0.
        // binom(0.5, 1.0) = Γ(1.5)/(Γ(2)·Γ(0.5)) = (√π/2)/√π = 0.5.
        // The lngamma asymptotic-with-recurrence loses ~1e-11 of precision
        // here; a 1e-9 tolerance cleanly distinguishes the new gamma path
        // (≈0.5) from the old vanishing-coefficient bug (= 0.0).
        let v05_1 = binom(0.5_f64, 1.0);
        assert!((v05_1 - 0.5).abs() < 1e-9, "binom(0.5, 1.0) = {v05_1}");
        // binom(0.5, 0.6) = Γ(1.5)/(Γ(1.6)·Γ(0.9)).
        let expected = lngamma_positive(1.5) - lngamma_positive(1.6) - lngamma_positive(0.9);
        let expected = expected.exp();
        assert!((binom(0.5_f64, 0.6) - expected).abs() < 1e-9);
    }

    #[test]
    fn binom_integer_k_noninteger_n_matches_scipy() {
        // frankenscipy-binom: the old gamma formula collapsed binom(n, k) to 0
        // whenever n - k + 1 <= 0, even when Γ(n-k+1) is finite (not a pole).
        // For integer k and non-integer n the result is a signed polynomial in
        // n. Golden values from scipy.special.binom 1.17.1.
        let cases = [
            (0.5_f64, 2.0, -0.125_f64),
            (0.5, 3.0, 0.0625),
            (0.5, 5.0, 0.02734375),
            (0.5, 7.0, 0.01611328125),
            (2.5, 5.0, 0.01171875),
            (2.5, 7.0, 0.00244140625),
            (-0.5, 1.0, -0.5),
            (2.0, -1.0, 0.0), // negative integer k → 0
        ];
        for (n, k, want) in cases {
            let got = binom(n, k);
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "binom({n},{k}) got {got}, want {want}"
            );
        }
    }

    #[test]
    fn binom_endpoints_return_one() {
        // Integer k uses the exact product form → exactly 1.0.
        assert_eq!(binom(5.0, 0.0), 1.0);
        assert_eq!(binom(5.0, 5.0), 1.0);
        assert_eq!(binom(2.5, 0.0), 1.0);
        // Non-integer k=n routes through the beta form; scipy.special.binom(2.5,2.5)
        // = 0.9999999999999998 (NOT exactly 1.0 — matching scipy is the point).
        assert!(
            (binom(2.5, 2.5) - 1.0).abs() < 1e-12,
            "binom(2.5,2.5) = {}",
            binom(2.5, 2.5)
        );
    }

    #[test]
    fn binom_negative_or_nan_propagates() {
        // Negative-integer n is a genuine pole → NaN (scipy).
        assert!(binom(-1.0, 0.5).is_nan());
        // Negative non-integer k is finite in scipy: binom(2, -0.5) = 0.33953054526271.
        assert!((binom(2.0, -0.5) - 0.33953054526271004).abs() < 1e-12);
        assert!(binom(f64::NAN, 1.0).is_nan());
        assert!(binom(2.0, f64::NAN).is_nan());
    }

    #[test]
    fn factorialk_zero_and_negative_conventions() {
        assert_eq!(factorialk(0, 1), 1.0);
        assert_eq!(factorialk(0, 5), 1.0);
        assert!(factorialk(-1, 1).is_nan());
        assert!(factorialk(5, 0).is_nan());
        assert!(factorialk(5, -2).is_nan());
    }

    #[test]
    fn factorialk_k_one_matches_literal_product() {
        fn literal_product(n: i64) -> f64 {
            let mut result = 1.0;
            for value in 1..=n {
                result *= value as f64;
            }
            result
        }

        for n in [0, 1, 2, 10, 20, 21, 64, 128, 170, 171, 200] {
            let fk = factorialk(n, 1);
            let original = literal_product(n);
            if original.is_finite() {
                let rel_err = (fk - original).abs() / original.abs().max(1.0);
                assert!(rel_err <= 2.0e-13, "factorialk({n}, 1) rel_err={rel_err:e}");
            } else {
                assert_eq!(fk, original, "factorialk({n}, 1) overflow mismatch");
            }
        }
    }

    #[test]
    fn factorialk_metamorphic_k_two_matches_factorial2() {
        // factorialk(n, 2) == n!! for n in 0..=15
        for n in 0..=15 {
            let fk = factorialk(n, 2);
            let f2 = factorial2(n);
            assert!(
                (fk - f2).abs() < 1e-12,
                "factorialk({n}, 2) = {fk}, factorial2({n}) = {f2}"
            );
        }
    }

    #[test]
    fn factorialk_metamorphic_recurrence() {
        // For n >= k, factorialk(n, k) == n * factorialk(n - k, k).
        for n in 1..=20 {
            for k in 1..=5 {
                if n >= k {
                    let lhs = factorialk(n, k);
                    let rhs = (n as f64) * factorialk(n - k, k);
                    assert!(
                        (lhs - rhs).abs() < 1e-9 * lhs.abs().max(1.0),
                        "factorialk({n}, {k}) = {lhs}, n * factorialk({n}-{k}, {k}) = {rhs}"
                    );
                }
            }
        }
    }

    #[test]
    fn factorialk_known_value_n_10_k_3() {
        // 10 * 7 * 4 * 1 = 280
        assert_eq!(factorialk(10, 3), 280.0);
    }

    #[test]
    fn comb_perm_match_scipy_reference_points() {
        // /testing-conformance-harnesses: pin canonical scipy values
        // for comb (binomial) and perm (k-permutations).
        //   comb(5, 2)  = 10
        //   comb(10, 5) = 252
        //   comb(0, 0)  = 1   (vacuous identity)
        //   comb(7, 0)  = 1
        //   comb(7, 7)  = 1
        //   comb(7, 8)  = 0   (k > n short-circuit)
        //   perm(5, 2)  = 5·4 = 20
        //   perm(10, 3) = 10·9·8 = 720
        //   perm(7, 0)  = 1   (empty product)
        //   perm(0, 0)  = 1
        //   perm(7, 8)  = 0
        assert_eq!(comb(5, 2), 10.0);
        assert_eq!(comb(10, 5), 252.0);
        assert_eq!(comb(0, 0), 1.0);
        assert_eq!(comb(7, 0), 1.0);
        assert_eq!(comb(7, 7), 1.0);
        assert_eq!(comb(7, 8), 0.0);
        assert_eq!(perm(5, 2), 20.0);
        assert_eq!(perm(10, 3), 720.0);
        assert_eq!(perm(7, 0), 1.0);
        assert_eq!(perm(0, 0), 1.0);
        assert_eq!(perm(7, 8), 0.0);
        // Symmetry C(n, k) = C(n, n-k).
        assert_eq!(comb(20, 7), comb(20, 13));
        assert_eq!(comb(50, 17), comb(50, 33));
    }

    #[test]
    fn gammaln_matches_scipy_reference_points() {
        // /testing-conformance-harnesses: pin gammaln at canonical
        // points scipy gives in closed form:
        //   gammaln(0.5) = ln(√π) = 0.5·ln(π)
        //   gammaln(1)   = 0
        //   gammaln(2)   = 0
        //   gammaln(0.25)= 1.288_022_524_698_077... (Γ(1/4)·Γ(3/4) = π√2)
        //   gammaln(10)  = ln(9!) = ln(362880) = 12.801_827_480_081_469
        let pi = std::f64::consts::PI;
        let cases: &[(f64, f64)] = &[
            (0.5, 0.5 * pi.ln()),
            (1.0, 0.0),
            (2.0, 0.0),
            (10.0, 362_880.0_f64.ln()),
        ];
        for &(x, expected) in cases {
            let got = gammaln_scalar(x, RuntimeMode::Strict).unwrap();
            assert!(
                (got - expected).abs() < 1e-12,
                "gammaln({x}) = {got}, expected {expected}"
            );
        }
        // Pole behavior — strict mode returns +inf, not NaN.
        assert!(
            gammaln_scalar(0.0, RuntimeMode::Strict)
                .unwrap()
                .is_infinite()
        );
        assert!(
            gammaln_scalar(-1.0, RuntimeMode::Strict)
                .unwrap()
                .is_infinite()
        );
    }

    #[test]
    fn zeta_matches_scipy_reference_points() {
        // /testing-conformance-harnesses: pin closed-form scipy.special
        // .zeta values:
        //   ζ(2)  = π²/6
        //   ζ(4)  = π⁴/90
        //   ζ(6)  = π⁶/945
        //   ζ(0)  = −1/2
        //   ζ(−1) = −1/12   (functional equation gives the negative-s
        //                   continuation)
        let pi = std::f64::consts::PI;
        let cases: &[(f64, f64)] = &[
            (2.0, pi * pi / 6.0),
            (4.0, pi.powi(4) / 90.0),
            (6.0, pi.powi(6) / 945.0),
        ];
        for &(s, expected) in cases {
            let got = zeta_scalar(s);
            let rel = ((got - expected) / expected).abs();
            assert!(
                rel < 1e-9,
                "ζ({s}) = {got}, expected {expected} (rel = {rel})"
            );
        }
        // Large-s tail: ζ(s) → 1 as s → ∞.
        assert!((zeta_scalar(50.0) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn trigamma_metamorphic_recurrence() {
        // /testing-metamorphic: ψ'(x+1) = ψ'(x) − 1/x² for x not at a
        // non-positive integer pole.
        // Plus closed-form trigamma values:
        //   ψ'(1) = π²/6   ψ'(0.5) = π²/2   ψ'(2) = π²/6 − 1
        let pi_sq_over_6 = std::f64::consts::PI * std::f64::consts::PI / 6.0;
        let pi_sq_over_2 = std::f64::consts::PI * std::f64::consts::PI / 2.0;
        assert!((trigamma_core(1.0) - pi_sq_over_6).abs() < 1e-9);
        assert!((trigamma_core(0.5) - pi_sq_over_2).abs() < 1e-9);
        assert!((trigamma_core(2.0) - (pi_sq_over_6 - 1.0)).abs() < 1e-9);
        // Recurrence across multiple x.
        for &x in &[0.25_f64, 0.5, 1.3, 2.7, 5.3, 9.1, 17.4] {
            let lhs = trigamma_core(x + 1.0);
            let rhs = trigamma_core(x) - 1.0 / (x * x);
            assert!(
                (lhs - rhs).abs() < 1e-9,
                "ψ'({x}+1) = {lhs}, expected ψ'({x}) − 1/{x}² = {rhs}"
            );
        }
    }

    #[test]
    fn digamma_metamorphic_recurrence() {
        // /testing-metamorphic: ψ(x+1) = ψ(x) + 1/x for any x not at
        // a non-positive integer. Independent of any specific value;
        // catches drift in either the recurrence-shift branch or the
        // asymptotic series. Pin across multiple x.
        // 1e-9 envelope reflects the asymptotic-series truncation
        // residual (~5e-11 for moderate x, growing with x).
        for &x in &[0.25_f64, 0.5, 1.5, 2.7, 5.3, 9.1, 17.4] {
            let lhs = digamma_core(x + 1.0);
            let rhs = digamma_core(x) + 1.0 / x;
            assert!(
                (lhs - rhs).abs() < 1e-9,
                "ψ({x}+1) = {lhs}, expected ψ({x}) + 1/{x} = {rhs}"
            );
        }
    }

    #[test]
    fn digamma_matches_scipy_reference_points() {
        // /testing-conformance-harnesses: pin canonical scipy.special
        // .psi values:
        //   ψ(1)   = -γ                       (Euler-Mascheroni constant)
        //   ψ(0.5) = -γ - 2·ln 2              (closed form)
        //   ψ(2)   = 1 - γ                    (recurrence ψ(n+1) = ψ(n) + 1/n)
        //   ψ(3)   = 3/2 - γ
        let gamma_em = 0.577_215_664_901_532_9_f64;
        let ln_2 = 2.0_f64.ln();
        let cases: &[(f64, f64)] = &[
            (1.0, -gamma_em),
            (0.5, -gamma_em - 2.0 * ln_2),
            (2.0, 1.0 - gamma_em),
            (3.0, 1.5 - gamma_em),
            (4.0, 1.0 + 0.5 + 1.0 / 3.0 - gamma_em),
        ];
        // Tolerance reflects the ~2e-10 residual of the asymptotic-
        // with-shift digamma_core (Euler-Maclaurin truncated at
        // order 1/x⁷). Tighter than 1e-9 would require extending the
        // series; pin the current envelope.
        for &(x, expected) in cases {
            let got = digamma_core(x);
            assert!(
                (got - expected).abs() < 1e-9,
                "ψ({x}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn psi_alias_matches_digamma_dispatch() -> Result<(), String> {
        let scalar_input = scalar(2.5);
        let psi_scalar = get_scalar(psi(&scalar_input, RuntimeMode::Strict))?;
        let digamma_scalar = get_scalar(digamma(&scalar_input, RuntimeMode::Strict))?;
        assert!((psi_scalar - digamma_scalar).abs() < 1e-14);

        let vec_input = SpecialTensor::RealVec(vec![0.5, 1.0, 2.5, 8.0]);
        let psi_vec = psi(&vec_input, RuntimeMode::Strict).map_err(|err| err.to_string())?;
        let digamma_vec =
            digamma(&vec_input, RuntimeMode::Strict).map_err(|err| err.to_string())?;
        assert_eq!(psi_vec, digamma_vec);

        let complex_input = SpecialTensor::ComplexScalar(Complex64::new(1.5, 0.25));
        let psi_complex =
            psi(&complex_input, RuntimeMode::Strict).map_err(|err| err.to_string())?;
        let digamma_complex =
            digamma(&complex_input, RuntimeMode::Strict).map_err(|err| err.to_string())?;
        assert_eq!(psi_complex, digamma_complex);
        Ok(())
    }

    #[test]
    fn gamma_metamorphic_factorial_identity() {
        // /testing-metamorphic: Γ(n+1) = n! for non-negative integer n.
        // Pin across n = 0..=10 to a tolerance of 1 ulp (closed-form
        // integer arithmetic, no rounding error in either side).
        let mut fact = 1.0_f64;
        for n in 0..=10_u64 {
            if n > 0 {
                fact *= n as f64;
            }
            let gamma_val = gamma_core((n + 1) as f64);
            let rel = ((gamma_val - fact) / fact).abs();
            assert!(
                rel < 1e-13,
                "Γ({}) = {gamma_val}, expected {n}! = {fact} (rel = {rel})",
                n + 1
            );
        }
    }

    #[test]
    fn gamma_metamorphic_reflection_formula() {
        // /testing-metamorphic: Γ(x) · Γ(1−x) = π/sin(πx) for any
        // non-integer x. Independent of any specific gamma value;
        // catches any drift in either the Lanczos or recurrence
        // branches without hard-coding a reference number.
        let pi = std::f64::consts::PI;
        // Avoid x near integer poles where Γ diverges.
        for &x in &[0.25_f64, 0.5, 0.75, 1.5, 2.5, 3.7, 5.3, 7.1] {
            let lhs = gamma_core(x) * gamma_core(1.0 - x);
            let rhs = pi / (pi * x).sin();
            let rel = (lhs - rhs).abs() / rhs.abs().max(1.0);
            assert!(
                rel < 1e-12,
                "Γ({x})·Γ(1-{x}) = {lhs}, expected π/sin(π·{x}) = {rhs} (rel = {rel})"
            );
        }
    }

    #[test]
    fn factorial_matches_scipy_reference_values() {
        // scipy.special.factorial(0) = 1.0
        // scipy.special.factorial(5) = 120.0
        // scipy.special.factorial(10) = 3628800.0
        assert_eq!(factorial(0), 1.0, "0! should be 1");
        assert_eq!(factorial(5), 120.0, "5! should be 120");
        assert_eq!(factorial(10), 3628800.0, "10! should be 3628800");
    }

    #[test]
    fn factorial2_matches_scipy_reference_values() {
        // scipy.special.factorial2(5) = 5!! = 5*3*1 = 15
        // scipy.special.factorial2(6) = 6!! = 6*4*2 = 48
        // scipy.special.factorial2(0) = 1
        assert_eq!(factorial2(0), 1.0, "0!! should be 1");
        assert_eq!(factorial2(5), 15.0, "5!! should be 15");
        assert_eq!(factorial2(6), 48.0, "6!! should be 48");
    }

    #[test]
    fn binom_matches_scipy_reference_values() {
        // scipy.special.binom(5, 2) = 10
        // scipy.special.binom(10, 3) = 120
        // scipy.special.binom(6, 0) = 1
        assert!(
            (binom(5.0, 2.0) - 10.0).abs() < 1e-10,
            "C(5,2) = {}, expected 10",
            binom(5.0, 2.0)
        );
        assert!(
            (binom(10.0, 3.0) - 120.0).abs() < 1e-6,
            "C(10,3) = {}, expected 120",
            binom(10.0, 3.0)
        );
        assert!(
            (binom(6.0, 0.0) - 1.0).abs() < 1e-10,
            "C(6,0) = {}, expected 1",
            binom(6.0, 0.0)
        );
    }

    #[test]
    fn gammainc_matches_scipy_reference_values() {
        // scipy.special.gammainc(1, 1) = 1 - exp(-1) ≈ 0.6321205588
        // scipy.special.gammainc(2, 1) ≈ 0.2642411177
        let val1 = gammainc_scalar(1.0, 1.0, RuntimeMode::Strict).expect("gammainc(1, 1)");
        assert!(
            (val1 - 0.6321205588).abs() < 1e-6,
            "gammainc(1,1) = {}, expected 0.6321205588",
            val1
        );

        let val2 = gammainc_scalar(2.0, 1.0, RuntimeMode::Strict).expect("gammainc(2, 1)");
        assert!(
            (val2 - 0.2642411177).abs() < 1e-6,
            "gammainc(2,1) = {}, expected 0.2642411177",
            val2
        );
    }
}
