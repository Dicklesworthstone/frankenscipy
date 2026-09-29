#![forbid(unsafe_code)]
//! frankenscipy-3cu8u.2: every audited public API fails closed exactly once.
//!
//! The README's error model (rule 4) promises that an error returned to the caller is also in
//! the audit ledger as a `FailClosed` event. For every public function that takes an audit
//! ledger (the inventory is in the bead), this test drives invalid inputs (non-finite, empty,
//! shape mismatch, out-of-domain, singular) in each runtime mode the function has, plus a valid
//! input per mode, and checks the ledger each call leaves behind:
//!
//! - a call that returns `Err` records exactly one `FailClosed`, as its last event, with the
//!   reason code the table names and a `blake3:` fingerprint of the call;
//! - the same call made again records the same fingerprint, and different rejected inputs of one
//!   API record different fingerprints;
//! - a call that succeeds records no `FailClosed`.
//!
//! Events a call records before it fails (a CASP decision, a bounded recovery) are allowed; the
//! `FailClosed` must come after them.

use std::collections::BTreeMap;
use std::fmt::Display;

use fsci_arrayapi::{
    ArangeRequest, CoreArrayBackend, CreationRequest, DType, ExecutionMode, IndexExpr,
    IndexRequest, IndexingMode, MemoryOrder, ScalarValue, Shape, SliceSpec, arange_with_audit,
    broadcast_shapes_with_audit, from_slice_with_audit, getitem_with_audit, reshape_with_audit,
};
use fsci_fft::{
    Complex64, FftOptions, fft_with_audit, fft2_with_audit, fftn_with_audit, hfft_with_audit,
    hfft2_with_audit, hfftn_with_audit, ifft_with_audit, ifft2_with_audit, ifftn_with_audit,
    ihfft_with_audit, ihfft2_with_audit, ihfftn_with_audit, irfft_with_audit, irfft2_with_audit,
    irfftn_with_audit, rfft_with_audit, rfft2_with_audit, rfftn_with_audit,
};
use fsci_integrate::{
    SolveIvpOptions, ToleranceValue, solve_ivp_with_audit, validate_first_step_with_audit,
    validate_max_step_with_audit, validate_tol_with_audit,
};
use fsci_linalg::{
    InvOptions, LstsqOptions, PinvOptions, SolveOptions, TriangularSolveOptions, det_with_audit,
    inv_with_audit, lstsq_with_audit, pinv_with_audit, solve_banded_with_audit,
    solve_triangular_with_audit, solve_with_audit,
};
use fsci_runtime::{
    AuditAction, AuditEvent, AuditLedger, RuntimeMode, SolverPortfolio, SparseSolverPortfolio,
    SyncSharedAuditLedger,
};

/// What a case must do.
#[derive(Clone, Copy)]
enum Expect {
    /// Return `Err` and record one `FailClosed` with this reason.
    Reject(&'static str),
    /// Return `Ok` and record no `FailClosed`.
    Accept,
}

type Call = Box<dyn Fn(&SyncSharedAuditLedger) -> Result<(), String>>;

struct Case {
    api: &'static str,
    class: &'static str,
    mode: &'static str,
    expect: Expect,
    call: Call,
}

fn case(
    api: &'static str,
    class: &'static str,
    mode: &'static str,
    expect: Expect,
    call: impl Fn(&SyncSharedAuditLedger) -> Result<(), String> + 'static,
) -> Case {
    Case {
        api,
        class,
        mode,
        expect,
        call: Box::new(call),
    }
}

fn done<T, E: Display>(result: Result<T, E>) -> Result<(), String> {
    result.map(drop).map_err(|error| error.to_string())
}

const MODES: [(RuntimeMode, &str); 2] = [
    (RuntimeMode::Strict, "Strict"),
    (RuntimeMode::Hardened, "Hardened"),
];

fn run(case: &Case) -> (Result<(), String>, Vec<AuditEvent>) {
    let ledger = AuditLedger::shared();
    let result = (case.call)(&ledger);
    let events = ledger.lock().expect("ledger").entries().to_vec();
    (result, events)
}

fn describe(events: &[AuditEvent]) -> String {
    events
        .iter()
        .map(|event| format!("{:?} -> {}", event.action, event.outcome))
        .collect::<Vec<_>>()
        .join("; ")
}

fn is_fingerprint(value: &str) -> bool {
    value
        .strip_prefix("blake3:")
        .is_some_and(|hex| hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit()))
}

fn linalg_cases(cases: &mut Vec<Case>) {
    let square = || vec![vec![4.0, 1.0], vec![1.0, 3.0]];
    let singular = || vec![vec![1.0, 2.0], vec![2.0, 4.0]];
    for (mode, label) in MODES {
        let solve_options = SolveOptions {
            mode,
            ..SolveOptions::default()
        };
        let solve = move |a: Vec<Vec<f64>>, b: Vec<f64>| {
            move |ledger: &SyncSharedAuditLedger| {
                let mut portfolio = SolverPortfolio::new(mode, 8);
                done(solve_with_audit(
                    &a,
                    &b,
                    solve_options,
                    &mut portfolio,
                    ledger,
                ))
            }
        };
        let rows: [(&str, Vec<Vec<f64>>, Vec<f64>, Expect); 7] = [
            (
                "ragged",
                vec![vec![1.0, 2.0], vec![3.0]],
                vec![1.0, 2.0],
                Expect::Reject("ragged_matrix"),
            ),
            (
                "non_square",
                vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
                vec![1.0, 2.0],
                Expect::Reject("non_square_matrix"),
            ),
            (
                "shape_mismatch",
                square(),
                vec![1.0, 2.0, 3.0],
                Expect::Reject("incompatible_shapes"),
            ),
            (
                "non_finite_matrix",
                vec![vec![f64::NAN, 1.0], vec![1.0, 3.0]],
                vec![1.0, 2.0],
                Expect::Reject("non_finite_matrix"),
            ),
            (
                "non_finite_vector",
                square(),
                vec![f64::INFINITY, 2.0],
                Expect::Reject("non_finite_vector"),
            ),
            (
                "singular",
                singular(),
                vec![1.0, 2.0],
                Expect::Reject("singular_matrix"),
            ),
            ("valid", square(), vec![1.0, 2.0], Expect::Accept),
        ];
        for (class, a, b, expect) in rows {
            cases.push(case(
                "linalg::solve_with_audit",
                class,
                label,
                expect,
                solve(a, b),
            ));
        }

        let inv_options = InvOptions {
            mode,
            ..InvOptions::default()
        };
        for (class, a, expect) in [
            (
                "non_square",
                vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
                Expect::Reject("non_square_matrix"),
            ),
            (
                "non_finite",
                vec![vec![f64::NAN, 1.0], vec![1.0, 3.0]],
                Expect::Reject("non_finite_input"),
            ),
            ("singular", singular(), Expect::Reject("singular_matrix")),
            ("valid", square(), Expect::Accept),
        ] {
            cases.push(case(
                "linalg::inv_with_audit",
                class,
                label,
                expect,
                move |l| done(inv_with_audit(&a, inv_options, l)),
            ));
        }

        for (class, a, expect) in [
            (
                "non_square",
                vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
                Expect::Reject("non_square_matrix"),
            ),
            (
                "non_finite",
                vec![vec![f64::NAN, 1.0], vec![1.0, 3.0]],
                Expect::Reject("non_finite_input"),
            ),
            ("valid", square(), Expect::Accept),
        ] {
            cases.push(case(
                "linalg::det_with_audit",
                class,
                label,
                expect,
                move |l| done(det_with_audit(&a, mode, true, l)),
            ));
        }

        let lstsq_options = LstsqOptions {
            mode,
            ..LstsqOptions::default()
        };
        for (class, b, expect) in [
            (
                "shape_mismatch",
                vec![1.0, 2.0, 3.0],
                Expect::Reject("incompatible_shapes"),
            ),
            (
                "non_finite",
                vec![f64::NAN, 2.0],
                Expect::Reject("non_finite_input"),
            ),
            ("valid", vec![1.0, 2.0], Expect::Accept),
        ] {
            cases.push(case(
                "linalg::lstsq_with_audit",
                class,
                label,
                expect,
                move |l| done(lstsq_with_audit(&square(), &b, lstsq_options, l)),
            ));
        }

        for (class, a, atol, expect) in [
            (
                "non_finite",
                vec![vec![f64::NAN, 1.0], vec![1.0, 3.0]],
                None,
                Expect::Reject("non_finite_input"),
            ),
            (
                "bad_threshold",
                square(),
                Some(-1.0),
                Expect::Reject("invalid_pinv_threshold"),
            ),
            ("valid", square(), None, Expect::Accept),
        ] {
            let options = PinvOptions {
                mode,
                atol,
                ..PinvOptions::default()
            };
            cases.push(case(
                "linalg::pinv_with_audit",
                class,
                label,
                expect,
                move |l| done(pinv_with_audit(&a, options, l)),
            ));
        }

        let triangular_options = TriangularSolveOptions {
            mode,
            lower: true,
            ..TriangularSolveOptions::default()
        };
        for (class, a, expect) in [
            (
                "non_square",
                vec![vec![1.0, 0.0, 0.0], vec![4.0, 5.0, 0.0]],
                Expect::Reject("non_square_matrix"),
            ),
            (
                "singular",
                vec![vec![2.0, 0.0], vec![1.0, 0.0]],
                Expect::Reject("singular_matrix"),
            ),
            (
                "valid",
                vec![vec![2.0, 0.0], vec![1.0, 3.0]],
                Expect::Accept,
            ),
        ] {
            cases.push(case(
                "linalg::solve_triangular_with_audit",
                class,
                label,
                expect,
                move |l| {
                    done(solve_triangular_with_audit(
                        &a,
                        &[1.0, 2.0],
                        triangular_options,
                        l,
                    ))
                },
            ));
        }

        for (class, ab, expect) in [
            (
                "bad_band",
                vec![vec![4.0, 4.0, 4.0]],
                Expect::Reject("invalid_band_shape"),
            ),
            (
                "valid",
                vec![
                    vec![0.0, 1.0, 1.0],
                    vec![4.0, 4.0, 4.0],
                    vec![1.0, 1.0, 0.0],
                ],
                Expect::Accept,
            ),
        ] {
            cases.push(case(
                "linalg::solve_banded_with_audit",
                class,
                label,
                expect,
                move |l| {
                    done(solve_banded_with_audit(
                        (1, 1),
                        &ab,
                        &[1.0, 2.0, 3.0],
                        solve_options,
                        l,
                    ))
                },
            ));
        }
    }
}

fn fft_cases(cases: &mut Vec<Case>) {
    // `fsci_fft::Complex64` is `(re, im)`.
    let c = |re: f64| -> Complex64 { (re, 0.0) };
    for (mode, label) in MODES {
        // `FftOptions` is not `Copy`; each call builds its own.
        let opts = move || FftOptions {
            mode,
            check_finite: true,
            ..FftOptions::default()
        };
        // 1-D complex-input transforms.
        type Complex1d = fn(
            &[Complex64],
            &FftOptions,
            &SyncSharedAuditLedger,
        ) -> Result<Vec<Complex64>, fsci_fft::FftError>;
        for (api, transform) in [
            ("fft::fft_with_audit", fft_with_audit as Complex1d),
            ("fft::ifft_with_audit", ifft_with_audit as Complex1d),
        ] {
            for (class, input, expect) in [
                ("empty", vec![], Expect::Reject("empty_input")),
                (
                    "non_finite",
                    vec![c(1.0), c(f64::NAN), c(3.0)],
                    Expect::Reject("non_finite_input"),
                ),
                ("valid", vec![c(1.0), c(2.0), c(3.0)], Expect::Accept),
            ] {
                cases.push(case(api, class, label, expect, move |l| {
                    done(transform(&input, &opts(), l))
                }));
            }
        }
        for (class, input, expect) in [
            ("empty", vec![], Expect::Reject("empty_input")),
            (
                "non_finite",
                vec![1.0, f64::NAN, 3.0],
                Expect::Reject("non_finite_input"),
            ),
            ("valid", vec![1.0, 2.0, 3.0], Expect::Accept),
        ] {
            let input_again = input.clone();
            cases.push(case(
                "fft::rfft_with_audit",
                class,
                label,
                expect,
                move |l| done(rfft_with_audit(&input, &opts(), l)),
            ));
            cases.push(case(
                "fft::ihfft_with_audit",
                class,
                label,
                expect,
                move |l| done(ihfft_with_audit(&input_again, None, &opts(), l)),
            ));
        }
        for (class, input, n, expect) in [
            ("empty", vec![], None, Expect::Reject("empty_input")),
            (
                "non_finite",
                vec![c(1.0), c(f64::NAN), c(3.0)],
                None,
                Expect::Reject("non_finite_input"),
            ),
            (
                "zero_length",
                vec![c(1.0), c(2.0), c(3.0)],
                Some(0),
                Expect::Reject("invalid_output_len"),
            ),
            ("valid", vec![c(1.0), c(2.0), c(3.0)], None, Expect::Accept),
        ] {
            let input_again = input.clone();
            cases.push(case(
                "fft::irfft_with_audit",
                class,
                label,
                expect,
                move |l| done(irfft_with_audit(&input, n, &opts(), l)),
            ));
            cases.push(case(
                "fft::hfft_with_audit",
                class,
                label,
                expect,
                move |l| done(hfft_with_audit(&input_again, n, &opts(), l)),
            ));
        }

        // 2-D and N-D transforms over a 2 × 4 array. The inverse real and Hermitian forward
        // transforms take the half spectrum, 2 × 3 values, and produce 2 × 4.
        let full: Vec<Complex64> = (0..8).map(|v| c(f64::from(v))).collect();
        let half: Vec<Complex64> = (0..6).map(|v| c(f64::from(v))).collect();
        let real: Vec<f64> = (0..8).map(f64::from).collect();
        let with_nan_c = |mut v: Vec<Complex64>| {
            v[1] = c(f64::NAN);
            v
        };
        let with_nan_r = |mut v: Vec<f64>| {
            v[1] = f64::NAN;
            v
        };
        let complex_rows = |input: &[Complex64], short: usize| {
            [
                (
                    "length_mismatch",
                    input[..short].to_vec(),
                    Expect::Reject("length_mismatch"),
                ),
                (
                    "non_finite",
                    with_nan_c(input.to_vec()),
                    Expect::Reject("non_finite_input"),
                ),
                ("valid", input.to_vec(), Expect::Accept),
            ]
        };
        type Complex2d = fn(
            &[Complex64],
            (usize, usize),
            &FftOptions,
            &SyncSharedAuditLedger,
        ) -> Result<Vec<Complex64>, fsci_fft::FftError>;
        type ComplexNd = fn(
            &[Complex64],
            &[usize],
            &FftOptions,
            &SyncSharedAuditLedger,
        ) -> Result<Vec<Complex64>, fsci_fft::FftError>;
        type HalfToReal2d = fn(
            &[Complex64],
            (usize, usize),
            &FftOptions,
            &SyncSharedAuditLedger,
        ) -> Result<Vec<f64>, fsci_fft::FftError>;
        type HalfToRealNd = fn(
            &[Complex64],
            &[usize],
            &FftOptions,
            &SyncSharedAuditLedger,
        ) -> Result<Vec<f64>, fsci_fft::FftError>;
        type Real2d = fn(
            &[f64],
            (usize, usize),
            &FftOptions,
            &SyncSharedAuditLedger,
        ) -> Result<Vec<Complex64>, fsci_fft::FftError>;
        type RealNd = fn(
            &[f64],
            &[usize],
            &FftOptions,
            &SyncSharedAuditLedger,
        ) -> Result<Vec<Complex64>, fsci_fft::FftError>;
        for (api, transform) in [
            ("fft::fft2_with_audit", fft2_with_audit as Complex2d),
            ("fft::ifft2_with_audit", ifft2_with_audit as Complex2d),
        ] {
            for (class, input, expect) in complex_rows(&full, 7) {
                cases.push(case(api, class, label, expect, move |l| {
                    done(transform(&input, (2, 4), &opts(), l))
                }));
            }
        }
        for (api, transform) in [
            ("fft::fftn_with_audit", fftn_with_audit as ComplexNd),
            ("fft::ifftn_with_audit", ifftn_with_audit as ComplexNd),
        ] {
            for (class, input, expect) in complex_rows(&full, 7) {
                cases.push(case(api, class, label, expect, move |l| {
                    done(transform(&input, &[2, 4], &opts(), l))
                }));
            }
        }
        for (api, transform) in [
            ("fft::irfft2_with_audit", irfft2_with_audit as HalfToReal2d),
            ("fft::hfft2_with_audit", hfft2_with_audit as HalfToReal2d),
        ] {
            for (class, input, expect) in complex_rows(&half, 5) {
                cases.push(case(api, class, label, expect, move |l| {
                    done(transform(&input, (2, 4), &opts(), l))
                }));
            }
        }
        for (api, transform) in [
            ("fft::irfftn_with_audit", irfftn_with_audit as HalfToRealNd),
            ("fft::hfftn_with_audit", hfftn_with_audit as HalfToRealNd),
        ] {
            for (class, input, expect) in complex_rows(&half, 5) {
                cases.push(case(api, class, label, expect, move |l| {
                    done(transform(&input, &[2, 4], &opts(), l))
                }));
            }
        }
        let real_rows = || {
            [
                (
                    "length_mismatch",
                    real[..7].to_vec(),
                    Expect::Reject("length_mismatch"),
                ),
                (
                    "non_finite",
                    with_nan_r(real.clone()),
                    Expect::Reject("non_finite_input"),
                ),
                ("valid", real.clone(), Expect::Accept),
            ]
        };
        for (api, transform) in [
            ("fft::rfft2_with_audit", rfft2_with_audit as Real2d),
            ("fft::ihfft2_with_audit", ihfft2_with_audit as Real2d),
        ] {
            for (class, input, expect) in real_rows() {
                cases.push(case(api, class, label, expect, move |l| {
                    done(transform(&input, (2, 4), &opts(), l))
                }));
            }
        }
        for (api, transform) in [
            ("fft::rfftn_with_audit", rfftn_with_audit as RealNd),
            ("fft::ihfftn_with_audit", ihfftn_with_audit as RealNd),
        ] {
            for (class, input, expect) in real_rows() {
                cases.push(case(api, class, label, expect, move |l| {
                    done(transform(&input, &[2, 4], &opts(), l))
                }));
            }
        }
    }
}

fn arrayapi_cases(cases: &mut Vec<Case>) {
    let modes = [
        (ExecutionMode::Strict, "Strict"),
        (ExecutionMode::Hardened, "Hardened"),
    ];
    for (shapes, class, expect) in [
        (
            vec![Shape::new(vec![2, 3]), Shape::new(vec![4])],
            "incompatible",
            Expect::Reject("broadcast_shapes::BroadcastIncompatible"),
        ),
        (
            vec![Shape::new(vec![2, 3]), Shape::new(vec![5, 1])],
            "incompatible_leading",
            Expect::Reject("broadcast_shapes::BroadcastIncompatible"),
        ),
        (
            vec![Shape::new(vec![2, 3]), Shape::new(vec![3])],
            "valid",
            Expect::Accept,
        ),
    ] {
        cases.push(case(
            "arrayapi::broadcast_shapes_with_audit",
            class,
            "-",
            expect,
            move |l| done(broadcast_shapes_with_audit(&shapes, l)),
        ));
    }
    for (execution, label) in modes {
        for (class, start, step, expect) in [
            ("zero_step", 0.0, 0.0, Expect::Reject("arange::InvalidStep")),
            (
                "non_finite",
                f64::NAN,
                0.5,
                Expect::Reject("arange::NonFiniteInput"),
            ),
            ("valid", 0.0, 0.5, Expect::Accept),
        ] {
            cases.push(case(
                "arrayapi::arange_with_audit",
                class,
                label,
                expect,
                move |l| {
                    let request = ArangeRequest {
                        start: ScalarValue::F64(start),
                        stop: ScalarValue::F64(2.0),
                        step: ScalarValue::F64(step),
                        dtype: None,
                    };
                    done(arange_with_audit(
                        &CoreArrayBackend::new(execution),
                        &request,
                        l,
                    ))
                },
            ));
        }
        for (class, len, expect) in [
            (
                "length_mismatch",
                2,
                Expect::Reject("from_slice::InvalidShape"),
            ),
            (
                "length_mismatch_long",
                4,
                Expect::Reject("from_slice::InvalidShape"),
            ),
            ("valid", 3, Expect::Accept),
        ] {
            cases.push(case(
                "arrayapi::from_slice_with_audit",
                class,
                label,
                expect,
                move |l| {
                    let values: Vec<ScalarValue> =
                        (0..len).map(|v| ScalarValue::F64(f64::from(v))).collect();
                    let request = CreationRequest {
                        shape: Shape::new(vec![3]),
                        dtype: DType::Float64,
                        order: MemoryOrder::C,
                    };
                    done(from_slice_with_audit(
                        &CoreArrayBackend::new(execution),
                        &values,
                        &request,
                        l,
                    ))
                },
            ));
        }
        let array = move || {
            let backend = CoreArrayBackend::new(execution);
            let values: Vec<ScalarValue> = (0..6).map(|v| ScalarValue::F64(f64::from(v))).collect();
            let request = CreationRequest {
                shape: Shape::new(vec![2, 3]),
                dtype: DType::Float64,
                order: MemoryOrder::C,
            };
            fsci_arrayapi::from_slice(&backend, &values, &request).expect("array")
        };
        for (class, index, expect) in [
            (
                "mask_shape",
                IndexRequest {
                    mode: IndexingMode::BooleanMask,
                    index: IndexExpr::BooleanMask {
                        mask_shape: Shape::new(vec![3, 2]),
                    },
                },
                Expect::Reject("getitem::InvalidShape"),
            ),
            (
                "mode_mismatch",
                IndexRequest {
                    mode: IndexingMode::Advanced,
                    index: IndexExpr::BooleanMask {
                        mask_shape: Shape::new(vec![2, 3]),
                    },
                },
                Expect::Reject("getitem::InvalidIndex"),
            ),
            (
                "valid",
                IndexRequest {
                    mode: IndexingMode::Basic,
                    index: IndexExpr::Basic {
                        // One slice per axis of the 2 × 3 array.
                        slices: vec![
                            SliceSpec {
                                start: Some(0),
                                stop: Some(1),
                                step: 1,
                            },
                            SliceSpec {
                                start: None,
                                stop: None,
                                step: 1,
                            },
                        ],
                    },
                },
                Expect::Accept,
            ),
        ] {
            cases.push(case(
                "arrayapi::getitem_with_audit",
                class,
                label,
                expect,
                move |l| {
                    done(getitem_with_audit(
                        &CoreArrayBackend::new(execution),
                        &array(),
                        &index,
                        l,
                    ))
                },
            ));
        }
        for (class, dims, expect) in [
            (
                "element_count",
                vec![4, 2],
                Expect::Reject("reshape::InvalidShape"),
            ),
            (
                "element_count_1d",
                vec![5],
                Expect::Reject("reshape::InvalidShape"),
            ),
            ("valid", vec![3, 2], Expect::Accept),
        ] {
            cases.push(case(
                "arrayapi::reshape_with_audit",
                class,
                label,
                expect,
                move |l| {
                    done(reshape_with_audit(
                        &CoreArrayBackend::new(execution),
                        &array(),
                        &Shape::new(dims.clone()),
                        l,
                    ))
                },
            ));
        }
    }
}

fn integrate_cases(cases: &mut Vec<Case>) {
    for (mode, label) in MODES {
        for (class, y0, max_step, t_eval, rhs_len, expect) in [
            (
                "empty",
                vec![],
                f64::INFINITY,
                None,
                1,
                Expect::Reject("empty_y0"),
            ),
            (
                "non_finite",
                vec![f64::NAN],
                f64::INFINITY,
                None,
                1,
                Expect::Reject("non_finite_y0"),
            ),
            (
                "bad_max_step",
                vec![1.0],
                0.0,
                None,
                1,
                Expect::Reject("max_step_must_be_positive"),
            ),
            (
                "t_eval_out_of_span",
                vec![1.0],
                f64::INFINITY,
                Some(vec![0.5, 2.0]),
                1,
                Expect::Reject("t_eval_out_of_span"),
            ),
            (
                "rhs_wrong_shape",
                vec![1.0],
                f64::INFINITY,
                None,
                2,
                Expect::Reject("rhs_wrong_shape"),
            ),
            ("valid", vec![1.0], f64::INFINITY, None, 1, Expect::Accept),
        ] {
            cases.push(case(
                "integrate::solve_ivp_with_audit",
                class,
                label,
                expect,
                move |l| {
                    let ivp = SolveIvpOptions {
                        t_span: (0.0, 1.0),
                        y0: &y0,
                        t_eval: t_eval.as_deref(),
                        max_step,
                        mode,
                        ..SolveIvpOptions::default()
                    };
                    let mut fun = |_t: f64, y: &[f64]| vec![-y[0]; rhs_len];
                    done(solve_ivp_with_audit(&mut fun, &ivp, l))
                },
            ));
        }
        for (class, rtol, atol, expect) in [
            (
                "nan_rtol",
                ToleranceValue::Scalar(f64::NAN),
                ToleranceValue::Scalar(1e-6),
                Expect::Reject("rtol_must_not_be_nan"),
            ),
            (
                "atol_shape",
                ToleranceValue::Scalar(1e-3),
                ToleranceValue::Vector(vec![1e-6; 3]),
                Expect::Reject("atol_wrong_shape"),
            ),
            (
                "valid",
                ToleranceValue::Scalar(1e-3),
                ToleranceValue::Scalar(1e-6),
                Expect::Accept,
            ),
        ] {
            cases.push(case(
                "integrate::validate_tol_with_audit",
                class,
                label,
                expect,
                move |l| {
                    done(validate_tol_with_audit(
                        rtol.clone(),
                        atol.clone(),
                        2,
                        mode,
                        Some(l),
                    ))
                },
            ));
        }
    }
    for (class, first_step, expect) in [
        (
            "non_finite",
            f64::NAN,
            Expect::Reject("first_step_must_be_finite"),
        ),
        ("zero", 0.0, Expect::Reject("first_step_must_be_positive")),
        ("too_long", 5.0, Expect::Reject("first_step_exceeds_bounds")),
        ("valid", 0.1, Expect::Accept),
    ] {
        cases.push(case(
            "integrate::validate_first_step_with_audit",
            class,
            "-",
            expect,
            move |l| {
                done(validate_first_step_with_audit(
                    first_step,
                    0.0,
                    1.0,
                    Some(l),
                ))
            },
        ));
    }
    for (class, max_step, expect) in [
        ("nan", f64::NAN, Expect::Reject("max_step_must_not_be_nan")),
        (
            "negative",
            -1.0,
            Expect::Reject("max_step_must_be_positive"),
        ),
        ("valid", 0.5, Expect::Accept),
    ] {
        cases.push(case(
            "integrate::validate_max_step_with_audit",
            class,
            "-",
            expect,
            move |l| done(validate_max_step_with_audit(max_step, Some(l))),
        ));
    }
}

fn sparse_cases(cases: &mut Vec<Case>) {
    use fsci_sparse::{
        CaspIterativeSolveOptions, CooMatrix, CsrMatrix, FormatConvertible, IterativeSolveOptions,
        Shape2D, casp_iterative_solve_with_audit, spsolve_with_audit,
    };
    fn csr(rows: usize, cols: usize, triplets: &[(usize, usize, f64)]) -> CsrMatrix {
        let (r, c, d): (Vec<_>, Vec<_>, Vec<_>) = triplets.iter().copied().fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |(mut rs, mut cs, mut ds), (r, c, v)| {
                rs.push(r);
                cs.push(c);
                ds.push(v);
                (rs, cs, ds)
            },
        );
        CooMatrix::from_triplets(Shape2D::new(rows, cols), d, r, c, false)
            .expect("coo")
            .to_csr()
            .expect("csr")
    }
    fn spd() -> CsrMatrix {
        csr(2, 2, &[(0, 0, 4.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)])
    }
    for (mode, label) in MODES {
        let options = fsci_sparse::SolveOptions {
            mode,
            ..fsci_sparse::SolveOptions::default()
        };
        let mut rows: Vec<(&str, CsrMatrix, Vec<f64>, Expect)> = vec![
            (
                "non_square",
                csr(2, 3, &[(0, 0, 1.0), (1, 1, 1.0)]),
                vec![1.0, 2.0],
                Expect::Reject("spsolve_with_casp::non_square"),
            ),
            (
                "shape_mismatch",
                spd(),
                vec![1.0, 2.0, 3.0],
                Expect::Reject("spsolve_with_casp::rhs_mismatch"),
            ),
            (
                "non_finite",
                spd(),
                vec![f64::NAN, 2.0],
                Expect::Reject("spsolve_with_casp::non_finite"),
            ),
            ("valid", spd(), vec![1.0, 2.0], Expect::Accept),
        ];
        rows.push(if mode == RuntimeMode::Hardened {
            (
                "empty_row",
                csr(2, 2, &[(0, 0, 1.0), (0, 1, 1.0)]),
                vec![1.0, 2.0],
                Expect::Reject("spsolve_with_casp::empty_structural_row"),
            )
        } else {
            // Strict returns an answer here, as SciPy's `spsolve` does (it warns rather than
            // raises on a singular matrix), so there is nothing to fail closed. Whether the
            // values match SciPy's is outside this test.
            (
                "singular",
                csr(2, 2, &[(0, 0, 1.0), (0, 1, 1.0)]),
                vec![1.0, 2.0],
                Expect::Accept,
            )
        });
        for (class, a, b, expect) in rows {
            cases.push(case(
                "sparse::spsolve_with_audit",
                class,
                label,
                expect,
                move |l| {
                    let mut portfolio = SparseSolverPortfolio::new(mode, 8);
                    done(spsolve_with_audit(&a, &b, options, &mut portfolio, l))
                },
            ));
        }

        let iterative = CaspIterativeSolveOptions {
            iterative: IterativeSolveOptions {
                mode,
                check_finite: true,
                ..IterativeSolveOptions::default()
            },
            ..CaspIterativeSolveOptions::default()
        };
        for (class, b, expect) in [
            (
                "shape_mismatch",
                vec![1.0, 2.0, 3.0],
                Expect::Reject("casp_iterative_solve::incompatible_shape"),
            ),
            (
                "non_finite",
                vec![f64::NAN, 2.0],
                Expect::Reject("casp_iterative_solve::non_finite_input"),
            ),
            ("valid", vec![1.0, 2.0], Expect::Accept),
        ] {
            cases.push(case(
                "sparse::casp_iterative_solve_with_audit",
                class,
                label,
                expect,
                move |l| {
                    done(casp_iterative_solve_with_audit(
                        &spd(),
                        &b,
                        None,
                        iterative,
                        l,
                    ))
                },
            ));
        }

        // Row 0 lists its columns out of order, so the matrix is not canonical.
        let unsorted = || {
            CsrMatrix::from_components(
                Shape2D::new(2, 2),
                vec![1.0, 2.0, 3.0],
                vec![1, 0, 1],
                vec![0, 2, 3],
                false,
            )
            .expect("unsorted csr")
        };
        let expect = if mode == RuntimeMode::Hardened {
            Expect::Reject("csr_to_csc::unsorted_indices")
        } else {
            Expect::Accept
        };
        cases.push(case(
            "sparse::csr_to_csc_with_mode_and_audit",
            "unsorted",
            label,
            expect,
            move |l| {
                done(fsci_sparse::ops::csr_to_csc_with_mode_and_audit(
                    &unsorted(),
                    mode,
                    "op",
                    l,
                ))
            },
        ));
        cases.push(case(
            "sparse::csr_to_csc_with_mode_and_audit",
            "valid",
            label,
            Expect::Accept,
            move |l| {
                done(fsci_sparse::ops::csr_to_csc_with_mode_and_audit(
                    &spd(),
                    mode,
                    "op",
                    l,
                ))
            },
        ));
    }
}

fn other_cases(cases: &mut Vec<Case>) {
    // fsci-stats: `try_fit_with_audit` is a trait method; it has no runtime mode.
    for (class, data, expect) in [
        ("empty", vec![], Expect::Reject("fit::insufficient_data")),
        (
            "non_finite",
            vec![1.0, f64::NAN, 3.0],
            Expect::Reject("fit::unsupported_data"),
        ),
        ("valid", vec![1.0, 2.0, 4.0], Expect::Accept),
    ] {
        cases.push(case(
            "stats::ContinuousDistribution::try_fit_with_audit",
            class,
            "-",
            expect,
            move |l| {
                use fsci_stats::ContinuousDistribution;
                done(fsci_stats::Normal::try_fit_with_audit(&data, l))
            },
        ));
    }

    let points = || {
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![5.0, 5.0],
        ]
    };
    for (mode, label) in MODES {
        // fsci-cluster
        for (class, data, k, expect) in [
            ("empty", vec![], 1, Expect::Reject("empty_data")),
            (
                "non_finite",
                vec![vec![0.0, f64::NAN], vec![1.0, 0.0]],
                1,
                Expect::Reject("non_finite_input"),
            ),
            ("bad_k", points(), 0, Expect::Reject("invalid_argument")),
            ("valid", points(), 2, Expect::Accept),
        ] {
            cases.push(case(
                "cluster::kmeans_with_mode",
                class,
                label,
                expect,
                move |l| {
                    done(fsci_cluster::kmeans_with_mode(
                        &data,
                        k,
                        50,
                        7,
                        mode,
                        Some(l),
                    ))
                },
            ));
        }

        // fsci-spatial
        for (class, data, expect) in [
            ("empty", vec![], Expect::Reject("empty_data")),
            (
                "ragged",
                vec![vec![0.0, 0.0], vec![1.0]],
                Expect::Reject("dimension_mismatch"),
            ),
            (
                "non_finite",
                vec![vec![0.0, f64::INFINITY], vec![1.0, 0.0]],
                Expect::Reject("non_finite_input"),
            ),
            ("valid", points(), Expect::Accept),
        ] {
            cases.push(case(
                "spatial::KDTree::new_with_mode",
                class,
                label,
                expect,
                move |l| done(fsci_spatial::KDTree::new_with_mode(&data, mode, Some(l))),
            ));
        }

        // fsci-signal: an empty signal is classified as an invalid input length.
        for (class, x, expect) in [
            ("empty", vec![], Expect::Reject("invalid_input_length")),
            (
                "non_finite",
                vec![1.0, f64::NAN, 3.0],
                Expect::Reject("non_finite_input"),
            ),
            ("valid", vec![1.0, 2.0, 3.0], Expect::Accept),
        ] {
            cases.push(case(
                "signal::czt_with_mode_and_audit",
                class,
                label,
                expect,
                move |l| {
                    done(fsci_signal::czt_with_mode_and_audit(
                        &x,
                        3,
                        None,
                        None,
                        mode,
                        Some(l),
                    ))
                },
            ));
        }

        // fsci-interpolate
        let interp_options = fsci_interpolate::Interp1dOptions {
            mode,
            ..fsci_interpolate::Interp1dOptions::default()
        };
        let non_finite_y = if mode == RuntimeMode::Hardened {
            Expect::Reject("non_finite_input")
        } else {
            Expect::Accept
        };
        for (class, x, y, expect) in [
            (
                "length_mismatch",
                vec![0.0, 1.0, 2.0],
                vec![0.0, 1.0],
                Expect::Reject("length_mismatch"),
            ),
            (
                "too_few",
                vec![0.0],
                vec![0.0],
                Expect::Reject("too_few_points"),
            ),
            (
                "non_finite_x",
                vec![0.0, f64::NAN, 2.0],
                vec![0.0, 1.0, 2.0],
                Expect::Reject("non_finite_x"),
            ),
            (
                "unsorted",
                vec![0.0, 2.0, 1.0],
                vec![0.0, 1.0, 2.0],
                Expect::Reject("unsorted_x"),
            ),
            (
                "non_finite_y",
                vec![0.0, 1.0, 2.0],
                vec![0.0, f64::NAN, 2.0],
                non_finite_y,
            ),
            (
                "valid",
                vec![0.0, 1.0, 2.0],
                vec![0.0, 1.0, 4.0],
                Expect::Accept,
            ),
        ] {
            cases.push(case(
                "interpolate::Interp1d::new_with_audit",
                class,
                label,
                expect,
                move |l| {
                    done(fsci_interpolate::Interp1d::new_with_audit(
                        &x,
                        &y,
                        interp_options,
                        Some(l),
                    ))
                },
            ));
        }

        // fsci-io
        for (class, content, expect) in [
            (
                "bad_header",
                "not a matrix market file\n",
                Expect::Reject("invalid_format"),
            ),
            (
                "symmetric_non_square",
                "%%MatrixMarket matrix coordinate real symmetric\n2 3 1\n1 1 3.0\n",
                Expect::Reject("invalid_format"),
            ),
            (
                "valid",
                "%%MatrixMarket matrix coordinate real general\n2 2 1\n1 1 3.0\n",
                Expect::Accept,
            ),
        ] {
            cases.push(case(
                "io::mmread_with_mode",
                class,
                label,
                expect,
                move |l| done(fsci_io::mmread_with_mode(content, mode, Some(l))),
            ));
        }

        // fsci-ndimage
        let image = || {
            fsci_ndimage::NdArray::new((0..9).map(f64::from).collect(), vec![3, 3]).expect("image")
        };
        let non_finite_sigma = if mode == RuntimeMode::Hardened {
            Expect::Reject("non_finite_input")
        } else {
            Expect::Reject("invalid_argument")
        };
        for (class, sigma, expect) in [
            ("negative_sigma", -1.0, Expect::Reject("invalid_argument")),
            ("non_finite_sigma", f64::NAN, non_finite_sigma),
            ("valid", 1.0, Expect::Accept),
        ] {
            cases.push(case(
                "ndimage::gaussian_filter_with_mode",
                class,
                label,
                expect,
                move |l| {
                    done(fsci_ndimage::gaussian_filter_with_mode(
                        &image(),
                        sigma,
                        fsci_ndimage::BoundaryMode::Reflect,
                        0.0,
                        mode,
                        Some(l),
                    ))
                },
            ));
        }

        // fsci-opt
        for (class, x0, expect) in [
            (
                "empty",
                vec![],
                Expect::Reject("minimize::invalid_argument"),
            ),
            ("valid", vec![1.0, -1.0], Expect::Accept),
        ] {
            cases.push(case(
                "opt::minimize_with_audit",
                class,
                label,
                expect,
                move |l| {
                    let options = fsci_opt::MinimizeOptions {
                        mode,
                        ..fsci_opt::MinimizeOptions::default()
                    };
                    let objective =
                        |x: &[f64]| x.iter().map(|v| (v - 0.5) * (v - 0.5)).sum::<f64>();
                    done(fsci_opt::minimize_with_audit(objective, &x0, options, l))
                },
            ));
        }

        // fsci-special: Strict follows SciPy (a negative-integer pole is NaN) and does not
        // reject; Hardened rejects such a pole, its one failure mode. NaN in is NaN out in both.
        for (class, z, hardened_reason) in [
            ("pole", -2.0, "gamma::PoleInput"),
            ("other_pole", -3.0, "gamma::PoleInput"),
            ("nan", f64::NAN, ""),
            ("valid", 2.5, ""),
        ] {
            let expect = if mode == RuntimeMode::Hardened && !hardened_reason.is_empty() {
                Expect::Reject(hardened_reason)
            } else {
                Expect::Accept
            };
            cases.push(case(
                "special::gamma_with_audit",
                class,
                label,
                expect,
                move |l| {
                    done(fsci_special::gamma_with_audit(
                        &fsci_special::SpecialTensor::RealScalar(z),
                        mode,
                        l,
                    ))
                },
            ));
        }
    }
}

#[test]
fn every_audited_error_is_one_fail_closed_event() {
    let mut cases = Vec::new();
    linalg_cases(&mut cases);
    fft_cases(&mut cases);
    arrayapi_cases(&mut cases);
    integrate_cases(&mut cases);
    sparse_cases(&mut cases);
    other_cases(&mut cases);

    let mut failures = Vec::new();
    let mut compared = 0;
    let mut rejected = 0;
    // API → fingerprints of its rejected calls, to check that different inputs differ.
    let mut fingerprints: BTreeMap<&str, Vec<(String, String)>> = BTreeMap::new();
    for case in &cases {
        let (result, events) = run(case);
        compared += 1;
        let label = format!("{} [{} / {}]", case.api, case.class, case.mode);
        let fail_closed: Vec<&AuditEvent> = events
            .iter()
            .filter(|event| matches!(event.action, AuditAction::FailClosed { .. }))
            .collect();
        match (case.expect, &result) {
            (Expect::Reject(reason), Err(error)) => {
                rejected += 1;
                let last_is_fail_closed = events
                    .last()
                    .is_some_and(|event| matches!(event.action, AuditAction::FailClosed { .. }));
                if fail_closed.len() != 1 || !last_is_fail_closed {
                    failures.push(format!(
                        "{label}: returned Err({error}) and recorded {} FailClosed \
                         (last event FailClosed: {last_is_fail_closed}); events: [{}]",
                        fail_closed.len(),
                        describe(&events)
                    ));
                    continue;
                }
                let event = fail_closed[0];
                if event.action
                    != (AuditAction::FailClosed {
                        reason: reason.to_string(),
                    })
                {
                    failures.push(format!(
                        "{label}: returned Err({error}); reason {:?}, expected {reason:?}",
                        event.action
                    ));
                }
                if !is_fingerprint(&event.input_fingerprint) {
                    failures.push(format!(
                        "{label}: fingerprint {:?} is not blake3:<64 hex>",
                        event.input_fingerprint
                    ));
                }
                let (_, again) = run(case);
                let again = again
                    .iter()
                    .rev()
                    .find(|event| matches!(event.action, AuditAction::FailClosed { .. }))
                    .map(|event| event.input_fingerprint.clone());
                if again.as_deref() != Some(event.input_fingerprint.as_str()) {
                    failures.push(format!(
                        "{label}: the same call recorded fingerprint {again:?}, first {:?}",
                        event.input_fingerprint
                    ));
                }
                fingerprints
                    .entry(case.api)
                    .or_default()
                    .push((label.clone(), event.input_fingerprint.clone()));
            }
            (Expect::Reject(reason), Ok(())) => failures.push(format!(
                "{label}: expected Err with FailClosed {reason:?}, returned Ok; events: [{}]",
                describe(&events)
            )),
            (Expect::Accept, Ok(())) => {
                if !fail_closed.is_empty() {
                    failures.push(format!(
                        "{label}: returned Ok but recorded FailClosed; events: [{}]",
                        describe(&events)
                    ));
                }
            }
            (Expect::Accept, Err(error)) => failures.push(format!(
                "{label}: expected Ok, returned Err({error}); events: [{}]",
                describe(&events)
            )),
        }
    }
    for (api, recorded) in &fingerprints {
        for (i, (label_a, a)) in recorded.iter().enumerate() {
            for (label_b, b) in &recorded[i + 1..] {
                if a == b {
                    failures.push(format!(
                        "{api}: {label_a} and {label_b} share fingerprint {a}"
                    ));
                }
            }
        }
    }

    let apis: std::collections::BTreeSet<&str> = cases.iter().map(|case| case.api).collect();
    eprintln!(
        "audit fail-closed property: {compared} cases over {} APIs, {rejected} rejected, {} \
         failures",
        apis.len(),
        failures.len()
    );
    for failure in &failures {
        eprintln!("  FAIL {failure}");
    }
    assert_eq!(compared, cases.len(), "every case in the table was run");
    assert_eq!(
        apis.len(),
        46,
        "the inventory has 46 audited APIs: {apis:?}"
    );
    assert!(
        failures.is_empty(),
        "{} failures (listed above)",
        failures.len()
    );
}
