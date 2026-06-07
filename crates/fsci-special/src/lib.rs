#![forbid(unsafe_code)]

pub mod airy;
pub mod audit;
pub mod bessel;
pub mod beta;
pub mod convenience;
pub mod elliptic;
pub mod error;
pub mod gamma;
pub mod hyper;
pub mod orthopoly;
pub mod types;

pub use airy::{
    AIRY_DISPATCH_PLAN, AIRYE_DISPATCH_PLAN, AiryResult, ai, ai_zeros, airy, airye, bi, bi_zeros,
};
pub use audit::{
    SyncSharedAuditLedger, record_bounded_recovery, record_fail_closed, sync_audit_ledger,
};
pub use bessel::{
    BESSEL_DISPATCH_PLAN, h1vp, h2vp, hankel1, hankel1e, hankel2, hankel2e, i0, i0_scalar, i0e,
    i0e_scalar, i1, i1_scalar, i1e, i1e_scalar, iv, ive, ive_scalar, ivp, j0, j1, jn, jn_zeros,
    jnp_zeros, jv, jve, jvp, k0, k0_scalar, k0e, k0e_scalar, k1, k1_scalar, k1e, k1e_scalar, kn,
    kn_scalar, kv, kve, kve_scalar, kvp, log_wright_bessel, riccati_jn, riccati_yn, spherical_in,
    spherical_jn, spherical_kn, spherical_yn, wright_bessel, y0, y0_zeros, y1, y1_zeros, y1p_zeros,
    yn, yn_zeros, ynp_zeros, yv, yve, yvp,
};
pub use beta::{
    BETA_DISPATCH_PLAN, bdtr, bdtrc, bdtri, bdtrik, bdtrin, beta, betainc, betainc_scalar,
    betaincc, betaincc_scalar, betainccinv, betainccinv_scalar, betaln, betaln_scalar, btdtr,
    btdtrc, btdtri, btdtria, btdtrib, fdtr, fdtrc, fdtri, fdtridfd, nbdtr, nbdtrc, nbdtri, nbdtrik,
    nbdtrin, stdtr, stdtrc, stdtridf, stdtrit,
};
pub use convenience::{
    CONVENIENCE_DISPATCH_PLAN,
    // Numerical differentiation and AGM
    agm,
    arccosh,
    arcsinh,
    arctanh,
    bei,
    beip,
    ber,
    bernoulli,
    berp,
    // Convenience wrappers
    betaincinv,
    betaincinv_scalar,
    binary_cross_entropy,
    boxcox,
    boxcox_scalar,
    boxcox_transform,
    boxcox_transform_scalar,
    boxcox1p,
    boxcox1p_scalar,
    cauchit,
    cauchit_inv,
    cbrt,
    ceil,
    celu,
    central_diff,
    central_diff2,
    clausen,
    clip,
    cloglog,
    cloglog_inv,
    copysign,
    cosdg,
    cosm1,
    cosm1_scalar,
    cospi,
    cotdg,
    dawsn,
    dawsn_scalar,
    debye,
    deg2rad,
    digamma_scalar,
    diric,
    divmod,
    elu,
    entr,
    erfc_conv,
    erfcinv_conv,
    erfcx,
    erfi,
    euler,
    exp2,
    exp10,
    expi_scalar,
    expit,
    expm1,
    expn,
    exprel,
    fabs,
    fdiff,
    fix,
    floor,
    fmax,
    fmin,
    fresnel,
    frexp,
    gamma_mod_squared,
    gammainc_conv,
    gammaincc_conv,
    gammainccinv,
    gammainccinv_scalar,
    gammaincinv,
    gammaincinv_scalar,
    gelu,
    gradient_approx,
    hard_sigmoid,
    hard_swish,
    hard_swish_scalar,
    hard_tanh,
    hardshrink,
    heaviside,
    hessian_approx,
    huber,
    hurwitz_zeta,
    hypot,
    inv_boxcox,
    inv_boxcox_scalar,
    inv_boxcox1p,
    inv_boxcox1p_scalar,
    isfinite,
    isinf,
    isnan,
    isneginf,
    isposinf,
    it2struve0,
    itmodstruve0,
    itstruve0,
    jacobian_approx,
    kei,
    keip,
    kelvin,
    ker,
    kerp,
    kl_div,
    kolmogi,
    kolmogorov,
    lambertw_scalar,
    ldexp,
    leaky_relu,
    log_comb,
    log_cosh,
    log_expit,
    log_expit_scalar,
    log_ndtr,
    log_ndtr_scalar,
    log_softmax,
    log1mexp,
    log1p,
    log1pexp,
    log1pmx_scalar,
    log2,
    log10,
    logaddexp,
    logaddexp2,
    logit,
    loglog,
    loglog_inv,
    logsigmoid,
    logsumexp,
    logsumexp_axis_2d,
    logsumexp_axis_2d_with_b,
    logsumexp_with_b,
    maximum,
    minimum,
    mish,
    modf,
    modstruve,
    modstruve_scalar,
    nan_to_num,
    ndtr,
    ndtr_scalar,
    ndtri,
    ndtri_exp,
    ndtri_exp_scalar,
    ndtri_scalar,
    negative,
    negentropy,
    nextafter,
    nrdtrimn,
    nrdtrisd,
    owens_t,
    owens_t_scalar,
    pentagamma,
    poch,
    positive,
    power,
    powm1,
    powm1_scalar,
    pseudo_huber,
    rad2deg,
    radian,
    reciprocal,
    rel_entr,
    relu,
    rint,
    round,
    selu,
    shichi,
    sici,
    sign,
    signbit,
    silu,
    sinc,
    sinc_squared,
    sindg,
    sinpi,
    smirnov,
    smirnovi,
    softmax,
    softplus,
    softshrink,
    softsign,
    spacing,
    spence,
    spence_scalar,
    square,
    stirling2,
    stirling2_scalar,
    struve,
    struve_scalar,
    swish,
    tandg,
    tanhshrink,
    tetragamma,
    threshold,
    tklmbda,
    trigamma,
    trunc,
    voigt_profile,
    voigt_profile_real_gamma_zero,
    wofz,
    wofz_real,
    wofz_scalar,
    wrightomega,
    wrightomega_scalar,
    xlog1py,
    xlogx,
    xlogy,
    zeta_scalar,
};
pub use elliptic::{
    ELLIPTIC_DISPATCH_PLAN, ellipe, ellipeinc, ellipj, ellipk, ellipkinc, ellipkm1, elliprc,
    elliprd, elliprf, elliprg, elliprj, exp1, expi, expn_scalar, lambertw,
};
pub use error::{
    ERROR_DISPATCH_PLAN, erf, erf_scalar, erfc, erfc_scalar, erfcinv, erfinv, erfinv_scalar,
};
pub use gamma::{
    GAMMA_DISPATCH_PLAN, binom, chdtr, chdtrc, chdtri, chdtriv, comb, digamma, factorial,
    factorial2, factorialk, gamma, gamma_with_audit, gammainc, gammainc_scalar, gammaincc,
    gammaincc_scalar, gammaln, gammaln_scalar, gammasgn, gammasgn_scalar, gdtr, gdtrc, gdtria,
    gdtrib, gdtrix, loggamma, loggamma_scalar, multigammaln, pdtr, pdtrc, pdtri, pdtrik, perm,
    polygamma, psi, rgamma, zeta, zetac,
};
pub use hyper::{
    HYPER_DISPATCH_PLAN, HyperCaspDecision, HyperCaspProblem, HypergeometricBranch,
    HypergeometricFunction, hyp0f1, hyp0f1_scalar, hyp1f1, hyp2f1, hyperu, hyperu_scalar,
    select_hypergeometric_branch,
};
pub use orthopoly::{
    assoc_laguerre, c_roots, cg_roots, eval_chebyc, eval_chebys, eval_chebyt, eval_chebyu,
    eval_gegenbauer, eval_genlaguerre, eval_hermite, eval_hermitenorm, eval_jacobi, eval_laguerre,
    eval_legendre, eval_sh_chebyt, eval_sh_chebyu, eval_sh_jacobi, eval_sh_legendre, h_roots,
    he_roots, j_roots, js_roots, l_roots, la_roots, lpmn, lpmv, lpn, lqmn, lqn, p_roots, ps_roots,
    roots_chebyc, roots_chebys, roots_chebyt, roots_chebyu, roots_gegenbauer, roots_genlaguerre,
    roots_hermite, roots_hermitenorm, roots_jacobi, roots_laguerre, roots_legendre,
    roots_sh_chebyt, roots_sh_chebyu, roots_sh_jacobi, roots_sh_legendre, s_roots, sph_harm,
    sph_harm_y, t_roots, ts_roots, u_roots, us_roots,
};
pub use types::{
    Complex64, DispatchPlan, DispatchStep, KernelRegime, SpecialError, SpecialErrorKind,
    SpecialResult, SpecialTensor, SpecialTraceEntry, take_special_traces,
};

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, OnceLock};

    use fsci_runtime::RuntimeMode;

    use super::*;

    #[test]
    fn scalar_kernels_follow_primary_contract_points() {
        let one = SpecialTensor::RealScalar(1.0);
        let two = SpecialTensor::RealScalar(2.0);
        let zero = SpecialTensor::RealScalar(0.0);

        let gamma_one = gamma(&one, RuntimeMode::Strict).expect("gamma(1) should evaluate");
        assert_real_scalar_close(gamma_one, 1.0, 1e-12);

        let beta_half = beta(&one, &two, RuntimeMode::Strict).expect("beta(1,2) should evaluate");
        assert_real_scalar_close(beta_half, 0.5, 1e-12);

        let erf_zero = erf(&zero, RuntimeMode::Strict).expect("erf(0) should evaluate");
        assert_real_scalar_close(erf_zero, 0.0, 1e-12);

        let erfc_zero = erfc(&zero, RuntimeMode::Strict).expect("erfc(0) should evaluate");
        assert_real_scalar_close(erfc_zero, 1.0, 1e-12);

        let inv_zero = erfinv(&zero, RuntimeMode::Strict).expect("erfinv(0) should evaluate");
        assert_real_scalar_close(inv_zero, 0.0, 1e-12);

        let gammainc_one =
            gammainc(&one, &one, RuntimeMode::Strict).expect("gammainc(1,1) should evaluate");
        assert_real_scalar_close(gammainc_one, 1.0 - (-1.0_f64).exp(), 1e-10);

        let gammaincc_one =
            gammaincc(&one, &one, RuntimeMode::Strict).expect("gammaincc(1,1) should evaluate");
        assert_real_scalar_close(gammaincc_one, (-1.0_f64).exp(), 1e-10);

        let j0_zero = j0(&zero, RuntimeMode::Strict).expect("j0(0) should evaluate");
        assert_real_scalar_close(j0_zero, 1.0, 1e-8);

        let j1_zero = j1(&zero, RuntimeMode::Strict).expect("j1(0) should evaluate");
        assert_real_scalar_close(j1_zero, 0.0, 1e-8);
    }

    #[test]
    fn root_namespace_exports_orthopoly_extras() {
        let assoc = assoc_laguerre(0.5, 3, 1.25);
        let assoc_via = eval_genlaguerre(3, 1.25, 0.5);
        assert!((assoc - assoc_via).abs() < 1e-12);

        assert!((eval_chebyc(3, 1.5) - (1.5_f64.powi(3) - 3.0 * 1.5)).abs() < 1e-12);
        assert!((eval_chebys(3, 1.5) - (1.5_f64.powi(3) - 2.0 * 1.5)).abs() < 1e-12);

        let shifted = eval_sh_jacobi(4, 0.5, 1.25, 0.3);
        let via = eval_jacobi(4, -0.75, 0.25, -0.4);
        assert!((shifted - via).abs() < 1e-12);

        let (shifted_nodes, shifted_weights) = roots_sh_jacobi(3, 0.5, 1.25);
        let (jacobi_nodes, jacobi_weights) = roots_jacobi(3, -0.75, 0.25);
        let scale = 2.0_f64.powf(-0.5);
        for i in 0..3 {
            assert!((shifted_nodes[i] - 0.5 * (1.0 + jacobi_nodes[i])).abs() < 1e-12);
            assert!((shifted_weights[i] - scale * jacobi_weights[i]).abs() < 1e-12);
        }

        let (p_nodes, p_weights) = p_roots(3);
        let (leg_nodes, leg_weights) = roots_legendre(3);
        let (js_nodes, js_weights) = js_roots(3, 0.5, 1.25);
        assert_eq!(p_nodes, leg_nodes);
        assert_eq!(p_weights, leg_weights);
        assert_eq!(js_nodes, shifted_nodes);
        assert_eq!(js_weights, shifted_weights);
    }

    #[test]
    fn erf_infinity_regression() {
        use fsci_runtime::RuntimeMode;
        let inf = SpecialTensor::RealScalar(f64::INFINITY);
        let neg_inf = SpecialTensor::RealScalar(f64::NEG_INFINITY);

        let res_inf = erf(&inf, RuntimeMode::Strict).unwrap();
        let res_neg_inf = erf(&neg_inf, RuntimeMode::Strict).unwrap();

        assert_eq!(
            expect_real_scalar(res_inf, "erf(inf)"),
            1.0,
            "erf(inf) should be 1.0"
        );
        assert_eq!(
            expect_real_scalar(res_neg_inf, "erf(-inf)"),
            -1.0,
            "erf(-inf) should be -1.0"
        );
    }

    #[test]
    fn iv_large_argument_regression() {
        use fsci_runtime::RuntimeMode;
        // I_0.5(500) = sqrt(2/(pi*500)) * sinh(500) ≈ 1.13e215
        let v = SpecialTensor::RealScalar(0.5);
        let z = SpecialTensor::RealScalar(500.0);
        let result = iv(&v, &z, RuntimeMode::Strict).expect("iv(0.5, 500)");
        let val = expect_real_scalar(result, "iv(0.5, 500)");
        assert!(
            val > 1e214 && val < 1e216,
            "iv(0.5, 500) should be ~1e215, got {val}"
        );
    }

    #[test]
    fn hardened_mode_rejects_domain_violations() {
        let out_of_domain = SpecialTensor::RealScalar(1.1);
        let pole = SpecialTensor::RealScalar(-2.0);

        let strict_inv = erfinv(&out_of_domain, RuntimeMode::Strict).expect("strict returns NaN");
        assert_real_scalar_nan(strict_inv);

        let hardened_inv =
            erfinv(&out_of_domain, RuntimeMode::Hardened).expect_err("hardened rejects domain");
        assert_eq!(hardened_inv.kind, SpecialErrorKind::DomainError);

        let strict_gamma = gamma(&pole, RuntimeMode::Strict).expect("strict returns NaN");
        assert_real_scalar_nan(strict_gamma);

        let hardened_gamma =
            gamma(&pole, RuntimeMode::Hardened).expect_err("hardened rejects pole");
        assert_eq!(hardened_gamma.kind, SpecialErrorKind::PoleInput);
    }

    #[test]
    fn trace_log_captures_domain_policy_events_as_json_lines() {
        let _guard = trace_test_guard();
        let _ = take_special_traces();

        let out_of_domain = SpecialTensor::RealScalar(1.1);
        let _ = erfinv(&out_of_domain, RuntimeMode::Strict).expect("strict returns NaN");
        let _ = erfinv(&out_of_domain, RuntimeMode::Hardened).expect_err("hardened rejects");

        let traces = take_special_traces();
        assert!(!traces.is_empty(), "expected at least one trace entry");
        assert!(traces.iter().any(|entry| {
            entry.function == "erfinv"
                && entry.category == "domain_error"
                && entry.action_taken == "returned_nan"
        }));
        assert!(traces.iter().any(|entry| {
            entry.function == "erfinv"
                && entry.category == "domain_error"
                && entry.action_taken == "fail_closed"
        }));

        let first = traces.first().expect("trace entries exist");
        let json = first.to_json_line();
        assert!(json.starts_with('{'));
        assert!(json.contains("\"function\""));
        assert!(json.contains("\"action_taken\""));
        assert!(json.contains("\"category\""));
    }

    #[test]
    fn beta_overflow_path_diverges_by_runtime_mode() {
        let _guard = trace_test_guard();
        let _ = take_special_traces();

        let tiny = SpecialTensor::RealScalar(1.0e-308);
        let strict = beta(&tiny, &tiny, RuntimeMode::Strict).expect("strict must evaluate");
        let value = expect_real_scalar(strict, "beta tiny strict");
        assert!(
            value.is_infinite() || value > 1.0e308,
            "expected very large or infinite strict beta, got {value}"
        );

        let hardened = beta(&tiny, &tiny, RuntimeMode::Hardened).expect_err("hardened rejects");
        assert_eq!(hardened.kind, SpecialErrorKind::OverflowRisk);

        let traces = take_special_traces();
        assert!(traces.iter().any(|entry| {
            entry.function == "beta"
                && entry.category == "overflow_risk"
                && entry.action_taken == "returned_inf"
        }));
        assert!(traces.iter().any(|entry| {
            entry.function == "beta"
                && entry.category == "overflow_risk"
                && entry.action_taken == "fail_closed"
        }));
    }

    #[test]
    fn erfcinv_domain_policy_diverges_by_runtime_mode() {
        let _guard = trace_test_guard();
        let _ = take_special_traces();
        let out_of_domain = SpecialTensor::RealScalar(3.0);

        let strict = erfcinv(&out_of_domain, RuntimeMode::Strict).expect("strict returns NaN");
        assert_real_scalar_nan(strict);

        let hardened =
            erfcinv(&out_of_domain, RuntimeMode::Hardened).expect_err("hardened rejects");
        assert_eq!(hardened.kind, SpecialErrorKind::DomainError);

        let traces = take_special_traces();
        assert!(traces.iter().any(|entry| {
            entry.function == "erfcinv"
                && entry.category == "domain_error"
                && entry.action_taken == "returned_nan"
        }));
        assert!(traces.iter().any(|entry| {
            entry.function == "erfcinv"
                && entry.category == "domain_error"
                && entry.action_taken == "fail_closed"
        }));
    }

    #[test]
    fn betaln_and_betainc_domain_policy_diverge_by_runtime_mode() {
        let _guard = trace_test_guard();
        let _ = take_special_traces();
        // -1.5 is a negative NON-integer: a valid betaln input that SciPy
        // evaluates finitely (= gammaln(-1.5)+gammaln(1)-gammaln(-0.5)).
        let neg = SpecialTensor::RealScalar(-1.5);
        let pos = SpecialTensor::RealScalar(1.0);
        let x_bad = SpecialTensor::RealScalar(2.0);

        // Strict now COMPUTES the value (matches scipy.special.betaln(-1.5, 1.0));
        // Hardened still conservatively fail-closes on a nonpositive parameter.
        let strict_betaln = betaln(&neg, &pos, RuntimeMode::Strict).expect("strict computes value");
        assert_real_scalar_close(strict_betaln, -0.4054651081081643, 1e-12);
        let hardened_betaln =
            betaln(&neg, &pos, RuntimeMode::Hardened).expect_err("hardened rejects");
        assert_eq!(hardened_betaln.kind, SpecialErrorKind::DomainError);

        let strict_betainc =
            betainc(&pos, &pos, &x_bad, RuntimeMode::Strict).expect("strict returns NaN");
        assert_real_scalar_nan(strict_betainc);
        let hardened_betainc =
            betainc(&pos, &pos, &x_bad, RuntimeMode::Hardened).expect_err("hardened rejects");
        assert_eq!(hardened_betainc.kind, SpecialErrorKind::DomainError);

        let traces = take_special_traces();
        assert!(traces.iter().any(|entry| {
            entry.function == "betaln"
                && entry.category == "domain_error"
                && entry.action_taken == "fail_closed"
        }));
        assert!(traces.iter().any(|entry| {
            entry.function == "betainc"
                && entry.category == "domain_error"
                && entry.action_taken == "returned_nan"
        }));
        assert!(traces.iter().any(|entry| {
            entry.function == "betainc"
                && entry.category == "domain_error"
                && entry.action_taken == "fail_closed"
        }));
    }

    #[test]
    fn gammainc_domain_policy_diverges_by_runtime_mode() {
        let _guard = trace_test_guard();
        let _ = take_special_traces();
        let invalid_a = SpecialTensor::RealScalar(0.0);
        let one = SpecialTensor::RealScalar(1.0);

        let strict = gammainc(&invalid_a, &one, RuntimeMode::Strict).expect("strict returns NaN");
        assert_real_scalar_nan(strict);

        let hardened =
            gammainc(&invalid_a, &one, RuntimeMode::Hardened).expect_err("hardened rejects");
        assert_eq!(hardened.kind, SpecialErrorKind::DomainError);

        let traces = take_special_traces();
        assert!(traces.iter().any(|entry| {
            entry.function == "gammainc"
                && entry.category == "domain_error"
                && entry.action_taken == "returned_nan"
        }));
        assert!(traces.iter().any(|entry| {
            entry.function == "gammainc"
                && entry.category == "domain_error"
                && entry.action_taken == "fail_closed"
        }));
    }

    #[test]
    fn bessel_subset_wrappers_execute_required_surface() {
        let _guard = trace_test_guard();
        let _ = take_special_traces();
        let zero = SpecialTensor::RealScalar(0.0);
        let one = SpecialTensor::RealScalar(1.0);
        let three = SpecialTensor::RealScalar(3.0);
        let neg_one = SpecialTensor::RealScalar(-1.0);

        let y0_zero = y0(&zero, RuntimeMode::Strict).expect("strict y0(0) should diverge");
        let value = expect_real_scalar(y0_zero, "y0(0)");
        assert!(value.is_infinite() && value.is_sign_negative());

        let y1_zero = y1(&zero, RuntimeMode::Strict).expect("strict y1(0) should diverge");
        let value = expect_real_scalar(y1_zero, "y1(0)");
        assert!(value.is_infinite() && value.is_sign_negative());

        let jn_zero = jn(&zero, &one, RuntimeMode::Strict).expect("jn(0,x)=j0(x)");
        let j0_one = j0(&one, RuntimeMode::Strict).expect("j0(1)");
        assert_real_scalars_close(jn_zero, j0_one, 1e-12);

        let yn_one = yn(&one, &three, RuntimeMode::Strict).expect("yn(1,x)=y1(x)");
        let y1_three = y1(&three, RuntimeMode::Strict).expect("y1(3)");
        assert_real_scalars_close(yn_one, y1_three, 1e-12);

        let strict_neg =
            y0(&neg_one, RuntimeMode::Strict).expect("strict returns NaN for negative domain");
        assert_real_scalar_nan(strict_neg);
        let hardened_neg = y0(&neg_one, RuntimeMode::Hardened).expect_err("hardened rejects");
        assert_eq!(hardened_neg.kind, SpecialErrorKind::DomainError);
    }

    #[test]
    fn error_function_unit_matrix_and_structured_logs() {
        let _guard = trace_test_guard();
        clear_test_logs();
        let seed = 17_005_u64;
        let mode = RuntimeMode::Strict;

        let zero = SpecialTensor::RealScalar(0.0);
        let inf = SpecialTensor::RealScalar(f64::INFINITY);
        let neg_inf = SpecialTensor::RealScalar(f64::NEG_INFINITY);

        let erf_zero = erf(&zero, mode).expect("erf(0)");
        assert_real_scalar_close(erf_zero.clone(), 0.0, 1e-12);
        push_test_log(test_log_json(
            "error-erf-zero",
            "erf",
            "x=0",
            "0",
            scalar_to_string(&erf_zero),
            1e-12,
            mode,
            seed,
            "pass",
        ));

        let erf_inf = erf(&inf, mode).expect("erf(inf)");
        assert_real_scalar_close(erf_inf.clone(), 1.0, 1e-12);
        push_test_log(test_log_json(
            "error-erf-inf",
            "erf",
            "x=inf",
            "1",
            scalar_to_string(&erf_inf),
            1e-12,
            mode,
            seed,
            "pass",
        ));

        let erf_neg_inf = erf(&neg_inf, mode).expect("erf(-inf)");
        assert_real_scalar_close(erf_neg_inf.clone(), -1.0, 1e-12);
        push_test_log(test_log_json(
            "error-erf-neg-inf",
            "erf",
            "x=-inf",
            "-1",
            scalar_to_string(&erf_neg_inf),
            1e-12,
            mode,
            seed,
            "pass",
        ));

        let erfc_zero = erfc(&zero, mode).expect("erfc(0)");
        assert_real_scalar_close(erfc_zero.clone(), 1.0, 1e-12);
        push_test_log(test_log_json(
            "error-erfc-zero",
            "erfc",
            "x=0",
            "1",
            scalar_to_string(&erfc_zero),
            1e-12,
            mode,
            seed,
            "pass",
        ));

        let erfc_inf = erfc(&inf, mode).expect("erfc(inf)");
        assert_real_scalar_close(erfc_inf.clone(), 0.0, 1e-12);
        push_test_log(test_log_json(
            "error-erfc-inf",
            "erfc",
            "x=inf",
            "0",
            scalar_to_string(&erfc_inf),
            1e-12,
            mode,
            seed,
            "pass",
        ));

        let erfc_neg_inf = erfc(&neg_inf, mode).expect("erfc(-inf)");
        assert_real_scalar_close(erfc_neg_inf.clone(), 2.0, 1e-12);
        push_test_log(test_log_json(
            "error-erfc-neg-inf",
            "erfc",
            "x=-inf",
            "2",
            scalar_to_string(&erfc_neg_inf),
            1e-12,
            mode,
            seed,
            "pass",
        ));

        for x in [-3.0, -1.0, -0.2, 0.0, 0.2, 1.0, 3.0] {
            let tensor = SpecialTensor::RealScalar(x);
            let erf_v = erf(&tensor, mode).expect("erf finite");
            let erfc_v = erfc(&tensor, mode).expect("erfc finite");
            let lhs = scalar_value(&erf_v) + scalar_value(&erfc_v);
            assert!(
                (lhs - 1.0).abs() <= 2.0e-7,
                "erf+erfc identity failed for x={x}: {lhs}"
            );
            push_test_log(test_log_json(
                "error-erf-plus-erfc",
                "erf_erfc_identity",
                format!("x={x}"),
                "1",
                format!("{lhs:.16e}"),
                2.0e-7,
                mode,
                seed,
                "pass",
            ));
        }

        let nan = SpecialTensor::RealScalar(f64::NAN);
        assert_real_scalar_nan(erf(&nan, mode).expect("erf NaN propagation"));
        assert_real_scalar_nan(erfc(&nan, mode).expect("erfc NaN propagation"));
        assert_real_scalar_nan(erfinv(&nan, mode).expect("erfinv NaN propagation"));
        assert_real_scalar_nan(erfcinv(&nan, mode).expect("erfcinv NaN propagation"));

        let logs = take_test_logs();
        assert!(!logs.is_empty(), "expected structured test logs");
        assert_logs_follow_schema(&logs);
    }

    #[test]
    fn gamma_beta_unit_matrix_and_structured_logs() {
        let _guard = trace_test_guard();
        clear_test_logs();
        let seed = 17_006_u64;
        let mode = RuntimeMode::Strict;

        for (n, expected) in [(1.0, 1.0), (2.0, 1.0), (3.0, 2.0), (4.0, 6.0), (5.0, 24.0)] {
            let n_tensor = SpecialTensor::RealScalar(n);
            let actual = gamma(&n_tensor, mode).expect("gamma integer point");
            assert_real_scalar_close(actual.clone(), expected, 1e-9);
            push_test_log(test_log_json(
                "gamma-integer-factorial",
                "gamma",
                format!("x={n}"),
                format!("{expected}"),
                scalar_to_string(&actual),
                1e-9,
                mode,
                seed,
                "pass",
            ));
        }

        let gammaln_one = gammaln(&SpecialTensor::RealScalar(1.0), mode).expect("gammaln(1)");
        assert_real_scalar_close(gammaln_one.clone(), 0.0, 1e-12);
        push_test_log(test_log_json(
            "gamma-gammaln-one",
            "gammaln",
            "x=1",
            "0",
            scalar_to_string(&gammaln_one),
            1e-12,
            mode,
            seed,
            "pass",
        ));

        let gammaln_five = gammaln(&SpecialTensor::RealScalar(5.0), mode).expect("gammaln(5)");
        assert_real_scalar_close(gammaln_five.clone(), 24.0_f64.ln(), 1e-9);
        push_test_log(test_log_json(
            "gamma-gammaln-five",
            "gammaln",
            "x=5",
            "ln(24)",
            scalar_to_string(&gammaln_five),
            1e-9,
            mode,
            seed,
            "pass",
        ));

        let a = SpecialTensor::RealScalar(2.0);
        let x0 = SpecialTensor::RealScalar(0.0);
        let x1 = SpecialTensor::RealScalar(1.0);
        let xinf = SpecialTensor::RealScalar(f64::INFINITY);

        assert_real_scalar_close(gammainc(&a, &x0, mode).expect("gammainc(a,0)"), 0.0, 1e-12);
        assert_real_scalar_close(
            gammaincc(&a, &x0, mode).expect("gammaincc(a,0)"),
            1.0,
            1e-12,
        );
        assert_real_scalar_close(
            gammainc(&a, &xinf, mode).expect("gammainc(a,inf)"),
            1.0,
            1e-12,
        );
        assert_real_scalar_close(
            gammaincc(&a, &xinf, mode).expect("gammaincc(a,inf)"),
            0.0,
            1e-12,
        );

        let p = gammainc(&a, &x1, mode).expect("gammainc");
        let q = gammaincc(&a, &x1, mode).expect("gammaincc");
        let complement = scalar_value(&p) + scalar_value(&q);
        assert!((complement - 1.0).abs() <= 2.0e-12);
        push_test_log(test_log_json(
            "gamma-complement",
            "gammainc+gammaincc",
            "a=2,x=1",
            "1",
            format!("{complement:.16e}"),
            2.0e-12,
            mode,
            seed,
            "pass",
        ));

        let beta_left = beta(
            &SpecialTensor::RealScalar(0.5),
            &SpecialTensor::RealScalar(2.0),
            mode,
        )
        .expect("beta left");
        let beta_right = beta(
            &SpecialTensor::RealScalar(2.0),
            &SpecialTensor::RealScalar(0.5),
            mode,
        )
        .expect("beta right");
        assert_real_scalars_close(beta_left.clone(), beta_right.clone(), 1e-10);
        push_test_log(test_log_json(
            "beta-symmetry",
            "beta",
            "a=0.5,b=2 vs a=2,b=0.5",
            "equal",
            format!(
                "lhs={},rhs={}",
                scalar_to_string(&beta_left),
                scalar_to_string(&beta_right)
            ),
            1e-10,
            mode,
            seed,
            "pass",
        ));

        assert_real_scalar_close(
            betainc(
                &SpecialTensor::RealScalar(2.5),
                &SpecialTensor::RealScalar(3.0),
                &SpecialTensor::RealScalar(0.0),
                mode,
            )
            .expect("betainc x=0"),
            0.0,
            1e-12,
        );
        assert_real_scalar_close(
            betainc(
                &SpecialTensor::RealScalar(2.5),
                &SpecialTensor::RealScalar(3.0),
                &SpecialTensor::RealScalar(1.0),
                mode,
            )
            .expect("betainc x=1"),
            1.0,
            1e-12,
        );

        let logs = take_test_logs();
        assert!(!logs.is_empty(), "expected structured test logs");
        assert_logs_follow_schema(&logs);
    }

    #[test]
    fn bessel_reference_points_and_structured_logs() {
        let _guard = trace_test_guard();
        clear_test_logs();
        let seed = 17_007_u64;
        let mode = RuntimeMode::Strict;

        let y0_one = y0(&SpecialTensor::RealScalar(1.0), mode).expect("y0(1)");
        assert_real_scalar_close(y0_one.clone(), 0.088_256_964_2, 7.5e-3);
        push_test_log(test_log_json(
            "bessel-y0-one",
            "y0",
            "x=1",
            "0.0882569642",
            scalar_to_string(&y0_one),
            7.5e-3,
            mode,
            seed,
            "pass",
        ));

        let y1_one = y1(&SpecialTensor::RealScalar(1.0), mode).expect("y1(1)");
        assert_real_scalar_close(y1_one.clone(), -0.781_212_821_3, 7.5e-3);
        push_test_log(test_log_json(
            "bessel-y1-one",
            "y1",
            "x=1",
            "-0.7812128213",
            scalar_to_string(&y1_one),
            7.5e-3,
            mode,
            seed,
            "pass",
        ));

        let j0_zero_cross = j0(&SpecialTensor::RealScalar(2.404_825_557_7), mode).expect("j0 zero");
        assert!(
            scalar_value(&j0_zero_cross).abs() <= 7.5e-4,
            "j0 first root mismatch: {}",
            scalar_value(&j0_zero_cross)
        );
        push_test_log(test_log_json(
            "bessel-j0-first-root",
            "j0",
            "x=2.4048255577",
            "0",
            scalar_to_string(&j0_zero_cross),
            7.5e-4,
            mode,
            seed,
            "pass",
        ));

        for x in [20.0, 30.0, 40.0, 50.0] {
            let j0_v = j0(&SpecialTensor::RealScalar(x), mode).expect("j0 large x");
            let y0_v = y0(&SpecialTensor::RealScalar(x), mode).expect("y0 large x");
            let lhs = scalar_value(&j0_v).powi(2) + scalar_value(&y0_v).powi(2);
            let rhs = 2.0 / (std::f64::consts::PI * x);
            assert!(
                (lhs - rhs).abs() <= 3.5e-3,
                "Bessel asymptotic envelope mismatch at x={x}: lhs={lhs} rhs={rhs}"
            );
            push_test_log(test_log_json(
                "bessel-asymptotic-envelope",
                "j0,y0",
                format!("x={x}"),
                format!("{rhs:.16e}"),
                format!("{lhs:.16e}"),
                3.5e-3,
                mode,
                seed,
                "pass",
            ));
        }

        let logs = take_test_logs();
        assert!(!logs.is_empty(), "expected structured test logs");
        assert_logs_follow_schema(&logs);
    }

    #[test]
    fn property_erf_plus_erfc_identity_over_grid() {
        let _guard = trace_test_guard();
        clear_test_logs();
        let seed = 17_101_u64;
        let mode = RuntimeMode::Strict;

        for i in -48..=48 {
            let x = i as f64 / 8.0;
            let tensor = SpecialTensor::RealScalar(x);
            let lhs = scalar_value(&erf(&tensor, mode).expect("erf"))
                + scalar_value(&erfc(&tensor, mode).expect("erfc"));
            assert!((lhs - 1.0).abs() <= 2.0e-7, "identity mismatch for x={x}");
            push_test_log(test_log_json(
                "prop-erf-plus-erfc",
                "erf+erfc",
                format!("x={x}"),
                "1",
                format!("{lhs:.16e}"),
                2.0e-7,
                mode,
                seed,
                "pass",
            ));
        }
        assert_logs_follow_schema(&take_test_logs());
    }

    #[test]
    fn property_gamma_recurrence_and_beta_symmetry() {
        let _guard = trace_test_guard();
        clear_test_logs();
        let seed = 17_102_u64;
        let mode = RuntimeMode::Strict;

        for k in 1..=30 {
            let x = 0.1 * k as f64;
            let g_x = scalar_value(&gamma(&SpecialTensor::RealScalar(x), mode).expect("gamma(x)"));
            let g_x1 = scalar_value(
                &gamma(&SpecialTensor::RealScalar(x + 1.0), mode).expect("gamma(x+1)"),
            );
            let rhs = x * g_x;
            let rel = ((g_x1 - rhs) / rhs.max(1e-12)).abs();
            assert!(
                rel <= 2.5e-9,
                "gamma recurrence mismatch at x={x}: rel={rel}"
            );
            push_test_log(test_log_json(
                "prop-gamma-recurrence",
                "gamma",
                format!("x={x}"),
                format!("{rhs:.16e}"),
                format!("{g_x1:.16e}"),
                2.5e-9,
                mode,
                seed,
                "pass",
            ));
        }

        for a in [0.5, 1.0, 1.5, 2.0, 5.0] {
            for b in [0.25, 1.0, 2.5, 4.0] {
                let lhs = beta(
                    &SpecialTensor::RealScalar(a),
                    &SpecialTensor::RealScalar(b),
                    mode,
                )
                .expect("beta lhs");
                let rhs = beta(
                    &SpecialTensor::RealScalar(b),
                    &SpecialTensor::RealScalar(a),
                    mode,
                )
                .expect("beta rhs");
                assert_real_scalars_close(lhs.clone(), rhs.clone(), 1.0e-10);
                push_test_log(test_log_json(
                    "prop-beta-symmetry",
                    "beta",
                    format!("a={a},b={b}"),
                    scalar_to_string(&rhs),
                    scalar_to_string(&lhs),
                    1.0e-10,
                    mode,
                    seed,
                    "pass",
                ));
            }
        }

        assert_logs_follow_schema(&take_test_logs());
    }

    #[test]
    fn property_inverse_composition_and_gamma_complement() {
        let _guard = trace_test_guard();
        clear_test_logs();
        let seed = 17_103_u64;
        let mode = RuntimeMode::Strict;

        for x in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0] {
            let erf_x = erf(&SpecialTensor::RealScalar(x), mode).expect("erf(x)");
            let inv = erfinv(&erf_x, mode).expect("erfinv(erf(x))");
            let back = scalar_value(&inv);
            assert!(
                (back - x).abs() <= 2.0e-6,
                "inverse mismatch at x={x}: got {back}"
            );
            push_test_log(test_log_json(
                "prop-erfinv-erf",
                "erfinv(erf(x))",
                format!("x={x}"),
                format!("{x:.16e}"),
                format!("{back:.16e}"),
                2.0e-6,
                mode,
                seed,
                "pass",
            ));
        }

        for a in [0.25, 0.5, 1.0, 2.0, 5.0] {
            for x in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0] {
                let p = gammainc(
                    &SpecialTensor::RealScalar(a),
                    &SpecialTensor::RealScalar(x),
                    mode,
                )
                .expect("gammainc");
                let q = gammaincc(
                    &SpecialTensor::RealScalar(a),
                    &SpecialTensor::RealScalar(x),
                    mode,
                )
                .expect("gammaincc");
                let sum = scalar_value(&p) + scalar_value(&q);
                assert!(
                    (sum - 1.0).abs() <= 3.0e-12,
                    "gammainc complement mismatch at a={a} x={x}"
                );
                push_test_log(test_log_json(
                    "prop-gammainc-complement",
                    "gammainc+gammaincc",
                    format!("a={a},x={x}"),
                    "1",
                    format!("{sum:.16e}"),
                    3.0e-12,
                    mode,
                    seed,
                    "pass",
                ));
            }
        }

        assert_logs_follow_schema(&take_test_logs());
    }

    #[test]
    fn unit_vector_broadcast_and_shape_edge_cases() {
        let _guard = trace_test_guard();
        clear_test_logs();
        let seed = 17_200_u64;
        let mode = RuntimeMode::Strict;

        let vec_input = SpecialTensor::RealVec(vec![-1.0, 0.0, 1.0]);
        let erf_vec = erf(&vec_input, mode).expect("erf vec");
        let erfc_vec = erfc(&vec_input, mode).expect("erfc vec");
        let erf_vals = expect_real_vec_ref(&erf_vec, "erf vector");
        let erfc_vals = expect_real_vec_ref(&erfc_vec, "erfc vector");
        assert_eq!(erf_vals.len(), 3);
        assert_eq!(erfc_vals.len(), 3);
        for (idx, (lhs, rhs)) in erf_vals.iter().zip(erfc_vals.iter()).enumerate() {
            let sum = lhs + rhs;
            assert!(
                (sum - 1.0).abs() <= 3.0e-7,
                "vector erf+erfc identity failed at idx={idx}: {sum}"
            );
            push_test_log(test_log_json(
                "unit-erf-erfc-vector",
                "erf,erfc",
                format!("idx={idx},x={}", [-1.0_f64, 0.0, 1.0][idx]),
                "1",
                format!("{sum:.16e}"),
                3.0e-7,
                mode,
                seed,
                "pass",
            ));
        }

        let beta_vec = beta(
            &SpecialTensor::RealVec(vec![1.0, 2.0, 3.0]),
            &SpecialTensor::RealScalar(2.0),
            mode,
        )
        .expect("beta vector+scalar");
        let beta_vals = expect_real_vec(beta_vec, "beta vector+scalar");
        assert_eq!(beta_vals.len(), 3);
        assert!((beta_vals[0] - 0.5).abs() <= 1e-12);
        assert!((beta_vals[1] - (1.0 / 6.0)).abs() <= 1e-12);
        assert!((beta_vals[2] - (1.0 / 12.0)).abs() <= 1e-12);
        push_test_log(test_log_json(
            "unit-beta-vector-scalar",
            "beta",
            "a=[1,2,3],b=2",
            "[0.5,0.1666...,0.0833...]",
            format!(
                "[{:.16e},{:.16e},{:.16e}]",
                beta_vals[0], beta_vals[1], beta_vals[2]
            ),
            1e-12,
            mode,
            seed,
            "pass",
        ));

        let mismatch = beta(
            &SpecialTensor::RealVec(vec![1.0, 2.0]),
            &SpecialTensor::RealVec(vec![1.0]),
            RuntimeMode::Hardened,
        )
        .expect_err("mismatched vectors should fail");
        assert_eq!(mismatch.kind, SpecialErrorKind::DomainError);
        push_test_log(test_log_json(
            "unit-beta-mismatch-domain",
            "beta",
            "a_len=2,b_len=1,hardened",
            "DomainError",
            format!("{:?}", mismatch.kind),
            0.0,
            RuntimeMode::Hardened,
            seed,
            "pass",
        ));

        let j0_vec = j0(&SpecialTensor::RealVec(vec![0.0, 1.0, 2.0]), mode).expect("j0 vec");
        let j0_vals = expect_real_vec(j0_vec, "j0 vector");
        assert_eq!(j0_vals.len(), 3);
        assert!((j0_vals[0] - 1.0).abs() <= 1e-8);
        push_test_log(test_log_json(
            "unit-j0-vector",
            "j0",
            "x=[0,1,2]",
            "len=3",
            format!(
                "[{:.16e},{:.16e},{:.16e}]",
                j0_vals[0], j0_vals[1], j0_vals[2]
            ),
            1e-8,
            mode,
            seed,
            "pass",
        ));

        assert_logs_follow_schema(&take_test_logs());
    }

    #[test]
    fn unit_domain_boundary_matrix_strict_vs_hardened() {
        let _guard = trace_test_guard();
        clear_test_logs();
        let seed = 17_201_u64;

        let erfinv_left =
            erfinv(&SpecialTensor::RealScalar(-1.0), RuntimeMode::Strict).expect("erfinv(-1)");
        let erfinv_right =
            erfinv(&SpecialTensor::RealScalar(1.0), RuntimeMode::Strict).expect("erfinv(1)");
        let left = expect_real_scalar(erfinv_left, "erfinv(-1)");
        let right = expect_real_scalar(erfinv_right, "erfinv(1)");
        assert!(left.is_infinite() && left.is_sign_negative());
        assert!(right.is_infinite() && right.is_sign_positive());

        let erfcinv_zero =
            erfcinv(&SpecialTensor::RealScalar(0.0), RuntimeMode::Strict).expect("erfcinv(0)");
        let erfcinv_two =
            erfcinv(&SpecialTensor::RealScalar(2.0), RuntimeMode::Strict).expect("erfcinv(2)");
        let zero = expect_real_scalar(erfcinv_zero, "erfcinv(0)");
        let two = expect_real_scalar(erfcinv_two, "erfcinv(2)");
        assert!(zero.is_infinite() && zero.is_sign_positive());
        assert!(two.is_infinite() && two.is_sign_negative());

        let strict_bad_order = jn(
            &SpecialTensor::RealScalar(0.5),
            &SpecialTensor::RealScalar(1.0),
            RuntimeMode::Strict,
        )
        .expect("strict returns NaN for non-integer order");
        assert_real_scalar_nan(strict_bad_order);
        let hardened_bad_order = jn(
            &SpecialTensor::RealScalar(0.5),
            &SpecialTensor::RealScalar(1.0),
            RuntimeMode::Hardened,
        )
        .expect_err("hardened rejects non-integer order");
        assert_eq!(hardened_bad_order.kind, SpecialErrorKind::DomainError);

        let strict_bad_gamma = gammaincc(
            &SpecialTensor::RealScalar(-1.0),
            &SpecialTensor::RealScalar(0.5),
            RuntimeMode::Strict,
        )
        .expect("strict returns NaN for invalid shape");
        assert_real_scalar_nan(strict_bad_gamma);
        let hardened_bad_gamma = gammaincc(
            &SpecialTensor::RealScalar(-1.0),
            &SpecialTensor::RealScalar(0.5),
            RuntimeMode::Hardened,
        )
        .expect_err("hardened rejects invalid shape");
        assert_eq!(hardened_bad_gamma.kind, SpecialErrorKind::DomainError);

        push_test_log(test_log_json(
            "unit-boundary-policy-matrix",
            "erfinv/erfcinv/jn/gammaincc",
            "boundary + malformed domain set",
            "strict nonfinite fallback / hardened domain error",
            "validated",
            0.0,
            RuntimeMode::Strict,
            seed,
            "pass",
        ));
        assert_logs_follow_schema(&take_test_logs());
    }

    #[test]
    fn property_erfcinv_erfc_roundtrip_grid() {
        let _guard = trace_test_guard();
        clear_test_logs();
        let seed = 17_300_u64;
        let mode = RuntimeMode::Strict;

        for i in 1..=120 {
            let y = (i as f64) / 60.0;
            let inv = erfcinv(&SpecialTensor::RealScalar(y), mode).expect("erfcinv(y)");
            let back = erfc(&inv, mode).expect("erfc(erfcinv(y))");
            let back_v = scalar_value(&back);
            assert!(
                (back_v - y).abs() <= 4.0e-6,
                "erfcinv/erfc roundtrip mismatch y={y}: {back_v}"
            );
            push_test_log(test_log_json(
                "prop-erfcinv-erfc-roundtrip",
                "erfc(erfcinv(y))",
                format!("y={y:.16e}"),
                format!("{y:.16e}"),
                format!("{back_v:.16e}"),
                4.0e-6,
                mode,
                seed,
                "pass",
            ));
        }

        assert_logs_follow_schema(&take_test_logs());
    }

    #[test]
    fn erfinv_tail_contract_points() {
        let mode = RuntimeMode::Strict;
        let pairs = [
            (0.999_999, 3.458_910_737_275_499, 1.0e-7),
            (0.999_999_999_999, 5.042_031_898_572_696, 2.0e-5),
            (-0.999_999, -3.458_910_737_275_499, 1.0e-7),
        ];

        for (y, expected, tol) in pairs {
            let actual = erfinv(&SpecialTensor::RealScalar(y), mode).expect("erfinv tail point");
            assert_real_scalar_close(actual, expected, tol);
        }
    }

    #[test]
    fn erfcinv_tiny_argument_remains_finite_and_accurate() {
        let mode = RuntimeMode::Strict;
        let actual =
            erfcinv(&SpecialTensor::RealScalar(1.0e-20), mode).expect("erfcinv tiny argument");
        assert_real_scalar_close(actual, 6.601_580_622_355_143, 1.0e-8);
    }

    #[test]
    fn property_bessel_envelope_large_x_grid() {
        let _guard = trace_test_guard();
        clear_test_logs();
        let seed = 17_301_u64;
        let mode = RuntimeMode::Strict;

        for k in 0..200 {
            let x = 20.0 + 0.5 * k as f64;
            let j0_v = j0(&SpecialTensor::RealScalar(x), mode).expect("j0");
            let y0_v = y0(&SpecialTensor::RealScalar(x), mode).expect("y0");
            let lhs = scalar_value(&j0_v).powi(2) + scalar_value(&y0_v).powi(2);
            let rhs = 2.0 / (std::f64::consts::PI * x);
            assert!(
                (lhs - rhs).abs() <= 3.5e-3,
                "bessel envelope mismatch at x={x}: lhs={lhs} rhs={rhs}"
            );
            push_test_log(test_log_json(
                "prop-bessel-envelope",
                "j0^2+y0^2",
                format!("x={x:.16e}"),
                format!("{rhs:.16e}"),
                format!("{lhs:.16e}"),
                3.5e-3,
                mode,
                seed,
                "pass",
            ));
        }

        assert_logs_follow_schema(&take_test_logs());
    }

    #[test]
    fn property_jn_three_term_recurrence_grid() {
        let _guard = trace_test_guard();
        clear_test_logs();
        let seed = 17_302_u64;
        let mode = RuntimeMode::Strict;

        for n in 1..=8 {
            for step in 1..=120 {
                let x = 0.5 + 0.05 * step as f64;
                let n_tensor = SpecialTensor::RealScalar(n as f64);
                let n_prev_tensor = SpecialTensor::RealScalar((n - 1) as f64);
                let n_next_tensor = SpecialTensor::RealScalar((n + 1) as f64);
                let x_tensor = SpecialTensor::RealScalar(x);

                let jn_v = scalar_value(&jn(&n_tensor, &x_tensor, mode).expect("jn"));
                let jn_prev = scalar_value(&jn(&n_prev_tensor, &x_tensor, mode).expect("jn-1"));
                let jn_next = scalar_value(&jn(&n_next_tensor, &x_tensor, mode).expect("jn+1"));
                let rhs = (2.0 * n as f64 / x) * jn_v - jn_prev;
                assert!(
                    (jn_next - rhs).abs() <= 2.0e-6,
                    "jn recurrence mismatch n={n} x={x}: lhs={jn_next} rhs={rhs}"
                );
                push_test_log(test_log_json(
                    "prop-jn-recurrence",
                    "J_{n+1}=(2n/x)J_n-J_{n-1}",
                    format!("n={n},x={x:.16e}"),
                    format!("{rhs:.16e}"),
                    format!("{jn_next:.16e}"),
                    2.0e-6,
                    mode,
                    seed,
                    "pass",
                ));
            }
        }

        assert_logs_follow_schema(&take_test_logs());
    }

    #[test]
    fn previously_pending_families_now_implemented() {
        let scalar = SpecialTensor::RealScalar(1.0);

        // jv is now implemented — verify it returns a value
        let jv_result = jv(&scalar, &scalar, RuntimeMode::Strict);
        assert!(
            jv_result.is_ok(),
            "jv(1,1) should succeed now that it's implemented"
        );

        // hyp2f1 is now implemented — verify it returns a value instead
        let hyper_result = hyp2f1(&scalar, &scalar, &scalar, &scalar, RuntimeMode::Strict);
        assert!(
            hyper_result.is_ok(),
            "hyp2f1(1,1,1,1) should succeed now that it's implemented"
        );
    }

    #[test]
    fn dispatch_plans_cover_packet_boundaries() {
        assert!(
            GAMMA_DISPATCH_PLAN
                .iter()
                .any(|entry| entry.function == "gamma")
        );
        assert!(
            BETA_DISPATCH_PLAN
                .iter()
                .any(|entry| entry.function == "beta")
        );
        assert!(
            BESSEL_DISPATCH_PLAN
                .iter()
                .any(|entry| entry.function == "jv")
        );
        assert!(
            ERROR_DISPATCH_PLAN
                .iter()
                .any(|entry| entry.function == "erf")
        );
        assert!(
            HYPER_DISPATCH_PLAN
                .iter()
                .any(|entry| entry.function == "hyp2f1")
        );
    }

    #[test]
    fn ndtr_contract_points() {
        assert!((ndtr_scalar(0.0) - 0.5).abs() <= 1.0e-12);
        assert!((ndtr_scalar(1.959_963_984_540_054) - 0.975).abs() <= 5.0e-5);
        let tail = ndtr_scalar(-8.0);
        assert!(tail > 0.0 && tail < 1.0e-14, "ndtr(-8) tail={tail}");
        assert!(ndtr_scalar(8.0) > 1.0 - 1.0e-12, "ndtr(8) should be ~1");
        assert_eq!(ndtr_scalar(f64::NEG_INFINITY), 0.0);
        assert_eq!(ndtr_scalar(f64::INFINITY), 1.0);
    }

    #[test]
    fn ndtr_ndtri_roundtrip() {
        for x in [-4.0, -1.5, -0.25, 0.0, 0.5, 2.0, 4.0] {
            let back = ndtri_scalar(ndtr_scalar(x));
            assert!(
                (back - x).abs() <= 2.0e-4,
                "ndtri(ndtr(x)) mismatch at x={x}: {back}"
            );
        }
    }

    #[test]
    fn ndtri_domain_and_endpoints() {
        assert_eq!(ndtri_scalar(0.0), f64::NEG_INFINITY);
        assert_eq!(ndtri_scalar(1.0), f64::INFINITY);
        assert!(ndtri_scalar(-0.1).is_nan());
        assert!(ndtri_scalar(1.1).is_nan());
    }

    #[test]
    fn beta_distribution_cdf_and_inverse_roundtrip() {
        let value = btdtr(2.0, 5.0, 0.3);
        assert!(
            (value - 0.579_825).abs() <= 1.0e-6,
            "unexpected beta cdf: {value}"
        );

        let x = btdtri(2.0, 5.0, 0.3);
        let back = btdtr(2.0, 5.0, x);
        assert!(
            (back - 0.3).abs() <= 1.0e-8,
            "beta inverse roundtrip failed: {back}"
        );
    }

    #[test]
    fn beta_inverse_endpoints_and_invalid_inputs() {
        assert_eq!(btdtri(2.0, 5.0, 0.0), 0.0);
        assert_eq!(btdtri(2.0, 5.0, 1.0), 1.0);
        assert!(btdtr(2.0, 5.0, -0.1).is_nan());
        assert!(btdtri(-1.0, 5.0, 0.5).is_nan());
    }

    #[test]
    fn f_distribution_cdf_and_inverse_roundtrip() {
        let x = 1.75;
        let y = fdtr(5.0, 7.0, x);
        let back = fdtri(5.0, 7.0, y);
        assert!(
            (back - x).abs() <= 1.0e-8,
            "f inverse roundtrip failed: {back}"
        );
    }

    #[test]
    fn f_distribution_endpoints() {
        assert_eq!(fdtr(5.0, 7.0, 0.0), 0.0);
        assert_eq!(fdtri(5.0, 7.0, 0.0), 0.0);
        assert_eq!(fdtri(5.0, 7.0, 1.0), f64::INFINITY);
        assert!(fdtr(-1.0, 7.0, 1.0).is_nan());
    }

    #[test]
    fn gamma_distribution_cdf_and_inverse_roundtrip() {
        let x = 3.2;
        let y = gdtr(2.5, 1.25, x);
        let back = gdtrix(2.5, 1.25, y);
        assert!(
            (back - x).abs() <= 1.0e-8,
            "gamma inverse roundtrip failed: {back}"
        );
    }

    #[test]
    fn gamma_inverse_parameters_roundtrip() {
        let a = 1.2;
        let b = 3.4;
        let x = 5.6;
        let p = gdtr(a, b, x);
        let back_a = gdtria(p, b, x);
        let back_b = gdtrib(a, p, x);
        assert!(
            (back_a - a).abs() <= 1.0e-8,
            "gamma rate inverse failed: {back_a}"
        );
        assert!(
            (back_b - b).abs() <= 1.0e-8,
            "gamma shape inverse failed: {back_b}"
        );
    }

    #[test]
    fn gamma_distribution_exponential_special_case() {
        let x = 1.5_f64;
        let scale = 2.0_f64;
        let expected = 1.0 - (-(x / scale)).exp();
        assert!((gdtr(1.0 / scale, 1.0, x) - expected).abs() <= 1.0e-10);
    }

    #[test]
    fn gamma_inverse_endpoints_and_invalid_inputs() {
        assert_eq!(gdtrix(2.0, 3.0, 0.0), 0.0);
        assert_eq!(gdtrix(2.0, 3.0, 1.0), f64::INFINITY);
        assert_eq!(gdtria(0.0, 3.0, 2.0), 0.0);
        assert_eq!(gdtria(1.0, 3.0, 2.0), f64::INFINITY);
        assert_eq!(gdtrib(2.0, 0.0, 3.0), f64::INFINITY);
        assert_eq!(gdtrib(2.0, 1.0, 3.0), 0.0);
        assert_eq!(gdtrib(2.0, 0.5, 0.0), 0.0);
        assert!(gdtr(-1.0, 3.0, 1.0).is_nan());
        assert!(gdtrix(2.0, -3.0, 0.5).is_nan());
        assert!(gdtria(0.5, 3.0, 0.0).is_nan());
        assert!(gdtrib(2.0, 0.0, 0.0).is_nan());
    }

    #[test]
    fn kl_div_matches_rel_entr_offset() {
        let x = 0.7;
        let y = 1.4;
        let rel = rel_entr(
            &SpecialTensor::RealScalar(x),
            &SpecialTensor::RealScalar(y),
            RuntimeMode::Strict,
        )
        .expect("rel_entr");
        let expected = scalar_value(&rel) - x + y;
        assert!((kl_div(x, y) - expected).abs() <= 1.0e-12);
        assert_eq!(kl_div(0.0, 2.0), 2.0);
        assert_eq!(kl_div(1.0, 1.0), 0.0);
    }

    #[test]
    fn kl_div_invalid_inputs_are_infinite() {
        assert!(kl_div(-1.0, 1.0).is_infinite());
        assert!(kl_div(1.0, 0.0).is_infinite());
    }

    #[test]
    fn kl_div_nan_inputs_propagate() {
        assert!(kl_div(f64::NAN, 1.0).is_nan());
        assert!(kl_div(1.0, f64::NAN).is_nan());
    }

    #[test]
    fn complex_erf_erfc_support_known_values_and_identities() {
        let mode = RuntimeMode::Strict;
        let i = SpecialTensor::ComplexScalar(Complex64::new(0.0, 1.0));

        let erf_i = erf(&i, mode).expect("erf(i)");
        assert_complex_scalar_close(
            erf_i.clone(),
            Complex64::new(0.0, 1.650_425_758_797_542_8),
            1.0e-11,
        );

        let erfc_i = erfc(&i, mode).expect("erfc(i)");
        assert_complex_scalar_close(
            erfc_i,
            Complex64::new(1.0, -1.650_425_758_797_542_8),
            1.0e-11,
        );

        let z = SpecialTensor::ComplexScalar(Complex64::new(1.0, 1.0));
        let z_conj = SpecialTensor::ComplexScalar(Complex64::new(1.0, -1.0));
        let erf_z = erf(&z, mode).expect("erf(1+i)");
        let erfc_z = erfc(&z, mode).expect("erfc(1+i)");
        let erf_conj = erf(&z_conj, mode).expect("erf(1-i)");

        assert_complex_scalar_close(
            add_complex_scalars(&erf_z, &erfc_z),
            Complex64::from_real(1.0),
            1.0e-11,
        );
        assert_complex_scalar_close(erf_conj, complex_scalar_value(&erf_z).conj(), 1.0e-11);
    }

    #[test]
    fn complex_vector_erf_preserves_shape_and_odd_symmetry() {
        let mode = RuntimeMode::Strict;
        let input =
            SpecialTensor::ComplexVec(vec![Complex64::new(0.0, 0.5), Complex64::new(0.0, -0.5)]);

        let output = erf(&input, mode).expect("complex vector erf");
        let values = expect_complex_vec(output, "complex vector erf");

        assert_eq!(values.len(), 2);
        assert_complex_close(values[0], -values[1], 1.0e-12);
    }

    fn expect_real_scalar(tensor: SpecialTensor, context: &str) -> f64 {
        match tensor {
            SpecialTensor::RealScalar(value) => value,
            other => unexpected_tensor(&other, context, "real scalar output"),
        }
    }

    fn expect_real_scalar_ref(tensor: &SpecialTensor, context: &str) -> f64 {
        match tensor {
            SpecialTensor::RealScalar(value) => *value,
            other => unexpected_tensor(other, context, "real scalar output"),
        }
    }

    fn expect_real_vec(tensor: SpecialTensor, context: &str) -> Vec<f64> {
        match tensor {
            SpecialTensor::RealVec(values) => values,
            other => unexpected_tensor(&other, context, "real vector output"),
        }
    }

    fn expect_real_vec_ref<'a>(tensor: &'a SpecialTensor, context: &str) -> &'a [f64] {
        match tensor {
            SpecialTensor::RealVec(values) => values,
            other => unexpected_tensor(other, context, "real vector output"),
        }
    }

    fn expect_complex_scalar(tensor: SpecialTensor, context: &str) -> Complex64 {
        match tensor {
            SpecialTensor::ComplexScalar(value) => value,
            other => unexpected_tensor(&other, context, "complex scalar output"),
        }
    }

    fn expect_complex_vec(tensor: SpecialTensor, context: &str) -> Vec<Complex64> {
        match tensor {
            SpecialTensor::ComplexVec(values) => values,
            other => unexpected_tensor(&other, context, "complex vector output"),
        }
    }

    fn unexpected_tensor<T>(actual: &SpecialTensor, context: &str, expected: &str) -> T {
        std::panic::panic_any(format!("{context}: expected {expected}, got {actual:?}"))
    }

    fn assert_real_scalar_close(actual: SpecialTensor, expected: f64, tol: f64) {
        let value = expect_real_scalar(actual, "assert_real_scalar_close");
        assert!(
            (value - expected).abs() <= tol,
            "expected {expected}, got {value}"
        );
    }

    fn assert_real_scalar_nan(actual: SpecialTensor) {
        let value = expect_real_scalar(actual, "assert_real_scalar_nan");
        assert!(value.is_nan(), "expected NaN");
    }

    fn assert_real_scalars_close(actual: SpecialTensor, expected: SpecialTensor, tol: f64) {
        let lhs = expect_real_scalar(actual, "assert_real_scalars_close actual");
        let rhs = expect_real_scalar(expected, "assert_real_scalars_close expected");
        assert!((lhs - rhs).abs() <= tol, "expected {rhs}, got {lhs}");
    }

    fn assert_complex_scalar_close(actual: SpecialTensor, expected: Complex64, tol: f64) {
        assert_complex_close(
            expect_complex_scalar(actual, "assert_complex_scalar_close"),
            expected,
            tol,
        );
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64, tol: f64) {
        let delta = (actual - expected).abs();
        assert!(
            delta <= tol,
            "expected {}+{}i, got {}+{}i (|delta|={delta})",
            expected.re,
            expected.im,
            actual.re,
            actual.im
        );
    }

    fn trace_test_guard() -> std::sync::MutexGuard<'static, ()> {
        static TRACE_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        TRACE_TEST_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("trace test lock poisoned")
    }

    fn scalar_value(tensor: &SpecialTensor) -> f64 {
        expect_real_scalar_ref(tensor, "scalar_value")
    }

    fn complex_scalar_value(tensor: &SpecialTensor) -> Complex64 {
        match tensor {
            SpecialTensor::ComplexScalar(value) => *value,
            other => unexpected_tensor(other, "complex_scalar_value", "complex scalar tensor"),
        }
    }

    fn add_complex_scalars(lhs: &SpecialTensor, rhs: &SpecialTensor) -> SpecialTensor {
        SpecialTensor::ComplexScalar(complex_scalar_value(lhs) + complex_scalar_value(rhs))
    }

    fn scalar_to_string(tensor: &SpecialTensor) -> String {
        format!("{:.16e}", scalar_value(tensor))
    }

    fn test_logs() -> &'static Mutex<Vec<String>> {
        static TEST_LOGS: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
        TEST_LOGS.get_or_init(|| Mutex::new(Vec::new()))
    }

    fn clear_test_logs() {
        if let Ok(mut logs) = test_logs().lock() {
            logs.clear();
        }
    }

    fn push_test_log(line: String) {
        if let Ok(mut logs) = test_logs().lock() {
            logs.push(line);
        }
    }

    fn take_test_logs() -> Vec<String> {
        if let Ok(mut logs) = test_logs().lock() {
            let mut out = Vec::with_capacity(logs.len());
            std::mem::swap(&mut out, &mut *logs);
            return out;
        }
        Vec::new()
    }

    #[allow(clippy::too_many_arguments)]
    fn test_log_json(
        test_id: impl Into<String>,
        function: impl Into<String>,
        input: impl Into<String>,
        expected: impl Into<String>,
        actual: impl Into<String>,
        tolerance: f64,
        mode: RuntimeMode,
        seed: u64,
        result: impl Into<String>,
    ) -> String {
        let test_id = test_id.into();
        let function = function.into();
        let input = input.into();
        let expected = expected.into();
        let actual = actual.into();
        let result = result.into();
        format!(
            "{{\"test_id\":\"{}\",\"function\":\"{}\",\"input\":\"{}\",\"expected\":\"{}\",\"actual\":\"{}\",\"tolerance\":{},\"mode\":\"{:?}\",\"seed\":{},\"timestamp_ms\":{},\"result\":\"{}\"}}",
            escape_json(&test_id),
            escape_json(&function),
            escape_json(&input),
            escape_json(&expected),
            escape_json(&actual),
            tolerance,
            mode,
            seed,
            now_unix_ms(),
            escape_json(&result),
        )
    }

    fn assert_logs_follow_schema(logs: &[String]) {
        assert!(!logs.is_empty(), "expected non-empty structured logs");
        for line in logs {
            assert!(line.starts_with('{'));
            assert!(line.contains("\"test_id\""));
            assert!(line.contains("\"function\""));
            assert!(line.contains("\"input\""));
            assert!(line.contains("\"expected\""));
            assert!(line.contains("\"actual\""));
            assert!(line.contains("\"tolerance\""));
            assert!(line.contains("\"mode\""));
            assert!(line.contains("\"seed\""));
            assert!(line.contains("\"timestamp_ms\""));
            assert!(line.contains("\"result\""));
        }
    }

    fn now_unix_ms() -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |duration| duration.as_millis())
    }

    fn escape_json(input: &str) -> String {
        input
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    }

    // ── Combinatorial function tests ────────────────────────────────

    #[test]
    fn factorial_small_values() {
        assert_eq!(factorial(0), 1.0);
        assert_eq!(factorial(1), 1.0);
        assert_eq!(factorial(5), 120.0);
        assert_eq!(factorial(10), 3_628_800.0);
        assert_eq!(factorial(20), 2_432_902_008_176_640_000.0);
    }

    #[test]
    fn factorial_large_values() {
        // 21! should use gamma path
        let f21 = factorial(21);
        assert!((f21 - 51_090_942_171_709_440_000.0).abs() / f21 < 1e-10);
        // Large factorials overflow to infinity in f64.
        // 170! is finite, while 171! ≈ 1.24e309 exceeds f64::MAX.
        let f170 = factorial(170);
        assert!(f170.is_finite());
        let f171 = factorial(171);
        assert!(f171.is_infinite());
    }

    #[test]
    fn comb_basic() {
        assert_eq!(comb(5, 0), 1.0);
        assert_eq!(comb(5, 5), 1.0);
        assert_eq!(comb(5, 2), 10.0);
        assert_eq!(comb(10, 3), 120.0);
        assert_eq!(comb(20, 10), 184_756.0);
    }

    #[test]
    fn comb_edge_cases() {
        assert_eq!(comb(0, 0), 1.0);
        assert_eq!(comb(5, 6), 0.0); // k > n
        assert_eq!(comb(100, 1), 100.0);
        assert_eq!(comb(100, 99), 100.0); // symmetry
    }

    #[test]
    fn comb_large() {
        // C(200, 100) is a huge number; verify it's finite and reasonable
        let c = comb(200, 100);
        assert!(c.is_finite());
        assert!(c > 1e58); // known to be ~9.05e58
    }

    #[test]
    fn perm_basic() {
        assert_eq!(perm(5, 0), 1.0);
        assert_eq!(perm(5, 1), 5.0);
        assert_eq!(perm(5, 2), 20.0);
        assert_eq!(perm(5, 5), 120.0);
        assert_eq!(perm(10, 3), 720.0);
    }

    #[test]
    fn perm_edge_cases() {
        assert_eq!(perm(0, 0), 1.0);
        assert_eq!(perm(5, 6), 0.0); // k > n
    }

    #[test]
    fn perm_relation_to_comb() {
        // P(n, k) = C(n, k) * k!
        for n in 0..15_u64 {
            for k in 0..=n {
                let p = perm(n, k);
                let c = comb(n, k);
                let kfact = factorial(k);
                assert!(
                    (p - c * kfact).abs() < 1e-6 * p.max(1.0),
                    "P({n},{k})={p} != C({n},{k})*{k}!={}",
                    c * kfact
                );
            }
        }
    }

    // ── Zeta function tests ─────────────────────────────────────────

    #[test]
    fn zeta_known_values() {
        // ζ(2) = π²/6
        let z2 = zeta(2.0);
        let expected = std::f64::consts::PI * std::f64::consts::PI / 6.0;
        assert!(
            (z2 - expected).abs() < 1e-3,
            "zeta(2) = {z2}, expected {expected}"
        );

        // ζ(4) = π⁴/90
        let z4 = zeta(4.0);
        let expected4 = std::f64::consts::PI.powi(4) / 90.0;
        assert!(
            (z4 - expected4).abs() < 1e-4,
            "zeta(4) = {z4}, expected {expected4}"
        );
    }

    #[test]
    fn zeta_at_zero() {
        assert!((zeta(0.0) - (-0.5)).abs() < 1e-12, "zeta(0) = -1/2");
    }

    #[test]
    fn zeta_pole_at_one() {
        assert!(zeta(1.0).is_infinite());
    }

    #[test]
    fn zeta_negative_even_integers() {
        // ζ(-2n) = 0 for positive integers n (trivial zeros)
        assert!(zeta(-2.0).abs() < 1e-6, "zeta(-2) = {}", zeta(-2.0));
        assert!(zeta(-4.0).abs() < 1e-4, "zeta(-4) = {}", zeta(-4.0));
    }

    #[test]
    fn zeta_nan_passthrough() {
        assert!(zeta(f64::NAN).is_nan());
    }

    // ── Spherical Bessel function tests ──────────────────────────────

    #[test]
    fn spherical_jn_j0_is_sinc() {
        // j_0(x) = sin(x)/x
        let mode = RuntimeMode::Strict;
        for x in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0] {
            let result = spherical_jn(
                &SpecialTensor::RealScalar(0.0),
                &SpecialTensor::RealScalar(x),
                mode,
            )
            .expect("spherical_jn(0, x)");
            let expected = x.sin() / x;
            assert_real_scalar_close(result, expected, 1e-12);
        }
    }

    #[test]
    fn spherical_jn_at_zero() {
        let mode = RuntimeMode::Strict;
        // j_0(0) = 1
        let j0_0 = spherical_jn(
            &SpecialTensor::RealScalar(0.0),
            &SpecialTensor::RealScalar(0.0),
            mode,
        )
        .expect("j_0(0)");
        assert_real_scalar_close(j0_0, 1.0, 1e-12);

        // j_n(0) = 0 for n > 0
        for n in 1..=5 {
            let jn_0 = spherical_jn(
                &SpecialTensor::RealScalar(n as f64),
                &SpecialTensor::RealScalar(0.0),
                mode,
            )
            .expect("j_n(0)");
            assert_real_scalar_close(jn_0, 0.0, 1e-12);
        }
    }

    #[test]
    fn spherical_jn_recurrence() {
        // (2n+1) j_n(z) = z (j_{n-1}(z) + j_{n+1}(z))
        let mode = RuntimeMode::Strict;
        for n in 1..=6 {
            for x in [1.0, 2.5, 5.0, 10.0] {
                let jn_v = scalar_value(
                    &spherical_jn(
                        &SpecialTensor::RealScalar(n as f64),
                        &SpecialTensor::RealScalar(x),
                        mode,
                    )
                    .expect("jn"),
                );
                let jn_prev = scalar_value(
                    &spherical_jn(
                        &SpecialTensor::RealScalar((n - 1) as f64),
                        &SpecialTensor::RealScalar(x),
                        mode,
                    )
                    .expect("jn-1"),
                );
                let jn_next = scalar_value(
                    &spherical_jn(
                        &SpecialTensor::RealScalar((n + 1) as f64),
                        &SpecialTensor::RealScalar(x),
                        mode,
                    )
                    .expect("jn+1"),
                );
                let lhs = (2.0 * n as f64 + 1.0) * jn_v;
                let rhs = x * (jn_prev + jn_next);
                assert!(
                    (lhs - rhs).abs() < 1e-8,
                    "spherical jn recurrence n={n} x={x}: lhs={lhs} rhs={rhs}"
                );
            }
        }
    }

    #[test]
    fn spherical_yn_y0_is_neg_cos_over_x() {
        // y_0(x) = -cos(x)/x
        let mode = RuntimeMode::Strict;
        for x in [0.5, 1.0, 2.0, 5.0, 10.0] {
            let result = spherical_yn(
                &SpecialTensor::RealScalar(0.0),
                &SpecialTensor::RealScalar(x),
                mode,
            )
            .expect("spherical_yn(0, x)");
            let expected = -x.cos() / x;
            assert_real_scalar_close(result, expected, 1e-12);
        }
    }

    #[test]
    fn spherical_yn_at_zero_is_neg_inf() {
        let mode = RuntimeMode::Strict;
        let result = spherical_yn(
            &SpecialTensor::RealScalar(0.0),
            &SpecialTensor::RealScalar(0.0),
            mode,
        )
        .expect("y_0(0)");
        let value = expect_real_scalar(result, "spherical_yn(0, 0)");
        assert!(value.is_infinite() && value.is_sign_negative());
    }

    #[test]
    fn spherical_in_i0_is_sinh_over_x() {
        // i_0(x) = sinh(x)/x
        let mode = RuntimeMode::Strict;
        for x in [0.5, 1.0, 2.0, 5.0] {
            let result = spherical_in(
                &SpecialTensor::RealScalar(0.0),
                &SpecialTensor::RealScalar(x),
                mode,
            )
            .expect("spherical_in(0, x)");
            let expected = x.sinh() / x;
            assert_real_scalar_close(result, expected, 1e-12);
        }
    }

    #[test]
    fn spherical_in_at_zero() {
        let mode = RuntimeMode::Strict;
        let i0_0 = spherical_in(
            &SpecialTensor::RealScalar(0.0),
            &SpecialTensor::RealScalar(0.0),
            mode,
        )
        .expect("i_0(0)");
        assert_real_scalar_close(i0_0, 1.0, 1e-12);

        let i1_0 = spherical_in(
            &SpecialTensor::RealScalar(1.0),
            &SpecialTensor::RealScalar(0.0),
            mode,
        )
        .expect("i_1(0)");
        assert_real_scalar_close(i1_0, 0.0, 1e-12);
    }

    #[test]
    fn spherical_kn_k0_uses_scipy_normalization() {
        // SciPy's k_0(x) = pi * exp(-x) / (2x)
        let mode = RuntimeMode::Strict;
        for x in [0.5, 1.0, 2.0, 5.0] {
            let result = spherical_kn(
                &SpecialTensor::RealScalar(0.0),
                &SpecialTensor::RealScalar(x),
                mode,
            )
            .expect("spherical_kn(0, x)");
            let expected = std::f64::consts::FRAC_PI_2 * (-x).exp() / x;
            assert_real_scalar_close(result, expected, 1e-12);
        }
    }

    #[test]
    fn spherical_kn_at_zero_is_inf() {
        let mode = RuntimeMode::Strict;
        let result = spherical_kn(
            &SpecialTensor::RealScalar(0.0),
            &SpecialTensor::RealScalar(0.0),
            mode,
        )
        .expect("k_0(0)");
        assert!(expect_real_scalar(result, "spherical_kn(0, 0)").is_infinite());
    }

    #[test]
    fn spherical_bessel_negative_order_rejected_in_hardened() {
        let mode = RuntimeMode::Hardened;
        let neg = SpecialTensor::RealScalar(-1.0);
        let x = SpecialTensor::RealScalar(1.0);

        let err = spherical_jn(&neg, &x, mode).expect_err("negative order");
        assert_eq!(err.kind, SpecialErrorKind::DomainError);
    }

    #[test]
    fn spherical_jn_j1_formula() {
        // j_1(x) = sin(x)/x² - cos(x)/x
        let mode = RuntimeMode::Strict;
        for x in [0.5, 1.0, 3.0, 7.0] {
            let result = spherical_jn(
                &SpecialTensor::RealScalar(1.0),
                &SpecialTensor::RealScalar(x),
                mode,
            )
            .expect("j_1(x)");
            let expected = x.sin() / (x * x) - x.cos() / x;
            assert_real_scalar_close(result, expected, 1e-12);
        }
    }

    #[test]
    fn spherical_kn_k1_formula() {
        // SciPy's k_1(x) = pi * exp(-x) / (2x) * (1 + 1/x)
        let mode = RuntimeMode::Strict;
        for x in [0.5, 1.0, 3.0] {
            let result = spherical_kn(
                &SpecialTensor::RealScalar(1.0),
                &SpecialTensor::RealScalar(x),
                mode,
            )
            .expect("k_1(x)");
            let expected = std::f64::consts::FRAC_PI_2 * (-x).exp() / x * (1.0 + 1.0 / x);
            assert_real_scalar_close(result, expected, 1e-12);
        }
    }

    #[test]
    fn spherical_bessel_complex_zero_order_closed_forms() {
        let mode = RuntimeMode::Strict;
        let z = Complex64::new(1.25, -0.75);
        let z_tensor = SpecialTensor::ComplexScalar(z);
        let order = SpecialTensor::RealScalar(0.0);

        assert_complex_scalar_close(
            spherical_jn(&order, &z_tensor, mode).expect("complex spherical_jn"),
            z.sin() * z.recip(),
            1.0e-12,
        );
        assert_complex_scalar_close(
            spherical_yn(&order, &z_tensor, mode).expect("complex spherical_yn"),
            -z.cos() * z.recip(),
            1.0e-12,
        );
        assert_complex_scalar_close(
            spherical_in(&order, &z_tensor, mode).expect("complex spherical_in"),
            z.sinh() * z.recip(),
            1.0e-12,
        );
        assert_complex_scalar_close(
            spherical_kn(&order, &z_tensor, mode).expect("complex spherical_kn"),
            Complex64::from_real(std::f64::consts::FRAC_PI_2) * (-z).exp() * z.recip(),
            1.0e-12,
        );
    }

    #[test]
    fn spherical_bessel_complex_real_axis_matches_real_outputs() {
        let mode = RuntimeMode::Strict;
        let x = 1.75;
        let z_complex = SpecialTensor::ComplexScalar(Complex64::from_real(x));

        for order in [0.0, 1.0, 3.0] {
            let order_tensor = SpecialTensor::RealScalar(order);

            let real_jn = spherical_jn(&order_tensor, &SpecialTensor::RealScalar(x), mode)
                .expect("real spherical_jn");
            let complex_jn =
                spherical_jn(&order_tensor, &z_complex, mode).expect("complex spherical_jn");
            assert_complex_close(
                complex_scalar_value(&complex_jn),
                Complex64::from_real(scalar_value(&real_jn)),
                1.0e-12,
            );

            let real_yn = spherical_yn(&order_tensor, &SpecialTensor::RealScalar(x), mode)
                .expect("real spherical_yn");
            let complex_yn =
                spherical_yn(&order_tensor, &z_complex, mode).expect("complex spherical_yn");
            assert_complex_close(
                complex_scalar_value(&complex_yn),
                Complex64::from_real(scalar_value(&real_yn)),
                1.0e-12,
            );

            let real_in = spherical_in(&order_tensor, &SpecialTensor::RealScalar(x), mode)
                .expect("real spherical_in");
            let complex_in =
                spherical_in(&order_tensor, &z_complex, mode).expect("complex spherical_in");
            assert_complex_close(
                complex_scalar_value(&complex_in),
                Complex64::from_real(scalar_value(&real_in)),
                1.0e-12,
            );

            let real_kn = spherical_kn(&order_tensor, &SpecialTensor::RealScalar(x), mode)
                .expect("real spherical_kn");
            let complex_kn =
                spherical_kn(&order_tensor, &z_complex, mode).expect("complex spherical_kn");
            assert_complex_close(
                complex_scalar_value(&complex_kn),
                Complex64::from_real(scalar_value(&real_kn)),
                1.0e-12,
            );
        }
    }

    #[test]
    fn spherical_bessel_complex_vector_preserves_conjugation() {
        let mode = RuntimeMode::Strict;
        let z = Complex64::new(1.0, 0.5);
        let inputs = SpecialTensor::ComplexVec(vec![z, z.conj()]);
        let order = SpecialTensor::RealScalar(2.0);

        let jn_values = expect_complex_vec(
            spherical_jn(&order, &inputs, mode).expect("complex spherical_jn"),
            "complex spherical_jn",
        );
        assert_complex_close(jn_values[1], jn_values[0].conj(), 1.0e-11);

        let yn_values = expect_complex_vec(
            spherical_yn(&order, &inputs, mode).expect("complex spherical_yn"),
            "complex spherical_yn",
        );
        assert_complex_close(yn_values[1], yn_values[0].conj(), 1.0e-11);

        let in_values = expect_complex_vec(
            spherical_in(&order, &inputs, mode).expect("complex spherical_in"),
            "complex spherical_in",
        );
        assert_complex_close(in_values[1], in_values[0].conj(), 1.0e-11);

        let kn_values = expect_complex_vec(
            spherical_kn(&order, &inputs, mode).expect("complex spherical_kn"),
            "complex spherical_kn",
        );
        assert_complex_close(kn_values[1], kn_values[0].conj(), 1.0e-11);
    }

    // ── Real-order Bessel function tests ─────────────────────────────

    #[test]
    fn jv_integer_order_matches_jn() {
        // jv(n, x) should match jn(n, x) for integer n
        let mode = RuntimeMode::Strict;
        for n in 0..=5 {
            let x = 2.5;
            let jv_result = scalar_value(
                &jv(
                    &SpecialTensor::RealScalar(n as f64),
                    &SpecialTensor::RealScalar(x),
                    mode,
                )
                .expect("jv"),
            );
            let jn_result = scalar_value(
                &jn(
                    &SpecialTensor::RealScalar(n as f64),
                    &SpecialTensor::RealScalar(x),
                    mode,
                )
                .expect("jn"),
            );
            assert!(
                (jv_result - jn_result).abs() < 1e-8,
                "jv({n}, {x}) = {jv_result}, jn = {jn_result}"
            );
        }
    }

    #[test]
    fn jv_half_order() {
        // J_{1/2}(z) = sqrt(2/(πz)) sin(z)
        let mode = RuntimeMode::Strict;
        for x in [1.0, 2.0, 5.0, 10.0] {
            let result = scalar_value(
                &jv(
                    &SpecialTensor::RealScalar(0.5),
                    &SpecialTensor::RealScalar(x),
                    mode,
                )
                .expect("jv(0.5)"),
            );
            let expected = (2.0 / (std::f64::consts::PI * x)).sqrt() * x.sin();
            assert!(
                (result - expected).abs() < 0.01,
                "J_1/2({x}): got {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn yv_positive_z() {
        // Y_v should be finite for positive z
        let mode = RuntimeMode::Strict;
        let result = yv(
            &SpecialTensor::RealScalar(0.5),
            &SpecialTensor::RealScalar(1.0),
            mode,
        )
        .expect("yv(0.5, 1)");
        let value = expect_real_scalar(result, "yv(0.5, 1)");
        assert!(value.is_finite(), "Y_0.5(1) should be finite: {value}");
    }

    #[test]
    fn yv_negative_z_strict_returns_nan() {
        let mode = RuntimeMode::Strict;
        let result = yv(
            &SpecialTensor::RealScalar(0.5),
            &SpecialTensor::RealScalar(-1.0),
            mode,
        )
        .expect("yv strict");
        assert_real_scalar_nan(result);
    }

    /// yv at non-integer orders v > 1 (frankenscipy-6avjb).
    ///
    /// Pre-fix the reflection formula Y_v = (J_v cos(vπ) − J_{−v}) / sin(vπ)
    /// called jv_series(-v, z), and that series used log(v+k+1) in its
    /// log-magnitude recurrence — NaN whenever v+k+1 < 0 — and ignored
    /// the sign of Γ(v+k+1). The result was NaN for every non-integer
    /// v > 1, even though Y_v is real and well-defined for z > 0.
    #[test]
    fn yv_finite_at_non_integer_orders_above_one() {
        let mode = RuntimeMode::Strict;
        // (v, z, scipy.special.yv(v, z))
        let cases: [(f64, f64, f64); 9] = [
            (1.5, 1.0, -1.1024955751601793),
            (1.5, 5.0, 0.321_924_442_961_140_2),
            (2.5, 5.0, 0.294_372_374_961_792_5),
            (2.5, 10.0, -0.16417847961494103),
            (1.3, 5.0, 0.26631384911040207),
            (1.7, 5.0, 0.356_264_127_687_647_9),
            (2.3, 5.0, 0.335_540_080_653_554_2),
            (3.7, 5.0, -0.09577011740148628),
            (5.5, 10.0, 0.23675446066584144),
        ];
        for (v, z, expected) in cases {
            let result = yv(
                &SpecialTensor::RealScalar(v),
                &SpecialTensor::RealScalar(z),
                mode,
            )
            .expect("yv strict");
            let value = expect_real_scalar(result, "yv at non-integer order");
            let diff = (value - expected).abs();
            let scale = expected.abs().max(1.0);
            assert!(
                value.is_finite(),
                "yv({v}, {z}) returned non-finite {value}"
            );
            assert!(
                diff < 1e-10 * scale,
                "yv({v}, {z}) = {value}, expected {expected}, diff = {diff}"
            );
        }
    }

    #[test]
    fn iv_at_zero() {
        let mode = RuntimeMode::Strict;
        // I_0(0) = 1
        let result = iv(
            &SpecialTensor::RealScalar(0.0),
            &SpecialTensor::RealScalar(0.0),
            mode,
        )
        .expect("iv(0,0)");
        assert_real_scalar_close(result, 1.0, 1e-12);
        // I_v(0) = 0 for v > 0
        let result = iv(
            &SpecialTensor::RealScalar(1.0),
            &SpecialTensor::RealScalar(0.0),
            mode,
        )
        .expect("iv(1,0)");
        assert_real_scalar_close(result, 0.0, 1e-12);
    }

    #[test]
    fn iv_half_order() {
        // I_{1/2}(z) = sqrt(2/(πz)) sinh(z)
        let mode = RuntimeMode::Strict;
        for x in [0.5, 1.0, 2.0] {
            let result = scalar_value(
                &iv(
                    &SpecialTensor::RealScalar(0.5),
                    &SpecialTensor::RealScalar(x),
                    mode,
                )
                .expect("iv(0.5)"),
            );
            let expected = (2.0 / (std::f64::consts::PI * x)).sqrt() * x.sinh();
            assert!(
                (result - expected).abs() < 0.01,
                "I_1/2({x}): got {result}, expected {expected}"
            );
        }
    }

    #[test]
    fn kv_positive_z() {
        let mode = RuntimeMode::Strict;
        let result = kv(
            &SpecialTensor::RealScalar(0.5),
            &SpecialTensor::RealScalar(1.0),
            mode,
        )
        .expect("kv(0.5, 1)");
        let value = expect_real_scalar(result, "kv(0.5, 1)");
        assert!(
            value.is_finite() && value > 0.0,
            "K_0.5(1) should be positive: {value}"
        );
    }

    #[test]
    fn kv_at_zero_is_inf() {
        let mode = RuntimeMode::Strict;
        let result = kv(
            &SpecialTensor::RealScalar(0.0),
            &SpecialTensor::RealScalar(0.0),
            mode,
        )
        .expect("kv(0,0)");
        assert!(expect_real_scalar(result, "kv(0,0)").is_infinite());
    }

    #[test]
    fn kv_integer_order_positive_and_finite() {
        // Small integer-order K_n(1) values should be finite and match reference values.
        let mode = RuntimeMode::Strict;
        for (n, target) in [
            (0.0, 0.421_024_438_240_708_34),
            (1.0, 0.601_907_230_197_234_6),
            (2.0, 1.624_838_898_635_177_4),
        ] {
            let result = kv(
                &SpecialTensor::RealScalar(n),
                &SpecialTensor::RealScalar(1.0),
                mode,
            )
            .expect("kv integer order should evaluate");
            let value = expect_real_scalar(result, "kv integer order");
            assert!(
                value.is_finite() && value > 0.0,
                "K_{n}(1) should be positive finite: {value}"
            );
            assert!(
                (value - target).abs() <= 2.0e-10,
                "K_{n}(1) mismatch: expected {target}, got {value}"
            );
        }
    }

    #[test]
    fn kv_half_order_formula() {
        // K_{1/2}(z) = sqrt(π/(2z)) * exp(-z)
        let mode = RuntimeMode::Strict;
        for x in [0.5, 1.0, 2.0] {
            let result = scalar_value(
                &kv(
                    &SpecialTensor::RealScalar(0.5),
                    &SpecialTensor::RealScalar(x),
                    mode,
                )
                .expect("kv(0.5)"),
            );
            let expected = (std::f64::consts::PI / (2.0 * x)).sqrt() * (-x).exp();
            assert!(
                (result - expected).abs() < 0.01,
                "K_1/2({x}): got {result}, expected {expected}"
            );
        }
    }

    // ── Bernoulli / Euler / Hurwitz zeta tests ───────────────────────

    #[test]
    fn bernoulli_known_values() {
        assert!((bernoulli(0) - 1.0).abs() < 1e-12);
        assert!((bernoulli(1) - (-0.5)).abs() < 1e-12);
        assert!((bernoulli(2) - (1.0 / 6.0)).abs() < 1e-12);
        assert!((bernoulli(3)).abs() < 1e-12); // odd > 1 are zero
        assert!((bernoulli(4) - (-1.0 / 30.0)).abs() < 1e-12);
    }

    #[test]
    fn euler_known_values() {
        assert!((euler(0) - 1.0).abs() < 1e-12);
        assert!((euler(1)).abs() < 1e-12); // odd are zero
        assert!((euler(2) - (-1.0)).abs() < 1e-12);
        assert!((euler(4) - 5.0).abs() < 1e-12);
        assert!((euler(6) - (-61.0)).abs() < 1e-12);
    }

    #[test]
    fn hurwitz_zeta_reduces_to_riemann() {
        // ζ(s, 1) = ζ(s) (Riemann zeta)
        let riemann = zeta(2.0);
        let hurwitz = hurwitz_zeta(2.0, 1.0);
        assert!(
            (hurwitz - riemann).abs() < 1e-12,
            "hurwitz(2,1) = {hurwitz}, riemann(2) = {riemann}"
        );
    }

    #[test]
    fn hurwitz_zeta_known_value() {
        // ζ(2, 0.5) = π² / 2 ≈ 4.9348
        let result = hurwitz_zeta(2.0, 0.5);
        let expected = std::f64::consts::PI * std::f64::consts::PI / 2.0;
        assert!(
            (result - expected).abs() < 1e-12,
            "ζ(2, 0.5) = {result}, expected {expected}"
        );
    }

    /// Anchor tests at small s near the pole (frankenscipy-3u8ze).
    ///
    /// Pre-fix the Euler-Maclaurin tail correction stopped at the integral
    /// term, so a 10 000-term direct sum still floored at ~1e-5 abs at
    /// s=1.1. The full Bernoulli expansion (j up to 4) brings precision
    /// back to floating-point limit (~1e-15) for both Hurwitz and Riemann.
    #[test]
    fn hurwitz_zeta_matches_scipy_at_small_s() {
        // (s, a, scipy.special.zeta(s, a))
        let cases: [(f64, f64, f64); 7] = [
            (1.1, 1.0, 10.584448464950802),
            (1.1, 2.0, 9.584448464950798),
            (1.1, 0.5, 12.103813495683745),
            (1.5, 2.0, 1.6123753486854886),
            (2.0, 0.5, 4.934_802_200_544_68),
            (2.0, 3.0, 0.3949340668482265),
            (3.0, 1.5, 0.414_398_322_117_16),
        ];
        for (s, a, expected) in cases {
            let got = hurwitz_zeta(s, a);
            let diff = (got - expected).abs();
            assert!(
                diff < 1e-12,
                "hurwitz_zeta({s}, {a}) = {got}, expected {expected}, diff = {diff}"
            );
        }
    }

    #[test]
    fn hurwitz_zeta_negative_a_matches_scipy() {
        // scipy.special.zeta(s, a) is finite for a<0 when s is an integer (shift
        // recurrence over the negative-base terms); we previously returned NaN.
        let cases: [(f64, f64, f64); 5] = [
            (2.0, -0.5, 8.934802200544679),
            (3.0, -1.5, 0.11810202582086413),
            (4.0, -2.5, 32.457979369864596),
            (2.0, -2.5, 9.539246644989124),
            (5.0, -0.5, 0.14476040944446772),
        ];
        for (s, a, expected) in cases {
            let got = hurwitz_zeta(s, a);
            assert!(
                (got - expected).abs() <= 1e-11 * expected.abs().max(1.0),
                "hurwitz_zeta({s}, {a}) = {got}, expected {expected}"
            );
        }
        // Nonpositive-integer a is a pole; non-integer s with a<0 is NaN (scipy).
        assert!(hurwitz_zeta(2.0, 0.0).is_infinite());
        assert!(hurwitz_zeta(2.0, -1.0).is_infinite());
        assert!(hurwitz_zeta(2.5, -0.5).is_nan());
    }

    #[test]
    fn riemann_zeta_matches_scipy_at_small_s() {
        // scipy.special.zeta(s) for s slightly above the pole at s=1.
        let cases: [(f64, f64); 5] = [
            (1.05, 20.580_844_302_036_99),
            (1.1, 10.584448464950802),
            (1.2, 5.591582441177753),
            (1.5, 2.612_375_348_685_488),
            (2.0, 1.6449340668482264),
        ];
        for (s, expected) in cases {
            let got = zeta(s);
            let diff = (got - expected).abs();
            assert!(
                diff < 1e-12,
                "zeta({s}) = {got}, expected {expected}, diff = {diff}"
            );
        }
    }

    #[test]
    fn hurwitz_zeta_nan_inputs() {
        assert!(hurwitz_zeta(f64::NAN, 1.0).is_nan());
        // a = -1 is a nonpositive-integer pole => +inf (scipy.special.zeta), not
        // NaN; negative NON-integer a with non-integer s is the NaN case.
        assert!(hurwitz_zeta(2.0, -1.0).is_infinite());
        assert!(hurwitz_zeta(2.5, -0.5).is_nan());
    }

    #[test]
    fn hurwitz_zeta_pole() {
        assert!(hurwitz_zeta(1.0, 1.0).is_infinite());
    }

    // ── sici tests ───────────────────────────────────────────────────

    #[test]
    fn sici_zero() {
        let (si, ci) = sici(0.0);
        assert_eq!(si, 0.0);
        assert!(ci.is_infinite() && ci < 0.0); // Ci(0) = -∞
    }

    #[test]
    fn sici_small_positive() {
        // Si(1) ≈ 0.9460831, Ci(1) ≈ 0.3374039
        let (si, ci) = sici(1.0);
        assert!((si - 0.9460831).abs() < 1e-5, "Si(1) = {si}");
        assert!((ci - 0.3374039).abs() < 1e-5, "Ci(1) = {ci}");
    }

    #[test]
    fn sici_large() {
        // Si(10) ≈ 1.6583475, Ci(10) ≈ -0.04545643
        let (si, ci) = sici(10.0);
        assert!((si - 1.6583475).abs() < 1e-5, "Si(10) = {si}");
        assert!((ci - (-0.04545643)).abs() < 1e-5, "Ci(10) = {ci}");
    }

    #[test]
    fn sici_negative() {
        // Si(-x) = -Si(x)
        let (si_pos, ci_pos) = sici(2.0);
        let (si_neg, ci_neg) = sici(-2.0);
        assert!((si_neg + si_pos).abs() < 1e-10);
        assert!((ci_neg - ci_pos).abs() < 1e-10);
    }

    // ── shichi tests ─────────────────────────────────────────────────

    #[test]
    fn shichi_zero() {
        let (shi, chi) = shichi(0.0);
        assert_eq!(shi, 0.0);
        assert!(chi.is_infinite() && chi < 0.0); // Chi(0) = -∞
    }

    #[test]
    fn shichi_small_positive() {
        // Shi(1) ≈ 1.0572509, Chi(1) ≈ 0.8378669
        let (shi, chi) = shichi(1.0);
        assert!((shi - 1.0572509).abs() < 1e-5, "Shi(1) = {shi}");
        assert!((chi - 0.8378669).abs() < 1e-5, "Chi(1) = {chi}");
    }

    #[test]
    fn shichi_negative() {
        // Shi(-x) = -Shi(x), Chi(-x) = Chi(x)
        let (shi_pos, chi_pos) = shichi(2.0);
        let (shi_neg, chi_neg) = shichi(-2.0);
        assert!((shi_neg + shi_pos).abs() < 1e-10);
        assert!((chi_neg - chi_pos).abs() < 1e-10);
    }

    // ── hyp0f1 tests ─────────────────────────────────────────────────

    #[test]
    fn hyp0f1_at_zero() {
        // 0F1(; b; 0) = 1 for any b
        let result = hyp0f1_scalar(2.0, 0.0, RuntimeMode::Strict).unwrap();
        assert_eq!(result, 1.0);
    }

    #[test]
    fn hyp0f1_small_positive() {
        // 0F1(; 1; z) = I_0(2*sqrt(z)) for z > 0
        // At z = 1: 0F1(; 1; 1) = I_0(2) ≈ 2.2795853
        let result = hyp0f1_scalar(1.0, 1.0, RuntimeMode::Strict).unwrap();
        assert!((result - 2.2795853).abs() < 0.01, "0F1(;1;1) = {result}");
    }

    #[test]
    fn hyp0f1_negative_z() {
        // 0F1(; 1; -z²/4) = J_0(z) for integer order
        // 0F1(; 1; -1) = J_0(2) ≈ 0.2238907
        let result = hyp0f1_scalar(1.0, -1.0, RuntimeMode::Strict).unwrap();
        assert!((result - 0.2238907).abs() < 0.01, "0F1(;1;-1) = {result}");
    }

    #[test]
    fn hyp0f1_pole() {
        // 0F1(; 0; z) is undefined (pole at b=0)
        let result = hyp0f1_scalar(0.0, 1.0, RuntimeMode::Strict).unwrap();
        assert!(result.is_nan());
    }

    #[test]
    fn hyp0f1_hardened_pole() {
        // In hardened mode, pole should return error
        let result = hyp0f1_scalar(-1.0, 1.0, RuntimeMode::Hardened);
        assert!(result.is_err());
    }

    #[test]
    fn i0_matches_scipy_reference() {
        // scipy.special.i0(0) = 1.0
        assert!((i0_scalar(0.0) - 1.0).abs() < 1e-14);
        // scipy.special.i0(1) ≈ 1.2660658777520082
        assert!((i0_scalar(1.0) - 1.2660658777520082).abs() < 1e-10);
        // scipy.special.i0(2) ≈ 2.2795853023360673
        assert!((i0_scalar(2.0) - 2.2795853023360673).abs() < 1e-10);
    }

    #[test]
    fn i1_matches_scipy_reference() {
        // scipy.special.i1(0) = 0.0
        assert!(i1_scalar(0.0).abs() < 1e-14);
        // scipy.special.i1(1) ≈ 0.5651591039924851
        assert!((i1_scalar(1.0) - 0.5651591039924851).abs() < 1e-10);
        // scipy.special.i1(2) ≈ 1.5906368546373291
        assert!((i1_scalar(2.0) - 1.590_636_854_637_329_5_f64).abs() < 1e-10);
    }

    #[test]
    fn i0_i1_negative_argument() {
        // I_0(-x) = I_0(x), I_1(-x) = -I_1(x)
        assert!((i0_scalar(-1.0) - i0_scalar(1.0)).abs() < 1e-14);
        assert!((i1_scalar(-1.0) + i1_scalar(1.0)).abs() < 1e-14);
    }

    #[test]
    fn erf_matches_scipy_reference_values() {
        // scipy.special.erf(1) = 0.8427007929497148
        assert!((erf_scalar(1.0) - 0.8427007929497148).abs() < 1e-10);
        // scipy.special.erf(0) = 0.0
        assert!(erf_scalar(0.0).abs() < 1e-14);
        // scipy.special.erf(-1) = -0.8427007929497148
        assert!((erf_scalar(-1.0) + 0.8427007929497148).abs() < 1e-10);
    }

    #[test]
    fn erfc_matches_scipy_reference_values() {
        // scipy.special.erfc(1) = 0.15729920705028516
        assert!((erfc_scalar(1.0) - 0.15729920705028516).abs() < 1e-10);
        // scipy.special.erfc(0) = 1.0
        assert!((erfc_scalar(0.0) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn digamma_matches_scipy_reference_values() {
        // scipy.special.digamma(5) = 1.5061176684318003
        let result = digamma_scalar(5.0);
        assert!(
            (result - 1.5061176684318003).abs() < 1e-6,
            "digamma(5) got {result}, expected 1.5061176684318003"
        );
        // scipy.special.digamma(1) = -0.5772156649015329 (Euler-Mascheroni)
        let result1 = digamma_scalar(1.0);
        assert!(
            (result1 + 0.5772156649015329).abs() < 1e-6,
            "digamma(1) got {result1}, expected -0.5772156649015329"
        );
    }

    #[test]
    fn betaln_matches_scipy_reference_values() {
        // scipy.special.betaln(2, 3) = -2.4849066497880004
        let result = betaln_scalar(2.0, 3.0, RuntimeMode::Strict).unwrap();
        assert!((result + 2.4849066497880004).abs() < 1e-10);
    }

    #[test]
    fn betainc_matches_scipy_reference_values() {
        // scipy.special.betainc(2, 3, 0.5) = 0.6875
        let result = betainc_scalar(2.0, 3.0, 0.5, RuntimeMode::Strict).unwrap();
        assert!((result - 0.6875).abs() < 1e-10);
    }

    #[test]
    fn softmax_matches_scipy_reference_values() {
        // scipy.special.softmax([1, 2, 3])
        let result = softmax(&[1.0, 2.0, 3.0]);
        let expected = [0.09003057317038046, 0.24472847105479764, 0.6652409557748218];
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "softmax[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn log_softmax_matches_scipy_reference_values() {
        // scipy.special.log_softmax([1, 2, 3])
        let result = log_softmax(&[1.0, 2.0, 3.0]);
        let expected = [
            -2.4076059644443806,
            -1.4076059644443804,
            -0.4076059644443804,
        ];
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "log_softmax[{i}] got {got}, expected {want}"
            );
        }
    }

    #[test]
    fn gammaln_scalar_matches_scipy_reference_values() {
        // scipy.special.gammaln([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
        use fsci_runtime::RuntimeMode;
        let inputs = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0];
        let expected = [
            0.5723649429247,
            0.0,
            -0.12078223763524526,
            0.0,
            std::f64::consts::LN_2,
            1.791759469228055,
            3.1780538303479458,
        ];
        for (i, (&x, &want)) in inputs.iter().zip(expected.iter()).enumerate() {
            let got =
                gammaln_scalar(x, RuntimeMode::Strict).expect("gammaln_scalar should succeed");
            assert!(
                (got - want).abs() < 1e-10,
                "gammaln({x}) got {got}, expected {want} at index {i}"
            );
        }
    }

    #[test]
    fn gammainc_scalar_matches_scipy_reference_values() {
        // scipy.special.gammainc(a, x) for (a, x) in [(1, 1), (2, 1), (3, 2), (0.5, 0.5)]
        use fsci_runtime::RuntimeMode;
        let inputs = [(1.0, 1.0), (2.0, 1.0), (3.0, 2.0), (0.5, 0.5)];
        let expected = [
            0.6321205588285577,
            0.2642411176571153,
            0.32332358381693654,
            0.6826894921370859,
        ];
        for (i, ((a, x), &want)) in inputs.iter().zip(expected.iter()).enumerate() {
            let got = gammainc_scalar(*a, *x, RuntimeMode::Strict)
                .expect("gammainc_scalar should succeed");
            assert!(
                (got - want).abs() < 1e-10,
                "gammainc({a}, {x}) got {got}, expected {want} at index {i}"
            );
        }
    }

    #[test]
    fn gammaincc_scalar_matches_scipy_reference_values() {
        // scipy.special.gammaincc(a, x) for (a, x) in [(1, 1), (2, 1), (3, 2), (0.5, 0.5)]
        use fsci_runtime::RuntimeMode;
        let inputs = [(1.0, 1.0), (2.0, 1.0), (3.0, 2.0), (0.5, 0.5)];
        let expected = [
            0.36787944117144245,
            0.7357588823428847,
            0.6766764161830634,
            0.31731050786291115,
        ];
        for (i, ((a, x), &want)) in inputs.iter().zip(expected.iter()).enumerate() {
            let got = gammaincc_scalar(*a, *x, RuntimeMode::Strict)
                .expect("gammaincc_scalar should succeed");
            assert!(
                (got - want).abs() < 1e-10,
                "gammaincc({a}, {x}) got {got}, expected {want} at index {i}"
            );
        }
    }

    #[test]
    fn ndtr_scalar_matches_scipy_reference_values() {
        // scipy.special.ndtr([-2, -1, 0, 1, 2])
        let inputs = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let expected = [
            0.022750131948179195,
            0.15865525393145707,
            0.5,
            0.8413447460685429,
            0.9772498680518208,
        ];
        for (i, (&x, &want)) in inputs.iter().zip(expected.iter()).enumerate() {
            let got = ndtr_scalar(x);
            assert!(
                (got - want).abs() < 1e-10,
                "ndtr({x}) got {got}, expected {want} at index {i}"
            );
        }
    }

    #[test]
    fn ndtri_scalar_matches_scipy_reference_values() {
        // scipy.special.ndtri([0.01, 0.1, 0.5, 0.9, 0.99])
        let inputs = [0.01, 0.1, 0.5, 0.9, 0.99];
        let expected = [
            -2.3263478740408408,
            -1.2815515655446004,
            0.0,
            1.2815515655446004,
            2.3263478740408408,
        ];
        for (i, (&p, &want)) in inputs.iter().zip(expected.iter()).enumerate() {
            let got = ndtri_scalar(p);
            assert!(
                (got - want).abs() < 1e-10,
                "ndtri({p}) got {got}, expected {want} at index {i}"
            );
        }
    }

    #[test]
    fn zeta_matches_scipy_reference_values() {
        // scipy.special.zeta([2, 3, 4])
        let inputs = [2.0, 3.0, 4.0];
        let expected = [1.6449340668482264, 1.2020569031595942, 1.0823232337111381];
        for (i, (&s, &want)) in inputs.iter().zip(expected.iter()).enumerate() {
            let got = zeta_scalar(s);
            assert!(
                (got - want).abs() < 1e-10,
                "zeta({s}) got {got}, expected {want} at index {i}"
            );
        }
    }

    #[test]
    fn cbrt_matches_scipy_reference_values() {
        // scipy.special.cbrt([1, 8, 27, -8])
        let inputs = [1.0, 8.0, 27.0, -8.0];
        let expected = [1.0, 2.0, 3.0, -2.0];
        for (i, (&x, &want)) in inputs.iter().zip(expected.iter()).enumerate() {
            let got = cbrt(x);
            assert!(
                (got - want).abs() < 1e-10,
                "cbrt({x}) got {got}, expected {want} at index {i}"
            );
        }
    }
}
