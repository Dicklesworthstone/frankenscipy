#!/usr/bin/env python3
"""SciPy-backed oracle capture for FrankenSciPy special function packet fixtures.

Reads a conformance fixture JSON and emits a normalized oracle capture JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _decode_arg(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value.keys()) == {"re", "im"}:
            return complex(float(value["re"]), float(value["im"]))
        if set(value.keys()) == {"values"} and isinstance(value["values"], list):
            return [_decode_arg(item) for item in value["values"]]
        return {key: _decode_arg(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_decode_arg(item) for item in value]
    return value


def _as_float(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _complex_payload(value: Any) -> Dict[str, float]:
    if hasattr(value, "item"):
        value = value.item()
    return {"re": float(value.real), "im": float(value.imag)}


def _normalize_value(value: Any, np: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [_normalize_value(item, np) for item in value]
    if isinstance(value, tuple):
        return [_normalize_value(item, np) for item in value]
    if np.iscomplexobj(value):
        return _complex_payload(value)
    return _as_float(value)


def _result_payload(case_id: str, result: Any, np: Any) -> Dict[str, Any]:
    if isinstance(result, tuple):
        values = [_normalize_value(value, np) for value in result]
        return {
            "case_id": case_id,
            "status": "ok",
            "result_kind": "tuple",
            "result": {"values": values},
            "error": None,
        }

    if np.isscalar(result) or (hasattr(result, "ndim") and result.ndim == 0):
        if np.iscomplexobj(result):
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "complex_scalar",
                "result": {"value": _complex_payload(result)},
                "error": None,
            }
        return {
            "case_id": case_id,
            "status": "ok",
            "result_kind": "scalar",
            "result": {"value": _as_float(result)},
            "error": None,
        }

    values = _normalize_value(result, np)
    if np.iscomplexobj(result):
        return {
            "case_id": case_id,
            "status": "ok",
            "result_kind": "complex_vector",
            "result": {"values": values},
            "error": None,
        }
    return {
        "case_id": case_id,
        "status": "ok",
        "result_kind": "vector",
        "result": {"values": values},
        "error": None,
    }


def _run_case(case: Dict[str, Any], special: Any, np: Any) -> Dict[str, Any]:
    case_id = case["case_id"]
    function_name = case.get("function")
    args = [_decode_arg(arg) for arg in case.get("args", [])]

    try:
        def rel_erf_erfc_identity(x: Any) -> Any:
            x_arr = np.asarray(x)
            return special.erf(x_arr) + special.erfc(x_arr)

        def rel_gamma_recurrence(x: Any) -> Any:
            x_arr = np.asarray(x)
            return special.gamma(x_arr + 1.0) - x_arr * special.gamma(x_arr)

        def rel_beta_symmetry(a: Any, b: Any) -> Any:
            a_arr = np.asarray(a)
            b_arr = np.asarray(b)
            return special.beta(a_arr, b_arr) - special.beta(b_arr, a_arr)

        def rel_gammainc_complement(a: Any, x: Any) -> Any:
            a_arr = np.asarray(a)
            x_arr = np.asarray(x)
            return special.gammainc(a_arr, x_arr) + special.gammaincc(a_arr, x_arr)

        def btdtrc(a: Any, b: Any, x: Any) -> Any:
            a_arr = np.asarray(a)
            b_arr = np.asarray(b)
            x_arr = np.asarray(x)
            return special.betainc(b_arr, a_arr, 1.0 - x_arr)

        def rel_jn_recurrence(n: Any, x: Any) -> Any:
            n_val = np.asarray(n, dtype=float)
            x_arr = np.asarray(x)
            with np.errstate(divide="ignore", invalid="ignore"):
                denom = np.where(x_arr == 0, np.nan, x_arr)
                return (
                    special.jv(n_val - 1.0, x_arr)
                    - (2.0 * n_val / denom) * special.jv(n_val, x_arr)
                    + special.jv(n_val + 1.0, x_arr)
                )

        def scalar_item(value: Any) -> Any:
            arr = np.asarray(value)
            if hasattr(arr, "item"):
                try:
                    return arr.item()
                except ValueError:
                    return arr
            return arr

        def ellipj_component(component_index: int) -> Any:
            def inner(u: Any, m: Any) -> Any:
                return special.ellipj(np.asarray(u), np.asarray(m))[component_index]

            return inner

        def fresnel_component(component_index: int) -> Any:
            def inner(x: Any) -> Any:
                return special.fresnel(np.asarray(x))[component_index]

            return inner

        def unary_tuple_component(tuple_fn: Any, component_index: int) -> Any:
            def inner(x: Any) -> Any:
                return tuple_fn(np.asarray(x))[component_index]

            return inner

        def roots_component(root_fn: Any, component_index: int) -> Any:
            def inner(*raw_args: Any) -> Any:
                if len(raw_args) < 2:
                    raise ValueError(
                        "root component lookup requires parameters plus a component index"
                    )
                *shape_args, element_index = raw_args
                order = int(scalar_item(shape_args[0]))
                evaluated_args = [order, *(scalar_item(arg) for arg in shape_args[1:])]
                roots = root_fn(*evaluated_args)
                return roots[component_index][int(scalar_item(element_index))]

            return inner

        def fdtridfd_refined(dfn: Any, p: Any, x: Any) -> Any:
            d0 = special.fdtridfd(dfn, p, x)
            try:
                dfn_f = float(scalar_item(dfn))
                p_f = float(scalar_item(p))
                x_f = float(scalar_item(x))
                d0_f = float(scalar_item(d0))
                if (
                    np.isfinite(d0_f)
                    and 1e-99 < d0_f < 1e99
                    and 0.0 < p_f < 1.0
                    and x_f > 0.0
                ):
                    from scipy import optimize as _opt

                    lo = d0_f * (1.0 - 1e-7)
                    hi = d0_f * (1.0 + 1e-7)
                    flo = float(special.fdtr(dfn_f, lo, x_f)) - p_f
                    fhi = float(special.fdtr(dfn_f, hi, x_f)) - p_f
                    if np.isfinite(flo) and np.isfinite(fhi) and flo * fhi <= 0.0:
                        return _opt.brentq(
                            lambda d: float(special.fdtr(dfn_f, d, x_f)) - p_f,
                            lo,
                            hi,
                            xtol=1e-16,
                            rtol=1e-15,
                        )
            except Exception:
                pass
            return d0

        # Map function name to scipy.special function
        func_map = {
            "gamma": special.gamma,
            "gammaln": special.gammaln,
            "gammainc": special.gammainc,
            "gammaincc": special.gammaincc,
            "digamma": special.digamma,
            "polygamma": special.polygamma,
            "rgamma": special.rgamma,
            "loggamma": special.loggamma,
            "beta": special.beta,
            "betaln": special.betaln,
            "betainc": special.betainc,
            "betaincinv": special.betaincinv,
            "btdtr": special.betainc,
            "btdtrc": btdtrc,
            "btdtri": special.betaincinv,
            "btdtria": special.btdtria,
            "btdtrib": special.btdtrib,
            "erf": special.erf,
            "erfc": special.erfc,
            "erfinv": special.erfinv,
            "erfcinv": special.erfcinv,
            "erfcx": special.erfcx,
            "erfi": special.erfi,
            "dawsn": special.dawsn,
            "i0": special.i0,
            "i1": special.i1,
            "i0e": special.i0e,
            "i1e": special.i1e,
            "iv": special.iv,
            "ive": special.ive,
            "j0": special.j0,
            "j1": special.j1,
            "jn": special.jn,
            "jv": special.jv,
            "jvp": special.jvp,
            "jve": special.jve,
            "y0": special.y0,
            "y1": special.y1,
            "yn": special.yn,
            "yv": special.yv,
            "yvp": special.yvp,
            "yve": special.yve,
            "k0": special.k0,
            "k1": special.k1,
            "k0e": special.k0e,
            "k1e": special.k1e,
            "kv": special.kv,
            "kvp": special.kvp,
            "kve": special.kve,
            "ivp": special.ivp,
            "zeta": special.zeta,
            "zetac": special.zetac,
            "riemann_zeta": lambda x: special.zeta(x, 1),  # Riemann zeta
            "factorial": special.factorial,
            "factorial2": special.factorial2,
            "comb": special.comb,
            "perm": special.perm,
            "poch": special.poch,
            "multigammaln": special.multigammaln,
            "bdtr": special.bdtr,
            "bdtrc": special.bdtrc,
            "bdtri": special.bdtri,
            "chdtr": special.chdtr,
            "chdtrc": special.chdtrc,
            "chdtri": special.chdtri,
            "chdtriv": special.chdtriv,
            "fdtr": special.fdtr,
            "fdtrc": special.fdtrc,
            "fdtri": special.fdtri,
            "fdtridfd": fdtridfd_refined,
            "gdtr": special.gdtr,
            "gdtrc": special.gdtrc,
            "gdtrix": special.gdtrix,
            "gdtria": special.gdtria,
            "gdtrib": special.gdtrib,
            "nbdtr": special.nbdtr,
            "nbdtrc": special.nbdtrc,
            "nbdtri": special.nbdtri,
            "nrdtrimn": special.nrdtrimn,
            "nrdtrisd": special.nrdtrisd,
            "pdtr": special.pdtr,
            "pdtrc": special.pdtrc,
            "pdtri": special.pdtri,
            "pdtrik": special.pdtrik,
            "expit": special.expit,
            "logit": special.logit,
            "log_expit": special.log_expit,
            "log_ndtr": special.log_ndtr,
            "ndtr": special.ndtr,
            "ndtri": special.ndtri,
            "stdtr": special.stdtr,
            "stdtrc": lambda df, t: special.stdtr(df, -np.asarray(t)),
            "stdtridf": special.stdtridf,
            "stdtrit": special.stdtrit,
            "log1p": np.log1p,
            "expm1": np.expm1,
            "logaddexp": np.logaddexp,
            "logaddexp2": np.logaddexp2,
            "sinc": np.sinc,
            "softplus": special.softplus,
            "xlogx": lambda x: special.xlogy(x, x),
            "xlogy": special.xlogy,
            "xlog1py": special.xlog1py,
            "entr": special.entr,
            "rel_entr": special.rel_entr,
            "kl_div": special.kl_div,
            "huber": special.huber,
            "pseudo_huber": special.pseudo_huber,
            "softmax": special.softmax,
            "log_softmax": special.log_softmax,
            "logsumexp": special.logsumexp,
            "exp1": special.exp1,
            "expi": special.expi,
            "expn": special.expn,
            "exp10": special.exp10,
            "exp2": special.exp2,
            "exprel": special.exprel,
            "cbrt": special.cbrt,
            "radian": special.radian,
            "sindg": special.sindg,
            "cosdg": special.cosdg,
            "tandg": special.tandg,
            "cotdg": special.cotdg,
            "airy": special.airy,
            "airye": special.airye,
            "ai_zeros": special.ai_zeros,
            "bi_zeros": special.bi_zeros,
            "fresnel_c": fresnel_component(1),
            "fresnel_s": fresnel_component(0),
            "sici_si": unary_tuple_component(special.sici, 0),
            "sici_ci": unary_tuple_component(special.sici, 1),
            "shichi_shi": unary_tuple_component(special.shichi, 0),
            "shichi_chi": unary_tuple_component(special.shichi, 1),
            "hyp0f1": special.hyp0f1,
            "hyp1f1": special.hyp1f1,
            "hyp2f1": special.hyp2f1,
            "hankel1": special.hankel1,
            "hankel2": special.hankel2,
            "wright_bessel": special.wright_bessel,
            "spherical_jn": special.spherical_jn,
            "spherical_yn": special.spherical_yn,
            "spherical_in": special.spherical_in,
            "spherical_kn": special.spherical_kn,
            "ber": special.ber,
            "bei": special.bei,
            "ker": special.ker,
            "kei": special.kei,
            "struve": special.struve,
            "modstruve": special.modstruve,
            "eval_legendre": special.eval_legendre,
            "eval_chebyt": special.eval_chebyt,
            "eval_chebyu": special.eval_chebyu,
            "eval_hermite": special.eval_hermite,
            "eval_hermitenorm": special.eval_hermitenorm,
            "eval_laguerre": special.eval_laguerre,
            "eval_genlaguerre": special.eval_genlaguerre,
            "eval_jacobi": special.eval_jacobi,
            "eval_gegenbauer": special.eval_gegenbauer,
            "eval_sh_legendre": special.eval_sh_legendre,
            "eval_sh_chebyt": special.eval_sh_chebyt,
            "eval_sh_chebyu": special.eval_sh_chebyu,
            "roots_legendre": special.roots_legendre,
            "roots_chebyt": special.roots_chebyt,
            "roots_chebyu": special.roots_chebyu,
            "roots_hermite": special.roots_hermite,
            "roots_hermitenorm": special.roots_hermitenorm,
            "roots_laguerre": special.roots_laguerre,
            "roots_genlaguerre": special.roots_genlaguerre,
            "roots_jacobi": special.roots_jacobi,
            "roots_gegenbauer": special.roots_gegenbauer,
            "roots_legendre_node": roots_component(special.roots_legendre, 0),
            "roots_legendre_weight": roots_component(special.roots_legendre, 1),
            "roots_chebyt_node": roots_component(special.roots_chebyt, 0),
            "roots_chebyt_weight": roots_component(special.roots_chebyt, 1),
            "roots_chebyu_node": roots_component(special.roots_chebyu, 0),
            "roots_chebyu_weight": roots_component(special.roots_chebyu, 1),
            "roots_hermite_node": roots_component(special.roots_hermite, 0),
            "roots_hermite_weight": roots_component(special.roots_hermite, 1),
            "roots_hermitenorm_node": roots_component(special.roots_hermitenorm, 0),
            "roots_hermitenorm_weight": roots_component(special.roots_hermitenorm, 1),
            "roots_laguerre_node": roots_component(special.roots_laguerre, 0),
            "roots_laguerre_weight": roots_component(special.roots_laguerre, 1),
            "roots_genlaguerre_node": roots_component(special.roots_genlaguerre, 0),
            "roots_genlaguerre_weight": roots_component(special.roots_genlaguerre, 1),
            "roots_jacobi_node": roots_component(special.roots_jacobi, 0),
            "roots_jacobi_weight": roots_component(special.roots_jacobi, 1),
            "roots_gegenbauer_node": roots_component(special.roots_gegenbauer, 0),
            "roots_gegenbauer_weight": roots_component(special.roots_gegenbauer, 1),
            "lpmv": special.lpmv,
            "sph_harm_y": special.sph_harm_y,
            "rel_erf_erfc_identity": rel_erf_erfc_identity,
            "rel_erfinv_composition": lambda x: special.erfinv(special.erf(np.asarray(x))),
            "rel_gamma_recurrence": rel_gamma_recurrence,
            "rel_beta_symmetry": rel_beta_symmetry,
            "rel_gammainc_complement": rel_gammainc_complement,
            "rel_jn_recurrence": rel_jn_recurrence,
            "ellipk": special.ellipk,
            "ellipkm1": special.ellipkm1,
            "ellipe": special.ellipe,
            "ellipj": special.ellipj,
            "ellipj_sn": ellipj_component(0),
            "ellipj_cn": ellipj_component(1),
            "ellipj_dn": ellipj_component(2),
            "ellipj_ph": ellipj_component(3),
            "ellipkinc": special.ellipkinc,
            "ellipeinc": special.ellipeinc,
            "elliprc": special.elliprc,
            "elliprd": special.elliprd,
            "elliprf": special.elliprf,
            "elliprg": special.elliprg,
            "elliprj": special.elliprj,
            "lambertw": special.lambertw,
            "wrightomega": special.wrightomega,
            "hurwitz_zeta": lambda x, q: special.zeta(x, q),
            "owens_t": special.owens_t,
            "boxcox": special.boxcox,
            "boxcox1p": special.boxcox1p,
            "inv_boxcox": special.inv_boxcox,
            "inv_boxcox1p": special.inv_boxcox1p,
            "kolmogorov": special.kolmogorov,
            "kolmogi": special.kolmogi,
            "spence": special.spence,
        }

        if function_name not in func_map:
            return {
                "case_id": case_id,
                "status": "error",
                "result_kind": "unsupported_function",
                "result": {},
                "error": f"unsupported function: {function_name}",
            }

        func = func_map[function_name]
        if function_name in {"logsumexp", "softmax", "log_softmax"}:
            if len(args) == 1 and isinstance(args[0], (list, np.ndarray)):
                result = func(np.asarray(args[0], dtype=float))
            else:
                result = func(np.asarray([scalar_item(a) for a in args], dtype=float))
            return _result_payload(case_id, result, np)

        def _int_if_integral(val: Any) -> Any:
            s = scalar_item(val)
            if np.isfinite(s) and float(s).is_integer():
                return int(round(s))
            return val

        if function_name in {
            "polygamma",
            "expn",
            "eval_legendre",
            "eval_chebyt",
            "eval_chebyu",
            "eval_hermite",
            "eval_hermitenorm",
            "eval_laguerre",
            "eval_genlaguerre",
            "eval_jacobi",
            "eval_gegenbauer",
            "eval_sh_legendre",
            "eval_sh_chebyt",
            "eval_sh_chebyu",
            "spherical_jn",
            "spherical_yn",
            "spherical_in",
            "spherical_kn",
            "jn",
            "yn",
            "kn",
            "factorial",
            "factorial2",
            "bdtr",
            "bdtrc",
            "bdtri",
            "nbdtr",
            "nbdtrc",
            "nbdtri",
            "pdtr",
            "pdtrc",
            "pdtri",
            "stdtr",
            "stdtrit",
            "smirnov",
            "smirnovi",
        } and len(args) >= 1:
            args = [_int_if_integral(args[0]), *args[1:]]
        elif function_name in {"multigammaln", "lambertw"} and len(args) >= 2:
            args = [args[0], _int_if_integral(args[1]), *args[2:]]
        elif function_name in {"comb", "perm", "lpmv", "sph_harm_y"} and len(args) >= 2:
            args = [_int_if_integral(args[0]), _int_if_integral(args[1]), *args[2:]]
        elif function_name in {"jvp", "yvp", "ivp", "kvp"} and len(args) >= 3:
            args = [args[0], args[1], _int_if_integral(args[2]), *args[3:]]

        result = func(*args)
        return _result_payload(case_id, result, np)

    except (ArithmeticError, FloatingPointError, OverflowError, TypeError, ValueError) as exc:
        return {
            "case_id": case_id,
            "status": "error",
            "result_kind": "exception",
            "result": {},
            "error": str(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture SciPy special oracle outputs")
    parser.add_argument("--fixture", required=True, help="Input packet fixture JSON path")
    parser.add_argument("--output", required=True, help="Output oracle capture JSON path")
    parser.add_argument("--oracle-root", required=False, default="",
                        help="(unused) legacy oracle root path — kept for CLI backwards compat")
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    output_path = Path(args.output)

    try:
        import numpy as np
        from scipy import special
    except ModuleNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in fixture: {exc}", file=sys.stderr)
        return 1

    case_outputs: List[Dict[str, Any]] = []

    for case in fixture.get("cases", []):
        case_outputs.append(_run_case(case, special=special, np=np))

    payload = {
        "packet_id": fixture.get("packet_id", "unknown"),
        "family": fixture.get("family", "unknown"),
        "generated_unix_ms": int(time.time() * 1000),
        "runtime": {
            "python_version": sys.version.split()[0],
            "numpy_version": getattr(np, "__version__", "unknown"),
            "scipy_version": getattr(sys.modules.get("scipy"), "__version__", "unknown"),
        },
        "case_outputs": case_outputs,
    }

    # Scrub non-JSON floats (Infinity, -Infinity, NaN) to serde-parseable
    # values BEFORE writing — Python's `json.dump(allow_nan=True)` (the
    # default) emits the literal tokens `Infinity` / `-Infinity` / `NaN`
    # which are NOT valid JSON and serde_json rejects them. Use the
    # f64-escape strings that serde-json's `f64` deserializer accepts via
    # `Number::from_f64(...).unwrap_or(...)` round-trip — `null` is the
    # safe portable sentinel; downstream code already handles None.
    import math as _math
    def _scrub(value):
        if isinstance(value, float):
            if _math.isnan(value) or _math.isinf(value):
                # Encode as a tagged object so the Rust side can recover
                # the sign of infinity if needed; for current consumers
                # `null` would also work, but tagged is friendlier for
                # debugging the oracle output by hand.
                if _math.isnan(value):
                    return {"__float__": "nan"}
                return {"__float__": "+inf" if value > 0 else "-inf"}
            return value
        if isinstance(value, dict):
            return {k: _scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_scrub(v) for v in value]
        return value

    cleaned = _scrub(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(cleaned, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
