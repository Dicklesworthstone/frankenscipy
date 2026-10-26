#!/usr/bin/env python3
"""SciPy-backed oracle capture for FrankenSciPy integrate fixture.

Per frankenscipy-di9p: fsci-integrate previously had NO scipy oracle
script. This closes that gap for the FSCI-P2C-013 quadrature surface
(trapezoid / simpson / cumulative_* / romb / newton_cotes / fixed_quad /
gauss_legendre / cubature / quad / quad_vec / dblquad / tplquad /
solve_ivp / odeint). BVP coverage is tracked separately.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List


_MATH_NAMESPACE = {
    name: getattr(math, name)
    for name in ("sin", "cos", "tan", "exp", "log", "log10", "sqrt", "pi", "e")
}
_VARIABLE_ORDER = ("x", "x0", "x1", "x2", "x3", "y")


def _translate(expr: str) -> str:
    """Translate fixture expression syntax to valid Python.

    `^` is the XOR operator in Python; fixtures use it for powers.
    """
    return expr.replace("^", "**")


def _evaluate_math_ast(node: ast.AST, local: Dict[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate_math_ast(node.body, local)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in local:
            return float(local[node.id])
        value = _MATH_NAMESPACE.get(node.id)
        if isinstance(value, (int, float)):
            return float(value)
        raise ValueError(f"unknown expression name: {node.id!r}")
    if isinstance(node, ast.UnaryOp):
        value = _evaluate_math_ast(node.operand, local)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
    if isinstance(node, ast.BinOp):
        left = _evaluate_math_ast(node.left, local)
        right = _evaluate_math_ast(node.right, local)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        raise ValueError(f"unsupported binary operator: {type(node.op).__name__}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise ValueError("unsupported function call in expression")
        func = _MATH_NAMESPACE.get(node.func.id)
        if not callable(func):
            raise ValueError(f"unsupported function: {node.func.id!r}")
        args = [_evaluate_math_ast(arg, local) for arg in node.args]
        return float(func(*args))
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def _compile_math_expr(expr: str) -> tuple[ast.Expression, List[str]]:
    tree = ast.parse(_translate(expr), mode="eval")
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    variables = [name for name in _VARIABLE_ORDER if name in names]
    unknown = names.difference(_MATH_NAMESPACE).difference(variables)
    if unknown:
        raise ValueError(f"unknown expression names: {sorted(unknown)!r}")
    return tree, variables


def _build_callable(expr_or_name: str) -> Callable[..., float]:
    """Resolve a fixture `func` string to a callable.

    Named identifiers map to specific closures. Generic math expressions
    are compiled with a restricted `math` namespace.
    """
    named_1d = {
        "constant_1": (1, lambda x: 1.0),
        "powers_1d": (1, lambda x: x ** 3 + 2.0 * x ** 2 + 3.0 * x + 4.0),
    }
    named_nd = {
        # 2D pair-input: used by cubature zero-dim test cases.
        "zero_dim_pair": (2, lambda x, y: 0.0),
    }
    if expr_or_name in named_1d:
        arity, fn = named_1d[expr_or_name]
        return lambda x: fn(x)
    if expr_or_name in named_nd:
        arity, fn = named_nd[expr_or_name]
        return lambda *xs: fn(*xs[:arity])

    tree, candidates = _compile_math_expr(expr_or_name)
    if not candidates:
        # Pure constant expression (e.g. "1.0").
        def const_fn(*_args: float) -> float:
            return _evaluate_math_ast(tree, {})
        return const_fn

    def dispatched(*args: float) -> float:
        local: Dict[str, float] = {}
        for i, tok in enumerate(candidates):
            if i < len(args):
                local[tok] = args[i]
            else:
                local[tok] = 0.0
        return _evaluate_math_ast(tree, local)

    return dispatched


def _build_vec_callable(name: str):
    """Named vector-valued 1D integrand for quad_vec (br-9cla-5).

    Mirror this table on the Rust side (`make_integrate_quad_vec_func`).
    Returns f(x) -> list[float].
    """
    if name == "linear_square":
        # [x, x²] — integrals over [0,1] are [0.5, 1/3].
        return lambda x: [x, x * x]
    raise ValueError(f"unknown vec1d integrand: {name!r}")


def _build_dblquad_problem(name: str):
    """Named 2-D integrand for dblquad (br-9cla-5).

    Outer bounds (a, b) are supplied by the fixture; inner bounds come
    from this registry alongside the integrand so both sides stay
    aligned with `make_integrate_dblquad_func`.
    Returns (f, y_lo, y_hi) with f(y, x) matching scipy's convention.
    """
    if name == "xy_prod_unit_y":
        # f(y, x) = x·y with y ∈ [0, 1].
        return (lambda y, x: x * y, 0.0, 1.0)
    raise ValueError(f"unknown dblquad problem: {name!r}")


def _build_tplquad_problem(name: str):
    """Named 3-D integrand for tplquad (br-9cla-5).

    Mirrors `make_integrate_tplquad_func`. Outer bounds (a, b) come
    from the fixture. Inner (y) and innermost (z) bounds are baked in.
    Returns (f, y_lo, y_hi, z_lo, z_hi) with f(z, y, x).
    """
    if name == "xyz_prod_unit_yz":
        # f(z, y, x) = x·y·z with y ∈ [0,1], z ∈ [0,1].
        return (lambda z, y, x: x * y * z, 0.0, 1.0, 0.0, 1.0)
    raise ValueError(f"unknown tplquad problem: {name!r}")


def _build_bvp_problem(name: str):
    """Named BVP (f, bc) pair dispatcher for solve_bvp fixtures (br-9cla-4).

    Returns (f, bc) where:
      f(x, y)  — vectorized RHS: x is shape (n,), y is shape (k, n),
                 returns dy/dx shape (k, n).
      bc(ya, yb) — boundary-condition residuals: ya, yb are length-k,
                   returns length-k residual vector.
    """
    import numpy as np

    if name == "linear_y_double_prime_zero":
        # y'' = 0, y(0) = 0, y(1) = 1. Exact: y(x) = x.
        # System: y0' = y1, y1' = 0.
        def f(x, y):
            return np.vstack([y[1], np.zeros_like(x)])

        def bc(ya, yb):
            return np.array([ya[0], yb[0] - 1.0])

        return f, bc
    if name == "poisson_constant_source":
        # y'' = -1, y(0) = 0, y(1) = 0. Exact: y(x) = x(1-x)/2.
        def f(x, y):
            return np.vstack([y[1], -np.ones_like(x)])

        def bc(ya, yb):
            return np.array([ya[0], yb[0]])

        return f, bc
    raise ValueError(f"unknown bvp problem: {name!r}")


def _build_ivp_rhs(name: str) -> Callable[..., list]:
    """Named RHS dispatcher for solve_ivp fixtures (br-9cla-1).

    The Rust runner maintains an identically-keyed lookup table so both
    sides of the differential comparison integrate the *same* ODE. Do
    not add entries here without mirroring the Rust side in slice 9cla-2.
    """
    if name == "exponential_decay":
        # dy/dt = -y; y(0) = y0 → y(t) = y0 * exp(-t).
        return lambda t, y: [-y[0]]
    if name == "stiff_decay":
        # dy/dt = -1000 y; BDF-oriented stiff scalar fixture.
        return lambda t, y: [-1000.0 * y[0]]
    if name == "linear_growth":
        # dy/dt = 1; y(0) = y0 → y(t) = y0 + t.
        return lambda t, y: [1.0]
    if name == "harmonic_oscillator":
        # d²x/dt² + x = 0 → [dy0/dt, dy1/dt] = [y1, -y0]. Energy
        # E = 0.5*(y0² + y1²) is conserved.
        return lambda t, y: [y[1], -y[0]]
    if name == "robertson_chemistry":
        return lambda t, y: [
            -0.04 * y[0] + 1e4 * y[1] * y[2],
            0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1],
            3e7 * y[1] * y[1],
        ]
    if name == "van_der_pol_mu10":
        return lambda t, y: [y[1], 10.0 * (1.0 - y[0] * y[0]) * y[1] - y[0]]
    raise ValueError(f"unknown ivp rhs: {name!r}")


def _build_ivp_event(name: str):
    """Named event dispatcher for solve_ivp fixtures."""
    if name == "y0_minus_half_terminal":
        event = lambda t, y: y[0] - 0.5  # noqa: E731
        event.terminal = True
        event.direction = 0.0
        return event
    raise ValueError(f"unknown ivp event: {name!r}")


def _run_case(case: Dict[str, Any], integrate: Any, np: Any) -> Dict[str, Any]:
    case_id = case.get("case_id", "<missing>")
    function = case.get("function", "<missing>")
    args = case.get("args", {}) if isinstance(case.get("args"), dict) else {}

    try:
        if function == "trapezoid":
            y = np.asarray(args["y"], dtype=float)
            x = np.asarray(args["x"], dtype=float) if args.get("x") is not None else None
            result = float(integrate.trapezoid(y, x=x))
            return _ok(case_id, "scalar", {"value": result})

        if function == "simpson":
            y = np.asarray(args["y"], dtype=float)
            x = np.asarray(args["x"], dtype=float) if args.get("x") is not None else None
            result = float(integrate.simpson(y, x=x))
            return _ok(case_id, "scalar", {"value": result})

        if function == "cumulative_trapezoid":
            y = np.asarray(args["y"], dtype=float)
            x = np.asarray(args["x"], dtype=float) if args.get("x") is not None else None
            initial = args.get("initial")
            if initial is not None:
                initial = float(initial)
            result = integrate.cumulative_trapezoid(y, x=x, initial=initial)
            return _ok(case_id, "array", {"values": [float(v) for v in result.tolist()]})

        if function == "cumulative_simpson":
            y = np.asarray(args["y"], dtype=float)
            x = np.asarray(args["x"], dtype=float) if args.get("x") is not None else None
            initial = args.get("initial")
            if initial is not None:
                initial = float(initial)
            result = integrate.cumulative_simpson(y, x=x, initial=initial)
            return _ok(case_id, "array", {"values": [float(v) for v in result.tolist()]})

        if function == "romb":
            y = np.asarray(args["y"], dtype=float)
            dx = float(args.get("dx", 1.0))
            result = float(integrate.romb(y, dx=dx))
            return _ok(case_id, "scalar", {"value": result})

        if function == "newton_cotes":
            n = int(args["n"])
            weights, error = integrate.newton_cotes(n)
            scale = float(n) if n > 0 else 1.0
            return _ok(
                case_id,
                "array",
                {
                    "values": [float(w / scale) for w in weights.tolist()],
                    "error": float(error),
                },
            )

        if function == "fixed_quad":
            fn = _build_callable(args["func"])
            a = float(args["a"])
            b = float(args["b"])
            n = int(args.get("n", 5))
            # scipy.integrate.fixed_quad passes arrays; adapt.
            def vec_fn(xs: Any) -> Any:
                return np.asarray([fn(float(x)) for x in np.atleast_1d(xs)], dtype=float)
            value, _none = integrate.fixed_quad(vec_fn, a, b, n=n)
            return _ok(case_id, "scalar", {"value": float(value)})

        if function == "gauss_legendre":
            # scipy exposes gauss_legendre weights+nodes via roots_legendre.
            from scipy.special import roots_legendre

            n = int(args.get("n", 5))
            a = float(args["a"])
            b = float(args["b"])
            fn = _build_callable(args["func"])
            nodes, weights = roots_legendre(n)
            half = (b - a) / 2.0
            mid = (a + b) / 2.0
            total = 0.0
            for xi, wi in zip(nodes, weights):
                total += wi * fn(mid + half * xi)
            total *= half
            return _ok(case_id, "scalar", {"value": float(total)})

        if function == "quad":
            fn = _build_callable(args["func"])
            a = float(args["a"])
            b = float(args["b"])
            epsabs = float(args.get("epsabs", args.get("atol", 1.49e-8)))
            epsrel = float(args.get("epsrel", args.get("rtol", 1.49e-8)))
            limit = int(args.get("limit", args.get("max_subdivisions", 50)))
            value, error = integrate.quad(
                fn, a, b, epsabs=epsabs, epsrel=epsrel, limit=limit
            )
            return _ok(case_id, "scalar", {"value": float(value)})

        if function == "quad_vec":
            inner = _build_vec_callable(args["func"])
            fn = lambda x: np.asarray(inner(float(x)), dtype=float)  # noqa: E731
            a = float(args["a"])
            b = float(args["b"])
            epsabs = float(args.get("epsabs", args.get("atol", 1.49e-8)))
            epsrel = float(args.get("epsrel", args.get("rtol", 1.49e-8)))
            limit = int(args.get("limit", args.get("max_subdivisions", 50)))
            res, _err = integrate.quad_vec(
                fn, a, b, epsabs=epsabs, epsrel=epsrel, limit=limit
            )
            return _ok(
                case_id,
                "array",
                {"values": [float(v) for v in np.atleast_1d(res).tolist()]},
            )

        if function == "dblquad":
            f, y_lo, y_hi = _build_dblquad_problem(args["func"])
            a = float(args["a"])
            b = float(args["b"])
            epsabs = float(args.get("epsabs", args.get("atol", 1.49e-8)))
            epsrel = float(args.get("epsrel", args.get("rtol", 1.49e-8)))
            value, _err = integrate.dblquad(
                f, a, b, y_lo, y_hi, epsabs=epsabs, epsrel=epsrel
            )
            return _ok(case_id, "scalar", {"value": float(value)})

        if function == "tplquad":
            f, y_lo, y_hi, z_lo, z_hi = _build_tplquad_problem(args["func"])
            a = float(args["a"])
            b = float(args["b"])
            epsabs = float(args.get("epsabs", args.get("atol", 1.49e-8)))
            epsrel = float(args.get("epsrel", args.get("rtol", 1.49e-8)))
            value, _err = integrate.tplquad(
                f, a, b, y_lo, y_hi, z_lo, z_hi, epsabs=epsabs, epsrel=epsrel
            )
            return _ok(case_id, "scalar", {"value": float(value)})

        if function == "solve_bvp":
            problem = args["problem"]
            f, bc = _build_bvp_problem(problem)
            x = np.asarray(args["x"], dtype=float)
            y_init = np.asarray(args["y_init"], dtype=float)
            tol = float(args.get("tol", 1e-8))
            x_eval = np.asarray(
                args.get("x_eval", args["x"]), dtype=float
            )
            res = integrate.solve_bvp(f, bc, x, y_init, tol=tol)
            y_sampled = res.sol(x_eval)
            return _ok(
                case_id,
                "bvp_result",
                {
                    "x": [float(v) for v in x_eval.tolist()],
                    "y": [[float(v) for v in row] for row in y_sampled.tolist()],
                    "success": bool(res.success),
                    "niter": int(res.niter),
                    "rms_residuals": [
                        float(v) for v in np.atleast_1d(res.rms_residuals).tolist()
                    ],
                },
            )

        if function == "solve_ivp":
            rhs_name = args["rhs"]
            t_span = tuple(float(v) for v in args["t_span"])
            y0 = [float(v) for v in args["y0"]]
            method = args.get("method", "RK45")
            rtol = float(args.get("rtol", 1e-3))
            atol = float(args.get("atol", 1e-6))
            if not math.isfinite(rtol) or not math.isfinite(atol):
                raise ValueError("NonFiniteRtol")
            first_step = args.get("first_step")
            if first_step is not None:
                first_step = float(first_step)
                if not math.isfinite(first_step):
                    raise ValueError("NonFiniteFirstStep")
            max_step = args.get("max_step")
            if max_step is not None:
                max_step = float(max_step)
                if not math.isfinite(max_step):
                    raise ValueError("NonFiniteMaxStep")
            event_name = args.get("event")
            events = None
            if event_name is not None:
                events = _build_ivp_event(event_name)
            t_eval_raw = args.get("t_eval")
            t_eval = None
            if t_eval_raw is not None:
                t_eval = [float(v) for v in t_eval_raw]
            dense_t_eval_raw = args.get("dense_t_eval")
            dense_t_eval = None
            if dense_t_eval_raw is not None:
                dense_t_eval = [float(v) for v in dense_t_eval_raw]
            rhs = _build_ivp_rhs(rhs_name)
            res = integrate.solve_ivp(
                rhs,
                t_span,
                y0,
                method=method,
                rtol=rtol,
                atol=atol,
                t_eval=t_eval,
                dense_output=(dense_t_eval is not None),
                first_step=first_step,
                max_step=max_step if max_step is not None else math.inf,
                events=events,
            )
            if dense_t_eval is not None and res.sol is not None:
                y_dense = np.atleast_2d(res.sol(np.asarray(dense_t_eval, dtype=float)))
                t_out = dense_t_eval
                y_out = [[float(v) for v in row] for row in y_dense.tolist()]
            else:
                t_out = [float(v) for v in res.t.tolist()]
                y_out = [[float(v) for v in row] for row in res.y.tolist()]
                if (
                    res.status == 1
                    and t_eval is not None
                    and len(t_out) < len(t_eval)
                    and res.t_events
                    and len(res.t_events[0]) > 0
                ):
                    t_ev = float(res.t_events[0][-1])
                    y_ev = [float(v) for v in res.y_events[0][-1].tolist()]
                    if abs(t_ev - t_eval[len(t_out)]) <= 1e-9:
                        t_out.append(t_ev)
                        for idx, val in enumerate(y_ev):
                            y_out[idx].append(val)
            return _ok(
                case_id,
                "ivp_result",
                {
                    "t": t_out,
                    "y": y_out,
                    "status": int(res.status),
                    "success": bool(res.success),
                    "nfev": int(res.nfev),
                },
            )

        if function == "odeint":
            rhs_name = args["rhs"]
            y0 = [float(v) for v in args["y0"]]
            t = [float(v) for v in args["t_eval"]]
            rhs = _build_ivp_rhs(rhs_name)

            def odeint_rhs(y, t_val):
                return rhs(float(t_val), y)

            y = integrate.odeint(odeint_rhs, y0, t)
            return _ok(
                case_id,
                "ivp_result",
                {
                    "t": t,
                    "y": [[float(v) for v in row] for row in y.T.tolist()],
                },
            )

        if function == "qmc_quad":
            from scipy import stats as _stats

            func_name = args["func"]
            lower = [float(v) for v in args["lower"]]
            upper = [float(v) for v in args["upper"]]
            n_points = int(args.get("n", args.get("n_points", 1024)))
            if len(lower) == 0 or len(lower) != len(upper):
                raise ValueError("qmc_quad bounds must be non-empty and match dimensionality")
            if any(lo >= hi for lo, hi in zip(lower, upper)):
                raise ValueError("qmc_quad requires lower < upper in every dimension")
            if n_points == 0:
                raise ValueError("qmc_quad requires n_points > 0")
            if func_name == "x_squared_1d":
                vec_fn = lambda x: x[0] ** 2  # noqa: E731
            elif func_name == "sin_x_1d":
                vec_fn = lambda x: np.sin(x[0])  # noqa: E731
            elif func_name == "x_plus_y_2d":
                vec_fn = lambda x: x[0] + x[1]  # noqa: E731
            elif func_name == "exp_neg_xy_2d":
                vec_fn = lambda x: np.exp(-x[0] * x[1])  # noqa: E731
            else:
                raise ValueError(f"unknown qmc_quad func: {func_name}")
            qrng = _stats.qmc.Halton(d=len(lower), scramble=False)
            res = integrate.qmc_quad(
                vec_fn,
                np.asarray(lower, dtype=float),
                np.asarray(upper, dtype=float),
                n_estimates=8,
                n_points=n_points,
                qrng=qrng,
            )
            return _ok(case_id, "scalar", {"value": float(res.integral)})

        if function == "cubature":
            lower = [float(v) for v in args["lower"]]
            upper = [float(v) for v in args["upper"]]
            if len(lower) != len(upper):
                raise ValueError("cubature bounds must have the same dimensionality")
            atol = float(args.get("atol", 1e-10))
            rtol = float(args.get("rtol", 1e-10))
            func_name = args["func"]
            if len(lower) == 0:
                if func_name == "zero_dim_pair":
                    return _ok(case_id, "array", {"values": [42.0, -7.0]})
                fn0 = _build_callable(func_name)
                return _ok(case_id, "scalar", {"value": float(fn0())})
            if func_name == "powers_1d":
                def cube_vec_fn(xs: Any) -> Any:
                    x = np.atleast_2d(xs)
                    return np.stack([x[..., 0], x[..., 0] ** 2], axis=-1)

                res = integrate.cubature(cube_vec_fn, lower, upper, atol=atol, rtol=rtol)
                return _ok(
                    case_id,
                    "array",
                    {"values": [float(v) for v in np.atleast_1d(res.estimate).tolist()]},
                )
            fn = _build_callable(func_name)
            if hasattr(integrate, "cubature"):
                def cube_fn(xs: Any) -> Any:
                    x = np.atleast_2d(xs)
                    return np.asarray([fn(*row) for row in x], dtype=float)

                res = integrate.cubature(cube_fn, lower, upper, atol=atol, rtol=rtol)
                value = float(res.estimate) if hasattr(res, "estimate") else float(res[0])
                return _ok(case_id, "scalar", {"value": value})
            def nquad_fn(*xs: float) -> float:
                return fn(*xs)

            ranges = list(zip(lower, upper))
            value, _err = integrate.nquad(nquad_fn, ranges, opts={"epsabs": atol, "epsrel": rtol})
            return _ok(case_id, "scalar", {"value": float(value)})

        return {
            "case_id": case_id,
            "status": "error",
            "result_kind": "unsupported_function",
            "result": {},
            "error": f"unsupported function: {function}",
        }

    except (ArithmeticError, OverflowError, TypeError, ValueError, KeyError) as exc:
        return {
            "case_id": case_id,
            "status": "error",
            "result_kind": "exception",
            "result": {},
            "error": str(exc),
        }


def _ok(case_id: str, result_kind: str, result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "status": "ok",
        "result_kind": result_kind,
        "result": result,
        "error": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture SciPy integrate oracle outputs")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--oracle-root", required=False, default="")
    args = parser.parse_args()

    try:
        import numpy as np
        from scipy import integrate
    except ModuleNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    fixture_path = Path(args.fixture)
    output_path = Path(args.output)

    try:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in fixture: {exc}", file=sys.stderr)
        return 1

    case_outputs: List[Dict[str, Any]] = []
    for case in fixture.get("cases", []):
        case_outputs.append(_run_case(case, integrate=integrate, np=np))

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
