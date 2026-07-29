#!/usr/bin/env python3
"""SciPy-backed oracle capture for FrankenSciPy interpolate packet fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List


def _float_list(values: Any) -> List[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(value) for value in values]


def _ok(case_id: str, result_kind: str, result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "status": "ok",
        "result_kind": result_kind,
        "result": result,
        "error": None,
    }


def _err(case_id: str, error: str) -> Dict[str, Any]:
    return _ok(case_id, "error", {"error": error})


def _fixture_error(case: Dict[str, Any], fallback: str) -> str:
    expected = case.get("expected", {})
    if expected.get("kind") == "error":
        return str(expected.get("error") or fallback)
    return fallback


def _coerce_maybe_nan_f64(value: Any) -> float:
    if isinstance(value, str):
        marker = value.strip().lower()
        if marker == "nan":
            return float("nan")
        if marker in {"inf", "+inf", "infinity", "+infinity"}:
            return float("inf")
        if marker in {"-inf", "-infinity"}:
            return float("-inf")
    return float(value)


def _cubic_spline_bc_type(case: Dict[str, Any]) -> Any:
    bc = case.get("bc", {"kind": "natural"})
    if isinstance(bc, str):
        return bc.replace("_", "-")

    kind = str(bc.get("kind", "natural"))
    if kind in {"natural", "not_a_knot", "periodic"}:
        return kind.replace("_", "-")
    if kind == "clamped":
        return (
            (1, _coerce_maybe_nan_f64(bc["left_derivative"])),
            (1, _coerce_maybe_nan_f64(bc["right_derivative"])),
        )
    if kind == "tuple":
        return (
            (int(bc["left_order"]), _coerce_maybe_nan_f64(bc["left_value"])),
            (int(bc["right_order"]), _coerce_maybe_nan_f64(bc["right_value"])),
        )
    raise ValueError(f"unsupported CubicSpline bc kind: {kind}")


def _run_interp1d(case: Dict[str, Any], interpolate: Any, np: Any) -> Dict[str, Any]:
    case_id = case["case_id"]
    expected = case.get("expected", {})

    # FrankenSciPy's current strict contract rejects unsorted x at construction
    # while scipy.interpolate.interp1d silently sorts by default. Preserve the
    # packet's declared reject-path category until a separate parity slice
    # decides whether the Rust kernel should adopt SciPy's sorting behavior.
    x = [float(value) for value in case["x"]]
    if expected.get("kind") == "error" and any(right <= left for left, right in zip(x, x[1:])):
        return _err(case_id, _fixture_error(case, "x values must be strictly increasing"))

    try:
        y = np.array(case["y"], dtype=np.float64)
        x_new = np.array(case["x_new"], dtype=np.float64)
        kwargs: Dict[str, Any] = {"kind": case.get("kind", "linear")}
        if "bounds_error" in case:
            kwargs["bounds_error"] = bool(case["bounds_error"])
        if "fill_value" in case:
            kwargs["fill_value"] = float(case["fill_value"])

        interpolator = interpolate.interp1d(np.array(x, dtype=np.float64), y, **kwargs)
        values = interpolator(x_new)
        return _ok(case_id, "vector", {"values": _float_list(values)})
    except (ArithmeticError, OverflowError, TypeError, ValueError) as exc:
        return _err(case_id, _fixture_error(case, str(exc)))


def _run_regular_grid_interpolator(
    case: Dict[str, Any], interpolate: Any, np: Any
) -> Dict[str, Any]:
    case_id = case["case_id"]
    try:
        points = [np.array(axis, dtype=np.float64) for axis in case["points"]]
        shape = tuple(len(axis) for axis in points)
        values = np.array(case["values"], dtype=np.float64).reshape(shape)
        xi = np.array(case["xi"], dtype=np.float64)
        kwargs: Dict[str, Any] = {
            "method": case.get("method", "linear"),
            "bounds_error": bool(case.get("bounds_error", True)),
        }
        if "fill_value" in case:
            kwargs["fill_value"] = float(case["fill_value"])

        interpolator = interpolate.RegularGridInterpolator(points, values, **kwargs)
        return _ok(case_id, "vector", {"values": _float_list(interpolator(xi))})
    except (ArithmeticError, OverflowError, TypeError, ValueError) as exc:
        return _err(case_id, _fixture_error(case, str(exc)))


def _run_cubic_spline(case: Dict[str, Any], interpolate: Any, np: Any) -> Dict[str, Any]:
    case_id = case["case_id"]
    try:
        bc = _cubic_spline_bc_type(case)
        spline = interpolate.CubicSpline(
            np.array(case["x"], dtype=np.float64),
            np.array(case["y"], dtype=np.float64),
            bc_type=bc,
        )
        values = spline(np.array(case["x_new"], dtype=np.float64))
        return _ok(case_id, "vector", {"values": _float_list(values)})
    except (ArithmeticError, OverflowError, TypeError, ValueError) as exc:
        return _err(case_id, _fixture_error(case, str(exc)))


def _run_bspline(case: Dict[str, Any], interpolate: Any, np: Any) -> Dict[str, Any]:
    case_id = case["case_id"]
    try:
        spline = interpolate.BSpline(
            np.array(case["knots"], dtype=np.float64),
            np.array(case["coefficients"], dtype=np.float64),
            int(case["degree"]),
        )
        values = spline(np.array(case["x_new"], dtype=np.float64))
        return _ok(case_id, "vector", {"values": _float_list(values)})
    except (ArithmeticError, OverflowError, TypeError, ValueError) as exc:
        return _err(case_id, _fixture_error(case, str(exc)))


def _run_case(case: Dict[str, Any], interpolate: Any, np: Any) -> Dict[str, Any]:
    operation = case.get("operation")
    if operation == "interp1d":
        return _run_interp1d(case, interpolate, np)
    if operation == "regular_grid_interpolator":
        return _run_regular_grid_interpolator(case, interpolate, np)
    if operation == "cubic_spline":
        return _run_cubic_spline(case, interpolate, np)
    if operation == "bspline":
        return _run_bspline(case, interpolate, np)
    return {
        "case_id": case.get("case_id", "<missing>"),
        "status": "error",
        "result_kind": "unsupported_operation",
        "result": {},
        "error": f"unsupported operation: {operation}",
    }


def _read_live_vector(prefix: str, expected: int, np: Any) -> Any:
    line = sys.stdin.readline()
    marker = f"{prefix} "
    if not line.startswith(marker):
        raise ValueError(f"expected {prefix} vector, got {line.strip()!r}")
    values = np.fromstring(line[len(marker) :], dtype=np.float64, sep=",")
    if values.size != expected:
        raise ValueError(f"{prefix} vector length {values.size} != {expected}")
    return values


def _observed_os_threads() -> int:
    """Return native process threads, including pools invisible to `threading`."""
    task_dir = Path("/proc/self/task")
    if task_dir.is_dir():
        return sum(1 for _entry in task_dir.iterdir())
    return threading.active_count()


def _run_cursor_live(
    interpolate: Any, np: Any, scipy: Any, cursor_kind: str
) -> int:
    scipy_path = Path(scipy.__file__).resolve()
    cursor_type = {
        "pchip": interpolate.PchipInterpolator,
        "cubic": interpolate.CubicSpline,
        "akima": interpolate.Akima1DInterpolator,
        "hermite": interpolate.CubicHermiteSpline,
    }[cursor_kind]
    cursor_module = cursor_type.__module__
    module_file = Path(sys.modules[cursor_module].__file__).resolve()
    scipy_engine_sha256 = hashlib.sha256(module_file.read_bytes()).hexdigest()
    actual_observed_worker_threads = _observed_os_threads()
    fsci_loaded = any(
        name == "fsci_interpolate" or name.startswith("fsci_")
        for name in sys.modules
    )
    genuine = (
        cursor_module == "scipy.interpolate._cubic"
        and scipy_path.parent in module_file.parents
        and not fsci_loaded
    )
    print(
        f"READY scipy={scipy.__version__} numpy={np.__version__} "
        f"file={scipy_path} cursor_kind={cursor_kind} cursor_mod={cursor_module} "
        f"scipy_engine_path={module_file} "
        f"scipy_engine_sha256={scipy_engine_sha256} "
        f"actual_observed_worker_threads={actual_observed_worker_threads} "
        f"fsci_loaded={fsci_loaded} genuine={genuine}",
        flush=True,
    )
    if not genuine:
        print("FATAL not-genuine-scipy-cursor", flush=True)
        return 2

    incumbent = None
    queries = None
    for raw in sys.stdin:
        fields = raw.strip().split()
        if not fields:
            continue
        command = fields[0]
        try:
            if command == "INIT":
                if len(fields) != 3:
                    raise ValueError("INIT requires knot and query counts")
                knot_count = int(fields[1])
                query_count = int(fields[2])
                x = _read_live_vector("X", knot_count, np)
                y = _read_live_vector("Y", knot_count, np)
                queries = _read_live_vector("Q", query_count, np)
                derivatives = _read_live_vector("D", knot_count, np)
                if cursor_kind == "pchip":
                    incumbent = interpolate.PchipInterpolator(
                        x, y, extrapolate=True
                    )
                elif cursor_kind == "cubic":
                    incumbent = interpolate.CubicSpline(
                        x, y, bc_type="natural", extrapolate=True
                    )
                elif cursor_kind == "akima":
                    incumbent = interpolate.Akima1DInterpolator(
                        x, y, extrapolate=True
                    )
                else:
                    incumbent = interpolate.CubicHermiteSpline(
                        x, y, derivatives, extrapolate=True
                    )
                sorted_queries = bool(np.all(queries[1:] >= queries[:-1]))
                finite = bool(
                    np.all(np.isfinite(x))
                    and np.all(np.isfinite(y))
                    and np.all(np.isfinite(derivatives))
                    and np.all(np.isfinite(queries))
                )
                print(
                    f"CASE cursor_kind={cursor_kind} knots={x.size} "
                    f"queries={queries.size} "
                    f"sorted={sorted_queries} finite={finite}",
                    flush=True,
                )
            elif command == "PARITY":
                if incumbent is None or queries is None:
                    raise ValueError("PARITY before INIT")
                values = np.asarray(incumbent(queries), dtype=np.float64)
                print(f"RESULT components={values.size}", flush=True)
                print(
                    "Y " + ",".join(format(float(value), ".17g") for value in values),
                    flush=True,
                )
            elif command == "SOLVE":
                if incumbent is None or queries is None:
                    raise ValueError("SOLVE before INIT")
                if len(fields) != 2:
                    raise ValueError("SOLVE requires a repetition count")
                repetitions = int(fields[1])
                if repetitions < 1:
                    raise ValueError("SOLVE repetitions must be positive")
                values = None
                started = time.perf_counter_ns()
                for _ in range(repetitions):
                    values = incumbent(queries)
                elapsed_seconds = (time.perf_counter_ns() - started) * 1.0e-9
                values = np.asarray(values, dtype=np.float64)
                checksum = int(np.bitwise_xor.reduce(values.view(np.uint64)))
                print(
                    f"TIME {elapsed_seconds:.17g} {values.size} {checksum:016x}",
                    flush=True,
                )
            elif command == "QUIT":
                return 0
            else:
                raise ValueError(f"unknown command {command!r}")
        except (ArithmeticError, OverflowError, TypeError, ValueError) as exc:
            print(f"FATAL {type(exc).__name__}: {exc}", flush=True)
            return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture SciPy interpolate oracle outputs")
    parser.add_argument("--fixture", required=False, help="Input packet fixture JSON path")
    parser.add_argument("--output", required=False, help="Output oracle capture JSON path")
    parser.add_argument(
        "--pchip-live",
        action="store_true",
        help="Run the persistent live-SciPy PCHIP protocol (legacy alias)",
    )
    parser.add_argument(
        "--cursor-live",
        choices=("pchip", "cubic", "akima", "hermite"),
        help="Run a persistent live-SciPy cubic-cursor protocol on stdin",
    )
    parser.add_argument(
        "--oracle-root",
        required=False,
        default="",
        help="(unused) legacy oracle root path, kept for CLI backwards compatibility",
    )
    args = parser.parse_args()

    try:
        import numpy as np
        import scipy
        from scipy import interpolate
    except ModuleNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.pchip_live and args.cursor_live:
        parser.error("--pchip-live and --cursor-live are mutually exclusive")
    if args.pchip_live:
        return _run_cursor_live(interpolate, np, scipy, "pchip")
    if args.cursor_live:
        return _run_cursor_live(interpolate, np, scipy, args.cursor_live)
    if not args.fixture or not args.output:
        parser.error("--fixture and --output are required outside live cursor mode")
    fixture_path = Path(args.fixture)
    output_path = Path(args.output)

    try:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in fixture: {exc}", file=sys.stderr)
        return 1

    payload = {
        "packet_id": fixture.get("packet_id", "unknown"),
        "family": fixture.get("family", "unknown"),
        "generated_unix_ms": int(time.time() * 1000),
        "runtime": {
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
        },
        "case_outputs": [
            _run_case(case, interpolate, np) for case in fixture.get("cases", [])
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
