#!/usr/bin/env python3
"""SciPy-backed oracle capture for FrankenSciPy FFT packet fixtures.

Reads a conformance fixture JSON and emits a normalized oracle capture JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _json_float(value: Any) -> Any:
    """Keep non-finite SciPy output representable in strict JSON captures."""
    value = float(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _to_complex_list(arr: Any) -> List[List[Any]]:
    """Convert numpy array to [[real, imag], ...] format."""
    result = []
    for val in arr.flat:
        result.append([_json_float(val.real), _json_float(val.imag)])
    return result


def _to_real_list(arr: Any) -> List[Any]:
    """Convert numpy array to list of floats."""
    return [_json_float(v) for v in arr.flat]


def _coerce_maybe_nan_f64(value: Any) -> float:
    """Accept either a JSON number or a NaN/Inf string sentinel per br-gr99.

    Mirror of the Rust-side maybe_nan_f64 deserializer: fixtures can
    encode non-finite inputs as "NaN" / "Infinity" / "-Infinity" (case
    insensitive) since stock JSON has no literal for these.
    """
    if isinstance(value, str):
        key = value.strip().lower()
        if key == "nan":
            return float("nan")
        if key in ("infinity", "inf", "+infinity", "+inf"):
            return float("inf")
        if key in ("-infinity", "-inf"):
            return float("-inf")
        return float(value)  # fall through for exponent-string numerics
    return float(value)


def _complex_input_to_numpy(data: List[List[float]], np: Any) -> Any:
    """Convert [[real, imag], ...] to numpy complex array.

    Accepts NaN/Inf string sentinels in either component per br-gr99.
    """
    return np.array(
        [complex(_coerce_maybe_nan_f64(r), _coerce_maybe_nan_f64(i)) for r, i in data],
        dtype=np.complex128,
    )


def _shape_tuple(case: Dict[str, Any], field: str) -> Any:
    """Normalize optional shape fields into tuples of ints."""
    shape = case.get(field)
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)


def _resolve_fft_shapes(case: Dict[str, Any]) -> tuple[Any, Any]:
    """Resolve reshape vs output-shape hints without ambiguous legacy mixing."""
    input_shape = _shape_tuple(case, "input_shape")
    output_shape = _shape_tuple(case, "output_shape")
    legacy_shape = _shape_tuple(case, "shape")

    if legacy_shape is not None and (input_shape is not None or output_shape is not None):
        raise ValueError(
            "shape is deprecated for multi-dimensional FFT hints; use input_shape/output_shape without mixing"
        )

    if input_shape is None:
        input_shape = legacy_shape
    if output_shape is None:
        output_shape = legacy_shape

    return input_shape, output_shape


def _irfft_input_shape(output_shape: Any) -> Any:
    """Compute the (complex) input shape for irfft2/irfftn from the
    real output shape. scipy convention: irfft along the last axis maps
    n//2 + 1 complex samples back to n real samples; other axes are
    full-length. Returns None when output_shape is None.

    Per br-ocsl.
    """
    if output_shape is None:
        return None
    last = output_shape[-1] // 2 + 1
    return (*output_shape[:-1], last)


def _reshape_input(case: Dict[str, Any], x: Any, np: Any) -> Any:
    """Reshape flat inputs for multi-dimensional transforms when possible."""
    input_shape, _ = _resolve_fft_shapes(case)
    if input_shape is not None:
        expected = int(np.prod(input_shape))
        if x.size != expected:
            raise ValueError(
                f"input size {x.size} does not match input_shape {list(input_shape)}"
            )
        return x.reshape(input_shape)
    return x


def _run_case(case: Dict[str, Any], fft: Any, np: Any) -> Dict[str, Any]:
    case_id = case["case_id"]
    transform = case.get("transform", "fft")
    normalization = case.get("normalization", "backward")

    try:
        # Build input array. real_input entries may be NaN/Inf string
        # sentinels per br-gr99 (stock JSON can't encode them as numbers).
        if case.get("complex_input") is not None:
            x = _complex_input_to_numpy(case["complex_input"], np)
        elif case.get("real_input") is not None:
            x = np.array(
                [_coerce_maybe_nan_f64(v) for v in case["real_input"]],
                dtype=np.float64,
            )
        else:
            return {
                "case_id": case_id,
                "status": "error",
                "result_kind": "missing_input",
                "result": {},
                "error": "no input provided",
            }

        n = case.get("output_len")

        # Execute transform
        if transform == "fft":
            result = fft.fft(x, n=n, norm=normalization)
        elif transform == "ifft":
            result = fft.ifft(x, n=n, norm=normalization)
        elif transform == "rfft":
            result = fft.rfft(x, n=n, norm=normalization)
        elif transform == "irfft":
            result = fft.irfft(x, n=n, norm=normalization)
        elif transform == "hfft":
            result = fft.hfft(x, n=n, norm=normalization)
        elif transform == "ihfft":
            result = fft.ihfft(x, n=n, norm=normalization)
        elif transform in ("dct", "dct_i", "dct_iii", "dct_iv"):
            dct_type = {"dct_i": 1, "dct_iii": 3, "dct_iv": 4}.get(
                transform, case.get("dct_type", 2)
            )
            result = fft.dct(x, type=dct_type, norm=normalization)
        elif transform == "idct":
            dct_type = case.get("dct_type", 2)
            result = fft.idct(x, type=dct_type, norm=normalization)
        elif transform in ("dst", "dst_i", "dst_ii", "dst_iii", "dst_iv"):
            dst_type = {"dst_i": 1, "dst_ii": 2, "dst_iii": 3, "dst_iv": 4}.get(
                transform, case.get("dst_type", 2)
            )
            result = fft.dst(x, type=dst_type, norm=normalization)
        elif transform == "idst":
            dst_type = case.get("dst_type", 2)
            result = fft.idst(x, type=dst_type, norm=normalization)
        elif transform == "fft2":
            _, output_shape = _resolve_fft_shapes(case)
            result = fft.fft2(_reshape_input(case, x, np), s=output_shape, norm=normalization)
        elif transform == "ifft2":
            _, output_shape = _resolve_fft_shapes(case)
            result = fft.ifft2(_reshape_input(case, x, np), s=output_shape, norm=normalization)
        elif transform == "fftn":
            _, output_shape = _resolve_fft_shapes(case)
            result = fft.fftn(_reshape_input(case, x, np), s=output_shape, norm=normalization)
        elif transform == "ifftn":
            _, output_shape = _resolve_fft_shapes(case)
            result = fft.ifftn(_reshape_input(case, x, np), s=output_shape, norm=normalization)
        elif transform == "rfft2":
            # br-ocsl: real 2D FFT. Input is real, reshape via shape/input_shape.
            _, output_shape = _resolve_fft_shapes(case)
            result = fft.rfft2(_reshape_input(case, x, np), s=output_shape, norm=normalization)
        elif transform == "irfft2":
            # br-ocsl: inverse real 2D FFT. The fixture's `shape` is the
            # *output* (real) shape; the complex input shape is derived as
            # (s[0], s[-1]//2 + 1). Cannot reuse _reshape_input because it
            # would reshape to the output shape and fail size-match.
            _, output_shape = _resolve_fft_shapes(case)
            input_shape = _irfft_input_shape(output_shape)
            x_nd = x.reshape(input_shape) if input_shape is not None else x
            result = fft.irfft2(x_nd, s=output_shape, norm=normalization)
        elif transform == "rfftn":
            # br-ocsl: real ND FFT. axes default to all; output is complex with
            # last axis truncated to s[-1]//2+1.
            _, output_shape = _resolve_fft_shapes(case)
            result = fft.rfftn(_reshape_input(case, x, np), s=output_shape, norm=normalization)
        elif transform == "irfftn":
            # br-ocsl: inverse real ND FFT. Same reshape rule as irfft2.
            _, output_shape = _resolve_fft_shapes(case)
            input_shape = _irfft_input_shape(output_shape)
            x_nd = x.reshape(input_shape) if input_shape is not None else x
            result = fft.irfftn(x_nd, s=output_shape, norm=normalization)
        elif transform == "fftfreq":
            n_pts = int(case.get("n_points", len(x)))
            d = case.get("sample_spacing")
            if d is None:
                d = 1.0
            result = fft.fftfreq(n_pts, d=d)
        elif transform == "rfftfreq":
            n_pts = int(case.get("n_points", len(x)))
            d = case.get("sample_spacing")
            if d is None:
                d = 1.0
            result = fft.rfftfreq(n_pts, d=d)
        elif transform == "fftshift":
            result = fft.fftshift(x)
        elif transform == "ifftshift":
            result = fft.ifftshift(x)
        elif transform == "hilbert":
            from scipy import signal
            result = signal.hilbert(x.real if np.iscomplexobj(x) else x)
        else:
            return {
                "case_id": case_id,
                "status": "error",
                "result_kind": "unsupported_transform",
                "result": {},
                "error": f"unsupported transform: {transform}",
            }

        # Format output
        if np.iscomplexobj(result):
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "complex_vector",
                "result": {"values": _to_complex_list(result)},
                "error": None,
            }
        else:
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "real_vector",
                "result": {"values": _to_real_list(result)},
                "error": None,
            }

    # br-p3be: narrow catch; let MemoryError / OSError propagate.
    except (
        ArithmeticError,
        OverflowError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
    ) as exc:
        return {
            "case_id": case_id,
            "status": "error",
            "result_kind": "exception",
            "result": {},
            "error": f"{type(exc).__name__}: {exc}",
        }


def _oracle_values_to_complex_array(values: List[List[float]], np: Any) -> Any:
    """Convert oracle [[real, imag], ...] payloads back into numpy complex arrays."""
    return np.array([complex(real, imag) for real, imag in values], dtype=np.complex128)


def _self_check_shape_and_complex_paths() -> List[str]:
    """Verify complex DCT/DST support and split shape semantics."""
    try:
        import numpy as np
        from scipy import fft
    except ModuleNotFoundError as exc:
        return [str(exc)]

    errors: List[str] = []

    complex_input = [[1.0, 2.0], [3.0, -4.0], [0.5, 0.25]]
    complex_array = _complex_input_to_numpy(complex_input, np)

    dct_case = {
        "case_id": "self_check_dct_complex",
        "transform": "dct",
        "complex_input": complex_input,
    }
    dct_result = _run_case(dct_case, fft=fft, np=np)
    expected_dct = fft.dct(complex_array, type=2, norm="backward")
    if dct_result["status"] != "ok" or dct_result["result_kind"] != "complex_vector":
        errors.append(f"complex dct self-check failed: {dct_result}")
    else:
        observed_dct = _oracle_values_to_complex_array(dct_result["result"]["values"], np)
        if not np.allclose(observed_dct, expected_dct, atol=1.0e-12, rtol=1.0e-12):
            errors.append("complex dct self-check mismatch")

    dst_case = {
        "case_id": "self_check_dst_complex",
        "transform": "dst",
        "complex_input": complex_input,
    }
    dst_result = _run_case(dst_case, fft=fft, np=np)
    expected_dst = fft.dst(complex_array, type=2, norm="backward")
    if dst_result["status"] != "ok" or dst_result["result_kind"] != "complex_vector":
        errors.append(f"complex dst self-check failed: {dst_result}")
    else:
        observed_dst = _oracle_values_to_complex_array(dst_result["result"]["values"], np)
        if not np.allclose(observed_dst, expected_dst, atol=1.0e-12, rtol=1.0e-12):
            errors.append("complex dst self-check mismatch")

    shape_case = {
        "case_id": "self_check_fft2_split_shape",
        "transform": "fft2",
        "real_input": [float(i) for i in range(9)],
        "input_shape": [3, 3],
        "output_shape": [5, 5],
    }
    shape_result = _run_case(shape_case, fft=fft, np=np)
    expected_shape = fft.fft2(np.arange(9.0).reshape((3, 3)), s=(5, 5), norm="backward")
    if shape_result["status"] != "ok":
        errors.append(f"split shape self-check failed: {shape_result}")
    else:
        observed_shape = _oracle_values_to_complex_array(shape_result["result"]["values"], np)
        # _oracle_values_to_complex_array always returns a 1-D vector
        # (the oracle serialises any N-D output as a flat [[re, im], ...]
        # list — see br-gr99). Compare against the ravelled expected
        # rather than the un-ravelled (5, 5) shape, which can never match
        # the 1-D vector and would always mark the self-check as failed.
        if observed_shape.size != expected_shape.size or not np.allclose(
            observed_shape, expected_shape.ravel(), atol=1.0e-12, rtol=1.0e-12
        ):
            errors.append("split shape self-check mismatch")

    ambiguous_case = {
        "case_id": "self_check_fft2_ambiguous_shape",
        "transform": "fft2",
        "real_input": [float(i) for i in range(9)],
        "shape": [3, 3],
        "output_shape": [5, 5],
    }
    ambiguous_result = _run_case(ambiguous_case, fft=fft, np=np)
    if ambiguous_result["status"] != "error" or "deprecated" not in (
        ambiguous_result.get("error") or ""
    ):
        errors.append(f"ambiguous shape self-check should fail closed: {ambiguous_result}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture SciPy FFT oracle outputs")
    parser.add_argument("--fixture", help="Input packet fixture JSON path")
    parser.add_argument("--output", help="Output oracle capture JSON path")
    parser.add_argument("--oracle-root", default="",
                        help="(unused) legacy oracle root path — kept for CLI backwards compat")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Verify split shape semantics and complex DCT/DST handling",
    )
    args = parser.parse_args()

    try:
        import numpy as np
        from scipy import fft
    except ModuleNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.self_check:
        errors = _self_check_shape_and_complex_paths()
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 1
        print("fft oracle shape/complex self-check passed")
        return 0

    if not args.fixture or not args.output:
        parser.error("--fixture and --output are required unless --self-check is used")

    fixture_path = Path(args.fixture)
    output_path = Path(args.output)

    try:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Invalid JSON in fixture: {exc}", file=sys.stderr)
        return 1

    case_outputs: List[Dict[str, Any]] = []

    for case in fixture.get("cases", []):
        case_outputs.append(_run_case(case, fft=fft, np=np))

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
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
