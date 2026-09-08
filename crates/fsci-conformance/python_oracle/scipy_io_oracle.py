#!/usr/bin/env python3
"""SciPy/NumPy-backed oracle capture for FrankenSciPy I/O packet fixtures."""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0.0 else "-Infinity"
        return value
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


def _float_list(values: Any) -> List[float]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, list):
        out: List[float] = []
        for value in values:
            if isinstance(value, list):
                out.extend(_float_list(value))
            else:
                out.append(float(value))
        return out
    return [float(values)]


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


def _matrix_payload(array: Any, np: Any) -> Dict[str, Any]:
    if hasattr(array, "toarray"):
        array = array.toarray()
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape((1, 1))
    elif arr.ndim == 1:
        arr = arr.reshape((1, arr.shape[0]))
    return {
        "rows": int(arr.shape[0]),
        "cols": int(arr.shape[1]),
        "values": _float_list(arr.reshape(-1)),
    }


def _run_mmread(case: Dict[str, Any], scipy_io: Any, np: Any) -> Dict[str, Any]:
    case_id = case["case_id"]
    try:
        stream = io.BytesIO(case["content"].encode("utf-8"))
        return _ok(case_id, "matrix", _matrix_payload(scipy_io.mmread(stream), np))
    except Exception as exc:
        return _err(case_id, _fixture_error(case, str(exc)))


def _run_mmwrite(case: Dict[str, Any], scipy_io: Any, np: Any) -> Dict[str, Any]:
    case_id = case["case_id"]
    try:
        rows = int(case["rows"])
        cols = int(case["cols"])
        matrix = np.asarray(case["data"], dtype=np.float64).reshape((rows, cols))
        stream = io.BytesIO()
        scipy_io.mmwrite(stream, matrix)
        stream.seek(0)
        return _ok(case_id, "matrix", _matrix_payload(scipy_io.mmread(stream), np))
    except Exception as exc:
        return _err(case_id, _fixture_error(case, str(exc)))


def _run_loadmat(case: Dict[str, Any], scipy_io: Any, np: Any) -> Dict[str, Any]:
    case_id = case["case_id"]
    try:
        stream = io.BytesIO(bytes.fromhex(str(case["content_hex"])))
        loaded = scipy_io.loadmat(stream)
        keys = [key for key in loaded.keys() if not key.startswith("__")]
        if not keys:
            return _err(case_id, _fixture_error(case, "MAT file did not contain any arrays"))
        return _ok(case_id, "matrix", _matrix_payload(loaded[keys[0]], np))
    except Exception as exc:
        return _err(case_id, _fixture_error(case, str(exc)))


def _run_savemat(case: Dict[str, Any], scipy_io: Any, np: Any) -> Dict[str, Any]:
    case_id = case["case_id"]
    try:
        rows = int(case["rows"])
        cols = int(case["cols"])
        name = str(case["name"])
        matrix = np.asarray(case["data"], dtype=np.float64).reshape((rows, cols))
        stream = io.BytesIO()
        scipy_io.savemat(stream, {name: matrix}, format="4")
        stream.seek(0)
        loaded = scipy_io.loadmat(stream)
        return _ok(case_id, "matrix", _matrix_payload(loaded[name], np))
    except Exception as exc:
        return _err(case_id, _fixture_error(case, str(exc)))


def _run_loadtxt(case: Dict[str, Any], np: Any) -> Dict[str, Any]:
    case_id = case["case_id"]
    try:
        stream = io.StringIO(case["content"])
        return _ok(case_id, "matrix", _matrix_payload(np.loadtxt(stream), np))
    except Exception as exc:
        return _err(case_id, _fixture_error(case, str(exc)))


def _run_savetxt(case: Dict[str, Any], np: Any) -> Dict[str, Any]:
    case_id = case["case_id"]
    try:
        rows = int(case["rows"])
        cols = int(case["cols"])
        matrix = np.asarray(case["data"], dtype=np.float64).reshape((rows, cols))
        stream = io.StringIO()
        np.savetxt(stream, matrix, delimiter=str(case.get("delimiter", " ")))
        stream.seek(0)
        return _ok(case_id, "matrix", _matrix_payload(np.loadtxt(stream), np))
    except Exception as exc:
        return _err(case_id, _fixture_error(case, str(exc)))


def _run_wav_write(case: Dict[str, Any], wavfile: Any, np: Any) -> Dict[str, Any]:
    case_id = case["case_id"]
    try:
        channels = int(case["channels"])
        data = np.asarray(case["data"], dtype=np.float64)
        pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
        if channels > 1:
            pcm = pcm.reshape((-1, channels))
        stream = io.BytesIO()
        wavfile.write(stream, int(case["sample_rate"]), pcm)
        stream.seek(0)
        sample_rate, read = wavfile.read(stream)
        read_arr = np.asarray(read)
        bits_per_sample = int(read_arr.dtype.itemsize * 8)
        values = read_arr.astype(np.float64).reshape(-1) / 32768.0
        return _ok(
            case_id,
            "wav",
            {
                "sample_rate": int(sample_rate),
                "channels": channels,
                "bits_per_sample": bits_per_sample,
                "values": _float_list(values),
            },
        )
    except Exception as exc:
        return _err(case_id, _fixture_error(case, str(exc)))


def _run_case(case: Dict[str, Any], scipy_io: Any, wavfile: Any, np: Any) -> Dict[str, Any]:
    try:
        operation = case.get("operation")
        if operation == "mmread":
            return _run_mmread(case, scipy_io, np)
        if operation == "mmwrite":
            return _run_mmwrite(case, scipy_io, np)
        if operation == "loadmat":
            return _run_loadmat(case, scipy_io, np)
        if operation == "savemat":
            return _run_savemat(case, scipy_io, np)
        if operation == "loadtxt":
            return _run_loadtxt(case, np)
        if operation == "savetxt":
            return _run_savetxt(case, np)
        if operation == "wav_write":
            return _run_wav_write(case, wavfile, np)
        return {
            "case_id": case.get("case_id", "<missing>"),
            "status": "error",
            "result_kind": "unsupported_operation",
            "result": {},
            "error": f"unsupported operation: {operation}",
        }
    except Exception as exc:
        return _err(case.get("case_id", "<missing>"), _fixture_error(case, str(exc)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture SciPy I/O oracle outputs")
    parser.add_argument("--fixture", required=True, help="Input packet fixture JSON path")
    parser.add_argument("--output", required=True, help="Output oracle capture JSON path")
    parser.add_argument(
        "--oracle-root",
        required=False,
        default="",
        help="(unused) legacy oracle root path, kept for CLI backwards compatibility",
    )
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    output_path = Path(args.output)

    try:
        import numpy as np
        import scipy
        import scipy.io as scipy_io
        from scipy.io import wavfile
    except ModuleNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

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
            _run_case(case, scipy_io, wavfile, np) for case in fixture.get("cases", [])
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
