#!/usr/bin/env python3
"""SciPy-backed oracle capture for FrankenSciPy linalg packet fixtures.

Reads a conformance fixture JSON and emits a normalized oracle capture JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _to_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _as_float(value: Any) -> float:
    return float(value)


def _run_case(case: Dict[str, Any], linalg: Any, np: Any) -> Dict[str, Any]:
    operation = case["operation"]
    case_id = case["case_id"]

    try:
        if operation == "solve":
            a = np.asarray(case["a"], dtype=np.float64)
            b = np.asarray(case["b"], dtype=np.float64)
            assume_a = case.get("assume_a")
            transposed = bool(case.get("transposed", False))
            check_finite = bool(case.get("check_finite", True))
            if transposed:
                a = a.T

            if assume_a == "diagonal":
                diag = np.diag(a)
                if np.any(diag == 0.0):
                    raise linalg.LinAlgError("singular matrix")
                x = b / diag
            elif assume_a == "upper_triangular":
                x = linalg.solve_triangular(
                    a,
                    b,
                    lower=False,
                    trans=0,
                    unit_diagonal=False,
                    check_finite=check_finite,
                )
            elif assume_a == "lower_triangular":
                x = linalg.solve_triangular(
                    a,
                    b,
                    lower=True,
                    trans=0,
                    unit_diagonal=False,
                    check_finite=check_finite,
                )
            else:
                scipy_assume = {
                    None: "gen",
                    "general": "gen",
                    "symmetric": "sym",
                    "hermitian": "her",
                    "positive_definite": "pos",
                }.get(assume_a, "gen")
                x = linalg.solve(a, b, assume_a=scipy_assume, check_finite=check_finite)

            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "vector",
                "result": {"values": [float(v) for v in _to_list(x)]},
                "error": None,
            }

        if operation == "solve_triangular":
            a = np.asarray(case["a"], dtype=np.float64)
            b = np.asarray(case["b"], dtype=np.float64)
            trans_name = case.get("trans", "no_transpose")
            trans = {
                "no_transpose": 0,
                "transpose": 1,
                "conjugate_transpose": 2,
            }.get(trans_name, 0)
            x = linalg.solve_triangular(
                a,
                b,
                trans=trans,
                lower=bool(case.get("lower", False)),
                unit_diagonal=bool(case.get("unit_diagonal", False)),
                check_finite=bool(case.get("check_finite", True)),
            )
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "vector",
                "result": {"values": [float(v) for v in _to_list(x)]},
                "error": None,
            }

        if operation == "solve_banded":
            l_and_u = case["l_and_u"]
            ab = np.asarray(case["ab"], dtype=np.float64)
            b = np.asarray(case["b"], dtype=np.float64)
            x = linalg.solve_banded(
                (int(l_and_u[0]), int(l_and_u[1])),
                ab,
                b,
                check_finite=bool(case.get("check_finite", True)),
            )
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "vector",
                "result": {"values": [float(v) for v in _to_list(x)]},
                "error": None,
            }

        if operation == "inv":
            a = np.asarray(case["a"], dtype=np.float64)
            x = linalg.inv(a, check_finite=bool(case.get("check_finite", True)))
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "matrix",
                "result": {"values": [[float(v) for v in row] for row in _to_list(x)]},
                "error": None,
            }

        if operation == "det":
            a = np.asarray(case["a"], dtype=np.float64)
            value = linalg.det(a, check_finite=bool(case.get("check_finite", True)))
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "scalar",
                "result": {"value": _as_float(value)},
                "error": None,
            }

        if operation == "lstsq":
            a = np.asarray(case["a"], dtype=np.float64)
            b = np.asarray(case["b"], dtype=np.float64)
            x, residuals, rank, singular_values = linalg.lstsq(
                a,
                b,
                cond=case.get("cond", None),
                check_finite=bool(case.get("check_finite", True)),
            )
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "lstsq",
                "result": {
                    "x": [float(v) for v in _to_list(x)],
                    "residuals": [float(v) for v in _to_list(residuals)],
                    "rank": int(rank),
                    "singular_values": [float(v) for v in _to_list(singular_values)],
                },
                "error": None,
            }

        if operation == "pinv":
            a = np.asarray(case["a"], dtype=np.float64)
            x = linalg.pinv(
                a,
                atol=case.get("atol", 0.0),
                rtol=case.get("rtol", None),
                check_finite=bool(case.get("check_finite", True)),
            )
            singular_values = np.linalg.svd(a, compute_uv=False)
            max_s = float(np.max(singular_values)) if singular_values.size > 0 else 0.0
            atol = float(case.get("atol", 0.0))
            if case.get("rtol", None) is None:
                rtol = float(max(a.shape) * np.finfo(np.float64).eps)
            else:
                rtol = float(case["rtol"])
            threshold = atol + rtol * max_s
            rank = int(np.sum(singular_values > threshold))
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "pinv",
                "result": {
                    "values": [[float(v) for v in row] for row in _to_list(x)],
                    "rank": rank,
                },
                "error": None,
            }

        return {
            "case_id": case_id,
            "status": "error",
            "result_kind": "unsupported_operation",
            "result": {},
            "error": f"unsupported operation: {operation}",
        }

    except Exception as exc:  # noqa: BLE001
        return {
            "case_id": case_id,
            "status": "error",
            "result_kind": "exception",
            "result": {},
            "error": str(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture SciPy linalg oracle outputs")
    parser.add_argument("--fixture", required=True, help="Input packet fixture JSON path")
    parser.add_argument("--output", required=True, help="Output oracle capture JSON path")
    parser.add_argument("--oracle-root", required=True, help="Legacy oracle root path")
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    output_path = Path(args.output)

    try:
        import numpy as np
        from scipy import linalg
    except ModuleNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    case_outputs: List[Dict[str, Any]] = []

    for case in fixture.get("cases", []):
        case_outputs.append(_run_case(case, linalg=linalg, np=np))

    payload = {
        "packet_id": fixture.get("packet_id", "unknown"),
        "family": fixture.get("family", "unknown"),
        "generated_unix_ms": int(time.time() * 1000),
        "case_outputs": case_outputs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
