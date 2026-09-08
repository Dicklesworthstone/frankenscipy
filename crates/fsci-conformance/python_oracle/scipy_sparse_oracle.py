#!/usr/bin/env python3
"""SciPy-backed oracle capture for FrankenSciPy sparse helper fixtures."""

from __future__ import annotations

import argparse
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


def _coerce_maybe_nan_f64(value: Any) -> float:
    """Accept either a JSON number or a NaN/Inf string sentinel."""
    if isinstance(value, str):
        key = value.strip().lower()
        if key == "nan":
            return float("nan")
        if key in ("infinity", "inf", "+infinity", "+inf"):
            return float("inf")
        if key in ("-infinity", "-inf"):
            return float("-inf")
        return float(value)
    return float(value)


def _matrix_from_spec(spec: Dict[str, Any], sparse: Any, np: Any) -> Any:
    shape = tuple(spec["shape"])
    row = np.asarray(spec["row"], dtype=np.int64)
    col = np.asarray(spec["col"], dtype=np.int64)
    data = np.asarray([_coerce_maybe_nan_f64(v) for v in spec["data"]], dtype=np.float64)
    coo = sparse.coo_array((data, (row, col)), shape=shape)
    fmt = spec.get("format", "coo")
    if fmt == "coo":
        return coo
    if fmt == "csr":
        return coo.tocsr()
    if fmt == "csc":
        return coo.tocsc()
    raise ValueError(f"unsupported sparse fixture format: {fmt}")


def _matrix_result(matrix: Any) -> Dict[str, Any]:
    coo = matrix.tocoo()
    return {
        "format": getattr(matrix, "format", None),
        "shape": [int(coo.shape[0]), int(coo.shape[1])],
        "row": [int(v) for v in coo.row.tolist()],
        "col": [int(v) for v in coo.col.tolist()],
        "data": [float(v) for v in coo.data.tolist()],
    }


def _find_result(matrix: Any, sparse: Any) -> Dict[str, Any]:
    row, col, data = sparse.find(matrix)
    return {
        "row": [int(v) for v in row.tolist()],
        "col": [int(v) for v in col.tolist()],
        "data": [float(v) for v in data.tolist()],
    }


def _csr_components_result(matrix: Any) -> Dict[str, Any]:
    csr = matrix if getattr(matrix, "format", None) == "csr" else matrix.tocsr()
    return {
        "shape": [int(csr.shape[0]), int(csr.shape[1])],
        "data": [float(v) for v in csr.data.tolist()],
        "indices": [int(v) for v in csr.indices.tolist()],
        "indptr": [int(v) for v in csr.indptr.tolist()],
        "has_sorted_indices": bool(csr.has_sorted_indices),
        "has_canonical_format": bool(csr.has_canonical_format),
    }


def _run_case(case: Dict[str, Any], sparse: Any, np: Any) -> Dict[str, Any]:
    case_id = case["case_id"]
    operation = case["operation"]

    try:
        if operation == "find":
            matrix = _matrix_from_spec(case["matrix"], sparse, np)
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "find_triplets",
                "result": _find_result(matrix, sparse),
                "error": None,
            }

        if operation == "tril":
            matrix = _matrix_from_spec(case["matrix"], sparse, np)
            result = sparse.tril(matrix, k=int(case.get("k", 0)))
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "matrix_triplets",
                "result": _matrix_result(result),
                "error": None,
            }

        if operation == "triu":
            matrix = _matrix_from_spec(case["matrix"], sparse, np)
            result = sparse.triu(matrix, k=int(case.get("k", 0)))
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "matrix_triplets",
                "result": _matrix_result(result),
                "error": None,
            }

        if operation == "tocsc":
            matrix = _matrix_from_spec(case["matrix"], sparse, np)
            result = matrix.tocsc()
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "matrix_triplets",
                "result": _matrix_result(result),
                "error": None,
            }

        if operation == "vstack":
            blocks = [_matrix_from_spec(block, sparse, np) for block in case["blocks"]]
            result = sparse.vstack(blocks, format=case.get("format"))
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "matrix_triplets",
                "result": _matrix_result(result),
                "error": None,
            }

        if operation == "hstack":
            blocks = [_matrix_from_spec(block, sparse, np) for block in case["blocks"]]
            result = sparse.hstack(blocks, format=case.get("format"))
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "matrix_triplets",
                "result": _matrix_result(result),
                "error": None,
            }

        if operation == "csr_matmul":
            lhs = _matrix_from_spec(case["blocks"][0], sparse, np)
            rhs = _matrix_from_spec(case["blocks"][1], sparse, np)
            result = lhs @ rhs
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "csr_components",
                "result": _csr_components_result(result),
                "error": None,
            }

        # --- FSCI-P2C-004_sparse_ops dispatchers (per frankenscipy-dvd6) ---
        # The primary sparse conformance fixture uses a flat schema
        # (rows/cols/data/row_indices/col_indices/format/rhs/scalar) rather
        # than the nested "matrix": {...} schema used by the above 7
        # dispatchers. Build a scipy.sparse matrix from the flat schema and
        # run the requested operation, returning the result in the shape
        # the Rust runner already expects (vector/scalar/matrix_triplets/
        # csr_components per compare_sparse_outcome).
        if operation in {"eigsh", "eigs", "svds"}:
            # br-gorz: top-k iterative eigenvalue / SVD parity. Construct
            # the sparse matrix from flat schema, run scipy.sparse.linalg
            # routine, return only the eigenvalues / singular values
            # (ordering normalized via abs-desc sort on the Rust side).
            from scipy.sparse import coo_matrix
            from scipy.sparse.linalg import eigs, eigsh, svds

            rows = int(case["rows"])
            cols = int(case["cols"])
            data = [_coerce_maybe_nan_f64(v) for v in case.get("data", [])]
            row_indices = [int(v) for v in case.get("row_indices", [])]
            col_indices = [int(v) for v in case.get("col_indices", [])]
            mat = coo_matrix(
                (data, (row_indices, col_indices)), shape=(rows, cols)
            ).tocsr()
            k = int(case.get("k", 1))
            if operation == "eigsh":
                values = eigsh(mat, k=k, return_eigenvectors=False)
                values = sorted([float(v) for v in values], key=lambda v: abs(v), reverse=True)
            elif operation == "eigs":
                ev = eigs(mat, k=k, return_eigenvectors=False)
                values = sorted([float(v.real) for v in ev], key=lambda v: abs(v), reverse=True)
            else:  # svds
                # scipy.sparse.linalg.svds requires k < min(rows, cols)
                u, s, vt = svds(mat, k=k)
                values = sorted([float(v) for v in s], key=lambda v: abs(v), reverse=True)
            return {
                "case_id": case_id,
                "status": "ok",
                "result_kind": "eigenvalues_abs_sorted",
                "result": {"values": values},
                "error": None,
            }

        if operation in {"spmv", "spsolve", "add", "scale", "format_roundtrip"}:
            rows = int(case["rows"])
            cols = int(case["cols"])
            data = [_coerce_maybe_nan_f64(v) for v in case.get("data", [])]
            row_indices = [int(v) for v in case.get("row_indices", [])]
            col_indices = [int(v) for v in case.get("col_indices", [])]
            fmt = str(case.get("format", "csr")).lower()
            # Build a COO first, then coerce to requested format.
            coo = sparse.coo_matrix(
                (data, (row_indices, col_indices)),
                shape=(rows, cols),
            )
            if fmt == "csr":
                mat = coo.tocsr()
            elif fmt == "csc":
                mat = coo.tocsc()
            elif fmt == "coo":
                mat = coo
            elif fmt == "dok":
                mat = coo.todok()
            elif fmt == "lil":
                mat = coo.tolil()
            else:
                mat = coo.tocsr()

            if operation == "spmv":
                rhs = np.asarray([_coerce_maybe_nan_f64(v) for v in case.get("rhs", [])])
                y = mat @ rhs
                return {
                    "case_id": case_id,
                    "status": "ok",
                    "result_kind": "vector",
                    "result": {"values": [float(v) for v in y.tolist()]},
                    "error": None,
                }

            if operation == "spsolve":
                rhs = np.asarray([_coerce_maybe_nan_f64(v) for v in case.get("rhs", [])])
                from scipy.sparse import linalg as splinalg
                x = splinalg.spsolve(mat.tocsr(), rhs)
                return {
                    "case_id": case_id,
                    "status": "ok",
                    "result_kind": "vector",
                    "result": {"values": [float(v) for v in np.atleast_1d(x).tolist()]},
                    "error": None,
                }

            if operation == "scale":
                scalar = _coerce_maybe_nan_f64(case.get("scalar", 1.0))
                scaled = (mat.tocoo() * scalar).tocoo()
                return {
                    "case_id": case_id,
                    "status": "ok",
                    "result_kind": "matrix_triplets",
                    "result": _matrix_result(scaled),
                    "error": None,
                }

            if operation == "add":
                # fixture convention: rhs is either a list of data values
                # (same sparsity as lhs) OR None/missing meaning "add to
                # self" (shape-preservation test).
                rhs_raw = case.get("rhs")
                if rhs_raw is None or (hasattr(rhs_raw, "__len__") and len(rhs_raw) == 0):
                    rhs_mat = mat.tocsr()
                else:
                    rhs_data = [_coerce_maybe_nan_f64(v) for v in rhs_raw]
                    rhs_mat = sparse.coo_matrix(
                        (rhs_data, (row_indices, col_indices)),
                        shape=(rows, cols),
                    ).tocsr()
                result = (mat.tocsr() + rhs_mat).tocoo()
                return {
                    "case_id": case_id,
                    "status": "ok",
                    "result_kind": "matrix_triplets",
                    "result": _matrix_result(result),
                    "error": None,
                }

            if operation == "format_roundtrip":
                # Round-trip through CSR->COO->CSC->COO->requested format;
                # the expected output is structurally identical to input.
                result = mat.tocsr().tocoo().tocsc().tocoo()
                return {
                    "case_id": case_id,
                    "status": "ok",
                    "result_kind": "matrix_triplets",
                    "result": _matrix_result(result),
                    "error": None,
                }

        return {
            "case_id": case_id,
            "status": "error",
            "result_kind": "unsupported_operation",
            "result": {},
            "error": f"unsupported operation: {operation}",
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture SciPy sparse helper oracle outputs")
    parser.add_argument("--fixture", required=True, help="Input fixture JSON path")
    parser.add_argument("--output", required=True, help="Output oracle capture JSON path")
    parser.add_argument("--oracle-root", required=False, default="", help="Optional path to legacy scipy reference tree")
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    output_path = Path(args.output)

    try:
        import numpy as np
        from scipy import sparse
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
        case_outputs.append(_run_case(case, sparse=sparse, np=np))

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
        json.dumps(_json_safe(payload), allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
