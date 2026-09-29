#!/usr/bin/env python3
"""SciPy-backed oracle capture for FrankenSciPy spatial fixture.

Closes the spatial slice of frankenscipy-di9p. Covers distance
functions (pdist / cdist / euclidean / cityblock / chebyshev / cosine /
correlation), squareform, kdtree_query, directed_hausdorff,
convex_hull, halfspace_intersection, procrustes, geometric_slerp.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _run_case(case: Dict[str, Any], np: Any, spatial: Any, distance: Any) -> Dict[str, Any]:
    case_id = case.get("case_id", "<missing>")
    function = case.get("function", "<missing>")
    args = case.get("args", [])

    try:
        if function == "pdist":
            x = np.asarray(args[0], dtype=float)
            metric = args[1] if len(args) > 1 else "euclidean"
            result = distance.pdist(x, metric=metric)
            return _ok(case_id, "array", {"values": [float(v) for v in result.tolist()]})

        if function == "cdist":
            xa = np.asarray(args[0], dtype=float)
            xb = np.asarray(args[1], dtype=float)
            metric = args[2] if len(args) > 2 else "euclidean"
            result = distance.cdist(xa, xb, metric=metric)
            return _ok(case_id, "matrix", {
                "values": [[float(v) for v in row] for row in result.tolist()],
            })

        if function == "squareform_to_matrix":
            condensed = np.asarray(args[0], dtype=float)
            result = distance.squareform(condensed)
            return _ok(case_id, "matrix", {
                "values": [[float(v) for v in row] for row in result.tolist()],
            })

        if function == "kdtree_query":
            points = np.asarray(args[0], dtype=float)
            query = np.asarray(args[1], dtype=float)
            tree = spatial.cKDTree(points)
            dist, idx = tree.query(query)
            return _ok(case_id, "kdtree_result", {
                "distance": float(dist) if np.isscalar(dist) else [float(v) for v in np.atleast_1d(dist).tolist()],
                "index": int(idx) if np.isscalar(idx) else [int(v) for v in np.atleast_1d(idx).tolist()],
            })

        if function == "kdtree_query_k":
            # br-nmh2: k-nearest with k > 1.
            points = np.asarray(args[0], dtype=float)
            query = np.asarray(args[1], dtype=float)
            k = int(args[2])
            tree = spatial.cKDTree(points)
            dist, idx = tree.query(query, k=k)
            return _ok(case_id, "kdtree_query_k_result", {
                "indices": [int(v) for v in np.atleast_1d(idx).tolist()],
                "distances": [float(v) for v in np.atleast_1d(dist).tolist()],
            })

        if function == "kdtree_query_ball_point":
            # br-nmh2: all indices within radius r.
            points = np.asarray(args[0], dtype=float)
            query = np.asarray(args[1], dtype=float)
            r = float(args[2])
            tree = spatial.cKDTree(points)
            indices = sorted(int(v) for v in tree.query_ball_point(query, r))
            return _ok(case_id, "kdtree_ball_point_result", {
                "indices": indices,
            })

        if function == "directed_hausdorff":
            a = np.asarray(args[0], dtype=float)
            b = np.asarray(args[1], dtype=float)
            dist, _i, _j = spatial.distance.directed_hausdorff(a, b)
            return _ok(case_id, "scalar", {"value": float(dist)})

        if function == "convex_hull":
            points = np.asarray(args[0], dtype=float)
            hull = spatial.ConvexHull(points)
            return _ok(case_id, "convex_hull", {
                "vertices": [int(v) for v in hull.vertices.tolist()],
                "volume": float(hull.volume),
                "area": float(hull.area),
            })

        if function == "voronoi":
            # br-d1jx: scipy.spatial.Voronoi parity. Emit only
            # permutation-invariant scalars + lex-sorted vertex
            # multiset since region / ridge orderings have no
            # canonical sort. Cospherical inputs collapse to one
            # vertex per merged cell, in scipy's Qhull and in fsci.
            points = np.asarray(args[0], dtype=float)
            v = spatial.Voronoi(points)
            sorted_v = sorted(v.vertices.tolist(), key=lambda p: (p[0], p[1]))
            return _ok(case_id, "voronoi_result", {
                "vertex_count": int(len(v.vertices)),
                "ridge_count": int(len(v.ridge_points)),
                "sorted_vertices": [[float(p[0]), float(p[1])] for p in sorted_v],
            })

        if function == "halfspace_intersection":
            halfspaces = np.asarray(args[0], dtype=float)
            interior_point = np.asarray(args[1], dtype=float)
            hs = spatial.HalfspaceIntersection(halfspaces, interior_point)
            return _ok(case_id, "halfspace_intersection", {
                "intersections": [[float(v) for v in row] for row in hs.intersections.tolist()],
            })

        if function == "procrustes":
            data1 = np.asarray(args[0], dtype=float)
            data2 = np.asarray(args[1], dtype=float)
            mtx1, mtx2, disparity = spatial.procrustes(data1, data2)
            return _ok(case_id, "procrustes_result", {
                "disparity": float(disparity),
                "mtx1": [[float(v) for v in row] for row in mtx1.tolist()],
                "mtx2": [[float(v) for v in row] for row in mtx2.tolist()],
            })

        if function == "geometric_slerp":
            start = np.asarray(args[0], dtype=float)
            end = np.asarray(args[1], dtype=float)
            t = np.asarray(args[2], dtype=float)
            result = spatial.geometric_slerp(start, end, t)
            return _ok(case_id, "matrix", {
                "values": [[float(v) for v in row] for row in result.tolist()],
            })

        # Scalar pairwise-distance functions (br-nmh2 expanded set).
        if function in {"euclidean", "cityblock", "chebyshev", "cosine", "correlation",
                        "hamming", "jaccard", "canberra", "braycurtis", "sqeuclidean"}:
            a = np.asarray(args[0], dtype=float)
            b = np.asarray(args[1], dtype=float)
            fn = getattr(distance, function)
            return _ok(case_id, "scalar", {"value": float(fn(a, b))})

        if function == "minkowski":
            a = np.asarray(args[0], dtype=float)
            b = np.asarray(args[1], dtype=float)
            p = float(args[2])
            return _ok(case_id, "scalar", {"value": float(distance.minkowski(a, b, p))})

        if function == "seuclidean":
            a = np.asarray(args[0], dtype=float)
            b = np.asarray(args[1], dtype=float)
            v = np.asarray(args[2], dtype=float)
            return _ok(case_id, "scalar", {"value": float(distance.seuclidean(a, b, v))})

        if function == "mahalanobis":
            a = np.asarray(args[0], dtype=float)
            b = np.asarray(args[1], dtype=float)
            vi = np.asarray(args[2], dtype=float)
            return _ok(case_id, "scalar", {"value": float(distance.mahalanobis(a, b, vi))})

        if function == "wminkowski":
            # scipy 1.10+ removed wminkowski in favor of minkowski(w=...).
            # The fsci impl follows the legacy convention
            #   ( Σᵢ (wᵢ · |aᵢ-bᵢ|)^p )^(1/p)
            # which differs from the modern minkowski(w=…) that applies
            # weights AFTER raising to p. Match the legacy form so both
            # sides agree.
            a = np.asarray(args[0], dtype=float)
            b = np.asarray(args[1], dtype=float)
            p = float(args[2])
            w = np.asarray(args[3], dtype=float)
            value = float((np.sum((w * np.abs(a - b)) ** p)) ** (1.0 / p))
            return _ok(case_id, "scalar", {"value": value})

        return {
            "case_id": case_id,
            "status": "error",
            "result_kind": "unsupported_function",
            "result": {},
            "error": f"unsupported function: {function}",
        }

    # RuntimeError catches scipy.spatial._qhull.QhullError (RuntimeError
    # subclass) for degenerate halfspace/convex-hull inputs. We deliberately
    # do NOT catch bare Exception — MemoryError/RecursionError/OSError
    # indicate oracle-side failure and should propagate per br-p3be.
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


def _ok(case_id: str, result_kind: str, result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "status": "ok",
        "result_kind": result_kind,
        "result": result,
        "error": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture SciPy spatial oracle outputs")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--oracle-root", required=False, default="")
    args = parser.parse_args()

    try:
        import numpy as np
        from scipy import spatial
        from scipy.spatial import distance
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
        case_outputs.append(_run_case(case, np=np, spatial=spatial, distance=distance))

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
