"""SciPy comparator for diff_minkowski_tsearch_svoronoi.

Emits the same `name|v;v;v` lines as the Rust probe. Inputs MUST match the Rust side exactly.
Covers scipy.spatial minkowski_distance, minkowski_distance_p, tsearch and SphericalVoronoi --
entry points with no differential coverage (frankenscipy-ivxx6).

`tsearch` simplex indices and `SphericalVoronoi` vertex/region ORDER are implementation-dependent,
so this emits the same order-independent invariants the Rust side does: inside/outside for
tsearch, and vertex count / lexicographically sorted vertices / sorted region sizes for the
Voronoi diagram. Comparing the raw indices would manufacture a divergence out of a legal
difference in triangulation.
"""
import numpy as np
from scipy.spatial import (Delaunay, SphericalVoronoi, minkowski_distance,
                           minkowski_distance_p, tsearch)

XA = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 2.0, -1.0],
    [-3.0, 0.5, 2.0],
    [4.0, -4.0, 1.5],
])
XB = np.array([
    [1.0, -1.0, 2.0],
    [0.0, 0.0, 0.0],
    [2.5, 1.5, -2.0],
    [-1.0, 3.0, 0.0],
])

PTS = np.array([
    [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
    [0.35, 0.45], [0.7, 0.25], [0.25, 0.75],
])
QUERIES = np.array([
    [0.5, 0.5], [0.2, 0.2], [0.85, 0.6], [0.1, 0.9], [0.5, 0.05],
    [-0.5, 0.5], [1.5, 0.5], [0.5, -0.4],
])

SPHERE = np.array([
    [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
    [0.57735026918962584, 0.57735026918962584, 0.57735026918962584],
    [-0.57735026918962584, -0.57735026918962584, 0.57735026918962584],
])


def dump(name, arr):
    flat = np.asarray(arr, dtype=float).ravel()
    print(f"{name}|" + ";".join(f"{v:.17e}" for v in flat))


def main():
    for p in (0.5, 1.0, 1.5, 2.0, 3.0):
        tag = str(p).replace(".", "_")
        dump(f"mink_p{tag}", minkowski_distance(XA, XB, p))
        dump(f"minkp_p{tag}", minkowski_distance_p(XA, XB, p))
    dump("mink_pinf", minkowski_distance(XA, XB, np.inf))

    tri = Delaunay(PTS)
    found = tsearch(tri, QUERIES)
    dump("tsearch_inside", (found >= 0).astype(float))
    # The containment property is checked on the Rust side against its OWN triangulation; SciPy's
    # value here is 1.0 by construction, and is emitted so the group exists on both sides.
    dump("tsearch_containment", [1.0])

    sv = SphericalVoronoi(SPHERE, radius=1.0, center=np.zeros(3))
    dump("svor_nvertices", [float(len(sv.vertices))])
    order = np.lexsort((sv.vertices[:, 2], sv.vertices[:, 1], sv.vertices[:, 0]))
    dump("svor_vertices_sorted", sv.vertices[order])
    dump("svor_region_sizes_sorted", sorted(float(len(r)) for r in sv.regions))
    dev = np.max(np.abs(np.linalg.norm(sv.vertices, axis=1) - 1.0))
    dump("svor_on_sphere_maxdev", [dev])


if __name__ == "__main__":
    main()
