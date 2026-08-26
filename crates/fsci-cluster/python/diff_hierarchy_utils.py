"""SciPy comparator for diff_hierarchy_utils.

Emits the same `name|v;v;v` lines as the Rust probe. Inputs MUST match the Rust side exactly.
Covers scipy.cluster.hierarchy cut_tree, leaders, maxdists, maxinconsts, is_isomorphic and
from_mlab_linkage -- entry points with no differential coverage (frankenscipy-ivxx6).

`linkage` is emitted as a CONTROL: every other function here consumes Z, so if Z disagreed the
downstream groups would too and the probe would blame the wrong function.
"""
import numpy as np
from scipy.cluster.hierarchy import (cut_tree, from_mlab_linkage, inconsistent,
                                     is_isomorphic, leaders, linkage, maxdists,
                                     maxinconsts)

DATA = np.array([
    [0.0, 0.0], [0.3, 0.2], [0.1, 0.4],
    [3.0, 3.1], [3.4, 2.8], [3.1, 3.5],
    [6.2, 0.4], [6.0, 0.1], [6.5, 0.6],
    [1.7, 5.9], [2.0, 6.2], [9.4, 9.1],
])

MLAB = np.array([
    [1.0, 2.0, 0.5],
    [3.0, 4.0, 0.8],
    [5.0, 6.0, 1.4],
])


def dump(name, arr):
    flat = np.asarray(arr, dtype=float).ravel()
    print(f"{name}|" + ";".join(f"{v:.17e}" for v in flat))


def canonical(labels):
    seen, out, nxt = {}, [], 0
    for l in np.asarray(labels).ravel():
        l = int(l)
        if l not in seen:
            seen[l] = nxt
            nxt += 1
        out.append(float(seen[l]))
    return out


def main():
    z = linkage(DATA, method="average")
    dump("linkage_average", z)
    dump("maxdists", maxdists(z))

    r = inconsistent(z, 2)
    dump("inconsistent_d2", r)
    dump("maxinconsts_d2", maxinconsts(z, r))

    for k in (2, 3, 4, 6):
        labels = cut_tree(z, n_clusters=k).ravel()
        dump(f"cuttree_k{k}_raw", labels)
        dump(f"cuttree_k{k}_canon", canonical(labels))

    t = cut_tree(z, n_clusters=3).ravel() + 1
    L, M = leaders(z, t.astype(np.int32))
    dump("leaders_L", L)
    dump("leaders_M", M)

    a = [0, 0, 1, 1, 2, 2]
    relabelled = [5, 5, 9, 9, 1, 1]
    different = [0, 1, 1, 1, 2, 2]
    dump("isomorphic", [float(is_isomorphic(a, relabelled)),
                        float(is_isomorphic(a, different)),
                        float(is_isomorphic(a, a))])

    dump("from_mlab", from_mlab_linkage(MLAB))


if __name__ == "__main__":
    main()
