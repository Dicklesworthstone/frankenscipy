"""SciPy comparator for diff_disttransform_splinefilter.

Emits the same `name|v;v;v` lines as the Rust probe. Inputs MUST match the Rust side exactly.
Covers scipy.ndimage.distance_transform_bf, distance_transform_cdt and spline_filter1d --
three entry points with no differential coverage that existing perf bins already time
(frankenscipy-ivxx6).
"""
import numpy as np
from scipy.ndimage import (binary_dilation, distance_transform_bf,
                           distance_transform_cdt, spline_filter1d)

IMG = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1],
    [0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0],
], dtype=float)

RAMP = np.array([np.sin(0.7 * i) * 3.0 + 0.25 * i for i in range(12)])


def dump(name, arr):
    flat = np.asarray(arr, dtype=float).ravel()
    print(f"{name}|" + ";".join(f"{v:.17e}" for v in flat))


DENSE = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 1],
    [1, 1, 1, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 1, 0, 1, 1, 0],
    [1, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 0, 1, 1, 1],
], dtype=bool)

DENSE_ST = np.array([
    [1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1],
], dtype=bool)


def main():
    dump("dilate_asym", binary_dilation(DENSE, structure=DENSE_ST).astype(float))
    for mname, metric in (("euclidean", "euclidean"),
                          ("taxicab", "taxicab"),
                          ("chessboard", "chessboard")):
        dump(f"bf_{mname}", distance_transform_bf(IMG, metric=metric))

    dump("bf_euclidean_sampling",
         distance_transform_bf(IMG, metric="euclidean", sampling=[2.0, 0.5]))

    for mname, metric in (("taxicab", "taxicab"), ("chessboard", "chessboard")):
        dump(f"cdt_{mname}", distance_transform_cdt(IMG, metric=metric))

    for order in (2, 3, 4, 5):
        for bname, mode in (("reflect", "reflect"), ("nearest", "nearest"),
                            ("wrap", "wrap"), ("mirror", "mirror"),
                            ("constant", "constant")):
            dump(f"spf1d_o{order}_{bname}",
                 spline_filter1d(RAMP, order=order, axis=0, mode=mode))


if __name__ == "__main__":
    main()
