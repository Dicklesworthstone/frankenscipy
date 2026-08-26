"""SciPy comparator for diff_convolve2d_cola_c2d.

Emits the same `name|v;v;v` lines as the Rust probe. Inputs MUST match the Rust side exactly.
Covers scipy.signal.convolve2d, check_COLA, check_NOLA and cont2discrete -- four entry points
with no differential coverage (frankenscipy-ivxx6).

The windows are built with numpy rather than by calling scipy's window helpers, so a window
convention difference cannot be mistaken for a COLA/NOLA divergence.
"""
import numpy as np
from scipy.signal import check_COLA, check_NOLA, cont2discrete, convolve2d

A = np.array([
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [6.0, 7.0, 8.0, 9.0, 1.0],
    [2.0, 4.0, 6.0, 8.0, 3.0],
    [5.0, 7.0, 9.0, 2.0, 4.0],
])
V = np.array([
    [1.0, -2.0],
    [3.0, 0.5],
    [-1.5, 2.5],
])

N = 16
HANN = np.sin(np.pi * np.arange(N) / N) ** 2
BOXCAR = np.ones(N)

NUM = [1.0]
DEN = [1.0, 0.7, 1.0]
DT = 0.1


def dump(name, arr):
    flat = np.asarray(arr, dtype=float).ravel()
    print(f"{name}|" + ";".join(f"{v:.17e}" for v in flat))


def main():
    for mode in ("full", "same", "valid"):
        dump(f"conv2d_{mode}", convolve2d(A, V, mode=mode))

    for bname, boundary in (("fill", "fill"), ("wrap", "wrap"), ("symm", "symm")):
        dump(f"conv2d_same_{bname}",
             convolve2d(A, V, mode="same", boundary=boundary, fillvalue=0.0))

    cola, nola = [], []
    for w in (HANN, BOXCAR):
        for noverlap in (0, 4, 8, 12, 13):
            try:
                cola.append(1.0 if check_COLA(w, N, noverlap) else 0.0)
            except Exception:
                cola.append(-1.0)
            try:
                nola.append(1.0 if check_NOLA(w, N, noverlap) else 0.0)
            except Exception:
                nola.append(-1.0)
    dump("check_cola", cola)
    dump("check_nola", nola)

    for method in ("zoh", "bilinear", "euler", "backward_diff"):
        numd, dend, _dt = cont2discrete((NUM, DEN), DT, method=method)
        dump(f"c2d_{method}_num", numd)
        dump(f"c2d_{method}_den", dend)

    numd, dend, _dt = cont2discrete((NUM, DEN), DT, method="gbt", alpha=0.3)
    dump("c2d_gbt03_num", numd)
    dump("c2d_gbt03_den", dend)


if __name__ == "__main__":
    main()
