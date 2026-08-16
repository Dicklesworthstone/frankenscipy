"""Incumbent arm for `laplacian_matches_scipy_in_both_modes`."""
import numpy as np, scipy, scipy.sparse as sp, scipy.sparse.csgraph as csg
np.set_printoptions(suppress=True, linewidth=130)
print("scipy", scipy.__version__)
for label, g in [
    ("star", np.array([[0., 1., 1.], [1., 0., 0.], [1., 0., 0.]])),
    ("isolated node 2", np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])),
    ("weighted", np.array([[0., 2., 3.], [2., 0., 0.], [3., 0., 0.]])),
]:
    G = sp.csr_matrix(g)
    print(f"-- {label} normed=False:\n", csg.laplacian(G).toarray())
    print(f"-- {label} normed=True:\n", np.round(csg.laplacian(G, normed=True).toarray(), 9))
