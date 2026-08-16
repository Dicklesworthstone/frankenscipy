"""Incumbent arm for `sparse_edge_cases_match_scipy`. Re-derive rather than trust."""
import numpy as np, scipy, scipy.sparse as sp, scipy.sparse.csgraph as csg
print("scipy", scipy.__version__)
A = sp.csr_matrix(np.array([[1., 2., 0.], [0., 3., 4.], [5., 0., 6.]]))
print("tril k=+10 nnz:", sp.tril(A, 10).nnz, " tril k=-10 nnz:", sp.tril(A, -10).nnz)
print("triu k=+10 nnz:", sp.triu(A, 10).nnz, " triu k=-10 nnz:", sp.triu(A, -10).nnz)
print("A**0:\n", (A ** 0).toarray())
B = sp.csr_matrix(np.array([[1., 0.], [0., 2.]]))
print("kron(B,B) diag:   ", sp.kron(B, B).toarray().diagonal())
print("kronsum(B,B) diag:", sp.kronsum(B, B).toarray().diagonal())
g = sp.csr_matrix(np.array([[0., 1., 0., 0.], [0., 0., 0., 0.],
                           [0., 0., 0., 1.], [0., 0., 0., 0.]]))
print("connected_components weak:", csg.connected_components(g, directed=True, connection='weak'))
