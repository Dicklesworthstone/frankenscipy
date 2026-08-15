"""Incumbent arm for the csgraph semantics pinned in
`csgraph_self_loops_and_unreachable_nodes_match_scipy` (frankenscipy).
Run it to re-derive the expectations rather than trusting the numbers."""
import numpy as np, scipy, scipy.sparse as sp, scipy.sparse.csgraph as csg
np.set_printoptions(suppress=True, linewidth=120)
print("scipy", scipy.__version__)
g = np.array([[5., 1., 0.], [0., 0., 2.], [0., 0., 0.]])   # self-loop w=5 on node 0
G = sp.csr_matrix(g)
print("self-loop graph floyd_warshall:\n", csg.floyd_warshall(G, directed=True))
print("dijkstra(0):    ", csg.dijkstra(G, directed=True, indices=0))
print("bellman_ford(0):", csg.bellman_ford(G, directed=True, indices=0))
d = np.array([[0., 1., 0., 0.], [0., 0., 2., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]])
D = sp.csr_matrix(d)
print("disconnected floyd_warshall:\n", csg.floyd_warshall(D, directed=True))
print("disconnected dijkstra(0):", csg.dijkstra(D, directed=True, indices=0))
