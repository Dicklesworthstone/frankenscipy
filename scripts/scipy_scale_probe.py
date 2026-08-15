import numpy as np, scipy, scipy.sparse as sp, scipy.sparse.linalg as spl
print("scipy", scipy.__version__, "numpy", np.__version__)
SIDE=8; n=SIDE*SIDE
rows=[];cols=[];data=[]
for r in range(SIDE):
    for c in range(SIDE):
        i=r*SIDE+c
        def push(j,v): rows.append(i);cols.append(j);data.append(v)
        if r>0: push(i-SIDE,-1.0)
        if c>0: push(i-1,-1.2)
        push(i,4.001)
        if c+1<SIDE: push(i+1,-0.8)
        if r+1<SIDE: push(i+SIDE,-1.0)
A=sp.csr_matrix((data,(rows,cols)),shape=(n,n))
b=np.array([1.0+0.1*(i%7) for i in range(n)])
for scale in (1.0, 1e-15):
    As=A*scale; bs=b*scale
    print(f"--- scale={scale:g}  ||b||={np.linalg.norm(bs):.3e}")
    for name,fn in [("bicg",spl.bicg),("cgs",spl.cgs),("bicgstab",spl.bicgstab),("qmr",spl.qmr),("gmres",spl.gmres)]:
        x,info=fn(As,bs,rtol=1e-10,atol=0.0,maxiter=2000)
        rel=np.linalg.norm(As@x-bs)/np.linalg.norm(bs)
        print(f"   {name:9s} info={info:4d}  relative_residual={rel:.3e}")

# frankenscipy-pfet9: what does the incumbent do for a rhs whose norm is BELOW
# f64::EPSILON? Our solvers short-circuit there and return x = 0 with
# converged=true; the question this answers is whether SciPy does the same.
print("\n=== tiny-norm rhs: does the peer short-circuit? ===")
for bscale in (1e-17, 1e-20, 0.0):
    bs = b * bscale
    print(f"--- ||b||={np.linalg.norm(bs):.3e}")
    for name, fn in [("cg", spl.cg), ("gmres", spl.gmres)]:
        M = A if name != "cg" else (A + A.T) / 2 + sp.eye(n) * 8
        x, info = fn(M, bs, rtol=1e-10, atol=0.0, maxiter=2000)
        nb = np.linalg.norm(bs)
        rel = np.linalg.norm(M @ x - bs) / nb if nb > 0 else np.linalg.norm(M @ x)
        exact = spl.spsolve(sp.csc_matrix(M), bs) if nb > 0 else np.zeros(n)
        drift = np.linalg.norm(x - exact) / (np.linalg.norm(exact) if nb > 0 else 1.0)
        print(f"   {name:6s} info={info:3d} rel={rel:.3e} all_zero={np.all(x == 0)} "
              f"drift_vs_direct={drift:.3e}")

# frankenscipy-pfet9 item 2: the PIVOT GUARDS. Ours reject a pivot whose
# magnitude is below an ABSOLUTE floor (f64::EPSILON * 100 = 2.22e-14), so a
# uniformly scaled system makes every pivot look singular. The scale used here
# is a power of two, which scales a float exactly, so an implementation that is
# scale-invariant must return a bit-identically scaled answer -- there is no
# tolerance to argue about.
print("\n=== pivot guards: is the peer scale-invariant, and where does it fail? ===")
SCALE = 2.0 ** -60          # 8.674e-19, far below our absolute pivot floor
tri_rows, tri_cols, tri_vals = [], [], []
for i in range(n):
    tri_rows.append(i); tri_cols.append(i); tri_vals.append(2.0 + (i % 5) * 0.25)
    if i > 0:
        tri_rows.append(i); tri_cols.append(i - 1); tri_vals.append(-0.5 - (i % 3) * 0.125)
L = sp.csr_matrix((tri_vals, (tri_rows, tri_cols)), shape=(n, n))

x_unit = spl.spsolve_triangular(L, b, lower=True)
x_tiny = spl.spsolve_triangular(L * SCALE, b * SCALE, lower=True)
print(f"spsolve_triangular  min|diag| at scale 2^-60 = {np.min(np.abs(L.diagonal())) * SCALE:.3e}")
print(f"   solved at scale 1: yes    solved at scale 2^-60: yes")
print(f"   bit_identical_across_scale={np.array_equal(x_unit, x_tiny)}")

# The peer's own gate, so we know what it actually rejects rather than guessing.
for label, diag_value in (("exactly 0.0", 0.0), ("1e-300 (denormal-ish)", 1e-300)):
    Lz = L.copy().tolil(); Lz[n // 2, n // 2] = diag_value; Lz = Lz.tocsr()
    try:
        spl.spsolve_triangular(Lz, b, lower=True)
        verdict = "SOLVED (no error)"
    except Exception as exc:                                  # noqa: BLE001
        verdict = f"{type(exc).__name__}: {exc}"
    print(f"   diagonal {label:22s} -> {verdict}")

ilu_unit = spl.spilu(sp.csc_matrix(A), drop_tol=0.0, fill_factor=1.0)
ilu_tiny = spl.spilu(sp.csc_matrix(A * SCALE), drop_tol=0.0, fill_factor=1.0)
u_unit = np.sort(np.abs(ilu_unit.U.diagonal()))
u_tiny = np.sort(np.abs(ilu_tiny.U.diagonal()))
print(f"spilu               min|U diag| at scale 2^-60 = {u_tiny.min():.3e}")
print(f"   factored at scale 1: yes    factored at scale 2^-60: yes")
print(f"   U diagonals scale exactly: {np.array_equal(u_unit * SCALE, u_tiny)}")

# frankenscipy-xs7i2: the LEAST-SQUARES pair. Our lsmr returned the zero vector
# after zero iterations for a system it solves at scale 1, because it clamped
# alpha = ||A^T u|| against a bare f64::EPSILON. This section is the incumbent
# arm, and its point is the conlim=0 row: scipy's visible degradation at these
# scales is its stopping HEURISTIC, not its arithmetic.
print("\n=== least squares under scaling: heuristic or arithmetic? ===")
M, N = 96, 64
ls_rows, ls_cols, ls_vals = [], [], []
for i in range(M):
    ls_rows.append(i); ls_cols.append(i % N); ls_vals.append(3.0 + 0.25 * (i % 5))
    ls_rows.append(i); ls_cols.append((i * 7 + 3) % N); ls_vals.append(-0.75 - 0.125 * (i % 3))
A_ls = sp.csr_matrix((ls_vals, (ls_rows, ls_cols)), shape=(M, N))
x_true = np.array([1.0 + 0.1 * (i % 7) for i in range(N)])
b_ls = A_ls @ x_true

for k in (0, 40, 50, 52, 54, 56, 60):
    scale = 2.0 ** -k
    As, bs = A_ls * scale, b_ls * scale
    nb = np.linalg.norm(bs)
    row = [f"2^-{k:<3d}"]
    for name, kwargs in (
        ("lsqr", dict(atol=1e-12, btol=1e-12, iter_lim=2000)),
        ("lsmr", dict(atol=1e-12, btol=1e-12, maxiter=2000)),
        ("lsmr conlim=0", dict(atol=1e-12, btol=1e-12, conlim=0, maxiter=2000)),
    ):
        fn = spl.lsqr if name == "lsqr" else spl.lsmr
        out = fn(As, bs, **kwargs)
        rel = np.linalg.norm(As @ out[0] - bs) / nb
        row.append(f"{name}: istop={out[1]} iters={out[2]:4d} rel={rel:.3e}")
    print("   " + "  ".join(row))

# frankenscipy-ze5a6: fsci-linalg's ldl. Ours skipped any pivot under an
# ABSOLUTE f64::EPSILON*1e3 and returned Ok with a factorization that missed A
# by 7.1% relative from 2^-46 down. scipy permutes (Bunch-Kaufman) instead of
# clamping, so this arm is what "no such failure exists" looks like.
print("\n=== ldl under scaling: does the peer's factorization survive? ===")
import scipy.linalg as sla_dense
LDL_N = 8
A_ldl = np.array([[(6.0 + 0.5 * (i % 3)) if i == j else -1.0 / (1.0 + abs(i - j))
                   for j in range(LDL_N)] for i in range(LDL_N)])
print(f"   cond(A) = {np.linalg.cond(A_ldl):.6f}")
for k in (0, 20, 40, 44, 46, 50, 60):
    scale = 2.0 ** -k
    As = A_ldl * scale
    lu, d, _perm = sla_dense.ldl(As)
    err = np.linalg.norm(lu @ d @ lu.T - As) / np.linalg.norm(As)
    print(f"   2^-{k:<3d} scipy ldl relative_reconstruction_error={err:.3e} "
          f"min|d|={np.min(np.abs(np.diag(d))):.3e}")

# frankenscipy-kwi99: eig/eigvals. The 1x1-vs-2x2 test on the real Schur form
# decides real eigenvalue vs complex conjugate pair, and ours compared the
# subdiagonal against an ABSOLUTE eps*100. Below 2^-46 we reported four REAL
# eigenvalues for a matrix that has none. This arm is the equivariance the peer
# delivers at every scale.
print("\n=== eig: do complex pairs stay complex under scaling? ===")
A_eig = np.array([[0.0, -1.0, 0.0, 0.0],
                  [1.0,  0.0, 0.0, 0.0],
                  [0.0,  0.0, 2.0, -3.0],
                  [0.0,  0.0, 3.0,  2.0]])
print(f"   eigenvalues at scale 1: {np.round(sla_dense.eigvals(A_eig), 6)}")
for k in (0, 20, 40, 43, 45, 46, 50, 60):
    scale = 2.0 ** -k
    w = sla_dense.eigvals(A_eig * scale)
    print(f"   2^-{k:<3d} scipy max|Im|={np.max(np.abs(w.imag)):.6e}  expected={3.0 * scale:.6e}  "
          f"all_real={np.all(w.imag == 0)}")

# frankenscipy-i8gy5: ordqz. Ours gated the QZ beta -- the diagonal of the B
# factor, carrying the scale of the pencil -- against an absolute eps, so a
# scaled pencil had every eigenvalue marked unselected and came back UNSORTED.
# Scaling both A and B leaves every ratio unchanged, so the sort must not move.
print("\n=== ordqz: does the peer still sort a scaled pencil? ===")
A_qz = np.diag([2.0, 0.25, -3.0])
B_qz = np.eye(3)
for k in (0, 20, 50, 60):
    scale = 2.0 ** -k
    AA, BB, _al, _be, _Q, _Z = sla_dense.ordqz(A_qz * scale, B_qz * scale, sort='iuc')
    ratios = np.array([AA[i, i] / BB[i, i] for i in range(3)])
    print(f"   2^-{k:<3d} scipy ordqz(iuc) ratios={np.round(ratios, 6)}")
