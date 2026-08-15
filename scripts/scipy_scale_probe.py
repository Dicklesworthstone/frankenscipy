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
