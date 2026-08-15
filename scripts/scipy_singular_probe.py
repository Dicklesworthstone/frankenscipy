import warnings, numpy as np, scipy, scipy.sparse as sp, scipy.sparse.linalg as spl
warnings.filterwarnings("ignore")
print("scipy",scipy.__version__)

def show(name, fn):
    try:
        out = fn()
        print(f"  {name:34s} -> {out}")
    except Exception as e:
        print(f"  {name:34s} -> RAISES {type(e).__name__}: {str(e)[:70]}")

# 1. exactly singular square: rank 2 of 3
S = sp.csc_matrix(np.array([[1.,2.,3.],[2.,4.,6.],[1.,0.,1.]]))
b = np.array([1.,2.,1.])
print("== exactly singular 3x3 (row1 = 2*row0), consistent rhs ==")
show("spsolve", lambda: np.round(spl.spsolve(S,b),6))
show("splu", lambda: spl.splu(S) and "factorized")
show("lsqr", lambda: (np.round(spl.lsqr(S,b)[0],6), spl.lsqr(S,b)[1]))
show("lsmr", lambda: (np.round(spl.lsmr(S,b)[0],6), spl.lsmr(S,b)[1]))

# 2. inconsistent rhs on the same singular matrix
b2 = np.array([1.,3.,1.])
print("== same matrix, INCONSISTENT rhs ==")
show("spsolve", lambda: np.round(spl.spsolve(S,b2),6))
show("lsqr", lambda: (np.round(spl.lsqr(S,b2)[0],6), spl.lsqr(S,b2)[1]))

# 3. zero matrix
Z = sp.csc_matrix((3,3))
print("== all-zero matrix ==")
show("spsolve", lambda: np.round(spl.spsolve(Z,b),6))
show("lsqr", lambda: (np.round(spl.lsqr(Z,b)[0],6), spl.lsqr(Z,b)[1]))

# 4. rectangular overdetermined, full rank
R = sp.csr_matrix(np.array([[1.,0.],[0.,1.],[1.,1.]]))
br = np.array([1.,2.,4.])
print("== overdetermined 3x2 full rank (least squares) ==")
show("lsqr", lambda: (np.round(spl.lsqr(R,br)[0],8), spl.lsqr(R,br)[1], round(float(spl.lsqr(R,br)[3]),8)))
show("lsmr", lambda: (np.round(spl.lsmr(R,br)[0],8), spl.lsmr(R,br)[1]))

# 5. rank-deficient rectangular
D = sp.csr_matrix(np.array([[1.,2.],[2.,4.],[3.,6.]]))
bd = np.array([1.,2.,3.])
print("== rank-deficient 3x2 ==")
show("lsqr", lambda: (np.round(spl.lsqr(D,bd)[0],8), spl.lsqr(D,bd)[1]))
show("lsmr", lambda: (np.round(spl.lsmr(D,bd)[0],8), spl.lsmr(D,bd)[1]))
