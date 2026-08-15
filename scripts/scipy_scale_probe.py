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
