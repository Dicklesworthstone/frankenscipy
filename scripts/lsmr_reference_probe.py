"""Transcription of fsci-sparse lsmr() into Python, to check the frankenscipy-7crv5
criterion arithmetic without the fleet. Mirrors the Rust line for line."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl

def sym_orth(a, b):
    if b == 0.0:
        if a == 0.0: return (1.0, 0.0, 0.0)
        return (np.sign(a) if a else 1.0, 0.0, abs(a))
    if a == 0.0: return (0.0, np.sign(b), abs(b))
    r = np.hypot(a, b)
    return (a/r, b/r, r)

def fsci_lsmr(A, b, tol, max_iter, use_new_criterion):
    m, n = A.shape
    b_norm = np.linalg.norm(b)
    if b_norm == 0.0: return np.zeros(n), True, 0
    u = b / b_norm
    v = A.T @ u
    alpha = np.linalg.norm(v)
    if alpha == 0.0: return np.zeros(n), 1.0 <= tol, 0
    v = v / alpha
    beta = b_norm; alpha_bar = alpha; rho = 1.0; rho_bar = 1.0
    c_bar = 1.0; s_bar = 0.0; zeta_bar = alpha*beta
    h = v.copy(); h_bar = np.zeros(n)
    beta_dd = beta; beta_d = 0.0; rho_d_old = 1.0
    tau_tilde_old = 0.0; theta_tilde = 0.0; zeta = 0.0
    residual_squared = 0.0; a_norm_sq = 0.0
    x = np.zeros(n)
    for it in range(max_iter):
        u = A @ v - alpha*u
        beta = np.linalg.norm(u)
        if beta > 0.0:
            u = u/beta
            v = A.T @ u - beta*v
            alpha = np.linalg.norm(v)
            if alpha > 0.0: v = v/alpha
        c_hat, s_hat, alpha_hat = sym_orth(alpha_bar, 0.0)
        rho_old = rho
        c, s, rho = sym_orth(alpha_hat, beta)
        theta_new = s*alpha; alpha_bar = c*alpha
        rho_bar_old = rho_bar; zeta_old = zeta
        theta_bar = s_bar*rho; rho_temp = c_bar*rho
        c_bar, s_bar, rho_bar = sym_orth(rho_temp, theta_new)
        if rho == 0.0 or rho_bar == 0.0:
            return x, np.linalg.norm(A@x-b)/b_norm <= tol, it+1
        zeta = c_bar*zeta_bar; zeta_bar *= -s_bar
        hbs = -(theta_bar*rho/(rho_old*rho_bar_old))
        h_bar = h + hbs*h_bar
        x = x + zeta/(rho*rho_bar)*h_bar
        h = v - theta_new/rho*h
        beta_acute = c_hat*beta_dd; beta_check = -s_hat*beta_dd
        beta_hat = c*beta_acute; beta_dd = -s*beta_acute
        theta_tilde_old = theta_tilde
        c_tilde_old, s_tilde_old, rho_tilde_old = sym_orth(rho_d_old, theta_bar)
        theta_tilde = s_tilde_old*rho_bar; rho_d_old = c_tilde_old*rho_bar
        beta_d = -s_tilde_old*beta_d + c_tilde_old*beta_hat
        tau_tilde_old = (zeta_old - theta_tilde_old*tau_tilde_old)/rho_tilde_old
        tau_d = (zeta - theta_tilde*tau_tilde_old)/rho_d_old
        residual_squared += beta_check*beta_check
        est = np.sqrt(residual_squared + (beta_d-tau_d)**2 + beta_dd**2)/b_norm
        if use_new_criterion:
            a_norm_sq += alpha*alpha + beta*beta
            a_norm = np.sqrt(a_norm_sq)
            normar = abs(zeta_bar); normr = est*b_norm
            if normar > 0 and a_norm > 0 and normr > 0 and normar/(a_norm*normr) <= tol:
                return x, True, it+1
        if est <= tol or alpha == 0.0 or beta == 0.0:
            return x, np.linalg.norm(A@x-b)/b_norm <= tol, it+1
    return x, np.linalg.norm(A@x-b)/b_norm <= tol, max_iter

A=np.array([[1.,2.,3.],[2.,4.,6.],[1.,0.,1.]]); b=np.array([1.,3.,1.])
Asp=sp.csr_matrix(A)
want=np.linalg.pinv(A)@b
print("target (pinv/scipy):", np.round(want,6))
for flag in (False, True):
    x,conv,it=fsci_lsmr(A,b,1e-12,500,flag)
    print(f"  new_criterion={str(flag):5s} -> x={np.round(x,6)} conv={conv} iters={it} "
          f"drift={np.linalg.norm(x-want)/np.linalg.norm(want):.3e}")
# also confirm the consistent cases are untouched by the new criterion
for label,A2,b2 in [("singular consistent",A,np.array([1.,2.,1.])),
                    ("overdetermined",np.array([[1.,0.],[0.,1.],[1.,1.]]),np.array([1.,2.,4.])),
                    ("rank-deficient 3x2",np.array([[1.,2.],[2.,4.],[3.,6.]]),np.array([1.,2.,3.]))]:
    w=np.linalg.pinv(A2)@b2
    xa,_,ia=fsci_lsmr(A2,b2,1e-12,500,False); xb,_,ib=fsci_lsmr(A2,b2,1e-12,500,True)
    print(f"  {label:20s} old={np.round(xa,6)} ({ia} it)  new={np.round(xb,6)} ({ib} it)  "
          f"target={np.round(w,6)}")
