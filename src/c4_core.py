# -*- coding: utf-8 -*-
"""Cycle 4 core: generalized instrumented top88 (BC/filter/mesh) + spectral tools."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, eigs, LinearOperator
from scipy.linalg import eig

E0, Emin, nu, XMIN = 1.0, 1e-9, 0.3, 1e-3
P, MOVE = 3.0, 0.2

def lk():
    k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,
                  -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
    return 1/(1-nu**2)*np.array([
        [k[0],k[1],k[2],k[3],k[4],k[5],k[6],k[7]],
        [k[1],k[0],k[7],k[6],k[5],k[4],k[3],k[2]],
        [k[2],k[7],k[0],k[5],k[6],k[3],k[4],k[1]],
        [k[3],k[6],k[5],k[0],k[7],k[2],k[1],k[4]],
        [k[4],k[5],k[6],k[7],k[0],k[1],k[2],k[3]],
        [k[5],k[4],k[3],k[2],k[1],k[0],k[7],k[6]],
        [k[6],k[3],k[4],k[1],k[2],k[7],k[0],k[5]],
        [k[7],k[2],k[1],k[4],k[3],k[6],k[5],k[0]]])

def setup(nelx, nely, rmin, bc="mbb"):
    ndof = 2*(nelx+1)*(nely+1)
    elx, ely = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
    n1 = ((nely+1)*elx + ely).ravel()
    n2 = ((nely+1)*(elx+1) + ely).ravel()
    edof = np.column_stack([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,
                            2*n2, 2*n2+1, 2*n1, 2*n1+1])
    iK = np.kron(edof, np.ones((8,1), dtype=int)).flatten()
    jK = np.kron(edof, np.ones((1,8), dtype=int)).flatten()
    r = int(np.ceil(rmin)); iH, jH, sH = [], [], []
    for i in range(nelx):
        for j in range(nely):
            row = i*nely + j
            for kk in range(max(i-(r-1), 0), min(i+r, nelx)):
                for ll in range(max(j-(r-1), 0), min(j+r, nely)):
                    fac = rmin - np.hypot(i-kk, j-ll)
                    if fac > 0:
                        iH.append(row); jH.append(kk*nely+ll); sH.append(fac)
    H = coo_matrix((sH, (iH, jH)), shape=(nelx*nely, nelx*nely)).tocsr()
    Hs = np.asarray(H.sum(1)).ravel()
    f = np.zeros(ndof)
    if bc == "mbb":
        f[1] = -1.0
        fixed = np.union1d(np.arange(0, 2*(nely+1), 2), np.array([ndof-1]))
    elif bc == "cantilever":
        fixed = np.arange(0, 2*(nely+1))
        nmid = (nely+1)*nelx + nely//2
        f[2*nmid+1] = -1.0
    else:
        raise ValueError(bc)
    free = np.setdiff1d(np.arange(ndof), fixed)
    # reduced assembly indices (free-free block only)
    red = -np.ones(ndof, dtype=int); red[free] = np.arange(free.size)
    mK = (red[iK] >= 0) & (red[jK] >= 0)
    return dict(KE=lk(), edof=edof, H=H, Hs=Hs, f=f, free=free, ndof=ndof,
                nel=nelx*nely, nelx=nelx, nely=nely, nfree=free.size,
                iKr=red[iK[mK]], jKr=red[jK[mK]], mK=mK)

def step(x, volfrac, eta, S, filt):
    xph = np.asarray(S["H"]@x).ravel()/S["Hs"] if filt == "dens" else x
    sK = ((S["KE"].flatten()[np.newaxis]).T*(Emin + xph**P*(E0-Emin))).flatten(order="F")
    Kff = coo_matrix((sK[S["mK"]], (S["iKr"], S["jKr"])),
                     shape=(S["nfree"], S["nfree"])).tocsc()
    u = np.zeros(S["ndof"])
    u[S["free"]] = spsolve(Kff, S["f"][S["free"]], permc_spec="MMD_AT_PLUS_A")
    ce = np.einsum("ij,jk,ik->i", u[S["edof"]], S["KE"], u[S["edof"]])
    dc = -P*xph**(P-1)*(E0-Emin)*ce
    if filt == "sens":
        dcf = np.asarray(S["H"]@(x*dc)).ravel()/S["Hs"]/np.maximum(1e-3, x)
        dvf = np.ones_like(x)
    else:
        dcf = np.asarray(S["H"]@(dc/S["Hs"])).ravel()
        dvf = np.asarray(S["H"]@(np.ones_like(x)/S["Hs"])).ravel()
    l1, l2 = 0.0, 1e9
    lo = np.maximum(XMIN, x-MOVE); hi = np.minimum(1.0, x+MOVE)
    while (l2-l1)/(l1+l2) > 1e-10:
        lm = 0.5*(l1+l2)
        B = np.maximum(1e-30, -dcf/(dvf*lm))
        xn = np.clip(x*B**eta, lo, hi)
        vol = (np.asarray(S["H"]@xn).ravel()/S["Hs"]).mean() if filt == "dens" else xn.mean()
        if vol > volfrac: l1 = lm
        else: l2 = lm
    return xn

def converge(S, volfrac, eta0, filt, iters=200):
    x = volfrac*np.ones(S["nel"])
    for _ in range(iters):
        x = step(x, volfrac, eta0, S, filt)
    amps = []
    for _ in range(10):
        xn = step(x, volfrac, eta0, S, filt)
        amps.append(np.mean(np.abs(xn-x))); x = xn
    return x, float(np.median(amps))

def free_set(x0, tol=2e-3):
    return np.where((x0 > XMIN+tol) & (x0 < 1-tol))[0]

def build_J(x0, S, volfrac, filt, eta=0.5, eps=1e-5):
    F = free_set(x0)
    base = np.log(step(x0, volfrac, eta, S, filt)[F])
    J = np.empty((F.size, F.size))
    for j, e in enumerate(F):
        xp = x0.copy(); xp[e] *= np.exp(eps)
        J[:, j] = (np.log(step(xp, volfrac, eta, S, filt)[F]) - base)/eps
    return J, F

def spectrum(J, eta=0.5):
    mu, V = eig(J)
    unif = np.abs(V.sum(0))/(np.sqrt(V.shape[0])*np.linalg.norm(V, axis=0))
    k0 = int(np.argmax(unif))
    keep = np.ones(len(mu), bool); keep[k0] = False
    mu, V = mu[keep], V[:, keep]
    s = (1.0 - mu)/eta
    order = np.argsort(-s.real)
    return s[order], V[:, order]

def arnoldi_smax(x0, S, volfrac, filt, eta=0.5, eps=1e-5, k=6, ncv=30, tol=2e-3):
    """Top of the s-spectrum via ARPACK on the FD s-operator A=(I-J)/eta,
    with mean-centering to deflate the volume null direction."""
    F = free_set(x0)
    base = np.log(step(x0, volfrac, eta, S, filt)[F])
    nmv = [0]
    def mv(v):
        v = np.real(v)
        v = v - v.mean()
        nv = np.linalg.norm(v)
        if nv < 1e-14:
            return np.zeros_like(v)
        u = v/nv
        xp = x0.copy(); xp[F] = x0[F]*np.exp(eps*u)
        w = (np.log(step(xp, volfrac, eta, S, filt)[F]) - base)/eps
        w = w - w.mean()
        nmv[0] += 1
        return nv*(u - w)/eta
    A = LinearOperator((F.size, F.size), matvec=mv, dtype=float)
    rng = np.random.default_rng(1)
    v0 = rng.normal(size=F.size); v0 -= v0.mean()
    vals, vecs = eigs(A, k=k, which="LR", ncv=ncv, tol=tol, v0=v0)
    order = np.argsort(-vals.real)
    return vals[order], np.real(vecs[:, order]), F, nmv[0]

def cont_amp(x0, eta, S, volfrac, filt, iters=60, kick=0.02, seed=5):
    x = x0.copy()
    if kick:
        rng = np.random.default_rng(seed)
        m = (x > XMIN+2e-3) & (x < 1-2e-3)
        d = np.zeros_like(x); d[m] = rng.normal(0, kick, m.sum()); d[m] -= d[m].mean()
        x = np.clip(x*np.exp(d), XMIN, 1.0)
    amps = []
    for _ in range(iters):
        xn = step(x, volfrac, eta, S, filt)
        amps.append(np.mean(np.abs(xn-x))); x = xn
    return float(np.median(amps[-12:]))

def eta_star_complex(s_vals):
    """Overall flip/NS threshold: min over modes of 2*Re(s)/|s|^2 (Re(s)>0)."""
    s = np.asarray(s_vals)
    ok = s.real > 1e-6
    etas = 2.0*s.real[ok]/np.abs(s[ok])**2
    k = int(np.argmin(etas))
    return float(etas[k]), s[ok][k]

def growth_mu(x0, eta, S, volfrac, filt, kick=0.02, iters=14, seed=5):
    """Post-kick linear growth: dominant |multiplier| from norm ratios of
    centered free-set increments (robust to complex modes)."""
    rng = np.random.default_rng(seed)
    x = x0.copy()
    m0 = (x > XMIN+2e-3) & (x < 1-2e-3)
    d = np.zeros_like(x); d[m0] = rng.normal(0, kick, m0.sum()); d[m0] -= d[m0].mean()
    x = np.clip(x*np.exp(d), XMIN, 1.0)
    dzs, masks = [], []
    for _ in range(iters):
        xn = step(x, volfrac, eta, S, filt)
        dzs.append(np.log(xn)-np.log(x))
        masks.append((x > XMIN+1e-6) & (x < 1-1e-6) &
                     (xn > XMIN+1e-6) & (xn < 1-1e-6) &
                     (np.abs(xn-x) < MOVE-1e-6))
        x = xn
    gs, mus = [], []
    for k in range(2, iters-1):
        m = masks[k] & masks[k+1]
        if m.sum() < 20: continue
        z0 = dzs[k][m];   z0 = z0 - z0.mean()
        z1 = dzs[k+1][m]; z1 = z1 - z1.mean()
        n0 = np.linalg.norm(z0)
        if n0 < 1e-12: continue
        gs.append(np.linalg.norm(z1)/n0)
        mus.append(float(z1@z0/(n0*n0)))
    return (float(np.median(gs[-6:])) if gs else np.nan,
            float(np.median(mus[-6:])) if mus else np.nan)

def mode_probe(x0, v, F, S, volfrac, filt, etas, delta=0.005, iters=18):
    """x-space mode-aligned growth probe. Returns [(eta, |ratio|, signed)]."""
    w = x0[F]*v
    mid = (x0[F] > 0.05) & (x0[F] < 0.95)
    wm = w[mid] - w[mid].mean()
    nw = np.linalg.norm(wm)
    if nw < 1e-12:
        return [(float(e), np.nan, np.nan) for e in etas]
    wm /= nw
    out = []
    for eta in etas:
        x = x0.copy(); x[F] = x0[F]*np.exp(delta*v)
        x = np.clip(x, XMIN, 1.0)
        prev, rs = None, []
        for _ in range(iters):
            xn = step(x, volfrac, eta, S, filt)
            dx = (xn - x)[F][mid]; dx = dx - dx.mean()
            a = float(dx @ wm)
            if prev is not None and abs(prev) > 1e-14:
                rs.append(a/prev)
            prev = a; x = xn
        r = np.array(rs[3:])
        out.append((float(eta), float(np.median(np.abs(r))), float(np.median(r))))
    return out
