# -*- coding: utf-8 -*-
"""
Cycle 2 v2: instrumented top88 + perturbation probe + two-mode decomposition.

New vs v1:
  * After the main run, kick the design with a small log-perturbation and
    record the impulse response (both modes excited).
  * Fit a rank-2 linear recurrence dz_{k+2} = a dz_{k+1} + b dz_k over the
    probe window; roots r1 (creep, ~+1) and r2 (flip candidate, negative).
  * Prediction: flip root crosses -1 at onset; from STABLE runs,
    s_flip = (1 - Re r2)/eta should be ~constant and eta* = 2/s_flip.
"""
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

E0, Emin, nu, XMIN = 1.0, 1e-9, 0.3, 1e-3
rng = np.random.default_rng(7)

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

def setup(nelx, nely, rmin):
    ndof = 2*(nelx+1)*(nely+1)
    elx, ely = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
    n1 = ((nely+1)*elx + ely).ravel()
    n2 = ((nely+1)*(elx+1) + ely).ravel()
    edofMat = np.column_stack([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,
                               2*n2, 2*n2+1, 2*n1, 2*n1+1])
    iK = np.kron(edofMat, np.ones((8,1), dtype=int)).flatten()
    jK = np.kron(edofMat, np.ones((1,8), dtype=int)).flatten()
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
    f = np.zeros(ndof); f[1] = -1.0
    fixed = np.union1d(np.arange(0, 2*(nely+1), 2), np.array([ndof-1]))
    free = np.setdiff1d(np.arange(ndof), fixed)
    return dict(KE=lk(), edofMat=edofMat, iK=iK, jK=jK, H=H, Hs=Hs,
                f=f, free=free, ndof=ndof, nel=nelx*nely)

def fe(x, p, S):
    sK = ((S["KE"].flatten()[np.newaxis]).T*(Emin + x**p*(E0-Emin))).flatten(order="F")
    K = coo_matrix((sK, (S["iK"], S["jK"])), shape=(S["ndof"], S["ndof"])).tocsc()
    u = np.zeros(S["ndof"]); fr = S["free"]
    u[fr] = spsolve(K[fr][:, fr], S["f"][fr])
    ce = np.einsum("ij,jk,ik->i", u[S["edofMat"]], S["KE"], u[S["edofMat"]])
    return ce

def oc(x, dcf, volfrac, eta, move):
    l1, l2 = 0.0, 1e9
    lo = np.maximum(XMIN, x-move); hi = np.minimum(1.0, x+move)
    while (l2-l1)/(l1+l2) > 1e-10:
        lmid = 0.5*(l1+l2)
        B = np.maximum(1e-30, -dcf/lmid)
        xnew = np.clip(x*B**eta, lo, hi)
        if xnew.mean() > volfrac: l1 = lmid
        else: l2 = lmid
    return xnew

def step(x, p, volfrac, eta, move, S):
    ce = fe(x, p, S)
    c = ((Emin + x**p*(E0-Emin))*ce).sum()
    dc = -p*x**(p-1)*(E0-Emin)*ce
    dcf = np.asarray(S["H"] @ (x*dc)).ravel()/S["Hs"]/np.maximum(1e-3, x)
    xnew = oc(x, dcf, volfrac, eta, move)
    inter = ((x > XMIN+1e-6) & (x < 1-1e-6) &
             (xnew > XMIN+1e-6) & (xnew < 1-1e-6) &
             (np.abs(xnew-x) < move-1e-6))
    return xnew, np.log(xnew)-np.log(x), inter, np.log(np.maximum(1e-300, -dcf)), c

def two_mode_fit(dzs, masks, k0, k1):
    A = np.zeros((2,2)); b = np.zeros(2); n = 0
    for k in range(k0, k1-2):
        m = masks[k] & masks[k+1] & masks[k+2]
        if m.sum() < 20: continue
        z0 = dzs[k][m];   z0 = z0 - z0.mean()
        z1 = dzs[k+1][m]; z1 = z1 - z1.mean()
        z2 = dzs[k+2][m]; z2 = z2 - z2.mean()
        A += np.array([[z1@z1, z1@z0], [z1@z0, z0@z0]])
        b += np.array([z2@z1, z2@z0]); n += 1
    if n == 0 or abs(np.linalg.det(A)) < 1e-300:
        return np.nan, np.nan
    al, be = np.linalg.solve(A, b)
    disc = al*al + 4*be
    if disc >= 0:
        r1, r2 = (al+np.sqrt(disc))/2, (al-np.sqrt(disc))/2
    else:
        r1 = complex(al/2, np.sqrt(-disc)/2); r2 = r1.conjugate()
    return r1, r2   # r2 = smaller/most-negative

def run(nelx, nely, volfrac, p, rmin, eta, iters, move, S,
        probe_iters=34, probe_amp=0.05):
    x = volfrac*np.ones(S["nel"])
    dzs, masks, amps = [], [], []
    mu_t, s_t = [], []
    logD_prev = None
    for it in range(iters):
        x2, dz, inter, logD, c = step(x, p, volfrac, eta, move, S)
        dzs.append(dz); masks.append(inter); amps.append(np.mean(np.abs(x2-x)))
        if len(dzs) >= 2:
            m = masks[-2] & masks[-1]
            if m.sum() >= 20:
                z0 = dzs[-2][m]; z0 = z0 - z0.mean()
                den = z0 @ z0
                if den/m.sum() > 1e-16:
                    z1 = dzs[-1][m]; z1 = z1 - z1.mean()
                    dD = (logD - logD_prev)[m]; dD = dD - dD.mean()
                    mu_t.append((z1@z0)/den); s_t.append(-(dD@z0)/den)
        logD_prev = logD; x = x2
    c_main = c
    # --- perturbation probe ---
    inter_now = (x > XMIN+1e-6) & (x < 1-1e-6)
    dlt = np.zeros(S["nel"])
    dlt[inter_now] = rng.normal(0, probe_amp, inter_now.sum())
    dlt[inter_now] -= dlt[inter_now].mean()
    xp = np.clip(x*np.exp(dlt), XMIN, 1.0)
    xp = np.clip(xp*(volfrac/xp.mean()), XMIN, 1.0)
    k_probe = len(dzs)
    xq = xp
    for it in range(probe_iters):
        xq2, dz, inter, logD, c = step(xq, p, volfrac, eta, move, S)
        dzs.append(dz); masks.append(inter); xq = xq2
    r1, r2 = two_mode_fit(dzs, masks, k_probe+2, len(dzs))
    tail = lambda a, n: np.array(a[-n:]) if len(a) else np.array([np.nan])
    return dict(amp_tail=float(np.median(np.array(amps)[-40:])),
                mu_run=float(np.median(tail(mu_t, 60))),
                s_run=float(np.median(tail(s_t, 60))),
                mu_all=np.array(mu_t), s_all=np.array(s_t),
                r1=r1, r2=r2, x_final=x, c=c_main)


if __name__ == "__main__":
    # ================= sweep =================
    nelx, nely, volfrac, p, move, iters = 60, 20, 0.5, 3.0, 0.2, 200
    etas = np.round(np.arange(0.3, 1.61, 0.1), 2)
    rmins = [1.1, 2.4]
    setups = {rm: setup(nelx, nely, rm) for rm in rmins}
    res = {}
    hdr = f"{'rmin':>5} {'eta':>5} {'amp':>9} {'mu_tail':>8} {'r_creep':>8} {'r_flip':>8} {'s_flip':>7} {'c':>8}"
    print(hdr)
    for rm in rmins:
        for eta in etas:
            r = run(nelx, nely, volfrac, p, rm, eta, iters, move, setups[rm])
            res[(rm, eta)] = r
            rf = r["r2"]; rc = r["r1"]
            rf_re = rf.real if isinstance(rf, complex) else rf
            s_flip = (1-rf_re)/eta
            r["s_flip"] = s_flip; r["rf_re"] = rf_re
            tag = "C" if isinstance(rf, complex) else " "
            print(f"{rm:5.1f} {eta:5.2f} {r['amp_tail']:9.2e} {r['mu_run']:8.3f} "
                  f"{(rc.real if isinstance(rc,complex) else rc):8.3f} {rf_re:8.3f}{tag} "
                  f"{s_flip:7.3f} {r['c']:8.2f}")

    # ================= classification & prediction =================
    print("\n=== onset vs extrapolated prediction ===")
    summ = {}
    for rm in rmins:
        amp = np.array([res[(rm, e)]["amp_tail"] for e in etas])
        mu = np.array([res[(rm, e)]["mu_run"] for e in etas])
        osc = (amp > 3e-3) & (mu < -0.5)
        if osc.any() and not osc[0]:
            ih = int(np.argmax(osc)); il = ih-1
            eta_lo, eta_hi = etas[il], etas[ih]
            stable_mask = ~osc & (np.arange(len(etas)) < ih)
            s_stab = np.array([res[(rm, e)]["s_flip"] for e in etas[stable_mask]])
            s_med = float(np.median(s_stab)); s_spread = float(np.ptp(s_stab))
            pred = 2.0/s_med
            summ[rm] = dict(eta_lo=eta_lo, eta_hi=eta_hi, s_med=s_med,
                            s_spread=s_spread, pred=pred)
            inb = "INSIDE bracket" if eta_lo < pred <= eta_hi else "outside bracket"
            print(f"rmin={rm}: onset in ({eta_lo}, {eta_hi}] | stable-side s_flip "
                  f"median={s_med:.3f} (spread {s_spread:.3f}) -> eta*_pred={pred:.3f} [{inb}]")
        else:
            summ[rm] = None
            print(f"rmin={rm}: no clean onset")

    # ================= figure =================
    fig = plt.figure(figsize=(13.5, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0])
    axA = fig.add_subplot(gs[0,0]); axB = fig.add_subplot(gs[0,1]); axC = fig.add_subplot(gs[0,2])
    cols = {1.1: "tab:blue", 2.4: "tab:red"}

    for rm in rmins:
        amp = [res[(rm, e)]["amp_tail"] for e in etas]
        axA.semilogy(etas, amp, "o-", color=cols[rm], label=f"rmin={rm}")
        if summ[rm]:
            axA.axvline(summ[rm]["pred"], color=cols[rm], ls="--", lw=1.3,
                        label=f"pred $2/\\hat s_{{flip}}$={summ[rm]['pred']:.2f}")
    axA.axvline(0.5, color="gray", ls=":", lw=1, label=r"determinate bound $2/(p{+}1)$")
    axA.set_xlabel(r"$\eta$"); axA.set_ylabel(r"tail mean $|\Delta x|$")
    axA.set_title("(A) oscillation amplitude & predicted onset")
    axA.legend(fontsize=7)

    for rm in rmins:
        rf = [res[(rm, e)]["rf_re"] for e in etas]
        rc = [(res[(rm, e)]["r1"].real if isinstance(res[(rm, e)]["r1"], complex)
               else res[(rm, e)]["r1"]) for e in etas]
        axB.plot(etas, rf, "o-", color=cols[rm], label=f"flip root, rmin={rm}")
        axB.plot(etas, rc, "s--", color=cols[rm], alpha=0.4, ms=4,
                 label=f"creep root, rmin={rm}")
        if summ[rm]:
            axB.axvspan(summ[rm]["eta_lo"], summ[rm]["eta_hi"], color=cols[rm], alpha=0.08)
    axB.axhline(-1, color="k", lw=0.8, ls=":")
    axB.set_xlabel(r"$\eta$"); axB.set_ylabel("recurrence roots (Re)")
    axB.set_title("(B) probe modes: flip root crosses $-1$ at onset")
    axB.legend(fontsize=6.5)

    for rm in rmins:
        for e in etas:
            r = res[(rm, e)]
            n = min(len(r["s_all"]), 60)
            if n > 0:
                axC.plot(1-e*r["s_all"][-n:], r["mu_all"][-n:], ".", ms=2.5,
                         alpha=0.35, color=cols[rm])
    axC.plot([-2.2, 1.1], [-2.2, 1.1], "k--", lw=1)
    axC.set_xlabel(r"$1-\eta\,\hat s$"); axC.set_ylabel(r"$\hat\mu$")
    axC.set_title(r"(C) identity $\hat\mu = 1-\eta\hat s$ on free set")

    def show(ax, arr, title):
        ax.imshow(-arr.reshape(nelx, nely).T, cmap="gray", origin="upper",
                  interpolation="nearest", vmin=-1, vmax=0)
        ax.set_title(title, fontsize=9); ax.axis("off")

    axD = fig.add_subplot(gs[1,0]); axE = fig.add_subplot(gs[1,1]); axF = fig.add_subplot(gs[1,2])
    show(axD, res[(2.4, 0.5)]["x_final"], f"(D) rmin=2.4, eta=0.5, c={res[(2.4,0.5)]['c']:.0f}")
    rm_o = 2.4; eta_o = summ[rm_o]["eta_hi"] if summ[rm_o] else etas[-1]
    show(axE, res[(rm_o, eta_o)]["x_final"],
         f"(E) rmin={rm_o}, eta={eta_o}, c={res[(rm_o,eta_o)]['c']:.0f} (oscillating)")
    S = setups[rm_o]; x = res[(rm_o, eta_o)]["x_final"].copy()
    acc = np.zeros_like(x)
    for _ in range(10):
        xn, dz, inter, logD, c = step(x, p, volfrac, eta_o, move, S)
        acc += np.abs(xn-x); x = xn
    im = axF.imshow((acc/10).reshape(nelx, nely).T, cmap="inferno", origin="upper",
                    interpolation="nearest")
    axF.set_title(f"(F) mean |dx|/iter at rmin={rm_o}, eta={eta_o}", fontsize=9)
    axF.axis("off"); fig.colorbar(im, ax=axF, shrink=0.8)

    fig.tight_layout()
    fig.savefig(Path(__file__).resolve().parents[1] / "figs" / "fig_cycle2.png", dpi=150)
    print("figure saved.")
