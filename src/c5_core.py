# -*- coding: utf-8 -*-
"""Cycle 5 core: exact volume-tangent projection Q^T J Q, gray-layer stats,
compliance evaluation, and a mode probe with an adjustable mid-density mask.

Extends c4_core by import; cycle-4 behavior is untouched.
"""
import numpy as np
from scipy.linalg import eig
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

import c4_core as core
from c4_core import XMIN, E0, Emin


def vol_weights(x0, F, S, filt):
    """Log-coordinate volume weights w on the free set F, so that the
    volume constraint reads sum_F w_e dz_e = 0 (dz = d log x).
    sens: vol = mean(x)        -> w_e = x_e
    dens: vol = mean(H x / Hs) -> w_e = x_e * colsum_e(H_ie / Hs_i)."""
    if filt == "dens":
        c = np.asarray(S["H"].T @ (1.0 / S["Hs"])).ravel()
        return x0[F] * c[F]
    return x0[F].copy()


def perp_basis(w):
    """Orthonormal basis Q (n x n-1) of the hyperplane {v : w^T v = 0},
    built from a single Householder reflector mapping w to +-e1."""
    u = np.asarray(w, float) / np.linalg.norm(w)
    v = u.copy()
    v[0] += np.copysign(1.0, u[0] if u[0] != 0.0 else 1.0)
    v /= np.linalg.norm(v)
    H = np.eye(u.size) - 2.0 * np.outer(v, v)
    return H[:, 1:]


def tangent_spectrum(J, x0, F, S, filt, eta=0.5):
    """Exact projected spectrum eig(Q^T J Q) on the volume tangent space.
    Returns (s sorted by descending Re, modes lifted to free-set coords,
    row-null residual ||w^T J|| / (||w|| ||J||_F))."""
    w = vol_weights(x0, F, S, filt)
    Q = perp_basis(w)
    mu, V = eig(Q.T @ J @ Q)
    s = (1.0 - mu) / eta
    order = np.argsort(-s.real)
    s, V = s[order], V[:, order]
    rnull = float(np.linalg.norm(w @ J) /
                  (np.linalg.norm(w) * np.linalg.norm(J)))
    return s, Q @ V, rnull


def gray_stats(x0, lo=0.05, hi=0.95):
    g = (x0 > lo) & (x0 < hi)
    return dict(n_gray=int(g.sum()), mass_gray=float(x0[g].sum()),
                n_free=int(core.free_set(x0).size))


def compliance(x, S, filt):
    xph = np.asarray(S["H"] @ x).ravel() / S["Hs"] if filt == "dens" else x
    sK = ((S["KE"].flatten()[np.newaxis]).T *
          (Emin + xph**core.P * (E0 - Emin))).flatten(order="F")
    Kff = coo_matrix((sK[S["mK"]], (S["iKr"], S["jKr"])),
                     shape=(S["nfree"], S["nfree"])).tocsc()
    u = np.zeros(S["ndof"])
    u[S["free"]] = spsolve(Kff, S["f"][S["free"]], permc_spec="MMD_AT_PLUS_A")
    return float(S["f"] @ u)


def mode_probe2(x0, v, F, S, volfrac, filt, etas, delta=0.005, iters=18,
                mid_lo=0.05, mid_hi=0.95, min_mid=20):
    """c4_core.mode_probe with an adjustable mid-density mask.  Falls back to
    the full interior free set when too few mid-density elements exist
    (large p, thin gray layer).  Returns (probe rows, mask size)."""
    mid = (x0[F] > mid_lo) & (x0[F] < mid_hi)
    if mid.sum() < min_mid:
        mid = (x0[F] > XMIN + 2e-3) & (x0[F] < 1 - 2e-3)
    w = x0[F] * v
    wm = w[mid] - w[mid].mean()
    nw = np.linalg.norm(wm)
    if nw < 1e-12:
        return [(float(e), np.nan, np.nan) for e in etas], int(mid.sum())
    wm /= nw
    out = []
    for eta in etas:
        x = x0.copy()
        x[F] = x0[F] * np.exp(delta * v)
        x = np.clip(x, XMIN, 1.0)
        prev, rs = None, []
        for _ in range(iters):
            xn = core.step(x, volfrac, eta, S, filt)
            dx = (xn - x)[F][mid]
            dx = dx - dx.mean()
            a = float(dx @ wm)
            if prev is not None and abs(prev) > 1e-16:
                rs.append(a / prev)
            prev = a
            x = xn
        r = np.array(rs[3:])
        out.append((float(eta), float(np.median(np.abs(r))),
                    float(np.median(r))))
    return out, int(mid.sum())


def onset_bracket(probe):
    """(last stable eta, first flip eta) from probe rows sorted by eta.
    flip := signed ratio < 0 and |ratio| >= 1."""
    lo, hi = None, None
    for e, g, rs in sorted(probe):
        if rs < 0 and g >= 1.0:
            hi = e
            break
        lo = e
    return lo, hi
