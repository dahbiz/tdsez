"""Exact reference for the multipole validation: the TDSE of the driven oscillator in a truncated Hermite basis
(no grid), integrated by SciPy DOP853 (rtol 1e-11); the pulse is E0 sin^2(pi t/tp) sin(W t)."""
import numpy as np
from scipy.integrate import solve_ivp

E0, W, TP = 0.3, 1.0, 20.0


def field(t, tp=TP):
    """E, E', E'' of the sin^2 pulse (analytic)."""
    a = np.pi / tp
    s, c = np.sin(a * t) ** 2, np.sin(2 * a * t) * a          # env, env'
    c2 = 2 * a * a * np.cos(2 * a * t)                          # env''
    on = t < tp
    e = E0 * s * np.sin(W * t)
    e1 = E0 * (c * np.sin(W * t) + s * W * np.cos(W * t))
    e2 = E0 * (c2 * np.sin(W * t) + 2 * c * W * np.cos(W * t) - s * W * W * np.sin(W * t))
    return np.where(on, e, 0), np.where(on, e1, 0), np.where(on, e2, 0)


import scipy.sparse as sp


def hermite_reference(ws, m, q, c, mp, n, pol, ts, nb, tp):
    """q<r_a>(t) for the oscillator frequencies ws, field along axis pol, direction n."""
    dim = len(ws)
    lad = sp.diags(np.sqrt(np.arange(1, nb)), 1, format="csr")
    eye = sp.identity(nb, format="csr")

    def embed(op, a):
        mats = [eye] * dim
        mats[a] = op
        out = mats[0]
        for k in mats[1:]:
            out = sp.kron(out, k, format="csr")
        return out

    R = [embed((lad + lad.T) / np.sqrt(2 * m * w), a) for a, w in enumerate(ws)]
    P = [embed(1j * np.sqrt(m * w / 2) * (lad.T - lad), a) for a, w in enumerate(ws)]
    zero = sp.csr_matrix((nb**dim, nb**dim))
    R3, P3 = R + [zero] * (3 - dim), P + [zero] * (3 - dim)
    L = [R3[1] @ P3[2] - R3[2] @ P3[1], R3[2] @ P3[0] - R3[0] @ P3[2], R3[0] @ P3[1] - R3[1] @ P3[0]]
    levels = sum(np.meshgrid(*[w * (np.arange(nb) + 0.5) for w in ws], indexing="ij"))
    H0 = sp.diags(levels.ravel())
    nr = sum(n[a] * R[a] for a in range(dim))
    ex = np.zeros(3); ex[pol] = 1.0
    cvec = np.cross(n, ex)                                   # C = n x E = cvec E(t)
    r2 = sum(Ra @ Ra for Ra in R)
    Ep = R[pol]
    quad = nr @ Ep                                           # (n.r)(r.e)
    octu = nr @ nr @ Ep                                      # (n.r)^2 (r.e)
    mag = sum(cvec[b] * L[b] for b in range(3))              # L.C / E
    magq = 0.5 * (nr @ mag + mag @ nr)                       # {n.r, L}/2 . C / E
    cr = sum(cvec[b] * R3[b] for b in range(3))
    dia = r2 - cr @ cr                                       # |c x r|^2 = |c|^2 r^2 - (c.r)^2, |c| = 1
    s = -q

    def H(t):
        e, e1, e2 = (float(v) for v in field(np.array(t), tp))
        h = H0 + s * e * Ep
        if mp >= 2:
            h = h + q / (2 * c) * e1 * quad - q / (2 * m * c) * e * mag
        if mp >= 3:
            h = h - q / (6 * c * c) * e2 * octu + q / (3 * m * c * c) * e1 * magq + q * q / (8 * m * c * c) * e * e * dia
        return h

    psi0 = np.zeros(nb**dim, complex)
    psi0[0] = 1.0
    sol = solve_ivp(lambda t, y: -1j * (H(t) @ y), (0, ts[-1]), psi0, t_eval=ts, method="DOP853", rtol=1e-11, atol=1e-13)
    psi = sol.y
    occ = (np.abs(psi[:, -1]) ** 2).reshape((nb,) * dim)
    edge = sum(np.take(occ, range(nb - 3, nb), axis=a).sum() for a in range(dim))
    return np.stack([q * np.real(np.einsum("ik,ik->k", psi.conj(), Ra @ psi)) for Ra in R], 1), edge


