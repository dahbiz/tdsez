"""Exact DQW and mass-step eigenvalues by transfer matrices in 40-digit arithmetic (mpmath).

    python exact_dqw.py     -> exact.json (DQW E0..E3 for w_b = 2..10 nm; mass-step ground state)

DQW: GaAs wells (8 nm, m_w = 0.067, V = 0) between Al0.3Ga0.7As (m_b = 0.0879, V0 = 0.00893 Ha):
barrier w_b, wells, 5 nm buffers, Neumann walls at |x| = w_b/2 + 13 nm. In each layer
(psi, psi'/m) is propagated exactly; psi and psi'/m are continuous at the interfaces (BenDaniel-Duke).
Symmetric potential: even states start at x = 0 with (1, 0), odd ones with (0, 1); the eigenvalue
condition is psi'(L) = 0. Mass step: m = 1 on (-1, 0), m = 4 on (0, 1), V = 0, Dirichlet walls.
"""
import json

import mpmath as mp

mp.mp.dps = 40
NM = mp.mpf("18.8973")
HA_MEV = mp.mpf("27211.386")
MW, MB, V0 = mp.mpf("0.067"), mp.mpf("0.0879"), mp.mpf("0.00893")


def step(u, w, m, V, E, d):
    """Propagate (psi, psi'/m) over a layer of width d."""
    if E < V:
        q = mp.sqrt(2 * m * (V - E))
        c, s = mp.cosh(q * d), mp.sinh(q * d)
        return u * c + w * m / q * s, u * q / m * s + w * c
    k = mp.sqrt(2 * m * (E - V))
    c, s = mp.cos(k * d), mp.sin(k * d)
    return u * c + w * m / k * s, -u * k / m * s + w * c


def layers(wb):
    a = wb / 2 * NM
    return [(MB, V0, a), (MW, 0, 8 * NM), (MB, V0, 5 * NM)]


def f(E, wb, parity):
    u, w = (mp.mpf(1), mp.mpf(0)) if parity == 0 else (mp.mpf(0), mp.mpf(1))
    for m, V, d in layers(wb):
        u, w = step(u, w, m, V, E, d)
    return w


def roots(fun, lo, hi, n=4000):
    """All sign changes of fun on (lo, hi), refined by bisection to 40 digits."""
    xs = [lo + (hi - lo) * i / n for i in range(1, n)]
    out, prev = [], fun(xs[0])
    for a, b in zip(xs[:-1], xs[1:]):
        fb = fun(b)
        if prev * fb < 0:
            out.append(mp.findroot(fun, (a, b), solver="bisect", tol=mp.mpf(10) ** -38))
        prev = fb
    return out


res = {"dqw": {}}
for wb in (2, 4, 6, 8, 10):
    E = sorted(roots(lambda e: f(e, wb, 0), mp.mpf("1e-6"), V0) + roots(lambda e: f(e, wb, 1), mp.mpf("1e-6"), V0))
    res["dqw"][str(wb)] = [mp.nstr(e * HA_MEV, 30) for e in E[:4]]
    print(f"w_b = {wb:2d} nm: E0..E3 = " + ", ".join(mp.nstr(e * HA_MEV, 14) for e in E[:4]) + " meV")

# mass step: psi_L = sin(k1 (x+1)), psi_R = C sin(k2 (1-x)); continuity of psi and psi'/m at 0
g = lambda E: mp.sqrt(2 * E) * mp.cos(mp.sqrt(2 * E)) * mp.sin(mp.sqrt(8 * E)) \
    + mp.sqrt(8 * E) / 4 * mp.cos(mp.sqrt(8 * E)) * mp.sin(mp.sqrt(2 * E))
Es = roots(g, mp.mpf("0.01"), mp.mpf(5))
res["massstep"] = [mp.nstr(e, 30) for e in Es[:3]]
print("mass step E0..E2 =", [mp.nstr(e, 16) for e in Es[:3]])
json.dump(res, open("exact.json", "w"), indent=1)
