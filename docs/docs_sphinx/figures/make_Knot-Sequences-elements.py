"""Element widths on the half axis [0, Lmax] for the knot sequences of
src/core_knots.cpp (uniform, symexp, hydrogenic with and without
HydrogenicHMax). The generator formulas are reproduced exactly.

Parameters: Lmax = 65 au (the x axis of the HHG study in Knot-Sequences.rst),
HydrogenicR1 = 0.05, HydrogenicNLin = 5, HydrogenicNExp = 10, and the same
total element count (30) for uniform and symexp (KnotAlpha default 0.9).

Running the script also prints the DOF counts of the grids quoted in
Knot-Sequences.rst as a consistency check (degree 3: n = elements + 3).

Output: Knot-Sequences-elements.svg next to this script.
"""
from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def clean(x):
    return round(x * 1e12) / 1e12


def hydrogenic_half(Lmax, n_lin, r1, n_exp, hmax):
    """Half-axis knots [0, ..] of TDSEZHydrogenicKnots (without Lmax)."""
    r_cross = n_lin * r1
    log_ratio = math.log(Lmax / r_cross)
    half = [clean(i * r1) for i in range(n_lin + 1)]
    half += [clean(r_cross * math.exp(k * log_ratio / n_exp)) for k in range(1, n_exp)]
    if hmax > 0:
        half.append(Lmax)
        for j in range(len(half) - 1):
            if half[j + 1] - half[j] <= hmax:
                continue
            r = half[j]
            nu = math.ceil((Lmax - r) / hmax - 1e-9)
            half = half[: j + 1] + [clean(r + k * (Lmax - r) / nu) for k in range(1, nu + 1)]
            break
        half.pop()
    return half + [Lmax]


def symexp_full(Lmin, Lmax, nint, alpha):
    t = 2.0 * (np.arange(nint) + 1.0) / (nint + 1) - 1.0
    m = (np.exp(alpha * np.abs(t)) - 1.0) / (np.exp(alpha) - 1.0)
    inner = np.where(t >= 0, m * Lmax, m * Lmin)
    return np.concatenate(([Lmin], inner, [Lmax]))


def nfun(Lmax, n_lin, r1, n_exp, hmax, p=3):
    return 2 * (len(hydrogenic_half(Lmax, n_lin, r1, n_exp, hmax)) - 1) + p


# --- consistency check of the tabulated DOF counts -------------------------
if __name__ == "__main__":
    print("3D H, +-10:", nfun(10, 5, 0.05, 10, 0) ** 3, "(35 937)")
    print("3D H, +-20, NExp 15:", nfun(20, 5, 0.05, 15, 0) ** 3, "(79 507)")
    print("3D H, +-10, HMax 1:", nfun(10, 5, 0.05, 10, 1.0) ** 3, "(68 921)")
    for hm, r1 in ((0, 0.05), (2.0, 0.05), (1.0, 0.05), (0.7, 0.05), (0.5, 0.05),
                   (0.35, 0.05), (0.5, 0.1), (0.5, 0.2), (0.25, 0.02)):
        print(f"2D HHG HMax={hm} R1={r1}:",
              nfun(65, 5, r1, 10, hm) * nfun(40, 5, r1, 10, hm))
    print("largest uncapped element on 65 au:",
          65 - hydrogenic_half(65, 5, 0.05, 10, 0)[-2])

Lmax = 65.0
grids = {
    "uniform (30 elements)": np.linspace(-Lmax, Lmax, 31),
    "symexp, KnotAlpha 0.9 (30 elements)": symexp_full(-Lmax, Lmax, 29, 0.9),
}
fig, ax = plt.subplots(figsize=(7.0, 3.8))
styles = [("#888888", "s", "-"), ("#a5583b", "^", "-"), ("#3b6ea5", "o", "-"), ("#3b6ea5", "o", "--")]
for (label, k), (c, mk, ls) in zip(grids.items(), styles):
    k = k[k >= -1e-12]
    if k[0] > 1e-12:
        k = np.concatenate(([0.0], k))
    h = np.diff(k)
    ax.step(k, np.append(h, h[-1]), where="post", color=c, ls=ls, lw=1.2, label=label)
for (label, hm), (c, mk, ls) in zip((("hydrogenic, no width cap", -1), ("hydrogenic, HydrogenicHMax = 1", 1.0)),
                                    styles[2:]):
    k = np.array(hydrogenic_half(Lmax, 5, 0.05, 10, hm if hm > 0 else 0))
    h = np.diff(k)
    ax.step(k, np.append(h, h[-1]), where="post", color=c, ls=ls, lw=1.2, label=label)
ax.set_yscale("log")
ax.set_xlim(0, Lmax)
ax.set_xlabel(r"$x$ (a.u.)")
ax.set_ylabel(r"element width $h$ (a.u.)")
ax.legend(fontsize=8, frameon=False, loc="lower right")
ax.grid(True, which="major", color="#dddddd", lw=0.5)
fig.tight_layout()
fig.savefig(Path(__file__).with_name("Knot-Sequences-elements.svg"))
