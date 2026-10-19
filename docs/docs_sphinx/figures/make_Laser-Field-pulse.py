"""Field E(t) and vector potential A(t) of the flat-top pulse of
examples/hhg/hhg_1d.inp, written as E = -dA/dt with
A(t) = (E0/w) f(t) sin(w t) and a sin^2-ramped flat-top envelope f(t).

Output: Laser-Field-pulse.svg next to this script.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Parameters of examples/hhg/hhg_1d.inp
E0, w = 0.0534, 0.057
Tr, Tf, Tp = 220.4625, 1543.2375, 1763.7

t = np.linspace(0.0, Tp, 20001)

# envelope f(t) and its derivative, exactly as in the LaserX expression
f = np.where(t < Tr, np.sin(np.pi * t / (2 * Tr)) ** 2,
    np.where(t < Tf, 1.0,
    np.where(t < Tp, np.cos(np.pi * (t - Tf) / (2 * Tr)) ** 2, 0.0)))
df = np.where(t < Tr, (np.pi / (2 * Tr)) * np.sin(np.pi * t / Tr),
     np.where(t < Tf, 0.0,
     np.where(t < Tp, -(np.pi / (2 * Tr)) * np.sin(np.pi * (t - Tf) / Tr), 0.0)))

A = (E0 / w) * f * np.sin(w * t)
E = -(E0 / w) * (df * np.sin(w * t) + w * f * np.cos(w * t))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 4.2), sharex=True)
ax1.plot(t, E, color="#3b6ea5", lw=0.8)
ax1.plot(t, E0 * f, color="#555555", lw=0.8, ls="--")
ax1.plot(t, -E0 * f, color="#555555", lw=0.8, ls="--")
ax1.set_ylabel(r"$E(t)$ (a.u.)")
ax2.plot(t, A, color="#a5583b", lw=0.8)
ax2.set_ylabel(r"$A(t)$ (a.u.)")
ax2.set_xlabel(r"$t$ (a.u.)")
for ax, x in ((ax1, Tr), (ax1, Tf), (ax2, Tr), (ax2, Tf)):
    ax.axvline(x, color="#999999", lw=0.6, ls=":")
ax1.set_xlim(0.0, Tp)
fig.tight_layout()
fig.savefig(Path(__file__).with_name("Laser-Field-pulse.svg"))
