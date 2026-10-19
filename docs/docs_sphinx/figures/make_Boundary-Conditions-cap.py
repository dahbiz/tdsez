"""Manolopoulos CAP W(x) as assembled by TDSEZformCap / Manolopoulos_CAP_profile
(src/assembly/assembly.cpp): layer width w = c/(2 delta kmin) at both walls,
r = c u/w with u the depth into the layer, and
W = Emin (a r - b r^3 + 4/(c-r)^2 - 4/(c+r)^2), Emin = kmin^2/2,
a = 1 - 16/c^3, b = (1 - 17/c^3)/c^2, c = 2.62206, delta = 0.2.

Example: axis [-65, 65], CAPKmin = 0.2 (w = 32.8 au).
Output: Boundary-Conditions-cap.svg next to this script.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

c, delta = 2.62206, 0.2
a = 1.0 - 16.0 / c**3
b = (1.0 - 17.0 / c**3) / c**2
lo, hi, kmin = -65.0, 65.0, 0.2
w = c / (2 * delta * kmin)
Emin = 0.5 * kmin**2

x = np.linspace(lo, hi, 40001)
s = np.minimum(x - lo, hi - x)
u = np.minimum(w - s, w * (1 - 1e-10))
r = c * u / w
W = np.where(s < w, Emin * (a * r - b * r**3 + 4 / (c - r) ** 2 - 4 / (c + r) ** 2), 0.0)

fig, ax = plt.subplots(figsize=(7.0, 3.4))
ax.plot(x, W / Emin, color="#3b6ea5", lw=1.2)
for xe in (lo + w, hi - w):
    ax.axvline(xe, color="#999999", lw=0.8, ls=":")
ax.axvspan(lo, lo + w, color="#eef3fb")
ax.axvspan(hi - w, hi, color="#eef3fb")
ax.text(0.0, 6.0, "free core", ha="center", fontsize=9, color="#555555")
ax.text(lo + 0.62 * w, 4.6, f"layer\nw = {w:.1f} a.u.", ha="center", fontsize=9, color="#555555")
ax.text(hi - 0.62 * w, 4.6, f"layer\nw = {w:.1f} a.u.", ha="center", fontsize=9, color="#555555")
ax.set_ylim(0, 8)
ax.set_xlim(lo, hi)
ax.set_xlabel(r"$x$ (a.u.)")
ax.set_ylabel(r"$W(x)/E_{\min}$")
fig.tight_layout()
fig.savefig(Path(__file__).with_name("Boundary-Conditions-cap.svg"))
