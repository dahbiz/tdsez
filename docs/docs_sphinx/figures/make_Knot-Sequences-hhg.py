"""Harmonic-yield error against HydrogenicHMax for the 2D HHG study of
Knot-Sequences.rst (R1 = 0.05, NLin 5, NExp 10). The values are exactly the
tabulated ones; the row "HMax = 0.7 / 0.5 / 0.35" of the table gives the same
errors for the three widths, and they are plotted at each of them.

Output: Knot-Sequences-hhg.svg next to this script.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

hmax = [2.0, 1.0, 0.7, 0.5, 0.35]
median = [0.088, 0.007, 0.007, 0.007, 0.007]
pmax = [0.48, 0.031, 0.031, 0.031, 0.031]
cutoff = [4.4, 0.018, 0.010, 0.010, 0.010]
nocap = (1.18, 2.35, 2.98)          # elements up to 28 au

fig, ax = plt.subplots(figsize=(7.0, 3.8))
ax.plot(hmax, median, "o-", color="#3b6ea5", label="plateau, median")
ax.plot(hmax, pmax, "s-", color="#a5583b", label="plateau, max")
ax.plot(hmax, cutoff, "^-", color="#555555", label="H27-H35, max")
for v, c, m in zip(nocap, ("#3b6ea5", "#a5583b", "#555555"), ("o", "s", "^")):
    ax.plot([3.0], [v], m, color=c, mfc="none")
ax.axvline(1.0, color="#999999", lw=0.8, ls=":")
ax.text(1.03, 2.0e-3, r"$\pi/(2k_{\max})\approx 1.0$", fontsize=8, color="#555555")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([0.35, 0.5, 0.7, 1.0, 2.0, 3.0])
ax.set_xticklabels(["0.35", "0.5", "0.7", "1.0", "2.0", "no cap"])
ax.minorticks_off()
ax.set_ylim(1e-3, 10)
ax.invert_xaxis()
ax.set_xlabel("HydrogenicHMax (a.u.)")
ax.set_ylabel(r"error $|\log_{10} I/I_{\mathrm{ref}}|$ (decades)")
ax.legend(fontsize=8, frameon=False, loc="upper right")
ax.grid(True, color="#dddddd", lw=0.5)
fig.tight_layout()
fig.savefig(Path(__file__).with_name("Knot-Sequences-hhg.svg"))
