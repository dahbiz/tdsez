"""Measured speed-ups of the optimisations (numbers of PERFORMANCE.md, Summary)."""
import os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# (label, before [s], after [s]) exactly as tabulated in PERFORMANCE.md
rows = [
    ("Banded insertion, 3D p=7, 20$^3$ el.: H, M", 258, 11.5),
    ("Magnus-4 vs Crank–Nicolson, 1D driven HO", 11.4, 0.77),
    ("Sum-factorised assembly, 3D p=7: TD operators", 464, 32),
    ("Sum-factorised assembly, 3D p=7: H, M", 114, 8.0),
    ("Sum-factorised assembly, 3D z-pol. p=5: run", 62.6, 7.1),
    ("Banded insertion, 3D p=7, 10$^3$ el.: TD operators", 54.2, 6.5),
    ("Eigensolver inner solves, 3D p=7", 49.0, 19.8),
    ("Quadrature monitors, 3D var. mass: run", 142, 90),
    ("Allocation-free 1/m derivatives: force assembly", 28.1, 18.5),
    ("Monitor products + batched HDF5, 1D HO", 15.8, 12.3),
    ("Extrapolated guess + fused update: stepping", 3.40, 2.90),
    ("Padé/Magnus stage solver, 3D HO", 249, 219),
]
rows.sort(key=lambda r: r[1] / r[2])
mpl.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.linewidth": 0.8})
fig, ax = plt.subplots(figsize=(7.2, 4.6))
y = range(len(rows))
gain = [b / a for _, b, a in rows]
ax.barh(list(y), gain, color="#3b6ea5", height=0.62)
for i, g in enumerate(gain):
    ax.text(g * 1.06, i, f"{g:.1f}×", va="center", fontsize=8)
ax.set_yticks(list(y)); ax.set_yticklabels([r[0] for r in rows])
ax.set_xscale("log"); ax.set_xlim(1, 60)
ax.set_xlabel("speed-up (wall time before / after)")
ax.grid(axis="x", which="both", color="#dddddd", lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.savefig(os.path.join(HERE, "performance_speedups.svg"), bbox_inches="tight")
