"""Owned-row imbalance of PetIGA's partition (MPI Notes, Load Balance).

Per axis, n elements are split over P ranks (leftover elements to the first
ranks); a rank owns one basis function per element and the last rank also owns
the p trailing ones. The 3D ratio is (max owned / min owned)^3 on P^3 ranks.
"""
import os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def owned(n, P, p):
    base, extra = divmod(n, P)
    rows = [base + (1 if r < extra else 0) for r in range(P)]
    rows[-1] += p
    return rows


def ratio3d(n, P, p):
    rows = owned(n, P, p)
    return (max(rows) / min(rows)) ** 3


P = 2
ns = list(range(16, 401))
mpl.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.linewidth": 0.8})
fig, ax = plt.subplots(figsize=(6.4, 3.6))
colors = ["#9fbbe0", "#3b6ea5", "#1f3f66", "#c0504d"]
for p, c in zip([1, 3, 5, 7], colors):
    ax.plot([n / P for n in ns], [ratio3d(n, P, p) for n in ns], color=c, lw=1.4, label=f"p = {p}")
for n in (32, 33, 200):   # the rows of the Load Balance table (p = 3)
    ax.plot(n / P, ratio3d(n, P, 3), "o", color="#3b6ea5", ms=4, zorder=3)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([8, 16, 32, 64, 128, 200])
ax.set_xticklabels(["8", "16", "32", "64", "128", "200"])
ax.set_yticks([1, 1.25, 1.5, 2, 3, 5, 8])
ax.set_yticklabels(["1", "1.25", "1.5", "2", "3", "5", "8"])
ax.minorticks_off()
ax.set_xlabel("elements per rank and axis")
ax.set_ylabel("max / min owned rows (3D, 2×2×2 ranks)")
ax.set_ylim(1.0, 9.0)
ax.grid(alpha=0.3, lw=0.5)
ax.legend(frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "mpi_load_balance.svg"))
