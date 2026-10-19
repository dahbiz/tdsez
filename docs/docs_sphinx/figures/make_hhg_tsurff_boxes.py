"""Simulation-box layouts of the two strong-field example decks.

hhg_box.svg    examples/hhg/hhg_1d.inp     Domain = -70, 70, CAPKmin = 0.3
tsurff_box.svg examples/tsurff/ati_1d.inp  Domain = -80, 80, CAPKmin = 0.2,
                                           SurffRadius = 35, Rc = 20, d = 2

Quantities drawn (no simulation data):
  CAP layer width per wall  w = 2.62206 / (2 * 0.2 * CAPKmin) = 6.555 / CAPKmin
  quiver amplitude          alpha = E0 / omega^2 (E0 = 0.0534, omega = 0.057)
  free-core requirement     half width >= 2 alpha + 10 (start-up CAP report)
  potentials                exactly the Potential expressions of the decks

Output: hhg_box.svg and tsurff_box.svg next to this script.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

E0, OMEGA = 0.0534, 0.057
ALPHA = E0 / OMEGA ** 2
BLUE, GREY, RED, GREEN = "#3b6ea5", "#555555", "#a53b3b", "#3b8a4a"


def cap_width(kmin):
    return 2.62206 / (2.0 * 0.2 * kmin)


def draw(ax, L, kmin, V, surff=None, cutoff=None):
    w = cap_width(kmin)
    x = np.linspace(-L, L, 4001)
    ymin, ymax = -1.15, 0.55
    for lo, hi in ((-L, -L + w), (L - w, L)):
        ax.axvspan(lo, hi, color=RED, alpha=0.13, lw=0)
    ax.text(-L + w / 2, 0.38, f"CAP\nw = {w:.1f}", ha="center", va="center", fontsize=8, color=RED)
    ax.text(L - w / 2, 0.38, f"CAP\nw = {w:.1f}", ha="center", va="center", fontsize=8, color=RED)
    ax.plot(x, V(x), color=BLUE, lw=1.2, label="potential $V(x)$")
    # free core and the 2 alpha + 10 requirement of the start-up CAP report
    core = L - w
    need = 2 * ALPHA + 10
    ax.annotate("", xy=(core, -0.78), xytext=(-core, -0.78),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.9))
    ax.text(0, -0.74, f"free core, half width {core:.1f}", ha="center", va="bottom", fontsize=8, color=GREY)
    ax.annotate("", xy=(need, -0.98), xytext=(-need, -0.98),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.9, ls="--"))
    ax.text(0, -0.94, rf"required: $2\alpha + 10 = {need:.1f}$", ha="center", va="bottom", fontsize=8, color=GREY)
    ax.annotate("", xy=(ALPHA, 0.12), xytext=(-ALPHA, 0.12),
                arrowprops=dict(arrowstyle="<->", color=GREEN, lw=0.9))
    ax.text(0, 0.16, rf"quiver $\pm\alpha = \pm{ALPHA:.1f}$", ha="center", va="bottom", fontsize=8, color=GREEN)
    if cutoff is not None:
        for s in (-1, 1):
            ax.axvline(s * cutoff, color=BLUE, lw=0.6, ls=":")
        ax.text(cutoff + 1, -0.55, rf"cut-off $R_c = {cutoff:g}$", fontsize=8, color=BLUE)
    if surff is not None:
        for s in (-1, 1):
            ax.axvline(s * surff, color="black", lw=1.1, ls="--")
        ax.text(surff + 1, 0.12, f"t-SURFF\nsurface\n$R = {surff:g}$", fontsize=8, va="bottom")
    ax.set_xlim(-L, L)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"$x$ (a.u.)")
    ax.set_ylabel(r"$V(x)$ (Ha)")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(Path(__file__).with_name(name))
    plt.close(fig)


# examples/hhg/hhg_1d.inp
fig, ax = plt.subplots(figsize=(7.0, 3.0))
draw(ax, 70.0, 0.3, lambda x: -1.0 / np.sqrt(x * x + 1.0))
save(fig, "hhg_box.svg")

# examples/tsurff/ati_1d.inp
Rc, d = 20.0, 2.0
fig, ax = plt.subplots(figsize=(7.0, 3.0))
draw(ax, 80.0, 0.2, lambda x: -0.5 * (1 - np.tanh((np.abs(x) - Rc) / d)) / np.sqrt(x * x + 1),
     surff=35.0, cutoff=Rc)
save(fig, "tsurff_box.svg")
