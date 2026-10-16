"""Static Kramers-Henneberger states of a 1D soft-core atom versus the quiver amplitude.

V(x) = -1 / sqrt(x^2 + a^2), a = 1.44, in a CW field E0 sin(w t) with w = 1, so
the quiver amplitude is alpha0 = E0 / w^2 = E0. For each alpha0 TDSE-Z
(Mode = kh) solves the eigenstates of the cycle-averaged potential
V_KH(x) = < V(x + alpha(t)) >; alpha0 = 0 is the bare atom (Mode = td).

    python kh_soft_core_1d.py [--binary build/tdsez.x] [--out kh_out]

Writes kh_potential.png, kh_states.png, kh_density_map.png, kh_energies.png
and kh_levels.csv.
"""

import argparse
import csv
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import zkit
from zkit.io.wfs_field import reconstruct_static_wfs

A = 1.44
L = 140.0                  # box half width: room for x ~ +-alpha0 and the long-range tail
N_STATES = 8
ALPHAS = np.round(np.arange(0.0, 20.0001, 0.25), 2)
SHOWN = (0.0, 2.0, 5.0, 10.0, 20.0)

DECK = """Verbose = 0
Mode = {mode}
EnablePropagation = 0
Dimension = 1
Domain = -{L}, {L}
SplineDegree = 6
NQuadratures = 10
KnotSequence = uniform
NSplines = 1400
Potential = -1/sqrt(x*x + {a2})
NBoundStates = {n}
NBoundStatesSave = 1
TargetEigenvalue = -1.0
Polarization = x
LaserX = {E0}*sin(t)
Omegax = 1.0
FinalTime = 20
"""


def run(binary, out, alpha):
    name = f"kh_a{alpha:05.2f}".replace(".", "p")
    deck = DECK.format(mode="kh" if alpha > 0 else "td", L=L, a2=A * A, n=N_STATES, E0=alpha)
    (out / f"{name}.inp").write_text(deck)
    r = subprocess.run([binary, "-inp", f"{name}.inp"], cwd=out, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{name}: TDSE-Z failed\n{r.stdout[-1500:]}")
    return alpha, zkit.Run(out, f"{name}.inp")


def v_kh(x, run, alpha):
    """V_KH from the quiver TDSE-Z used (run_metadata/kh_alpha, kh_weights)."""
    v = lambda xx: -1.0 / np.sqrt(xx * xx + A * A)  # noqa: E731
    if alpha == 0:
        return v(x)
    with h5py.File(run.path("eigen")) as f:
        g = f["run_metadata"]
        w, al = g["kh_weights"][:], g["kh_alpha"][:, 0]
    return sum(wk * v(x + ak) for wk, ak in zip(w, al))


def states(run, x):
    """Bound states (E < 0) sorted by energy: energies and normalised densities on x."""
    E = run.spectrum.values
    out = []
    for i in np.argsort(E):
        if E[i] >= 0:
            continue
        psi = reconstruct_static_wfs(run.path("eigen"), int(i), axes=[x])["psi"]
        rho = np.abs(np.asarray(psi).ravel()) ** 2
        out.append((E[i], rho / np.trapezoid(rho, x)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--binary", default=None, help="tdsez.x (default: TDSEZ_BINARY or PATH)")
    ap.add_argument("--out", default="kh_out", help="run and figure directory")
    ap.add_argument("--jobs", type=int, default=4, help="runs in parallel")
    args = ap.parse_args()
    binary = os.path.abspath(args.binary or zkit.find_binary())
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    with ThreadPoolExecutor(args.jobs) as pool:
        runs = dict(pool.map(lambda a: run(binary, out, a), ALPHAS))

    levels = {a: np.sort(r.spectrum.values) for a, r in runs.items()}
    with open(out / "kh_levels.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["alpha0"] + [f"E{n}" for n in range(N_STATES)])
        for a in ALPHAS:
            wr.writerow([a] + [f"{e:.12f}" for e in levels[a][:N_STATES]])

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
    cmap = plt.get_cmap("viridis")
    colors = {a: cmap(i / (len(SHOWN) - 1)) for i, a in enumerate(SHOWN)}

    # 1. the dressed potential
    x = np.linspace(-35, 35, 4001)
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for a in SHOWN:
        ax.plot(x, v_kh(x, runs[a], a), color=colors[a], lw=1.8, label=rf"$\alpha_0 = {a:g}$")
    ax.set_xlim(-30, 30)
    ax.set_ylim(-0.75, 0.02)
    ax.set_xlabel("x (a.u.)")
    ax.set_ylabel(r"$V_{\mathrm{KH}}(x)$ (a.u.)")
    ax.set_title(rf"Kramers-Henneberger potential, soft core $a = {A}$")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "kh_potential.png", dpi=180)

    # 2. ground and first excited dressed states, V_KH on a twin axis
    xs = np.linspace(-32, 32, 3201)
    fig, axs = plt.subplots(2, len(SHOWN), figsize=(13, 5.6), sharex=True, sharey="row")
    for j, a in enumerate(SHOWN):
        st = states(runs[a], xs)
        vk = v_kh(xs, runs[a], a)
        for i, ax in enumerate(axs[:, j]):
            if i >= len(st):
                continue
            E, rho = st[i]
            c = plt.get_cmap("plasma")(0.15 + 0.45 * i)
            ax.fill_between(xs, 0, rho, color=c, alpha=0.35, lw=0)
            ax.plot(xs, rho, color=c, lw=1.4)
            tw = ax.twinx()
            tw.plot(xs, vk, color="0.45", lw=0.9, ls="--")
            tw.set_ylim(-0.75, 0.05)
            if j == len(SHOWN) - 1:
                tw.tick_params(colors="0.45")
            else:
                tw.set_yticks([])
            tw.grid(False)
            if j == len(SHOWN) - 1:
                tw.set_ylabel(r"$V_{\mathrm{KH}}$ (a.u.)", color="0.45")
            for sgn in (-1, 1):
                ax.axvline(sgn * a, color="0.6", lw=0.7, ls=":")
            ax.text(0.03, 0.93, rf"$E_{i} = {E:.4f}$", transform=ax.transAxes, fontsize=8, va="top")
            if i == 0:
                ax.set_title(rf"$\alpha_0 = {a:g}$", fontsize=10)
        axs[1, j].set_xlabel("x (a.u.)")
    axs[0, 0].set_ylabel(r"$|\psi_0(x)|^2$")
    axs[1, 0].set_ylabel(r"$|\psi_1(x)|^2$")
    axs[0, 0].set_xlim(-32, 32)
    fig.suptitle(r"KH ground and first excited states: dichotomy into lobes at $x \approx \pm\alpha_0$ (dotted)")
    fig.tight_layout()
    fig.savefig(out / "kh_states.png", dpi=180)

    # 2b. ground-state density versus x and the quiver amplitude
    xm = np.linspace(-26, 26, 1041)
    dens = np.array([states(runs[a], xm)[0][1] for a in ALPHAS])
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.grid(False)
    im = ax.pcolormesh(xm, ALPHAS, dens, cmap="magma", shading="auto", vmax=np.percentile(dens, 99.5))
    ax.plot(ALPHAS, ALPHAS, color="w", lw=0.8, ls="--")
    ax.plot(-ALPHAS, ALPHAS, color="w", lw=0.8, ls="--", label=r"$x = \pm\alpha_0$")
    ax.set_xlabel("x (a.u.)")
    ax.set_ylabel(r"quiver amplitude $\alpha_0$ (a.u.)")
    ax.set_title(r"KH ground-state density $|\psi_0(x)|^2$")
    ax.legend(loc="upper center", fontsize=8, framealpha=0.3)
    fig.colorbar(im, ax=ax, label=r"$|\psi_0|^2$ (a.u.)")
    fig.tight_layout()
    fig.savefig(out / "kh_density_map.png", dpi=180)

    # 3. levels versus the quiver amplitude
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for n in range(6):
        E = np.array([levels[a][n] if len(levels[a]) > n else np.nan for a in ALPHAS])
        E = np.where(E < 0, E, np.nan)
        ax.plot(ALPHAS, E, lw=1.8, color=plt.get_cmap("plasma")(n / 6), label=f"n = {n}")
    for a in SHOWN[1:]:
        ax.axvline(a, color="0.75", lw=0.7, ls=":")
    ax.set_xlabel(r"quiver amplitude $\alpha_0 = E_0/\omega^2$ (a.u.)")
    ax.set_ylabel(r"$E_n$ (a.u.)")
    ax.set_title(r"KH energies versus the quiver amplitude")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "kh_energies.png", dpi=180)
    files = ("kh_potential.png", "kh_states.png", "kh_density_map.png", "kh_energies.png", "kh_levels.csv")
    print("wrote", ", ".join(str(out / f) for f in files))


if __name__ == "__main__":
    main()
