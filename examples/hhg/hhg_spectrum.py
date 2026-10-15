#!/usr/bin/env python3
"""HHG spectrum from a TDSE-Z run.

    python hhg_spectrum.py td/TimeEvolutionData_hhg_1d.inp.h5 --omega 0.057

Reads the dipole table (t, E, D = q<x>, A = q d<v>/dt per axis) and the
current table (J = q<v>), and computes the emitted spectrum three equivalent
ways:

    S_A(w) = |FT[A](w)|^2,   S_J(w) = w^2 |FT[J](w)|^2,   S_D(w) = w^4 |FT[D](w)|^2

Each signal is multiplied by a window (Hann by default) over the whole record
before the transform. S_A is the reference; the three agree up to the plateau
cutoff when the run is converged and the box absorbs the electrons that do not
return (CAP). A dipole spectrum S_D that lies far above S_A at low and plateau
orders shows that free electrons drift (reflections, too small a box, or a
field with a non-zero area).
Writes <stem>_hhg.dat: order, S_A, S_J, S_D; optional --plot file.
"""

import argparse
import os

import h5py
import numpy as np


def spectrum(t, s, window):
    """|FT[window * s]|^2 on the positive angular frequencies."""
    dt = t[1] - t[0]
    w = {"hann": np.hanning, "blackman": np.blackman, "none": np.ones}[window](len(t))
    f = np.fft.rfft(s * w) * dt
    return 2 * np.pi * np.fft.rfftfreq(len(t), dt), np.abs(f) ** 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5", help="td/TimeEvolutionData_<input>.h5")
    ap.add_argument("--omega", type=float, required=True, help="laser angular frequency (a.u.)")
    ap.add_argument("--axis", default="x", choices="xyz", help="emission axis (default x)")
    ap.add_argument("--window", default="hann", choices=["hann", "blackman", "none"])
    ap.add_argument("--max-order", type=float, default=60)
    ap.add_argument("--plot", help="write a log-scale plot (png/pdf)")
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        dip = np.asarray(f["dipoles"])
        cur = np.asarray(f["currents"])
    dim = {4: 1, 7: 2, 10: 3}[dip.shape[1]]
    a = "xyz".index(args.axis)
    if a >= dim:
        raise SystemExit(f"axis {args.axis} not present in a {dim}D run")
    t = dip[:, 0]
    if not np.allclose(np.diff(t), t[1] - t[0], rtol=1e-6):
        raise SystemExit("time samples are not uniform")
    D, A = dip[:, 1 + dim + a], dip[:, 1 + 2 * dim + a]
    J = cur[:, 3 + a]

    w, SA = spectrum(t, A, args.window)
    _, SJ = spectrum(t, J, args.window)
    _, SD = spectrum(t, D, args.window)
    SJ, SD = w ** 2 * SJ, w ** 4 * SD
    order = w / args.omega
    keep = order <= args.max_order
    stem = os.path.splitext(os.path.basename(args.h5))[0]
    out = f"{stem}_hhg.dat"
    np.savetxt(out, np.column_stack([order, SA, SJ, SD])[keep],
               header="order S_A=|A(w)|^2 S_J=w^2|J(w)|^2 S_D=w^4|D(w)|^2")
    print(f"wrote {out}  ({keep.sum()} frequencies, dw = {w[1]:.4g} a.u. = {w[1] / args.omega:.3g} orders)")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for s, lab in ((SA, r"$|\tilde A|^2$"), (SJ, r"$\omega^2|\tilde J|^2$"), (SD, r"$\omega^4|\tilde D|^2$")):
            ax.semilogy(order[keep], s[keep], lw=0.8, label=lab)
        ax.set_xlabel("harmonic order")
        ax.set_ylabel("spectrum (a.u.)")
        ax.set_xticks(np.arange(1, args.max_order + 1, 4))
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
