#!/usr/bin/env python3
"""Read a TDSE-Z t-SURFF file (td/<name>_surff.h5) and write the spectrum.

    python plot_tsurff.py td/ati_1d_surff.h5 --energy --omega 0.057 --plot ati.png

The file holds the canonical-momentum amplitudes b(k) (``amplitude``, re/im)
and dP/d^dk = |b(k)|^2 (``pes``) on the momenta in ``momenta``; a tensor grid
also stores the axes ``k_x``/``k_y``/``k_z`` and ``pes_grid`` (x slowest).
The final kinetic momentum is k + A(T) (attribute ``A_final``).

1D: writes k, dP/dk; with --energy also E = k^2/2m and
    dP/dE = (m/|k|) (|b(k)|^2 + |b(-k)|^2)  (both directions, needs a k grid
    symmetric about 0).
2D/3D grids: writes the angle-integrated dP/dE from the grid (2D: rings,
    3D: shells), binned in energy.
Prints the total yield int |b|^2 d^dk over the grid.
"""

import argparse
import os

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("h5")
    ap.add_argument("--energy", action="store_true", help="also write dP/dE")
    ap.add_argument("--omega", type=float, help="photon energy: print energies in units of omega too")
    ap.add_argument("--bins", type=int, default=400, help="energy bins (2D/3D)")
    ap.add_argument("--plot", help="write a plot (needs matplotlib)")
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        k = np.asarray(f["momenta"])
        pes = np.asarray(f["pes"])
        attrs = {a: np.asarray(f.attrs[a]) for a in f.attrs}
        grid = "pes_grid" in f
        axes = [np.asarray(f[n]) for n in ("k_x", "k_y", "k_z") if n in f]
    dim = k.shape[1]
    m = float(attrs["exterior_mass"][0])
    stem = os.path.splitext(os.path.basename(args.h5))[0]
    print(f"{args.h5}: {dim}D, {len(pes)} momenta, T = {attrs['final_time'][0]:g}, "
          f"R = {attrs['radius'][0]:g}, A(T) = {attrs['A_final']}, interior = {int(attrs['interior'][0])}")
    if grid:
        dk = np.prod([a[1] - a[0] if len(a) > 1 else 1.0 for a in axes])
        print(f"total yield over the grid: {pes.sum() * dk:.6g}")

    out = [np.column_stack([k, pes])]
    head = " ".join(f"k{'xyz'[d]}" for d in range(dim)) + " dP/dk"
    np.savetxt(f"{stem}_pes.dat", out[0], header=head)
    print(f"wrote {stem}_pes.dat")

    E = dPdE = None
    if args.energy:
        if dim == 1:
            kk = k[:, 0]
            pos = kk > 0
            neg = np.interp(-kk[pos], kk, pes)          # |b(-k)|^2 on the same |k|
            E = kk[pos] ** 2 / (2 * m)
            dPdE = m / kk[pos] * (pes[pos] + neg)
        else:
            if not grid:
                raise SystemExit("--energy in 2D/3D needs a momentum grid")
            e = (k ** 2).sum(axis=1) / (2 * m)
            hist, edges = np.histogram(e, bins=args.bins, range=(0, e.max()), weights=pes * dk)
            E = 0.5 * (edges[1:] + edges[:-1])
            dPdE = hist / np.diff(edges)
        cols = [E, dPdE] + ([E / args.omega] if args.omega else [])
        np.savetxt(f"{stem}_dPdE.dat", np.column_stack(cols),
                   header="E dP/dE" + (" E/omega" if args.omega else ""))
        print(f"wrote {stem}_dPdE.dat")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        if E is not None:
            ax.semilogy(E / args.omega if args.omega else E, dPdE, lw=0.8)
            ax.set_xlabel("E / omega" if args.omega else "E (a.u.)")
            ax.set_ylabel("dP/dE")
        elif dim == 1:
            ax.semilogy(k[:, 0], pes, lw=0.8)
            ax.set_xlabel("k (a.u.)")
            ax.set_ylabel("dP/dk")
        else:
            g = np.asarray(h5py.File(args.h5, "r")["pes_grid"])
            g = g.sum(axis=2) if g.ndim == 3 else g
            ax.pcolormesh(axes[0], axes[1], np.log10(g.T + 1e-30 * g.max()), shading="auto")
            ax.set_xlabel("k_x")
            ax.set_ylabel("k_y")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
