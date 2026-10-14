#!/usr/bin/env python3
"""Dipole <x>(t) of the 1D driven harmonic oscillator for the three propagators.

Reads run_<prop>/td/TimeEvolutionData_*.h5 (dipoles dataset: [t, E_x, q<x>, q a_x])
and the propagation wall time from run_<prop>/run.log, and writes
propagator_dipoles.png next to this script.
"""
import glob, os, re
import h5py, numpy as np, matplotlib as mpl
mpl.use('Agg'); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
E0, W = 1.0, 0.6

def x_exact(t):
    return -E0 / (1.0 - W * W) * (np.cos(W * t) - np.cos(t))

def charge(run_dir):
    """Charge q of the run (dipole column = q<x>), from the EigenData metadata."""
    for path in glob.glob(os.path.join(run_dir, 'static', 'EigenData_*.h5')):
        with h5py.File(path, 'r') as g:
            if 'run_metadata' in g and 'Charge' in g['run_metadata'].attrs:
                return float(g['run_metadata'].attrs['Charge'])
    return -1.0  # TDSE-Z default (electron)

def load(prop):
    d = os.path.join(HERE, f'run_{prop}')
    f = h5py.File(glob.glob(os.path.join(d, 'td', 'TimeEvolutionData_*.h5'))[0], 'r')
    dip = np.real(f['dipoles'][()]).reshape(-1, 4)
    log = open(os.path.join(d, 'run.log'), errors='ignore').read()
    m = re.search(r'^\s+Propagation\s+([0-9.]+)', log, re.M)
    dt = re.search(r'^TimeStep\s*=\s*([0-9.eE+-]+)', open(os.path.join(d, f'ho1d_{prop}.inp')).read(), re.M)
    return dip[:, 0], dip[:, 2] / charge(d), float(m.group(1)) if m else np.nan, dt.group(1)

mpl.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10,
                     "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True, "legend.frameon": False})
style = {'cn': ('#1f77b4', 'Crank–Nicolson'), 'pade22': ('#ff7f0e', 'Padé(2,2) midpoint'),
         'magnus4': ('#2ca02c', 'Magnus-4 (CF4)')}

fig, ax = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True, gridspec_kw=dict(hspace=0.08))
tt = np.linspace(0, 50, 4001)
ax[0].plot(tt, x_exact(tt), 'k-', lw=2.4, alpha=0.35, label='exact')
for prop in ['cn', 'pade22', 'magnus4']:
    t, x, wall, dt = load(prop)
    col, lab = style[prop]
    ax[0].plot(t, x, '-', color=col, lw=1.0, label=rf'{lab}, $\Delta t={dt}$')
    err = np.abs(x - x_exact(t))
    ax[1].semilogy(t[1:], err[1:], '-', color=col, lw=0.9,
                   label=rf'{lab}: max {err.max():.1e}, {wall:.1f} s')
    print(f'{prop:8s} dt={dt:6s} max|x-x_ex|={err.max():.2e} propagation={wall:.2f} s')
ax[0].set_ylabel(r'$\langle x\rangle(t)$ (a.u.)'); ax[0].set_ylim(-3.4, 4.8); ax[0].legend(fontsize=8, loc='upper right', ncol=2)
ax[1].set_ylabel(r'$|\langle x\rangle-x_{\mathrm{ex}}|$ (a.u.)'); ax[1].set_xlabel(r'$t$ (a.u.)')
ax[1].set_xlim(0, 50); ax[1].legend(fontsize=8, loc='lower right')
for a, l in zip(ax, 'ab'):
    a.text(0.01, 0.95, f'({l})', transform=a.transAxes, fontweight='bold', va='top')
fig.savefig(os.path.join(HERE, 'propagator_dipoles.png'), dpi=160, bbox_inches='tight')
