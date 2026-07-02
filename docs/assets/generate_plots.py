#!/usr/bin/env python3
"""Generate clean 2D waveform plots for TDSE-ⵣ docs."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DPI = 120
FONTSIZE = 10
LABELSIZE = 11
TITLEFONTSIZE = 12

def env(t):
    return np.sin(t * 0.057 / 10) ** 2

def save_2d_plot(components, name, title, xlim, fig_size=(7.5, 4.2)):
    fig, ax = plt.subplots(figsize=fig_size)
    
    colors = ['#ef4444', '#3b82f6', '#22c55e']
    
    for i, (label, fn) in enumerate(components):
        t = np.linspace(xlim[0], xlim[1], 1500)
        y = np.array([fn(t_) for t_ in t])
        ax.plot(t, y, color=colors[i], linewidth=1.2, label=label)
    
    ax.set_title(title, fontsize=TITLEFONTSIZE, fontweight='bold', pad=15)
    ax.set_xlabel('t (a.u.)', fontsize=LABELSIZE)
    ax.set_ylabel('E (a.u.)', fontsize=LABELSIZE)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(True, alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(name, dpi=DPI, bbox_inches='tight', facecolor='#111827', edgecolor='none')
    plt.close(fig)
    print(f"Saved {name}")

# === lici: Chirped-Rotating (XZ plane) ===
lici_comp = [
    ('Ex', lambda t: env(t) * 0.107159 * np.cos(0.005*t) * np.sin(0.057*t + 0.00005*t*t)),
    ('Ez', lambda t: env(t) * 0.107159 * np.sin(0.005*t) * np.sin(0.057*t + 0.00005*t*t)),
]
save_2d_plot(lici_comp, 
             '/Users/dahbi/Desktop/JCP/TDSEZ/docs/assets/wf_lici.png',
             'CHIRPED-ROTATING PULSE  (XZ Plane)',
             (0, 10 * np.pi / 0.057 + 0.1))

# === liss: OTC Figure-8 Lissajous (XY plane) ===
liss_comp = [
    ('Ex', lambda t: env(t) * 0.077369 * np.sin(0.057*t)),
    ('Ey', lambda t: env(t) * 0.077369 * np.sin(2*0.057*t + np.pi/2)),
]
save_2d_plot(liss_comp,
             '/Users/dahbi/Desktop/JCP/TDSEZ/docs/assets/wf_liss.png',
             'OTC FIGURE-8 LISSAJOUS  (XY Plane)',
             (0, 10 * np.pi / 0.057 + 0.1))

# === saw: Sawtooth OTC (XY plane) ===
saw_comp = [
    ('Ex', lambda t: env(t) * 0.08578 * np.cos(0.057*t)),
    ('Ey', lambda t: env(t) * 0.08578 * np.cos(2*0.057*t + np.pi/2)),
]
save_2d_plot(saw_comp,
             '/Users/dahbi/Desktop/JCP/TDSEZ/docs/assets/wf_saw.png',
             'SAWTOOTH OTC WAVEFORM  (XY Plane)',
             (0, 10 * np.pi / 0.057 + 0.1))

# === tre: Counter-Rotating Trefoil (XY plane) ===
tre_comp = [
    ('Ex', lambda t: env(t) * 0.053964 * (np.cos(0.057*t) + np.cos(2*0.057*t))),
    ('Ey', lambda t: env(t) * 0.053964 * (np.sin(0.057*t) - np.sin(2*0.057*t))),
]
save_2d_plot(tre_comp,
             '/Users/dahbi/Desktop/JCP/TDSEZ/docs/assets/wf_tre.png',
             'COUNTER-ROTATING TREFOIL  (XY Plane)',
             (0, 10 * np.pi / 0.057 + 0.1))

print("Done — 4 plots generated.")
