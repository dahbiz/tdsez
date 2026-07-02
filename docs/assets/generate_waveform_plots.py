#!/usr/bin/env python3
"""Generate clean 2D waveform plot images for TDSEZ docs."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def env(t):
    return np.sin(t * 0.057 / 10) ** 2

t_max = 10 * np.pi / 0.057
t = np.linspace(0, t_max, 2000)

configs = [
    {
        'key': 'lici',
        'title': 'CHIRPED-ROTATING PULSE',
        'plane': '(XZ Plane)',
        'xlabel': 't (a.u.)',
        'ylabel': r'$E_x$, $E_z$ (a.u.)',
        'components': [
            ('$E_x$', env(t) * 0.107159 * np.cos(0.005*t) * np.sin(0.057*t + 0.00005*t*t), '#ef4444'),
            ('$E_z$', env(t) * 0.107159 * np.sin(0.005*t) * np.sin(0.057*t + 0.00005*t*t), '#3b82f6'),
        ],
        'subtitle': 'XZ chirped-rotating pulse · 10 cycles · ω = 0.057 a.u. · |E| = 0.107 a.u.',
    },
    {
        'key': 'liss',
        'title': 'OTC FIGURE-8 LISSAJOUS',
        'plane': '(XY Plane)',
        'xlabel': 't (a.u.)',
        'ylabel': r'$E_x$, $E_y$ (a.u.)',
        'components': [
            ('$E_x$', env(t) * 0.077369 * np.sin(0.057*t), '#ef4444'),
            ('$E_y$', env(t) * 0.077369 * np.sin(2*0.057*t + np.pi/2), '#3b82f6'),
        ],
        'subtitle': 'OTC figure-8 · 10 cycles · A_ω = A_2ω = 0.077 a.u. · |E| = 0.107 a.u.',
    },
    {
        'key': 'saw',
        'title': 'SAWTOOTH OTC WAVEFORM',
        'plane': '(XY Plane)',
        'xlabel': 't (a.u.)',
        'ylabel': r'$E_x$, $E_y$ (a.u.)',
        'components': [
            ('$E_x$', env(t) * 0.08578 * np.cos(0.057*t), '#ef4444'),
            ('$E_y$', env(t) * 0.08578 * np.cos(2*0.057*t + np.pi/2), '#3b82f6'),
        ],
        'subtitle': 'Sawtooth OTC · 10 cycles · A_ω = A_2ω = 0.086 a.u. · |E| = 0.107 a.u.',
    },
    {
        'key': 'tre',
        'title': 'COUNTER-ROTATING TREFOIL',
        'plane': '(XY Plane)',
        'xlabel': 't (a.u.)',
        'ylabel': r'$E_x$, $E_y$ (a.u.)',
        'components': [
            ('$E_x$', env(t) * 0.053964 * (np.cos(0.057*t) + np.cos(2*0.057*t)), '#ef4444'),
            ('$E_y$', env(t) * 0.053964 * (np.sin(0.057*t) - np.sin(2*0.057*t)), '#3b82f6'),
        ],
        'subtitle': 'Counter-rotating · 10 cycles · A_ω = A_2ω = 0.054 a.u. · |E| = 0.107 a.u.',
    },
]

for cfg in configs:
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#0a0f14')
    
    for label, y_data, color in cfg['components']:
        ax.plot(t, y_data, color=color, linewidth=2.0, label=label)
    
    ax.set_title(f"{cfg['title']}", fontsize=13, fontweight='bold', 
                 color='#e2e8f0', pad=12, fontname='Inter')
    ax.set_xlabel(cfg['xlabel'], fontsize=11, color='#94a3b8', fontname='Inter')
    ax.set_ylabel(cfg['ylabel'], fontsize=11, color='#94a3b8', fontname='Inter')
    
    ax.tick_params(axis='both', labelsize=9, colors='#64748b')
    ax.spines['bottom'].set_color('#334155')
    ax.spines['left'].set_color('#334155')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.grid(True, alpha=0.08, color='white', linestyle='-')
    
    plt.tight_layout()
    
    path = f'/Users/dahbi/Desktop/JCP/TDSEZ/docs/assets/wf_{cfg["key"]}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='#111827', edgecolor='none')
    plt.close(fig)
    print(f"Generated {path}")

print("All 4 waveform plots generated successfully.")
