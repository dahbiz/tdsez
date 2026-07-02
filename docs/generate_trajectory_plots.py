#!/usr/bin/env python3
"""Generate trajectory shape plots for TDSEZ docs."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Parameters from .inp files
A_lici = 0.107159
omega = 0.057
Omega = 0.005  # chirped rotation rate
beta = 0.00005  # chirp rate
A_liss = 0.077369
A_saw = 0.08578
A_tre = 0.053964

# Time range (10 cycles)
t_max = 10 * np.pi / omega
t = np.linspace(0, t_max, 3000)

# Envelope (sin²)
env = np.sin(omega * t / 10) ** 2

# === LICI: Chirped-Rotating (XZ plane) ===
Ex_lici = env * A_lici * np.cos(Omega * t) * np.sin(omega * t + beta * t**2)
Ez_lici = env * A_lici * np.sin(Omega * t) * np.sin(omega * t + beta * t**2)

fig, ax = plt.subplots(figsize=(5, 5))
fig.patch.set_facecolor('#111827')
ax.set_facecolor('#0a0f14')

# Plot trajectory
ax.plot(Ex_lici, Ez_lici, color='#8b5cf6', linewidth=1.2)

# Add arrow to show direction at a few points
for frac in [0.2, 0.5, 0.8]:
    idx = int(frac * len(t))
    dx = Ex_lici[idx+10] - Ex_lici[idx]
    dy = Ez_lici[idx+10] - Ez_lici[idx]
    ax.annotate('', xy=(Ex_lici[idx+10], Ez_lici[idx+10]),
                xytext=(Ex_lici[idx], Ez_lici[idx]),
                arrowprops=dict(arrowstyle='->', color='#a78bfa', lw=1.5))

ax.axhline(y=0, color='rgba(255,255,255,0.08)', linewidth=0.5)
ax.axvline(x=0, color='rgba(255,255,255,0.08)', linewidth=0.5)

ax.set_title('XZ TRAJECTORY SHAPE', fontsize=12, fontweight='bold',
             color='#e2e8f0', pad=15, fontname='Inter')
ax.set_xlabel(r'$E_x$ (a.u.)', fontsize=11, color='#94a3b8', fontname='Inter')
ax.set_ylabel(r'$E_z$ (a.u.)', fontsize=11, color='#94a3b8', fontname='Inter')

ax.tick_params(axis='both', labelsize=10, colors='#64748b')
ax.spines['bottom'].set_color('#334155')
ax.spines['left'].set_color('#334155')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.08, color='white', linestyle='-')
ax.set_aspect('equal')

plt.tight_layout()
fig.savefig('docs/assets/traf_lici.png', dpi=150, bbox_inches='tight',
            facecolor='#111827', edgecolor='none')
plt.close()
print("Generated: docs/assets/traf_lici.png")

# === LISS: OTC Figure-8 (XY plane) ===
Ex_liss = env * A_liss * np.sin(omega * t)
Ey_liss = env * A_liss * np.sin(2 * omega * t + np.pi / 2)

fig, ax = plt.subplots(figsize=(5, 5))
fig.patch.set_facecolor('#111827')
ax.set_facecolor('#0a0f14')

ax.plot(Ex_liss, Ey_liss, color='#8b5cf6', linewidth=1.2)

for frac in [0.25, 0.5, 0.75]:
    idx = int(frac * len(t))
    dx = Ex_liss[idx+10] - Ex_liss[idx]
    dy = Ey_liss[idx+10] - Ey_liss[idx]
    ax.annotate('', xy=(Ex_liss[idx+10], Ey_liss[idx+10]),
                xytext=(Ex_liss[idx], Ey_liss[idx]),
                arrowprops=dict(arrowstyle='->', color='#a78bfa', lw=1.5))

ax.axhline(y=0, color='rgba(255,255,255,0.08)', linewidth=0.5)
ax.axvline(x=0, color='rgba(255,255,255,0.08)', linewidth=0.5)

ax.set_title('XY TRAJECTORY SHAPE', fontsize=12, fontweight='bold',
             color='#e2e8f0', pad=15, fontname='Inter')
ax.set_xlabel(r'$E_x$ (a.u.)', fontsize=11, color='#94a3b8', fontname='Inter')
ax.set_ylabel(r'$E_y$ (a.u.)', fontsize=11, color='#94a3b8', fontname='Inter')

ax.tick_params(axis='both', labelsize=10, colors='#64748b')
ax.spines['bottom'].set_color('#334155')
ax.spines['left'].set_color('#334155')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.08, color='white', linestyle='-')
ax.set_aspect('equal')

plt.tight_layout()
fig.savefig('docs/assets/traf_liss.png', dpi=150, bbox_inches='tight',
            facecolor='#111827', edgecolor='none')
plt.close()
print("Generated: docs/assets/traf_liss.png")

# === SAW: Sawtooth OTC (XY plane) ===
Ex_saw = env * A_saw * np.cos(omega * t)
Ey_saw = env * A_saw * np.cos(2 * omega * t + np.pi / 2)

fig, ax = plt.subplots(figsize=(5, 5))
fig.patch.set_facecolor('#111827')
ax.set_facecolor('#0a0f14')

ax.plot(Ex_saw, Ey_saw, color='#8b5cf6', linewidth=1.2)

for frac in [0.25, 0.5, 0.75]:
    idx = int(frac * len(t))
    dx = Ex_saw[idx+10] - Ex_saw[idx]
    dy = Ey_saw[idx+10] - Ey_saw[idx]
    ax.annotate('', xy=(Ex_saw[idx+10], Ey_saw[idx+10]),
                xytext=(Ex_saw[idx], Ey_saw[idx]),
                arrowprops=dict(arrowstyle='->', color='#a78bfa', lw=1.5))

ax.axhline(y=0, color='rgba(255,255,255,0.08)', linewidth=0.5)
ax.axvline(x=0, color='rgba(255,255,255,0.08)', linewidth=0.5)

ax.set_title('XY TRAJECTORY SHAPE', fontsize=12, fontweight='bold',
             color='#e2e8f0', pad=15, fontname='Inter')
ax.set_xlabel(r'$E_x$ (a.u.)', fontsize=11, color='#94a3b8', fontname='Inter')
ax.set_ylabel(r'$E_y$ (a.u.)', fontsize=11, color='#94a3b8', fontname='Inter')

ax.tick_params(axis='both', labelsize=10, colors='#64748b')
ax.spines['bottom'].set_color('#334155')
ax.spines['left'].set_color('#334155')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.08, color='white', linestyle='-')
ax.set_aspect('equal')

plt.tight_layout()
fig.savefig('docs/assets/traf_saw.png', dpi=150, bbox_inches='tight',
            facecolor='#111827', edgecolor='none')
plt.close()
print("Generated: docs/assets/traf_saw.png")

# === TRE: Counter-Rotating Trefoil (XY plane) ===
Ex_tre = env * A_tre * (np.cos(omega * t) + np.cos(2 * omega * t))
Ey_tre = env * A_tre * (np.sin(omega * t) - np.sin(2 * omega * t))

fig, ax = plt.subplots(figsize=(5, 5))
fig.patch.set_facecolor('#111827')
ax.set_facecolor('#0a0f14')

ax.plot(Ex_tre, Ey_tre, color='#8b5cf6', linewidth=1.2)

for frac in [0.25, 0.5, 0.75]:
    idx = int(frac * len(t))
    dx = Ex_tre[idx+10] - Ex_tre[idx]
    dy = Ey_tre[idx+10] - Ey_tre[idx]
    ax.annotate('', xy=(Ex_tre[idx+10], Ey_tre[idx+10]),
                xytext=(Ex_tre[idx], Ey_tre[idx]),
                arrowprops=dict(arrowstyle='->', color='#a78bfa', lw=1.5))

ax.axhline(y=0, color='rgba(255,255,255,0.08)', linewidth=0.5)
ax.axvline(x=0, color='rgba(255,255,255,0.08)', linewidth=0.5)

ax.set_title('XY TRAJECTORY SHAPE', fontsize=12, fontweight='bold',
             color='#e2e8f0', pad=15, fontname='Inter')
ax.set_xlabel(r'$E_x$ (a.u.)', fontsize=11, color='#94a3b8', fontname='Inter')
ax.set_ylabel(r'$E_y$ (a.u.)', fontsize=11, color='#94a3b8', fontname='Inter')

ax.tick_params(axis='both', labelsize=10, colors='#64748b')
ax.spines['bottom'].set_color('#334155')
ax.spines['left'].set_color('#334155')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.08, color='white', linestyle='-')
ax.set_aspect('equal')

plt.tight_layout()
fig.savefig('docs/assets/traf_tre.png', dpi=150, bbox_inches='tight',
            facecolor='#111827', edgecolor='none')
plt.close()
print("Generated: docs/assets/traf_tre.png")

print("\nAll trajectory shape plots generated successfully.")
