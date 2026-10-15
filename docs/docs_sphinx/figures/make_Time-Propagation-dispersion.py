"""Phase error of the field-free step of Crank-Nicolson and Pade(2,2).

For an eigencomponent of energy E the exact step multiplies it by
exp(-i theta), theta = E h. Crank-Nicolson (Pade(1,1)) applies
(1 - i theta/2) / (1 + i theta/2), Pade(2,2) applies
(1 - i theta/2 - theta^2/12) / (1 + i theta/2 - theta^2/12). Both have
modulus one; their phases are 2 atan(theta/2) and
2 atan2(theta/2, 1 - theta^2/12). The plot shows the relative phase error
|phi/theta - 1| (pure formulas, no simulation data).
Output: Time-Propagation-dispersion.svg next to this script.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

theta = np.logspace(-2, np.log10(3.0), 400)
phi_cn = 2.0 * np.arctan(theta / 2.0)
phi_p22 = 2.0 * np.arctan2(theta / 2.0, 1.0 - theta**2 / 12.0)

err_cn = np.abs(phi_cn / theta - 1.0)
err_p22 = np.abs(phi_p22 / theta - 1.0)

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.loglog(theta, err_cn, color="#3b6ea5", lw=2.0, label="Crank-Nicolson, Pade(1,1)")
ax.loglog(theta, err_p22, color="#c0612b", lw=2.0, label="Pade(2,2) (pade22, magnus4)")
ax.loglog(theta, theta**2 / 12.0, color="#3b6ea5", lw=0.8, ls="--", label=r"$\theta^2/12$")
ax.loglog(theta, theta**4 / 720.0, color="#c0612b", lw=0.8, ls="--", label=r"$\theta^4/720$")
ax.set_xlabel(r"$\theta = E\,h$ (energy $\times$ step length)")
ax.set_ylabel(r"relative phase error $|\varphi/\theta - 1|$")
ax.set_ylim(1e-14, 1.0)
ax.grid(True, which="major", color="#dddddd", lw=0.6)
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
fig.tight_layout()
fig.savefig(Path(__file__).with_name("Time-Propagation-dispersion.svg"), transparent=True)
