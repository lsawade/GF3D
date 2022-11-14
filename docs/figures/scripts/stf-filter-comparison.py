import os
import numpy as np
import matplotlib.pyplot as plt
from lwsspy.GF.signal.plot import compare_lowpass_filters

# File paths
SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
FIGUREDIR = os.path.dirname(SCRIPTDIR)

# Sampling time and frequency
dt = 0.2
fs = 1/dt

# Time vecotr
t = np.arange(5000)*dt - 200

# Approximate specfem STF
k = 0.25 * fs
H = 1.0/(1 + np.exp(-k*t))

# Set filter parameters

# Cutoff for butter
cutoff = 1/20

# Cutoff for bessel filter
bcutoff = 1.5*cutoff

# Cheby check db diff
rp1 = 0.05
rp2 = 20

# Filter orders
border = 6     # Butter
corder = 10    # Chebychev
bessorder = 7  # Bessel

# New sample spacing
ndt = 1/(cutoff*5)

fig, axes = compare_lowpass_filters(
    t, H, cutoff, bcutoff, ndt,
    border=border, corder=corder, bessorder=bessorder,
    rp1=rp1, rp2=rp2)

# Reset legend and axes limits
axes[0].legend(frameon=False)
axes[1].set_xlim(-50, 50)
axes[1].legend(frameon=False)
axes[2].set_xlim(-50, 50)
axes[2].legend(frameon=False)

plt.savefig(os.path.join(FIGUREDIR, 'stf-comparison.pdf'))
