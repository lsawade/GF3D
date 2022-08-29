import os
from scipy.special import erf
from numpy import *
from matplotlib.pyplot import *
from lwsspy.GF.stf import gaussian, erf

# File paths
SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))
FIGUREDIR = os.path.dirname(SCRIPTDIR)


t0 = -5
tc = 0.0
dt = 0.01
nt = 1001

hdur = 1.5

t = arange(t0, t0+nt*dt, dt)

# In specfem the hdur is used like standard deviation for the force.
g = exp(-(t-tc)**2/hdur**2) / (sqrt(pi) * hdur)

e = 0.5*erf((t-tc)/hdur)+0.5

figure()
plot(t,g)
plot(t,e)
plot(t, np.gradient(e, t))
plot(t, np.gradient(e, t)-g)
xlim(min(t), max(t))
title('Comparing Erf derivative and Gaussian')


plt.savefig(os.path.join(FIGUREDIR, 'erf-gaussian.pdf'))
