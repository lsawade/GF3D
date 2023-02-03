# %%
# LOAD THE KDTREE FIRST!!!!!!!!!!!!!
# If other packages are loaded, the wrong libstdc++ is picked up and
# doesn't contain the right GLIBCXX version

import matplotlib
from copy import deepcopy
import toml
from lwsspy.GF.seismograms import SGTManager
from obspy import read, Stream
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from lwsspy.plot import plot_label
from lwsspy.GF.source import CMTSOLUTION
# from lwsspy.GF.postprocess import Adios2HDF5
from lwsspy.GF.seismograms import get_seismograms, get_seismograms_sub, get_frechet
from lwsspy.GF.simulation import Simulation
from lwsspy.GF.stf import create_stf

import matplotlib.dates as mdates
from scipy import integrate, fft

# Get file name
specfemmagic = '/scratch/gpfs/lsawade/SpecfemMagicGF'
stationdir = os.path.join(specfemmagic, 'DB', 'II', 'BFO')
h5file = os.path.join(stationdir,  'II.BFO.h5')

# %% Get CMTSOLUTION
cmt = CMTSOLUTION.read('CMTSOLUTION')

# %% Define Processing function


def process_stream(st: Stream):
    st.filter('bandpass', freqmin=1/300.0, freqmax=1/40.0, zerophase=True)

# %% Get seismograms


dl = 0.4
minl = -4.0
maxl = 4.0 + dl
pert = np.arange(minl, maxl, dl)
traces = []
for i in pert:

    ocmt = deepcopy(cmt)
    # ocmt.longitude = cmt.longitude + i
    ocmt.longitude = cmt.longitude + i

    tr = get_seismograms(h5file, ocmt).select(component='Z')[0]
    traces.append(tr)

st = Stream(traces=traces)

# %% Process stream
process_stream(st)

# %% Get time limits

starttime = st[0].stats.starttime + 700
endtime = starttime + 9000
limits = (starttime.datetime, endtime.datetime)

# %% Plot figure

plt.rcParams["font.family"] = "monospace"
plt.close('all')
plt.figure()
fig = plt.figure(figsize=(7, 4))
ax = plt.gca()
lw = 1

absmax = np.max(np.abs(st.max()))

for _tr, _pert in zip(st, pert):

    plt.plot(_tr.times("matplotlib"), _tr.data /
             absmax*dl+_pert, 'k', label=f"{_pert:.1f}",
             lw=lw)

plot_label(ax, f'dLon', dist=0.05, location=13,
           fontsize='medium', box=False)

plot_label(
    ax, f'max|A|: {absmax:.5g} m', dist=0, location=2, fontsize='xx-small', box=False)

ax.xaxis_date()
ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
# ax.tick_params(labelleft=True, left=True)
ax.spines.right.set_visible(False)
# ax.spines.left.set_visible(False)
ax.spines.top.set_visible(False)
ax.set_xlim(limits)

network = st[0].stats.network
station = st[0].stats.station
plot_label(ax,
           f"{cmt.cmt_time.strftime('%Y-%m-%d %H:%M:%S')}  Loc: {cmt.latitude}dg, {cmt.longitude}dg, {cmt.depth}km\n"
           f"{network}.{station} -- BP: [40s, 300s]",
           fontsize='medium', box=False, location=6)
plt.xlabel('Time')


fig.autofmt_xdate()


plt.savefig('eventsection.pdf')
