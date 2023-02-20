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
import matplotlib.pyplot as plt
from lwsspy.GF.plot.util import plot_label
from lwsspy.GF.source import CMTSOLUTION
# from lwsspy.GF.postprocess import Adios2HDF5
from lwsspy.GF.seismograms import get_seismograms, get_seismograms_sub, get_frechet
from lwsspy.GF.simulation import Simulation
from lwsspy.GF.stf import create_stf
from lwsspy.GF.process import process_stream

import matplotlib.dates as mdates
from scipy import integrate, fft

# Get file name
specfemmagic = '/scratch/gpfs/lsawade/SpecfemMagicTurkey'
cmtfile = os.path.join(specfemmagic, 'DATA_default', 'CMTSOLUTION')
stationdir = os.path.join(specfemmagic, 'DB_test', 'II', 'KIV')
h5file = os.path.join(stationdir,  'II.KIV.h5')
stationdir = os.path.join(specfemmagic, 'DB_test', 'GE', 'MSBI')
h5file = os.path.join(stationdir,  'GE.MSBI.h5')

# %% Get CMTSOLUTION
cmt = CMTSOLUTION.read(cmtfile)

# %% Define Processing function

# %% Get Reciprocal seismograms
rp = get_seismograms(h5file, cmt)

# %% Get forward seismograms

fw = read(os.path.join(specfemmagic,
                       'specfem3d_globe_forward', 'OUTPUT_FILES', 'GE.MSBI*.sac'))

# %%
bp = [20, 50]
duration = 1800.0
rp = process_stream(rp, bandpass=bp)
fw = process_stream(fw, bandpass=bp)

# %% Set up time limits
starttime = fw[0].stats.starttime + 0
endtime = starttime + 1800
limits = (starttime.datetime, endtime.datetime)

# %% Plot comparison figure

plt.rcParams["font.family"] = "monospace"
plt.close('all')
fig = plt.figure(figsize=(7, 4))
lw = 0.25

for _i, comp in enumerate(['N', 'E', 'Z']):

    forward = fw.select(component=comp)[0]
    recipro = rp.select(component=comp)[0]

    trn = recipro.copy()
    trn.trim(starttime=starttime, endtime=endtime)
    absmax = np.max(np.abs(recipro.data))

    absmax_off = 0.0  # 0.1*absmax
    ax = plt.subplot(3, 1, _i+1)
    plt.plot(forward.times("matplotlib"), forward.data+absmax_off,
             'k-', lw=lw, label='Standard Specfem')
    plt.plot(recipro.times("matplotlib"), recipro.data-absmax_off,
             'r--', lw=lw, label='Reciprocal GF DB')
    # plt.plot(forward.times("matplotlib")[
    #          ::21], forward.data[::21], 'r--', lw=lw, label='Fw. subsampled')

    # Ylabel
    # plt.ylabel(f'{comp}  ')
    plot_label(ax, f'{comp}', dist=0.025, location=13,
               fontsize='medium', box=False)

    # Axis limits and indicator
    ax.set_ylim(-1.2*absmax, 1.2*absmax)
    plot_label(
        ax, f'A: {absmax:.5g} m', dist=0,
        fontsize='xx-small', box=False)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.tick_params(labelleft=False, left=False)
    ax.spines.right.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_xlim(limits)

    if _i == 2:
        plt.xlabel('Time')
    else:
        ax.spines.bottom.set_visible(False)
        ax.tick_params(bottom=False)

    if _i == 0:
        # Add title with event info
        network = rp[0].stats.network
        station = rp[0].stats.station
        plot_label(ax,
                   f"{cmt.cmt_time.strftime('%Y-%m-%d %H:%M:%S')}  Loc: {cmt.latitude}dg, {cmt.longitude}dg, {cmt.depth}km\n"
                   f"{network}.{station} Loc: {rp[0].stats.latitude}dg, {rp[0].stats.longitude}dg -- BP: [{bp[0]:.1f}s, {bp[1]:.1f}s]",
                   fontsize='medium', box=False, location=6)

        # Add legend
        plt.legend(frameon=False, loc='upper right',
                   ncol=3, fontsize='x-small')


fig.autofmt_xdate()

# Removes datestamp from N and E axes
plt.subplots_adjust(
    left=0.1, right=0.9, bottom=0.2, top=0.85, hspace=0.0)

plt.savefig('single_seismogram.pdf', dpi=300)


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

# %%
process_stream(st)
# %%

starttime = st[0].stats.starttime + 700
endtime = starttime + 9000
limits = (starttime.datetime, endtime.datetime)

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

plot_label(ax, f'dLat', dist=0.05, location=13,
           fontsize='medium', box=False)

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
           f"{cmt.cmt_time.ctime()}    Loc: {cmt.latitude}dg, {cmt.longitude}dg, {cmt.depth}km\n"
           f"{network}.{station} -- BP: [40s, 300s]",
           fontsize='medium', box=False, location=6)
plt.xlabel('Time')
fig.autofmt_xdate()

plt.savefig('eventsection.pdf', dpi=300)

#  %%  Get Fréchet derivatives

fr = get


# %%

# %%


# %%


rp = get_seismograms(h5file, cmt)
rpsub = get_seismograms_sub(h5file, cmt)

fw = read(os.path.join(specfemmagic,
                       'specfem3d_globe_forward', 'OUTPUT_FILES', '*.sac'))

for tr in fw:
    tr.stats.starttime -= 60


starttime = fw[0].stats.starttime + 300
endtime = starttime + 500
limits = (starttime.datetime, endtime.datetime)


# This test proves that spatial subsampling is indeed possible, at the cost of
# period.

def process_trace(tr, period=40.0):
    tr.taper(0.25, type='cosine', side='both')
    fdict = dict(freq=1.0/period, zerophase=True)
    tr.filter("lowpass", **fdict)


plt.close('all')
fig = plt.figure(figsize=(10, 6))
lw = 0.25
for _i, comp in enumerate(['N', 'E', 'Z']):

    forward = fw.select(component=comp)[0]
    recipro = rp.select(component=comp)[0]
    reciprosub = rpsub.select(component=comp)[0]

    period = 20.0
    process_trace(forward, period=period)
    process_trace(recipro, period=period)
    process_trace(reciprosub, period=period)

    ax = plt.subplot(3, 1, _i+1)
    plt.plot(forward.times("matplotlib")[
             ::28], forward.data[::32], 'r--', lw=lw, label='Fw. subsampled')
    plt.plot(recipro.times("matplotlib"), recipro.data,
             'k', lw=lw, label='125 Nodes')
    plt.plot(reciprosub.times("matplotlib"), reciprosub.data,
             'b', lw=lw, label='27 Nodes')
    # plt.plot(forward.times("matplotlib"), forward.data,
    #          'r-', lw=lw, label='Forward')
    plt.ylabel(f'{comp}  ', rotation=0)

    trn = recipro.copy()
    trn.trim(starttime=starttime, endtime=endtime)
    absmax = np.max(np.abs(trn.data))
    ax.set_ylim(-1.375*absmax, 1.375*absmax)
    plot_label(
        ax, f'max|u|: {absmax:.5g} m',
        fontsize='x-small', box=False)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.tick_params(labelleft=False, left=False)
    ax.spines.right.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.top.set_visible(False)

    ax.set_xlim(limits)

    if _i == 2:
        plt.xlabel('Time')

    if _i == 0:
        # Add title with event info
        network = rp[0].stats.network
        station = rp[0].stats.station
        plt.title(
            f"LP {period:.0f} s -- {network}.{station} -- {cmt.cmt_time} Loc: {cmt.latitude}dg, {cmt.longitude}dg, {cmt.depth}km")

        # Add legend
        plt.legend(frameon=False, loc='lower right',
                   ncol=3, fontsize='x-small')

fig.autofmt_xdate()
plt.subplots_adjust(
    left=0.05, right=0.95, bottom=0.1, top=0.95)
plt.savefig(f'proof_spatial_subsampling.pdf', dpi=300)


# %%

rp = get_seismograms(h5file, cmt)
rpsub = get_seismograms_sub(h5file, cmt)

fw = read(os.path.join(specfemmagic,
                       'specfem3d_globe_forward', 'OUTPUT_FILES', '*.sac'))
for tr in fw:
    tr.stats.starttime -= 60

plt.close('all')
fig = plt.figure(figsize=(10, 6))
lw = 0.25
for _i, comp in enumerate(['N', 'E', 'Z']):

    forward = fw.select(component=comp)[0]
    recipro = rp.select(component=comp)[0]
    reciprosub = rpsub.select(component=comp)[0]
    period = 55
    process_trace(forward, period=period)
    process_trace(recipro, period=period)
    process_trace(reciprosub, period=period)

    ax = plt.subplot(3, 1, _i+1)
    plt.plot(forward.times("matplotlib")[
             ::28], forward.data[::28], 'r--', lw=lw, label='Fw. subsampled')
    plt.plot(recipro.times("matplotlib"), recipro.data,
             'k', lw=lw, label='125 Nodes')
    plt.plot(reciprosub.times("matplotlib"), reciprosub.data,
             'b', lw=lw, label='27 Nodes')
    # plt.plot(forward.times("matplotlib"), forward.data,
    #          'r-', lw=lw, label='Forward')
    plt.ylabel(f'{comp}  ', rotation=0)

    trn = recipro.copy()
    trn.trim(starttime=starttime, endtime=endtime)
    absmax = np.max(np.abs(trn.data))
    ax.set_ylim(-1.375*absmax, 1.375*absmax)
    plot_label(
        ax, f'max|u|: {absmax:.5g} m',
        fontsize='x-small', box=False)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.tick_params(labelleft=False, left=False)
    ax.spines.right.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.top.set_visible(False)

    ax.set_xlim(limits)

    if _i == 2:
        plt.xlabel('Time')

    if _i == 0:
        # Add title with event info
        network = rp[0].stats.network
        station = rp[0].stats.station
        plt.title(
            f"LP {period:.0f} s -- {network}.{station} -- {cmt.cmt_time} Loc: {cmt.latitude}dg, {cmt.longitude}dg, {cmt.depth}km")

        # Add legend
        plt.legend(frameon=False, loc='lower right',
                   ncol=3, fontsize='x-small')

fig.autofmt_xdate()
plt.subplots_adjust(
    left=0.05, right=0.95, bottom=0.1, top=0.95)
plt.savefig(f'proof_spatial_subsampling_low_pass_{period:.0f}.pdf', dpi=300)
