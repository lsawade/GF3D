
import os
from obspy import read, Stream
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from lwsspy.GF.plot.util import plot_label
from lwsspy.GF.source import CMTSOLUTION
from lwsspy.GF.seismograms import get_seismograms, get_seismograms_sub

# %%

# Get file name
specfemmagic = '/scratch/gpfs/lsawade/SpecfemMagicGF'
db = os.path.join(specfemmagic, 'DB')
stationdir = os.path.join(db, 'II', 'BFO')

# Get CMT solution to convert
cmt = CMTSOLUTION.read(os.path.join(specfemmagic, 'CMTSOLUTION'))

# HDF5 File for reading
h5file = os.path.join(stationdir, 'II.BFO.h5')

# %%

rp = get_seismograms(h5file, cmt)
rp_sub = get_seismograms_sub(h5file, cmt)

# %% Get forward synthetics
fw = read(os.path.join(
    specfemmagic, 'specfem3d_globe_forward', 'OUTPUT_FILES', 'II*.sac'))


def process_stream(st: Stream):
    st.filter('bandpass', freqmin=1/300.0, freqmax=1/40.0, zerophase=True)


if True:
    process_stream(rp)
    process_stream(fw)
    process_stream(rp_sub)

# %%
starttime = fw[0].stats.starttime + 1800
endtime = starttime + 1800
limits = (starttime.datetime, endtime.datetime)

# %%
plt.rcParams["font.family"] = "monospace"
plt.close('all')
fig = plt.figure(figsize=(7, 5))
lw = 3

for _i, comp in enumerate(['N', 'E', 'Z']):

    forward = fw.select(component=comp)[0]
    recipro = rp.select(component=comp, network='II', station='BFO')[0]
    recipro_sub = rp_sub.select(component=comp)[0]

    trn = forward.copy()
    trn.trim(starttime=starttime, endtime=endtime)
    absmax = np.max(np.abs(forward.data))

    absmax_off = 0.1*absmax
    ax = plt.subplot(3, 1, _i+1)

    error = np.abs(recipro.data-recipro_sub.data)

    plt.plot(forward.times("matplotlib"), forward.data,  # +absmax_off,
             'k-', lw=lw, label='specfem3D_globe')
    plt.plot(recipro.times("matplotlib"), recipro.data,  # -absmax_off,
             'r-', lw=lw/3, label='GF-DB-125')
    plt.plot(recipro_sub.times("matplotlib"), recipro_sub.data,  # ,-absmax_off,
             'w-', lw=lw/9, label='GF-DB-27')
    plt.plot(recipro.times("matplotlib"), 10*error-1.19*absmax,  # -absmax_off,
             'k-', lw=lw/3, label='|Error|', alpha=0.5)

    # Ylabel
    # plt.ylabel(f'{comp}  ')
    plot_label(ax, f'{comp}', dist=0.025, location=13,
               fontsize='medium', box=False)

    # Axis limits and indicator
    ax.set_ylim(-1.2*absmax, 1.2*absmax)
    plot_label(
        ax, f'A: {absmax:.5g} m', dist=0,
        fontsize='xx-small', box=False)
    plot_label(
        ax, f'E: {np.max(error):.5g} m', dist=0,
        fontsize='xx-small', box=False, location=4)
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
        ax.spines['bottom'].set_position(('outward', 5))
    else:
        ax.spines.bottom.set_visible(False)
        ax.tick_params(bottom=False)
        # Cover timestamp
        plot_label(ax, f'           ', dist=0.02, location=10,
                   fontsize='medium', box={'facecolor': 'w', 'edgecolor': 'None'})

    if _i == 0:
        # Add title with event info
        network = rp[0].stats.network
        station = rp[0].stats.station
        plot_label(ax,
                   f"{cmt.cmt_time.ctime()} Loc: {cmt.latitude}dg, {cmt.longitude}dg, {cmt.depth}km\n"
                   f"{network}.{station} -- BP: [40s, 300s]",
                   fontsize='medium', box=False, location=6)

        # Add legend
        plt.legend(frameon=False, loc='upper right',
                   ncol=2, fontsize='x-small')

fig.autofmt_xdate()

# Removes datestamp from N and E axes
plt.subplots_adjust(
    left=0.1, right=0.9, bottom=0.2, top=0.85, hspace=0.2)

plt.savefig('element_subsampling.pdf', dpi=300)
