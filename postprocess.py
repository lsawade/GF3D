# %%
# LOAD THE KDTREE FIRST!!!!!!!!!!!!!
# If other packages are loaded, the wrong libstdc++ is picked up and
# doesn't contain the right GLIBCXX version

import time
from lwsspy.GF.seismograms import SGTManager
from obspy import read, Stream
import os
import numpy as np
import matplotlib.pyplot as plt
from lwsspy.plot import plot_label
from lwsspy.GF.source import CMTSOLUTION
from lwsspy.GF.postprocess import Adios2HDF5
from lwsspy.GF.seismograms import get_seismograms, get_seismograms_sub
import matplotlib.dates as mdates

# %%
# Only import the KDTree after setting the LD_LIBRARY PATH, e.g.
# $ export LD_LIBRARY_PATH='/home/lsawade/.conda/envs/gf/lib'

# %% Get CMT solution to convert
cmt = CMTSOLUTION.read('CMTSOLUTION')

# Get file name
specfemmagic = '/scratch/gpfs/lsawade/SpecfemMagicGF'
config_file = os.path.join(specfemmagic, 'reci.toml')

compdict = dict()
for _i, comp in enumerate(['N', 'E', 'Z']):
    compdict[comp] = os.path.join(
        specfemmagic, 'specfem3d_globe', f'run000{_i+1}', 'OUTPUT_FILES', 'save_forward_arrays_GF.bp')


# %%
# Write H5py Database file
h5file = '/scratch/gpfs/lsawade/testdb_compression.h5'

# %%

# compressors = [None, 'lzf', 'gzip']
# compressors = ['gzip']
# compressor_opts = dict()
# compressor_opts[None] = [None]
# compressor_opts['gzip'] = [5, 9]
# compressor_opts['lzf'] = [None]


# for _compressor in compressors:
#     for _opt in compressor_opts[_compressor]:

# h5file = f'/scratch/gpfs/lsawade/testdb_compression_{_compressor}'
# if _compressor == 'gzip':
#     h5file += f'{_opt:d}'
# h5file += '.h5'

h5file = f'/scratch/gpfs/lsawade/testdb.h5'
with Adios2HDF5(
        h5file, compdict['N'], compdict['E'], compdict['Z'],
        config_file, subspace=False,
        precision='half',
        compression='lzf',
        compression_opts=None) as A2H:

    A2H.write()

 # %%
# h5file = f'/scratch/gpfs/lsawade/permanentnew.h5'
# with h5py.File(h5file, 'r') as db:
#     ibool = db['ibool'][:]
#     xyz = db['xyz'][:]

#     # save_coordinates = np.savez('coords.npz', xyz=xyz, ibool=ibool)
#     del ibool, xyz

# h5file = f'/scratch/gpfs/lsawade/SA_subset/II/BFO.h5'

# %% Get CMT solution to convert
h5file = f'/scratch/gpfs/lsawade/testdb.h5'
cmt = CMTSOLUTION.read('CMTSOLUTION')

# %% Get seismograms
rp = get_seismograms(h5file, cmt)

# %%

fw = read(os.path.join(specfemmagic,
          'specfem3d_globe_forward', 'OUTPUT_FILES', '*.sac'))

for tr in fw:
    tr.stats.starttime -= 120


starttime = fw[0].stats.starttime + 300
endtime = starttime + 7200
limits = (starttime.datetime, endtime.datetime)


def process_stream(st: Stream):
    st.filter('bandpass', freqmin=1/300.0, freqmax=1/40.0, zerophase=True)


process_stream(rp)
process_stream(fw)

# %%

plt.close('all')
fig = plt.figure(figsize=(10, 6))
lw = 0.25
for _i, comp in enumerate(['N', 'E', 'Z']):

    forward = fw.select(component=comp)[0]
    recipro = rp.select(component=comp)[0]

    ax = plt.subplot(3, 1, _i+1)
    plt.plot(recipro.times("matplotlib"), recipro.data,
             'k', lw=lw, label='Reciprocal')
    plt.plot(forward.times("matplotlib"), forward.data,
             'r-', lw=lw, label='Forward')
    plt.plot(forward.times("matplotlib")[
             ::32], forward.data[::32], 'r--', lw=lw, label='Fw. subsampled')
    plt.ylabel(f'{comp}  ', rotation=0)

    trn = recipro.copy()
    trn.trim(starttime=starttime, endtime=endtime)
    absmax = np.max(np.abs(recipro.data))
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
            f"{network}.{station} -- {cmt.cmt_time} Loc: {cmt.latitude}dg, {cmt.longitude}dg, {cmt.depth}km")

        # Add legend
        plt.legend(frameon=False, loc='lower right',
                   ncol=3, fontsize='x-small')

fig.autofmt_xdate()
plt.subplots_adjust(
    left=0.05, right=0.95, bottom=0.1, top=0.95)
plt.savefig('proof.pdf', dpi=300)


# %% Get seismograms


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
