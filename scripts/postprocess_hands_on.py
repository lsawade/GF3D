# %%
# LOAD THE KDTREE FIRST!!!!!!!!!!!!!
# If other packages are loaded, the wrong libstdc++ is picked up and
# doesn't contain the right GLIBCXX version

import os
from glob import glob
from copy import deepcopy
from obspy import read, Stream
from obspy.geodetics.base import locations2degrees
import numpy as np
import matplotlib.axes
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from lwsspy.plot import plot_label
from lwsspy.GF.source import CMTSOLUTION
from lwsspy.GF.postprocess import Adios2HDF5
from lwsspy.GF.seismograms import \
    get_seismograms, get_seismograms_sub, get_frechet, SGTManager
from lwsspy.GF.simulation import Simulation
from lwsspy.GF.stf import create_stf


# import matplotlib.dates as mdates
# from scipy import integrate, fft
# %%
# Only import the KDTree after setting the LD_LIBRARY PATH, e.g.
# $ export LD_LIBRARY_PATH='/home/lsawade/.conda/envs/gf/lib'

# %% Get CMT solution to convert
cmt = CMTSOLUTION.read('/scratch/gpfs/lsawade/SpecfemMagicGF/CMTSOLUTION')

# Get file name
specfemmagic = '/scratch/gpfs/lsawade/SpecfemMagicGF'
config_file = os.path.join(specfemmagic, 'reci.toml')
db = '/scratch/gpfs/lsawade/SpecfemMagicGF/DB'
stationdir = os.path.join(db, 'II', 'BFO')
compdict = dict()
for _i, comp in enumerate(['N', 'E', 'Z']):
    compdict[comp] = os.path.join(
        stationdir, comp, 'specfem', 'OUTPUT_FILES', 'save_forward_arrays_GF.bp')

# %% Test seismograms at the source location
st = read(os.path.join(os.path.dirname(compdict['E']), 'EQ*.sac'))

st.plot(outfile='seistest1.png', dpi=300)


st = read(os.path.join(
    specfemmagic, 'specfem3d_globe_forward', 'OUTPUT_FILES', 'II*.sac'))

st.plot(outfile='seistest2.png', dpi=300)


# %%
# Write H5py Database file
# h5file = '/scratch/gpfs/lsawade/testdb_compression.h5'
print('hello')
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

    # A2H.write()
    A2H.write()

# %%
# h5file = f'/scratch/gpfs/lsawade/permanentnew.h5'
# with h5py.File(h5file, 'r') as db:
#     ibool = db['ibool'][:]
#     xyz = db['xyz'][:]

#     # save_coordinates = np.savez('coords.npz', xyz=xyz, ibool=ibool)
#     del ibool, xyz

# h5file = f'/scratch/gpfs/lsawade/SA_subset/II/BFO.h5'

# %%

h5files = f'/scratch/gpfs/lsawade/SpecfemMagicGF/DB/*/*/*.h5'
cmt = CMTSOLUTION.read('/scratch/gpfs/lsawade/SpecfemMagicGF/CMTSOLUTION')


sgt = SGTManager(glob(h5files)[:])

sgt.load_header_variables()

sgt.get_elements(cmt.latitude, cmt.longitude, cmt.depth, 30)


# %% Get a bunch of seismograms

cmt = CMTSOLUTION.read('/scratch/gpfs/lsawade/SpecfemMagicGF/CMTSOLUTION')
rp = sgt.get_seismograms(cmt)

# %% Process seismograms


def process_stream(st: Stream):
    st.filter('bandpass', freqmin=1/300.0, freqmax=1/40.0, zerophase=True)


process_stream(rp)

# %%
plt.rcParams["font.family"] = "monospace"


def plotsection(st: Stream, cmt: CMTSOLUTION, *args, ax: matplotlib.axes.Axes | None = None, comp='Z', **kwargs):

    # Axes to plot in
    if ax is None:
        ax = plt.gca()

    # Get a single component
    pst = st.select(component=comp)

    # Get station event distances, labels
    for tr in pst:
        tr.stats.distance = locations2degrees(
            tr.stats.latitude, tr.stats.longitude,
            cmt.latitude, cmt.longitude)

    # Sort the stream
    pst.sort(keys=['distance', 'network', 'station'])

    # Get scaling
    absmax = np.max(pst.max())
    plot_label(ax, f'max|u|: {absmax:.5g} m',
               fontsize='small', box=False, dist=0.0, location=4)

    # Number of stations
    y = np.arange(1, len(pst)+1)

    # Set ylabels
    # Set text labels and properties.
    # , rotation=20)
    ax.set_yticks(y, [f"{tr.stats.network}.{tr.stats.station}" for tr in pst])

    # TO have epicentral distances on the right
    ax2 = ax.secondary_yaxis("right")
    ax2.set_yticks(y, [f"{tr.stats.distance:>6.2f}" for tr in pst])
    ax2.spines.right.set_visible(False)
    ax2.tick_params(left=False, right=False)

    # Normalize
    for _tr, _y in zip(pst, y):
        plt.plot(_tr.times('matplotlib'), _tr.data /
                 absmax + _y, *args, **kwargs)


starttime = rp[0].stats.starttime + 0
endtime = starttime + 10800
limits = (starttime.datetime, endtime.datetime)


plt.figure(figsize=(9, 6))

ax = plt.axes()
plotsection(rp, cmt, 'k', comp='Z', lw=0.75)

# Remove all spines
ax.spines.top.set_visible(False)
ax.spines.left.set_visible(False)
ax.spines.right.set_visible(False)

ax.tick_params(left=False, right=False)

# Cover timestamp


ax.xaxis_date()
ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.set_xlim(limits)

plt.title(f"{cmt.cmt_time.ctime()} Loc: {cmt.latitude}dg, {cmt.longitude}dg, {cmt.depth}km - BP: [40s, 300s]",
          fontsize='medium')

plt.xlabel('Time')
plt.subplots_adjust(left=0.1, right=0.9, top=0.95)
plt.savefig('testsection.pdf', dpi=300)


# %% Get CMT solution to convert
h5file = f'/scratch/gpfs/lsawade/SpecfemMagicGF/DB/II/BFO/II.BFO.h5'
# h5file = f'/scratch/gpfs/lsawade/SpecfemMagicGF/h5test2.h5'
cmt = CMTSOLUTION.read('/scratch/gpfs/lsawade/SpecfemMagicGF/CMTSOLUTION')


# %% Get reciprocal synthetics
# for pert in np.arange(-1.0, 1.0):
rp = sgt.get_seismogram(cmt)

# %%


rp = get_seismograms(h5file, cmt)
rp_sub = get_seismograms_sub(h5file, cmt)

#
# %% Get forward synthetics
fw = read(os.path.join(specfemmagic,
                       'specfem3d_globe_forward', 'OUTPUT_FILES', 'II*.sac'))


# %%  Process tracess


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

plt.savefig('proof.pdf', dpi=300)


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


# %%
# Everything below here are function that used the storage of both the
# Displacement and the strain. The strain is no longer stored as part of the
# Green function database files and this will therefore not work anymore!

rp, rp2, sepsilon, epsilon_disp = get_seismograms(h5file, cmt)

plt.rcParams["font.family"] = "monospace"
plt.close('all')
plt.figure(figsize=(10, 6))
lw = 0.25
lab = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']

for _i in range(6):
    ax = plt.subplot(6, 1, _i+1)
    plt.plot(sepsilon[0, _i, :], 'k', lw=lw, label='Interpolated')
    plt.plot(epsilon_disp[0, _i, :], 'r--', lw=lw,
             label='Computed from Displacement')

    plot_label(ax, f'{lab[_i]}', dist=0.025, location=13,
               fontsize='medium', box=False)

    ax.tick_params(labelleft=False, left=False)
    ax.spines.right.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_xlim(100, 900)
    if _i == 0:
        plt.title('Strain components')
        # Add legend
        plt.legend(frameon=False, loc='upper left',
                   ncol=3, fontsize='x-small')

    if _i == 5:
        plt.xlabel('Samples')
    else:
        ax.spines.bottom.set_visible(False)
        ax.tick_params(bottom=False, labelbottom=False)


plt.subplots_adjust(
    left=0.075, right=0.925, bottom=0.1, top=0.9, hspace=0.0)


plt.savefig('teststrain.pdf')


# %%
fw = read(os.path.join(specfemmagic,
                       'specfem3d_globe_forward', 'OUTPUT_FILES', 'II*.sac'))

# %%

starttime = rp[0].stats.starttime + 0
endtime = starttime + 3600
limits = (starttime.datetime, endtime.datetime)


plt.rcParams["font.family"] = "monospace"
plt.close('all')
fig = plt.figure(figsize=(7, 4))
lw = 1

for _i, comp in enumerate(['N', 'E', 'Z']):

    recipro = rp.select(component=comp)[0]
    recipro2 = rp2.select(component=comp)[0]
    forward = fw.select(component=comp)[0]

    absmax = np.max(np.abs(recipro.data))

    absmax_off = 0.1*absmax

    ax = plt.subplot(3, 1, _i+1)

    plt.plot(forward.times("matplotlib"), forward.data,
             'lightgray', lw=lw, label='Standard Specfem')
    plt.plot(recipro.times("matplotlib"), recipro.data,
             'k-.', lw=lw, label='From Strain')
    plt.plot(recipro2.times("matplotlib"), recipro2.data,
             'r--', lw=lw, label='From Displacement')

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
                   f"{cmt.cmt_time.ctime()} Loc: {cmt.latitude}dg, {cmt.longitude}dg, {cmt.depth}km\n"
                   f"{network}.{station} -- BP: [40s, 300s]",
                   fontsize='medium', box=False, location=6)

        # Add legend
        plt.legend(frameon=False, loc='upper right',
                   ncol=3, fontsize='x-small')


fig.autofmt_xdate()

# Removes datestamp from N and E axes
plt.subplots_adjust(
    left=0.1, right=0.9, bottom=0.2, top=0.85, hspace=0.0)

plt.savefig('strain2displacement.pdf', dpi=300)
