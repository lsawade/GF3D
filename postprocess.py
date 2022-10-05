# %%
# LOAD THE KDTREE FIRST!!!!!!!!!!!!!
# If other packages are loaded, the wrong libstdc++ is picked up and
# doesn't contain the right GLIBCXX version

from lwsspy.GF.seismograms import SGTManager
from obspy import read, Stream
from copy import deepcopy
import datetime
import logging
import os
import h5py
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from lwsspy.plot import plot_label
from lwsspy.GF.source import CMTSOLUTION
from lwsspy.GF.postprocess import Adios2HDF5
from lwsspy.GF.seismograms import get_seismograms
import matplotlib.dates as mdates

# Only import the KDTree after setting the LD_LIBRARY PATH, e.g.
# $ export LD_LIBRARY_PATH='/home/lsawade/.conda/envs/gf/lib'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
h5file = '/scratch/gpfs/lsawade/testdb.h5'

# %%
with Adios2HDF5(
        h5file, compdict['N'], compdict['E'], compdict['Z'],
        config_file) as A2H:

    A2H.write()

# %%
with h5py.File(h5file, 'r') as db:
    ibool = db['ibool'][:]
    xyz = db['xyz'][:]

    # save_coordinates = np.savez('coords.npz', xyz=xyz, ibool=ibool)
    del ibool, xyz


# %% Get CMT solution to convert
cmt = CMTSOLUTION.read('CMTSOLUTION')

# %% Get seismograms
rp = get_seismograms(h5file, cmt)

# %%
fw = read(os.path.join(specfemmagic,
          'specfem3d_globe_forward', 'OUTPUT_FILES', '*.sac'))
for tr in fw:
    tr.stats.starttime -= 60


starttime = fw[0].stats.starttime + 300
endtime = starttime + 500
limits = (starttime.datetime, endtime.datetime)


# def process_stream(st: Stream):
#     st.filter('lowpass', freq=1/35.0,zerophase=True)


# process_stream(rp)
# process_stream(fw)

# %%

plt.close('all')
fig = plt.figure(figsize=(10, 6))
lw = 1
for _i, comp in enumerate(['N', 'E', 'Z']):

    forward = fw.select(component=comp)[0]
    recipro = rp.select(component=comp)[0]

    ax = plt.subplot(3, 1, _i+1)
    plt.plot(recipro.times("matplotlib"), recipro.data,
             'k', lw=lw, label='Reciprocal')
    plt.plot(forward.times("matplotlib"), forward.data,
             'r-', lw=lw, label='Forward')
    plt.plot(forward.times("matplotlib")[
             ::28], forward.data[::28], 'r--', lw=lw, label='Fw. subsampled')
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

    # ax.set_xlim(limits)

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


# %%

# %% Get CMT solution to convert
cmt = CMTSOLUTION.read('CMTSOLUTION')


lst = []
depth_range = list(range(-10, 11, 2))
for dd in depth_range:
    tcmt = deepcopy(cmt)
    tcmt.depth = tcmt.depth + dd

    lst.append(get_seismograms(h5file, tcmt))

# %%
starttime = lst[0][0].stats.starttime + 250
endtime = starttime + 500
limits = (starttime.datetime, endtime.datetime)

# %%
plt.close('all')
fig = plt.figure(figsize=(10, 3))
ax = plt.axes()
alpha = np.linspace(-0.95, 0.95, 11)

for _al, _reci, _dd in zip(alpha, lst, list(depth_range)):

    recipro = _reci.select(component='Z')[0]
    plt.plot(recipro.times("matplotlib"), recipro.data,
             c=(0.1, 0.1, 0.8), lw=0.25, label=f'{_dd:>3d} km', alpha=1-np.abs(_al))

recipro = lst[5].select(component='Z')[0]
plt.plot(recipro.times("matplotlib"), recipro.data,
         'k', lw=1.0, alpha=1)
plt.ylabel(f'Z  ', rotation=0)
absmax = np.max(np.abs(recipro.data))
ax.set_ylim(-1.2*absmax, 1.2*absmax)
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

plt.xlabel('Time')

# Add title with event info
network = _reci[0].stats.network
station = _reci[0].stats.station
plt.title(
    f"{network}.{station} -- {cmt.cmt_time} Loc: {cmt.latitude}dg, {cmt.longitude}dg, {cmt.depth}km")

# Add legend
plt.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))

fig.autofmt_xdate()
plt.subplots_adjust(
    left=0.125, right=0.875, bottom=0.2, top=0.875)
plt.savefig('testz.pdf', dpi=300)


# %%


# Write H5py Database file
h5file = '/scratch/gpfs/lsawade/testdb.h5'

dbfiles = [h5file]

SGTM = SGTManager(dbfiles)

# Get base
cmt = CMTSOLUTION.read('CMTSOLUTION')
lat, lon, dep = cmt.latitude, cmt.longitude, cmt.depth

SGTM.load_header_variables()

# %%

SGTM.header['res_topo']
SGTM.header['topography']
SGTM.header['ellipticity']

# %%
SGTM.get_elements(lat, lon, dep, k=10)
