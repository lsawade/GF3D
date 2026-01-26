# %%
# convolution testing
import json
import sys
import datetime
import os
import glob
from copy import deepcopy
import obspy
import numpy as np
import matplotlib.pyplot as plt
import obsplotlib.plot as opl
from copy import deepcopy   
import shutil
import obstflib as osl
import obsplotlib.plot as opl
from gf3d.source import CMTSOLUTION
import obsnumpy as onp
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.integrate import cumtrapz

# %%


cmtfile = """ PDEW2018  1 23  9 31 40.90  56.0000 -149.1700  14.1 0.0 7.9 GULF OF ALASKA
event name:     201801230931A
time shift:     23.0700
half duration:  22.3000
latitude:       56.2200
longitude:    -149.1200
depth:          33.6100
Mrr:       2.360000e+27
Mtt:      -4.850000e+27
Mpp:       2.500000e+27
Mrt:       1.940000e+27
Mrp:      -3.620000e+27
Mtp:       7.910000e+27
"""

cmt = CMTSOLUTION.read(cmtfile)

# %%
# Get the event record at II BFO for the Alaska event using the GFManager
cmt3 = CMTSOLUTION.read(cmtfile)

# Store one copy for the STF
cmt3_stf = deepcopy(cmt3)

cmt3.time_shift = 0.0


# %%
osl.utils.log("Loading Green function subset")

# Read the Green functions
from gf3d.seismograms import GFManager
gfm = GFManager("/lustre/orion/geo111/scratch/lsawade/alaska_subset.h5")
gfm.load()

# %%
# Get Seismograms
def get_ds(gfm: GFManager, cmt: CMTSOLUTION, raw=True):
    array, meta = gfm.get_seismograms(cmt=cmt, raw=raw, array=True)
    new_meta = onp.Meta.from_dict(meta)
    return onp.Dataset(data=array, meta=new_meta)


rp_ds_base = get_ds(gfm, cmt3, raw=True)

#%%
# Keep the geometry the same for plotting
rp_ds_base.compute_geometry(cmt3.latitude, cmt3.longitude)

# %%
# Download the corresponding data
osl.utils.log("Getting and/or preprocessing the data")

# Base parameters
tshift = 200.0
starttime = cmt3.origin_time - tshift
sps = 2
length_in_s = 3600 + tshift

# Preprocess variables
pre_filt = (0.005, 0.006, 1, 2)

# Define sampling rate as a function of the pre_filt
sampling_rate = pre_filt[3] * 2.5

# Response output
rr_output = "DISP"
water_level = 60

# overwrite data if directory exists by downloading new data
overwrite = False

# %%
stationdict = dict()
st = obspy.read("DATA/alaska_seismograms/*.sac")
for tr in st:
    net, sta = tr.stats.network, tr.stats.station
    if net not in stationdict:
        stationdict[net] = dict()
    stationdict[net][sta] = dict()
    stationdict[net][sta]["latitude"] = tr.stats.sac.stla
    stationdict[net][sta]["longitude"] = tr.stats.sac.stlo

sfs = onp.Dataset.from_stream(st, 
                               components=["N", "E", "Z"],
                               event_latitude=cmt3.latitude,
                               event_longitude=cmt3.longitude,
                               station_dict=stationdict)



sfs.compute_geometry(cmt3.latitude, cmt3.longitude)


# %%
def process_ds(ds: onp.Dataset, starttime, length_in_s, sps, step=False):

    # Taper the seismograms
    ds.taper(max_percentage=0.05, type='cosine')

    # Bandpass the seismograms
    ds.filter('bandpass', freqmin=1/300.0, freqmax=1/40.0, corners=4, zerophase=True)

    # Resample the seismograms
    ds.interpolate(sampling_rate=sps, starttime=starttime, npts=int((length_in_s) * sps),
                   method='weighted_average_slopes')

    # Rotate the seismograms
    ds.rotate('->RT')

    # Taper the seismograms
    ds.taper(max_percentage=0.05, type='cosine')

    # Convolve the seismograms with a step function
    if step:

        # Vector
        t = np.arange(0, length_in_s, 1/sps)

        # Create Error function
        error = osl.STF.error(origin=obspy.UTCDateTime(0), t=t, tshift=150.0, hdur=1e-6, tc=0.0)

        # Convolve the seismograms
        ds.convolve(error.f, 150.0)

        # Taper the seismograms
        ds.taper(max_percentage=0.05, type='cosine')


# %%
osl.utils.log("Processing data")

sf_ds = sfs.copy()
rp_ds = rp_ds_base.copy()

tshift = 200.0
starttime = cmt3.origin_time - tshift
sps = 10
length_in_s = 11800 + tshift

osl.utils.log("Processing specfem traces")
process_ds(sf_ds, starttime, length_in_s, sps, step=False)
# sf_ds.data = np.gradient(sf_ds.data, sf_ds.meta.delta, axis=-1)

osl.utils.log("Processing reciprocal synthetics")
process_ds(rp_ds, starttime, length_in_s, sps, step=True)


# %% 
# Convolve with STF for the respective cmt solution
t = np.arange(0, rp_ds.meta.npts * rp_ds.meta.delta, rp_ds.meta.delta)

hdur_diff = np.sqrt((cmt3_stf.hdur / 1.628) ** 2 - (gfm.header["hdur"]) ** 2) * 1.628

cstf = osl.STF.gaussian(
    origin=cmt3_stf.origin_time, t=t, tc=cmt3_stf.time_shift,
    tshift=tshift, hdur=hdur_diff, M0=cmt3.M0)

plt.figure(figsize=(8,2.5))
cstf.plot(normalize=1e7, label='CMT3D+')
plt.xlim(0, 125)
plt.xlabel('Time since origin [s]')
plt.ylabel('N$\cdot$m')
plt.legend(frameon=False, fontsize='small')
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.25, top=0.9)
plt.savefig(os.path.join('teststf.pdf'), dpi=300)

# %%
# Convolve the CMT3D with the STF but not the Green functions

ds_rp, ds_sf = rp_ds.intersection(sf_ds)
ds_rp.convolve(cstf.f/cmt3.M0, tshift)

# %%
# Get all stations where the epicentral distance is large than 30 dg
_component = "Z"

fig = plt.figure(figsize=(8, 4.5))
ax, twin_ax = osl.plot.plot_check_section([ds_sf, ds_rp], component=_component, scale=20.0,
                        labels=['Forward', 'Reciprocal'], mindist=30, 
                        lw=[1.25, 0.5], ls=['-', '-'], colors=['k', 'w'],
                        plot_misfit=True, plot_right_annotations=False,
                        step_idx=4, tickfontsize='x-small')

lim = (-2, 30.5)
ax.set_ylim(*lim)
twin_ax.set_ylim(*lim)
plt.subplots_adjust(left=0.1, right=0.9, top=0.975, bottom=0.1)
fig.savefig(os.path.join('./', f'A{_component}_sf_rp.pdf'))
plt.close(fig)