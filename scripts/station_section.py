# %% Imports
# External
from glob import glob
from copy import deepcopy
import matplotlib.axes
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import numpy as np
from obspy import read, Stream, Inventory
from obspy.geodetics.base import locations2degrees
import os
import typing as tp

# Internal
from lwsspy.GF.plot.util import plot_label
from lwsspy.GF.source import CMTSOLUTION
from lwsspy.GF.seismograms import SGTManager
from lwsspy.GF.process import process_stream, select_pairs
from lwsspy.GF.plot.section import plotsection
from lwsspy.seismo.download_data import download_data


# %% Files

# DB files
specfemmagic = '/scratch/gpfs/lsawade/SpecfemMagicGF'
h5files = os.path.join(specfemmagic, 'DB', '*', '*', '*.h5')

# CMTSOLUTION
cmt = CMTSOLUTION.read('/home/lsawade/lwsspy/lwsspy.GF/scripts/DATA/CHILE_CMT')

# %% Initialize the GF manager
sgt = SGTManager(glob(h5files)[:])
sgt.load_header_variables()
sgt.get_elements(cmt.latitude, cmt.longitude, cmt.depth, 30)

# %% Download data

raw, inv = download_data(
    cmt.origin_time,
    duration=4*3600,
    network='II,IU',
    station=','.join(sgt.stations),
    location='00',
    channel='LH*',
    starttimeoffset=-300,
    endtimeoffset=300
)

# %% Get a bunch of seismograms

# cmt = CMTSOLUTION.read('/scratch/gpfs/lsawade/SpecfemMagicGF/CMTSOLUTION')
rp = sgt.get_seismograms(cmt)


# %% Process

obs = process_stream(raw, inv=inv, cmt=cmt, duration=4*3600)
syn = process_stream(rp, cmt=cmt, duration=4*3600)

# %%

pobs, psyn = select_pairs(obs, syn)

# %% Plot section

starttime = psyn[0].stats.starttime + 0
endtime = starttime + 4*3600
limits = (starttime.datetime, endtime.datetime)

# Plots a section of observed and synthetic
plotsection(pobs, psyn, cmt, comp='Z', lw=0.75, limits=limits)
plt.savefig('testsection_data.pdf', dpi=300)
