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
elementdir = '/home/lsawade/lwsspy/lwsspy.GF/scripts/DATA/single_element_read/'
subsetfilename = os.path.join(elementdir, 'single_element.h5')
tracedir = os.path.join(elementdir, "traces")
stationxml = os.path.join(elementdir, "station.xml")

h5files = os.path.join(specfemmagic, 'DB', '*', '*', '*.h5')

# CMTSOLUTION
cmt = CMTSOLUTION.read('/home/lsawade/lwsspy/lwsspy.GF/scripts/DATA/CHILE_CMT')

# %% Initialize the GF manager
sgt = SGTManager(glob(h5files)[:])
sgt.load_header_variables()
sgt.get_elements(cmt.latitude, cmt.longitude, cmt.depth, 1, NGLL=3)

# %% Write a subset
sgt.write_subset(subsetfilename, duration=3600.0)

# %% load a subset

sgtsub = SGTManager(subsetfilename)
sgtsub.load()


# %% Download data

raw, inv = download_data(
    cmt.origin_time,
    duration=3600,
    network='II,IU',
    station=','.join(sgtsub.stations),
    location='00',
    channel='LH*',
    starttimeoffset=-300,
    endtimeoffset=300
)

# Write traces to directory
for tr in raw:
    tr.write(os.path.join(tracedir, tr.id + ".sac"), format="SAC")

# Write metadata
inv.write(stationxml, format='STATIONXML')

# %% Get a bunch of seismograms

rp = sgtsub.get_seismograms(cmt)

# %% Process

obs = process_stream(raw, inv=inv, cmt=cmt, duration=3600)
syn = process_stream(rp, cmt=cmt, duration=3600)

# %%

pobs, psyn = select_pairs(obs, syn)

# %% Plot section

starttime = psyn[0].stats.starttime + 0
endtime = starttime + 3600
limits = (starttime.datetime, endtime.datetime)

# Plots a section of observed and synthetic
plotsection(pobs, psyn, cmt, comp='Z', lw=0.75, limits=limits)
plt.savefig('subset_IO_section.pdf', dpi=300)
