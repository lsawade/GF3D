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
from gf3d.plot.util import plot_label, reset_mpl
from gf3d.source import CMTSOLUTION
from gf3d.seismograms import GFManager
from gf3d.process import process_stream, select_pairs
from gf3d.plot.section import plotsection
from gf3d.download import download_stream


# %% Files

# DB files
specfemmagic = '/scratch/gpfs/lsawade/SpecfemMagicGF'
database = os.path.join(specfemmagic, 'DB_force_10_b')
forwarddir = database + "_forward_test"
h5files = os.path.join(database, '*', '*', '*.h5')

# CMTSOLUTION
cmt = CMTSOLUTION.read(os.path.join(forwarddir, 'DATA', 'CMTSOLUTION'))

print(cmt)
print(glob(h5files))
# %% Initialize the GF manager
gfm = GFManager(glob(h5files)[:])
gfm.load_header_variables()
gfm.get_elements(cmt.latitude, cmt.longitude, cmt.depth, 30)

# %% Download data

# raw, inv = download_stream(
#     cmt.origin_time,
#     duration=4*3600,
#     network='II,IU',
#     station=','.join(gfm.stations),
#     location='00',
#     channel='LH*',
#     starttimeoffset=-300,
#     endtimeoffset=300
# )

fw = read(os.path.join(forwarddir, 'OUTPUT_FILES', '*.*.*.sac'))

# %% Get a bunch of seismograms

# cmt = CMTSOLUTION.read('/scratch/gpfs/lsawade/SpecfemMagicGF/CMTSOLUTION')
rp = gfm.get_seismograms(cmt)


# %% Process

obs = process_stream(fw, cmt=cmt, starttimeoffset=120, duration=4*3600-120)
syn = process_stream(rp, cmt=cmt, starttimeoffset=120, duration=4*3600-120)

# obs = fw.copy()
# syn = rp.copy()
# %%

pobs, psyn = select_pairs(obs, syn)

# %% Plot section

starttime = psyn[0].stats.starttime + 0
endtime = starttime + 4*3600
limits = (starttime, endtime)

# Plots a section of observed and synthetic
reset_mpl('','')
plotsection(pobs, psyn, cmt, comp='Z', lw=0.25, limits=limits)
plt.savefig('testsection_data.pdf', dpi=300)
