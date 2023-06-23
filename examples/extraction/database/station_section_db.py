#!/bin/env python
"""
Station Section
===============

The tutorial will go over the reading of station files from a database by
loading one that is included in the github repo. At the end, we plot a station
section of waveforms using built-in plotting tools.

Loading all modules
-------------------

"""
# sphinx_gallery_thumbnail_number = 1
# sphinx_gallery_dummy_images = 1

#%%
# External
from glob import glob
import matplotlib.pyplot as plt
from obspy import read, read_inventory, Stream

# Internal
from gf3d.source import CMTSOLUTION
from gf3d.seismograms import GFManager
from gf3d.process import process_stream, select_pairs
from gf3d.plot.section import plotsection

# %%
# Read observed data and event parameters

# CMTSOLUTION
cmt = CMTSOLUTION.read('../../DATA/single_element_read/CMTSOLUTION')

# Load Observed Data
raw = read("../../DATA/single_element_read/traces/*.sac")
inv = read_inventory("../../DATA/single_element_read/station.xml")

# %%
# Get seismograms from the database

# Load subset
gfm = GFManager(glob('../../DATA/single_element_read/DB/*/*/*.*.h5'))
gfm.load_header_variables()

# You will have to load a subset of elements first (just one here for storage reasons)
gfm.get_elements(cmt.latitude, cmt.longitude, cmt.depth, 100, NGLL=5)

# Finally you can read the seismograms.
rp = gfm.get_seismograms(cmt)

# %% Process

obs = process_stream(raw, inv=inv, cmt=cmt, duration=3600)
syn = process_stream(rp, cmt=cmt, duration=3600)

# obs = raw
# syn = rp

# %%
# Note that the only 3 stations (II.BFO, IU.ANMO, IU.HRV) are in the example
# database, which means that the observed data that we downloaded for the
# example subset file includes other stations

pobs, psyn = select_pairs(obs, syn)

# %%
# So, don't worry about the 'Cant find <Network>.<Station>..<Component>.
#
# Plot section with the data

starttime = pobs[0].stats.starttime + 0
endtime = starttime + 3600
limits = (starttime, endtime)

# Plots a section of observed and synthetic
plotsection(pobs, psyn, cmt, comp='Z', lw=0.75, limits=limits)

plt.show()
