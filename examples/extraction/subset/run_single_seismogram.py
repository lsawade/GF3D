#!/bin/env python
"""
Single Seismogram
=================

The tutorial will go over the reading of a subset file by loading one that is
included in the Github repository. At the end, we plot the waveforms using
built-in plotting tools.

Loading all modules
-------------------

"""
# sphinx_gallery_thumbnail_number = 1
# sphinx_gallery_dummy_images = 1

# External
import matplotlib.pyplot as plt
from obspy import read, read_inventory

# Internal
from gf3d.source import CMTSOLUTION
from gf3d.seismograms import GFManager
from gf3d.process import process_stream
from gf3d.plot.seismogram import plotseismogram

# %%
# Getting a CMTSOLUTION

# CMTSOLUTION
cmt = CMTSOLUTION.read('../../DATA/single_element_read/CMTSOLUTION')
print(cmt)

# %%
# Loading the subset database

gfsub = GFManager("../../DATA/single_element_read/single_element.h5")
gfsub.load()

# %%
# Load Observed Data and response files
raw = read("../../DATA/single_element_read/traces/*.sac")
inv = read_inventory("../../DATA/single_element_read/station.xml")

# %%
# Note that the data has been downloaded from IRIS previously, but it has not
# been processed. Now, we can query the database for synthetic seismograms
# corresponding to the observed ones.

# %% Get seismograms from the database

rp = gfsub.get_seismograms(cmt)


# %%
# Then, we select the station in question and process both observed and
# synthetics seismograms for a band-pass of [40-300].

network = 'II'
station = 'BFO'
obs = process_stream(
    raw.select(network=network, station=station), inv=inv, cmt=cmt, duration=3300)
syn = process_stream(
    rp.select(network=network, station=station), cmt=cmt, duration=3300)

# %%
# Set plot limits and plot seismogram

starttimeoffset = 500.0
endtimeoffset = 0.0
duration = 1000.0
limits = \
    syn[0].stats.starttime + starttimeoffset, \
    syn[0].stats.starttime + starttimeoffset + duration + endtimeoffset

plotseismogram(obs, syn, cmt, limits=limits)
plt.show()

# %%
# Note that the seismograms are offset for clarity.
