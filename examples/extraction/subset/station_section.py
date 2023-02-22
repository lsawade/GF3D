#!/bin/env python
"""
Station Section
===============

The tutorial will go over the reading of a subset file by loading one that is
included in the directory. At the end we plot a station section of waveforms
using built-in plotting tools.

Loading all modules
-------------------


"""
# sphinx_gallery_thumbnail_number = 1
# sphinx_gallery_dummy_images = 1

# External
import matplotlib.pyplot as plt
from obspy import read, read_inventory, Stream

# Internal
from gf3d.source import CMTSOLUTION
from gf3d.seismograms import GFManager
from gf3d.process import process_stream, select_pairs
from gf3d.plot.section import plotsection

# %%
#

# CMTSOLUTION
cmt = CMTSOLUTION.read('../../DATA/single_element_read/CMTSOLUTION')

# Load subset
gfsub = GFManager("../../DATA/single_element_read/single_element.h5")
gfsub.load()

# Load Observed Data
raw = read("../../DATA/single_element_read/traces/*.sac")
inv = read_inventory("../../DATA/single_element_read/station.xml")

# %% Get seismograms from the database

rp = gfsub.get_seismograms(cmt)


# %% Process

obs = process_stream(raw, inv=inv, cmt=cmt, duration=3600)
syn = process_stream(rp, cmt=cmt, duration=3600)

# %%

pobs, psyn = select_pairs(obs, syn)

# %% Plot section

starttime = psyn[0].stats.starttime + 0
endtime = starttime + 3600
limits = (starttime, endtime)

# Plots a section of observed and synthetic
plotsection(pobs, psyn, cmt, comp='Z', lw=0.75, limits=limits)

plt.show()
