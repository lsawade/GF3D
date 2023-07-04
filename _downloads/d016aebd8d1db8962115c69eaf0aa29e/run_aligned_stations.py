#!/bin/env python
"""
Aligned Station Section
=======================

The tutorial will go over the reading of a subset file by loading one that is
included in the directory. At the end we plot an aligned section of waveforms
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
from gf3d.plot.util import set_default_color
from gf3d.plot.section_aligned import plotsection_aligned, get_azimuth_distance_traveltime, filter_stations

# %%
# Get Sources and Seismograms
# ---------------------------
#
# Load CMTSOLUTION, observed data, and Green functions

cmt = CMTSOLUTION.read('../../DATA/single_element_read/CMTSOLUTION')

# %%
# Load Observed Data

raw = read("../../DATA/single_element_read/traces/*.sac")
inv = read_inventory("../../DATA/single_element_read/station.xml")

# %%
# Get synthetics

gfsub = GFManager("../../DATA/single_element_read/single_element.h5")
gfsub.load()
rp = gfsub.get_seismograms(cmt)


# %%
# Process
# -------
#

obs = process_stream(raw, inv=inv, cmt=cmt, duration=3300)
syn = process_stream(rp, cmt=cmt, duration=3300)

# %%
# Windowing thee traces
#

windowP = (-100, 250)
windowS = (-100, 250)
pobs, psyn = select_pairs(obs, syn)

# For pwaves
Pobs, Psyn = get_azimuth_distance_traveltime(
    cmt, pobs, psyn, comp='Z',
    traveltime_window=('P', windowP))

Sobs, Ssyn = get_azimuth_distance_traveltime(
    cmt, pobs, psyn, comp='Z',
    traveltime_window=('S', windowS))

# %%
# Selecting matching traces
#

selection = filter_stations(Pobs, Sobs)

Pobs = Stream([Pobs[_i] for _i in selection])
Psyn = Stream([Psyn[_i] for _i in selection])
Sobs = Stream([Sobs[_i] for _i in selection])
Ssyn = Stream([Ssyn[_i] for _i in selection])

# %%
# Plot section
# ------------
#


# Plots a section of observed and synthetic
fig = plt.figure(figsize=(6, 4))

# Plot Arrivals around ak135 P arrival
ax = plt.subplot(1, 2, 1)
plotsection_aligned(
    Pobs, Psyn, cmt, comp='Z', lw=1.0, ax=ax,
    traveltime_window=('P', windowP), labelright=False)

# Plot Arrivals around ak135 S arrival
ax = plt.subplot(1, 2, 2)
plotsection_aligned(
    Sobs, Ssyn, cmt, comp='Z', lw=1.0, ax=ax,
    traveltime_window=('S', windowS), labelleft=False)

# Set Title
title = (f"{cmt.cmt_time.ctime()} "
         f"Loc: {cmt.latitude:.2f}dg, {cmt.longitude:.2f}dg, {cmt.depth:.1f}km"
         f" - BP: [40s, 300s]")
fig.suptitle(title, fontsize='small')

# Adjust plot
plt.subplots_adjust(left=0.1, right=0.85, bottom=0.1, top=0.9, wspace=0.1)

plt.show()
