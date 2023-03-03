#!/bin/env python
# %%
"""

Recreating the logo
===================

This example will go over the creation of the logo which is based on the
observed data and database subset that come are stored in the example directory
of this package.


"""

# External
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from obspy import read, read_inventory, Stream

# Internal
from gf3d.constants import GRAY, ORANGE, BLUE, DARKGRAY
from gf3d.source import CMTSOLUTION
from gf3d.seismograms import GFManager
from gf3d.process import process_stream, select_pairs
from gf3d.plot.section_aligned import plotsection_aligned, get_azimuth_distance_traveltime, filter_stations
from gf3d.plot.util import plot_label


# %%
# Getting a CMTSOLUTION

# CMTSOLUTION
cmt = CMTSOLUTION.read('../DATA/single_element_read/CMTSOLUTION')
print(cmt)

# %%
# Loading the subset database

gfsub = GFManager("../DATA/single_element_read/single_element.h5")
gfsub.load()

# %%
# Load Observed Data and response files
raw = read("../DATA/single_element_read/traces/*.sac")
inv = read_inventory("../DATA/single_element_read/station.xml")

# %% Get a bunch of seismograms

rp = gfsub.get_seismograms(cmt)


# %%

obs = process_stream(raw, inv=inv, cmt=cmt, duration=3600)
syn = process_stream(rp, cmt=cmt, duration=3600)

# %%
windowP = (-50, 150)
pobs, psyn = select_pairs(obs, syn)

# For pwaves
Pobs, Psyn = get_azimuth_distance_traveltime(
    cmt, pobs, psyn, comp='Z',
    traveltime_window=('P', windowP))


# %%
# Plots a section of observed and synthetic
plt.close('all')
plt.rcParams["font.family"] = "monospace"

fig = plt.figure(figsize=(2.25, 1/2))
gs = GridSpec(1, 3, width_ratios=[0.8, 0.3, 1.0])
ax = fig.add_subplot(gs[0:2], zorder=1)
plotsection_aligned(
    Pobs[1:2], Psyn[1:2], cmt, comp='Z', lw=1.0, ax=ax,
    traveltime_window=('P', windowP), labels=False,
    obsc=GRAY, sync=ORANGE, newsync=BLUE)
ax.axis('off')
ax.set_ylim(-0.2, 1.75)
cmt.axbeach(ax, 0.0, 0.45, width=200/np.sqrt(2), facecolor=ORANGE, edgecolor=GRAY,
            bgcolor=DARKGRAY, linewidth=0.5)

ax = fig.add_subplot(gs[1:], zorder=0)
ax.axis('off')
plot_label(ax, 'GF  ', location=0, box=False,
           fontsize=44, color=DARKGRAY, zorder=10)
plot_label(ax, '  3 ', location=0, box=False,
           fontsize=44, color=ORANGE, zorder=10)
plot_label(ax, '   D', location=0, box=False,
           fontsize=44, color=DARKGRAY, zorder=10)
# cmt.axbeach(ax, 1.0, 0.5, width=400/np.sqrt(2), facecolor=ORANGE, edgecolor=GRAY,
#             bgcolor=DARKGRAY, linewidth=0.5, zorder=0)
# plt.subplots_adjust(left=0.0, right=0.875, bottom=0.0, top=1.0, wspace=0.00)
plt.subplots_adjust(left=0.0, right=0.975, bottom=0.0, top=1.0, wspace=0.00)

# Save the plot
plt.savefig('logo.png', dpi=600, transparent=True)

plt.show()

# %% Favicon

plt.close('all')
fig = plt.figure(figsize=(1.0, 1.0))
ax = plt.axes()
ax.axis('off')
cmt.axbeach(ax, 0.5, 0.5, width=30, facecolor=ORANGE, edgecolor='w',
            bgcolor=[_g for _g in GRAY], linewidth=0.5)
plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.00)

# Save the figure
plt.savefig('favicon.png', dpi=32, transparent=True)

plt.show()
