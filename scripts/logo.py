# %% Imports
# External
from copy import deepcopy
import os
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


# %% Files

HOME = os.environ['HOME']

# CMTSOLUTION
cmt = CMTSOLUTION.read(
    os.path.join(HOME, 'GF3D/scripts/DATA/single_element_read/CMTSOLUTION'))

# Load subset
gfsub = GFManager(
    os.path.join(HOME, "GF3D/scripts/DATA/single_element_read/single_element.h5"))
gfsub.load()

# Load Observed Data
raw = read(
    os.path.join(HOME, "GF3D/scripts/DATA/single_element_read/traces/*.sac"))

inv = read_inventory(
    os.path.join(HOME, "GF3D/scripts/DATA/single_element_read/station.xml"))

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

selection = filter_stations(Pobs, Sobs)

Pobs = Stream([Pobs[_i] for _i in selection])
Psyn = Stream([Psyn[_i] for _i in selection])


# %%
# Plots a section of observed and synthetic
plt.close('all')
fig = plt.figure(figsize=(2.5, 1/2))
gs = GridSpec(1, 3, width_ratios=[0.8, 0.2, 1.0])
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
plot_label(ax, 'GF3D', location=0, box=False, fontsize=44, color=DARKGRAY)
plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.00)
plt.savefig('logo.png', dpi=600, transparent=True)

# %% Favicon

plt.close('all')
fig = plt.figure(figsize=(1.0, 1.0))
ax = plt.axes()
ax.axis('off')
cmt.axbeach(ax, 0.5, 0.5, width=30, facecolor=ORANGE, edgecolor='w',
            bgcolor=[_g for _g in GRAY], linewidth=0.5)
plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.00)
plt.savefig('favicon.png', dpi=32, transparent=True)
