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
import numpy as np
from gf3d.signal.filter import butter_low_two_pass_filter
import matplotlib.pyplot as plt
from obspy import read, read_inventory, Stream

# Internal
from gf3d.source import CMTSOLUTION
from gf3d.seismograms import GFManager
from gf3d.process import process_stream, select_pairs
from gf3d.plot.section import plotsection

# %%
# half duration:  33.4000

# CMTSOLUTION
cmt = CMTSOLUTION.read('../../DATA/single_element_read/CMTSOLUTION')

cmt = CMTSOLUTION.read("""
 PDEW2015  9 16 22 54 32.90 -31.5700  -71.6700  22.4 0.0 8.3 NEAR COAST OF CENTRAL CH
event name:     201509162254A
time shift:     49.9800
half duration:  33.4000
latitude:      -31.1300
longitude:     -72.0900
depth:          17.3500
Mrr:       1.950000e+28
Mtt:      -4.360000e+26
Mpp:      -1.910000e+28
Mrt:       7.420000e+27
Mrp:      -2.480000e+28
Mtp:       9.420000e+26
"""
                       )

# Load subset
# gfsub = GFManager("../../DATA/single_element_read/single_element.h5")
# gfsub = GFManager("../../../single_element_not_fortran.h5")
gfsub = GFManager("../../../single_element.h5")
gfsub.load()

# Load Observed Data
# raw = read("../../DATA/single_element_read/traces/*.sac")
# inv = read_inventory("../../DATA/single_element_read/station.xml")

raw = read("../../../specfem_traces/*")
# %% Get seismograms from the database

rp = gfsub.get_seismograms(cmt)


# %% Process

# obs = process_stream(raw, inv=inv, cmt=cmt, duration=3600)
obs = raw  # process_stream(raw, cmt=cmt, duration=3600)
syn = rp  # process_stream(rp, cmt=cmt, duration=3600)

for tr in obs:
    tr.data = butter_low_two_pass_filter(
        tr.data, 1/40.0, 1.0/tr.stats.delta, order=5)

for tr in syn:
    tr.data = butter_low_two_pass_filter(
        tr.data, 1/40.0, 1.0/tr.stats.delta, order=5)


obs.differentiate()
obs = process_stream(obs, cmt=cmt, duration=14400)
syn = process_stream(syn, cmt=cmt, duration=14400)

# %%

pobs, psyn = select_pairs(obs, syn)

for _o, _s in zip(pobs, psyn):

    print(np.sum(_o.data*_s.data)/np.sum(_o.data*_o.data))

# %% Plot section

starttime = psyn[0].stats.starttime + 0
endtime = starttime + 14400
limits = (starttime, endtime)

# Plots a section of observed and synthetic
plotsection(pobs, psyn, cmt, comp='Z', lw=0.75, limits=limits)

plt.show()
