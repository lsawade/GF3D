#!/bin/env python
"""
Single Seismogram
=================

The tutorial will go over the reading of a three component seismogram from a
station file in the database by loading one that is included in the Github repo.
At the end, we plot the seismograms using built-in plotting tools.

Loading all modules
-------------------

"""
# sphinx_gallery_thumbnail_number = 1
# sphinx_gallery_dummy_images = 1

#%% Import things

# External
import matplotlib.pyplot as plt
from obspy import read, read_inventory, Stream

# Internal
from gf3d.source import CMTSOLUTION
from gf3d.seismograms import GFManager, get_seismograms
from gf3d.process import process_stream
from gf3d.plot.seismogram import plotseismogram


# %%
# Getting a CMTSOLUTION and load data

# CMTSOLUTION
# cmt = CMTSOLUTION.read('../../DATA/single_element_read/CMTSOLUTION')
# print(cmt)

cmt = CMTSOLUTION.read("/scratch/gpfs/lsawade/SpecfemMagicGF/specfem3d_globe_forward/DATA/CMTSOLUTION")
raw = read("/scratch/gpfs/lsawade/SpecfemMagicGF/specfem3d_globe_forward/OUTPUT_FILES/II.BFO*.sac")

# raw = read("../../DATA/single_element_read/traces/*.sac")
# inv = read_inventory("../../DATA/single_element_read/station.xml")

# %%
# Note that the data has been downloaded from IRIS previously, but it has not
# been processed. Now, we can query the database for synthetic seismograms
# corresponding to the observed ones.

# %%
# Loading a single station file from the database

# Load subset
# filename = '../../DATA/single_element_read/DB/II/BFO/II.BFO.h5'
filename = '/scratch/gpfs/lsawade/SpecfemMagicGF/DB_hdur_test/II/BFO/II.BFO.h5'
filename = '/scratch/gpfs/lsawade/SpecfemMagicGF/II.BFO_half_10.h5'
gfm = GFManager([filename])
gfm.load_header_variables()

# You will have to load a subset of elements first (just one here for storage reasons)
gfm.get_elements(cmt.latitude, cmt.longitude, cmt.depth, 100, NGLL=5)

# Finally you can read the seismograms.
rp = gfm.get_seismograms(cmt)

# rp = get_seismograms(filename, cmt)


# %%
# Then, we select the station in question and process both observed and
# synthetics seismograms for a band-pass of [40-300].

network = 'II'
station = 'BFO'
# obs = process_stream(
#     raw.select(network=network, station=station), inv=inv, cmt=cmt, duration=3600)
# syn = process_stream(
#     rp.select(network=network, station=station), cmt=cmt, duration=3600)

# obs = rp
obs = raw.select(network=network, station=station)
syn = rp.select(network=network, station=station)
from gf3d.signal.filter import butter_low_two_pass_filter

# def process_stream(st: Stream):
#     st.filter('bandpass', freqmin=1/500.0, freqmax=1/40.0, zerophase=True)
# # def process_stream(st: Stream):
# #     st.filter('lowpass', freq=1/40.0, zerophase=True)
# process_stream(obs)
# process_stream(syn)


obs = process_stream(obs, cmt=cmt, duration=14000, starttimeoffset=10)
syn = process_stream(syn, cmt=cmt, duration=14000, starttimeoffset=10)

# obs.differentiate()
# syn = process_stream(
#     rp.select(network=network, station=station), cmt=cmt, duration=14400)
# syn = None #raw

# %%
# Set plot limits and plot seismogram

starttimeoffset = 0.0
endtimeoffset = 0.0
duration = 14400
limits = \
    obs[0].stats.starttime + starttimeoffset, \
    obs[0].stats.starttime + starttimeoffset + duration + endtimeoffset

plotseismogram(obs, syn, cmt, limits=limits, nooffset=True, lw=0.25)

plt.savefig('testseis.pdf', dpi=300)
plt.show()

# %%
# The seismograms are offset for clarity so that they are easy to read.s
