# %% Imports
# External
import matplotlib.pyplot as plt
from obspy import read, read_inventory

# Internal
from lwsspy.GF.source import CMTSOLUTION
from lwsspy.GF.seismograms import SGTManager
from lwsspy.GF.process import process_stream, select_pairs
from lwsspy.GF.plot.section import plotsection


# %% Files

# CMTSOLUTION
cmt = CMTSOLUTION.read(
    '/home/lsawade/lwsspy/lwsspy.GF/scripts/DATA/single_element_read/CMTSOLUTION')

# Load subset
sgtsub = SGTManager(
    "/home/lsawade/lwsspy/lwsspy.GF/scripts/DATA/single_element_read/single_element.h5")
sgtsub.load()

# Load Observed Data
raw = read(
    "/home/lsawade/lwsspy/lwsspy.GF/scripts/DATA/single_element_read/traces/*.SAC")

inv = read_inventory(
    "/home/lsawade/lwsspy/lwsspy.GF/scripts/DATA/single_element_read/station.xml")

# %% Get a bunch of seismograms

rp = sgtsub.get_seismograms(cmt)


# %% Process

obs = process_stream(raw, inv=inv, cmt=cmt, duration=4*3600)
syn = process_stream(rp, cmt=cmt, duration=4*3600)

# %%

pobs, psyn = select_pairs(obs, syn)

# %% Plot section

starttime = psyn[0].stats.starttime + 0
endtime = starttime + 4*3600
limits = (starttime.datetime, endtime.datetime)

# Plots a section of observed and synthetic
plotsection(pobs, psyn, cmt, comp='Z', lw=0.75, limits=limits)
plt.savefig('subset_single_element_read_section.pdf', dpi=300)
