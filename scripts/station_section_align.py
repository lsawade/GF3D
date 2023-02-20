# %% Imports
# External
import matplotlib.pyplot as plt
from obspy import read, read_inventory, Stream

# Internal
from lwsspy.GF.source import CMTSOLUTION
from lwsspy.GF.seismograms import SGTManager
from lwsspy.GF.process import process_stream, select_pairs
from lwsspy.GF.plot.section_aligned import plotsection_aligned, get_azimuth_distance_traveltime, filter_stations


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

obs = process_stream(raw, inv=inv, cmt=cmt, duration=3600)
syn = process_stream(rp, cmt=cmt, duration=3600)

# %%
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

selection = filter_stations(Pobs, Sobs)

Pobs = Stream([Pobs[_i] for _i in selection])
Psyn = Stream([Psyn[_i] for _i in selection])
Sobs = Stream([Sobs[_i] for _i in selection])
Ssyn = Stream([Ssyn[_i] for _i in selection])
# %% Plot section

# Plots a section of observed and synthetic
plt.close('all')
fig = plt.figure(figsize=(8, 5))
ax = plt.subplot(1, 2, 1)
plotsection_aligned(
    Pobs, Psyn, cmt, comp='Z', lw=0.75, ax=ax,
    traveltime_window=('P', windowP), labelright=False)

ax = plt.subplot(1, 2, 2)
plotsection_aligned(
    Sobs, Ssyn, cmt, comp='Z', lw=0.75, ax=ax,
    traveltime_window=('S', windowS), labelleft=False)
title = (f"{cmt.cmt_time.ctime()} "
         f"Loc: {cmt.latitude:.2f}dg, {cmt.longitude:.2f}dg, {cmt.depth:.1f}km"
         f" - BP: [40s, 300s]")
fig.suptitle(title, fontsize='small')
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1)
plt.savefig('station_section_align.pdf', dpi=300)

# %%
