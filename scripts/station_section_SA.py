# %% Imports
# External
import matplotlib.pyplot as plt

# Internal
from gf3d.source import CMTSOLUTION
from gf3d.seismograms import GFManager
from gf3d.process import process_stream, select_pairs
from gf3d.plot.section import plotsection
from gf3d.download import download_stream


# %% Files

# DB files
C202206031454A = """
PDEW2022  6  3 14 54 34.40 -23.1600  -68.2600  98.6 0.0 5.3 NORTHERN CHILE
event name:     202206031454A
time shift:      4.5100
half duration:   1.1
latitude:      -23.1200
longitude:     -68.3900
depth:         131.4200
Mrr:      -7.080000e+23
Mtt:       1.510000e+23
Mpp:       5.570000e+23
Mrt:       3.540000e+23
Mrp:      -4.610000e+23
Mtp:      -5.540000e+23
"""

# CMTSOLUTION
cmt = CMTSOLUTION.read(C202206031454A)

# %% Initialize the GF manager
gfm = GFManager(
    "/Users/lucassawade/Downloads/subset_S23_W68_Z150_R175_NGLL3.h5")
gfm.load()

# %% Download data

raw, inv = download_stream(
    cmt.origin_time,
    duration=4*3600,
    network='II,IU',
    station=','.join(gfm.stations),
    location='00',
    channel='LH*',
    starttimeoffset=-300,
    endtimeoffset=300
)

# %% Get a bunch of seismograms

# cmt = CMTSOLUTION.read('/scratch/gpfs/lsawade/SpecfemMagicGF/CMTSOLUTION')
rp = gfm.get_seismograms(cmt)


# %% Process

obs = process_stream(raw, inv=inv, cmt=cmt, duration=4*3600)
syn = process_stream(rp, cmt=cmt, duration=4*3600)
syn.differentiate()
# %%

pobs, psyn = select_pairs(obs, syn)

# %% Plot section

starttime = psyn[0].stats.starttime + 0
endtime = starttime + 4*3600
limits = (starttime, endtime)

# Plots a section of observed and synthetic
plotsection(pobs, psyn, cmt, comp='Z', lw=0.75, limits=limits)
plt.savefig('testsection_data.pdf', dpi=300)
