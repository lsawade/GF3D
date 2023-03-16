# %% Imports
# External
from glob import glob
import matplotlib.pyplot as plt
import os, sys

# Internal
from gf3d.source import CMTSOLUTION
from gf3d.seismograms import GFManager
from gf3d.process import process_stream, select_pairs
from gf3d.plot.section import plotsection
from gf3d.download import download_stream


# %% Files

# DB files
specfemmagic = '/scratch/gpfs/lsawade/SpecfemMagicGF'
elementdir = '/home/lsawade/gf3d/examples/DATA/single_element_read/'
subsetfilename = os.path.join(elementdir, 'single_element.h5')
tracedir = os.path.join(elementdir, "traces")
stationxml = os.path.join(elementdir, "station.xml")

h5files = os.path.join(specfemmagic, 'DB', '*', '*', '*.h5')

# CMTSOLUTION
cmt = CMTSOLUTION.read(
    '/home/lsawade/GF3D/examples/DATA/single_element_read/CMTSOLUTION')

# %% Initialize the GF manager
gfm = GFManager(glob(h5files)[:])
gfm.load_header_variables()
gfm.get_elements(cmt.latitude, cmt.longitude, cmt.depth, 1, NGLL=3)

# %% Write a subset
gfm.write_subset(subsetfilename, duration=3600.0)

# %% load a subset

gfsub = GFManager(subsetfilename)
gfsub.load()


# %% Download data

raw, inv = download_stream(
    cmt.origin_time,
    duration=3600,
    network='II,IU',
    station=','.join(gfsub.stations),
    location='00',
    channel='LH*',
    starttimeoffset=-300,
    endtimeoffset=300
)

# Write traces to directory
for tr in raw:
    tr.write(os.path.join(tracedir, tr.id + ".sac"), format="SAC")

# Write metadata
inv.write(stationxml, format='STATIONXML')

# %% Get a bunch of seismograms

rp = gfsub.get_seismograms(cmt)

# %% Process

obs = process_stream(raw, inv=inv, cmt=cmt, duration=3600)
syn = process_stream(rp, cmt=cmt, duration=3600)

# %%

pobs, psyn = select_pairs(obs, syn)

# %% Plot section

starttime = psyn[0].stats.starttime + 0
endtime = starttime + 3600
limits = (starttime.datetime, endtime.datetime)

# Plots a section of observed and synthetic
plotsection(pobs, psyn, cmt, comp='Z', lw=0.75, limits=limits)
plt.savefig('subset_IO_section.pdf', dpi=300)

sys.exit()


# %%

# %% Initialize the GF manager
gfm = GFManager(glob(h5files)[:])
gfm.load_header_variables()

# %%
gfm.get_elements(-23.0, -68.0, 150, dist_in_km=175.0, NGLL=3)

# %%
gfm.write_subset(
    '/scratch/gpfs/lsawade/subset_S23_W68_Z150_NGLL3.h5', duration=4*3600.0)


# %%
gfm2 = GFManager(
    '/Users/lucassawade/Downloads/subset_S23_W68_Z150_R175_NGLL3.h5')
gfm2.load()

# %%
st = gfm2.get_seismograms(cmt)
