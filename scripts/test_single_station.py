# %%
# Get data from syngine
import os
from glob import glob
from gf3d.seismograms import GFManager
from gf3d.client import GF3DClient
from gf3d.source import CMTSOLUTION
import matplotlib.pyplot as plt
from gf3d.plot.seismogram import plotseismogram, add_header
from gf3d.download import download_stream
import obspy
import numpy as np
from obspy.geodetics.base import gps2dist_azimuth
from gf3d.plot import util as putil
import matplotlib as mpl
import obsplotlib.plot as opl

# For JT
# plt.style.use("dark_background")
mpl.rcParams["font.family"] = "Serif"

# Standard
# mpl.rcParams["font.family"] = "Monospace"

# %%

cmtfile = """ PDEW2023  4 14  9 55 45.20  -6.0400  112.0500 597.0 0.0 7.0 JAVA, INDONESIA
event name:     202304140955A
time shift:      6.2800
half duration:   8.4000
latitude:       -6.1900
longitude:     112.1600
depth:         606.8500
Mrr:      -4.300000e+26
Mtt:       4.070000e+26
Mpp:       2.280000e+25
Mrt:       2.190000e+26
Mrp:      -1.520000e+26
Mtp:      -1.060000e+26
"""

cmt = CMTSOLUTION.read(cmtfile)

# %%
# Get data from database

if not os.path.exists("/lustre/orion/geo111/scratch/lsawade/java_single_station.h5"):

    # Client for downloading data
    # gfc = GF3DClient(db="glad-m25")
    # gfc.get_subset("java.h5", cmt.latitude, cmt.longitude, cmt.depth, radius_in_km=70)

    # Local file
    databasename = '/lustre/orion/geo111/scratch/lsawade/DB_new/GLAD-M25/128_single'

    # Check for files given database path
    db_globstr = os.path.join(databasename, '*', '*', '*.h5')

    # Get all files
    db_files = glob(db_globstr)

    # Get subset
    GFM = GFManager(db_files)
    GFM.load_header_variables()
    GFM.write_subset_directIO('/lustre/orion/geo111/scratch/lsawade/java_single_station.h5', cmt.latitude, cmt.longitude,
                              cmt.depth, 70.0, NGLL=5,
                              fortran=False)

network = "G"
station = "AIS"

# %%
# Download data from IRIS
raw, inv = download_stream(
    cmt.origin_time,
    7200,
    starttimeoffset=-120,
    endtimeoffset=120,
    network="G",
    station="AIS",
    channel="LH*",
)


# %%
# Get the synthetics
gfm = GFManager("/lustre/orion/geo111/scratch/lsawade/java_single_station.h5")
gfm.load()
rp = gfm.get_seismograms(cmt)

# %%


# Process just ever so slightly
def process(
    st,
    baz,
    inv: obspy.Inventory | None = None,
    remove_response=False,
    starttime=None,
    npts=None,
    sampling_rate=1,
    bandpass=[200, 500],
):
    out = st.copy()
    out.detrend("demean")
    out.detrend("linear")
    out.taper(max_percentage=0.05, type="cosine")

    if inv is not None and remove_response:
        out.remove_response(
            inventory=inv, output="DISP", pre_filt=(0.001, 0.005, 0.1, 0.2),
            water_level=100
        )

    if inv is not None:
        out.rotate("->ZNE", inventory=inv)

    out.filter(
        "bandpass",
        freqmin=1 / bandpass[1],
        freqmax=1 / bandpass[0],
        corners=2,
        zerophase=True,
    )

    # out.rotate("NE->RT", back_azimuth=baz)

    if isinstance(starttime, obspy.UTCDateTime) and isinstance(npts, int):
        out.interpolate(starttime=starttime, npts=npts, sampling_rate=sampling_rate)

    out.taper(max_percentage=0.05, type="cosine")

    return out

bp = [50, 150]
dt_proc = process(
    raw.select(network=network, station=station),
    0,
    starttime=cmt.origin_time,
    npts=7200,
    bandpass=bp,
    inv=inv,
    remove_response=True,
)

rp_proc = process(
    rp.select(network=network, station=station),
    0,
    starttime=cmt.origin_time,
    npts=7200,
    bandpass=bp,
)


# %%
# Attach geometry

opl.attach_geometry(rp_proc, cmt.latitude, cmt.longitude)
opl.attach_geometry(dt_proc, cmt.latitude, cmt.longitude)


# %%
# Select intersections
rp_proc.sort(keys=["distance", "network", "station"])
dt_plot, rp_plot = opl.select_intersection([dt_proc, rp_proc.copy()])

# %%
# Plot section
plt.close("all")
fig = plt.figure(figsize=(9, 11))

channel_axes = opl.station(
    [dt_plot, rp_plot],
    components="ZNE",
    labels=["Observed", "3D Numerical"],
    colors=["k", (0.9, 0.15, 0.15)],
    ls=["-", "-"],
    origin_time=cmt.origin_time,
    lw=[0.625, 0.625],
    limits=(0, 7200),
)
# axl.tick_params(labelleft=False, bottom=False, labelbottom=False)
# axr.tick_params(labelright=False)

# opl.plot_label(
#     ax,
#     "Bandpass: 50-150s -- Mw: 7.1",
#     location=6,
#     box=False,
#     fontsize="medium",
#     dist=0.025,
# )
# axl.set_ylim(None, 1)
# axl.set_xlabel("")
# axl.spines["bottom"].set_visible(False)

# Compute station event distance

headerdict = dict(
    event=cmt.eventname,
    event_time=cmt.origin_time,
    event_latitude=cmt.latitude,
    event_longitude=cmt.longitude,
    event_depth_in_km=cmt.depth,
    event_Mw=cmt.Mw,
    bandpass=bp,
    location=6,
    fontsize='small',
    color='black'
)
opl.add_header(channel_axes[0], **headerdict)
# plt.subplots_adjust(left=0.075, right=0.925, bottom=0.0, top=1.0)
plt.show(block=False)
plt.savefig("java_station_section.pdf")

