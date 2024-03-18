# %%
# Get data from syngine
import os
import matplotlib.pyplot as plt
from gf3d.source import CMTSOLUTION
from gf3d.download import download_stream
import obspy
import numpy as np
from obspy.geodetics.base import gps2dist_azimuth
import matplotlib as mpl
import obsplotlib.plot as opl

mpl.rcParams["font.family"] = "Times"
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
# Download data from IRIS
raw, inv = download_stream(
    cmt.origin_time,
    7200,
    starttimeoffset=-120,
    endtimeoffset=120,
    network="G,GE,II,IU,MN,IC",
    station=None,
    channel="LH*",
)


# %%
# Process just ever so slightly


def process(
    st,
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
            inventory=inv, output="DISP", pre_filt=(0.001, 0.005, 0.1, 0.2)
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

    if isinstance(starttime, obspy.UTCDateTime) and isinstance(npts, int):
        out.interpolate(starttime=starttime, npts=npts, sampling_rate=sampling_rate)

    out.taper(max_percentage=0.05, type="cosine")

    return out


dt_proc = process(
    raw.select(component="Z"),
    starttime=cmt.origin_time,
    npts=7200,
    bandpass=[50, 150],
    inv=inv,
    remove_response=True,
)

# %%

# Artifical synthetics: Here you could put your own stuff
rp_proc = dt_proc.copy()

for tr in rp_proc:
    tr.data += np.random.randn(len(tr.data)) * 0.05 * np.max(np.abs(tr.data))

# %%
# Attach geometry
opl.attach_geometry(dt_proc, cmt.latitude, cmt.longitude)


# %%
# Select intersections
component = "Z"
dt_plot, rp_plot = opl.select_intersection(
    [dt_proc.select(component=component), rp_proc.select(component=component)]
)
opl.attach_geometry(dt_plot, cmt.latitude, cmt.longitude, inv=inv)
opl.attach_geometry(rp_plot, cmt.latitude, cmt.longitude)
opl.copy_geometry(dt_plot, rp_plot)

for tr in dt_plot.select(station="TIXI"):
    dt_plot.remove(tr)
for tr in rp_plot.select(station="TIXI"):
    rp_plot.remove(tr)

# %%
# Plot section
plt.close("all")
fig = plt.figure(figsize=(8, 9))

(ax, _) = opl.section(
    [dt_plot, rp_plot],
    labels=["Observed", "3D Numerical"],
    colors=["k", "k"],
    ls=["-", "--"],
    origin_time=cmt.origin_time,
    plot_geometry=False,
    lw=[0.5, 0.5],
    plot_amplitudes=False,
    scale=4,
    skip_station=2,
    plot_stations_right=True,  # Skip station must be even!
    legendargs=dict(loc="lower right", ncol=2, frameon=False, bbox_to_anchor=(1, 1)),
    limits=(0, 7200),
)

# Adding label in the top-left outside corner of the plot
# opl.plot_label(
#     ax,
#     "Bandpass: 50-150s -- Mw: 7.1",
#     location=6,
#     box=False,
#     fontsize="medium",
#     dist=0.025,
# )

# This is just to offset the bottom axis
ax.set_ylim(-1.5, None)

headerdict = dict(
    event=cmt.eventname,
    event_time=None,
    event_latitude=cmt.latitude,
    event_longitude=cmt.longitude,
    event_depth_in_km=cmt.depth,
    location=6,
    # fontsize='small'
)
# Adding Header (title, at the top of the plot)
# opl.add_header(ax, **headerdict)

plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.95)
plt.show(block=False)
plt.savefig("java_station_section.pdf")
