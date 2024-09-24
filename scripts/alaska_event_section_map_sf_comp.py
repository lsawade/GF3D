# %%
# Get data from syngine
import os
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

mpl.rcParams["font.family"] = "monospace"
# %%

cmtfile = """ PDEW2018  1 23  9 31 40.90  56.0000 -149.1700  14.1 0.0 7.9 GULF OF ALASKA
event name:     201801230931A
time shift:     23.0700
half duration:  22.3000
latitude:       56.2200
longitude:    -149.1200
depth:          33.6100
Mrr:       2.360000e+27
Mtt:      -4.850000e+27
Mpp:       2.500000e+27
Mrt:       1.940000e+27
Mrp:      -3.620000e+27
Mtp:       7.910000e+27
"""

cmt = CMTSOLUTION.read(cmtfile)

# %%
# Load seismograms

sfs = obspy.read("DATA/alaska_seismos/*.sac")


# %%
# Get data from database
if not os.path.exists("alaska.h5"):
    gfc = GF3DClient(db="glad-m25")
    gfc.get_subset("alaska.h5", cmt.latitude, cmt.longitude, cmt.depth, radius_in_km=40)

# %%

gfm = GFManager("alaska_subset.h5")
gfm.load()
rp = gfm.get_seismograms(cmt)

# %%
# Attach geometry

opl.attach_geometry(rp, cmt.latitude, cmt.longitude)
opl.attach_geometry(sfs, cmt.latitude, cmt.longitude)



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

    out.rotate("NE->RT", back_azimuth=baz)

    if isinstance(starttime, obspy.UTCDateTime) and isinstance(npts, int):
        out.interpolate(starttime=starttime, npts=npts, sampling_rate=1)

    out.taper(max_percentage=0.05, type="cosine")

    return out

#%%

rp_proc = process(
    rp, 0, starttime=cmt.origin_time, npts=10800, bandpass=[50, 500]
).select(component="Z")

sf_proc = process(
    sfs, 0, starttime=cmt.origin_time, npts=10800, bandpass=[50, 500]
).select(component="Z")

# %%

# %%
# Plot section
plt.close("all")
fig = plt.figure(figsize=(8, 5))
(ax, _) = opl.section(
    [sf_proc, rp_proc],
    origin_time=cmt.origin_time,
    plot_geometry=False,
    lw=[0.5, 0.5],
    colors="k",
    plot_amplitudes=False,
    scale=4,
    skip_station=1,
    limits=(0, 10800),
    labels=["Forward", "Reciprocal"],
        
)
opl.plot_label(
    ax,
    "Bandpass: 50-500s -- Mw: 7.9",
    location=6,
    box=False,
    fontsize="medium",
    dist=0.025,
)

headerdict = dict(
    event=cmt.eventname,
    event_time=None,
    event_latitude=cmt.latitude,
    event_longitude=cmt.longitude,
    event_depth_in_km=cmt.depth,
    location=6,
    # fontsize='small'
)
opl.add_header(ax, **headerdict)
plt.show(block=False)
plt.savefig("alaska_station_section.pdf")

# %%

import cartopy.crs as ccrs


class HighResMollweide(ccrs.Mollweide):
    @property
    def threshold(self):
        return 100.0


class LowerThresholdRobinson(ccrs.Robinson):
    @property
    def threshold(self):
        return 1e3


def plot_event_geometry(gfm: GFManager, cmt: CMTSOLUTION):
    import cartopy
    from cartopy.crs import PlateCarree

    # Get midpoint
    mlat, mlon = cmt.latitude, cmt.longitude

    # Get unique networks
    networks = {net for net in gfm.networks}

    net_dict = dict()

    for net in networks:
        if net not in net_dict:
            net_dict[net] = dict()
            net_dict[net]["latitudes"] = []
            net_dict[net]["longitudes"] = []

        for _i, gfnet in enumerate(gfm.networks):
            if gfnet == net:
                net_dict[net]["latitudes"].append(gfm.latitudes[_i])
                net_dict[net]["longitudes"].append(gfm.longitudes[_i])

    # Projection
    projection = ccrs.Mollweide(central_longitude=mlon)
    projection._threshold = projection._threshold / 50

    plt.figure(figsize=(8.0, 4.0))

    mapax = plt.subplot(1, 1, 1, projection=projection)
    mapax.set_global()
    mapax.add_feature(
        cartopy.feature.OCEAN,
        zorder=-1,
        edgecolor="none",
        linewidth=0.25,
        facecolor=(1.0, 1.0, 1.0),
    )
    mapax.add_feature(
        cartopy.feature.LAND,
        zorder=-1,
        edgecolor="k",
        linewidth=0.25,
        facecolor=(0.9, 0.9, 0.9),
    )
    mapax.gridlines(lw=0.25, ls="-", color=(0.75, 0.75, 0.75), zorder=-1)

    cmap = plt.get_cmap("rainbow")
    colors = [*cmap(np.linspace(0, 1, len(net_dict), endpoint=True))]

    for _i, net in enumerate(net_dict):
        for _j, (_lat, _lon) in enumerate(
            zip(net_dict[net]["latitudes"], net_dict[net]["longitudes"])
        ):
            mapax.plot(
                [cmt.longitude, _lon],
                [cmt.latitude, _lat],
                "-k",
                lw=0.15,
                transform=ccrs.Geodetic(),
                zorder=0,
            )

    for _i, net in enumerate(net_dict):
        mapax.plot(
            net_dict[net]["longitudes"],
            net_dict[net]["latitudes"],
            "v",
            markerfacecolor=colors[_i],
            markersize=8,
            markeredgewidth=0.25,
            markeredgecolor="k",
            transform=PlateCarree(),
            label=net,
        )

    mapax.plot(
        cmt.longitude,
        cmt.latitude,
        "*",
        markerfacecolor=(0.2, 0.2, 0.2),
        markersize=20,
        markeredgecolor="k",
        markeredgewidth=0.25,
        transform=PlateCarree(),
    )
    plt.legend(
        loc="lower center",
        ncol=len(net_dict),
        fancybox=False,
        frameon=False,
        prop={"size": "xx-large"},
        bbox_to_anchor=(0.5, 1.0),
        columnspacing=0.0,
    )
    plt.show(block=False)


plot_event_geometry(gfm, cmt)
plt.savefig("alaska_station_event_geometry.pdf", transparent=True)

# %%

# %%


def plot_map_single_station(rlat, rlon, cmt: CMTSOLUTION):
    import cartopy
    from gf3d.geoutils import geomidpointv
    from cartopy.crs import Orthographic, PlateCarree, Geodetic

    mlat, mlon = geomidpointv(rlat, rlon, cmt.latitude, cmt.longitude)

    # Projection
    projection = Orthographic(central_longitude=mlon, central_latitude=mlat)

    plt.figure(figsize=(2.0, 4))

    mapax = plt.subplot(2, 1, 1, projection=projection)
    mapax.set_global()
    mapax.add_feature(
        cartopy.feature.LAND,
        zorder=-1,
        edgecolor="k",
        linewidth=0.25,
        facecolor=(0.9, 0.9, 0.9),
    )
    mapax.gridlines(lw=0.25, ls="-", color=(0.75, 0.75, 0.75), zorder=-1)

    mapax.plot(
        rlon,
        rlat,
        "v",
        markerfacecolor=(0.8, 0.2, 0.2),
        markersize=9,
        markeredgecolor="k",
        transform=PlateCarree(),
    )

    mapax.plot(
        cmt.longitude,
        cmt.latitude,
        "*",
        markerfacecolor=(0.2, 0.2, 0.8),
        markersize=12,
        markeredgecolor="k",
        transform=PlateCarree(),
    )

    mapax.plot(
        [cmt.longitude, rlon],
        [cmt.latitude, rlat],
        "-k",
        transform=Geodetic(),
        zorder=0,
    )

    ax = plt.subplot(2, 1, 2)

    cmt.ax_beach(ax, 0.5, 0.5, 200, clip_on=False, zorder=10, linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.subplots_adjust(hspace=0.0, left=0.0, right=1.0, bottom=0.0, top=1.0)
    plt.show(block=False)


ii_aru = rp_proc.select(network="II", station="ARU")[0]

plot_map_single_station(ii_aru.stats.latitude, ii_aru.stats.longitude, cmt)


# %%

# %%
plt.figure(figsize=(2, 2))
ax = plt.subplot(1, 1, 1)
cmt.ax_beach(ax, 0.5, 0.5, 390, clip_on=False, zorder=10, linewidth=0.25)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
plt.subplots_adjust(hspace=0.0, left=0.0, right=1.0, bottom=0.0, top=1.0)
plt.show(block=False)
plt.savefig("alaska_event_beachball.png", transparent=True, dpi=600)
# %%


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

    out.rotate("NE->RT", back_azimuth=baz)

    if isinstance(starttime, obspy.UTCDateTime) and isinstance(npts, int):
        out.interpolate(starttime=starttime, npts=npts, sampling_rate=1)

    out.taper(max_percentage=0.05, type="cosine")

    return out


slat = inv.select(network=network, station=station)[0][0].latitude
slon = inv.select(network=network, station=station)[0][0].longitude

dist_in_m, az, baz = gps2dist_azimuth(cmt.latitude, cmt.longitude, slat, slon)

# Bandpass
bp = [200, 500]

obs = process(
    raw,
    baz,
    inv=inv,
    remove_response=True,
    starttime=raw[0].stats.starttime,
    npts=10800,
    bandpass=bp,
)
is_syn = process(
    st, baz, inv=inv, starttime=raw[0].stats.starttime, npts=10800, bandpass=bp
)
rp_syn = process(prp, baz, starttime=raw[0].stats.starttime, npts=10800, bandpass=bp)

# %%
# obs.plot()


def traceL2(tr1, tr2, norm=True):
    if norm:
        return np.sum((tr1.data - tr2.data) ** 2) / np.sum(tr1.data**2)
    else:
        return 0.5 * np.sum((tr1.data - tr2.data) ** 2)


def diffstream(st1, st2):
    st = st1.copy()

    for tr in st:
        tr2 = st2.select(component=tr.stats.component)[0]
        tr.data = tr.data - tr2.data

    return st


# %%


# headerdict = dict(
#     event=cmt.eventname,
#     event_time=cmt.cmt_time,
#     event_latitude=cmt.latitude,
#     event_longitude=cmt.longitude,
#     event_depth_in_km=cmt.depth,
#     station=f"{network}.{station}",
#     station_latitude=slat,
#     station_longitude=slon,
#     station_azimuth=az,
#     station_back_azimuth=baz,
#     station_distance_in_degree=dist_in_m / 1000.0 / (40000 / 360.0),
#     location=6,
#     # fontsize='small'
# )

plotdict = dict(
    limits=(0, 10800),
    nooffset=False,
    components=["Z"],
    absmax=1.3e-4,
    event_origin_time=cmt.origin_time,
)
fig = plt.figure(figsize=(8, 4))


ax1 = plt.subplot(2, 1, 1)
plotseismogram(
    [obs, is_syn],
    ax=ax1,
    labels=["Observed", "Instaseis PREM 1D"],
    headerdict=headerdict,
    **plotdict,
)

ax1.spines["bottom"].set_visible(False)
ax1.tick_params(bottom=False, labelbottom=False)

misfit = traceL2(
    obs.select(component="Z")[0], is_syn.select(component="Z")[0], norm=True
)

putil.plot_label(
    ax1,
    f"L2_N: {misfit:4g}\nBP: {bp[0]:d}-{bp[1]:d}s",
    location=1,
    box=False,
    fontsize="small",
    dist=0.0,
)


ax2 = plt.subplot(2, 1, 2)
plotseismogram(
    [obs, rp_syn], ax=ax2, labels=["Observed", "SPECFEM3D_GLOBE GLAD-M25"], **plotdict
)

misfit = traceL2(
    obs.select(component="Z")[0], rp_syn.select(component="Z")[0], norm=True
)

putil.plot_label(
    ax2,
    f"L2_N: {misfit:4g}\nBP: {bp[0]:d}-{bp[1]:d}s",
    location=1,
    box=False,
    dist=0.0,
    fontsize="small",
)

plt.subplots_adjust(hspace=-0.2, bottom=0.125, top=0.85, left=0.05, right=0.95)

plt.show(block=False)
plt.savefig("alaska_event.pdf")
