scripts/bolivia_event_old.py# %%
# Get data from syngine
from gf3d.seismograms import GFManager
from obspy.clients.syngine import Client
from gf3d.source import CMTSOLUTION
import matplotlib.pyplot as plt
# from gf3d.plot.seismogram import plotseismogram, add_header
import obsplotlib.plot as opl
from gf3d.download import download_stream
import obspy
import numpy as np
from obspy.geodetics.base import gps2dist_azimuth
from gf3d.plot import util as putil
import matplotlib as mpl
mpl.rcParams["font.family"] = "monospace"

# %%
network = 'II'
station = 'BFO'

cmtfile = \
"""
PDE 1994  6  9  0 33 16.40 -13.8300  -67.5600 637.0 6.9 6.8 NORTHERN BOLIVIA
event name:     060994A
time shift:     29.0000
half duration:  20.0000
latitude:      -13.8200
longitude:     -67.2500
depth:         647.1000
Mrr:      -7.590000e+27
Mtt:       7.750000e+27
Mpp:      -1.600000e+26
Mrt:      -2.503000e+28
Mrp:       4.200000e+26
Mtp:      -2.480000e+27
"""

cmt = CMTSOLUTION.read(cmtfile)


# %%
# Download Syngine synthetics

client = Client()
models = client.get_available_models()

st = client.get_waveforms(model='prem_a_10s', network=network, station=station,
                          eventid=f"GCMT:{cmt.eventname}",
                          starttime=cmt.origin_time,
                          endtime=cmt.origin_time+10800)

# %%
# Download observed data
# Data not available at this station
raw, inv = download_stream(cmt.origin_time + 10*365*24*3600, 10800,
                           network=network, station=station,
                           location="*", channel="BH*",)

# %%

gfm = GFManager('/lustre/orion/geo111/scratch/lsawade/subset_bolivia.h5')
gfm.load()
rp = gfm.get_seismograms(cmt)
prp = rp.select(network=network, station=station)

# %%


def process(st, baz, inv: obspy.Inventory | None = None, remove_response=False,
            starttime=None, npts=None, sampling_rate=1,
            bandpass=[200, 500]):

    out = st.copy()
    out.detrend('demean')
    out.detrend('linear')
    out.taper(max_percentage=0.05, type='cosine')

    if inv is not None and remove_response:
        out.remove_response(inventory=inv, output="DISP",
                            pre_filt=(0.001, 0.005, 0.1, 0.2))

    if inv is not None:
        out.rotate('->ZNE', inventory=inv)

    out.filter('bandpass',
               freqmin=1 / bandpass[1], freqmax=1 / bandpass[0],
               corners=2, zerophase=True)

    out.rotate('NE->RT', back_azimuth=baz)

    if isinstance(starttime, obspy.UTCDateTime) \
            and isinstance(npts, int):
        out.interpolate(starttime=starttime, npts=npts, sampling_rate=1)

    out.taper(max_percentage=0.05, type='cosine')

    return out


slat = inv.select(network=network, station=station)[0][0].latitude
slon = inv.select(network=network, station=station)[0][0].longitude

dist_in_m, az, baz = gps2dist_azimuth(
    cmt.latitude, cmt.longitude, slat, slon)

# Bandpass
bp = [17, 500]

# obs = process(raw, baz, inv=inv, remove_response=True,
#               starttime=raw[0].stats.starttime, npts=10800,
#               bandpass=bp)
is_syn = process(st, baz, inv=inv,
                 starttime=st[0].stats.starttime, npts=10800,
                 bandpass=bp)
rp_syn = process(prp, baz, starttime=st[0].stats.starttime, npts=10800,
                 bandpass=bp)

# %%
# obs.plot()


def traceL2(tr1, tr2, norm=True):
    if norm:
        return np.sum((tr1.data - tr2.data)**2) / np.sum(tr1.data**2)
    else:
        return 0.5 * np.sum((tr1.data - tr2.data)**2)


def diffstream(st1, st2):
    st = st1.copy()

    for tr in st:
        tr2 = st2.select(component=tr.stats.component)[0]
        tr.data = tr.data - tr2.data

    return st

# %%


headerdict = dict(
    event=cmt.eventname,
    event_time=cmt.cmt_time,
    event_latitude=cmt.latitude,
    event_longitude=cmt.longitude,
    event_depth_in_km=cmt.depth,
    station=f"{network}.{station}",
    station_latitude=slat,
    station_longitude=slon,
    station_azimuth=az,
    station_back_azimuth=baz,
    station_distance_in_degree=dist_in_m/1000.0/(40000/360.0),
    location=6,
    # fontsize='small'
)

plotdict = dict(
    limits=(0, 10800),
    nooffset=False,
    components=['Z'],
    absmax=1.3e-4,
    event_origin_time=cmt.origin_time
)
fig = plt.figure(figsize=(8, 4))


ax1 = plt.subplot(1, 1, 1)
opl.trace([is_syn.select(comp), rp_syn], ax=ax1, labels=['Observed', 'Instaseis PREM 1D'],
               headerdict=headerdict, **plotdict)

# ax1.spines['bottom'].set_visible(False)
# ax1.tick_params(bottom=False, labelbottom=False)

misfit = traceL2(is_syn.select(component='Z')[
                 0], rp_syn.select(component='Z')[0], norm=True)

putil.plot_label(ax1, f'L2_N: {misfit:4g}\nBP: {bp[0]:d}-{bp[1]:d}s',
                 location=1, box=False, fontsize='small',  dist=0.0)


# ax2 = plt.subplot(2, 1, 2)
# plotseismogram([obs, rp_syn], ax=ax2,
#                labels=['Observed', 'SPECFEM3D_GLOBE GLAD-M25'],
#                **plotdict)

# misfit = traceL2(obs.select(component='Z')[
#                  0], rp_syn.select(component='Z')[0], norm=True)

# putil.plot_label(ax2, f'L2_N: {misfit:4g}\nBP: {bp[0]:d}-{bp[1]:d}s',
#                  location=1, box=False, dist=0.0,
#                  fontsize='small')

plt.subplots_adjust(hspace=-0.2, bottom=0.125, top=.85, left=0.05, right=0.95)

plt.show(block=False)
plt.savefig('bolivia_event.pdf')
