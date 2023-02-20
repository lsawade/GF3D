# %% Imports
# External
from obspy.clients.fdsn.header import URL_MAPPINGS
from glob import glob
from copy import deepcopy
import matplotlib.axes
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import numpy as np
from obspy import read, read_inventory, Stream
from obspy.geodetics.base import locations2degrees
import os
import typing as tp

# Internal
from lwsspy.GF.plot.util import plot_label
from lwsspy.GF.source import CMTSOLUTION
from lwsspy.GF.seismograms import SGTManager
from lwsspy.GF.process import process_stream, select_pairs, process_stream_trace_by_trace
from lwsspy.GF.plot.section import plotsection
from lwsspy.GF.plot.section_aligned import plotsection_aligned, get_azimuth_distance_traveltime, filter_stations
from lwsspy.seismo.download_waveforms_to_storage import download_waveforms_to_storage


# %% Files

# DB files
specfemmagic = '/scratch/gpfs/lsawade/SpecfemMagicTurkey'
h5files = os.path.join(specfemmagic, 'DB_test', '*', '*', '*.h5')

# CMTSOLUTION
cmt = CMTSOLUTION.read(
    '/home/lsawade/lwsspy/lwsspy.GF/scripts/TURKEY/DATA/CMTSOLUTION')

# %% Initialize the GF manager
sgt = SGTManager(glob(h5files)[:])
sgt.load_header_variables()
sgt.get_elements(cmt.latitude, cmt.longitude, cmt.depth, 1)

# %% Download data

# raw, inv = download_data(
#     cmt.origin_time,
#     duration=1800,

#     ,
#     location='*',
#     channel='*',
#     starttimeoffset=-300,
#     endtimeoffset=300
# )

datastorage = '/home/lsawade/lwsspy/lwsspy.GF/scripts/TURKEY/DATA/data'
starttime = cmt.origin_time - 300
endtime = cmt.origin_time + 1800 + 300
minimum_length = 0.9
reject_channels_with_gaps = False
network = 'KO,TU,MP,GO,II,IU,A2,GE,AB,HC,HP,HT,HL,RO'
station = ','.join(sgt.stations)
channel = None,
location = None,
providers = ['EIDA', 'IRIS', 'GEOFON', 'GFZ']  # list(URL_MAPPINGS.keys())
minlatitude = -90.0,
maxlatitude = 90.0,
minlongitude = -180.0,
maxlongitude = 180.0,
location_priorities = None  # ["00", "10", "20", "30"]
# ["LH[ZNE12]", "BH[ZNE12]", "HH[ZNE12]", "HH[ZNE12]"]
channel_priorities = None
limit_stations_to_inventory = None
waveform_storage = None
station_storage = None
logfile = '/home/lsawade/lwsspy/lwsspy.GF/scripts/TURKEY/DATA/downloadlog.txt'


download_waveforms_to_storage(
    datastorage,
    starttime,
    endtime,
    minimum_length=minimum_length,
    reject_channels_with_gaps=reject_channels_with_gaps,
    network=network,
    station=station,
    channel=None,
    location=None,
    providers=providers,
    minlatitude=-90.0,
    maxlatitude=90.0,
    minlongitude=-180.0,
    maxlongitude=180.0,
    location_priorities=location_priorities,
    channel_priorities=channel_priorities,
    limit_stations_to_inventory=None,
    waveform_storage=None,
    station_storage=None,
    logfile=None,
    threads_per_client=1
)

# %% Read downloaded data

raw = read(
    '/home/lsawade/lwsspy/lwsspy.GF/scripts/TURKEY/DATA/data/waveforms/*.mseed')
inv = read_inventory(
    '/home/lsawade/lwsspy/lwsspy.GF/scripts/TURKEY/DATA/data/stations/*.xml')

raw.merge(fill_value=0)

# %% Get a bunch of seismograms

# cmt = CMTSOLUTION.read('/scratch/gpfs/lsawade/SpecfemMagicGF/CMTSOLUTION')
rp = sgt.get_seismograms(cmt)


# %% Process
duration = 1000
bp = [15, 50]
obs = process_stream_trace_by_trace(raw, inv=inv, cmt=cmt,
                                    duration=duration, bandpass=bp,
                                    starttimeoffset=-120)
syn = process_stream(rp, cmt=cmt, duration=duration, bandpass=bp,
                     starttimeoffset=-120)

# %%

pobs, psyn = select_pairs(obs, syn)

# %% Plot section

starttime = psyn[0].stats.starttime
endtime = starttime + 800
limits = (starttime, endtime)

# Plots a section of observed and synthetic
plotsection(pobs, psyn, cmt, comp='Z', lw=0.75, limits=limits, scale=2.0)
plt.savefig('testsection_data.pdf', dpi=300)

# %%

window1 = (-75, 75)
window2 = (-50, 50)
phase1 = 'Rayleigh'
phase2 = 'Rayleigh'

# For pwaves
obs1, syn1 = get_azimuth_distance_traveltime(
    cmt, pobs, psyn, comp='Z',
    traveltime_window=(phase1, window1), vrayleigh=3.0)

obs2, syn2 = get_azimuth_distance_traveltime(
    cmt, pobs, psyn, comp='Z',
    traveltime_window=(phase2, window2), vrayleigh=3.0)

# %%
selection = filter_stations(obs1, obs2)

# %%
obs1 = Stream([obs1[_i] for _i in selection])
syn1 = Stream([syn1[_i] for _i in selection])
obs2 = Stream([obs2[_i] for _i in selection])
syn2 = Stream([syn2[_i] for _i in selection])
# %% Plot section

# Plots a section of observed and synthetic
plt.close('all')
fig = plt.figure(figsize=(6, 6))
ax = plt.subplot(1, 1, 1)
plotsection_aligned(
    obs1, syn1, cmt, comp='Z', lw=0.75, ax=ax,
    traveltime_window=(phase1, window1), labelright=True)
title = (f"{cmt.cmt_time.ctime()} "
         f"Loc: {cmt.latitude:.2f}dg, {cmt.longitude:.2f}dg, {cmt.depth:.1f}km"
         f" - BP: [15s, 200s]")
fig.suptitle(title, fontsize='small')
plt.subplots_adjust(left=0.1, right=0.825, bottom=0.1, top=0.9, wspace=0.1)
plt.savefig('station_section_align.pdf', dpi=300)
