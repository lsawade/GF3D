# %%
# Get data from syngine
from gf3d.seismograms import GFManager
from obspy.clients.syngine import Client
from gf3d.source import CMTSOLUTION
import matplotlib.pyplot as plt
from gf3d.plot.seismogram import plotseismogram
from gf3d.download import download_stream

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

client = Client()

# %%
models = client.get_available_models()

# %%
st = client.get_waveforms(model='prem_a_10s', network="II", station="ARU",
                          eventid=f"GCMT:{cmt.eventname}",
                          starttime=cmt.origin_time,
                          endtime=cmt.origin_time+10800)
st.filter('lowpass', freq=1/40, corners=2, zerophase=True)
# %%

obs, inv = download_stream(cmt.origin_time, 10800,
                           network="II", station="ARU", channel="LH*",)

# %%
obs.detrend('demean')
obs.detrend('linear')
obs.taper(max_percentage=0.05, type='cosine')
obs.remove_response(inventory=inv, output="DISP",
                    pre_filt=(0.001, 0.005, 0.1, 0.2))
obs.filter('lowpass', freq=1/40, corners=2, zerophase=True)
obs.rotate('->ZNE', inventory=inv)
# obs.plot()
# %%

gfm = GFManager('subset.h5')
gfm.load()

# %%

rp = gfm.get_seismograms(cmt)
prp = rp.select(network='II', station='ARU')
prp.filter('lowpass', freq=1/40, corners=2, zerophase=True)

plotseismogram(obs, st, cmt, newsyn=prp)
plt.show()
