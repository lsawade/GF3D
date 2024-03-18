# %%
import os
from gf3d.seismograms import get_seismograms
from gf3d.source import CMTSOLUTION
from gf3d.plot.seismogram import plotseismogram
from obspy import read
import matplotlib.pyplot as plt

# %%
network = 'II'
station = 'BFO'
maindir = '/Users/lucassawade/PDrive/Python/GF3D/dbfiles/'
st = read(os.path.join(
    maindir, f'RECIPROCAL_SPECFEMS/GLAD-M25/128/specfem3d_globe/OUTPUT_FILES/{network}.{station}.MX*.sem.sac'))

cmt = CMTSOLUTION.read(os.path.join(
    maindir, f'RECIPROCAL_SPECFEMS/GLAD-M25/128/specfem3d_globe/DATA/CMTSOLUTION'))
# %%


rp = get_seismograms(os.path.join(
    maindir, f'DB/GLAD-M25/128_single_test/{network}/{station}/{network}.{station}.h5'),
    cmt)


st.taper(max_percentage=0.05)
st.filter('bandpass', freqmin=1/1000, freqmax=1/40, zerophase=True)
st.interpolate(starttime=st[0].stats.starttime, sampling_rate=1, npts=10800)
rp.taper(max_percentage=0.05)
rp.filter('bandpass', freqmin=1/1000, freqmax=1/40, zerophase=True)
rp.interpolate(starttime=st[0].stats.starttime, sampling_rate=1, npts=10800)

plotseismogram(st, rp, cmt, nooffset=True)

plt.show(block=False)
