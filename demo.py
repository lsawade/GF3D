# %%

# External
import os
from obspy import read

# Internal
from lwsspy.GF.source import CMTSOLUTION
from lwsspy.GF.seismograms import SGTManager
from lwsspy.GF.seismograms import get_seismograms, get_frechet
from lwsspy.GF.plot.compare_fw_reci import compare_fw_reci, plot_drp
# from lwsspy.GF.utils import blockPrint, enablePrint

# Define directory to save figures to
demodir = '/home/lsawade/lwsspy/lwsspy.GF/demo2'

# %% Define a CMTSOLUTION of interest
cmtfile = '/home/lsawade/lwsspy/lwsspy.GF/CMTSOLUTION'
cmt = CMTSOLUTION.read(cmtfile)
print(cmt)

# %%
db = '/scratch/gpfs/lsawade/SA_database'
stationfile = os.path.join(db, 'II', 'BFO.h5')
# stationfile = '/scratch/gpfs/lsawade/testdb.h5'
# %%
# Get reciprocal synthetics
# blockPrint()
rp = get_seismograms(stationfile, cmt)
# enablePrint()
print(rp)

# %%
# Get forward synthetics
fw = read(os.path.join(
    '/scratch/gpfs/lsawade/SpecfemMagicGF',
    'specfem3d_globe_forward', 'OUTPUT_FILES', '*.sac'))

# Simply used time offset for Timing offset (reciprocal is corrected)
for tr in fw:
    tr.stats.starttime -= 120

starttime = fw[0].stats.starttime + 300
endtime = starttime + 10000
limits = (starttime.datetime, endtime.datetime)

# %%
compare_fw_reci(cmt, rp, fw, limits, demodir)

# %% Compute Frechet's with respect to M

blockPrint()
drp = get_frechet(cmt, stationfile)
enablePrint()

drp

# %%

starttime = fw[0].stats.starttime + 100
endtime = starttime + 1000
limits = (starttime.datetime, endtime.datetime)

plot_drp(cmt, rp, drp, limits, demodir, comp='Z')


# %% First get subset of elements the

dbfiles = [stationfile]

SGTM = SGTManager(dbfiles)

# Get base
lat, lon, dep = cmt.latitude, cmt.longitude, cmt.depth

SGTM.load_header_variables()

# %%
blockPrint()
SGTM.get_elements(lat, lon, dep, k=10)  # TO BE CHANGED into radius.
enablePrint()

print('The strain subset has the shape')
SGTM.epsilon.shape

# %%
blockPrint()
s = SGTM.get_seismogram(cmt)
enablePrint()

# %%
compare_fw_reci(cmt, s, fw, limits, '.')
