# %%

from gf3d.seismograms import GFManager
from gf3d.source import CMTSOLUTION

# %%

cmtfile = \
"""
PDE 1994  6  9  0 33 16.40 -13.8300  -67.5600 637.0 6.9 6.8 NORTHERN BOLIVIA
event name:     C060994A
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

gfm = GFManager('/lustre/orion/geo111/scratch/lsawade/subset_bolivia.h5')
gfm.load()

# %%
rp = gfm.get_seismograms(cmt)





# %%
from gf3d.plot.seismogram import plotseismogram, add_header


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

