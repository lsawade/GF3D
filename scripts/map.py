# %%
from obspy.imaging.mopad_wrapper import beach as bb2
from obspy.imaging.beachball import beach as bb1
import numpy as np
import io
import requests
import os
import matplotlib
import matplotlib.pyplot as plt
import cartopy
from cartopy.crs import Orthographic, PlateCarree, Geodetic
from lwsspy.GF.source import CMTSOLUTION
from lwsspy.GF.seismograms import get_seismograms
from lwsspy.GF.geoutils import geomidpointv

# Get station dir
specfemmagic = '/scratch/gpfs/lsawade/SpecfemMagicGF'
stationdir = os.path.join(specfemmagic, 'DB', 'II', 'BFO')
h5file = os.path.join(stationdir,  'II.BFO.h5')

# %% Get CMTSOLUTION
cmt = CMTSOLUTION.read('CMTSOLUTION')

# %% Get seismograms
st = get_seismograms(h5file, cmt)

# rlat, rlon
rlat = st[0].stats.latitude
rlon = st[0].stats.longitude

# %%
mlat, mlon = geomidpointv(rlat, rlon, cmt.latitude, cmt.longitude)

# Projection
projection = Orthographic(
    central_longitude=mlon, central_latitude=mlat)

# %%
plt.figure(figsize=(2.0, 4))


mapax = plt.subplot(2, 1, 1, projection=projection)
mapax.set_global()
mapax.add_feature(cartopy.feature.LAND, zorder=-1, edgecolor='k',
                  linewidth=0.25, facecolor=(0.9, 0.9, 0.9))
mapax.gridlines(lw=0.25, ls='-', color=(0.75, 0.75, 0.75), zorder=-1)

mapax.plot(
    rlon, rlat, 'v',
    markerfacecolor=(0.8, 0.2, 0.2), markersize=9,
    markeredgecolor='k', transform=PlateCarree())

mapax.plot(
    cmt.longitude, cmt.latitude, '*',
    markerfacecolor=(0.2, 0.2, 0.8), markersize=12,
    markeredgecolor='k', transform=PlateCarree())

mapax.plot(
    [cmt.longitude, rlon], [cmt.latitude, rlat], '-k',
    transform=Geodetic(), zorder=0)

ax = plt.subplot(2, 1, 2)

cmt.axbeach(ax, 72, 72, 110, clip_on=False, zorder=10, linewidth=1)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
plt.subplots_adjust(hspace=0.0, left=0.0, right=1.0, bottom=0.0, top=1.0)
plt.savefig('map.pdf')
