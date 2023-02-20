# %%
import os
import numpy as np
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from lwsspy.seismo import CMTCatalog
from lwsspy.GF.utils import filedir
from lwsspy.GF.coordinate.Ra2b import Ra2b


DIRNAME = f'{os.getenv("HOME")}/lwsspy/lwsspy.GF/scripts/TURKEY'
# %%

# %% Locations of andular chunk
ANGULAR_WIDTH = 60.0
ANGULAR_HEIGHT = 60.0
CENTER_LATITUDE = 30.0
CENTER_LONGITUDE = 45.0

iminlat = -ANGULAR_HEIGHT/2.0
imaxlat = +ANGULAR_HEIGHT/2.0
iminlon = -ANGULAR_WIDTH/2.0
imaxlon = +ANGULAR_WIDTH/2.0
corners = np.array([
    [iminlat, iminlon],
    [iminlat, imaxlon],
    [imaxlat, imaxlon],
    [imaxlat, iminlon]])

rcorner = corners*np.pi/180.0
rintermediate = np.array([[0, CENTER_LONGITUDE], ])*np.pi/180.0
rcenter = np.array([[CENTER_LATITUDE, CENTER_LONGITUDE], ])*np.pi/180.0


print(rcenter)
print(rcorner)
# %% Rotate the points

# Points around x axis


def sphere2cart(thetaphi):
    r = 1.0
    x = r*np.cos(thetaphi[:, 0])*np.cos(thetaphi[:, 1])
    y = r*np.cos(thetaphi[:, 0])*np.sin(thetaphi[:, 1])
    z = r*np.sin(thetaphi[:, 0])*np.ones_like(thetaphi[:, 1])

    return np.vstack((x, y, z)).T


# Convert to cartesian
xyz_corner = sphere2cart(rcorner)
xyz_intermediate = sphere2cart(rcenter)
xyz_center = sphere2cart(rcenter)

# Rotate from Z-axis to other points
R1 = Ra2b((1, 0, 0), xyz_center[0, :])
R2 = Ra2b(xyz_center[0, :], xyz_center[0, :])

xyz_ncorner = (R2 @ R1 @ xyz_corner.T).T


def cart2sphere(xyz):
    r = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2 + xyz[:, 2]**2)
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])

    # Catch phis below 0
    # phi = np.where(phi < 0, phi + 2*np.pi, phi)

    # Catch corner case of latitude computation
    theta = np.where((r == 0), 0, np.arcsin(xyz[:, 2]/r))

    return np.vstack((theta, phi)).T


# New corners
nrcorner = cart2sphere(xyz_ncorner)
ncorner = nrcorner/np.pi*180.0
ncorner[:, 0] = ncorner[:, 0]

print(ncorner)

#  ! corner            1
#  ! longitude in degrees =    18.434948822922006
#  ! latitude in degrees =    5.7679995259277172E-009
#  !
#  ! corner            2
#  ! longitude in degrees =    71.565051177077976
#  ! latitude in degrees =    5.7679995259277172E-009
#  !
#  ! corner            3
#  ! longitude in degrees =    5.7295779454167022E-006
#  ! latitude in degrees =    50.955917956667584
#  !
#  ! corner            4
#  ! longitude in degrees =    89.999999990940751
#  ! latitude in degrees =    50.955917956667584

# %%


class GD(ccrs.Geodetic):
    @property
    def threshold(self):
        return 0.00001


class PL(ccrs.PlateCarree):
    @property
    def threshold(self):
        return 0.00001


prj = ccrs.Orthographic(
    central_longitude=CENTER_LONGITUDE, central_latitude=CENTER_LATITUDE)
plt.figure()
ax = plt.axes(projection=prj)
prj.threshold = 100.0
plt.plot(
    np.append(corners[:, 1], corners[0, 1]),
    np.append(corners[:, 0], corners[0, 0]),
    '-ro', transform=ccrs.Geodetic())
plt.plot(
    np.append(ncorner[:, 1], ncorner[0, 1]),
    np.append(ncorner[:, 0], ncorner[0, 0]),
    '-bo', transform=ccrs.Geodetic())

ax.add_feature(cartopy.feature.LAND, zorder=-2, edgecolor='black',
               linewidth=0.5, facecolor=(0.9, 0.9, 0.9))
ax.set_global()
plt.savefig('testrot.png')
# %%


cat = CMTCatalog.load(os.path.join(
    DIRNAME, '../DATA/gcmt_catalog_20220924.pkl'))


# South America Bbox

bbox = [-89.296875, -47.421875, -57.515823, 14.774883]

# %% Define minmax dictionaries from bounding box
mindict = dict(latitude=bbox[2], longitude=bbox[0])
maxdict = dict(latitude=bbox[3], longitude=bbox[1])

# %%
cat, _ = cat.filter(mindict=mindict, maxdict=maxdict)

# %%


plt.figure(figsize=(5, 10))

mapax = plt.axes(projection=ccrs.Orthographic(
    central_longitude=sum(bbox[:2])/2, central_latitude=sum(bbox[2:])/2))
mapax.set_extent(bbox)

mapax.spines['geo'].set_visible(False)

cat.plot(ax=mapax)

plt.savefig('eventmap.pdf', dpi=300)
