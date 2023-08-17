import os
import numpy as np
import sys
import pickle
import shapefile
from gf3d.geoutils import geo2cart
from gf3d.coordinate.ingeopoly import ingeopoly
from gf3d.utils import downloadfile, unzip, downloadfile_progress


def msgeo(resolution=20):
    """Return the coordinates for plotting a sphere centered at (x,y,z)"""
    lon, lat = np.meshgrid(
        np.linspace(-180, 180, 2*resolution), np.linspace(-90, 90, resolution))
    return lon, lat


DIRNAME = f'.'


def read_land():
    fname = os.path.join(DIRNAME, 'DATA', 'ne_50m_land')
    zipname = os.path.join(DIRNAME, 'DATA', 'ne_50m_land.zip')
    url = 'https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/physical/ne_50m_land.zip'

    if os.path.exists(fname) is False:
        if os.path.exists(zipname) is False:
            print('Downloading landareas')
            try:
                # downloadfile(url, zipname)
                downloadfile_progress(url, zipname)
            except Exception as e:
                print(e)
                print('Download failed')
                sys.exit()
        print("Unzipping...")
        unzip(zipname, fname)

    land = shapefile.Reader(os.path.join(fname, 'ne_50m_land.shp'))

    return land


continents_file = './landareas.pkl'

if len(sys.argv) > 1:
    print('./landareas.pkl exists, but overwriting.')

elif os.path.exists(continents_file):
    print('use any commandline argument to overwrite.')
    sys.exit()
else:
    pass


gslon, gslat = msgeo(1440)
gsx, gsy, gsz = geo2cart(1.01, gslat, gslon)
flags = np.zeros_like(gsx, dtype=bool)

land = read_land()

nl = len(land.shapes())

xl = []
yl = []
zl = []

for _i, shape in enumerate(land.shapes()):

    lon, lat = zip(*shape.points)
    if len(lon) < 200:
        continue

    xt, yt, zt = geo2cart(1.01, lat, lon)

    # print(np.min(lon), np.min(lat))
    # print(np.max(lon), np.max(lat))

    # print(0, 0)
    # print(np.max(lon)-np.min(lon), np.max(lat)-np.min(lat))

    nflags = ingeopoly(lat, lon, gslat, gslon)

    if np.sum(nflags.astype(int)) == 0:
        continue

    # If more true values redo flags
    flags = np.where(nflags, nflags, flags)

    print(_i, 'of', nl, np.sum(flags.astype(int)))

    xl.extend([*xt, None])
    yl.extend([*yt, None])
    zl.extend([*zt, None])


# Get flags
coord = dict(
    x=np.where(flags, gsx, 'None').tolist(),
    y=np.where(flags, gsy, 'None').tolist(),
    z=np.where(flags, gsz, 'None').tolist(),
    xl=xl,
    yl=yl,
    zl=zl
)


with open(continents_file, 'wb') as f:

    pickle.dump(coord, f)
