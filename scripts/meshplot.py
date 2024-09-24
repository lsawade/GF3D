# %%
# LOAD THE KDTREE FIRST!!!!!!!!!!!!!
# If other packages are loaded, the wrong libstdc++ is picked up and
# doesn't contain the right GLIBCXX version

import os
import sys
from gf3d.plot.mesh import meshplot
from gf3d.utils import downloadfile, unzip
import requests
import shapefile
from gf3d.plot.util import plot_label
# Get file name

#%%
h5file = 'coord.h5'
DIRNAME = f'{os.getenv("HOME")}/PDrive/Python/Codes/GF3D/scripts'


def download_file(url, local_filename):
    # local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter below
    headers = {'user-agent': 'Mozilla/5.0'}

    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk:
                f.write(chunk)

    return local_filename


def read_land():
    fname = os.path.join(DIRNAME, 'DATA', 'ne_50m_land')
    print(fname)
    zipname = os.path.join(DIRNAME, 'DATA', 'ne_50m_land.zip')
    url = 'https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/physical/ne_50m_land.zip'
    if os.path.exists(fname) is False:
        if os.path.exists(zipname) is False:
            print('Downloading landareas')
            try:
                # downloadfile(url, zipname)
                download_file(url, zipname)
            except Exception as e:
                print(e)
                print('Download failed')
                sys.exit()
        print("Unzipping...")
        unzip(zipname, fname)

    land = shapefile.Reader(os.path.join(fname, 'ne_50m_land.shp'))

    return land


land = read_land()

# %%
meshplot(h5file, './outfile.html', land=None)




# %%
# Now using the the coordinates create histogram
import h5py
import numpy as np

with h5py.File(h5file, 'r') as db:
    xyz = db['xyz'][:]
    

#%%
def cart2geo(x,y,z):
    r = np.sqrt(x**2 + y**2 + z**2)
    lat = np.arcsin(z/r)
    lon = np.arctan2(y,x)
    return np.degrees(lat), np.degrees(lon)


lat, lon = cart2geo(xyz[:,0], xyz[:,1], xyz[:,2])


# %%
import matplotlib.pyplot as plt
import cartopy 
import cartopy.crs as ccrs

plt.figure(figsize=(10, 5))
data_crs = ccrs.PlateCarree()
ax = plt.axes(projection=ccrs.Mollweide())
my_cmap = plt.get_cmap('Greys')
# my_cmap.set_under('none')
# plot_label(ax, f"Total # of GLL points: {xyz.shape[0]}", location=1, dist=0.0, box=False)
plt.show(block=False)
latbins = np.linspace(-90, 90, 100)
lonbins = np.linspace(-180, 180, 200)

hex1 = ax.hexbin(lon, lat, cmap=my_cmap, vmax=None, gridsize=(200,100), transform=data_crs)
# hex1 = ax.hist2d(lon, lat, bins=(lonbins, latbins), cmap=my_cmap)[3]
ax.coastlines()
plt.colorbar(hex1, ax=ax, label=f'GLL points (N: {xyz.shape[0]})', pad=0.01)
plt.show()