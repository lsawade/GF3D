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
# Get file name

h5file = sys.argv[1]
DIRNAME = f'{os.getenv("HOME")}/GF3D/scripts'


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
meshplot(h5file, './outfile.html', land=land)

# %%
