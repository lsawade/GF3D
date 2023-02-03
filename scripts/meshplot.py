# %%
# LOAD THE KDTREE FIRST!!!!!!!!!!!!!
# If other packages are loaded, the wrong libstdc++ is picked up and
# doesn't contain the right GLIBCXX version

import os
from lwsspy.GF.plot.mesh import meshplot
from lwsspy.GF.utils import downloadfile, unzip
import shapefile
# Get file name
specfemmagic = '/scratch/gpfs/lsawade/SpecfemMagicGF'
stationdir = os.path.join(specfemmagic, 'DB', 'II', 'BFO')
h5file = os.path.join(stationdir,  'II.BFO.h5')
DIRNAME = f'{os.getenv("HOME")}/lwsspy/lwsspy.GF/scripts'


def read_land():
    fname = os.path.join(DIRNAME, 'DATA', 'ne_50m_land')
    zipname = os.path.join(DIRNAME, 'DATA', 'ne_50m_land.zip')
    url = 'https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/physical/ne_50m_land.zip'

    if os.path.exists(fname) is False:
        if os.path.exists(zipname) is False:
            print('Downloading landareas')
            downloadfile(url, zipname)
        print("Unzipping...")
        unzip(zipname, fname)

    land = shapefile.Reader(os.path.join(fname, 'ne_50m_land.shp'))

    return land

land = read_land()

# %%
meshplot(h5file, './outfile.html', land=land)

# %%
