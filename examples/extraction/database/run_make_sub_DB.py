"""
Create new sub DB from database files
-------------------------------------

"""
# sphinx_gallery_dummy_images = 1

# %%

# External
from glob import glob

# Internal
from gf3d.source import CMTSOLUTION
from gf3d.seismograms import GFManager

# %%
# Read the event data

# CMTSOLUTION
cmt = CMTSOLUTION.read('../../DATA/single_element_read/CMTSOLUTION')


# %%
# Read database files
gfm = GFManager(glob('../../DATA/single_element_read/DB/*/*/*.*.h5'))
gfm.load_header_variables()
gfm.write_DB_directIO('tempDB', cmt.latitude, cmt.longitude, cmt.depth,
                      dist_in_km=125, NGLL=3, duration=3600.0)

# %%
# Load a subset

gfsub = GFManager(glob('tempDB/*/*/*.*.h5'))
gfsub.load_header_variables()

# %%
# Finally you can read the seismograms.
rp = gfsub.get_seismograms(cmt)
print(rp)
