"""
Create subset from database files
---------------------------------

"""
# sphinx_gallery_dummy_images = 1

# %%

# External
import sys
from glob import glob

# Internal
from gf3d.source import CMTSOLUTION
from gf3d.seismograms import GFManager

# %%
# Read the event data

# CMTSOLUTION
cmt = CMTSOLUTION.read('examples/DATA/single_element_read/CMTSOLUTION')


# %%
# Read database files
gfm = GFManager(glob('/gpfs/alpine/geo111/scratch/lsawade/DB/GLAD-M25/128/*/*/*.*.h5'))
gfm.load_header_variables()


#%%
gfm.get_elements(cmt.latitude, cmt.longitude,
                 cmt.depth, dist_in_km=100, NGLL=5)


# %% Write a subset
gfm.write_subset('temp_subset.h5', duration=14400.0, fortran=True)

sys.exit()
# %%
# Load a subset

gfsub = GFManager('temp_subset.h5')
gfsub.load()

# %%
# Finally you can read the seismograms.
rp = gfsub.get_seismograms(cmt)
print(rp)
