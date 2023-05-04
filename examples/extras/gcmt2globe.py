#!/bin/env python
# %%
"""

Filtering the GCMT Catalog for South America
============================================

This example will go over the creation of the subset catalog used for the
AGU presentation in 2022. A regional Green Function catalog.

"""

import pickle
from gf3d.catalog.cmt import CMTCatalog

# %%
# Loading a list of sources

catalogfile = '../DATA/gcmtcatalog_list_of_CMTSOLUTIONS.pkl'

with open(catalogfile, 'rb') as f:
    cmts = pickle.load(f)

#%%
# Making a catalog out of it
cat = CMTCatalog(cmts)

# %%
# Get latitude, longitude and depth

latitude = cat.latitude
longitude = cat.longitude
depth = cat.depth

# %% Write catalog to GF_LOCATIONS FILE

# Open GF locations file for each compenent
locations_file = 'GF_LOCATIONS'

with open(locations_file, 'w') as f:

    # Loop over provided target locations
    for _lat, _lon, _dep in zip(
            latitude,
            longitude,
            depth):
        f.write(f'{_lat:9.4f}   {_lon:9.4f}   {_dep:9.4f}\n')

# %%
# Just a small check on much storage the Global 17s database would use.

x = [100753, 160553, 251777, 368535, 294368, 305935,
176917, 246136, 138208, 12263, 382244, 227071,
404337, 224512, 578171, 341973, 124549, 230091,
121430, 150595, 83643, 204260, 187607, 99032]

NGLOB = sum(x)


# If we divide the element into 8 elements (128--> 256) we have 8 new interior
# points per new element, 8 * 4 on the interior boundaries and then 6 * 4 * 4 new
# GLL points on the external element boundaries
new_gll_per_element = 8 * 8 + 8 * 4 + 6 * 4 * 4
GLL128 = sum(x)
NEL128 = 73682

GLL256 = GLL128 + NEL128 * new_gll_per_element