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

catalogfile = '../DATA/gcmtcatalog_list_of_CMTSOLUTIONS.pkl'

with open(catalogfile, 'rb') as f:
    cmts = pickle.load(f)


cat = CMTCatalog(cmts)


# %%
# South America Bbox
bbox = [-89.296875, -47.421875, -57.515823, 14.774883]


#%%

# Filter the catalog parameter for parameter low and high value are optional
# their defaults are -inf and +inf, respectively.
ncat, _ = cat.filter('latitude', low=bbox[2], high=bbox[3])
ncat, _ = ncat.filter('longitude', low=bbox[0], high=bbox[1])

# %% Get latitude, longitude and depth

latitude = ncat.latitude
longitude = ncat.longitude
depth = ncat.depth

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

