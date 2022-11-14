# %%

from lwsspy.seismo import CMTCatalog

cat = CMTCatalog.load('gcmt_catalog_20220924.pkl')

# %%
# South America Bbox
bbox = [-89.296875, -57.515823, -27.421875, 14.774883]

# %% Define minmax dictionaries from bounding box
mindict = dict(latitude=bbox[2], longitude=bbox[0])
maxdict = dict(latitude=bbox[3], longitude=bbox[1])

# %% Filter the catalog

ncat, dropcat = cat.filter(mindict=mindict, maxdict=maxdict)

# %% Get latitude, longitude and depth

latitude = ncat.getvals('latitude')
longitude = ncat.getvals('longitude')
depth = ncat.getvals('depth_in_m')/1000.0

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


# %% Write CMT solution to directory

ncat.cmts2dir('./SA_subset')
