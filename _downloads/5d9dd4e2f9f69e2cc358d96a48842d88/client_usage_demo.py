"""

First example of client usage to create subset.
-----------------------------------------------

Since the database will be very large, we almost never want to download the
entire thing to do some cmt inversions. So, the better workflow is the
following:

1. Create a regional subset using a server where the database is located
2. Download file from the server
3. Load the file with he ``GFManager`` or with the Fortran API ``read_GF``.
4. Extract Green functions

.. note::

    Note that this example cannot be run for the gallery since it requires a
    server to query from. As a result, all outputs are hand-written and may
    contain errors. Do note that the examples are created using the database
    files in the 'examples/DATA/single_element_read/DB' files.

"""
# %%
# Loading modules
from gf3d.client import GF3DClient
from gf3d.seismograms import GFManager
from gf3d.source import CMTSOLUTION

# %%
# GF3DClient
# ++++++++++
#
# The client automatically knows about server locations of a given database,
# i.e., they are hard coded.

gfcl = GF3DClient('example-db')


# %%
# With the client initialized, we can query a dictionary of general parameters

info = gfcl.get_info()
print(info)

# %%
# **OUTPUT:**
#
# .. code:: python
#
#    {'topography': True,
#     'ellipticity': True,
#     'nx_topo': 5400,
#     'ny_topo': 2700,
#     'res_topo': 4.0,
#     'nspl': 628,
#     'NSPEC': 1,
#     'NGLOB': 125,
#     'NGLLX': 5,
#     'NGLLY': 5,
#     'NGLLZ': 5,
#     'dt': 4.900000035000001,
#     'tc': 200.0,
#     'nsteps': 776,
#     'factor': 1e+17,
#     'hdur': 0.700000005,
#     'USE_BUFFER_ELEMENTS': False}
#


# %%
# We can also retrieve the stations available

stations = gfcl.stations_avail()
print(stations)

# %%
# **OUTPUT:**
#
# .. code:: python
#
#     ['IU.HRV', 'IU.ANMO', 'II.BFO']
#

# %%
# Retrieving a subset using the client
# ++++++++++++++++++++++++++++++++++++
#
# Now that we know that we can retrieve information from the server, we can
# query a subset quite easily as well

# Set query parameters
latitude = -31.1300
longitude = -72.0900
depth_in_km = 17.3500
radius_in_km = 100

# Make query
gfcl.get_subset('firstquery.h5', latitude=latitude, longitude=longitude,
                depth_in_km=depth_in_km, radius_in_km=radius_in_km)

# %%
# It'll take a minute for the server to create the dataset, and then the download
# should start and a a progress bar should show the progress.
# This downloads a regional subset of the database for all stations in the
# database. Let's load the downloaded subset using the Python API's GFManager

gfm = GFManager('firstquery.h5')
gfm.load()

# %%
# Now let's retrieve some seismograms using a CMTSOLUTION

# Read cmt
cmt = CMTSOLUTION.read('../DATA/single_element_read/CMTSOLUTION')

# Get seismos
rp = gfm.get_seismograms(cmt)

print(rp)

# %%
# **OUTPUT:**
#
# .. code:: python
#
#     9 Trace(s) in Stream:
#     IU.HRV..MXN  | 2015-09-16T22:51:12.900000Z - 2015-09-16T23:54:30.400027Z | 0.2 Hz, 776 samples
#     IU.HRV..MXE  | 2015-09-16T22:51:12.900000Z - 2015-09-16T23:54:30.400027Z | 0.2 Hz, 776 samples
#     IU.HRV..MXZ  | 2015-09-16T22:51:12.900000Z - 2015-09-16T23:54:30.400027Z | 0.2 Hz, 776 samples
#     IU.ANMO..MXN | 2015-09-16T22:51:12.900000Z - 2015-09-16T23:54:30.400027Z | 0.2 Hz, 776 samples
#     IU.ANMO..MXE | 2015-09-16T22:51:12.900000Z - 2015-09-16T23:54:30.400027Z | 0.2 Hz, 776 samples
#     IU.ANMO..MXZ | 2015-09-16T22:51:12.900000Z - 2015-09-16T23:54:30.400027Z | 0.2 Hz, 776 samples
#     II.BFO..MXN  | 2015-09-16T22:51:12.900000Z - 2015-09-16T23:54:30.400027Z | 0.2 Hz, 776 samples
#     II.BFO..MXE  | 2015-09-16T22:51:12.900000Z - 2015-09-16T23:54:30.400027Z | 0.2 Hz, 776 samples
#     II.BFO..MXZ  | 2015-09-16T22:51:12.900000Z - 2015-09-16T23:54:30.400027Z | 0.2 Hz, 776 samples
