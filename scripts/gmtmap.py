


# Make obspy client for IRIS and get stations of network CU, II, IU, GE, and IC
from obspy.clients.fdsn import Client

client = Client("IRIS")
inv = client.get_stations(
    network="CU,II,IU,GE,IC",
)


# %%
# Get the locations of the stations in the inventory
import numpy as np

lons = []
lats = []
for net in inv:
    for sta in net:
        lons.append(sta.longitude)
        lats.append(sta.latitude)
lons = np.array(lons)
lats = np.array(lats)

# Get a list of the network codes matching the latitudes and longitudes
nets = []
for net in inv:
    for sta in net:
        nets.append(net.code)
nets = np.array(nets)


#%%

# Make a global GMT figure with the stations plotted with colors corresponding to their network code
import pygmt
fig = pygmt.Figure()
fig.coast(
    region="g",
    projection="W0/15c",
    frame=True,
    land="lightgray",
    # borders="1/thick",
    water="white",
    shorelines="thinnest",
)


# Make a list of pygmt color names to use for the networks
cmap = ["red", "green", "blue", "yellow", "purple"]

# Make the list of colors have 'light' in from of each name
# cmap = ["light" + c for c in cmap]

# Use the colors to plot each network
for net, c in zip(np.unique(nets), cmap):
    fig.plot(
        x=lons[nets == net],
        y=lats[nets == net],
        style="t0.15c",
        fill=c,
        pen="black",
        label=net,
    )

# Add a horizontal legend below the map
fig.legend(position="JBR+jBR+o0.2c", box="+gwhite+p1p")



fig.savefig("gmtmap_stations.pdf")

# %%

from gf3d.catalog.cmt import CMTCatalog

# Download the catalog
# cat = CMTCatalog.from_gcmt() # Already downloaded

# Load the catalog
cat = CMTCatalog.load('gf3d.gcmt.cat.pkl')

# %%
