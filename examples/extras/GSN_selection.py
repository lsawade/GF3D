#!/bin/env python
# %%
"""

Retrieving Global Seismograph network metadata + extras
=======================================================

This example shows how to create a station lost for the database generation by
creating a list of stations from the Global Seismograph network and some extra
networks used in the GCMT analysis.

    "We use data from the IRIS-USGS Global Seismographic Network (seismic network
    codes II and IU) with additional data from sta- tions of the China Digital
    Seismograph Network (IC), the Geoscope (G), Mednet (MN), Geofon (GE), and
    Caribbean (CU) Networks."

-- from Ekström et al. 2012

"""
#%%

from obspy.clients.fdsn import Client
from obspy import Inventory, UTCDateTime


# %%
# First we'll just grab all the stations needed for GCMT analysis.

client = Client("IRIS")

starttime = UTCDateTime("1970-02-27T06:45:00.000")
endtime = UTCDateTime("2022-02-27T06:45:00.000000Z")

inventory = client.get_stations(
    network="II,IU,IC,G,MN,GE", station="*", location='00', channel="LH*",
    starttime=starttime,
    endtime=endtime,
    level='channel')

# %%
# Creating entry for each network/station combination.

counter = 0
net = []
sta = []
lat = []
lon = []
ele = []
bur = []
sen = []
for network in inventory:
    for station in network:
        print()
        print(f"{network.code}.{station.code}")
        print(80*"-")
        for channel in station:
            net.append(network.code)
            sta.append(station.code)
            _netsta = f"{network.code}.{station.code}"
            lat.append(channel.latitude)
            lon.append(channel.longitude)
            ele.append(channel.elevation)
            bur.append(channel.depth)
            sen.append(channel.sensor.description)
            print(
                f"{_netsta:<7s} {channel.code:<4s} -- ("
                f"{channel.latitude:9.4f}, "
                f"{channel.longitude:9.4f}, "
                f"{channel.elevation:9.4f}, "
                f"{channel.depth:9.4f}) -- "
                f"{channel.sensor.description}")

# %% Check for exotic sensors
fnet = []
fsta = []
fnetsta = set()
flat = []
flon = []
fele = []
fbur = []
fsen = []
print(f"# {'Net.Sta':<7s} -- ({'Lat':>9s}, {'Lon':>9s}, {'Elev':>9s}, {'Bur':>9s}) -- Sensor")
for _net, _sta, _lat, _lon, _ele, _bur, _sen in zip(net, sta, lat, lon, ele, bur, sen):
    _netsta = f"{_net}.{_sta}"

    # STS-2.5 only ever an extra sensor
    if 'STS-2.5' in _sen:
        continue
    if _netsta in fnetsta:
        continue

    print(
        f"  {_netsta:<7s} -- ({_lat:9.4f}, {_lon:9.4f}, {_ele:9.4f}, {_bur:9.4f}) -- {_sen}")
    fnet.append(_net)
    fsta.append(_sta)
    fnetsta.add(_netsta)
    flat.append(_lat)
    flon.append(_lon)
    fele.append(_ele)
    fbur.append(_bur)
    fsen.append(_sen)

#%% It's important to note here that I have gone through the works to check
# whether the sensors are substantially far away from each other. The largest
# distance I could find for different sensors at a single station was ~10m,
# which is simply irrelevant at global scale and long periods. So, the simple
# removal of all duplicates is fine. For regional studies at higher periods this
# may be non-negligible!
#
# It is also important that we keep track of topography here, but when we
# generate the GF database specfem automatically determines topography depending
# on the mesh. I.e., burial is the important value here.


# %%
with open('stations_gcmt.txt', 'w') as f:
    f.write('# Station list for the Reciprocal database\n')
    f.write('# Sensor information is only kept from the inventory. For most \n')
    f.write('# stations all sensors are at the same location.\n#\n')
    f.write(
        f"# {'Net.Sta':<7s}, {'Lat':>9s}, {'Lon':>9s}, {'Elev':>9s}, {'Bur':>9s}, Sensor\n")

    for _net, _sta, _lat, _lon, _ele, _bur, _sen in zip(fnet, fsta, flat, flon, fele, fbur, fsen):
        _netsta = f"{_net}.{_sta}"
        f.write(
            f"  {_netsta:<7s}, {_lat:9.4f}, {_lon:9.4f}, {_ele:9.4f}, {_bur:9.4f}, {_sen}\n")
