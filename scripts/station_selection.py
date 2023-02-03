# %%
from obspy.clients.fdsn import Client
from obspy import Inventory, UTCDateTime


# %%
client = Client("IRIS")

starttime = UTCDateTime("1970-02-27T06:45:00.000")
endtime = UTCDateTime("2022-02-27T06:45:00.000000Z")

inventory = client.get_stations(
    network="II,IU", station="*", location='00', channel="LH*",
    starttime=starttime,
    endtime=endtime, latitude=-16, longitude=-60,
    minradius=25, maxradius=180, level='channel')

# %%
# By eye geographical selection:
sel = ['II.AAK',
       'II.BFO',
       'II.BORG',
       'II.DGAR',
       'II.HOPE',
       'II.KAPI',
       'II.KDAK',
       'IU.ANMO',
       'IU.CTAO',
       'IU.DAV',
       'IU.HRV',
       'IU.KOWA',
       'IU.MAJO',
       'IU.NAI',
       'IU.SBA',
       'IU.SNZO',
       'IU.XMAS',
       'IU.YAK']


subinv = inventory.select(channel="*Z")

counter = 0
net = []
sta = []
lat = []
lon = []
ele = []
bur = []
sen = []
for network in subinv:
    for station in network:
        if f"{network.code}.{station.code}" in sel:
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
                    f"{_netsta:<7s} -- ("
                    f"{channel.latitude:6.1f}, "
                    f"{channel.longitude:6.1f}, "
                    f"{channel.elevation:6.1f}, "
                    f"{channel.depth:6.1f}) -- "
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
print(f"# {'Net.Sta':<7s} -- ({'Lat':>6s}, {'Lon':>6s}, {'Elev':>6s}, {'Bur':>6s}) -- Sensor")
for _net, _sta, _lat, _lon, _ele, _bur, _sen in zip(net, sta, lat, lon, ele, bur, sen):
    _netsta = f"{_net}.{_sta}"
    if 'STS-6' in _sen:
        continue
    if 'Trillium' in _sen:
        continue
    if 'STS-2.5' in _sen:
        continue
    if _netsta in fnetsta:
        continue

    print(
        f"  {_netsta:<7s} -- ({_lat:6.1f}, {_lon:6.1f}, {_ele:6.1f}, {_bur:6.1f}) -- {_sen}")
    fnet.append(_net)
    fsta.append(_sta)
    fnetsta.add(_netsta)
    flat.append(_lat)
    flon.append(_lon)
    fele.append(_ele)
    fbur.append(_bur)
    fsen.append(_sen)

# %%
with open('stations.txt', 'w') as f:
    f.write('# Station list for the Reciprocal database\n')
    f.write('# Sensor information is only kept from the inventory. For most \n')
    f.write('# stations all sensors are at the same location.\n#\n')
    f.write(
        f"# {'Net.Sta':<7s}, {'Lat':>6s}, {'Lon':>6s}, {'Elev':>6s}, {'Bur':>6s}, Sensor\n")

    for _net, _sta, _lat, _lon, _ele, _bur, _sen in zip(fnet, fsta, flat, flon, fele, fbur, fsen):
        _netsta = f"{_net}.{_sta}"
        f.write(
            f"  {_netsta:<7s}, {_lat:6.1f}, {_lon:6.1f}, {_ele:6.1f}, {_bur:6.1f}, {_sen}\n")
