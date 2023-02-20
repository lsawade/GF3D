# %%
from sys import argv


if __name__ == "__main__":

    if len(argv) == 1:
        print(
            """
    Converts a stations file to an nnodes workflow input file.
    Usage:

        python stations2nnodes.py STATIONS nnodes_file

    """
        )

    with open(argv[1], 'r') as f:

        lines = f.readlines()
        fnet = []
        fsta = []
        flat = []
        flon = []
        fele = []
        fbur = []
        fsen = []

        for _line in lines:

            _line = _line.strip()

            if _line[0] == '#':
                continue
            net, sta,
            _line
            _netsta = f"{_net}.{_sta}"
            f.write(
                f"  {_netsta:<7s}, {_lat:6.1f}, {_lon:6.1f}, {_ele:6.1f}, {_bur:6.1f}, {_sen}\n")

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
