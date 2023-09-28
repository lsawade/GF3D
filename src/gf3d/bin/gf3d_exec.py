#!/usr/bin/env python

"""

run using `$ gf3d COMMAND [SUBCOMMAND [SUBSUBCOMMAND]] [OPTIONS] ARG1 ARG2` etc.

GF3D CLI structure:

gf3d
- database
    - query
        - info
        - stations
        - subset
        - extract NOT IMPLEMENTED
    - extract NOT IMPLEMENTED
- subset
    - info
    - stations
    - extract

Make sure that for the query you have access to the server.

ssh -o "ServerAliveInterval=60" -N -f -F 'none' -L 5000:127.0.0.1:5000 lsawade@vrientius.princeton.edu

Call signature to query synthetic from a subsetfile:

gf3d subset extract -- SUBSETFILENAME YYYY MM DD HH MM SS MRR MTT MPP MRT MRP MTP LATITUDE LONGITUDE DEPTH
                             TIME_SHIFT HDUR ITYPSOKERN OUTDIR

Example:
gf3d extract subset -- subset.h5 2015 9 16 22 54 32.90 \
                            1.950000e+28 -4.360000e+26 -1.910000e+28 \
                            7.420000e+27 -2.480000e+28 9.420000e+26 \
                            -31.1300 -72.0900 17.3500 49.9800 33.4000 \
                            2 OUTPUT

"""
from gf3d.plot.section import plotsection
from gf3d.seismograms import GFManager
from gf3d.source import CMTSOLUTION
import matplotlib.pyplot as plt
import click


@click.command()
@click.option('--count', default=1, help='number of greetings')
@click.argument('name')
def subhello(count, name):
    for i in range(count):
        print(f"{i}. Hello {name}")


@click.group()
def cli():
    pass


@cli.group()
def database():
    '''Interface to a GF3D database'''
    pass


@database.group()
def query():
    '''Query info and data from a Green function database hosted with a flask server.'''
    pass


@query.command(name='info')
@click.argument('databasename', type=str)
@click.option('--debug',  is_flag=True, show_default=True, default=False, help='Only print query url', type=bool)
def query_info(databasename: str, debug: bool = False):
    '''Query info from a hosted database server.'''
    from gf3d.client import GF3DClient
    from pprint import pprint
    gfcl = GF3DClient(databasename, debug=debug)
    info = gfcl.get_info()
    if info:
        pprint(info)


@query.command(name='stations')
@click.argument('databasename', type=str)
@click.option('--debug',  is_flag=True, show_default=True, default=False, help='Only print query url', type=bool)
@click.option('--local', is_flag=True, default=False, help='Makes subset from local database databasename -> database root.', type=bool)
def query_stations(databasename: str, debug: bool, local: bool):
    """Query available stations from a hosted database server."""

    if local:

        import os
        from glob import glob
        from gf3d.seismograms import GFManager

        # Check for files given database path
        db_globstr = os.path.join(databasename, '*', '*', '*.h5')

        # Get all files
        db_files = glob(db_globstr)

        # Check if there are any files
        if len(db_files) == 0:
            raise ValueError(f'No files found in {database} directory. '
                             'Please check path.')

        if debug:
            print('Found files:')
            for file in db_files:
                print(file)
        else:
            # Get subset
            GFM = GFManager(db_files)
            GFM.load_header_variables()

            # Get station info:
            stations = GFM.get_stations()
            Nsta = len(stations)

            # Print copyable string array
            endofline = '\n'

            print('[', end="")
            for _i, (net, sta, lat, lon, bur) in enumerate(stations):

                # Remove newline character for the last station
                if _i == Nsta-1:
                    endofline = ''
                print(f"['{net}', '{sta}', {lat}, {lon}, {bur}],",
                      end=endofline)

            print(']')
    else:

        from gf3d.client import GF3DClient
        gfcl = GF3DClient(databasename, debug=debug)
        stations = gfcl.stations_avail()
        if stations:
            for station in stations:
                print(station)


@query.command(name='extract')
def query_extract():
    """Directly extract set of traces."""
    print('NOT YET IMPLEMENTED.')


@query.command(name='subset')
@click.argument('databasename', type=str)
@click.argument('subsetfilename', type=str)
@click.argument('latitude', type=float)
@click.argument('longitude', type=float)
@click.argument('depth_in_km', type=float)
@click.argument('radius_in_km', type=float)
@click.option('--fortran', is_flag=True, default=False, help='Return Fortran ordered subset.', type=bool)
@click.option('--ngll', default=5, help='Number of GLL points 5 or 3', type=int)
@click.option('--netsta', default=None, help='Station subselection. NOT IMPLEMENTED', type=str)
@click.option('--debug',  is_flag=True, show_default=True, default=False, help='Only print query url', type=bool)
@click.option('--local', is_flag=True, default=False, help='Makes subset from local database databasename -> database root.', type=bool)
@click.option('--nothreading', is_flag=True, default=False, help='When loading the data, not using joblib. Only if --local set.', type=bool)
def query_subset(
        databasename: str,
        subsetfilename: str,
        latitude: float,
        longitude: float,
        depth_in_km: float,
        radius_in_km: float = 100,
        ngll: int = 5,
        netsta: list | None = None,
        fortran: bool = False,
        local: bool = False,
        debug: bool = False,
        nothreading: bool = False):
    """Query a subset from a hosted database server.

    IMPORTANT: For negative latitudes and longitudes use following setup:

        gf3d query subset [--option = value] - - DATABASENAME SUBSETFILENAME LATITUDE ...

    """

    if local:

        import os
        from glob import glob
        from gf3d.seismograms import GFManager

        # Check for files given database path
        db_globstr = os.path.join(databasename, '*', '*', '*.h5')

        # Get all files
        db_files = glob(db_globstr)

        # Check if there are any files
        if len(db_files) == 0:
            raise ValueError(f'No files found in {database} directory. '
                             'Please check path.')

        if debug:
            print('Found files:')
            for file in db_files:
                print(file)
        else:
            # Get subset
            GFM = GFManager(db_files)
            GFM.load_header_variables()
            GFM.get_elements(
                latitude, longitude, depth_in_km, radius_in_km, NGLL=ngll,
                threading=not nothreading)
            GFM.write_subset(subsetfilename, fortran=fortran)

    else:
        from gf3d.client import GF3DClient

        gfcl = GF3DClient(databasename, debug=debug)
        gfcl.get_subset(
            subsetfilename,
            latitude=latitude,
            longitude=longitude,
            depth_in_km=depth_in_km,
            radius_in_km=radius_in_km,
            NGLL=ngll,
            netsta=netsta,
            fortran=fortran)


@query.command(name='subset-feature')
@click.argument('databasename', type=str)
@click.argument('subsetfilename', type=str)
@click.argument('latitude', type=float)
@click.argument('longitude', type=float)
@click.argument('depth_in_km', type=float)
@click.argument('radius_in_km', type=float)
@click.option('--fortran', is_flag=True, default=False, help='Return Fortran ordered subset.', type=bool)
@click.option('--ngll', default=5, help='Number of GLL points 5 or 3', type=int)
@click.option('--netsta', default=None, help='Station subselection. NOT IMPLEMENTED', type=str)
@click.option('--debug',  is_flag=True, show_default=True, default=False, help='Only print query url', type=bool)
@click.option('--nothreading', is_flag=True, default=False, help='When loading the data, not using joblib. Only if --local set.', type=bool)
def query_subset_feature(
        databasename: str,
        subsetfilename: str,
        latitude: float,
        longitude: float,
        depth_in_km: float,
        radius_in_km: float = 100,
        ngll: int = 5,
        netsta: list | None = None,
        fortran: bool = False,
        debug: bool = False,
        nothreading: bool = False):
    """Query a subset from a hosted database server.

    IMPORTANT: For negative latitudes and longitudes use following setup:

        gf3d query subset [--option = value] - - DATABASENAME SUBSETFILENAME LATITUDE ...

    """

    import os
    from glob import glob
    from gf3d.seismograms import GFManager

    # Check for files given database path
    db_globstr = os.path.join(databasename, '*', '*', '*.h5')

    # Get all files
    db_files = glob(db_globstr)

    # Check if there are any files
    if len(db_files) == 0:
        raise ValueError(f'No files found in {database} directory. '
                         'Please check path.')

    if debug:
        print('Found files:')
        for file in db_files:
            print(file)
    else:
        # Get subset
        GFM = GFManager(db_files)
        GFM.load_header_variables()
        GFM.write_subset_directIO(subsetfilename, latitude, longitude, depth_in_km,
                                  radius_in_km, NGLL=ngll, fortran=fortran)


@database.command(name='extract')
def database_extract():
    """To extract traces directly from the database. NOT YET IMPLEMENTED!"""
    print('NOT YET IMPLEMENTED!')


@database.group()
def plot():
    '''Interface to database plotting tools.'''
    pass


@plot.group()
def station():
    '''Interface to station plotting tools.'''
    pass


@station.command(name='seismogram')
@click.argument('databaseroot', type=click.Path(exists=True))
@click.argument('cmtsolutionfilename', type=click.Path(exists=True))
@click.argument('network', type=str)
@click.argument('station', type=str)
def plot_station(databaseroot, cmtsolutionfilename, network, station):

    # External
    import os
    import matplotlib.pyplot as plt

    # Internal
    from gf3d.source import CMTSOLUTION
    from gf3d.seismograms import get_seismograms
    from gf3d.plot.seismogram import plotseismogram

    # CMTSOLUTION
    cmt = CMTSOLUTION.read(cmtsolutionfilename)
    print(cmt)

    # %%
    # Loading the subset database
    file = os.path.join(databaseroot, network, station,
                        f'{network}.{station}.h5')

    if not os.path.exists(file):
        raise ValueError(f'File {file} does not exist. Please check path.')

    rp = get_seismograms(file, cmt)

    limits = rp[0].stats.starttime, rp[0].stats.endtime

    plotseismogram(rp, None, cmt, limits=limits)
    plt.show(block=True)


@cli.group()
def subset():
    """Interface to generated subset."""
    pass


@subset.command(name='info')
@click.argument('subsetfilename', type=click.Path(exists=True))
def subset_info(subsetfilename: str):
    """Prints all relevant subset info.
    """
    from pprint import pprint
    from gf3d.seismograms import GFManager

    GFM = GFManager(subsetfilename)
    GFM.load_subset_header_only()
    pprint(GFM.header)


@subset.command(name='stations')
@click.argument('subsetfilename', type=click.Path(exists=True))
def subset_stations(subsetfilename: str):
    """Prints all relevant subset info.
    """
    from pprint import pprint
    from gf3d.seismograms import GFManager

    GFM = GFManager(subsetfilename)
    GFM.load_subset_header_only()

    net, sta, lat, lon, bur = 'Network', 'Station', 'Latitude', 'Longitude', 'Burial'

    print(f'#{net:>10s},{sta:>10s},{lat:>15s},{lon:>15s},{bur:>15s}')
    print('#' + 69 * '-')
    for net, sta, lat, lon, bur in zip(
            GFM.networks, GFM.stations, GFM.latitudes, GFM.longitudes, GFM.burials*1000):
        print(f' {net:>10s},{sta:>10s},{lat:>15.5f},{lon:>15.5f},{bur:>15.5f}')


@subset.command(name='extract')
@click.argument('subsetfilename', type=click.Path(exists=True))
@click.argument('yr', type=int)
@click.argument('mo', type=int)
@click.argument('da', type=int)
@click.argument('ho', type=int)
@click.argument('mi', type=int)
@click.argument('se', type=float)
@click.argument('mrr', type=float)
@click.argument('mtt', type=float)
@click.argument('mpp', type=float)
@click.argument('mrt', type=float)
@click.argument('mrp', type=float)
@click.argument('mtp', type=float)
@click.argument('latitude', type=float)
@click.argument('longitude', type=float)
@click.argument('depth', type=float)
@click.argument('time_shift', type=float)
@click.argument('hdur', type=float)
@click.argument('itypsokern', type=int)
@click.argument('outdir', type=str)
def subset_extract(
        subsetfilename: str,
        yr: int,
        mo: int,
        da: int,
        ho: int,
        mi: int,
        se: float,
        mrr: float,
        mtt: float,
        mpp: float,
        mrt: float,
        mrp: float,
        mtp: float,
        latitude: float,
        longitude: float,
        depth: float,
        time_shift: float,
        hdur: float,
        itypsokern: int,
        outdir: str):
    """Gets synthetics from subset file.

    IMPORTANT: For negative latitudes and longitudes use following setup:

        gf3d query synthetics - - ARG1 ARG2

    \b
    OUTPUT FORMAT:
    --------------
    \b
                                                   #                          itypsokern
        <outdir>/NET.STA.S3.MX{N,E,Z}.mseed        # for the synthetic        0
        <outdir>/NET.STA.S3.MX{N,E,Z}.<par>.mseed  # for the kernels
                                                   # par = mrr, mtt, mpp,     1
                                                   #         mrt, mrp, mtp,
                                                   #       lat, lon, dep,     2
                                                   #       cmt, hdr           3

    \b
    PARAMETERS:
    -----------

    \b
    SUBSETFILENAME = file containing Green functions of a subset of elements
    YR = origin year
    MO = origin month
    DA = origin day
    HO = origin hour
    MI = origin minute
    SE = origin seconds
    MRR = Moment tensor element[dyn * cm]
    MTT = ---//---
    MPP = ---//---
    MRT = ---//---
    MRP = ---//---
    MTP = ---//---
    LATITUDE = centroid latitude[deg]
    LONGITUDE = centroid longitude[deg]
    DEPTH = centroid depth[km]
    TSHIFT = centroid time shift[s]
    HDUR = centroid half_duration[s]
    ITYPSOKERN = which seismograms to make, see below
    OUTDIR = directory to write synthetics to

    \b
    OUTPUT:
    -------

    \b
    if itypsokern = 0, the subroutine only returns the synthetic seismogram
       itypsokern = 1, the subroutine returns the synthetic seismogram,
                     and the 6 moment-tensor kernels
       itypsokern = 2, the subroutine returns the synthetic seismogram, the
                     6 moment-tensor kernels and 4 centroid kernels

    """
    import os
    from obspy import UTCDateTime
    from gf3d.seismograms import GFManager
    from gf3d.source import CMTSOLUTION

    # Create CMT source
    cmt = CMTSOLUTION(
        origin_time=UTCDateTime(yr, mo, da, ho, mi, se),
        pde_lat=0.0, pde_lon=0.0, pde_depth=0.0, mb=0.0, ms=0.0,
        region_tag='n/a', eventname='n/a',
        time_shift=time_shift, hdur=hdur, latitude=latitude,
        longitude=longitude, depth=depth,
        Mrr=mrr, Mtt=mtt, Mpp=mpp, Mrt=mrt, Mrp=mrp, Mtp=mtp)

    GFM = GFManager(subsetfilename)
    GFM.load()

    if itypsokern >= 0:
        # Get seismograms
        synt = GFM.get_seismograms(cmt)

        # Create synthetic output directory if it doesn't exist
        if os.path.exists(outdir) == False:
            os.makedirs(outdir)

        for tr in synt:
            tr.write(os.path.join(outdir, tr.id + '.mseed'), format='MSEED')

    if itypsokern >= 1:

        # pypars to F90
        pypars = {
            "Mrr": "mrr",
            "Mtt": "mtt",
            "Mpp": "mpp",
            "Mrt": "mrr",
            "Mrp": "mrp",
            "Mtp": "mtp",
            "latitude": "lat",
            "longitude": "lon",
            "depth": "dep",
            "time_shift": "cmt",
            "hdur": "hdr"
        }

        pardict = GFM.get_frechet(cmt, rtype=itypsokern)

        for par, kern in pardict.items():

            for tr in kern:
                tr.write(os.path.join(
                    outdir, tr.id + f'.{pypars[par]}.mseed'), format='MSEED')


@subset.command(name='extract-cmt')
@click.argument('subsetfilename', type=click.Path(exists=True))
@click.argument('cmtsolutionfilename', type=click.Path(exists=True))
@click.argument('itypsokern', type=int)
@click.argument('outdir', type=str)
@click.option('--inv', is_flag=True, default=False, help='Write inventory', type=bool)
def subset_extract_cmt(
        subsetfilename: str,
        cmtsolutionfilename: str,
        itypsokern: int,
        outdir: str,
        inv: bool = False):
    """Gets synthetics from subset file.

        gf3d subset extract-cmt subsetfilename cmtsolutionfilename itypsokern outdir

    \b
    OUTPUT FORMAT:
    --------------
    \b
                                                   #                          itypsokern
        <outdir>/NET.STA.S3.MX{N,E,Z}.mseed        # for the synthetic        0
        <outdir>/NET.STA.S3.MX{N,E,Z}.<par>.mseed  # for the kernels
                                                   # par = mrr, mtt, mpp,     1
                                                   #       mrt, mrp, mtp,
                                                   #       lat, lon, dep,     2
                                                   #       cmt, hdr           3

    \b
    PARAMETERS:
    -----------

    \b
    SUBSETFILENAME = file containing Green functions of a subset of elements
    CMTSOLUTIONFILENAME = file with a cmtsolution
    ITYPSOKERN = which seismograms to make, see below
    OUTDIR = directory to write synthetics to

    \b
    OUTPUT:
    -------

    \b
    if itypsokern = 0, the subroutine only returns the synthetic seismogram
       itypsokern = 1, the subroutine returns the synthetic seismogram,
                     and the 6 moment-tensor kernels
       itypsokern = 2, the subroutine returns the synthetic seismogram, the
                     6 moment-tensor kernels and 4 centroid kernels

    """
    import os
    from gf3d.seismograms import GFManager
    from gf3d.source import CMTSOLUTION

    # Create CMT source
    cmt = CMTSOLUTION.read(cmtsolutionfilename)

    GFM = GFManager(subsetfilename)
    GFM.load()

    if itypsokern >= 0:
        # Get seismograms
        synt = GFM.get_seismograms(cmt)

        # Create synthetic output directory if it doesn't exist
        if os.path.exists(outdir) == False:
            os.makedirs(outdir)

        for tr in synt:
            tr.write(os.path.join(outdir, tr.id + '.sac'), format='SAC')

    if itypsokern >= 1:

        # pypars to F90
        pypars = {
            "Mrr": "mrr",
            "Mtt": "mtt",
            "Mpp": "mpp",
            "Mrt": "mrr",
            "Mrp": "mrp",
            "Mtp": "mtp",
            "latitude": "lat",
            "longitude": "lon",
            "depth": "dep",
            "time_shift": "cmt",
            "hdur": "hdr"
        }

        pardict = GFM.get_frechet(cmt, rtype=itypsokern)

        for par, kern in pardict.items():

            for tr in kern:
                tr.write(os.path.join(
                    outdir, tr.id + f'.{pypars[par]}.mseed'), format='SAC')

    if inv:
        inventory = GFM.get_inventory()
        inventory.write(os.path.join(outdir, 'station.xml'),
                        format='STATIONXML')


@subset.group()
def plot():
    '''Interface to database plotting tools.'''
    pass


# @plot.command(name='elements')
# @click.argument('subsetfilename', type=click.Path(exists=True))
# @click.option('--cmt', default=None, help='Locate source and plot it in the mesh',
#               type=bool)
# def plot_subset_elements(subsetfilename):
#     from gf3d.seismograms import GFManager
#     from gf3d.plot.mesh import meshplot
#     from gf3d.locate_point import locate_point
#     # Loading the subset database

#     gfsub = GFManager(subsetfilename)
#     gfsub.load()

#     meshplot(subsetfilename, outfile="mesh.html")


@plot.group()
def station():
    '''Interface to database plotting tools.'''
    pass


@station.command(name='seismogram')
@click.argument('subsetfilename', type=click.Path(exists=True))
@click.argument('cmtsolutionfilename', type=click.Path(exists=True))
@click.argument('network', type=str)
@click.argument('station', type=str)
def plot_station_seismogram(subsetfilename, cmtsolutionfilename, network, station):

    # External
    import matplotlib.pyplot as plt

    # Internal
    from gf3d.source import CMTSOLUTION
    from gf3d.seismograms import GFManager
    from gf3d.plot.seismogram import plotseismogram

    # CMTSOLUTION
    cmt = CMTSOLUTION.read(cmtsolutionfilename)

    # %%
    # Loading the subset database

    gfsub = GFManager(subsetfilename)
    gfsub.load()

    rp = gfsub.get_seismograms(cmt)

    limits = rp[0].stats.starttime, rp[0].stats.endtime

    plotseismogram(rp.select(network=network, station=station),
                   None, cmt, limits=limits)
    plt.show(block=True)


@station.command(name='section')
@click.argument('subsetfilename', type=click.Path(exists=True))
@click.argument('cmtsolutionfilename', type=click.Path(exists=True))
@click.argument('component', type=str)
def plot_subset_section(
        subsetfilename: str,
        cmtsolutionfilename: str,
        component: str):

    # External

    # Internal

    # CMTSOLUTION
    cmt = CMTSOLUTION.read(cmtsolutionfilename)

    # Load subset
    gfsub = GFManager(subsetfilename)
    gfsub.load()

    # Get seismograms from the database

    rp = gfsub.get_seismograms(cmt)

    # Get time limits
    starttime = rp[0].stats.starttime
    endtime = rp[0].stats.endtime
    limits = (starttime, endtime)

    # Plots a section of observed and synthetic
    plotsection(rp, rp, cmt, comp=component, lw=0.75,
                limits=limits, sync='k', scale=3)

    plt.show()


if __name__ == "__main__":
    cli()
