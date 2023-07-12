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

gf3d extract subset -- SUBSETFILENAME YYYY MM DD HH MM SS MRR MTT MPP MRT MRP MTP LATITUDE LONGITUDE DEPTH
                             TIME_SHIFT HDUR ITYPSOKERN OUTDIR

Example:
gf3d extract subset -- subset.h5 2015 9 16 22 54 32.90 \
                            1.950000e+28 -4.360000e+26 -1.910000e+28 \
                            7.420000e+27 -2.480000e+28 9.420000e+26 \
                            -31.1300 -72.0900 17.3500 49.9800 33.4000 \
                            2 OUTPUT

"""
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
def query_info(databasename: str):
    '''Query info from a hosted database server.'''
    from gf3d.client import GF3DClient
    gfcl = GF3DClient(databasename)
    gfcl.get_info()


@query.command(name='stations')
@click.argument('databasename', type=str)
def query_stations(databasename: str):
    """Query available stations from a hosted database server."""
    from gf3d.client import GF3DClient
    gfcl = GF3DClient(databasename)
    stations = gfcl.stations_avail()
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
@click.option('--fortran', default=False, help='number of greetings', type=bool)
@click.option('--ngll', default=5, help='Number of GLL points 5 or 3', type=int)
@click.option('--netsta', default=None, help='Station subselection. NOT IMPLEMENTED', type=str)
def query_subset(
        databasename: str,
        subsetfilename: str,
        latitude: float,
        longitude: float,
        depth_in_km: float,
        radius_in_km: float = 100,
        ngll: int = 5,
        netsta: list | None = None,
        fortran: bool = False):
    """Query a subset from a hosted database server.

    IMPORTANT: For negative latitudes and longitudes use following setup:

        gf3d query subset [--option = value] - - DATABASENAME SUBSETFILENAME LATITUDE ...

    """

    from gf3d.client import GF3DClient
    print(ngll)

    gfcl = GF3DClient(databasename)
    gfcl.get_subset(
        subsetfilename,
        latitude=latitude,
        longitude=longitude,
        depth_in_km=depth_in_km,
        radius_in_km=radius_in_km,
        NGLL=ngll,
        netsta=netsta,
        fortran=fortran)


@database.command(name='extract')
def database_extract():
    """To extract traces directly from the database. NOT YET IMPLEMENTED!"""
    print('NOT YET IMPLEMENTED!')


@cli.group()
def subset():
    """Interface to generated subset."""
    pass


@subset.command(name='info')
@click.argument('subsetfilename', type=str)
def subset_info(subsetfilename: str):
    """Prints all relevant subset info.
    """
    from pprint import pprint
    from gf3d.seismograms import GFManager

    GFM = GFManager(subsetfilename)
    GFM.load_subset_header_only()
    pprint(GFM.header)


@subset.command(name='stations')
@click.argument('subsetfilename', type=str)
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
@click.argument('subsetfilename', type=str)
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
    PARAMETERS
    ----------

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


if __name__ == "__main__":
    cli()
