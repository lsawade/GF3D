import numpy as np
from .constants import ZERO


def get_topo_bathy(
        xlat, xlon, ibathy_topo, NX_BATHY: int, NY_BATHY: int, RESOLUTION_TOPO_FILE: float) -> float:
    """get elevation or ocean depth in meters at a given latitude and longitude

    Parameters
    ----------
    xlat : float
        location to interpolate
    xlon : _type_
        longitude to interpolat
    ibathy_topo : _type_
        bathymetry/topography values
    NX_BATHY : int
        Longitude dicretization
    NY_BATHY : int
        Latitude dicretization
    RESOLUTION_TOPO_FILE : float
        Lat/Lon resolution

    Returns
    -------
    float
        topogrpahy value.

    """

    # use constants

    # location latitude/longitude ( in degree)
    # double precision, intent(in ): : xlat, xlon

    # # returns elevation ( in meters)
    # double precision, intent(out): : value

    # # use integer array to store values
    # integer, dimension(NX_BATHY, NY_BATHY), intent(in ) : : ibathy_topo

    # # local parameters
    # integer:: iadd1, iel1

    # double precision: : samples_per_degree_topo
    # double precision: : xlo
    # double precision:: lon_corner, lat_corner, ratio_lon, ratio_lat

    # initializes elevation
    value = ZERO

    # longitude within range[0, 360] degrees
    xlo = xlon
    xlo = xlo + 360.0 if (xlo < 0.0) else xlo
    xlo = xlo - 360.0 if (xlo > 360.0) else xlo

    # compute number of samples per degree
    samples_per_degree_topo = RESOLUTION_TOPO_FILE / 60.0

    # compute offset in data file and avoid edge effects
    iadd1 = 0 + int((90.0-xlat)/samples_per_degree_topo)
    iadd1 = 0 if (iadd1 < 0) else iadd1
    iadd1 = NY_BATHY-1 if (iadd1 > NY_BATHY-1) else iadd1

    iel1 = int(xlo/samples_per_degree_topo)
    iel1 = NX_BATHY-1 if (iel1 <= 0 or iel1 > NX_BATHY-1) else iel1

    # Use bilinear interpolation rather nearest point interpolation

    # convert integer value to double precision
    #  value = dble(ibathy_topo(iel1, iadd1))

    lon_corner = iel1 * samples_per_degree_topo
    lat_corner = 90.0 - iadd1 * samples_per_degree_topo

    ratio_lon = (xlo-lon_corner)/samples_per_degree_topo
    ratio_lat = (xlat-lat_corner)/samples_per_degree_topo

    ratio_lon = 0.0 if (ratio_lon < 0.0) else ratio_lon
    ratio_lon = 1.0 if (ratio_lon > 1.0) else ratio_lon
    ratio_lat = 0.0 if (ratio_lat < 0.0) else ratio_lat
    ratio_lat = 1.0 if (ratio_lat > 1.0) else ratio_lat

    # convert integer value to double precision
    if (iadd1 <= NY_BATHY-1 and iel1 <= NX_BATHY-1):
        # interpolates for points within boundaries
        value = ibathy_topo[iel1, iadd1] * (1.0-ratio_lon) * (1.0-ratio_lat) \
            + ibathy_topo[iel1+1, iadd1] * ratio_lon * (1.0-ratio_lat) \
            + ibathy_topo[iel1+1, iadd1+1] * ratio_lon * ratio_lat \
            + ibathy_topo[iel1, iadd1+1] * (1.0-ratio_lon) * ratio_lat

    elif (iadd1 <= NY_BATHY-1 and iel1 == NX_BATHY):
        # interpolates for points on longitude border
        value = ibathy_topo[iel1, iadd1] * (1.0 - ratio_lon)*(1.0 - ratio_lat) \
            + ibathy_topo[1, iadd1] * ratio_lon*(1.0 - ratio_lat) \
            + ibathy_topo[1, iadd1+1] * ratio_lon*ratio_lat \
            + ibathy_topo[iel1, iadd1+1] * (1.0 - ratio_lon)*ratio_lat

    else:
        # for points on latitude boundaries
        value = ibathy_topo(iel1, iadd1)

    return value
