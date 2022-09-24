"""
Coordinate conversion routines
------------------------------

Include conversions from and to

* Cartesian
* Geocentric
* Geographic

Most of the routines are take from ``specfem3d_globe`` and adapted to Python.

"""

from typing import Iterable
import numpy as np
from ..constants import \
    ZERO, SMALL_VAL_ANGLE, PI_OVER_TWO, DEGREES_TO_RADIANS, RADIANS_TO_DEGREES, \
    ONE_MINUS_F_SQUARED, ASSUME_PERFECT_SPHERE, ZERO, PI, TWO_PI, TINYVAL, \
    HUGEVAL


def rthetaphi_2_xyz(r, theta, phi):
    """Geocentric to cartesian

    Parameters
    ----------
    r : numeric
        radius
    theta : numeric
        colatitude, [0, PI]
    phi : numeric
        longitude, [0, 2PI]
    """

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z


def xyz_2_rthetaphi(x, y, z):
    """Convert Cartesian x, y, z to geocentric r, theta, phi

    Parameters
    ----------
    x : numeric
        x coordinate
    y : numeric
        y coordinate
    z : numeric
        z coordinate

    Returns
    -------
    r, theta, phi
        tuple of numeric coordinates r theta phi

    """

    xmesh = x
    ymesh = y
    zmesh = z

    # Fix Z
    zmesh = np.where(np.logical_and(zmesh > - SMALL_VAL_ANGLE, zmesh <=
                     ZERO), -SMALL_VAL_ANGLE, zmesh)
    zmesh = np.where(np.logical_and(zmesh < SMALL_VAL_ANGLE, zmesh >=
                     ZERO), SMALL_VAL_ANGLE, zmesh)

    theta = np.arctan2(np.sqrt(xmesh*xmesh+ymesh*ymesh), zmesh)

    # Fix X
    xmesh = np.where(np.logical_and(xmesh > - SMALL_VAL_ANGLE, xmesh <=
                     ZERO), -SMALL_VAL_ANGLE, xmesh)
    xmesh = np.where(np.logical_and(xmesh < SMALL_VAL_ANGLE, xmesh >=
                     ZERO), SMALL_VAL_ANGLE, xmesh)

    phi = np.arctan2(ymesh, xmesh)

    r = np.sqrt(xmesh*xmesh + ymesh*ymesh + zmesh*zmesh)

    return r, theta, phi


def reduce_geocentric(theta: float, phi: float):
    """bring theta between 0 and PI, and phi between 0 and 2*PI.
    Routine taken from specfem3d_globe."""

    # slightly move points to avoid roundoff problem when exactly on the polar axis
    theta = theta + 0.0000001 if (np.abs(theta) < TINYVAL) else theta
    phi = phi + 0.0000001 if (np.abs(phi) < TINYVAL) else phi

    # colatitude
    th = theta
    # longitude
    ph = phi

    # brings longitude between 0 and 2*PI
    if (ph < ZERO or ph > TWO_PI):
        i = abs(int(ph/TWO_PI))
        if (ph < ZERO):
            ph = ph+(i+1)*TWO_PI
        else:
            if (ph > TWO_PI):
                ph = ph-i*TWO_PI
        phi = ph

    # brings colatitude between 0 and PI
    if (th < ZERO or th > PI):
        i = int(th/PI)
        if (th > ZERO):
            if (np.mod(i, 2) != 0):
                th = (i+1)*PI-th
                # switches hemisphere
                if (ph < PI):
                    ph = ph+PI
                else:
                    ph = ph-PI

            else:
                th = th-i*PI

        else:
            if (np.mod(i, 2) == 0):
                th = -th+i*PI

                # switches hemisphere
                if (ph < PI):
                    ph = ph+PI
                else:
                    ph = ph-PI

            else:
                th = th-i*PI

        theta = th
        phi = ph

    # checks ranges
    if (theta < ZERO or theta > PI):
        raise ValueError('theta out of range in reduce')

    if (phi < ZERO or phi > TWO_PI):
        raise ValueError('phi out of range in reduce')

    return theta, phi


def lat_2_geocentric_colat(lat_prime: float) -> float:
    """Converts geographic latitude (lat_prime) (in degrees) to geocentric
    colatitude (theta) ( in radians)
    See Dahlen and Tromp Chapter 14 (I think).

    Parameters
    ----------
    lat_prime : float
        geographical latitude

    Returns
    -------
    float
        geocentric colatitude
    """

    if not ASSUME_PERFECT_SPHERE:
        # converts geographic(lat_prime) to geocentric latitude and converts
        # to co-latitude(theta)
        theta = PI_OVER_TWO - np.arctan(
            ONE_MINUS_F_SQUARED * np.tan(lat_prime * DEGREES_TO_RADIANS))
    else:
        # for perfect sphere, geocentric and geographic latitudes are the same
        # converts latitude (in degrees to co-latitude ( in radians)
        theta = PI_OVER_TWO - lat_prime * DEGREES_TO_RADIANS

    return theta


def geocentric_2_geographic(theta):
    """
    note: september, 2014`
    factor: 1/(1 - e ^ 2) = 1/(1 - (1 - (1-f) ^ 2)) = 1/((1-f) ^ 2)`
    with eccentricity e ^ 2 = 1 - (1-f) ^ 2`
    see about Earth flattening in constants.h: flattening factor changed to 1/299.8`
                                               f = 1/299.8 -> 1/((1-f) ^ 2) = 1.0067046409645724`

    """

    # factor
    FACTOR_TAN = 1.0 / ONE_MINUS_F_SQUARED

    # note: instead of 1/tan(theta) we take cos(theta)/sin(theta) and avoid division by zero

    if not ASSUME_PERFECT_SPHERE:
        # mesh is elliptical
        # converts geocentric colatitude theta to geographic colatitude theta_prime
        theta_prime = PI_OVER_TWO - \
            np.arctan(FACTOR_TAN*np.cos(theta) /
                      np.maximum(TINYVAL, np.sin(theta)))

    else:
        # mesh is spherical, thus geocentric and geographic colatitudes are identical
        theta_prime = theta

    return theta_prime


def xyz_2_rlatlon(x, y, z):
    """converts geocentric coordinates x/y/z to geographic radius/latitude/longitude (in degrees)

    Parameters
    ----------
    x : array
        x coordinate
    y : array
        y coordinate
    z : array
        z coordinate

    Returns
    -------
    tuple(r, theta, phi)
        output coordinate in geographic coordinates
    """

    # converts location to radius/colatitude/longitude
    r, theta, phi = xyz_2_rthetaphi(x, y, z)

    # reduces range for colatitude to 0 and PI, for longitude to 0 and 2*PI
    if isinstance(theta, Iterable):
        vreduce = np.vectorize(reduce_geocentric)
        theta, phi = vreduce(theta, phi)

    else:
        theta, phi = reduce_geocentric(theta, phi)

    # converts geocentric to geographic colatitude
    # note: for example, the moho/topography/3D-model information is given in geographic latitude/longitude
    #       (lat/lon given with respect to a reference ellipsoid).
    #       we need to convert the geocentric mesh positions (theta,phi) to geographic ones (lat/lon),
    #       thus correcting geocentric latitude for ellipticity
    theta_prime = geocentric_2_geographic(theta)

    # gets geographic latitude and longitude in degrees
    lat = (PI_OVER_TWO - theta_prime) * RADIANS_TO_DEGREES
    lon = phi * RADIANS_TO_DEGREES

    return r, lat, lon


def xyz_2_latlon_minmax(x, y, z):
    """Compute range of latitudes and longitudes from cartesian coordinates.

    Parameters
    ----------
    x : arraylike
        x coordinate
    y : arraylike
        y coordinate
    z : arraylike
        z coordinate

    Returns
    -------
    tuple(lat_min, lat_max, lon_min, lon_max)
        extent in latlon/
    """

    # returns minimum and maximum values of latitude/longitude of given mesh points
    # latitude in degree between[-90, 90], longitude in degree between[0, 360]

    # loops only over corners
    r, lat, lon = xyz_2_rlatlon(x, y, z)

    # Compute mins and maxes
    lat_min = np.min(lat)
    lat_max = np.max(lat)
    lon_min = np.min(lon)
    lon_max = np.max(lon)

    # limits latitude to[-90.0, 90.0]
    lat_min = -90.0 if (lat_min < -90.0) else lat_min
    lat_max = 90.0 if (lat_max > 90.0) else lat_max

    # limits longitude to[0.0, 360.0]
    lon_min = lon_min + 360.0 if (lon_min < 0.0) else lon_min
    lon_min = lon_min - 360.0 if (lon_min > 360.0) else lon_min
    lon_max = lon_max + 360.0 if (lon_max < 0.0) else lon_max
    lon_max = lon_max - 360.0 if (lon_max > 360.0) else lon_max

    return lat_min, lat_max, lon_min, lon_max
