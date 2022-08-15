"""
Coordinate conversion routines
------------------------------

Include conversions from and to

* Cartesian
* Geocentric
* Geographic

Most of the routines are take from ``specfem3d_globe`` and adapted to Python.

"""

import numpy as np
from ..constants import \
    ZERO, SMALL_VAL_ANGLE, PI_OVER_TWO, DEGREES_TO_RADIANS, \
    ONE_MINUS_F_SQUARED, ASSUME_PERFECT_SPHERE, ZERO, PI, TWO_PI, TINYVAL


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
    zmesh = -SMALL_VAL_ANGLE if (
        zmesh > - SMALL_VAL_ANGLE and zmesh <= ZERO) else zmesh
    zmesh = SMALL_VAL_ANGLE if (
        zmesh < SMALL_VAL_ANGLE and zmesh >= ZERO) else zmesh

    theta = np.arctan2(np.sqrt(xmesh*xmesh+ymesh*ymesh), zmesh)

    # Fix X
    xmesh = -SMALL_VAL_ANGLE if (
        xmesh > - SMALL_VAL_ANGLE and xmesh <= ZERO) else xmesh
    xmesh = SMALL_VAL_ANGLE if (
        xmesh < SMALL_VAL_ANGLE and xmesh >= ZERO) else xmesh

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
