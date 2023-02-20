from multiprocessing.sharedctypes import Value
import numpy as np

from .constants import PI_OVER_TWO, DEGREES_TO_RADIANS, ONE_MINUS_F_SQUARED, \
    ASSUME_PERFECT_SPHERE, ZERO, PI, TWO_PI, TINYVAL, R_UNIT_SPHERE, R_PLANET

from .get_topo_bathy import get_topo_bathy
from .spline_eval import spline_evaluation
from .transformations.rthetaphi_xyz import lat_2_geocentric_colat, reduce_geocentric
from .logger import logger


def source2xyz(
        lat: float, lon: float, depth_in_km: float, M: None | np.ndarray = None,
        topography: bool = False, ellipticity: bool = False,
        ibathy_topo: np.ndarray | None = None,
        NX_BATHY: int | None = None,
        NY_BATHY: int | None = None,
        RESOLUTION_TOPO_FILE: float | None = None,
        rspl: np.ndarray | None = None,
        ellipicity_spline: np.ndarray | None = None,
        ellipicity_spline2: np.ndarray | None = None):

    # Make sure the longitude in [0.0, 360.0]
    lon = lon + 360.0 if (lon < 0.0) else lon
    lon = lon - 360.0 if (lon > 360.0) else lon

    # convert geographic latitude lat(degrees) to geocentric colatitude theta(radians)
    # lat in degrees --> theta in radians
    theta = lat_2_geocentric_colat(lat)

    phi = lon*DEGREES_TO_RADIANS
    theta, phi = reduce_geocentric(theta, phi)

    logger.debug(f"lat  {lat}, lon {lon}")
    logger.debug(f"theta {theta}, phi {phi}")

    sint = np.sin(theta)
    cost = np.cos(theta)
    sinp = np.sin(phi)
    cosp = np.cos(phi)

    if M is not None:
        # get the moment tensor
        Mrr = M[0]
        Mtt = M[1]
        Mpp = M[2]
        Mrt = M[3]
        Mrp = M[4]
        Mtp = M[5]

        # convert from a spherical to a Cartesian representation of the moment tensor
        Mxx = sint*sint*cosp*cosp*Mrr + cost*cost*cosp*cosp*Mtt + sinp*sinp*Mpp \
            + 2.0*sint*cost*cosp*cosp*Mrt - 2.0*sint*sinp*cosp*Mrp - 2.0*cost*sinp*cosp*Mtp

        Myy = sint*sint*sinp*sinp*Mrr + cost*cost*sinp*sinp*Mtt + cosp*cosp*Mpp \
            + 2.0*sint*cost*sinp*sinp*Mrt + 2.0*sint*sinp*cosp*Mrp + 2.0*cost*sinp*cosp*Mtp

        Mzz = cost*cost*Mrr + sint*sint*Mtt - 2.0*sint*cost*Mrt

        Mxy = sint*sint*sinp*cosp*Mrr + cost*cost*sinp*cosp*Mtt - sinp*cosp*Mpp \
            + 2.0*sint*cost*sinp*cosp*Mrt + sint * \
            (cosp*cosp-sinp*sinp)*Mrp + cost*(cosp*cosp-sinp*sinp)*Mtp

        Mxz = sint*cost*cosp*Mrr - sint*cost*cosp*Mtt \
            + (cost*cost-sint*sint)*cosp*Mrt - cost*sinp*Mrp + sint*sinp*Mtp

        Myz = sint*cost*sinp*Mrr - sint*cost*sinp*Mtt \
            + (cost*cost-sint*sint)*sinp*Mrt + cost*cosp*Mrp - sint*cosp*Mtp

    nu_source = np.zeros((3, 3))

    # record three components for each station
    for iorientation in range(3):
        #   North
        if (iorientation == 0):
            stazi = 0.0
            stdip = 0.0
        #   East
        elif (iorientation == 1):
            stazi = 90.0
            stdip = 0.0
        #   Vertical
        elif (iorientation == 2):
            stazi = 0.0
            stdip = - 90.0
        else:
            raise ValueError('incorrect orientation')

        #   get the orientation of the seismometer
        thetan = (90.0 + stdip) * DEGREES_TO_RADIANS
        phin = stazi * DEGREES_TO_RADIANS

        # we use the same convention as in Harvard normal modes for the orientation
        n = np.zeros(3)
        #   vertical component
        n[0] = np.cos(thetan)
        #   N-S component
        n[1] = - np.sin(thetan)*np.cos(phin)
        #   E-W component
        n[2] = np.sin(thetan)*np.sin(phin)

        #   get the Cartesian components of n in the model: nu
        nu_source[iorientation, 0] = n[0]*sint * \
            cosp + n[1]*cost*cosp - n[2]*sinp
        nu_source[iorientation, 1] = n[0]*sint * \
            sinp + n[1]*cost*sinp + n[2]*cosp
        nu_source[iorientation, 2] = n[0]*cost - n[1]*sint

    # point depth (in m)
    depth = depth_in_km*1000.0

    # normalized source radius
    r0 = R_UNIT_SPHERE

    logger.debug(f'depth  {depth}')
    logger.debug(f'RPLANET {R_PLANET}')

    # finds elevation of position
    if topography:
        if NX_BATHY is None:
            raise ValueError(
                'NX_BATHY needs to be defined to compute topography.')
        if NY_BATHY is None:
            raise ValueError(
                'NY_BATHY needs to be defined to compute topography.')
        if RESOLUTION_TOPO_FILE is None:
            raise ValueError(
                'RESOLUTION_TOPO_FILE needs to be defined to compute topography.')
        if ibathy_topo is None:
            raise ValueError(
                'ibathy_topo needs to be defined to compute topography.')

        # Compute local elevatoin
        elevation = get_topo_bathy(
            lat, lon, ibathy_topo, NX_BATHY, NY_BATHY, RESOLUTION_TOPO_FILE)

        logger.debug(f'elevation: {elevation}')

        r0 = r0 + elevation/R_PLANET

    # ellipticity
    if (ellipticity):
        if rspl is None:
            raise ValueError(
                'radius spline needs to be defined to evaluate ellipticity.')

        if ellipicity_spline is None:
            raise ValueError(
                'ellipticity spline needs to be defined to evaluate ellipticity.')

        if ellipicity_spline2 is None:
            raise ValueError(
                'ellipticity spline coefficients need to be defined to evaluate ellipticity.')

        # this is the Legendre polynomial of degree two, P2(cos(theta)),
        # see the discussion above eq(14.4) in Dahlen and Tromp (1998)
        p20 = 0.5*(3.0*cost*cost-1.0)

        # todo: check if we need radius or r0 for evaluation below...
        #       (receiver location routine takes r0)
        radius = r0 - depth/R_PLANET

        # get ellipticity using spline evaluation
        ell = spline_evaluation(rspl, ellipicity_spline,
                                ellipicity_spline2, radius)

        logger.debug(f"ELL: {ell}, p20: {p20}")
        # this is eq(14.4) in Dahlen and Tromp(1998)
        r0 = r0*(1.0-(2.0/3.0)*ell*p20)

    # stores surface radius for info output
    r0_source = r0

    logger.debug("rsource: {r0_source}")

    # subtracts source depth(given in m)
    r_target = r0 - depth/R_PLANET

    # compute the Cartesian position of the source
    x = r_target*sint*cosp
    y = r_target*sint*sinp
    z = r_target*cost

    if M is not None:
        # Collect M_cartesian
        M_cartesian = np.array([Mxx, Myy, Mzz, Mxy, Mxz, Myz])

        return x, y, z, M_cartesian

    else:
        return x, y, z
