from ..transformations.Ra2b import Ra2b
from matplotlib import path
from ..geoutils import geo2cart, cart2geo
import numpy as np


def ingeopoly(polylat, polylon, lat, lon):

    # Get mean location for the rotations to 0,0 -- Least amount of distortion
    mlat = np.mean(polylat)
    mlon = np.mean(polylon)

    # Get origin of rotation
    x0 = geo2cart(1, mlat, mlon)
    x1 = np.array([1,0,0])

    # Rotation matrix
    R = Ra2b(x0, x1)

    # Create point arrays
    px = np.vstack(geo2cart(1, polylat, polylon))
    x = np.vstack(geo2cart(1, lat.flatten(), lon.flatten()))

    # Rotate points
    rpx = R @ px
    rx = R @ x

    # Transform back to Centered geo coordinates
    _, cplat, cplon = cart2geo(rpx[0,:], rpx[1,:], rpx[2,:])
    _, clat, clon = cart2geo(rx[0,:], rx[1,:], rx[2,:])

    # Create polygon
    p = path.Path([(_x,_y) for _x, _y in zip(cplon, cplat)])  # square with legs length 1 and bottom left corner at the origin

    # Check if in polygon
    flags = p.contains_points([(_x,_y) for _x, _y in zip(clon, clat)], radius=0.01)

    return np.array(flags).reshape(lat.shape)



