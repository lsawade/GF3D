import numpy as np
import typing as tp

def isiterable(var):
    if (type(var) is list) or (type(var) is tuple):
        return True
    else:
        return False

def geo2cart(r: float or np.ndarray or list,
             theta: float or np.ndarray or list,
             phi: float or np.ndarray or list) \
    -> tp.Tuple[float or np.ndarray or list,
                float or np.ndarray or list,
                float or np.ndarray or list]:
    """Computes Cartesian coordinates from geographical coordinates.
    Parameters
    ----------
    r : float or numpy.ndarray or list
        Radius
    theta : float or numpy.ndarray or list
        Latitude (-90, 90)
    phi : float or numpy.ndarray or list
        Longitude (-180, 180)
    Returns
    -------
    float or np.ndarray or list, float or np.ndarray or list, float or np.ndarray or list
        (x, y, z)
    """

    if isiterable(r) or isiterable(theta) or isiterable(phi):
        r = np.array(r)
        theta = np.array(theta)
        phi = np.array(phi)

    # Convert to Radians
    thetarad = theta * np.pi/180.0
    phirad = phi * np.pi/180.0

    # Compute Transformation
    x = r * np.cos(thetarad) * np.cos(phirad)
    y = r * np.cos(thetarad) * np.sin(phirad)
    z = r * np.sin(thetarad)

    if isiterable(r) or isiterable(theta) or isiterable(phi):
        x = x.tolist()
        y = y.tolist()
        z = z.tolist()

    return x, y, z


def cart2geo(x: float or np.ndarray or list,
             y: float or np.ndarray or list,
             z: float or np.ndarray or list) \
    -> tp.Tuple[float or np.ndarray or list,
                float or np.ndarray or list,
                float or np.ndarray or list]:
    """Computes Cartesian coordinates from geographical coordinates.
    Parameters
    ----------
    x : float or numpy.ndarray or list
        Radius
    y : float or numpy.ndarray or list
        Latitude (-90, 90)
    z : float or numpy.ndarray or list
        Longitude (-180, 180)
    Returns
    -------
    float or np.ndarray or list, float or np.ndarray or list, float or np.ndarray or list
        (r, theta, phi)
    """

    if type(x) is list:
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

    # Compute Transformation
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)

    # Catch corner case of latitude computation
    theta = np.where((r == 0), 0, np.arcsin(z/r))

    # Convert to Radians
    theta *= 180.0/np.pi
    phi *= 180.0/np.pi

    if type(r) is list:
        r = r.tolist()
        theta = theta.tolist()
        phi = phi.tolist()

    return r, theta, phi


def geomidpointv(lat1, lon1, lat2, lon2):
    """This is dumb simple..."""

    # Gets cartesian coordinates
    x1 = geo2cart(1, lat1, lon1)
    x2 = geo2cart(1, lat2, lon2)

    # vector between x1 and x2
    dx = np.array(x2) - np.array(x1)

    # Add vector to point
    _, lat3, lon3 = cart2geo(*(x1 + 0.5 * dx))

    return lat3, lon3
