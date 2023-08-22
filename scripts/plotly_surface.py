# %%

import numpy as np
import plotly.graph_objects as go


def plotlypoly(rangex, rangey):
    polys = []
    NX = len(rangex)
    NY = len(rangey)
    points = np.zeros(((NX) * (NY), 2))
    ijk = np.zeros((2*(NX-1) * (NY-1), 3), dtype=int)
    counter = 0

    xx, yy = np.meshgrid(rangex, rangey)
    points[:, 0] = xx.T.flatten()
    points[:, 1] = yy.T.flatten()
    print(xx.T.shape, yy.T.shape, points.T.shape)
    counter = 0

    for j in range(0, NY-1, 1):
        for i in range(0, NX-1, 1):
            i0j0 = np.ravel_multi_index((i, j, ), (NX, NY))
            i1j0 = np.ravel_multi_index((i+1, j), (NX, NY))
            i1j1 = np.ravel_multi_index((i+1, j+1), (NX, NY))
            i0j1 = np.ravel_multi_index((i, j+1), (NX, NY))

            # interpolate center point

            # Triangle 1
            ijk[counter, :] = [i0j0, i1j0, i1j1]
            counter += 1
            # Triangle 2
            ijk[counter, :] = [i0j0, i1j1, i0j1]
            counter += 1

    return points, ijk


Nlon = 36
Nlat = 18
points, ijk = plotlypoly(np.linspace(0, 360, Nlon, endpoint=True),
                         np.linspace(0, 180, Nlat, endpoint=True))


# %%
# Mathematical convention!!!
# phi = 0 north pole
# theta = 0 x axis, increasing towards y axis
def sphere2cartpoly(thetaphi, r=1):
    Npoly = thetaphi.shape[0]
    xyz = np.zeros((Npoly, 3))
    xyz[:, 0] = r * np.sin(np.deg2rad(thetaphi[:, 1])) * \
        np.cos(np.deg2rad(thetaphi[:, 0]))
    xyz[:, 1] = r * np.sin(np.deg2rad(thetaphi[:, 1])) * \
        np.sin(np.deg2rad(thetaphi[:, 0]))
    xyz[:, 2] = r * np.cos(np.deg2rad(thetaphi[:, 1]))

    return xyz


xyz = sphere2cartpoly(points, r=1)

# %%


def HueBeach(v, M, ColorIn='white', ColorOut='red'):
    if np.dot(np.dot(M, v), v) > 0:
        return ColorOut
    else:
        return ColorIn


def get_beach_colors(L, poly, ijk):
    Npoly = ijk.shape[0]
    colors = Npoly * [None]
    DL = np.diag(L)
    for i in range(Npoly):
        slices = ijk[i, :]
        colors[i] = HueBeach(np.mean(poly[slices, :], axis=0), DL)
    return colors


def phiN(theta, L):
    """phiN is the phi coordinate of the (upper) nodal point at theta. phiN
    comes from solving (lambda1 x, lambda2, y, lambda3 z) .dot. (x,y,z) = 0,
    which is the nodal curve, which is good for any (lambda1, lambda2, lambda3).
    But to get phi for any theta you need lambda1 > 0, lambda2 > 0, lambda3 < 0
    OR lambda1 < 0, lambda2 < 0, lambda3 > 0.
    So, (theta, lambda1, lambda2, lambda3) -> (theta, phiN(theta))

    Parameters
    ----------
    theta : float or arraylike
        spherical longitude
    L : iterable
        lambda 1, 2, 3

    Returns
    -------
    _type_
        _description_
    """
    numerator = L[0] * np.cos(np.deg2rad(theta)) ** 2 + \
        L[1] * np.sin(np.deg2rad(theta)) ** 2
    denominator = L[0] * np.cos(np.deg2rad(theta)) ** 2 + \
        L[1] * np.sin(np.deg2rad(theta)) ** 2 - L[2]
    return 180/np.pi*np.arccos(np.sqrt(numerator/denominator))


def sphere2cartline(theta, phi):
    """Mathematical convention!!!"""
    xyz = np.zeros((len(theta), 3))
    xyz[:, 0] = np.sin(np.deg2rad(phi)) * np.cos(np.deg2rad(theta))
    xyz[:, 1] = np.sin(np.deg2rad(phi)) * np.sin(np.deg2rad(theta))
    xyz[:, 2] = np.cos(np.deg2rad(phi))

    return xyz


# %%


L = [0.0, 19, -20]
facecolors = get_beach_colors(L, xyz, ijk)


xref = np.diag([-1, 1, 1])
yref = np.diag([1, -1, 1])
zref = np.diag([1, 1, -1])
theta = np.linspace(0, 360, 1000)
phi = phiN(theta, L)

Uxyz = sphere2cartline(theta, phi)
Lxyz = np.dot(zref, Uxyz.T).T

# %%
fig = go.Figure(data=[
    go.Mesh3d(
        # Triangle vertices in coordinates
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],

        # Vertices indices
        i=ijk[:, 0],
        j=ijk[:, 1],
        k=ijk[:, 2],
        # opacity=0.5,
        facecolor=facecolors,
        # color='#DC143C',
        flatshading=True
    )
])


fig.show()
# %%
