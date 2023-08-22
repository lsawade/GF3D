# %%
import subprocess
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import axes3d
from functools import partial
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

# %%
# Make random distribution of points on a sphere


def sunflower(n: int, alpha: float) -> np.ndarray:
    # Number of points respectively on the boundary and inside the cirlce.
    n_exterior = np.round(alpha * np.sqrt(n)).astype(int)
    n_interior = n - n_exterior

    # Ensure there are still some points in the inside...
    if n_interior < 1:
        raise RuntimeError(f"Parameter 'alpha' is too large ({alpha}), all "
                           f"points would end-up on the boundary.")
    # Generate the angles. The factor k_theta corresponds to 2*pi/phi^2.
    k_theta = np.pi * (3 - np.sqrt(5))
    angles = np.linspace(k_theta, k_theta * n, n)

    # Generate the radii.
    r_interior = np.sqrt(np.linspace(0, 1, n_interior))
    r_exterior = np.ones((n_exterior,))
    r = np.concatenate((r_interior, r_exterior))

    # Return Cartesian coordinates from polar ones.
    return r * np.stack((np.cos(angles), np.sin(angles)))

    # NOTE: say the returned array is called s. The layout is such that s[0,:]
    # contains X values and s[1,:] contains Y values. Change the above to
    #   return r.reshape(n, 1) * np.stack((np.cos(angles), np.sin(angles)), axis=1)
    # if you want s[:,0] and s[:,1] to contain X and Y values instead.


# Make quick figure to check
N = 100
points = sunflower(N, alpha=2).T
plt.figure()
plt.axes()
plt.plot(points[:, 0], points[:, 1], 'o')
plt.axis('equal')
plt.savefig('sunflower.png', dpi=300)


def uniform_sphere():
    # dimension = number of variables
    d = 3
    # sample size = number of obs
    N = 1000
    # radius of circle
    radius = 2
    # Y ~ MVN(0, I(d))
    Y = np.random.randn(N // d, "Normal")
    # U ~ U(0, 1)
    u = np.random.rand(N, "Uniform")
    # r proportional to d_th root of U
    r = radius * u ** (1/d)
    # Y[,##] is sum of squares for each row
    X = r * Y / np.sqrt(np.sum(Y**2, axis=1))
    return X, Y


# %%
def plotly_anim():
    # Helix equation
    t = np.linspace(0, 10, 50)
    x, y, z = np.cos(t), np.sin(t), t

    fig = go.Figure(go.Scatter3d(x=x, y=y, z=z, mode='markers'))

    x_eye = -1.25
    y_eye = 2
    z_eye = 0.5

    fig.update_layout(
        title='Animation Test',
        width=600,
        height=600,
        scene_camera_eye=dict(x=x_eye, y=y_eye, z=z_eye),
        updatemenus=[dict(type='buttons',
                          showactive=False,
                          y=1,
                          x=0.8,
                          xanchor='left',
                          yanchor='bottom',
                          pad=dict(t=45, r=10),
                          buttons=[dict(label='Play',
                                        method='animate',
                                        args=[None, dict(frame=dict(duration=5, redraw=True),
                                                         transition=dict(
                                              duration=0.0),
                                            fromcurrent=True,
                                            mode='immediate'
                                        )]
                                        )
                                   ]
                          )
                     ]
    )

    def rotate_z(x, y, z, theta):
        w = x+1j*y
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

    frames = []
    for t in np.arange(0, 6.26, 0.01):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(go.Frame(layout=dict(
            scene_camera_eye=dict(x=xe, y=ye, z=ze))))
    fig.frames = frames

    fig.show()


# %%

def pyplot_anim():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Grab some example data and plot a basic wireframe.
    X, Y, Z = axes3d.get_test_data(0.05)
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

    # Set the axis labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Rotate the axes and update
    for angle in range(0, 360*4 + 1):
        # Normalize the angle to the range [-180, 180] for display
        angle_norm = (angle + 180) % 360 - 180

        # Cycle through a full rotation of elevation, then azimuth, roll, and all
        elev = azim = roll = 0
        if angle <= 360:
            elev = angle_norm
        elif angle <= 360*2:
            azim = angle_norm
        elif angle <= 360*3:
            roll = angle_norm
        else:
            elev = azim = roll = angle_norm

        # Update the axis view and title
        print(elev, azim, roll)
        ax.view_init(elev, azim, vertical_axis='z')
        plt.title('Elevation: %d°, Azimuth: %d°, Roll: %d°' %
                  (elev, azim, roll))

        plt.draw()
        plt.pause(.001)


# %%

def BasicPolys(rangex, rangey):
    polys = []
    NX = len(rangex)
    NY = len(rangey)
    polys = np.zeros(((NX-1) * (NY-1), 4, 2))
    counter = 0
    for i in range(NX-1):
        for j in range(NY-1):
            polys[counter, 0, 0] = rangex[i]
            polys[counter, 0, 1] = rangey[j]
            polys[counter, 1, 0] = rangex[i+1]
            polys[counter, 1, 1] = rangey[j]
            polys[counter, 2, 0] = rangex[i+1]
            polys[counter, 2, 1] = rangey[j+1]
            polys[counter, 3, 0] = rangex[i]
            polys[counter, 3, 1] = rangey[j+1]
            counter += 1

    return polys


# %%
# Mathematical convention!!!
# phi = 0 north pole
# theta = 0 x axis, increasing towards y axis
def sphere2cartpoly(thetaphi, r=1):
    Npoly = thetaphi.shape[0]
    xyz = np.zeros((Npoly, 4, 3))
    xyz[:, :, 0] = r * np.sin(np.deg2rad(thetaphi[:, :, 1])) * \
        np.cos(np.deg2rad(thetaphi[:, :, 0]))
    xyz[:, :, 1] = r * np.sin(np.deg2rad(thetaphi[:, :, 1])) * \
        np.sin(np.deg2rad(thetaphi[:, :, 0]))
    xyz[:, :, 2] = r * np.cos(np.deg2rad(thetaphi[:, :, 1]))

    return xyz

# %%


def cartesianpolyshift(xyz, xshift=0.0, yshift=0.0, zshift=0.0):
    # Shift ball
    xyz[:, :, 0] += xshift
    xyz[:, :, 1] += yshift
    xyz[:, :, 2] += zshift


# cartesianpolyshift(xyz, xshift=0.5, yshift=0.5, zshift=0.5)

# %%


# %%


def HueBeach(v, M, ColorIn='white', ColorOut='red'):
    if np.dot(np.dot(M, v), v) > 0:
        return ColorOut
    else:
        return ColorIn


def get_beach_colors(L, poly):
    Npoly = poly.shape[0]
    colors = Npoly * [None]
    DL = np.diag(L)
    for i in range(Npoly):
        colors[i] = HueBeach(np.mean(poly[i, :, :], axis=0), DL)
    return colors


def plot_beachball(xyz, facecolors):

    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False,
                facecolor=None, computed_zorder=True)
    fig.add_axes(ax)
    ax.add_collection3d(Poly3DCollection(
        xyz, edgecolors='k', linewidth=0.1, facecolors=facecolors, alpha=1))
    # plt.axis('off')
    ax.axes.set_xlim3d(-1.1, 1.1)
    ax.axes.set_ylim3d(-1.1, 1.1)
    ax.axes.set_zlim3d(-1.1, 1.1)
    ax.set_box_aspect([1.0, 1.0, 1.0])
    ax.view_init(elev=20., azim=35)
    return fig, ax


def animate_beachball(xyz, facecolors, outfile='beachball.gif'):

    fig, ax = plot_beachball(xyz, facecolors)

    def init():
        return fig,

    def animate(i):
        print(i)
        ax.view_init(elev=20., azim=2*i)
        # fig.savefig(f'frames/frame{i:>010d}.png', transparent=True, dpi=300)
        return fig,

    ani = animation.FuncAnimation(
        fig, animate, 180, init_func=init, interval=1/60, blit=True)

    ani.save(outfile, writer='imagemagick', dpi=150, fps=60,
             savefig_kwargs={'transparent': True})


# %%

ddeg = 10
thetaphi = BasicPolys(np.arange(0, 360+ddeg, ddeg),
                      np.arange(0, 180+ddeg, ddeg))

xyz = sphere2cartpoly(thetaphi, r=1)

# %%
L = [1.0, 19, -20]

facecolors = get_beach_colors(np.array([0.0, 10, -10]), xyz)

plot_beachball(xyz, facecolors)
plt.show()
# animate_beachball(xyz, facecolors)

# %%


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


xref = np.diag([-1, 1, 1])
yref = np.diag([1, -1, 1])
zref = np.diag([1, 1, -1])
theta = np.linspace(0, 360, 1000)
# tL = np.dot(zref, L)
phi = phiN(theta, L)
# tphi = phiN(theta, tL)
# xyz = sphere2cartpoly(np.array([theta, phi]).T, r=1)
Uxyz = sphere2cartline(theta, phi)
# Lxyz = np.dot(zref, Uxyz.T).T
# Lxyz = sphere2cartline(theta, 180-phi)

fig, ax = plot_beachball(xyz, facecolors)
scale = 1.025
ax.plot(
    Uxyz[:, 0]*scale,
    Uxyz[:, 1]*scale,
    Uxyz[:, 2]*scale,
    'r', linewidth=5.0)
ax.plot(
    Lxyz[:, 0]*scale,
    Lxyz[:, 1]*scale,
    Lxyz[:, 2]*scale,
    'b', linewidth=5.0)
plt.show()

# %%


# def plotly_beachball(xyz, facecolors, L):

#     surf = go.Surface(
#         x=X, y=Y, z=z,
#         colorscale=[[0, 'rgb(255,0,0)'], [1, 'rgb(255,0,0)']],
#         showscale=False)
#     fig = go.Figure(data=[surf])


# %%

def subdividequad(point1, point2, point3, point4, f):
    """

    The quad is aligned like so
    .. code::

        4---3
        |   |
        1---2

    Parameters
    ----------
    point1 : Iterable of floats
        2 floats
    point2 : Iterable of floats
        2 floats
    point3 : Iterable of floats
        2 floats
    point4 : Iterable of floats
        2 floats
    f : callable
        function of x that intersects with quad


    Returns
    -------
    Set of polygons

    """
    # Corner points of the quad
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    x4, y4 = point4
    debug = False

    if debug:
        print(f'x1, y1 = {x1, y1}')
        print(f'x2, y2 = {x2, y2}')
        print(f'x3, y3 = {x3, y3}')
        print(f'x4, y4 = {x4, y4}')
        print(f'f(x1), f(x2), f(x3), f(x4) = {f(x1), f(x2), f(x3), f(x4)}')

    # Cuts quad's left and right sides
    # 4---3
    # x   x
    # 1---2
    if (((y1 <= f(x1)) and (f(x1) <= y4)) and
            ((y2 <= f(x2)) and (f(x2) <= y3))):
        print("Cuts quad's left and right sides")
        # Lower quad
        poly1 = [[x1, y1],
                 [x2, y2],
                 [x2, f(x2)],
                 [x1, f(x1)]]
        # Upper quad
        poly2 = [[x4, y4],
                 [x3, y3],
                 [x2, f(x2)],
                 [x1, f(x1)]]

    # Cuts quad's left and bottom side
    # 4---3
    # x   |
    # 1-x-2
    elif (((y1 <= f(x1)) and (f(x1) <= y4)) and
          (f(x2) <= y2)):
        print("Cuts quad's left and bottom side")

        # Lower left triangle
        poly1 = [[x1, f(x1)],
                 [x1, y1],
                 [x1 + ((y1 - f(x1)) * (x2 - x1))/(f(x2) - f(x1)), y1]]

        poly2 = [[x1, f(x1)],
                 [x1 + ((y1 - f(x1)) * (x2 - x1))/(f(x2) - f(x1)), y1],
                 [x2, y2],
                 [x3, y3],
                 [x4, y4]]

    # Cuts quad's bottom and right side
    # 4---3
    # |   x
    # 1-x-2
    elif ((f(x1) <= y1) and
          ((y2 <= f(x2)) and (f(x2) <= y3))):
        print("Cuts quad's bottom and right side")

        # Lower right triangle
        poly1 = [[x1 + ((y1 - f(x1)) * (x2 - x1))/(f(x2) - f(x1)), y1],
                 [x2, y2],
                 [x2, f(x2)]]

        # Remaining pentagon
        poly2 = [[x1 + ((y1 - f(x1)) * (x2 - x1))/(f(x2) - f(x1)), y1],
                 [x2, f(x2)],
                 [x3, y3],
                 [x4, y4],
                 [x1, y1]]

    # Cuts top and right side of quad:
    # 4-x-3
    # |   x
    # 1---2
    elif ((y4 <= f(x1)) and
            ((y2 <= f(x2)) and (f(x2) <= y3))):
        print("Cuts top and right side of quad")

        # Upper right triangle
        poly1 = [[x1 + ((y3 - f(x1)) * (x2 - x1))/(f(x2) - f(x1)), y3],
                 [x3, y3],
                 [x3, f(x3)]]

        # Remaining pentagon
        poly2 = [[x1 + ((y3 - f(x1)) * (x2 - x1))/(f(x2) - f(x1)), y3],
                 [x3, f(x3)],
                 [x2, y2],
                 [x1, y1],
                 [x4, y4]]

    # Cuts left and top side of quad:
    # 4-x-3
    # x   |
    # 1---2
    elif (((y1 <= f(x1)) and (f(x1) <= y4)) and
            (y3 <= f(x2))):
        print("Cuts left and top side of quad")

        # Upper left triangle
        poly1 = [[x1 + ((y3 - f(x1)) * (x2 - x1))/(f(x2) - f(x1)), y3],
                 [x4, y4],
                 [x1, f(x1)]]

        # Remaining pentagon
        poly2 = [[x1 + ((y3 - f(x1)) * (x2 - x1))/(f(x2) - f(x1)), y3],
                 [x3, y3],
                 [x2, y2],
                 [x1, y1],
                 [x1, f(x1)]]

    # Cuts top and bottom side of quad:
    # 4-x-3
    # |   |
    # 1-x-2
    elif ((f(x2) <= y2) and (y4 <= f(x4)) or
            (f(x1) <= y1) and (y3 <= f(x3))):
        print("Cuts top and bottom side of quad")

        # Left quad
        poly1 = [[x4 + ((y1-f(x4))*(x2-x4))/(f(x2)-f(x4)), y1],
                 [x2, y2],
                 [x2, y3],
                 [x4 + ((y4-f(x4))*(x2-x4))/(f(x2)-f(x4)), y3]]

        # Right quad
        poly2 = [[x4 + ((y1-f(x4))*(x2-x4))/(f(x2)-f(x4)), y1],
                 [x1, y1],
                 [x4, y4],
                 [x4 + ((y4-f(x4))*(x2-x4))/(f(x2)-f(x4)), y3]]

    else:
        return [[point1, point2, point3, point4],]

    return [poly1, poly2]


def triangleArea(tri):
    """Heron's formula"""

    a = np.linalg.norm(tri[1, :] - tri[0, :])
    b = np.linalg.norm(tri[2, :] - tri[1, :])
    c = np.linalg.norm(tri[0, :] - tri[2, :])

    s = 0.5 * (a + b + c)

    return np.sqrt(s * (s - a) * (s - b) * (s - c))


def pentagonCOM(poly):
    s1 = np.array([0, 1, 4])
    s2 = np.array([1, 3, 4])
    s3 = np.array([1, 2, 3])

    A1 = triangleArea(poly[s1, :])
    A2 = triangleArea(poly[s2, :])
    A3 = triangleArea(poly[s3, :])

    c1 = np.mean(poly[s1, :], axis=0)
    c2 = np.mean(poly[s2, :], axis=0)
    c3 = np.mean(poly[s3, :], axis=0)

    return (A1*c1 + A2*c2 + A3*c3) / (A1 + A2 + A3)


def rectangleCOM(poly):
    s1 = np.array([0, 1, 2])
    s2 = np.array([2, 3, 0])

    A1 = triangleArea(poly[s1, :])
    A2 = triangleArea(poly[s2, :])

    c1 = np.mean(poly[s1, :], axis=0)
    c2 = np.mean(poly[s2, :], axis=0)

    return (A1*c1 + A2*c2) / (A1 + A2)

# %%


def plot_split(poly, f):

    ax = plt.gca()

    colors = ['yellow', 'orange']

    polys = subdividequad(*poly, f)

    ax.add_collection(PolyCollection([poly], edgecolors='k', linewidth=0.25,
                                     facecolors='none',))

    ax.add_collection(PolyCollection(polys, edgecolors='k', linewidth=0.25,
                                     facecolors=colors[: len(polys)], alpha=0.75))

    for _poly in polys:
        if len(_poly) == 5:
            print('Pentagon')
            v = np.array(_poly)
            xc, yc = pentagonCOM(v)
        elif len(_poly) == 4:
            print('Rectangle')
            v = np.array(_poly)
            xc, yc = rectangleCOM(v)
        elif len(_poly) == 3:
            print('Triangle')
            v = np.array(_poly)
            xc, yc = np.mean(v, axis=0)

        else:
            continue

        print('COM(x,y):', xc, yc)
        ax.plot(xc, yc, 'kx', lw=2.0)

    x = np.linspace(-0.5, 2.5, 5)
    y = f(x)
    ax.plot(x, y, 'k', lw=2.0)

    x1, y1 = poly[0]
    x2, y2 = poly[1]
    x3, y3 = poly[2]
    x4, y4 = poly[3]

    kwargs = dict(ha='center', va='center', backgroundcolor='w')
    ax.text(x1-0.2, y1-0.2, '(x1,y1)', **kwargs)
    ax.text(x2+0.2, y2-0.2, '(x2,y2)', **kwargs)
    ax.text(x3+0.2, y3+0.2, '(x3,y3)', **kwargs)
    ax.text(x4-0.2, y4+0.2, '(x4,y4)', **kwargs)

    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.25, 1.25)

    ax.axis('off')

# %%


x1, x4 = 0, 0
x2, x3 = 2, 2
y1, y2 = 0, 0
y3, y4 = 1, 1
def f1(x): return 2 * x + 0.5
def f2(x): return 0.5 * x - 0.5
def f3(x): return 2.0 * x + 1.5


point1 = [x1, y1]
point2 = [x2, y2]
point3 = [x3, y3]
point4 = [x4, y4]

poly = np.array([point1, point2, point3, point4])

# %%
plt.figure(figsize=(9, 1.5))
plt.subplot(1, 3, 1)
plot_split(poly, f1)
plt.subplot(1, 3, 2)
plot_split(poly, f2)
plt.subplot(1, 3, 3)
plot_split(poly, f3)
plt.show()


# %%
# Use the subdivide function to subdivide a beachball using phin
# as the function that intersects the quad

# pphiN = partial(phiN, L=L)
# subdividewithlambda(poly, pphiN)


def subdividewithlambda(poly, L):
    """Subdivides using a function that takes in a function relative to the x coordinates of the quad

    Parameters
    ----------
    poly : iterable
        polygon, list of 4 lists of 2 floats
    f : callable
        function of x that intersects with quad

    Returns
    -------
    list of polygons
    """

    if (((L[0] >= 0) and (L[1] >= 0) and (L[2] >= 0)) or
            ((L[0] <= 0) and (L[1] <= 0) and (L[2] <= 0))):
        return [poly]

    else:
        pphiN = partial(phiN, L=L)
        polys = subdividequad(*poly, pphiN)

        if len(polys) == 2:
            return polys
        else:
            def tpphiN(theta):
                return 180-pphiN(theta)

            return subdividequad(*poly, tpphiN)


polys = []
ddeg = 10
thetaphi = BasicPolys(np.arange(0, 360+ddeg, ddeg),
                      np.arange(0, 180+ddeg, ddeg))
thetaphi = BasicPolys(np.arange(0, 20+ddeg, ddeg),
                      np.arange(70, 90+ddeg, ddeg))
L = np.array([1.0, 19, -20])
for i in range(thetaphi.shape[0]):
    polys.extend(subdividewithlambda(thetaphi[i, :, :], L))


# xyz = sphere2cartpoly(thetaphi, r=1)
# %%


def sphere2cartpoly(poly, r=1):
    outpolys = []
    Npoly = len(poly)

    for i in range(Npoly):
        tpoly = np.array(poly[i])
        x = r * np.sin(np.deg2rad(tpoly[:, 1])) * \
            np.cos(np.deg2rad(tpoly[:, 0]))
        y = r * np.sin(np.deg2rad(tpoly[:, 1])) * \
            np.sin(np.deg2rad(tpoly[:, 0]))
        z = r * np.cos(np.deg2rad(tpoly[:, 1]))
        outpolys.append(np.vstack((x, y, z)).T.tolist())

    return outpolys


xyzpolys = sphere2cartpoly(polys, r=1)

# %% get polygon colors


def get_beach_colors(L, poly):
    Npoly = len(poly)
    colors = Npoly * [None]
    DL = np.diag(L)
    for i in range(Npoly):
        v = np.array(poly[i])
        x = np.mean(v, axis=0)
        colors[i] = HueBeach(x, DL)

    return colors


facecolors = get_beach_colors(L, xyzpolys)

# %%
plot_beachball(xyzpolys, facecolors)
plt.show()
