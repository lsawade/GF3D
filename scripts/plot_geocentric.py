
# %%

import matplotlib.pyplot as plt
import numpy as np

# %%


def legendre(n, x):
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    else:
        return ((2*n-1)/n * x * legendre(n-1, x) - (n-1)/n * legendre(n-2, x))
#

# Derivative of Legendre polynomials


def dlegendre(n, x):
    if n == 0:
        return np.zeros_like(x)
    elif n == 1:
        return np.ones_like(x)
    else:
        sum = 0
        sum += (2*(n-1) + 1) * legendre(n-1, x)

        if n >= 2:
            for i in range(2, n, 2):
                sum += (2*((n-1) - i) + 1) * legendre((n-1)-i, x)

        return sum


# %% Remaking the plot from Wiki

x = np.linspace(-1, 1, 1000)

plt.figure()

ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)

for n in range(4):
    ax1.plot(x, legendre(n, x), label=f"n={n}")
    ax2.plot(x, dlegendre(n, x), label=f"dn={n}")

ax1.legend(ncol=3, frameon=False)
ax2.legend(ncol=3, frameon=False)

plt.show(block=False)


# %%

theta = np.linspace(0, 2*np.pi, 1000)
r = np.ones_like(theta)

# f = 1/299.8
f = 1/4
e = np.sqrt(2*f - f**2)
f2 = 1-np.sqrt(1-e**2)

ONE_MINUS_F_SQUARED = 1 - f**2
FACTOR_TAN = 1.0 / ONE_MINUS_F_SQUARED
# %%
TINYVAL = 1e-10
DEGREES_TO_RADIANS = np.pi/180
# 1/((1-f) ^ 2) = 1.0067046409645724
# rfix = r * (1 - 2/3 * (1-(1-f**2))())


def geocentric_to_geographic_colat(theta):
    return np.pi/2 - np.arctan(1/(1-f**2) * np.cos(theta) /
                               np.maximum(TINYVAL, np.sin(theta)))


# def geodetic_to_geocentric(lat_prime):
    # return np.pi - np.arctan((1 - f**2) * np.tan(lat * DEGREES_TO_RADIANS))


def pol2cart(r, theta):
    return r * np.sin(theta), r * np.cos(theta)


def elliptical_r(f, r, theta):
    return r * (1 - 2/3 * f * legendre(2, np.cos(theta)))


theta_prime = geocentric_to_geographic_colat(theta)
theta = np.linspace(0, 2*np.pi, 1000)
r = np.ones_like(theta)

rfix = elliptical_r(f, r, theta)

plt.figure()
plt.plot([-1.25, 1.25], [0, 0], 'k-')
plt.plot([0, 0], [-1.25, 1.25], 'k-')
plt.plot(*pol2cart(r, theta), label="Geocentric")
plt.plot(*pol2cart(rfix, theta), label="Geodetic")
plt.axis('equal')
# plt.plot(*pol2cart(r, theta_prime), label="Geocentric")
plt.show(block=False)

# %%


def er(theta):
    return np.array([np.sin(theta), np.cos(theta)])


def derdtheta(theta):
    return np.array([np.cos(theta), -np.sin(theta)])


def tangent(theta):
    return derdtheta(theta) * 1 + er(theta) * 0


def normal(theta):
    return er(theta) * 1 - derdtheta(theta) * 0


def plot_vector(x, y, v, scale, color='k', arrowkwargs: dict | None = None, **kwargs, ):
    ax = plt.gca()
    ax.plot([x, x+v[0]*scale], [y, y+v[1]*scale], c=color, **kwargs)

    if arrowkwargs is None:
        arrowkwargs = dict(head_width=0.05, head_length=0.1)

    ax.arrow(x+v[0]*scale, y+v[1]*scale, v[0]*scale*0.1, v[1]*scale*0.1, shape='full',
             lw=0, length_includes_head=True, fc=color, ec=color, **arrowkwargs)


t = tangent(np.pi/4)
n = normal(np.pi/4)

x, y = pol2cart(1, np.pi/4)

plt.figure()
plt.plot([-1.25, 1.25], [0, 0], 'k-')
plt.plot([0, 0], [-1.25, 1.25], 'k-')
plt.plot(*pol2cart(r, theta), 'k-', label="Geocentric")
plt.plot(*pol2cart(rfix, theta), 'k-', label="Geodetic")
plot_vector(x, y, t, 0.5, color='b', label="Tangent")
plot_vector(x, y, n, 0.5, color='r', label="Normal")
plt.axis('equal')
plt.legend()
# plt.plot(*pol2cart(r, theta_prime), label="Geocentric")
plt.show(block=False)


# %%

# Load ellipticity
ell = np.loadtxt("DATA/test_ellipticity.dat")

# %%
depsdr = np.gradient(ell[:, 1], ell[:, 0])

# %% Remove Nan values
nonnan = ~np.isnan(depsdr)
eps = ell[nonnan, 1]
r = ell[nonnan, 0]/6371.0
depsdr = depsdr[nonnan]


def depsdr_query(rq):
    return np.interp(rq, r, depsdr)


def eps_query(rq):
    return np.interp(rq, r, eps)


# %%
f_earth = eps_query(1)
df_earth = depsdr_query(1)

# %%


def tangent_ell(r, theta, f, df):
    return derdtheta(theta) * r * (1 - 2.0/3.0 * f * legendre(2, np.cos(theta))) + er(theta) * 2/3 * f * dlegendre(2, np.cos(theta)) * (np.sin(theta))


def normal_ell(r, theta, f, df):
    return er(theta) * r * (1 - 2.0/3.0 * f * legendre(2, np.cos(theta))) - derdtheta(theta) * 2/3 * f * dlegendre(2, np.cos(theta)) * (np.sin(theta))


theta = np.linspace(0, 2*np.pi, 1000)
r = np.ones_like(theta)

f = f_earth
df = df_earth

# f = 1/4
# df = 1

theta0 = np.pi/4
tc = tangent(theta0)
nc = normal(theta0)

xc, yc = pol2cart(1, theta0)

re = elliptical_r(f, 1, theta0)
te = tangent_ell(1, theta0, f, df)
ne = normal_ell(1, theta0, f, df)
xe, ye = pol2cart(re, theta0)

# degree offset
roffset = 0.01*np.pi/180
xp, yp = pol2cart(1, theta0+roffset)
xm, ym = pol2cart(1, theta0-roffset)


# %%

zoom = True
plt.close()
plt.figure(figsize=(6, 5))
plt.plot([-1.25, 1.25], [0, 0], 'k-')
plt.plot([0, 0], [-1.25, 1.25], 'k-')
plt.plot(*pol2cart(r, theta), 'k-', label="Geocentric")
plt.plot(*pol2cart(elliptical_r(f, r, theta), theta), 'k-', label="Geodetic")
# plt.arrow(xm*scale, ym*scale, (xp-xm)*scale, (yp-ym) *
#           scale, shape='full', head_starts_at_zero=True)


def norm(x):
    return np.sqrt(np.sum(x**2))


if zoom:
    scale = 0.999

    plt.plot([0, xm], [0, ym], 'k--', lw=1.0)
    plt.plot([0, xp], [0, yp], 'k--', lw=1.0)
    plt.plot([xm*scale, xp*scale], [ym*scale, yp*scale], 'k.-', lw=1.0)
    plt.text(0.5*(xp+xm)*scale, 0.5*(yp+ym)*scale,
             f'{2*roffset*180/np.pi:.2f}$^\circ$\n{2*roffset*6371*scale:.2f} km', ha='left', va='bottom')
    plt.text(xc+0.0001, yc,
             f'Angle between Normals:\n{np.arccos(np.dot(nc,ne)/(norm(nc)*norm(ne)))*180/np.pi:.4f}$^\circ$', ha='left', va='center')

    plt.text(*pol2cart(elliptical_r(f, 1.000025, theta0-0.0005), theta0-0.0005),
             f'Ellipsoid', ha='center', va='center', rotation=-45)
    plt.text(*pol2cart(1.000025, theta0-0.0005),
             f'Sphere', ha='center', va='center', rotation=-45)

    head_width = 0.001
    scale = 0.0004
    arrowkwargs = dict(
        width=0.00001
        # head_width=head_width, head_length=head_width*5
    )
else:
    scale = 0.5
    arrowkwargs = dict(
        width=0.01
        # head_width=head_width, head_length=head_width*5
    )

plot_vector(xc, yc, tc/np.linalg.norm(tc, ord=2), scale,
            color='b', label="T", arrowkwargs=arrowkwargs)
plot_vector(xc, yc, nc/np.linalg.norm(nc, ord=2), scale,
            color='r', label="N", arrowkwargs=arrowkwargs)
plot_vector(xe, ye, te/np.linalg.norm(te, ord=2), scale,
            color='b', ls='--', arrowkwargs=arrowkwargs)
plot_vector(xe, ye, ne/np.linalg.norm(ne, ord=2), scale,
            color='r', ls='--',  arrowkwargs=arrowkwargs)
plt.axis('equal')
plt.axis('off')

if zoom:
    offset = 0.001
    plt.xlim(1/np.sqrt(2) - offset, 1/np.sqrt(2) + offset)
    plt.ylim(1/np.sqrt(2) - offset + 0.2*offset,
             1/np.sqrt(2) + offset - 0.7*offset)
    plt.legend(loc='upper right', frameon=False)
else:
    plt.legend(loc='upper left', frameon=False)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
# plt.plot(*pol2cart(r, theta_prime), label="Geocentric")
plt.show(block=False)


# %%
# Yes, that’s exactly the situation. So the red and blue arrows change their
# orientation, meaning that the “vertical” and “radial” are changed, but the
# “transverse” (in-and-out of the plane) is not. By taking the gradient of
# the right-hand side of equation (14.4), which defines the surface of an
# ellipsoid, we can get an (unnormalized) normal vector to the ellipsoid:
r0 = 1
theta0 = 3*np.pi/4

epsilon = eps_query(r0)
eta = depsdr_query(r0)

epsilon = 1/4
eta = 0

re = elliptical_r(epsilon, r0, theta0)
xe, ye = pol2cart(re, theta0)
xc, yc = pol2cart(r0, theta0)

# The gradient components with respect to the geocentric coordinates are
drdr = 1 - (2/3)*(epsilon + eta) * legendre(2, np.cos(theta0))
drdtheta = (2/3)*epsilon * dlegendre(2, np.cos(theta0)) * np.sin(theta0)

# This is of the form n = a \hat{r} + b \hat{theta}, so we can normalize n
# by requiring a ^ 2+b ^ 2 = 1, i.e., dividing by(a ^ 2+b ^ 2) ^ {1/2}.
norm = np.sqrt(drdr**2 + drdtheta**2)

# where eta is given by(14.15).

# Note that we are using the geocentric colatitude
rhat = np.array([np.sin(theta0), np.cos(theta0)])
that = np.array([np.cos(theta0), -np.sin(theta0)])

# Thus the geodetic normal becomes
n = (drdr * rhat + drdtheta * that) / norm
t = (-drdtheta * rhat + drdr * that) / norm

# Thus the projection operator onto the ellipsoid is
# I – nn = (1 – a ^ 2) \hat{r}\hat{r} + (1 – b ^ 2) \hat{theta}\hat{theta}
#           + \hat{phi}\hat{phi} – ab(\hat{r} \hat{theta} + \hat{theta}  \hat{r})

# Thus, the new vertical component becomes
# Sz = n dot s

# and the new horizontal component becomes pointing in north/south(???) direction
# Ss = (I – n n) dot s

# Note that the \hat{phi} component, that is the transverse component,
# remains the same, as expected. So after this you obtain new vertical can
# radial components.


def norm(x):
    return np.sqrt(np.sum(x**2))


tc = tangent(theta0)
nc = normal(theta0)

# %%


def plot_base():

    r = np.ones_like(theta)

    # degree offset
    roffset = 0.01*np.pi/180
    xp, yp = pol2cart(1, theta0+roffset)
    xm, ym = pol2cart(1, theta0-roffset)

    plt.close()
    plt.figure(figsize=(6, 5))
    plt.plot([-1.25, 1.25], [0, 0], 'k-')
    plt.plot([0, 0], [-1.25, 1.25], 'k-')
    plt.plot(*pol2cart(r, theta), 'k-', lw=0.5)
    plt.plot(*pol2cart(elliptical_r(epsilon, r, theta), theta), 'k-', lw=0.5)

# %%


plot_base()

zoom = False

if zoom:

    plt.plot([0, xm], [0, ym], 'k--', lw=1.0)
    plt.plot([0, xp], [0, yp], 'k--', lw=1.0)
    plt.plot([xm*scale, xp*scale], [ym*scale, yp*scale], 'k.-', lw=1.0)
    plt.text(0.5*(xp+xm)*scale, 0.5*(yp+ym)*scale,
             f'{2*roffset*180/np.pi:.2f}$^\circ$\n{2*roffset*6371*scale:.2f} km', ha='left', va='bottom')
    plt.text(xc+0.0002, yc+0.0001,
             f'Angle between Normals:\n{np.arccos(np.dot(nc,n)/(norm(nc)*norm(n)))*180/np.pi:.4f}$^\circ$', ha='left', va='center')

    plt.text(*pol2cart(elliptical_r(epsilon, 1.00005, theta0-0.0005), theta0-0.0005),
             f'Ellipsoid', ha='center', va='center', rotation=-theta0*180/np.pi)
    plt.text(*pol2cart(1.00005, theta0-0.0005),
             f'Sphere', ha='center', va='center', rotation=-theta0*180/np.pi)

    head_width = 0.001
    scale = 0.0008
    arrowkwargs = dict(
        width=0.00002
        # head_width=head_width, head_length=head_width*5
    )

    width = 0.00002
    qdict = dict(
        scale=1250,
        # lw=1.5,
        angles='xy',
        scale_units='xy',
        width=0.00001,
        facecolor='none',
        linewidth=2,
        headwidth=1/width * 0.05,
        headlength=1/width * 0.05
    )
    dasheddict = dict(
        linestyle='dashed',

    )
else:
    width = 0.00001
    qdict = dict(
        scale=1250,
        # lw=1.5,
        angles='xy',
        scale_units='xy',
        width=0.00001,
        facecolor='none',
        linewidth=2,
        headwidth=1/width * 0.1,
        headlength=1/width * 0.1
    )
    dasheddict = dict(linestyle='dashed')

plt.quiver(xc, yc, nc[0], nc[1], edgecolor=['r'], **qdict)
plt.quiver(xc, yc, tc[0], tc[1], edgecolor=['b'], **qdict)
plt.quiver(xe, ye, n[0], n[1], edgecolor=['r'], **qdict, **dasheddict)
plt.quiver(xe, ye, t[0], t[1], edgecolor=['b'], **qdict, **dasheddict)

plt.plot([1000, 1000], [1001, 1001], 'r',
         label=r'$\mathbf{\hat{r}}$')
plt.plot([1000, 1000], [1001, 1001], 'b',
         label=r'$\mathbf{\hat{\theta}}$')
plt.axis('equal')
plt.axis('off')

if zoom:
    scale = 2.0
    xplot, yplot = (xc+xe)/2, (yc+ye)/2
    offset = 0.5 * (np.abs(elliptical_r(epsilon, 1, 0) - r0) +
                    np.abs(elliptical_r(epsilon, 1, np.pi/2) - r0))
    plt.xlim(xplot - offset, xplot + offset)
    plt.ylim(yplot - offset, yplot + offset)
    plt.legend(frameon=False)
else:
    plt.legend(frameon=False)
    plt.xlim(-1.25, 1.25)
    plt.ylim(-1.25, 1.25)

plt.title(f'Zoom at {theta0*180/np.pi:.2f}$^\circ$ Colatitude')
plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
# plt.plot(*pol2cart(r, theta_prime), label="Geocentric")
plt.show(block=False)
