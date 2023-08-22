# %%
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
# %%
#
# Legendre polynomials and derivatives.


def legendre(n, x):
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    else:
        return ((2*n-1)/n * x * legendre(n-1, x) - (n-1)/n * legendre(n-2, x))

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


def get_ellipticity(rq):
    """Loads ellipticity for given radius r in [0, 1] from specfem data.
    r=0 is center of spherical Earth and r=1 is the surface"""

    # Loading the elliptiity data from specfem
    ell = np.loadtxt("DATA/test_ellipticity.dat")

    # Get the gradient of the ellipticity
    rell = ell[:, 0]/6371.0
    eps = ell[:, 1]
    depsdr = ell[:, 2]/6371.0
    # depsdr = np.gradient(ell[:, 1], ell[:, 0])

    # # Remove Nan values (those are at discontinuities)
    # nonnan = ~np.isnan(depsdr)
    # eps = ell[nonnan, 1]
    # depsdr = depsdr[nonnan]

    # Interpolate values
    epsilon = np.interp(rq, rell, eps)
    eta = np.interp(rq, rell, depsdr)

    return epsilon, eta


def spherical_unit_vectors(theta, phi):
    """
    theta: tuple (phi, theta)
    """
    rhat = np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    that = np.array([
        np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    phat = np.array([-np.sin(phi), np.cos(phi), 0])
    return rhat, that, phat

# %%


def elliptical_r(f, r, theta):
    return r * (1 - 2/3 * f * legendre(2, np.cos(theta)))


def make_ellipsis(x, y, z, r, f=0, resolution=20):
    """Return the coordinates for plotting a sphere centered at (x,y,z)"""
    phi, theta = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]

    # Make ellipticial radius for each theta
    tmp_theta = np.linspace(0, np.pi, resolution*1, endpoint=True)
    tmp_radius = elliptical_r(f, r, tmp_theta)
    # Repeat the radius for each phi
    radius = np.tile(tmp_radius, (resolution*2, 1))

    X = radius * np.cos(phi)*np.sin(theta) + x
    Y = radius * np.sin(phi)*np.sin(theta) + y
    Z = radius * np.cos(theta) + z
    return (X, Y, Z)

# %%


def add_vector(fig, x, v, color='rgb(255,0,0)', scale=1,
               arrow_tip_ratio=0.1,
               arrow_starting_ratio=0.98,
               markerkwargs=dict(),
               linekwargs=dict(),
               conekwargs=dict(),
               name=None,
               annotation=None,
               annotation_scale=1.025
               ):

    markertrace = go.Scatter3d(
        x=[x[0],],
        y=[x[1],],
        z=[x[2],],
        mode='markers',
        line=dict(color=color),
        marker=dict(color=color, **markerkwargs),
        showlegend=False,
    )

    linetrace = go.Scatter3d(
        x=[x[0], x[0] + v[0]*scale],
        y=[x[1], x[1] + v[1]*scale],
        z=[x[2], x[2] + v[2]*scale],
        mode='lines',
        line=dict(color=color, **linekwargs),
        showlegend=name is not None,
        name=name
    )

    conetrace = go.Cone(
        x=[x[0] + arrow_starting_ratio*v[0],],
        y=[x[1] + arrow_starting_ratio*v[1],],
        z=[x[2] + arrow_starting_ratio*v[2],],
        u=[arrow_tip_ratio*v[0],],
        v=[arrow_tip_ratio*v[1],],
        w=[arrow_tip_ratio*v[2],],
        showlegend=False,
        showscale=False,
        colorscale=[[0, color], [1, color]],
        **conekwargs
    )

    fig.add_trace(markertrace)
    fig.add_trace(linetrace)
    fig.add_trace(conetrace)

    if annotation is not None:

        texttrace = go.Scatter3d(
            x=[x[0] + v[0]*annotation_scale,],
            y=[x[1] + v[1]*annotation_scale,],
            z=[x[2] + v[2]*annotation_scale,],
            mode='text',
            line=dict(color=color, **linekwargs),
            showlegend=name is not None,
            text=annotation,
            textposition="middle center",
        )

        fig.add_trace(texttrace)


def pol2cart(r, theta):
    return r * np.sin(theta), r * np.cos(theta)


def sphere2cart(r, theta, phi):
    return r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)

# Yes, that’s exactly the situation. So the red and blue arrows change their
# orientation, meaning that the “vertical” and “radial” are changed, but the
# “transverse” (in-and-out of the plane) is not. By taking the gradient of
# the right-hand side of equation (14.4), which defines the surface of an
# ellipsoid, we can get an (unnormalized) normal vector to the ellipsoid:


def get_tangent_normal(r0, theta0, epsilon=0, eta=0):
    """Due to ellipticity also returns:
        x, y, normal, tangent"""

    # Get actual radius
    re = elliptical_r(epsilon, r0, theta0)

    # Get cartesian coordinates
    x, y = pol2cart(re, theta0)

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

    # Thus the geodetic normal becomes in xyz coordinates
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

    return x, y, n, t


def get_tangent_normal_3D(r0, theta0, phi0, epsilon=0, eta=0):
    """Due to ellipticity also returns:
        x, y, normal, tangent"""

    # Get actual radius
    re = elliptical_r(epsilon, r0, theta0)

    # Get cartesian coordinates
    x, y, z = sphere2cart(re, theta0, phi0)

    # The gradient components with respect to the geocentric coordinates are
    drdr = 1 - (2/3)*(epsilon + eta) * legendre(2, np.cos(theta0))
    drdtheta = (2/3)*epsilon * dlegendre(2, np.cos(theta0)) * np.sin(theta0)
    drdphi = 0

    # This is of the form n = a \hat{r} + b \hat{theta}, so we can normalize n
    # by requiring a ^ 2+b ^ 2 = 1, i.e., dividing by(a ^ 2+b ^ 2) ^ {1/2}.
    norm = np.sqrt(drdr**2 + drdtheta**2 + drdphi**2)

    # where eta is given by(14.15).

    # Note that we are using the geocentric colatitude
    rhat, that, phat = spherical_unit_vectors(theta0, phi0)

    # Thus the geodetic normal becomes in xyz coordinates
    n = (drdr * rhat + drdtheta * that + drdphi * phat) / norm
    tt = (-drdtheta * rhat + drdr * that + drdphi * phat) / norm
    tp = np.cross(n, tt)

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

    return (x, y, z), n, tt, tp


def add_ellipsis(fig, x, r, f=0, color='rgb(255,0,0)', **kwargs):

    # Make ellitpticical sphere
    X = make_ellipsis(x[0], x[1], x[2], r, f=f)

    fig.add_trace(go.Surface(x=X[0], y=X[1], z=X[2],
                             colorscale=[[0, color], [1, color]],
                             showscale=False, **kwargs,
                             showlegend=False,))


# %%

# # plotly.offline.init_notebook_mode()
f = 0.25
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=-.3, y=1.2, z=0.6)
)

axdict = dict(
    range=[-2.2, 2.2],
    showspikes=False,
    gridcolor='rgba(0,0,0,0)',
    backgroundcolor='rgba(0,0,0,0)',
    zerolinecolor='rgba(0,0,0,0)',
    showticklabels=False,
    title_text='')

fig = go.Figure(layout=dict(
    scene=dict(
        xaxis=axdict,
        yaxis=axdict,
        zaxis=axdict,
        aspectratio=dict(x=1, y=1, z=1)
    ),
    margin=dict(r=0, l=0, b=0, t=0),
    scene_camera=camera,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    legend=dict(
        x=0,
        y=1,
        traceorder="normal"
    )
)
)

r0, theta0, phi0 = 1, np.pi*1/3, np.pi*1/2


axis_markerdict = dict(size=3)
for i in range(3):
    vmin, vmax = [0, 0, 0], [0, 0, 0]
    vmin[i], vmax[i] = -2.0, 4.0
    print(vmin, vmax)
    add_vector(fig, vmin, vmax, color='rgb(0,0,0)',
               markerkwargs=axis_markerdict, arrow_starting_ratio=0.99,
               arrow_tip_ratio=0.05, annotation='xyz'[i], annotation_scale=1.025)

add_ellipsis(fig, [0, 0, 0], 1, f=0,
             color='rgb(255,255,0)', opacity=0.4)
add_ellipsis(fig, [0, 0, 0], 1, f=f,
             color='rgb(0,255,255)', opacity=0.4)

vector_markerdict = dict(size=2)

for i in range(0, 180+15, 15):

    r0, theta0, phi0 = 1, np.pi/180 * i, np.pi/180 * 2 * i
    XC, NC, TTC, TPC = get_tangent_normal_3D(
        r0, theta0, phi0, epsilon=0, eta=0)
    XE, NE, TTE, TPE = get_tangent_normal_3D(
        r0, theta0, phi0, epsilon=f, eta=0)

    if i == 15:

        name_rhat_s = r'$\mathbf{\hat{r}}_S$'
        name_that_s = r'$\boldsymbol{\hat{\theta}}_S$'
        name_phat_s = r'$\boldsymbol{\hat{\phi}}_S$'
        name_rhat_e = r'$\mathbf{\hat{r}}_E$'
        name_that_e = r'$\boldsymbol{\hat{\theta}}_E$'
        name_phat_e = r'$\boldsymbol{\hat{\phi}}_E$'

    else:

        name_rhat_s = None
        name_that_s = None
        name_phat_s = None
        name_rhat_e = None
        name_that_e = None
        name_phat_e = None

    add_vector(fig, XC, NC, color='rgb(255,0,0)',
               name=name_rhat_s, markerkwargs=vector_markerdict)
    add_vector(fig, XC, TTC, color='rgb(0,0,255)',
               name=name_that_s, markerkwargs=vector_markerdict)
    add_vector(fig, XC, TPC, color='rgb(0,0,0,0.75)',
               name=name_phat_s, markerkwargs=vector_markerdict)
    add_vector(fig, XE, NE, color='rgb(255,0,0)',
               name=name_rhat_e, linekwargs=dict(dash='dash'),
               markerkwargs=vector_markerdict)
    add_vector(fig, XE, TTE, color='rgb(0,0,255)',
               name=name_that_e, linekwargs=dict(dash='dash'),
               markerkwargs=vector_markerdict)
    add_vector(fig, XE, TPE, color='rgb(0,0,0, 0.75)',
               name=name_phat_e, linekwargs=dict(dash='dash'),
               markerkwargs=vector_markerdict)


# fig.update_traces()
fig.show()

# plotly.offline.iplot(fig, filename='simple-3d-scatter')
# %%
pio.write_image(fig, 'figure.pdf', format='pdf',
                width=500, height=500, scale=3.0)

# %%
fig.write_html("figure.html", include_mathjax='cdn', include_plotlyjs='cdn')

# %%
