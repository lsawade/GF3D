import h5py
import plotly
import plotly.graph_objs as go
from lwsspy.GF.geoutils import geo2cart


def meshplot(stationfile, outfile, land=None):

    with h5py.File(stationfile, 'r') as db:

        ibool = db['ibool'][:]
        xyz = db['xyz'][:]

    NSPEC = ibool.shape[-1]

    # %%
    x = []
    y = []
    z = []

    for i in range(NSPEC):

        x.extend([
            xyz[ibool[0, 0, 0, i], 0], xyz[ibool[4, 0, 0, i], 0], None,
            xyz[ibool[0, 0, 0, i], 0], xyz[ibool[0, 4, 0, i], 0], None,
            xyz[ibool[0, 0, 0, i], 0], xyz[ibool[0, 0, 4, i], 0], None,
            xyz[ibool[4, 0, 0, i], 0], xyz[ibool[4, 0, 4, i], 0], None,
            xyz[ibool[4, 0, 0, i], 0], xyz[ibool[4, 4, 0, i], 0], None,
            xyz[ibool[0, 4, 0, i], 0], xyz[ibool[4, 4, 0, i], 0], None,
            xyz[ibool[0, 4, 0, i], 0], xyz[ibool[0, 4, 4, i], 0], None,
            xyz[ibool[0, 0, 4, i], 0], xyz[ibool[0, 4, 4, i], 0], None,
            xyz[ibool[0, 0, 4, i], 0], xyz[ibool[4, 0, 4, i], 0], None,
            xyz[ibool[4, 4, 4, i], 0], xyz[ibool[4, 4, 0, i], 0], None,
            xyz[ibool[4, 4, 4, i], 0], xyz[ibool[0, 4, 4, i], 0], None,
            xyz[ibool[4, 4, 4, i], 0], xyz[ibool[4, 0, 4, i], 0], None,
        ])

        y.extend([
            xyz[ibool[0, 0, 0, i], 1], xyz[ibool[4, 0, 0, i], 1], None,
            xyz[ibool[0, 0, 0, i], 1], xyz[ibool[0, 4, 0, i], 1], None,
            xyz[ibool[0, 0, 0, i], 1], xyz[ibool[0, 0, 4, i], 1], None,
            xyz[ibool[4, 0, 0, i], 1], xyz[ibool[4, 0, 4, i], 1], None,
            xyz[ibool[4, 0, 0, i], 1], xyz[ibool[4, 4, 0, i], 1], None,
            xyz[ibool[0, 4, 0, i], 1], xyz[ibool[4, 4, 0, i], 1], None,
            xyz[ibool[0, 4, 0, i], 1], xyz[ibool[0, 4, 4, i], 1], None,
            xyz[ibool[0, 0, 4, i], 1], xyz[ibool[0, 4, 4, i], 1], None,
            xyz[ibool[0, 0, 4, i], 1], xyz[ibool[4, 0, 4, i], 1], None,
            xyz[ibool[4, 4, 4, i], 1], xyz[ibool[4, 4, 0, i], 1], None,
            xyz[ibool[4, 4, 4, i], 1], xyz[ibool[0, 4, 4, i], 1], None,
            xyz[ibool[4, 4, 4, i], 1], xyz[ibool[4, 0, 4, i], 1], None,
        ])

        z.extend([
            xyz[ibool[0, 0, 0, i], 2], xyz[ibool[4, 0, 0, i], 2], None,
            xyz[ibool[0, 0, 0, i], 2], xyz[ibool[0, 4, 0, i], 2], None,
            xyz[ibool[0, 0, 0, i], 2], xyz[ibool[0, 0, 4, i], 2], None,
            xyz[ibool[4, 0, 0, i], 2], xyz[ibool[4, 0, 4, i], 2], None,
            xyz[ibool[4, 0, 0, i], 2], xyz[ibool[4, 4, 0, i], 2], None,
            xyz[ibool[0, 4, 0, i], 2], xyz[ibool[4, 4, 0, i], 2], None,
            xyz[ibool[0, 4, 0, i], 2], xyz[ibool[0, 4, 4, i], 2], None,
            xyz[ibool[0, 0, 4, i], 2], xyz[ibool[0, 4, 4, i], 2], None,
            xyz[ibool[0, 0, 4, i], 2], xyz[ibool[4, 0, 4, i], 2], None,
            xyz[ibool[4, 4, 4, i], 2], xyz[ibool[4, 4, 0, i], 2], None,
            xyz[ibool[4, 4, 4, i], 2], xyz[ibool[0, 4, 4, i], 2], None,
            xyz[ibool[4, 4, 4, i], 2], xyz[ibool[4, 0, 4, i], 2], None,
        ])

    if land is not None:

        xl = []
        yl = []
        zl = []
        for shape in land.shapes():
            lon, lat = zip(*shape.points)
            xt, yt, zt = geo2cart(1.001, lat, lon)

            xl.extend([*xt, None])
            yl.extend([*yt, None])
            zl.extend([*zt, None])

    # %%
    axdict = {
        'showgrid': False,  # thin lines in the background
        'zeroline': False,  # thick line at x=0
        'visible': False,  # numbers below
        'showgrid': False
    }

    fig = go.Figure(data=go.Scatter3d(x=x, y=y, z=z, mode='lines'))

    fig.update_layout(
        scene=dict(
            xaxis=axdict, yaxis=axdict, zaxis=axdict),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    plotly.offline.plot(fig, filename=outfile)
