import h5py
import pickle
import os
import numpy as np
import plotly
import plotly.graph_objs as go
from gf3d.geoutils import geo2cart
from gf3d.coordinate.ingeopoly import ingeopoly


def ms(x, y, z, radius, resolution=20):
    """Return the coordinates for plotting a sphere centered at (x,y,z)"""
    u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
    X = radius * np.cos(u)*np.sin(v) + x
    Y = radius * np.sin(u)*np.sin(v) + y
    Z = radius * np.cos(v) + z
    return (X, Y, Z)

def meshplot(stationfile, outfile, land=None):



    with h5py.File(stationfile, 'r') as db:

        ibool = db['ibool'][:]
        xyz = db['xyz'][:]

    NSPEC = ibool.shape[-1]

    # %%
    x = []
    y = []
    z = []

    # for i in range(NSPEC):
    for i in range(0,NSPEC):

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


    continents_file = './landareas.pkl'
    if os.path.exists(continents_file):
        land = True
        with open(continents_file, 'rb') as f:
            landareas = pickle.load(f)
    else:
        land = False

    # %%
    axdict = {
        'showgrid': False,  # thin lines in the background
        'zeroline': False,  # thick line at x=0
        'visible': False,  # numbers below
        'showgrid': False
    }

    data = [
        go.Scatter3d(x=x, y=y, z=z, mode='lines', line={'color': 'rgb(0, 0, 0)'}, name='Elements'),
        ]

    if land:
        data.append(
            go.Surface(
                x=landareas['x'], y=landareas['y'], z=landareas['z'],
                showscale=False,
                colorscale=[[0, '#0000FF'], [1,'#0000FF']],
                surfacecolor=np.zeros_like(np.array(landareas['x']), dtype=float),
                name='Lands',
                opacity=0.25,
                contours=go.surface.Contours(
                    x=go.surface.contours.X(highlight=False),
                    y=go.surface.contours.Y(highlight=False),
                    z=go.surface.contours.Z(highlight=False),
                    ),
                lightposition=dict(x=0, y=0, z=0),
                lighting=dict(ambient=1.0),
                hoverinfo='skip',
                hovertemplate=None
            ))

        # data.append(
        #     go.Scatter3d(
        #         x=landareas['xl'], y=landareas['yl'], z=landareas['zl'],
        #         mode='lines',
        #         line={'color': 'rgb(50, 50, 50)'},
        #         name='Continents'),)

    # Inner sphere
    X,Y,Z = ms(0.0, 0.0, 0.0, (6371-750)/6371, 180)
    colorscale = [[0, '#FFFFFF'],
                  [1, '#FFFFFF']]

    data.append(
        go.Surface(
                x=X, y=Y, z=Z, showscale=False, surfacecolor=[0 for _ in range(len(Z))],
                colorscale=colorscale, #[[0, 'white'], [1,'white']],
                name='Inner Sph.',
                opacity=1.0,
                contours=go.surface.Contours(
                x=go.surface.contours.X(highlight=False),
                y=go.surface.contours.Y(highlight=False),
                z=go.surface.contours.Z(highlight=False),
                    ),
                lightposition=dict(x=0, y=0, z=0),
                lighting=dict(ambient=1.0),
                hoverinfo='skip',
                hovertemplate=None
            )
        )

    print('data length', len(data))

    fig = go.Figure(data=data)

    fig.update_layout(
        scene=dict(
            xaxis=axdict, yaxis=axdict, zaxis=axdict),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        width=1920, height=1080,
    )
    # fig.update_scences(
    #     scene=dict(
    #         xaxis=go.layout.scene.XAxis(showspikes=False),
    #         yaxis=go.layout.scene.YAxis(showspikes=False),
    #         zaxis=go.layout.scene.ZAxis(showspikes=False)

    #     )
    # )

    plotly.offline.plot(fig, filename=outfile)
