from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
    from .cmt import CMTCatalog

try:
    import pygmt
except ImportError:
    print("You need to install pygmt to use this function. Try: pip install pygmt")


def plot_global_gmt_map(cat: CMTCatalog, outfile='global_eventmap.pdf',
                        projection='mercator', central_longitude=0.0,
                        title=None|str):
    """Plot global map of events in catalog.

    Parameters
    ----------
    cat : CMTCatalog
        Catalog to plot
    outfile : str, optional
        File to output the plot to, by default 'global_eventmap.pdf'
    projection : str, optional
        Projection to use, by default 'mercator', other values: 'mollweide'
    central_longitude : float, optional
        Central longitude to use, by default 0.0
    title : str, optional
        Title to use, by default None

    Notes
    -----

    I did very little to make this function, this was mainly done by Github
    Copilot.

    """

    # Get latitude and longitudes from catalog
    latitudes = cat.latitude
    longitudes = cat.longitude
    depth = cat.depth
    moment_tensors = cat.tensor

    # Get the exponents of the moment tensors
    exponents = np.floor(np.log10(np.abs(np.max(moment_tensors, axis=1))))

    # Create the spec array required by the GMT plotting tool
    spec = np.vstack((longitudes, latitudes, depth,
                      (moment_tensors/10**exponents[:, None]).T,
                      exponents)).T

    # Make global GMT figure
    if projection == 'mercator':
        projection = f"M{central_longitude}/0/15c"
        region=[0, 360, -80, 80]
        frame = ["afg"]

    elif projection == 'mollweide':
        projection = f'W{central_longitude}/15c'
        region="g"
        frame = ["afg"]

    else:
        raise ValueError(f"Projection {projection} not recognized.")

    # Add title
    if isinstance(title, str):
        frame = frame + [f"+t\"{title}\""]

    fig = pygmt.Figure()
    # Make global basemap
    fig.coast(
        region=region,
        projection=projection,
        land="lightgray",
        water="white",
        shorelines="thinnest",
    )

    # Load global topography data. This is a 1 arc-minute grid.
    grid_globe = pygmt.datasets.load_earth_relief(resolution="06m")

    # Load earth relief using pygmt
    pygmt.makecpt(cmap="geo", series=[-8000, 8000])
    fig.grdimage(grid=grid_globe, frame="a")

    # Plot the events with beachball using moment_tensors and moment_magnitude
    fig.meca(
        spec=spec,
        # borders="1/thick",
        convention="mt",
        compressionfill="red",
        # Fill extensive quadrants with color "cornsilk"
        # [Default is "white"]
        extensionfill="cornsilk",
        # Draw a 0.5 points thick dark gray ("gray30") solid outline via
        # the pen parameter [Default is "0.25p,black,solid"]
        pen="0.01p,gray30,solid",
        # label="",
        # verbose=True,
        scale=".2c+s5.5",
        # frame=['a']
        frame=frame

    )

    # Tranform moment tensors to pygmt spec format
    fig.savefig(outfile)

