import numpy as np
import typing as tp
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition


def adjust_spines(ax: Axes, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])


def plot_label(ax: Axes, label: str, aspect: float = 1,
               location: int = 1, dist: float = 0.025,
               box: tp.Union[bool, dict] = True, fontdict: dict = {},
               **kwargs):
    """Plots label one of the corners of the plot.
    Plot locations are set as follows::

        17  6  14  7  18
            --------
         5 |1  22  2| 8
        13 |21  0 23| 15
        12 |3  24  4| 9
            --------
        20  11 16 10  19

    Tee dist parameter defines the distance between the axes and the text.

    Parameters
    ----------
    label : str
        label
    aspect : float, optional
        aspect ratio length/height, by default 1.0
    location : int, optional
        corner as described by above code figure, by default 1
    aspect : float, optional
        aspect ratio length/height, by default 0.025
    box : bool
        plots bounding box st. the label is on a background, default true
    Notes
    -----
    :Author:
        Lucas Sawade (lsawade@princeton.edu)
    :Last Modified:
        2021.01.26 18.30
    """
    if type(box) is bool:
        if box:
            boxdict = {'facecolor': 'w', 'edgecolor': 'k'}
        else:
            boxdict = {'facecolor': 'none', 'edgecolor': 'none'}
    else:
        boxdict = box

    # Get aspect of the axes
    aspect = 1.0/get_aspect(ax)

    # Inside
    if location == 0:
        ax.text(0.5, 0.5, label,
                horizontalalignment='center', verticalalignment='center_baseline',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 1:
        ax.text(dist, 1.0 - dist * aspect, label, horizontalalignment='left',
                verticalalignment='top', transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 2:
        ax.text(1.0 - dist, 1.0 - dist * aspect, label,
                horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 3:
        ax.text(dist, dist * aspect, label, horizontalalignment='left',
                verticalalignment='bottom', transform=ax.transAxes,
                bbox=boxdict, fontdict=fontdict, **kwargs)
    elif location == 4:
        ax.text(1.0 - dist, dist * aspect, label,
                horizontalalignment='right', verticalalignment='bottom',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    # Outside
    elif location == 5:
        ax.text(-dist, 1.0, label, horizontalalignment='right',
                verticalalignment='top', transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 6:
        ax.text(0, 1.0 + dist * aspect, label, horizontalalignment='left',
                verticalalignment='bottom', transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 7:
        ax.text(1.0, 1.0 + dist * aspect, label,
                horizontalalignment='right', verticalalignment='bottom',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 8:
        ax.text(1.0 + dist, 1.0, label,
                horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 9:
        ax.text(1.0 + dist, 0.0, label,
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 10:
        ax.text(1.0, - dist * aspect, label,
                horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 11:
        ax.text(0.0, -dist * aspect, label, horizontalalignment='left',
                verticalalignment='top', transform=ax.transAxes,
                bbox=boxdict, fontdict=fontdict, **kwargs)
    elif location == 12:
        ax.text(-dist, 0.0, label, horizontalalignment='right',
                verticalalignment='bottom', transform=ax.transAxes,
                bbox=boxdict, fontdict=fontdict, **kwargs)
    elif location == 13:
        ax.text(-dist, 0.5, label, horizontalalignment='right',
                verticalalignment='center_baseline', transform=ax.transAxes,
                bbox=boxdict, fontdict=fontdict, **kwargs)
    elif location == 14:
        ax.text(0.5, 1.0 + dist * aspect, label, horizontalalignment='center',
                verticalalignment='bottom', transform=ax.transAxes,
                bbox=boxdict, fontdict=fontdict, **kwargs)
    elif location == 15:
        ax.text(1 + dist, 0.5, label, horizontalalignment='left',
                verticalalignment='center_baseline', transform=ax.transAxes,
                bbox=boxdict, fontdict=fontdict, **kwargs)
    elif location == 16:
        ax.text(0.5, -dist * aspect, label, horizontalalignment='center',
                verticalalignment='top', transform=ax.transAxes,
                bbox=boxdict, fontdict=fontdict, **kwargs)
    elif location == 17:
        ax.text(- dist, 1.0 + dist * aspect, label,
                horizontalalignment='right', verticalalignment='bottom',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 18:
        ax.text(1.0 + dist, 1.0 + dist * aspect, label,
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 19:
        ax.text(1.0 + dist, 0.0 - dist * aspect, label,
                horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 20:
        ax.text(0.0 - dist, 0.0 - dist * aspect, label,
                horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 21:
        ax.text(0.0 + dist, 0.5, label,
                horizontalalignment='left', verticalalignment='center_baseline',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 22:
        ax.text(0.5, 1.0 - dist * aspect, label,
                horizontalalignment='center', verticalalignment='top',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 23:
        ax.text(1.0 - dist, 0.5, label,
                horizontalalignment='right', verticalalignment='center_baseline',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 24:
        ax.text(0.5, 0.0 + dist * aspect, label,
                horizontalalignment='center', verticalalignment='bottom',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    else:
        raise ValueError("Other corners not defined.")


def axes_from_axes(
        ax: Axes, n: int,
        extent: tp.Iterable = [0.2, 0.2, 0.6, 1.0],
        **kwargs) -> Axes:
    """Uses the location of an existing axes to create another axes in relative
    coordinates. IMPORTANT: Unlike ``inset_axes``, this function propagates
    ``*args`` and ``**kwargs`` to the ``pyplot.axes()`` function, which allows
    for the use of the projection ``keyword``.
    Parameters
    ----------
    ax : Axes
        Existing axes
    n : int
        label, necessary, because matplotlib will replace nonunique axes
    extent : list, optional
        new position in axes relative coordinates,
        by default [0.2, 0.2, 0.6, 1.0]
    Returns
    -------
    Axes
        New axes
    Notes
    -----
    DO NOT CHANGE THE INITIAL POSITION, this position works DO NOT CHANGE!
    :Author:
        Lucas Sawade (lsawade@princeton.edu)
    :Last Modified:
        2021.07.13 18.30
    """

    # Create new axes DO NOT CHANGE THIS INITIAL POSITION
    newax = plt.axes([0.0, 0.0, 0.25, 0.1], label=str(n), **kwargs)

    # Get new position
    ip = InsetPosition(ax, extent)

    # Set new position
    newax.set_axes_locator(ip)

    # return new axes
    return newax


def get_aspect(ax: Axes) -> float:
    """Returns the aspect ratio of an axes in a figure. This works around the
    problem of matplotlib's ``ax.get_aspect`` returning strings if set to
    'equal' for example
    Parameters
    ----------
    ax : Axes
        Matplotlib Axes object
    Returns
    -------
    float
        aspect ratio
    Notes
    -----
    :Author:
        Lucas Sawade (lsawade@princeton.edu)
    :Last Modified:
        2021.01.20 11.30
    """

    # Total figure size
    figW, figH = ax.get_figure().get_size_inches()

    # Axis size on figure
    _, _, w, h = ax.get_position().bounds

    # Ratio of display units
    disp_ratio = (figH * h) / (figW * w)

    return disp_ratio
