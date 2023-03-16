from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from obspy.imaging.mopad_wrapper import beach

if TYPE_CHECKING:
    from ..source import CMTSOLUTION


def ax_beach(
        cmt: CMTSOLUTION,
        ax: Axes | None = None, x: float = 0.5, y: float = 0.5, width=200, facecolor='k',
        edgecolor='k', bgcolor='w', linewidth=2, alpha=1.0,
        clip_on=False, **kwargs):
    """Plots beach ball into given axes.
        Note that width heavily depends on the given screen size/dpi. Therefore
        often does not work.

    Parameters
    ----------
    cmt : CMTSOLUTION
        CMT solution to be plotted into an axes
    ax : Axes
        Target Axes
    x : float
        Relative x coordinate in [0, 1.0]
    y : float
        Relative y coordinate in [0, 1.0]
    width : int, optional
        beachball width , by default 50
    facecolor : str, optional
        facecolor, by default 'k'
    edgecolor : str, optional
        edgecolor, by default 'k'
    bgcolor : str, optional
        backgroundcolor, by default 'w'
    linewidth : int, optional
        edgecolor linewidth, by default 2
    alpha : float, optional
        tranparency of facecolor, by default 1.0
    clip_on : bool, optional
        whether to clip at axes boundaries, by default False
    """
    if ax is None:
        ax = plt.gca()

    # Plot beach ball
    bb = beach(cmt.tensor,
               linewidth=linewidth,
               facecolor=facecolor,
               bgcolor=bgcolor,
               edgecolor=edgecolor,
               alpha=alpha,
               xy=(x, y),
               width=width,
               size=100,  # Defines number of interpolation points
               axes=ax,
               **kwargs)

    bb.set(clip_on=clip_on)

    # adds beachball to axes
    ax.add_collection(bb)
