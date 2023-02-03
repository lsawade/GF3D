from obspy import Stream
from obspy.geodetics.base import locations2degrees
import typing as tp
import numpy as np
import matplotlib.axes
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from lwsspy.GF.source import CMTSOLUTION
from lwsspy.GF.plotutil import plot_label


def plotsection(obs: Stream, syn: Stream, cmt: CMTSOLUTION,
                *args,
                ax: matplotlib.axes.Axes | None = None, comp='Z',
                limits: tp.Tuple[float] | None = None,
                newsyn: Stream or None = None,
                newcmt: CMTSOLUTION or None = None,
                ** kwargs):

    plt.rcParams["font.family"] = "monospace"

    if ax is None:
        plt.figure(figsize=(9, 6))
        ax = plt.axes()
        plottitle = True
    else:
        plottitle = False

    # Get a single component
    pobs = obs.select(component=comp)
    psyn = syn.select(component=comp)

    if newsyn is not None:
        pnewsyn = newsyn.select(component=comp)
    else:
        pnewsyn = None

    # Get station event distances, labels
    for _i, (_obs, _syn) in enumerate(zip(pobs, psyn)):
        dist = locations2degrees(
            _syn.stats.latitude, _syn.stats.longitude,
            cmt.latitude, cmt.longitude)

        _obs.stats.distance = dist
        _syn.stats.distance = dist

        if pnewsyn:
            setattr(pnewsyn[_i].stats, 'distance', dist)

    # Sort the stream
    pobs.sort(keys=['distance', 'network', 'station'])
    psyn.sort(keys=['distance', 'network', 'station'])

    if pnewsyn:
        pnewsyn.sort(keys=['distance', 'network', 'station'])

    # Get scaling
    absmax = np.max(pobs.max())
    plot_label(ax, f'max|u|: {absmax:.5g} m',
               fontsize='small', box=False, dist=0.0, location=4)

    # Plot label
    plot_label(ax, f'{comp} component',
               fontsize='medium', box=False, dist=0.0, location=1)

    # Number of stations
    y = np.arange(1, len(pobs)+1)

    # Set ylabels
    # Set text labels and properties.
    # , rotation=20)
    ax.set_yticks(y, [f"{tr.stats.network}.{tr.stats.station}" for tr in pobs])

    # TO have epicentral distances on the right
    ax2 = ax.secondary_yaxis("right")
    ax2.set_yticks(y, [f"{tr.stats.distance:>6.2f}" for tr in pobs])
    ax2.spines.right.set_visible(False)
    ax2.tick_params(left=False, right=False)

    # Normalize
    for _i, (_obs, _syn, _y) in enumerate(zip(pobs, psyn, y)):
        plt.plot(_obs.times('matplotlib'), _obs.data / absmax + _y, 'k',
                 *args, **kwargs)
        plt.plot(_syn.times('matplotlib'), _syn.data / absmax + _y, 'r',
                 *args, **kwargs)

        if pnewsyn:
            plt.plot(pnewsyn[_i].times('matplotlib'), pnewsyn[_i].data / absmax + _y, 'b',
                     *args, **kwargs)

    # Remove all spines
    ax.spines.top.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.right.set_visible(False)
    ax.tick_params(left=False, right=False)

    # Format x axis to have the date
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    if limits is not None:
        ax.set_xlim(limits)

    plt.xlabel('Time')

    if plottitle:

        if newcmt:
            title = (
                f"    {cmt.cmt_time.ctime()} Loc: {cmt.latitude:.4f}dg, {cmt.longitude:.4f}dg, {cmt.depth:.4f}km, ts={cmt.time_shift:.4f}s, hdur={cmt.hdur:.4f}s - BP: [40s, 300s]\n"
                f"New {newcmt.cmt_time.ctime()} Loc: {newcmt.latitude:.4f}dg, {newcmt.longitude:.4f}dg, {cmt.depth:.4f}km, ts={newcmt.time_shift:.4f}s, hdur={newcmt.hdur:.4f}s")

        else:
            title = (
                f"{cmt.cmt_time.ctime()} Loc: {cmt.latitude:.2f}dg, {cmt.longitude:.2f}dg, {cmt.depth:.1f}km - BP: [40s, 300s]")
        ax.set_title(title, loc='left', ha='left', fontsize='small')
        plt.subplots_adjust(left=0.1, right=0.9, top=0.925)

        return ax
