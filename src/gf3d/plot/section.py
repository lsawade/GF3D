from obspy import Stream, UTCDateTime
from obspy.geodetics.base import locations2degrees, gps2dist_azimuth

import typing as tp
import numpy as np
import matplotlib.axes
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from gf3d.source import CMTSOLUTION
from gf3d.plot.util import plot_label


def plotsection(obs: Stream, syn: Stream, cmt: CMTSOLUTION,
                *args,
                ax: matplotlib.axes.Axes | None = None, comp='Z',
                limits: tp.Tuple[UTCDateTime] | None = None,
                newsyn: Stream or None = None,
                newcmt: CMTSOLUTION or None = None, scale: float = 1.0,
                **kwargs):

    plt.rcParams["font.family"] = "monospace"

    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.axes()
        plottitle = True
    else:
        plottitle = False

    # Get a single component
    pobs = obs.select(component=comp).copy()
    psyn = syn.select(component=comp).copy()

    if newsyn is not None:
        pnewsyn = newsyn.select(component=comp).copy()
    else:
        pnewsyn = None

        # Get station event distances, labels
    for _i, (_obs, _syn) in enumerate(zip(pobs, psyn)):
        # Assign lats/lons
        latA = cmt.latitude
        lonA = cmt.longitude
        latB = _syn.stats.latitude
        lonB = _syn.stats.longitude

        # Compute distance
        dist = locations2degrees(latA, lonA, latB, lonB)

        # Compute azimuth
        # dist_in_m, az_A2B_deg, az_B2A_deg = gps2dist_azimuth
        _, az_A2B_deg, az_B2A_deg = gps2dist_azimuth(latA, lonA, latB, lonB)

        # Add info to traces
        _obs.stats.distance = dist
        _syn.stats.distance = dist
        _obs.stats.azimuth = az_A2B_deg
        _syn.stats.azimuth = az_A2B_deg
        _obs.stats.backazimuth = az_B2A_deg
        _syn.stats.backazimuth = az_B2A_deg

        if pnewsyn:
            pnewsyn[_i].stats.distance = dist
            pnewsyn[_i].stats.azimuth = az_A2B_deg
            pnewsyn[_i].stats.backazimuth = az_B2A_deg

    # Sort the stream
    pobs.sort(keys=['distance', 'network', 'station'])
    psyn.sort(keys=['distance', 'network', 'station'])

    if pnewsyn:
        pnewsyn.sort(keys=['distance', 'network', 'station'])

    # Get scaling
    if limits:
        slicestart = limits[0]
        sliceend = limits[1]
        absmax = np.max([np.max(np.abs(_tr.copy().slice(slicestart, sliceend).data))
                        for _tr in pobs])
    else:
        absmax = np.max([np.max(np.abs(_tr.data)) for _tr in pobs])

    plot_label(ax, f'max|u|: {absmax:.5g} m',
               fontsize='small', box=False, dist=0.0, location=4)

    # Plot label
    plot_label(ax, f'{comp}', fontweight='bold',
               fontsize='medium', box=False, dist=0.0, location=1)

    # Number of stations
    y = np.arange(1, len(pobs)+1)

    # Set ylabels
    # Set text labels and properties.
    # , rotation=20)
    ax.set_yticks(
        y, [f"{tr.stats.network}.{tr.stats.station}" for tr in pobs],
        verticalalignment='center',
        horizontalalignment='right',
        fontsize='small')
    # TO have epicentral distances on the right
    ax2 = ax.secondary_yaxis("right")
    ax2.set_yticks(
        y, [f"D:{tr.stats.distance:>6.2f} A:{tr.stats.azimuth:>6.2f}" for tr in pobs],
        verticalalignment='center',
        horizontalalignment='left', fontsize='x-small')
    ax2.spines.right.set_visible(False)
    ax2.tick_params(left=False, right=False)

    # Normalize
    for _i, (_obs, _syn, _y) in enumerate(zip(pobs, psyn, y)):

        plt.plot(
            _obs.times('matplotlib'),
            _obs.data / absmax * scale + _y, 'k',
            *args, **kwargs)
        plt.plot(
            _syn.times('matplotlib'),
            _syn.data / absmax * scale + _y, 'r',
            *args, **kwargs)

        if pnewsyn:
            plt.plot(
                pnewsyn[_i].times('matplotlib'),
                pnewsyn[_i].data / absmax * scale + _y, 'b',
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
        ax.set_xlim([lim.datetime for lim in limits])

    plt.xlabel('Time')

    if plottitle:

        if newcmt:
            title = (
                f"    {cmt.cmt_time.ctime()} Loc: {cmt.latitude:.4f}dg, {cmt.longitude:.4f}dg, {cmt.depth:.4f}km, ts={cmt.time_shift:.4f}s, hdur={cmt.hdur:.4f}s - BP: [40s, 300s]\n"
                f"New {newcmt.cmt_time.ctime()} Loc: {newcmt.latitude:.4f}dg, {newcmt.longitude:.4f}dg, {cmt.depth:.4f}km, ts={newcmt.time_shift:.4f}s, hdur={newcmt.hdur:.4f}s")

        else:
            title = (
                f"{cmt.cmt_time.ctime()} Loc: {cmt.latitude:.2f}dg, {cmt.longitude:.2f}dg, {cmt.depth:.1f}km - BP: [20s, 50s]")
        ax.set_title(title, loc='left', ha='left', fontsize='small')
        plt.subplots_adjust(left=0.1, right=0.85, top=0.925)

        return ax
