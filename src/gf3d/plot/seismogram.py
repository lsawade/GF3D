from obspy import Stream, UTCDateTime
from obspy.geodetics.base import locations2degrees, gps2dist_azimuth

import typing as tp
import numpy as np
import matplotlib.axes
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from gf3d.source import CMTSOLUTION
from gf3d.plot.util import plot_label


def plotseismogram(
        obs: Stream, syn: Stream | None, cmt: CMTSOLUTION,
        *args,
        ax: matplotlib.axes.Axes | None = None,
        limits: tp.Tuple[UTCDateTime] | None = None,
        newsyn: Stream or None = None, bandpass=None,
        obsc='k', sync='r', newsync='b', nooffset=False, lw=1,
        **kwargs):

    if ax is None:
        plt.close('all')
        fig = plt.figure(figsize=(7, 4))
        ax = plt.axes()
        axnone = True
    else:
        axnone = False

    if limits is not None:
        starttime, endtime = limits

    for _i, comp in enumerate(['N', 'E', 'Z']):

        observed = obs.select(component=comp)[0]

        if syn is not None:
            synthetic = syn.select(component=comp)[0]

        if newsyn is not None:
            newsynthetic = newsyn.select(component=comp)[0]

        if limits is not None:
            trn = observed.copy()
            trn.trim(starttime=starttime, endtime=endtime)
            absmax = np.max(np.abs(trn.data))
        else:
            absmax = np.max(np.abs(observed.data))

        # Define offset
        if (newsyn is not None) or ((newsyn is None) and (syn is None)) or nooffset:
            absmax_off = 0.0
        else:
            absmax_off = 0.1*absmax

        # To sync x axes of the 3 plots.
        if _i == 0:
            ax = None

        ax = plt.subplot(3, 1, _i+1, sharex=ax)
        plt.plot(observed.times("matplotlib"), observed.data+absmax_off,
                 '-', *args, c=obsc, lw=lw, label='Observed', **kwargs)

        plt.plot([np.min(observed.times("matplotlib")), np.max(observed.times("matplotlib"))],
                 [0, 0], 'k--', lw=lw)
        if syn is not None:
            plt.plot(synthetic.times("matplotlib"), synthetic.data-absmax_off,
                     '-', *args, c=sync, lw=lw, label='Synthetic', **kwargs)

        if newsyn is not None:
            plt.plot(newsynthetic.times("matplotlib"), newsynthetic.data,
                     'b-', *args, c=newsync, lw=lw, label='New Synthetic',
                     **kwargs)

        # Ylabel
        # plt.ylabel(f'{comp}  ')
        plot_label(ax, f'{comp}', dist=0.025, location=13,
                   fontsize='medium', box=False)

        # Axis limits and indicator
        ax.set_ylim(-1.2*absmax, 1.2*absmax)
        plot_label(
            ax, f'A: {absmax:.5g} m', dist=0,
            fontsize='xx-small', box=False)
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.tick_params(labelleft=False, left=False)
        ax.spines.right.set_visible(False)
        ax.spines.left.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.set_xlim(limits)

        if _i == 2:
            plt.xlabel('Time')
        else:
            ax.spines.bottom.set_visible(False)
            ax.tick_params(bottom=False)

        if _i == 0:
            # Add title with event info
            network = obs[0].stats.network
            station = obs[0].stats.station

            if bandpass is not None:
                bandpass_string = f"- BP: {bandpass}s"
            else:
                bandpass_string = ""

            plot_label(ax,
                       f"{cmt.cmt_time.strftime('%Y-%m-%d %H:%M:%S')}  Loc: {cmt.latitude}dg, {cmt.longitude}dg, {cmt.depth}km\n"
                       f"{network}.{station} {bandpass_string}",
                       fontsize='medium', box=False, location=6)

            # Add legend
            plt.legend(frameon=False, loc='upper right',
                       ncol=3, fontsize='x-small')

    if axnone:
        fig.autofmt_xdate()

        # Removes datestamp from N and E axes
        plt.subplots_adjust(
            left=0.1, right=0.9, bottom=0.2, top=0.85, hspace=0.0)
