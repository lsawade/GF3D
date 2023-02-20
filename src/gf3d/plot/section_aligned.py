from obspy import Stream
from obspy.geodetics.base import locations2degrees, gps2dist_azimuth
from obspy.taup import TauPyModel

import typing as tp
import numpy as np
import matplotlib.axes
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from gf3d.source import CMTSOLUTION
from gf3d.plot.util import plot_label


def get_azimuth_distance_traveltime(
        cmt, obs, syn,
        traveltime_window: None | tp.Tuple[str, tp.Tuple[float, float]],
        comp='Z', newsyn=None, vlove=4.4, vrayleigh=3.7, orbit=1,
        model: str = "ak135"):

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

    if traveltime_window is not None:
        # Phase
        phase = traveltime_window[0]
        trims = traveltime_window[1]

        if phase.lower() == 'love':

            for _i, _tr in enumerate(pobs):
                dist = (orbit - 1) * 180.0 + ((orbit-1) % 2) * \
                    180 + (-1)**(orbit-1) * _tr.stats.distance
                distkm = dist * 111.11  # km/deg
                _tr.stats.traveltime = distkm/vlove
                _tr.stats.tt_correction_type = 'linear'
                _tr.stats.tt_correction_velocity = vlove
                _tr.stats.label = f'L{orbit}({vlove:.2f} km/s)'

        elif phase.lower() == 'rayleigh':

            for _i, _tr in enumerate(pobs):
                dist = (orbit - 1) * 180.0 + ((orbit-1) % 2) * \
                    180 + (-1)**(orbit-1) * _tr.stats.distance

                distkm = dist * 111.11  # km/deg
                _tr.stats.traveltime = distkm/vrayleigh
                _tr.stats.tt_correction_type = 'linear'
                _tr.stats.tt_correction_velocity = vrayleigh
                _tr.stats.label = f'R{orbit}({vrayleigh:.2f} km/s)'

        else:

            # Initialize Taup model
            model = TauPyModel(model=model)

            poppable = []
            for _i, _tr in enumerate(pobs):

                arrivals = model.get_travel_times(
                    source_depth_in_km=cmt.depth,
                    distance_in_degree=_tr.stats.distance,
                    phase_list=[phase, ])

                if len(arrivals) == 0:
                    _tr.stats.traveltime = None
                    poppable.append(_i)
                    _tr.stats.label = None
                else:
                    _tr.stats.traveltime = arrivals[0].time
                    _tr.stats.tt_correction_type = 'phase'
                    _tr.stats.tt_correction_velocity = None
                    _tr.stats.label = f'{phase}-Wave'

            for _pop in poppable[::-1]:

                pobs.pop(_pop)
                psyn.pop(_pop)
                if pnewsyn:
                    pnewsyn.pop(_pop)

        for _i, (_obstr, _syntr) in enumerate(zip(pobs, psyn)):

            # Set up trace by trace interpolation
            dt = 0.1
            starttime = cmt.origin_time + _obstr.stats.traveltime + trims[0]
            npts = int(np.round((trims[1] - trims[0])/dt))

            # Interpolation arguments for the inteprolation
            iargs = 1.0/dt,
            ikwargs = dict(method='lanczos', starttime=starttime,
                           npts=npts, time_shift=0.0, a=20)

            # Interpolation
            _obstr.interpolate(*iargs, **ikwargs)
            _syntr.interpolate(*iargs, **ikwargs)

            if pnewsyn:
                pnewsyn[_i].interpolate(*iargs, **ikwargs)

    if newsyn is not None:
        return pobs, psyn, pnewsyn
    else:
        return pobs, psyn


def filter_stations(obs1, obs2):
    selection = []
    for _i, (_obs1, _obs1) in enumerate(zip(obs1, obs2)):
        if (_obs1.stats.traveltime is None) and (_obs2.stats.traveltime is None):
            continue
        else:
            selection.append(_i)

    return selection


def plotsection_aligned(obs: Stream, syn: Stream, cmt: CMTSOLUTION,
                        traveltime_window: None | tp.Tuple[str, tp.Tuple[float, float]],
                        *args,
                        ax: matplotlib.axes.Axes | None = None, comp='Z',
                        newsyn: Stream | None = None,
                        labelright: bool = True, labelleft: bool = True,
                        **kwargs):

    plt.rcParams["font.family"] = "monospace"

    if ax is None:
        plt.figure(figsize=(9, 6))
        ax = plt.axes()
        plottitle = True
    else:
        plottitle = False

    # Get window and phase
    phase = traveltime_window[0]
    trims = traveltime_window[1]

    # Get a single component
    pobs = obs.select(component=comp).copy()
    psyn = syn.select(component=comp).copy()

    if newsyn is not None:
        pnewsyn = newsyn.select(component=comp).copy()
    else:
        pnewsyn = None

    # Get scaling
    # absmax = np.max(pobs.max())
    absmax = np.max([np.max(np.abs(_tr.data)) for _tr in pobs])

    plot_label(ax, f'max|u|: {absmax:.5g} m',
               fontsize='small', box=False, dist=0.0, location=7)

    # Plot label
    plot_label(ax, f'{comp}', fontweight='bold',
               fontsize='medium', box=False, dist=0.0, location=6)

    # Number of stations
    y = np.arange(1, len(pobs)+1)

    # Set ylabels
    # Set text labels and properties.
    # , rotation=20)
    ax.set_yticks(
        y, [f"{tr.stats.network}.{tr.stats.station}" for tr in pobs],
        verticalalignment='center',
        horizontalalignment='right', fontsize='x-small')
    ax.tick_params(
        left=False, right=False, labelleft=labelleft, pad=50.0)

    # TO have epicentral distances on the right
    ax2 = ax.secondary_yaxis("right")
    ax2.set_yticks(
        y, [f"D:{tr.stats.distance:>6.2f} A:{tr.stats.azimuth:>6.2f}" for tr in pobs],
        verticalalignment='center',
        horizontalalignment='left',
        fontsize='x-small')
    ax2.spines.right.set_visible(False)
    ax2.tick_params(
        left=False, right=False, labelright=labelright, pad=-10.0)

    # Normalize
    xlabel = None
    for _i, (_obs, _syn, _y) in enumerate(zip(pobs, psyn, y)):

        if _obs.stats.traveltime is None:
            continue

        if _obs.stats.label is not None:
            xlabel = _obs.stats.label

        # If the traveltime is used set the reference traveltime to the
        # P arrival time
        reftime = cmt.origin_time + _obs.stats.traveltime

        # Condition for "Planned sections"
        plt.plot(
            _obs.times(type='relative', reftime=reftime),
            _obs.data / absmax + _y, 'k',
            *args, **kwargs)
        plt.plot(
            _syn.times(type='relative', reftime=reftime),
            _syn.data / absmax + _y, 'r',
            *args, **kwargs)

        if pnewsyn:
            plt.plot(
                pnewsyn[_i].times(type='relative', reftime=reftime),
                pnewsyn[_i].data / absmax + _y, 'b',
                *args, **kwargs)

    top = len(pobs) + 1.0
    bottom = -0.5
    plt.plot([0, 0], [bottom, top], 'k-',
             lw=ax.spines.bottom.get_linewidth())

    ax.set_ylim(bottom, top)
    ax.spines.bottom.set_bounds(trims[0], trims[1])

    # Remove all spines
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    ax.tick_params(left=False, right=False, pad=0)
    ax.spines.left.set_visible(False)

    plt.xlabel(f'{xlabel} offset [s]')

    return
