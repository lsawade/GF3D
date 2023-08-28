from obspy import Stream, UTCDateTime
from obspy.geodetics.base import locations2degrees, gps2dist_azimuth

import typing as tp
import numpy as np
import matplotlib.axes
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from gf3d.source import CMTSOLUTION
from gf3d.plot.util import plot_label


def add_header(ax,
               station: str | None = None,
               station_latitude: float | None = None,
               station_longitude: float | None = None,
               station_azimuth: float | None = None,
               station_back_azimuth: float | None = None,
               station_distance_in_degree: float | None = None,
               event: str | None = None,
               event_time: UTCDateTime | None = None,
               event_latitude: float | None = None,
               event_longitude: float | None = None,
               event_depth_in_km: float | None = None,
               event_Mw: float | None = None,
               bandpass: tp.List[float] | None = None,
               **kwargs):

    label = ""

    if event is not None:
        label += f"{event}: "

    if event_time is not None:
        label += f"{event_time.strftime('%Y-%m-%d %H:%M:%S')}  "

    if event_latitude is not None:
        label += f"LA {event_latitude:.3f} "

    if event_longitude is not None:
        label += f"LO {event_longitude:.3f} "

    if event_depth_in_km is not None:
        label += f"DP {event_depth_in_km:.3f}km"

    if event_Mw is not None:
        label += f"Mw: {event_Mw:.2f}"

    if any([event, event_time, event_latitude, event_longitude,
            event_depth_in_km, event_Mw]):
        label += "\n"

    if station is not None:
        label += f"{station}: "

    if station_latitude is not None:
        label += f"LA {station_latitude:.3f} "

    if station_longitude is not None:
        label += f"LO: {station_longitude:.3f} "

    if station_azimuth is not None:
        label += f"AZ: {station_azimuth:.1f}dg "

    if station_back_azimuth is not None:
        label += f"BAZ: {station_back_azimuth:.1f}dg "

    if station_distance_in_degree is not None:
        label += f"DIST: {station_distance_in_degree:.3f}dg "

    if bandpass is not None:
        label += f"- BP: {bandpass[0]:d}-{bandpass[1]:d}s"

    # Set some default kwargs for plot_label
    if 'fontsize' not in kwargs:
        kwargs['fontsize'] = 'medium'

    if 'location' not in kwargs:
        kwargs['location'] = 6

    # Finally add label
    plot_label(ax, label, box=False, **kwargs)
