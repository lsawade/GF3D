from obspy import Stream, Inventory
from gf3d.source import CMTSOLUTION
from gf3d.logger import logger


def process_stream_trace_by_trace(
        st: Stream, inv: Inventory | None = None, cmt: CMTSOLUTION | None = None,
        duration: float | None = None,
        bandpass=[40.0, 300.0], starttimeoffset: float = 0.0):
    cpstream = st.copy()
    out = []

    for trace in cpstream:
        try:
            logger.debug(f"Processing {trace.id}")

            if inv:
                trace.detrend("linear")
                trace.detrend("demean")
                trace.taper(max_percentage=0.05, type='hann')
                trace.remove_response(output="DISP", pre_filt=[0.003, 0.005, 45, 50],
                                      zero_mean=False, taper=False,
                                      water_level=100, inventory=inv)

            # Filter
            trace.filter('bandpass', freqmin=1.0 /
                         bandpass[1], freqmax=1.0/bandpass[0], zerophase=True)

            if (cmt is not None) and (duration is not None):
                trace.interpolate(1.0, method='weighted_average_slopes',
                                  starttime=cmt.origin_time + starttimeoffset, npts=int(duration))

            out.append(trace)

        except Exception as e:

            print(72*'=')
            print(trace.id)
            print(72*'-')
            print(e)
            print(72*'=', '\n')

    out = Stream(traces=out)
    if inv:
        out.rotate('->ZNE', inventory=inv)

    return out


def process_stream(
        st: Stream, inv: Inventory | None = None, cmt: CMTSOLUTION | None = None,
        duration: float | None = None,
        bandpass=[40.0, 300.0], starttimeoffset: float = 0.0):
    out = st.copy()

    out.taper(max_percentage=0.05, type='hann')
    out.detrend("linear")
    if inv:
        out.detrend("linear")
        out.detrend("demean")
        out.taper(max_percentage=0.05, type='hann')
        out.remove_response(output="DISP", pre_filt=[0.003, 0.005, 45, 50],
                            zero_mean=False, taper=False,
                            water_level=100, inventory=inv)
        out.rotate('->ZNE', inventory=inv)

    # Filter
    out.filter('bandpass', freqmin=1.0 /
               bandpass[1], freqmax=1.0/bandpass[0], zerophase=True)

    if inv:
        out.merge(fill_value=0.0)

    if (cmt is not None) and (duration is not None):
        out.interpolate(1.0, method='weighted_average_slopes',
                        starttime=cmt.origin_time + starttimeoffset, npts=int(duration))
    return out


def select_pairs(obs, syn):

    stations = set()
    for tr in obs:
        stations.add((tr.stats.network, tr.stats.station))
    for tr in syn:
        stations.add((tr.stats.network, tr.stats.station))

    # Grabbing the actual pairs
    newobs, newsyn = [], []
    for _net, _sta in stations:
        for _comp in ['N', 'E', 'Z']:
            try:
                obstr = obs.select(
                    network=_net, station=_sta, component=_comp)[0]
                syntr = syn.select(
                    network=_net, station=_sta, component=_comp)[0]
                newobs.append(obstr)
                newsyn.append(syntr)
            except:
                print(f'Cant find {_net}.{_sta}..{_comp}')

    return Stream(traces=newobs), Stream(traces=newsyn)
