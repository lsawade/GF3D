from obspy import Stream, Inventory
from gf3d.source import CMTSOLUTION


def process_stream(st: Stream, inv: Inventory | None = None, cmt: CMTSOLUTION | None = None, duration: float | None = None):
    out = st.copy()

    if inv:
        out.detrend("linear")
        out.detrend("demean")
        out.taper(max_percentage=0.05, type='hann')
        out.attach_response(inv)
        out.remove_response(output="DISP", pre_filt=[0.001, 0.005, 45, 50],
                            zero_mean=False, taper=False,
                            water_level=100)
        out.rotate('->ZNE', inventory=inv)

    # Filter
    out.filter('bandpass', freqmin=1/300.0, freqmax=1/40.0, zerophase=True)

    if inv:
        out.merge(fill_value=0.0)

    if (cmt is not None) and (duration is not None):
        out.interpolate(1.0, method='weighted_average_slopes',
                        starttime=cmt.origin_time, npts=int(duration))
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
                newobs.append(obs.select(
                    network=_net, station=_sta, component=_comp)[0])
                newsyn.append(syn.select(
                    network=_net, station=_sta, component=_comp)[0])
            except:
                print(f'Cant find {_net}.{_sta}..{_comp}')

    return Stream(traces=newobs), Stream(traces=newsyn)
