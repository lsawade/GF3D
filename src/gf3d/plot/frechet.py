import os
from gf3d.plot.util import plot_label
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates


def plotfrechet(cmt, rp, drp, network, station, limits, outdir, comp='Z',
                rp2=None, drp2=None, lw=1.0):

    N = 1 + len(drp.keys())
    keys = list(drp.keys())

    factor = 1.5
    fig, axes = plt.subplots(N, 1, figsize=(8, 4.5), gridspec_kw={
        'height_ratios': [factor, *((1,)*(N-1))]})

    for i in range(N):

        if i == 0:
            key = 'Syn'
            tr = rp.select(network=network, station=station, component=comp)[0]

            if rp2 is not None:
                tr2 = rp2.select(
                    network=network, station=station, component=comp)[0]
            else:
                tr2 = None
        else:

            key = keys[i-1]
            tr = drp[key].select(
                network=network, station=station, component=comp)[0]

            if drp2 is not None:
                tr2 = drp2[key].select(network=network,
                                       station=station, component=comp)[0]
            else:
                tr2 = None

        plt.sca(axes[i])
        plt.plot(tr.times("matplotlib"), tr.data, 'k', lw=lw)
        if tr2 is not None:
            plt.plot(tr2.times("matplotlib"), tr2.data, 'r--', lw=lw)

        absmax = np.max(np.abs(tr.data))
        axes[i].set_ylim(-1.01*absmax*factor, 1.01*absmax*factor)
        plot_label(
            axes[i], f'{key}', location=1, dist=0.0,
            fontsize='small', box=False)

        plot_label(
            axes[i], f'max|A|: {absmax:.5g}', location=2, dist=0.001,
            fontsize='x-small', box=False)

        axes[i].xaxis_date()
        axes[i].xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(axes[i].xaxis.get_major_locator()))
        axes[i].tick_params(labelleft=False, left=False)
        axes[i].spines.right.set_visible(False)
        axes[i].spines.left.set_visible(False)
        axes[i].spines.top.set_visible(False)

        axes[i].set_xlim(limits)

        if i == N-1:
            plt.xlabel('Time')
        else:
            axes[i].spines.bottom.set_visible(False)

        if i == 0:
            # Add title with event info
            net = tr.stats.network
            sta = tr.stats.station
            loc = tr.stats.location
            cha = tr.stats.channel
            plt.title(
                f"{net}.{sta}.{loc}.{cha} -- {cmt.cmt_time.strftime('%Y-%m-%d %H:%M:%S')}  Loc: {cmt.latitude}dg, {cmt.longitude}dg, {cmt.depth}km", fontsize='medium')

    fig.autofmt_xdate()
    plt.subplots_adjust(
        left=0.075, right=0.925, bottom=0.15, top=0.95, hspace=0.0)
    plt.savefig(os.path.join(outdir, 'frechet.pdf'), dpi=300)
    plt.close()
