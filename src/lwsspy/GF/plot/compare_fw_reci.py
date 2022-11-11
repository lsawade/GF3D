import os
from lwsspy.plot import plot_label
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates


def compare_fw_reci(cmt, rp, fw, limits, outdir):

    fig = plt.figure(figsize=(10, 6))
    lw = 0.25
    for _i, comp in enumerate(['N', 'E', 'Z']):

        forward = fw.select(component=comp)[0]
        recipro = rp.select(component=comp)[0]

        ax = plt.subplot(3, 1, _i+1)
        plt.plot(recipro.times("matplotlib"), recipro.data,
                 'k', lw=lw, label='Reciprocal')
        plt.plot(forward.times("matplotlib"), forward.data,
                 'r-', lw=lw, label='Forward')
        plt.plot(forward.times("matplotlib")[
            ::28], forward.data[::28], 'r--', lw=lw, label='Fw. subsampled')
        plt.ylabel(f'{comp}  ', rotation=0)

        absmax = np.max(np.abs(recipro.data))
        ax.set_ylim(-1.375*absmax, 1.375*absmax)
        plot_label(
            ax, f'max|u|: {absmax:.5g} m',
            fontsize='x-small', box=False)
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

        if _i == 0:
            # Add title with event info
            network = rp[0].stats.network
            station = rp[0].stats.station
            plt.title(
                f"{network}.{station} -- {cmt.cmt_time} Loc: {cmt.latitude}dg, {cmt.longitude}dg, {cmt.depth}km")

            # Add legend
            plt.legend(frameon=False, loc='lower right',
                       ncol=3, fontsize='x-small')

    fig.autofmt_xdate()
    plt.subplots_adjust(
        left=0.05, right=0.95, bottom=0.1, top=0.95)
    plt.savefig(os.path.join(outdir, 'demo_single.pdf'), dpi=300)
    plt.close()


def plot_drp(cmt, rp, drp, limits, outdir, comp='Z'):

    fig = plt.figure(figsize=(10, 8))

    N = 1 + len(drp.keys())
    keys = list(drp.keys())
    for i in range(N):

        lw = 0.25

        if i == 0:
            key = 'Syn'
            tr = rp.select(component=comp)[0]
        else:
            key = keys[i-1]
            tr = drp[key].select(component=comp)[0]

        ax = plt.subplot(N, 1, i+1)
        plt.plot(tr.times("matplotlib"), tr.data, 'k', lw=lw)

        absmax = np.max(np.abs(tr.data))
        ax.set_ylim(-1.01*absmax, 1.01*absmax)
        plot_label(
            ax, f'{key}', location=1, dist=0.0,
            fontsize='small', box=False)
        plot_label(
            ax, f'max|u|: {absmax:.5g} m', location=2, dist=0.001,
            fontsize='small', box=False)
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.tick_params(labelleft=False, left=False)
        ax.spines.right.set_visible(False)
        ax.spines.left.set_visible(False)
        ax.spines.top.set_visible(False)

        ax.set_xlim(limits)

        if i == N-1:
            plt.xlabel('Time')
        else:
            ax.spines.bottom.set_visible(False)

        if i == 0:
            # Add title with event info
            network = rp[0].stats.network
            station = rp[0].stats.station
            plt.title(
                f"{network}.{station} -- {cmt.cmt_time} Loc: {cmt.latitude}dg, {cmt.longitude}dg, {cmt.depth}km")

    fig.autofmt_xdate()
    plt.subplots_adjust(
        left=0.05, right=0.95, bottom=0.1, top=0.95, hspace=0.0)
    plt.savefig(os.path.join(outdir, 'demo_frechet.pdf'), dpi=300)
    plt.close()
