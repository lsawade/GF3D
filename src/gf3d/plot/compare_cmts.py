import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from .util import plot_label
from ..source import CMTSOLUTION

def compare_cmts(
    ax, cmt0: CMTSOLUTION, cmt1: CMTSOLUTION | None = None, factor: float = 1.0,
    dpi=300, pdfmode=False):
        plot_label(
            ax,
            f'{cmt0.eventname}\n'
            f'Lat: {cmt0.latitude:.0f} Lon: {cmt0.longitude:.0f} '
            f'Z: {cmt0.depth:.2f}km\n'
            f'Mw: {cmt0.Mw:.1f}',
            location=1, box=False, fontsize='xx-small', family='monospace')

        # Adjust beach width
        widthperdpi = 0.5

        if pdfmode:
            beachwidth = widthperdpi * 72 * factor
        else:
            beachwidth = widthperdpi * dpi * factor

        linewidth = 0.75
        # plot beaches in the axes anyways
        cmt0.axbeach(
            ax, 0.35, 0.5, beachwidth, facecolor=(0.8, 0.2, 0.2),
            clip_on=False, linewidth=linewidth)
        if cmt1 is not None:
            cmt1.axbeach(
                ax, 0.75, 0.5, beachwidth, facecolor=(0.2, 0.2, 0.8),
                clip_on=False, linewidth=linewidth)

        plt.plot([0, 0], [0, 0.95], 'k', lw=0.75,
                 transform=ax.transAxes, clip_on=False)
        plt.plot([0, 0.9], [0, 0], 'k', lw=0.75,
                 transform=ax.transAxes, clip_on=False)

        # Get changes
        if cmt1 is not None:
            ddep = (cmt1.depth-cmt0.depth)
            dlat = cmt1.latitude-cmt0.latitude
            dlon = cmt1.longitude-cmt0.longitude
            dM0 = (cmt1.M0-cmt0.M0)/cmt0.M0
            n1 = np.sqrt(np.sum(cmt1.fulltensor**2))
            n0 = np.sqrt(np.sum(cmt0.fulltensor**2))
            angle = np.arccos(
                np.sum(cmt1.fulltensor*cmt0.fulltensor)/(n1*n0))/np.pi*180

            # Plot Label of changes
            plot_label(
                ax,
                f'{"<"}: {angle:5.1f}\n'
                f'dlat: {dlat:5.2f}  dz:    {ddep:5.2f}km\n'
                f'dlon: {dlon:5.2f}  dlnM0: {dM0:5.2f}',
                fontsize='x-small', box=False,
                family='monospace', location=3)
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)