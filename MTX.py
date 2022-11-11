# %%
# LOAD THE KDTREE FIRST!!!!!!!!!!!!!
# If other packages are loaded, the wrong libstdc++ is picked up and
# doesn't contain the right GLIBCXX version

import numpy as np
import matplotlib.pyplot as plt
from lwsspy.plot import plot_label
from lwsspy.GF.source import CMTSOLUTION
from lwsspy.GF.seismograms import SGTManager
from matplotlib.colors import Normalize, LogNorm
from lwsspy.plot import axes_from_axes, get_aspect

# from lwsspy.GF.seismograms import get_seismograms

# %%

# Write H5py Database file
h5file = '/scratch/gpfs/lsawade/db/II/BFO.h5'

dbfiles = [h5file]

SGTM = SGTManager(dbfiles)

# Get base
cmt = CMTSOLUTION.read('CMTSOLUTION')
lat, lon, dep = cmt.latitude, cmt.longitude, cmt.depth

SGTM.load_header_variables()

# %%

SGTM.header['res_topo']
SGTM.header['topography']
SGTM.header['ellipticity']

# %%

SGTM.get_elements(lat, lon, dep, k=26)


# %%

s = SGTM.get_seismogram(cmt)

# %%

fig = plt.figure(figsize=(10, 6))

for i in range(3):
    ax = plt.subplot(3, 1, i+1)
    plt.plot(s[0, i, :], 'k', lw=1.0)
    plt.ylabel(f'A', rotation=0)
    absmax = np.max(np.abs(s[0, i, :]))
    ax.set_ylim(-1.2*absmax, 1.2*absmax)
    plot_label(
        ax, f'max|u|: {absmax:.5g} m',
        fontsize='x-small', box=False)
    if i == 2:
        ax.tick_params(labelleft=False, left=False)
    else:
        ax.tick_params(labelleft=False, left=False, labelbottom=False)
    ax.spines.right.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.top.set_visible(False)

    ax.set_xlim(0, 400)

    if i == 0:
        # Add title with event info
        network = 'II'
        station = 'BFO'

        plt.title(
            f"{network}.{station} -- {cmt.cmt_time} Loc: {cmt.latitude}dg, {cmt.longitude}dg, {cmt.depth}km")


plt.xlabel('Time in samples')


# Add legend
plt.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))


plt.subplots_adjust(
    left=0.125, right=0.875, bottom=0.2, top=0.875)
plt.savefig('test_seismogram.pdf', dpi=300)

# %%


# Get base
cmt = CMTSOLUTION.read('CMTSOLUTION')


fig = plt.figure(figsize=(10, 6))

alpha = np.linspace(-0.95, 0.95, 21)
depth_range = np.arange(-10, 11, 1)

for i in range(3):
    ax = plt.subplot(3, 1, i+1)

    for _j, _al in enumerate(alpha):

        ncmt = cmt.pert('depth', depth_range[_j])
        ns = SGTM.get_seismogram(ncmt)

        if np.sign(depth_range[_j]) > 0.0:
            lc = (0.8, 0.1, 0.1)
        else:
            lc = (0.1, 0.1, 0.8)

        plt.plot(ns[0, i, :],
                 c=lc, lw=0.25, label=f'{depth_range[_j]:>3d} km', alpha=1-np.abs(_al))

    plt.plot(s[0, i, :], 'k', lw=1.0)
    plt.ylabel(f'A', rotation=0)
    absmax = np.max(np.abs(s[0, i, :]))
    ax.set_ylim(-1.5*absmax, 1.5*absmax)
    plot_label(
        ax, f'max|u|: {absmax:.5g} m',
        fontsize='x-small', box=False)
    if i == 2:
        ax.tick_params(labelleft=False, left=False)
    else:
        ax.tick_params(labelleft=False, left=False, labelbottom=False)
    ax.spines.right.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.top.set_visible(False)

    ax.set_xlim(75, 250)

    if i == 0:
        # Add title with event info
        network = 'II'
        station = 'BFO'

        plt.title(
            f"Depth -- {network}.{station} -- {cmt.cmt_time} Loc: {cmt.latitude}dg, {cmt.longitude}dg, {cmt.depth}km")

    if i == 1:
        # Add legend
        plt.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))

plt.xlabel('Time in samples')


plt.subplots_adjust(
    left=0.125, right=0.875, bottom=0.2, top=0.875)
plt.savefig('testz.pdf', dpi=300)

# %%

# Get base
cmt = CMTSOLUTION.read('CMTSOLUTION')

alpha = np.linspace(-0.95, 0.95, 11)
pert_range = np.linspace(-0.25, 0.25, 11)


# %%
for param in ['Mrr', 'Mtt', 'Mpp', 'Mrt', 'Mrp', 'Mtp']:
    fig = plt.figure(figsize=(10, 6))

    for i in range(3):
        ax = plt.subplot(3, 1, i+1)

        for _j, _al in enumerate(alpha):

            ncmt = cmt.pert(param, cmt.M0 * pert_range[_j])
            print(cmt.M0, getattr(ncmt, 'Mrr'))

            ns = SGTM.get_seismogram(ncmt)

            if np.sign(pert_range[_j]) > 0.0:
                lc = (0.8, 0.1, 0.1)
            else:
                lc = (0.1, 0.1, 0.8)

            plt.plot(ns[0, i, :],
                     c=lc, lw=0.25, label=f'{pert_range[_j]*100:>5.1f} %', alpha=1-np.abs(_al))

        plt.plot(s[0, i, :], 'k', lw=0.5)
        plt.ylabel(f'A', rotation=0)
        absmax = np.max(np.abs(s[0, i, :]))
        ax.set_ylim(-1.2*absmax, 1.2*absmax)
        plot_label(
            ax, f'max|u|: {absmax:.5g} m',
            fontsize='x-small', box=False)
        if i == 2:
            ax.tick_params(labelleft=False, left=False)
        else:
            ax.tick_params(labelleft=False, left=False, labelbottom=False)
        ax.spines.right.set_visible(False)
        ax.spines.left.set_visible(False)
        ax.spines.top.set_visible(False)

        ax.set_xlim(75, 250)

        if i == 0:
            # Add title with event info
            network = 'II'
            station = 'BFO'

            plt.title(
                f"{param} -- {network}.{station} -- {cmt.cmt_time} Loc: {cmt.latitude}dg, {cmt.longitude}dg, {cmt.depth}km")

        if i == 1:
            # Add legend
            legend = plt.legend(frameon=False, loc='center left',
                                bbox_to_anchor=(1, 0.5))

            hp = legend._legend_box.get_children()[1]
            for vp in hp.get_children():
                for row in vp.get_children():
                    row.set_width(70)  # need to adapt this manually
                    row.mode = "expand"
                    row.align = "right"

    plt.xlabel('Time in samples')

    plt.subplots_adjust(
        left=0.125, right=0.875, bottom=0.2, top=0.875)
    plt.savefig(f'test_{param}.pdf', dpi=300)

# %%


# Get base
cmt = CMTSOLUTION.read('CMTSOLUTION')

Nx = 7
Ny = 7
minlat, maxlat = -0.25, 0.25
minlon, maxlon = -0.25, 0.25

fig = plt.figure(figsize=(10, 6))

alpha = np.linspace(-0.95, 0.95, 21)
depth_range = np.linspace(-0.5, 0.5, 51)
lat_range = np.linspace(minlat, maxlat, Ny)
lon_range = np.linspace(minlon, maxlon, Nx)
dlat = np.diff(lat_range)[0]
dlon = np.diff(lon_range)[0]

# llat, llon = np.meshgrid(lat, lon)

x = np.linspace(-1, 1, Nx)
y = np.linspace(-1, 1, Ny)
xx, yy = np.meshgrid(x, y)
rr, tt = np.sqrt(xx**2 + yy**2), np.arctan2(yy, xx)+np.pi

# Get colormap and norm
norm = Normalize(0, 2*np.pi)
rnorm = Normalize(-0.25, 1)
cmap = plt.get_cmap('hsv')

# Fix values to get colors
colors = cmap(norm(tt))
colorfactor = rnorm(rr / rr.max())
pc = colors.copy()
pc[:, :, :3] = pc[:, :, :3] * (colorfactor[:, :, None])
pc[:, :, 3] = pc[:, :, 3] * (1-colorfactor[:, :])
extent = [minlon, maxlon, ]


for i in range(3):
    ax = plt.subplot(3, 1, i+1)
    counter = 0
    for _j, _lat in enumerate(lat_range):
        for _k, _lon in enumerate(lon_range):

            ncmt = cmt.pert('latitude', _lat)
            ncmt = ncmt.pert('longitude', _lon)

            ns = SGTM.get_seismogram(ncmt)

            # if np.sign(depth_range[_j]) > 0.0:
            #     lc = (0.8, 0.1, 0.1)
            # else:
            #     lc = (0.1, 0.1, 0.8)

            plt.plot(ns[0, i, :], c=pc[_k, _j], lw=0.25)

    plt.plot(s[0, i, :], 'k', lw=1.0)
    plt.ylabel(f'A', rotation=0)
    absmax = np.max(np.abs(s[0, i, :]))
    ax.set_ylim(-1.5*absmax, 1.5*absmax)
    plot_label(
        ax, f'max|u|: {absmax:.5g} m',
        fontsize='x-small', box=False)
    if i == 2:
        ax.tick_params(labelleft=False, left=False)
    else:
        ax.tick_params(labelleft=False, left=False, labelbottom=False)
    ax.spines.right.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.top.set_visible(False)

    ax.set_xlim(75, 250)

    if i == 0:
        # Add title with event info
        network = 'II'
        station = 'BFO'

        plt.title(
            f"Depth -- {network}.{station} -- {cmt.cmt_time} Loc: {cmt.latitude}dg, {cmt.longitude}dg, {cmt.depth}km")

    if i == 1:
        print(get_aspect(ax))
        cax = axes_from_axes(
            ax, 12983019, [1.05, 0.0, get_aspect(ax)*1.0, 1.0])
        cax.imshow(pc, extent=[minlon-dlon/2, maxlon +
                   dlon/2, minlat-dlat/2, maxlat+dlat/2], origin='lower')
plt.xlabel('Time in samples')


plt.subplots_adjust(
    left=0.05, right=0.8, bottom=0.2, top=0.875)
plt.savefig('test_geoloc.pdf', dpi=300)

# %%
