# %% Imports
# External
import time
from glob import glob
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
from obspy import Stream
import os
import typing as tp
from scipy.optimize import minimize
import cartopy
from cartopy import crs
# Internal
from gf3d.plot.util import plot_label
from gf3d.source import CMTSOLUTION
from gf3d.seismograms import GFManager
from gf3d.process import process_stream, select_pairs
from gf3d.plot.section import plotsection
from gf3d.plot.section_aligned import plotsection_aligned, get_azimuth_distance_traveltime, filter_stations
from gf3d.plot.compare_cmts import compare_cmts
# from gf3d.plot.frechet import plotfrechet
from gf3d.download import download_stream


# %% Files

# DB files
specfemmagic = '/scratch/gpfs/lsawade/SpecfemMagicGF'
h5files = os.path.join(specfemmagic, 'DB', '*', '*', '*.h5')

# CMTSOLUTION
cmt = CMTSOLUTION.read(
    '/home/lsawade/lwsspy/gf3d//scripts/DATA/CHILE_CMT')

# %% Initialize the GF manager
gfm = GFManager(glob(h5files)[:])
gfm.load_header_variables()

t0 = time.time()
gfm.get_elements(cmt.latitude, cmt.longitude, cmt.depth, 30)
print(72*"=")
print(f"Retrieving elements took {time.time() - t0:.0f} seconds.")

# %% Download data

raw, inv = download_stream(
    cmt.origin_time,
    duration=4*3600,
    network='II,IU',
    station=','.join(gfm.stations),
    location='00',
    channel='LH*',
    starttimeoffset=-300,
    endtimeoffset=300
)

# %%
stations = set()
for network in inv:
    for station in network:
        for channel in station:
            stations.add((network.code, station.code,
                         channel.latitude, channel.longitude))

slat = [station[2] for station in stations]
slon = [station[3] for station in stations]


# %% Process Observed

obs = process_stream(raw, inv, cmt=cmt, duration=4*3600)

# %%

scale = dict(
    Mrr=dict(scale=cmt.M0),
    Mtt=dict(scale=cmt.M0),
    Mpp=dict(scale=cmt.M0),
    Mrt=dict(scale=cmt.M0),
    Mrp=dict(scale=cmt.M0),
    Mtp=dict(scale=cmt.M0),
    time_shift=dict(scale=1.0),
    depth=dict(scale=5.0),
    latitude=dict(scale=0.1),
    longitude=dict(scale=0.1),
    hdur=dict(scale=1.0)
)

keys = ['time_shift', 'hdur', 'latitude', 'longitude', 'depth',
        'Mrr', 'Mtt', 'Mpp', 'Mrt', 'Mrp', 'Mtp']

scaling_vector = [scale[_key]['scale'] for _key in keys]
m_init = np.array([getattr(cmt, _key) for _key in keys])

m_init_scaled = m_init/scaling_vector


def optfunc(x_scaled, data, compute_gradient=True):

    # Get input data
    [initcmt, obs, gfm, keys, scaling_vector, duration] = data  # get data block

    # Create model vector
    newcmt = deepcopy(cmt)

    x = x_scaled * scaling_vector

    # Reset parameters
    for _key, _x in zip(keys, x):
        setattr(newcmt, _key, _x)

    # Forward model
    syn = process_stream(gfm.get_seismograms(
        newcmt), cmt=initcmt, duration=duration)

    # Sort pairs
    pobs, psyn = select_pairs(obs, syn)

    # Frechet derivatives
    if compute_gradient:
        fsyn = gfm.get_frechet(newcmt)
        pfsyn = dict()
        for _parameter, _fstream in fsyn.items():

            fsyn[_parameter] = process_stream(
                _fstream, cmt=initcmt, duration=duration)

            _, pfsyn[_parameter] = select_pairs(obs, fsyn[_parameter])

    # Compute cost and gradient
    cost = 0.0

    if compute_gradient:
        gradient = np.zeros_like(x)

    for _i, (_syn, _obs) in enumerate(zip(psyn, pobs)):
        norm = np.sum(_obs.data**2)
        cost += np.sum((_syn.data - _obs.data)**2)/norm

        if compute_gradient:
            for _j, _key in enumerate(keys):
                gradient[_j] += 2 * \
                    np.sum((_syn.data - _obs.data) * pfsyn[_key][_i].data)/norm

    cost /= len(psyn)

    if compute_gradient:
        gradient /= len(psyn)
        gradient *= scaling_vector

    print(70*"-")
    print('Cost:', cost)
    if compute_gradient:
        print('Grad:', np.array2string(gradient, max_line_width=10000000))

    if compute_gradient:
        return cost, gradient
    else:
        return cost, newcmt


# %% Setting up the inversion

# Data
data = [cmt, obs, gfm, keys, scaling_vector, 4.0*3600]

# Constraints
# cons = ({'type': 'ineq', 'fun': lambda x:  x[5] + x[6] + x[7]},)

Wits = []
iterations = [0]


def recordresult(x):
    #global Wits
    Wits.append(x)
    print('Modl:', np.array2string(x, max_line_width=10000000))
    print(f'End of summary for iteration {iterations[0]:03d}')
    print(70*"=")
    iterations[0] += 1
    return


t0 = time.time()
res = minimize(optfunc, m_init_scaled, data, method='BFGS', jac=True, tol=1e-4,
               options={'disp': True, 'maxiter': 150}, callback=recordresult)  # ,
#    constraints=cons)

print(72*"=")
print(f"Optimization took {time.time() - t0:.0f} seconds.")

# %% Get new synthetics
syn = process_stream(gfm.get_seismograms(cmt), cmt=cmt, duration=4*3600)


newcmt = []
newsyn = []
costs = []

# Computing cost iteration 0
cost, _newcmt = optfunc(m_init_scaled, data, compute_gradient=False)
costs.append(cost)
newsyn.append(syn)
newcmt.append(_newcmt)

for x in Wits:

    # Updating the CMT solution
    ncmt = deepcopy(cmt)
    for _key, _x, _scale in zip(keys, x, scaling_vector):
        setattr(ncmt, _key, _x * _scale)

    # Appending the CMT solution to list
    newcmt.append(ncmt)
    newsyn.append(process_stream(
        gfm.get_seismograms(ncmt), cmt=cmt, duration=4*3600))

    # Compute coset
    cost, _ = optfunc(x, data, compute_gradient=False)
    costs.append(cost)


# %% Get new synthetics

starttime = obs[0].stats.starttime + 300
endtime = starttime + 4*3600
limits = (starttime.datetime, endtime.datetime)

# Sort pairs
pobs, psyn = select_pairs(obs, syn)

pnewsyn = [select_pairs(obs, _newsyn)[1] for _newsyn in newsyn]

# %% Plots a section of observed and synthetic
plt.close('all')
for _comp in ['N', 'E', 'Z']:
    plotsection(pobs, psyn, cmt, newsyn=pnewsyn[-1], newcmt=newcmt[-1],
                comp=_comp, lw=0.25, limits=limits)
    plt.savefig(f'optimization_section_{_comp}.pdf', dpi=300)
    plt.close('all')

# %%
component = 'Z'
# Plot body wave section
# phase1 = 'P'
# phase2 = 'S'
# window1 = (-100, 250)
# window2 = (-150, 250)
phase1 = 'Rayleigh'
phase2 = 'Rayleigh'
window1 = (-300, 400)
window2 = (-400, 600)

pobs, psyn = select_pairs(obs, syn)

niter = len(newcmt)

depth = [_cmt.depth for _cmt in newcmt]
latitude = [_cmt.latitude for _cmt in newcmt]
longitude = [_cmt.longitude for _cmt in newcmt]
minlat, maxlat = np.min(latitude), np.max(latitude)
minlon, maxlon = np.min(longitude), np.max(longitude)
mindep, maxdep = np.min(depth), np.max(depth)
mincost, maxcost = np.min(costs), np.max(costs)
dlat = np.abs(maxlat - minlat)
dlon = np.abs(maxlon - minlon)
ddep = np.abs(maxdep - mindep)
dcost = np.abs(1 - mincost/maxcost)
lonlim = (longitude[niter-1] - dlon, longitude[niter-1] + dlon)
latlim = (latitude[niter-1] - dlat, latitude[niter-1] + dlat)
deplim = (depth[niter-1] - ddep, depth[niter-1] + ddep)

for iter in range(niter):

    print("iteration:", iter)

    # For pwaves
    obs1, syn1, newsyn1 = get_azimuth_distance_traveltime(
        cmt, pobs, psyn, newsyn=pnewsyn[iter], comp=component,
        traveltime_window=(phase1, window1))

    obs2, syn2, newsyn2 = get_azimuth_distance_traveltime(
        cmt, pobs, psyn, newsyn=pnewsyn[iter], comp=component,
        traveltime_window=(phase2, window2), orbit=2)

    selection = filter_stations(obs1, obs2)

    # Subselection of overlapping stations with arrivals
    obs1 = Stream([obs1[_i] for _i in selection])
    syn1 = Stream([syn1[_i] for _i in selection])
    newsyn1 = Stream([newsyn1[_i] for _i in selection])
    obs2 = Stream([obs2[_i] for _i in selection])
    syn2 = Stream([syn2[_i] for _i in selection])
    newsyn2 = Stream([newsyn2[_i] for _i in selection])

    # Plots a section of observed and synthetic
    plt.close('all')
    fig = plt.figure(figsize=(10.5, 6))
    maings = GridSpec(1, 2, wspace=0.175, width_ratios=[1.3, 2.0])
    seisgs = GridSpecFromSubplotSpec(1, 2, wspace=0.05, subplot_spec=maings[1])
    panelgs = GridSpecFromSubplotSpec(
        2, 2, subplot_spec=maings[0], hspace=0.2, height_ratios=[1, 2.25])
    costbeachgs = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=panelgs[0, :], wspace=0.1, width_ratios=[1.0, 2])
    lowerpanelgs = GridSpecFromSubplotSpec(
        2, 2, subplot_spec=panelgs[1, :], wspace=0.3, height_ratios=[1.25, 1.0])

    # Create figure
    plt.subplots_adjust(left=0.075, right=0.925, bottom=0.1, top=0.95)

    axbeach = fig.add_subplot(costbeachgs[1])

    if iter == 0:
        compare_cmts(axbeach, cmt, None, factor=1.3, pdfmode=False)
    else:
        compare_cmts(axbeach, cmt, newcmt[iter], factor=1.3, pdfmode=False)

    axmap = fig.add_subplot(lowerpanelgs[0, :], projection=crs.PlateCarree(
        central_longitude=newcmt[-1].longitude))
    axmap.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='k',
                      linewidth=0.5, facecolor=(0.9, 0.9, 0.9))
    axmap.scatter(slon, slat, s=35, c='r', marker='v',
                  edgecolor='k', linewidth=0.5, transform=crs.PlateCarree())
    axmap.scatter(newcmt[-1].longitude, newcmt[-1].latitude, s=100, c='b',
                  marker='*', edgecolor='k', linewidth=0.5, transform=crs.PlateCarree())

    markeriterdict = dict(markerfacecolor='w', markersize=3)
    markerfinaldict = dict(markerfacecolor=(0.9, 0.2, 0.2), markersize=5)

    axcost = fig.add_subplot(costbeachgs[0])
    axcost.ticklabel_format(useOffset=False)
    axcost.spines.top.set_visible(False)
    axcost.spines.right.set_visible(False)
    axcost.tick_params(right=False)
    plt.plot(np.arange(iter+1), costs[:iter+1] /
             np.max(costs), 'k-o', **markeriterdict)
    plt.xlim(0, niter)
    plt.ylim(1-dcost*1.05, 1)
    plt.xlabel('Iteration #')
    plot_label(axcost, 'Norm. Cost', location=6, box=False)
    axcost.tick_params(axis='both', which='major')
    axcost.set_yscale('log')
    # ax.tick_params(axis='both', which='minor', labelsize=8)
    if iter == niter-1:
        plt.plot(iter, costs[iter]/np.max(costs), 'kx', **markerfinaldict)

    axdepth = fig.add_subplot(lowerpanelgs[1, 0])
    axdepth.ticklabel_format(useOffset=False)
    axdepth.spines.top.set_visible(False)
    axdepth.spines.right.set_visible(False)
    axdepth.tick_params(right=False)
    plt.plot(np.arange(iter+1),
             [_cmt.depth for _cmt in newcmt[:iter+1]], 'k-o', **markeriterdict)
    plt.xlim(0, niter)
    plt.ylim(deplim[::-1])
    plt.xlabel("Iteration #")
    plot_label(axdepth, 'Depth [km]', location=6, box=False)
    if iter == niter-1:
        plt.plot(iter, newcmt[iter].depth, 'kx', **markerfinaldict)

    axloc = fig.add_subplot(lowerpanelgs[1, 1])
    axloc.ticklabel_format(useOffset=False)
    axloc.spines.top.set_visible(False)
    axloc.spines.right.set_visible(False)
    axloc.tick_params(right=False)
    plot_label(axloc, 'Lat', location=6, box=False)
    plt.xlabel('Lon')
    plt.plot(longitude[:iter+1], latitude[:iter+1], 'k-o', **markeriterdict)
    plt.xlim(lonlim)
    plt.ylim(latlim)
    if iter == niter-1:
        plt.plot(longitude[iter], latitude[iter], 'kx', **markerfinaldict)

    if iter == 0:

        ax0 = fig.add_subplot(seisgs[0])
        plotsection_aligned(
            obs1, syn1, cmt, newsyn=None, comp=component, lw=0.75, ax=ax0,
            traveltime_window=(phase1, window1), labelright=False)

        ax1 = fig.add_subplot(seisgs[1])
        plotsection_aligned(
            obs2, syn2, cmt, newsyn=None, comp=component, lw=0.75, ax=ax1,
            traveltime_window=(phase2, window2), labelleft=False)

    else:

        ax0 = fig.add_subplot(seisgs[0])
        plotsection_aligned(
            obs1, syn1, cmt, newsyn=newsyn1, comp=component, lw=0.75, ax=ax0,
            traveltime_window=(phase1, window1), labelright=False)

        ax1 = fig.add_subplot(seisgs[1])
        plotsection_aligned(
            obs2, syn2, cmt, newsyn=newsyn2, comp=component, lw=0.75, ax=ax1,
            traveltime_window=(phase2, window2), labelleft=False)

    # ax0.set_title(title, fontsize='small', loc='left', ha='left')
    plt.savefig(
        f'optimization_section_{phase1}_{phase2}_{iter:>02d}.png', dpi=300)


# %% Get cost reduction:

costs = []
parameters = dict()

for key in keys:
    parameters[key] = [getattr(cmt, key)]

# Computing cost
costs.append(optfunc(m_init_scaled, data, compute_gradient=False)[0])

for _i, x in enumerate(Wits):

    cost, _newcmt = optfunc(x, data, compute_gradient=False)

    costs.append(cost)

    for key in keys:
        parameters[key].append(getattr(newcmt, key))


# %% Plot

iterations = np.arange(len(costs))

xlim = (0, len(costs))
kwargs = dict(lw=1.0, marker='o', color='k', clip_on=False)
labelkwargs = dict(location=6, dist=0.0, box=False, fontsize='medium')

nrows = 2
ncols = 6
plt.figure(figsize=(14, 6))
ax0 = plt.subplot(nrows, ncols, 1)
plt.plot(iterations, np.array(costs)/np.max(np.array(costs)), **kwargs)
plt.xlim(xlim)
ax0.ticklabel_format(useOffset=False)
plot_label(ax0, 'Normalized Cost', **labelkwargs)

ax = plt.subplot(nrows, ncols, 2, sharex=ax0)
plt.plot(iterations, parameters['depth'], **kwargs)
plt.xlim(xlim)
ax.ticklabel_format(useOffset=False)
plot_label(ax, 'Depth [km]', **labelkwargs)

ax = plt.subplot(nrows, ncols, 3, sharex=ax0)
plt.plot(iterations, parameters['latitude'], **kwargs)
plt.xlim(xlim)
ax.ticklabel_format(useOffset=False)
plot_label(ax, 'Lat [deg]', **labelkwargs)

ax = plt.subplot(nrows, ncols, 4, sharex=ax0)
plt.plot(iterations, parameters['longitude'], **kwargs)
plt.xlim(xlim)
ax.ticklabel_format(useOffset=False)
plot_label(ax, 'Lon [deg]', **labelkwargs)

ax = plt.subplot(nrows, ncols, 5, sharex=ax0)
plt.plot(iterations, parameters['time_shift'], **kwargs)
plt.xlim(xlim)
ax.ticklabel_format(useOffset=False)
plot_label(ax, 'Time Shift [s]', **labelkwargs)

ax = plt.subplot(nrows, ncols, 6, sharex=ax0)
plt.plot(iterations, parameters['hdur'], **kwargs)
plt.xlim(xlim)
ax.ticklabel_format(useOffset=False)
plot_label(ax, 'Half Duration [s]', **labelkwargs)

ipl = 7
for _mt in ['Mrr', 'Mtt', 'Mpp', 'Mrt', 'Mrp', 'Mtp']:
    ax = plt.subplot(nrows, ncols, ipl, sharex=ax0)
    plt.plot(iterations, np.array(parameters[_mt])/cmt.M0, **kwargs)
    plt.xlim(xlim)
    plt.xlabel('Iteration #')
    ax.ticklabel_format(useOffset=False)
    plot_label(ax, f'{_mt}/M0', **labelkwargs)
    ipl += 1

plt.suptitle(
    f"Eventid: {cmt.eventname} - Mw: {cmt.Mw:.2f} - {cmt.cmt_time.ctime()}\nLoc: {cmt.latitude:.4f}dg, {cmt.longitude:.4f}dg, {cmt.depth:.4f}km, ",
    ha='left', x=0.05, y=.975)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.875, wspace=0.6)

plt.savefig('optimization_summary.png', dpi=300)

plt.close('all')
