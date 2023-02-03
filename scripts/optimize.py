# %% Imports
# External
import time
from glob import glob
from copy import deepcopy
import matplotlib.axes
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import numpy as np
from obspy import read, Stream, Inventory
from obspy.geodetics.base import locations2degrees
import os
import typing as tp
from scipy.optimize import minimize

# Internal
from lwsspy.plot import plot_label
from lwsspy.GF.source import CMTSOLUTION
from lwsspy.GF.seismograms import SGTManager
from lwsspy.GF.process import process_stream, select_pairs
from lwsspy.GF.plot.section import plotsection
from lwsspy.GF.plot.frechet import plotfrechet
from lwsspy.seismo.download_data import download_data

# %% Files

# DB files
specfemmagic = '/scratch/gpfs/lsawade/SpecfemMagicGF'
h5files = os.path.join(specfemmagic, 'DB', '*', '*', '*.h5')

# CMTSOLUTION
cmt = CMTSOLUTION.read(
    '/home/lsawade/lwsspy/lwsspy.GF//scripts/DATA/CHILE_CMT')

# %% Initialize the GF manager
sgt = SGTManager(glob(h5files)[:])
sgt.load_header_variables()

t0 = time.time()
sgt.get_elements(cmt.latitude, cmt.longitude, cmt.depth, 30)
print(72*"=")
print(f"Retrieving elements took {time.time() - t0:.0f} seconds.")

# %% Download data

raw, inv = download_data(
    cmt.origin_time,
    duration=4*3600,
    network='II,IU',
    station=','.join(sgt.stations),
    location='00',
    channel='LH*',
    starttimeoffset=-300,
    endtimeoffset=300
)


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
    [initcmt, obs, sgt, keys, scaling_vector, duration] = data  # get data block

    # Create model vector
    newcmt = deepcopy(cmt)

    x = x_scaled * scaling_vector

    # Reset parameters
    for _key, _x in zip(keys, x):
        setattr(newcmt, _key, _x)

    # Forward model
    syn = process_stream(sgt.get_seismograms(
        newcmt), cmt=initcmt, duration=duration)

    # Sort pairs
    pobs, psyn = select_pairs(obs, syn)

    # Frechet derivatives
    if compute_gradient:
        fsyn = sgt.get_frechet(newcmt)
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
data = [cmt, obs, sgt, keys, scaling_vector, 4.0*3600]

# Constraints
cons = ({'type': 'ineq', 'fun': lambda x:  x[5] + x[6] + x[7]},)

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
res = minimize(optfunc, m_init_scaled, data, method='SLSQP', jac=True, tol=1e-2,
               options={'disp': True, 'maxiter': 10}, callback=recordresult,
               constraints=cons)

print(72*"=")
print(f"Optimization took {time.time() - t0:.0f} seconds.")

# %% Get new synthetics

newcmt = deepcopy(cmt)
for _key, _x, _scale in zip(keys, Wits[-1], scaling_vector):
    setattr(newcmt, _key, _x * _scale)

syn = process_stream(sgt.get_seismograms(cmt), cmt=cmt, duration=4*3600)
newsyn = process_stream(sgt.get_seismograms(newcmt), cmt=cmt, duration=4*3600)


# %% Get new synthetics

starttime = obs[0].stats.starttime + 300
endtime = starttime + 4*3600
limits = (starttime.datetime, endtime.datetime)

# Sort pairs
pobs, psyn = select_pairs(obs, syn)
_, pnewsyn = select_pairs(obs, newsyn)

# %% Plots a section of observed and synthetic
for _comp in ['N', 'E', 'Z']:
    plotsection(pobs, psyn, cmt, newsyn=pnewsyn, newcmt=newcmt,
                comp=_comp, lw=0.25, limits=limits)
    plt.savefig(f'optimization_section_{_comp}.pdf', dpi=300)
    plt.close('all')


# %% Get cost reduction:

costs = []
parameters = dict()

for key in keys:
    parameters[key] = [getattr(cmt, key)]

# Computing cost
costs.append(optfunc(m_init_scaled, data, compute_gradient=False)[0])

for _i, x in enumerate(Wits):

    cost, newcmt = optfunc(x, data, compute_gradient=False)

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

plt.savefig('optimization_summary.pdf', dpi=300)

plt.close('all')
