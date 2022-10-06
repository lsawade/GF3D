# %%
# LOAD THE KDTREE FIRST!!!!!!!!!!!!!
# If other packages are loaded, the wrong libstdc++ is picked up and
# doesn't contain the right GLIBCXX version

from scipy.spatial import KDTree
import h5py
import plotly.graph_objs as go
from lwsspy.GF.get_topo_bathy import get_topo_bathy
from pprint import pprint
import os
import adios2
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from lwsspy.GF.source2xyz import source2xyz
from lwsspy.GF.source import CMTSOLUTION
from lwsspy.GF.constants_solver import NGLLX, NGLLY, NGLLZ, NGLL3, MIDX, MIDY, MIDZ
from lwsspy.GF.locate_point import locate_point
from lwsspy.GF.transformations.rthetaphi_xyz import xyz_2_rthetaphi
from lwsspy.GF.lagrange import lagrange_any, gll_nodes
from lwsspy.GF.constants_solver import NGLLX, NGLLY, NGLLZ, NGLL3, MIDX, MIDY, MIDZ
# Only import the KDTree after setting the LD_LIBRARY PATH, e.g.
# $ export LD_LIBRARY_PATH='/home/lsawade/.conda/envs/gf/lib'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# %% Get CMT solution to convert
cmt = CMTSOLUTION.read('CMTSOLUTION')


topography = True
ellipticity = True


# Get file name
specfemmagic = '/scratch/gpfs/lsawade/SpecfemMagicGF'
reciprocal_file = os.path.join(
    specfemmagic, 'specfem3d_globe', 'run0001', "OUTPUT_FILES",
    "save_forward_arrays_GF.bp")

F0 = 1e14

# %%
with adios2.open(reciprocal_file, "r", comm) as rh:
    # Need to load
    pprint(rh.available_variables())
    ELLIPTICITY = rh.read('ELLIPTICITY')[0]
    TOPOGRAPHY = rh.read('TOPOGRAPHY')[0]
    NX_BATHY = rh.read('NX_BATHY')[0]
    NY_BATHY = rh.read('NY_BATHY')[0]
    RESOLUTION_TOPO_FILE = rh.read('RESOLUTION_TOPO_FILE')[0]

    ibathy_topo = rh.read('ibathy_topo/array')[:NX_BATHY*NY_BATHY]

    # In
print(NX_BATHY, NY_BATHY, RESOLUTION_TOPO_FILE)

# Resolution is given in minutes hence divide by 60 to get degrees
dx = RESOLUTION_TOPO_FILE/60
x = np.arange(0, 360, dx)
y = np.arange(-90, 90, dx)
ibathy_topo = ibathy_topo.reshape(NX_BATHY, NY_BATHY, order='F')

# %%
# Plot topo
plot_topo = True
if plot_topo:
    fig = plt.figure()
    extent = [0, 360, -90, 90]
    im = plt.imshow(ibathy_topo.T, extent=extent, origin='upper')
    plt.colorbar(im)
    plt.savefig('test_topo.png', dpi=200)
    plt.close(fig)

# %%

with adios2.open(reciprocal_file, "r", comm) as rh:

    Nrspl = rh.read('rspl/local_dim')[0]
    Nellipticity_spline = rh.read('ellipicity_spline/local_dim')[0]
    Nellipticity_spline2 = rh.read('ellipicity_spline2/local_dim')[0]

    rspl = rh.read('rspl/array')[:Nrspl]
    ellipticity_spline = rh.read(
        'ellipicity_spline/array')[:Nellipticity_spline]
    ellipticity_spline2 = rh.read(
        'ellipicity_spline2/array')[:Nellipticity_spline2]

# %%
plot_spline = True

if plot_spline:
    fig = plt.figure()
    plt.plot(rspl, ellipticity_spline)
    plt.plot(rspl, ellipticity_spline2)
    plt.savefig('test_spline.png', dpi=200)
    plt.close(fig)


# %%
print(Nrspl)
print(Nellipticity_spline)
print(Nellipticity_spline2)

# %%


get_topo_bathy(cmt.latitude, cmt.longitude, ibathy_topo,
               NX_BATHY, NY_BATHY, RESOLUTION_TOPO_FILE)

# get_topo_bathy(46.852886, -121.760374, ibathy_topo,
#                NX_BATHY, NY_BATHY, RESOLUTION_TOPO_FILE)

# %%
with h5py.File('testdb.h5', 'r') as db:
    topography = db['TOPOGRAPHY']
    ellipticity = db['ELLIPTICITY']
    ibathy_topo = db['BATHY']
    NX_BATHY = db['NX_BATHY']
    NY_BATHY = db['NY_BATHY']
    RESOLUTION_TOPO_FILE = db['RESOLUTION_TOPO_FILE']
    rspl = db['rspl']
    ellipticity_spline = ['ellipticity_spline']
    ellipticity_spline2 = ['ellipticity_spline2']

# %%
x_target, y_target, z_target, Mx = source2xyz(
    cmt.latitude,
    cmt.longitude,
    cmt.depth,
    cmt.tensor,
    topography=topography,
    ellipticity=ellipticity,
    ibathy_topo=ibathy_topo,
    NX_BATHY=NX_BATHY,
    NY_BATHY=NY_BATHY,
    RESOLUTION_TOPO_FILE=RESOLUTION_TOPO_FILE,
    rspl=rspl,
    ellipicity_spline=ellipticity_spline,
    ellipicity_spline2=ellipticity_spline2,
)

# %%
"""
using moment tensor source:
xi coordinate of source in that element:  -0.99738383475014891
eta coordinate of source in that element:   0.81333723147864112
gamma coordinate of source in that element:  -0.95321514891448189
"""

with adios2.open(reciprocal_file, "r", comm) as rh:

    pprint(rh.available_variables())

    # First read the number of NGLOB in each slice
    NGLOB = rh.read('NGLOB')
    NGF_UNQIUE = rh.read('NGF_UNIQUE')
    NGF_UNQIUE_LOCAL = rh.read('NGF_UNIQUE_LOCAL')
    NPROC = rh.read('NPROC')[0]

    print(NGLOB)
    print(NGF_UNQIUE)
    print(NGF_UNQIUE_LOCAL)
    print(NPROC)

    nsteps = rh.steps()
    xyz = []
    ibool = []
    epsilon = []

    for i in range(NPROC):

        if NGLOB[i] > 0:

            # Getting coordinates
            xyz_sub = dict()
            for _l in ['x', 'y', 'z']:

                global_dim = rh.read(f'{_l}/global_dim')[i]
                local_dim = rh.read(f'{_l}/local_dim')[i]
                offset = rh.read(f'{_l}/offset')[i]

                xyz_sub[_l] = rh.read(
                    f'{_l}/array', start=[offset], count=[NGLOB[i], ], block_id=0)
            xyz.append(xyz_sub)

            # Getting ibools
            local_dim = rh.read('ibool_GF/local_dim')[i]
            offset = rh.read('ibool_GF/offset')[i]
            ibool_sub = rh.read(
                'ibool_GF/array', start=[offset], count=[NGLL3*NGF_UNQIUE_LOCAL[i], ],
                block_id=0)
            ibool.append(
                ibool_sub.reshape(NGLLX, NGLLY, NGLLZ, NGF_UNQIUE_LOCAL[i], order='F')-1)

            # Getting the epsilon
            epsilon_sub = dict()
            for _l in ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']:
                key = f'epsilon_{_l}'

                local_dim = rh.read(f'{key}/local_dim')[i]
                offset = rh.read(f'{key}/offset')[i]

                epsilon_sub[key] = rh.read(
                    f'{key}/array', start=[offset],
                    count=[NGLL3*NGF_UNQIUE_LOCAL[i], ],
                    step_start=0, step_count=nsteps, block_id=0).reshape(
                        NGLLX, NGLLY, NGLLZ, NGF_UNQIUE_LOCAL[i], nsteps)

            epsilon.append(epsilon_sub)

        else:
            xyz.append([])
            ibool.append([])
            epsilon.append([])


# %% Locating a point using a KDTree
slc = 4
# USing the xyz points that we saved earlier
midpoints = np.zeros((3, NGF_UNQIUE_LOCAL[slc]))

# All iglobs
iglobs = ibool[slc][MIDX, MIDY, MIDZ, :].astype(int)
midpoints = np.vstack(
    (xyz[slc]['x'][iglobs], xyz[slc]['y'][iglobs], xyz[slc]['z'][iglobs])).T
kdtree = KDTree(midpoints)

# %%

# Locate the point
ispec_selected, xi, eta, gamma, x, y, z, distmin_not_squared = locate_point(
    x_target, y_target, z_target, cmt.latitude, cmt.longitude,
    midpoints, xyz[slc]['x'], xyz[slc]['y'], xyz[slc]['z'], ibool[slc],
    POINT_CAN_BE_BURIED=True, kdtree=kdtree)


# %%

with h5py.File('testdb.h5', 'r') as db:
    NSPEC = db['NSPEC'][()]
    topography = db['TOPOGRAPHY'][()]
    ellipticity = db['ELLIPTICITY'][()]
    ibathy_topo = db['BATHY'][:]
    NX_BATHY = db['NX_BATHY'][()]
    NY_BATHY = db['NY_BATHY'][()]
    RESOLUTION_TOPO_FILE = db['RESOLUTION_TOPO_FILE'][()]
    rspl = db['rspl'][:]
    ellipticity_spline = db['ellipticity_spline'][:]
    ellipticity_spline2 = db['ellipticity_spline2'][:]
    ibool = db['ibool'][:]-1
    xyz = db['xyz'][:]


# %%

ellipticity
# %%

cmt = CMTSOLUTION.read('CMTSOLUTION')


x_target, y_target, z_target, Mx = source2xyz(
    cmt.latitude,
    cmt.longitude,
    cmt.depth,
    cmt.tensor,
    topography=topography,
    ellipticity=ellipticity,
    ibathy_topo=ibathy_topo,
    NX_BATHY=NX_BATHY,
    NY_BATHY=NY_BATHY,
    RESOLUTION_TOPO_FILE=RESOLUTION_TOPO_FILE,
    rspl=rspl,
    ellipicity_spline=ellipticity_spline,
    ellipicity_spline2=ellipticity_spline2,
)

# %%
topography


# %%
kdtree = KDTree(xyz[ibool[2, 2, 2, :], :])

# %%
# Locate the point
ispec_selected, xi, eta, gamma, x, y, z, distmin_not_squared = locate_point(
    x_target, y_target, z_target, cmt.latitude, cmt.longitude,
    xyz[ibool[2, 2, 2, :], :], xyz[:, 0], xyz[:, 1], xyz[:, 2], ibool,
    POINT_CAN_BE_BURIED=True, kdtree=kdtree)

# %%
# GLL points and weights (degree)
npol = 4
xigll, wxi, _ = gll_nodes(npol)
etagll, weta, _ = gll_nodes(npol)
gammagll, wgamma, _ = gll_nodes(npol)


# Get lagrange values at specific GLL poins
shxi, shpxi = lagrange_any(xi, xigll, npol)
sheta, shpeta = lagrange_any(eta, xigll, npol)
shgamma, shpgamma = lagrange_any(gamma, xigll, npol)


# %%
with h5py.File('testdb.h5', 'r') as db:
    print(db['epsilon'].shape)
    epsilon = db['epsilon'][:, :, :, :, ispec_selected, :] / (1e14*1e7)

# %%
sepsilon = np.zeros((6, epsilon.shape[-1]))

for k in range(NGLLZ):
    for j in range(NGLLY):
        for i in range(NGLLX):
            hlagrange = shxi[i] * sheta[j] * shgamma[k]
            sepsilon += hlagrange * epsilon[:, i, j, k, :]


# %%
# gravitational constant in m3 kg-1 s-2, or equivalently in N.(m/kg)^2
GRAV = 6.67384e-11
R_PLANET = 6371000.0    # radius of the Earth in m
RHOAV = 5514.3          # Avergage density of the Earth
scaleM = 1e7 * RHOAV * (R_PLANET ** 5) * np.pi * GRAV * RHOAV

M = Mx
sgt = np.array([1., 1., 1., 2., 2., 2.])[:, None] * sepsilon
z = np.sum(Mx[:, None] * sgt, axis=0)

# %%

plt.figure()
plt.plot(z)
plt.show()
# %%
NSPEC = ibool.shape[-1]

# %%
x = []
y = []
z = []

for i in range(NSPEC):

    x.extend([
        xyz[ibool[0, 0, 0, i], 0], xyz[ibool[4, 0, 0, i], 0], None,
        xyz[ibool[0, 0, 0, i], 0], xyz[ibool[0, 4, 0, i], 0], None,
        xyz[ibool[0, 0, 0, i], 0], xyz[ibool[0, 0, 4, i], 0], None,
        xyz[ibool[4, 0, 0, i], 0], xyz[ibool[4, 0, 4, i], 0], None,
        xyz[ibool[4, 0, 0, i], 0], xyz[ibool[4, 4, 0, i], 0], None,
        xyz[ibool[0, 4, 0, i], 0], xyz[ibool[4, 4, 0, i], 0], None,
        xyz[ibool[0, 4, 0, i], 0], xyz[ibool[0, 4, 4, i], 0], None,
        xyz[ibool[0, 0, 4, i], 0], xyz[ibool[0, 4, 4, i], 0], None,
        xyz[ibool[0, 0, 4, i], 0], xyz[ibool[4, 0, 4, i], 0], None,
        xyz[ibool[4, 4, 4, i], 0], xyz[ibool[4, 4, 0, i], 0], None,
        xyz[ibool[4, 4, 4, i], 0], xyz[ibool[0, 4, 4, i], 0], None,
        xyz[ibool[4, 4, 4, i], 0], xyz[ibool[4, 0, 4, i], 0], None,
    ])

    y.extend([
        xyz[ibool[0, 0, 0, i], 1], xyz[ibool[4, 0, 0, i], 1], None,
        xyz[ibool[0, 0, 0, i], 1], xyz[ibool[0, 4, 0, i], 1], None,
        xyz[ibool[0, 0, 0, i], 1], xyz[ibool[0, 0, 4, i], 1], None,
        xyz[ibool[4, 0, 0, i], 1], xyz[ibool[4, 0, 4, i], 1], None,
        xyz[ibool[4, 0, 0, i], 1], xyz[ibool[4, 4, 0, i], 1], None,
        xyz[ibool[0, 4, 0, i], 1], xyz[ibool[4, 4, 0, i], 1], None,
        xyz[ibool[0, 4, 0, i], 1], xyz[ibool[0, 4, 4, i], 1], None,
        xyz[ibool[0, 0, 4, i], 1], xyz[ibool[0, 4, 4, i], 1], None,
        xyz[ibool[0, 0, 4, i], 1], xyz[ibool[4, 0, 4, i], 1], None,
        xyz[ibool[4, 4, 4, i], 1], xyz[ibool[4, 4, 0, i], 1], None,
        xyz[ibool[4, 4, 4, i], 1], xyz[ibool[0, 4, 4, i], 1], None,
        xyz[ibool[4, 4, 4, i], 1], xyz[ibool[4, 0, 4, i], 1], None,
    ])

    z.extend([
        xyz[ibool[0, 0, 0, i], 2], xyz[ibool[4, 0, 0, i], 2], None,
        xyz[ibool[0, 0, 0, i], 2], xyz[ibool[0, 4, 0, i], 2], None,
        xyz[ibool[0, 0, 0, i], 2], xyz[ibool[0, 0, 4, i], 2], None,
        xyz[ibool[4, 0, 0, i], 2], xyz[ibool[4, 0, 4, i], 2], None,
        xyz[ibool[4, 0, 0, i], 2], xyz[ibool[4, 4, 0, i], 2], None,
        xyz[ibool[0, 4, 0, i], 2], xyz[ibool[4, 4, 0, i], 2], None,
        xyz[ibool[0, 4, 0, i], 2], xyz[ibool[0, 4, 4, i], 2], None,
        xyz[ibool[0, 0, 4, i], 2], xyz[ibool[0, 4, 4, i], 2], None,
        xyz[ibool[0, 0, 4, i], 2], xyz[ibool[4, 0, 4, i], 2], None,
        xyz[ibool[4, 4, 4, i], 2], xyz[ibool[4, 4, 0, i], 2], None,
        xyz[ibool[4, 4, 4, i], 2], xyz[ibool[0, 4, 4, i], 2], None,
        xyz[ibool[4, 4, 4, i], 2], xyz[ibool[4, 0, 4, i], 2], None,
    ])

# %%
fig = go.Figure(data=go.Scatter3d(x=x, y=y, z=z, mode='lines'))
fig.show()
