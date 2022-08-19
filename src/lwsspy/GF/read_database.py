
import os
import sys
from mpi4py import MPI
import numpy as np
import adios2
import matplotlib.pyplot as plt
from .lagrange import lagrange_any, gll_nodes
from obspy import read
from time import sleep
from copy import deepcopy
from lwsspy.plot import plot_label
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# From constants.h.in
# gravitational constant in m3 kg-1 s-2, or equivalently in N.(m/kg)^2
GRAV = 6.67384e-11
R_PLANET = 6371000.0    # radius of the Earth in m
RHOAV = 5514.3          # Avergage density of the Earth

# Scaling of the GCMT moment tensor parameters in specfem
scaleM = 1e7 * RHOAV * (R_PLANET ** 5) * np.pi * GRAV * RHOAV

# Scaling of the FORCESOLUTION parameters
scaleF = RHOAV * (R_PLANET ** 5) * np.pi * GRAV * RHOAV

# Hello
NGLLX, NGLLY, NGLLZ = 5, 5, 5


class Variable:

    # Name of the variable in the ADIOS file
    name: str

    # Array content
    array: np.ndarray

    # Total size
    size: np.ndarray

    # Local dimension
    local_dim: np.ndarray

    # Global dimension
    global_dim: np.ndarray

    # Block id
    block_id: int

    def __init__(
            self, fh, varname: str, block_id: int | None = None, rtype: str | None = None,
            nsteps: int | None = None, comm: None = None):

        self.name = varname

        if rtype != 'scalar':
            # nx = 6500
            # size = comm.Get_size()
            # rank = comm.Get_rank()
            # shape = [size * nx]
            # start = [rank * nx]
            # count = [nx]
            self.offset = fh.read(f'{varname}/offset')
            self.local_dim = fh.read(f'{varname}/local_dim')
            self.size = fh.read(f'{varname}/size')
            self.global_dim = fh.read(f'{varname}/global_dim')

            # print(self.offset)
            self.offset = self.offset[block_id]
            self.local_dim = self.local_dim[block_id]
            self.global_dim = self.global_dim[block_id]
            self.size = self.size[block_id]

        if nsteps is not None:
            self.array = fh.read(
                f'{varname}/array',
                [self.offset, ], [self.local_dim],
                0, nsteps).T
            # self.array = fh.read(
            #     f'{varname}/array',
            #      start=start, count=count, step_start=0, step_count=nsteps, block_id=block_id).T

        elif rtype == 'scalar':
            self.array = fh.read(
                f'{varname}', block_id=block_id)[block_id]
        else:
            self.array = fh.read(
                f'{varname}/array',
                [self.offset, ], [self.local_dim],
                block_id=blockid)


# if( rank == 0 ):
# with-as will call adios2.close on fh at the end
# if only one rank is active pass MPI.COMM_SELF

dirs = ['attenuation_only', 'attenuation_ellipticity_only',
        'no_gravity_rotation', 'no_rotation', 'all_on']

if rank == 0:
    fig = plt.figure(figsize=(10, 11))

counter = 0

for _dir in dirs:
    print(_dir)
    forward_file = os.path.join('data', _dir, "forward_database_gf.bp")
    reciprocal_file = os.path.join('data', _dir, "reciprocal_database_gf.bp")
    seismograms = os.path.join('data', _dir, "II.BFO.*Z*")

    with adios2.open(reciprocal_file, "r", comm) as rh:

        with adios2.open(forward_file, "r", comm) as fh:

            # Get number of steps in the file
            nsteps = fh.steps()

            if (rank == 0):

                blockid = 0
                D = Variable(fh, 'displacement', 0, nsteps=nsteps)
                rrot = Variable(fh, 'rec_rotation', 0)
                rxi = Variable(fh, 'rec_xi', 0)
                reta = Variable(fh, 'rec_eta', 0)
                rgamma = Variable(fh, 'rec_gamma', 0)
                rlat = Variable(fh, 'rec_latitude', 0)
                rlon = Variable(fh, 'rec_longitude', 0)
                rdep = Variable(fh, 'rec_depth', 0)
                rspec = Variable(fh, 'rec_spec', 0)
                rslice = Variable(fh, 'rec_slice', 0)
                scale_displ = Variable(fh, 'scale_displ', 0, rtype='scalar')

                # Receiver location in element
                print(rxi.array[0], reta.array[0], rgamma.array[0])

                # Source
                sxi = Variable(fh, 'xi_source', 4)
                seta = Variable(fh, 'eta_source', 4)
                sgamma = Variable(fh, 'gamma_source', 4)

                print(sxi.array[0], seta.array[0], sgamma.array[0])
                Mxx = Variable(fh, 'Mxx', 4)
                Myy = Variable(fh, 'Myy', 4)
                Mzz = Variable(fh, 'Mzz', 4)
                Mxy = Variable(fh, 'Mxy', 4)
                Mxz = Variable(fh, 'Mxz', 4)
                Myz = Variable(fh, 'Myz', 4)

                # GLL points and weights (degree)
                npol = 4
                xigll, wxi, _ = gll_nodes(npol)
                etagll, weta, _ = gll_nodes(npol)
                gammagll, wgamma, _ = gll_nodes(npol)

                # epsilon_xx = Variable(rh, 'epsilon_xx', block_id=4, nsteps=nsteps)
                # epsilon_yy = Variable(rh, 'epsilon_yy', block_id=4, nsteps=nsteps)
                # epsilon_zz = Variable(rh, 'epsilon_zz', block_id=4, nsteps=nsteps)
                # epsilon_xy = Variable(rh, 'epsilon_xy', block_id=4, nsteps=nsteps)
                # epsilon_xz = Variable(rh, 'epsilon_xz', block_id=4, nsteps=nsteps)
                # epsilon_yz = Variable(rh, 'epsilon_yz', block_id=4, nsteps=nsteps)
                block_id = 0
                start = [500]
                count = [125]
                nsteps = rh.steps()
                epsilon_xx = rh.read(
                    f'epsilon_xx/array',
                    start=start, count=count,
                    step_start=0, step_count=nsteps, block_id=block_id).T
                epsilon_yy = rh.read(
                    f'epsilon_yy/array',
                    start=start, count=count,
                    step_start=0, step_count=nsteps, block_id=block_id).T
                epsilon_zz = rh.read(
                    f'epsilon_zz/array',
                    start=start, count=count,
                    step_start=0, step_count=nsteps, block_id=block_id).T
                epsilon_xy = rh.read(
                    f'epsilon_xy/array',
                    start=start, count=count,
                    step_start=0, step_count=nsteps, block_id=block_id).T
                epsilon_xz = rh.read(
                    f'epsilon_xz/array',
                    start=start, count=count,
                    step_start=0, step_count=nsteps, block_id=block_id).T
                epsilon_yz = rh.read(
                    f'epsilon_yz/array',
                    start=start, count=count,
                    step_start=0, step_count=nsteps, block_id=block_id).T

                # epsilon_yz = np.loadtxt('data/epsilon_yz.txt').T
                epsilon_xx = epsilon_xx.reshape((5, 5, 5, -1), order='F')
                epsilon_yy = epsilon_yy.reshape((5, 5, 5, -1), order='F')
                epsilon_zz = epsilon_zz.reshape((5, 5, 5, -1), order='F')
                epsilon_xy = epsilon_xy.reshape((5, 5, 5, -1), order='F')
                epsilon_xz = epsilon_xz.reshape((5, 5, 5, -1), order='F')
                epsilon_yz = epsilon_yz.reshape((5, 5, 5, -1), order='F')

                # print(epsilon_yz[1, 1, 1, 5000])
                repsilon_xx = Variable(
                    fh, 'epsilon_xx', block_id=0, nsteps=nsteps)
                repsilon_yy = Variable(
                    fh, 'epsilon_yy', block_id=0, nsteps=nsteps)
                repsilon_zz = Variable(
                    fh, 'epsilon_zz', block_id=0, nsteps=nsteps)
                repsilon_xy = Variable(
                    fh, 'epsilon_xy', block_id=0, nsteps=nsteps)
                repsilon_xz = Variable(
                    fh, 'epsilon_xz', block_id=0, nsteps=nsteps)
                repsilon_yz = Variable(
                    fh, 'epsilon_yz', block_id=0, nsteps=nsteps)

                repsilon_xx.array = repsilon_xx.array.reshape(
                    (5, 5, 5, -1), order='F')
                repsilon_yy.array = repsilon_yy.array.reshape(
                    (5, 5, 5, -1), order='F')
                repsilon_zz.array = repsilon_zz.array.reshape(
                    (5, 5, 5, -1), order='F')
                repsilon_xy.array = repsilon_xy.array.reshape(
                    (5, 5, 5, -1), order='F')
                repsilon_xz.array = repsilon_xz.array.reshape(
                    (5, 5, 5, -1), order='F')
                repsilon_yz.array = repsilon_yz.array.reshape(
                    (5, 5, 5, -1), order='F')

                # Get lagrange values at specific GLL poins
                shxi, shpxi = lagrange_any(sxi.array[0], xigll, npol)
                sheta, shpeta = lagrange_any(seta.array[0], xigll, npol)
                shgamma, shpgamma = lagrange_any(sgamma.array[0], xigll, npol)
                sepsilon = np.zeros((6, nsteps))

                for k in range(NGLLZ):
                    for j in range(NGLLY):
                        for i in range(NGLLX):
                            hlagrange = shxi[i] * sheta[j] * shgamma[k]

                            sepsilon[0, :] += epsilon_xx[i,
                                                         j, k, :] * hlagrange
                            sepsilon[1, :] += epsilon_yy[i,
                                                         j, k, :] * hlagrange
                            sepsilon[2, :] += epsilon_zz[i,
                                                         j, k, :] * hlagrange
                            sepsilon[3, :] += epsilon_xy[i,
                                                         j, k, :] * hlagrange
                            sepsilon[4, :] += epsilon_xz[i,
                                                         j, k, :] * hlagrange
                            sepsilon[5, :] += epsilon_yz[i,
                                                         j, k, :] * hlagrange

                # Moment tensor
                fm = 1.0
                M = np.array(
                    [[Mxx.array[0], fm*Mxy.array[0], fm*Mxz.array[0]],
                        [fm*Mxy.array[0],    Myy.array[0], fm*Myz.array[0]],
                        [fm*Mxz.array[0], fm*Myz.array[0],    Mzz.array[0]]])*scaleM

                # print(Mxx.array[0],Mzz.array[0],Mzz.array[0],)
                # Build SGT for dot product
                f = 1.0  # /np.sqrt(2)
                sgt = np.zeros((3, 3, nsteps))
                sgt[0, 0, :], sgt[0, 1, :], sgt[0, 2, :] = sepsilon[0, :], f * \
                    sepsilon[3, :], f*sepsilon[4, :]
                sgt[1, 0, :], sgt[1, 1, :], sgt[1, 2, :] = f * \
                    sepsilon[3, :],   sepsilon[1, :], f*sepsilon[5, :]
                sgt[2, 0, :], sgt[2, 1, :], sgt[2, 2, :] = f * \
                    sepsilon[4, :], f*sepsilon[5, :],   sepsilon[2, :]

                sgt = sgt.transpose(1, 0, 2)
                # dot product
                z = np.einsum('ji,ijk->k', M.T, sgt)
                plt.figure()
                plt.plot(z)
                plt.savefig('sgt_displacement.png', dpi=300)

                # Strain array
                # sys.exit()
                # Load addressing
                ibool_GF = Variable(fh, 'ibool_GF', 0)

                # Fix arrays
                D.array = D.array.reshape((3, -1, nsteps), order='F')
                rrot.array = rrot.array.reshape((3, 3, -1), order='F')
                ibool_GF.array = ibool_GF.array.reshape(
                    (5, 5, 5, -1), order='F')

                NGLLX, NGLLY, NGLLZ = 5, 5, 5

                # GLL points and weights (degree)
                npol = 4
                xigll, wxi, _ = gll_nodes(npol)
                etagll, weta, _ = gll_nodes(npol)
                gammagll, wgamma, _ = gll_nodes(npol)

                # Get lagrange values at specific GLL poins
                rhxi, rhpxi = lagrange_any(rxi.array[0], xigll, npol)
                rheta, rhpeta = lagrange_any(reta.array[0], xigll, npol)
                rhgamma, rhpgamma = lagrange_any(rgamma.array[0], xigll, npol)

                # get global number of that receiver
                uxd = np.zeros(nsteps)
                uyd = np.zeros(nsteps)
                uzd = np.zeros(nsteps)

                s = np.zeros((3, nsteps))

                irec = 0

                for k in range(NGLLZ):
                    for j in range(NGLLY):
                        for i in range(NGLLX):
                            iglob = ibool_GF.array[i, j, k, 0]

                            hlagrange = rhxi[i] * rheta[j] * rhgamma[k]

                            uxd = uxd + D.array[0, iglob-1, :] * hlagrange
                            uyd = uyd + D.array[1, iglob-1, :] * hlagrange
                            uzd = uzd + D.array[2, iglob-1, :] * hlagrange

                s[:, :] = scale_displ.array * (
                    np.outer(rrot.array[:, 0, irec], uxd) +
                    np.outer(rrot.array[:, 1, irec], uyd) +
                    np.outer(rrot.array[:, 2, irec], uzd))

                # s[:, :] = scale_displ.array * np.vstack((uxd,uyd,uzd))

                tr = read(seismograms)[0]
                idx = [1, 0, 2]

                # plt.subplot(3, 1, (2*i)+1)
                print('subplot:', 511 + counter)
                ax = fig.add_subplot(511 + counter)

                ax.plot(tr.times(), tr.data, 'k',
                        lw=0.75, label='Forward')
                ax.plot(tr.times(), s[2, :], 'r--', lw=0.75, label='Fw-GF')
                ax.plot(tr.times(), z/z.max() *
                        tr.data.max(), 'b:', lw=0.75, label='Reciprocal')
                ax.plot(tr.times(), tr.data-z/z.max() *
                        tr.data.max(), 'b', lw=0.5, label='error*10')
                ax.set_xlim(200, 600)
                absmax = np.max(
                    np.abs(np.hstack((s[2, :], tr.data, z/z.max()*tr.data.max()))))
                ax.set_ylim(-absmax, absmax)
                plot_label(
                    ax, f'{_dir}\nMax Disp: {np.max(absmax): .5} m', fontsize='xx-small', box=False)

                ax.tick_params(labelleft=False, left=False)
                ax.spines.right.set_visible(False)
                ax.spines.left.set_visible(False)
                ax.spines.top.set_visible(False)

                if counter == 0:
                    ax.legend(frameon=False)

                if counter == 4:
                    ax.set_xlabel('Time [s]')

                counter += 1

if rank == 0:
    # plt.subplots_adjust(hspace=0.45)
    fig.suptitle(
        'Z - event (lat,lon,dep)= (35.1500,26.8300,20.0), Crete,Greece', fontsize='small')
    fig.savefig('test1.pdf', dpi=300)
