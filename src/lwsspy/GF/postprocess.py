import logging
from lwsspy.GF.constants_solver import NGLLX, NGLLY, NGLLZ
import numpy as np
from numpy.typing import ArrayLike
import adios2
import typing as tp
import traceback
import matplotlib.pyplot as plt
from mpi4py import MPI
from lwsspy.math.cart2geo import cart2geo
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

logger = logging.getLogger('lwsspy.GF')
logger.setLevel('DEBUG')


class ProcessAdios(object):

    F: adios2.File
    vars: dict  # [str, tp.Iterable | ArrayLike | int | bool | float]

    def __init__(self,  filename: str) -> None:

        self.filename = filename
        self.vars = dict()

    def load_base_vars(self):

        self.vars["NSTEPS"] = int(self.F.steps())
        self.vars["NPROC"] = int(self.F.read("NPROC")[0])
        self.vars["PROC"] = self.F.read("PROC")[0]
        self.vars["NGLLX"] = NGLLX  # self.F.read("NGLLX")[0]
        self.vars["NGLLY"] = NGLLY  # self.F.read("NGLLY")[0]
        self.vars["NGLLZ"] = NGLLZ  # self.F.read("NGLLZ")[0]
        # self.vars["DT"] = self.F.read("DT")
        self.vars["NGLOB_LOCAL"] = self.F.read("NGLOB")
        self.vars["NGLOB"] = int(np.sum(self.vars["NGLOB_LOCAL"]))
        self.vars["NSPEC_LOCAL"] = self.F.read("NGF_UNIQUE_LOCAL")
        self.vars["NSPEC"] = int(np.sum(self.vars["NSPEC_LOCAL"]))
        self.vars["ELLIPTICITY"] = bool(self.F.read("ELLIPTICITY")[0])
        self.vars["TOPOGRAPHY"] = bool(self.F.read("TOPOGRAPHY")[0])

        if self.vars["ELLIPTICITY"]:

            Nrspl = self.F.read('rspl/local_dim')[0]
            Nellipticity_spline = self.F.read('ellipicity_spline/local_dim')[0]
            Nellipticity_spline2 = self.F.read(
                'ellipicity_spline2/local_dim')[0]

            self.vars['rspl'] = self.F.read(
                'rspl/array',  start=[0], count=[Nrspl])
            self.vars['ellipticity_spline'] = self.F.read(
                'ellipicity_spline/array', start=[0], count=[Nellipticity_spline])
            self.vars['ellipticity_spline2'] = self.F.read(
                'ellipicity_spline2/array', start=[0], count=[Nellipticity_spline2])

        if self.vars["TOPOGRAPHY"]:

            # Get parameters
            self.vars["NX_BATHY"] = int(self.F.read('NX_BATHY')[0])
            self.vars["NY_BATHY"] = int(self.F.read('NY_BATHY')[0])
            self.vars["RESOLUTION_TOPO_FILE"] = float(self.F.read(
                'RESOLUTION_TOPO_FILE')[0])
            self.vars["DX_BATHY"] = self.vars["RESOLUTION_TOPO_FILE"]/60

            # Get topography
            self.vars["BATHY"] = self.F.read(
                'ibathy_topo/array', start=[0],
                count=[self.vars["NX_BATHY"]*self.vars["NY_BATHY"], ],
                block_id=0).reshape(
                    self.vars["NX_BATHY"], self.vars["NY_BATHY"], order='F')

    def load_large_vars(self):

        # GLOBAL ARRAY DIMENSIONS
        if "NGLOB" not in self.vars:
            self.load_base_vars()
            # self.sanity_check()

        # Cube gll points
        NGLL3 = self.vars['NGLLX'] * self.vars['NGLLY'] * self.vars['NGLLZ']

        # Allocate arrays
        xyz = np.zeros((self.vars["NGLOB"], 3), dtype=np.float32)

        # Define strain size convention xx,yy,zz,xy,xz,yz
        epsilon = np.zeros((
            6,
            self.vars["NGLLX"], self.vars["NGLLY"], self.vars["NGLLZ"],
            self.vars["NSPEC"], self.vars["NSTEPS"]),
            dtype=np.float32)

        # Define ibool size
        ibool = np.zeros((
            self.vars["NGLLX"], self.vars["NGLLY"], self.vars["NGLLZ"],
            self.vars["NSPEC"]),
            dtype=int)

        def plot_coords_slice(i, x, y, z):
            _, lat, lon = cart2geo(x, y, z)
            fig = plt.figure()
            ax = plt.axes()
            ax.scatter(lon, lat, s=2, marker='o')
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            plt.savefig(f'slicecoords{i:02d}.png', dpi=300)
            plt.close(fig)

        # cumulative offset indeces.
        CNSPEC = np.hstack(
            (np.array([0]), np.cumsum(self.vars["NSPEC_LOCAL"])))
        CNGLOB = np.hstack(
            (np.array([0]), np.cumsum(self.vars["NGLOB_LOCAL"])))

        # Loop over processors
        logger.debug(
            f"{'NPROC':>5} | {'Offset':>10}{'Global':>10}{'Local':>10}{'NGL_LOCAL':>10}{'NGL_Global':>10}")
        logger.debug("-" * 55)
        for i in range(self.vars['NPROC']):

            # To access rank specific variables
            rankname = f'{i:d}'.zfill(5)

            # Only store things if there are points
            if self.vars['NGLOB_LOCAL'][i] > 0:
                logger.debug(self.vars['NGLOB_LOCAL'][i], rankname)
                # Getting coordinates
                x = dict()
                for _i, _l in enumerate(['x', 'y', 'z']):

                    # Offset
                    local_dim = self.F.read(f'{_l}/local_dim')[i]
                    global_dim = self.F.read(f'{_l}/global_dim')[i]
                    offset = self.F.read(f'{_l}/offset')[i]
                    if _i == 0:
                        logger.debug(
                            f"{i:>5d} | {offset:>10}{global_dim:>10}{local_dim:>10}{self.vars['NGLOB_LOCAL'][i]:>10}{self.vars['NGLOB']:>10}")

                    # Read and assign to global array
                    # x[_l] = self.F.read(f'{_l}/array', start=[offset], count=[
                    #     self.vars["NGLOB_LOCAL"][i], ], block_id=0)

                    # Assign to global array
                    xyz[CNGLOB[i]:CNGLOB[i+1], _i] = self.F.read(f'{_l}/array', start=[offset], count=[
                        self.vars["NGLOB_LOCAL"][i], ], block_id=0)
                    # x[_l]

                    logger.debug(
                        f'{i}--{_l} min/max: {np.min(xyz[CNGLOB[i]:CNGLOB[i+1], _i])}/{np.max(xyz[CNGLOB[i]:CNGLOB[i+1], _i])}')

                # plot_coords_slice(i, x['x'], x['y'], x['z'])

                # Getting ibool dimension and offset
                local_dim = self.F.read(f'ibool_GF/local_dim')[i]
                global_dim = self.F.read(f'ibool_GF/global_dim')[i]
                offset = self.F.read(f'ibool_GF/offset')[i]

                logger.debug(
                    f"      | {local_dim*i:>10}{global_dim:>10}{local_dim:>10}{self.vars['NSPEC_LOCAL'][i]:>10}{self.vars['NSPEC']:>10}")

                # Getting ibool
                ibool[:, :, :, CNSPEC[i]:CNSPEC[i+1]] = self.F.read(
                    f'ibool_GF/array',
                    start=[offset],
                    count=[NGLL3*self.vars['NSPEC_LOCAL'][i], ],
                    block_id=0).reshape(
                        NGLLX, NGLLY, NGLLZ,
                        self.vars['NSPEC_LOCAL'][i], order='F') + CNGLOB[i]  # OFFSET

                # Getting the epsilon
                for _i, _l in enumerate(['xx', 'yy', 'zz', 'xy', 'xz', 'yz']):

                    key = f'epsilon_{_l}'
                    local_dim = self.F.read(f'{key}/local_dim')[i]
                    offset = self.F.read(f'{key}/offset')[i]

                    epsilon[_i, :, :, :, CNSPEC[i]:CNSPEC[i+1], :] = self.F.read(
                        f'{key}/array', start=[offset],
                        count=[NGLL3*self.vars['NSPEC_LOCAL'][i], ],
                        step_start=0, step_count=self.vars['NSTEPS'],
                        block_id=0).reshape(
                            NGLLX, NGLLY, NGLLZ,
                            self.vars['NSPEC_LOCAL'][i], self.vars['NSTEPS'], order='F')

            else:
                logger.debug(f"Proc {i:d} does not have elements.")

        self.vars['xyz'] = xyz
        self.vars['epsilon'] = epsilon
        self.vars['ibool'] = ibool

    def open(self):
        self.F = adios2.open(self.filename, 'r')

    def close(self):
        self.F.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()

        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
            return False  # uncomment to pass exception through

        return True

    def plot_splines(self, out: str | None = None):

        fig = plt.figure()
        plt.plot(self.vars['rspl'], self.vars['ellipticity_spline'])
        plt.plot(self.vars['rspl'], self.vars['ellipticity_spline2'])

        if out is not None:
            plt.savefig(out, dpi=200)
            plt.close(fig)
        else:
            plt.show()

    def plot_topo(self, out: str | None = None):

        fig = plt.figure(figsize=(8, 3))
        extent = [0, 360, -90, 90]
        im = plt.imshow(self.vars['BATHY'].T, extent=extent, origin='upper')
        plt.colorbar(im)

        if out is not None:
            plt.savefig(out, dpi=200)
            plt.close(fig)
        else:
            plt.show()


class ProcessAdios3(object):

    P: tp.Dict[str, ProcessAdios]

    def __init__(self,  Nfile: str, Efile: str, Zfile: str, only: str | None = None) -> None:

        # Define files to work on
        if only is not None:
            if only in ['', '', '']:
                self.filenames = dict()
            else:
                raise ValueError('N')
        else:
            self.filenames = dict(N=Nfile, E=Efile, Z=Zfile)

            self.vars = dict(N=dict(), E=dict(), Z=dict())

    def load_basic_vars(self):

        for _, P in self.P.items():
            P.load_base_vars()

    def check_consistency(self):
        """Check consistency in sizes across the different Process instances."""
        # Check topogrpahy NX and NY for topo if topo true

        # Check ellipticity splines if ellipticity True

        # Check NPROCS

        # If NPROCS consistent, check

        # CHECK
        # - NGLOB
        # - NSPEC_UNIQUE_LOCAL
        # -

        pass

    def open(self):
        for key, filename in self.filenames.items():
            self.P[key] = ProcessAdios(filename)
            self.P[key].open()

    def close(self):
        for _, P in self.P.items():
            P.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()

        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
            return False  # uncomment to pass exception through

        return True
