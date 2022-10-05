import logging
import toml
from lwsspy.GF.constants_solver import NGLLX, NGLLY, NGLLZ
import numpy as np
import adios2
import traceback
import matplotlib.pyplot as plt
from mpi4py import MPI
import h5py
from lwsspy.math.cart2geo import cart2geo
from lwsspy.GF.simulation import Simulation


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

        # We also need to save every X-TH frame so that NSTEP_BETWEEN_FRAMES
        # we can thoroughly show the subsampling rate
        self.vars["DT"] = float(self.F.read("DT")[0])

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
        epsilon = np.zeros((6,
                            self.vars["NGLLX"], self.vars["NGLLY"], self.vars["NGLLZ"],
                            self.vars["NSPEC"], self.vars["NSTEPS"]))

        # Define ibool size
        ibool = np.zeros((
            self.vars["NGLLX"], self.vars["NGLLY"], self.vars["NGLLZ"],
            self.vars["NSPEC"]),
            dtype=int)

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
                logger.debug(f'{self.vars["NGLOB_LOCAL"][i]} -- {rankname}')
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
                        self.vars['NSPEC_LOCAL'][i], order='F') \
                    + CNGLOB[i] \
                    - 1
                # CNGLOB[i] gives the global values a subset of values
                # - 1 is to transfrom fortran to python indexing

                # Getting the epsilon
                for _i, _l in enumerate(['xx', 'yy', 'zz', 'xy', 'xz', 'yz']):

                    key = f'epsilon_{_l}'
                    local_dim = self.F.read(f'{key}/local_dim')[i]
                    offset = self.F.read(f'{key}/offset')[i]

                    epsilon[_i, :, :, :, CNSPEC[i]:CNSPEC[i+1], :] = self.F.read(
                        f'{key}/array', start=[offset],
                        count=[NGLL3*self.vars['NSPEC_LOCAL'][i]],
                        step_start=0, step_count=self.vars['NSTEPS'],
                        block_id=0).transpose().reshape(
                        NGLLX, NGLLY, NGLLZ, self.vars['NSPEC_LOCAL'][i], self.vars['NSTEPS'], order='F')

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

# Read input file dict


def read_toml(file: str):
    return toml.load(file)


class Adios2HDF5(object):

    DB: h5py.File
    consistent: bool = False

    def __init__(self,
                 h5file: str,
                 Nfile: str, Efile: str, Zfile: str,
                 config_file: str) -> None:

        # Output filename
        self.h5file = h5file

        # Component filenames
        self.filenames = dict(N=Nfile, E=Efile, Z=Zfile)

        # Get important infor for the conversion
        self.config = read_toml(config_file)
        self.get_timing()

        # Check consistency between components
        self.check_consistency()

    def get_timing(self):
        # Get the timining from the mesh file
        S = Simulation(**self.config)
        S.get_timestep_period()
        self.dt = S.ndt
        self.t0 = S.t0

        # STF midpoint! Note that due to the actual dt used this may not be hit by a
        # sample!
        self.tc = S.tc

    def check_consistency(self):
        """
        First we run a simple check to see whether all the variable parameters
        across all components are correct. This parameters have to be all the same
        since they depend on tagged elements. Tagged elements have to be the same
        across components

        We don't save the variables of the consisteny check simply because
        the consistency-check, and reading of the shape/size arrays takes no
        time.
        """

        vardict = dict()

        # For each component load the correct variables
        for _comp, _afile in self.filenames.items():

            with ProcessAdios(_afile) as P:
                P.load_base_vars()
                vardict[_comp] = P.vars

        # Get components
        ckeys = list(self.filenames.keys())

        # Get list of keys in the files
        keys = list(vardict[ckeys[0]].keys())

        # Check whether arrays agree in shapes and values
        try:
            for _key in keys:
                np.testing.assert_almost_equal(
                    vardict[ckeys[0]][_key], vardict[ckeys[1]][_key])
                np.testing.assert_almost_equal(
                    vardict[ckeys[0]][_key], vardict[ckeys[2]][_key])
        except Exception as e:
            print(e)
            raise ValueError(
                'The error was raised in the consistency check.\n'
                'Somehow it seems like the ADIOS files and their array sizes\n'
                'are inconsistent.')

        # Tell the class that the files are consistent.
        self.consistent = True

    def write(self):

        if self.consistent is False:
            raise ValueError('Component files are inconsistent.')

        # Grab the things that were identified by simulation class:
        self.DB.create_dataset("DT", data=self.dt)
        self.DB.create_dataset("TC", data=self.tc)
        self.DB.create_dataset("T0", data=self.t0)
        self.DB.create_dataset("Network", data=self.config['network'])
        self.DB.create_dataset("Station", data=self.config['station'])

        # This factor combines both removin the initial force + conversion to dyn
        self.DB.create_dataset(
            "FACTOR", data=1e7*float(self.config['force_factor']))

        for _i, (_comp, _afile) in enumerate(self.filenames.items()):

            logger.debug(72*"=")
            logger.debug(28*"=" + f" EPSILON FOR: {_comp} " + 28*"=")
            logger.debug(72*"=")

            with ProcessAdios(_afile) as P:

                # We checked whether all components have the same type of
                # We
                if _i == 0:
                    P.load_base_vars()
                    for _key in list(P.vars.keys()):
                        if _key == "DT":
                            continue
                        self.DB.create_dataset(_key, data=P.vars[_key])

                # Once the small variables are written write the large ones
                P.load_large_vars()

                # Write ibool and coordinates only once
                if _i == 0:
                    self.DB.create_dataset('ibool', data=P.vars['ibool'])
                    self.DB.create_dataset('xyz', data=P.vars['xyz'])

                # NOTE Here we can put code for compression!!!!
                # I tried a little but it failed completely...
                # For now let's just save the data
                # Write component-wise epsilon
                # offset = np.min(P.vars['epsilon'])
                # norm = np.max(P.vars['epsilon'] - offset)

                # db.create_dataset(
                #     f'epsilon/{_comp}/array',
                #     data=((P.vars['epsilon']-offset)/norm).astype(np.float16))

                # db.create_dataset(f'epsilon/{_comp}/offset', data=offset)
                # db.create_dataset(f'epsilon/{_comp}/norm', data=norm)

                self.DB.create_dataset(
                    f'epsilon/{_comp}/array', data=P.vars['epsilon'].astype(np.float32))

    def open(self):
        self.DB = h5py.File(self.h5file, 'w')
        # self.DB.open()

    def close(self):
        self.DB.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()

        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
            return False  # uncomment to pass exception through

        return True
