from scipy.spatial import KDTree  # Only import to avoid a certain error message
import time
import logging
import sys
import toml
import typing as tp
from lwsspy.GF.constants_solver import NGLLX, NGLLY, NGLLZ
import numpy as np
import adios2
import traceback
import matplotlib.pyplot as plt
from mpi4py import MPI
import h5py
from lwsspy.GF.simulation import Simulation
from lwsspy.GF.utils import read_toml
from pprint import pprint
# from lwsspy.GF.postprocess import ProcessAdios

# if tp.TYPE_CHECKING:
#     from adios2 import File  # type: ignore
from mpi4py.MPI import Intracomm

logger = logging.getLogger('lwsspy.GF')
logger.setLevel('DEBUG')


class ProcessAdios(object):

    F: adios2.File
    vars: dict  # [str, tp.Iterable | ArrayLike | int | bool | float]

    def __init__(self,  filename: str, comm: Intracomm) -> None:

        self.filename = filename
        self.vars = dict()
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def load_base_vars(self):

        self.vars["NSTEPS"] = int(self.F.steps())
        self.vars["NPROC"] = int(self.F.read("NPROC")[0])
        self.vars["PROC"] = self.F.read("PROC")[0]
        self.vars["NGLLX"] = NGLLX  # self.F.read("NGLLX")[0]
        self.vars["NGLLY"] = NGLLY  # self.F.read("NGLLY")[0]
        self.vars["NGLLZ"] = NGLLZ  # self.F.read("NGLLZ")[0]

        self.vars["NGLOB_LOCAL"] = self.F.read("NGLOB")
        self.vars["NGLOB"] = int(np.sum(self.vars["NGLOB_LOCAL"]))
        self.vars["NSPEC_LOCAL"] = self.F.read("NGF_UNIQUE_LOCAL")
        self.vars["NSPEC"] = int(np.sum(self.vars["NSPEC_LOCAL"]))
        self.vars["ELLIPTICITY"] = bool(self.F.read("ELLIPTICITY")[0])
        self.vars["TOPOGRAPHY"] = bool(self.F.read("TOPOGRAPHY")[0])

        # Adjacency values
        self.vars["USE_BUFFER_ELEMENTS"] = bool(
            self.F.read("USE_BUFFER_ELEMENTS")[0])

        # Get neighbor values
        if self.vars["USE_BUFFER_ELEMENTS"]:
            self.vars["NEIGHBORS_LOCAL"] = self.F.read("NUM_NEIGHBORS")
            self.vars["NEIGHBORS"] = int(np.sum(self.vars["NEIGHBORS_LOCAL"]))

        # We also need to save every X-TH frame so that NSTEP_BETWEEN_FRAMES
        # we can thoroughly show the subsampling rate
        self.vars["DT"] = float(self.F.read("DT")[0])

        # cumulative offset indeces.
        self.vars['CNSPEC'] = np.hstack(
            (np.array([0]), np.cumsum(self.vars["NSPEC_LOCAL"])))
        self.vars['CNGLOB'] = np.hstack(
            (np.array([0]), np.cumsum(self.vars["NGLOB_LOCAL"])))

        # Get neighbor values
        if self.vars["USE_BUFFER_ELEMENTS"]:
            self.vars['CNEIGH'] = np.hstack(
                (np.array([0]), np.cumsum(self.vars["NEIGHBORS_LOCAL"])))

        # Full shapes for the HDF5 file
        # -----------------------------
        # ibool
        self.vars['ibool_shape'] = (
            self.vars["NGLLX"], self.vars["NGLLY"], self.vars["NGLLZ"],
            self.vars["NSPEC"])

        # epsilon/component
        self.vars['epsilon_shape'] = (
            6, self.vars["NGLLX"], self.vars["NGLLY"], self.vars["NGLLZ"],
            self.vars["NSPEC"], self.vars["NSTEPS"])

        # xyz
        self.vars['xyz_shape'] = (self.vars['NGLOB'], 3)

        # Adjacency shape
        if self.vars["USE_BUFFER_ELEMENTS"]:
            self.vars['xadj_shape'] = (self.vars["NSPEC"] + 1,)
            self.vars['adj_shape'] = (self.vars["NEIGHBORS"],)

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

    def get_xyz(self, i):
        """Gets ``xyz`` for a single slice ``i``."""

        # GLOBAL ARRAY DIMENSIONS
        if "NGLOB" not in self.vars:
            self.load_base_vars()

        # To access rank specific variables
        rankname = f'{i:d}'.zfill(5)

        # Only store things if there are points
        if self.vars['NGLOB_LOCAL'][i] > 0:
            logger.debug(f'{self.vars["NGLOB_LOCAL"][i]} -- {rankname}')

            # Make arrays
            xyz = np.zeros((self.vars["NGLOB_LOCAL"][i], 3), dtype='f')

            # Getting coordinates
            for _i, _l in enumerate(['x', 'y', 'z']):

                # Offset
                local_dim = self.F.read(f'{_l}/local_dim')[i]
                global_dim = self.F.read(f'{_l}/global_dim')[i]
                offset = self.F.read(f'{_l}/offset')[i]
                if _i == 0:
                    logger.debug(
                        f"{i:>5d} | {offset:>10}{global_dim:>10}{local_dim:>10}{self.vars['NGLOB_LOCAL'][i]:>10}{self.vars['NGLOB']:>10}")

                # Assign to global array
                xyz[:, _i] = self.F.read(f'{_l}/array', start=[offset], count=[
                    self.vars["NGLOB_LOCAL"][i], ], block_id=0)
                # x[_l]

                logger.debug(
                    f'{i}--{_l} min/max: {np.min(xyz[:, _i])}/{np.max(xyz[:, _i])}')

            return xyz
        else:
            logger.debug(f"Proc {i:d} does not have elements.")

            return None

    def get_epsilon_minmax(self):
        '''This gets the overall minimum and maximum for all epsilon arrays.'''

        mins = []
        maxs = []

        for _, _l in enumerate(['xx', 'yy', 'zz', 'xy', 'xz', 'yz']):
            key = f'epsilon_{_l}'
            maxs.append(np.float64(
                self.F.available_variables()[f'{key}/array']['Max']))
            mins.append(np.float64(
                self.F.available_variables()[f'{key}/array']['Min']))

        return np.min(mins), np.max(maxs)

    def get_epsilon(self, i, norm, dtype):
        """Gets ``epsilon`` for a single slice ``i``."""

        # GLOBAL ARRAY DIMENSIONS
        if "NGLOB" not in self.vars:
            self.load_base_vars()

        # To access rank specific variables
        rankname = f'{i:d}'.zfill(5)

        # Cube gll points
        NGLL3 = self.vars['NGLLX'] * self.vars['NGLLY'] * self.vars['NGLLZ']

        # Only store things if there are points
        if self.vars['NGLOB_LOCAL'][i] > 0:
            logger.debug(f'{self.vars["NGLOB_LOCAL"][i]} -- {rankname}')

            epsilon = np.zeros(
                (6, self.vars["NGLLX"], self.vars["NGLLY"], self.vars["NGLLZ"],
                 self.vars['NSPEC_LOCAL'][i], self.vars["NSTEPS"]))

            # Getting the epsilon
            for _i, _l in enumerate(['xx', 'yy', 'zz', 'xy', 'xz', 'yz']):
                logger.debug(f'... Loading strain component {_l}')
                key = f'epsilon_{_l}'
                local_dim = self.F.read(f'{key}/local_dim')[i]
                offset = self.F.read(f'{key}/offset')[i]

                epsilon[_i, :, :, :, :, :] = self.F.read(
                    f'{key}/array', start=[offset],
                    count=[NGLL3*self.vars['NSPEC_LOCAL'][i]],
                    step_start=0, step_count=self.vars['NSTEPS'],
                    block_id=0).transpose().reshape(
                    NGLLX, NGLLY, NGLLZ, self.vars['NSPEC_LOCAL'][i],
                    self.vars['NSTEPS'], order='F') / norm

            return epsilon.astype(dtype, copy=False)
        else:
            logger.debug(f"Proc {i:d} does not have elements.")
            return None

    def get_epsilon_comp(self, i, comp):
        """Gets ``epsilon`` for a single slice ``i``."""

        # GLOBAL ARRAY DIMENSIONS
        if "NGLOB" not in self.vars:
            self.load_base_vars()

        # To access rank specific variables
        rankname = f'{i:d}'.zfill(5)

        # Cube gll points
        NGLL3 = self.vars['NGLLX'] * self.vars['NGLLY'] * self.vars['NGLLZ']

        # Only store things if there are points
        if self.vars['NGLOB_LOCAL'][i] > 0:
            logger.debug(f'{self.vars["NGLOB_LOCAL"][i]} -- {rankname}')

            # Getting the epsilon
            logger.debug(f'... Loading strain component {comp}')
            key = f'epsilon_{comp}'
            local_dim = self.F.read(f'{key}/local_dim')[i]
            offset = self.F.read(f'{key}/offset')[i]

            epsilon_comp = self.F.read(
                f'{key}/array', start=[offset],
                count=[NGLL3*self.vars['NSPEC_LOCAL'][i]],
                step_start=0, step_count=self.vars['NSTEPS'],
                block_id=0).transpose().reshape(
                NGLLX, NGLLY, NGLLZ, self.vars['NSPEC_LOCAL'][i], self.vars['NSTEPS'], order='F')

            return epsilon_comp
        else:
            logger.debug(f"Proc {i:d} does not have elements.")
            return None

    def get_adjacency(self, i):

        local_dim = self.F.read(f'xadj_gf/local_dim')[i]
        offset = self.F.read(f'xadj_gf/offset')[i]

        xadj = self.F.read(
            f'xadj_gf/array',
            start=[offset],
            count=[self.vars['NSPEC_LOCAL'][i] + 1],
            block_id=0) \
            + self.vars['CNEIGH'][i]

        # count is always 1 more than number of elements

        local_dim = self.F.read(f'adjncy_gf/local_dim')[i]
        offset = self.F.read(f'adjncy_gf/offset')[i]
        adjacency = self.F.read(
            f'adjncy_gf/array',
            start=[offset],
            count=[self.vars['NEIGHBORS_LOCAL'][i]],
            block_id=0) \
            + self.vars['CNSPEC'][i] \
            - 1
        # + self.vars['CNSPEC'][i] --> Add number of elements from first slice
        # - 1                      --> Remove index for Python indexing.

        return xadj.astype('i'), adjacency.astype('i')

    def get_ibool(self, i):
        """Gets ibool for a single slice ``i``."""

        # GLOBAL ARRAY DIMENSIONS
        if "NGLOB" not in self.vars:
            self.load_base_vars()

        # To access rank specific variables
        rankname = f'{i:d}'.zfill(5)

        # Cube gll points
        NGLL3 = self.vars['NGLLX'] * self.vars['NGLLY'] * self.vars['NGLLZ']

        ibool = np.zeros(
            (self.vars["NGLLX"], self.vars["NGLLY"], self.vars["NGLLZ"],
             self.vars['NSPEC_LOCAL'][i]), dtype='i')

        # Only store things if there are points
        if self.vars['NGLOB_LOCAL'][i] > 0:
            logger.debug(f'{self.vars["NGLOB_LOCAL"][i]} -- {rankname}')

            ibool = np.zeros(
                (self.vars["NGLLX"], self.vars["NGLLY"], self.vars["NGLLZ"],
                 self.vars['NSPEC_LOCAL'][i]), dtype=int)

            # Getting ibool dimension and offset
            local_dim = self.F.read(f'ibool_GF/local_dim')[i]
            global_dim = self.F.read(f'ibool_GF/global_dim')[i]
            offset = self.F.read(f'ibool_GF/offset')[i]

            logger.debug(
                f"      | {local_dim*i:>10}{global_dim:>10}{local_dim:>10}{self.vars['NSPEC_LOCAL'][i]:>10}{self.vars['NSPEC']:>10}")

            # Getting ibool
            ibool[:, :, :, :] = self.F.read(
                f'ibool_GF/array',
                start=[offset],
                count=[NGLL3*self.vars['NSPEC_LOCAL'][i], ],
                block_id=0).reshape(
                    NGLLX, NGLLY, NGLLZ,
                    self.vars['NSPEC_LOCAL'][i], order='F') \
                + self.vars['CNGLOB'][i] \
                - 1
            # CNGLOB[i] gives the global values a subset of values
            # - 1 is to transfrom fortran to python indexing

            return ibool
        else:
            logger.debug(f"Proc {i:d} does not have elements.")
            return None

    def open(self):
        self.F = adios2.open(self.filename, 'r', self.comm)

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


class Adios2HDF5(object):

    DB: h5py.File
    consistent: bool = False
    subspace: bool = False     # Only saving the corners and center points if True
    precision: type = np.float32
    # LZF faster, but less compression,
    # GZIP slower but better compression
    # going with gzip as a default for now.
    compression: str | None = None
    compression_opts: int | None = None  # Only used for the strain

    comm: Intracomm

    def __init__(self,
                 h5file: str,
                 Nfile: str, Efile: str, Zfile: str,
                 config_file: str,
                 subspace: bool = False,
                 precision: str | None = None,
                 compression: str | None = None,
                 compression_opts: int | None = None,
                 comm: Intracomm | None = None) -> None:

        # MPI setup
        if isinstance(comm, Intracomm):
            self.comm = comm
        else:
            from mpi4py.MPI import COMM_WORLD
            self.comm = COMM_WORLD

        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Output filename
        self.h5file = h5file

        # Component filenames
        self.filenames = dict(N=Nfile, E=Efile, Z=Zfile)

        # Get important infor for the conversion
        if self.rank == 0:
            config = read_toml(config_file)

            # escape for nnodes config file
            if 'root' in config:
                self.config = config['root']['cfg']
            else:
                self.config = config

            self.get_timing()
        else:
            self.config = None
            self.dt = None
            self.t0 = None
            self.tc = None
            self.hdur = None

        self.setup_broadcast()

        # Get subspace if needed
        self.subspace = subspace

        # Precision decision
        if precision is not None:
            if precision == 'single':
                self.precision = np.float32
            elif precision == 'double':
                self.precision = np.float64
            elif precision == 'half':
                self.precision = np.float16
            else:
                raise ValueError(
                    f'Precision Value {precision} not implemented')

        # Compression
        if compression is not None:
            if compression in ['lzf', 'gzip', 'szip']:
                self.compression = compression
            else:
                raise ValueError(f'Compression {compression} not implemented')

        # Check default
        if compression is not None:
            self.compression_opts = compression_opts

        self.comm.Barrier()

        # Check consistency between components
        if self.rank == 0:
            self.check_consistency()
        else:
            self.consistent = None

        self.consistent = self.comm.bcast(self.consistent, root=0)

    def setup_broadcast(self):
        # Bcast variables
        self.config = self.comm.bcast(self.config, root=0)
        self.dt = self.comm.bcast(self.dt, root=0)
        self.t0 = self.comm.bcast(self.t0, root=0)
        self.tc = self.comm.bcast(self.tc, root=0)
        self.hdur = self.comm.bcast(self.hdur, root=0)

    def get_timing(self):

        # Get the timining from the mesh file
        S = Simulation(**self.config)
        S.get_timestep_period()
        self.dt = S.ndt
        self.t0 = S.t0

        # STF midpoint! Note that due to the actual dt used this may not be hit by a
        # sample!
        self.tc = S.tc
        self.hdur = S.hdur

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

            with ProcessAdios(_afile, MPI.COMM_SELF) as P:
                P.load_base_vars()
                vardict[_comp] = P.vars

                # Get actual number of slices
                slices = np.where(P.vars['NGLOB_LOCAL'] != 0)[0]

                if self.size != len(slices):
                    raise ValueError(
                        f'This is designed to be run on '
                        f'{len(slices)} ranks. Adjust mpi call.')

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
            print(f'Sizes of {_key} did not match!')
            raise ValueError(
                'The error was raised in the consistency check.\n'
                'Somehow it seems like the ADIOS files and their array sizes\n'
                'are inconsistent.')

        # Tell the class that the files are consistent.
        self.consistent = True

    def write(self):

        if self.consistent is False:
            raise ValueError('Component files are inconsistent.')

        # Set type for
        stringtype = h5py.string_dtype(encoding='utf-8', length=10)

        # Grab the things that were identified by simulation class:
        self.DB.create_dataset("DT", data=self.dt)
        self.DB.create_dataset("TC", data=self.tc)
        self.DB.create_dataset("T0", data=self.t0)
        self.DB.create_dataset("HDUR", data=self.hdur)
        self.DB.create_dataset(
            "Network", data=str.encode(self.config['network']), dtype=stringtype)
        self.DB.create_dataset(
            "Station", data=str.encode(self.config['station']), dtype=stringtype)
        self.DB.create_dataset(
            "latitude", data=self.config['station_latitude'])
        self.DB.create_dataset(
            "longitude", data=self.config['station_longitude'])
        self.DB.create_dataset("burial", data=self.config['station_burial'])

        # This factor combines both removin the initial force + conversion to dyn
        self.DB.create_dataset(
            "FACTOR", data=1e7*float(self.config['force_factor']))
        self.comm.Barrier()

        if self.rank == 0:
            t0 = time.time()

        for _i, (_comp, _afile) in enumerate(self.filenames.items()):

            if self.rank == 0:
                logger.debug(72*"=")
                logger.debug(28*"=" + f" EPSILON FOR: {_comp} " + 28*"=")
                logger.debug(72*"=")

            with ProcessAdios(_afile, self.comm) as P:
                if self.rank == 0:
                    t00 = time.time()
                P.load_base_vars()

                slices = np.where(P.vars['NGLOB_LOCAL'] != 0)[0]

                # We checked whether all components have the same type of
                # We
                if _i == 0:
                    for _key in list(P.vars.keys()):
                        if _key == "DT":
                            continue

                        if (self.subspace) \
                                and (_key in ["NGLLX", "NGLLY", "NGLLZ"]):
                            self.DB.create_dataset(_key, data=self.subspace)
                        else:
                            self.DB.create_dataset(_key, data=P.vars[_key])

                # Create the three large variable datasets
                # Only write the coordinates and ibool arrays once.
                if _i == 0:

                    if self.rank == 0:
                        print('Shape   XYZ', P.vars['xyz_shape'])

                    xyz_ds = self.DB.create_dataset(
                        'xyz', P.vars['xyz_shape'], dtype='f')

                    if self.rank == 0:
                        print('Shape IBOOL', P.vars['ibool_shape'])

                    ibool_ds = self.DB.create_dataset(
                        'ibool', P.vars['ibool_shape'], dtype='i')

                    if P.vars["USE_BUFFER_ELEMENTS"]:
                        if self.rank == 0:
                            print('Shape x neighbors', P.vars['xadj_shape'])

                        xadj_ds = self.DB.create_dataset(
                            'xadj', P.vars['xadj_shape'], dtype='i')
                        if self.rank == 0:
                            print('Shape neighbors', P.vars['adj_shape'])

                        adjacency_ds = self.DB.create_dataset(
                            'adjacency', P.vars['adj_shape'], dtype='i')

                # Create epsilon for each components
                disp_ds = self.DB.create_dataset(
                    f'displacement/{_comp}/array', P.vars['epsilon_shape'],
                    dtype=self.precision,
                    chunks=(6, 5, 5, 5, 1, P.vars['epsilon_shape'][-1]),
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                    shuffle=True
                )

                norm = np.abs(P.get_epsilon_minmax()).max()

                if self.rank == 0:
                    print(norm)

                norm_ds = self.DB.create_dataset(
                    f'epsilon/{_comp}/norm', data=norm)

                # Create epsilon for each components
                epsilon_ds = self.DB.create_dataset(
                    f'epsilon/{_comp}/array', P.vars['epsilon_shape'],
                    dtype=self.precision,
                    chunks=(6, 5, 5, 5, 1, P.vars['epsilon_shape'][-1]),
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                    shuffle=True
                )

                j = slices[self.rank]

                if self.rank == 0:
                    t000 = time.time()

                if _i == 0:

                    self.comm.Barrier()

                    with xyz_ds.collective:

                        if P.vars['NGLOB_LOCAL'][j] > 0:
                            # Getting the coordinates xyz
                            xyz = P.get_xyz(j)

                            print(self.rank, 'hello 1', xyz.shape,
                                  P.vars['CNGLOB'][j+1]-P.vars['CNGLOB'][j])

                            xyz_ds[
                                P.vars['CNGLOB'][j]:P.vars['CNGLOB'][j+1], :] = xyz

                            del xyz

                    self.comm.Barrier()

                    with ibool_ds.collective:
                        if P.vars['NGLOB_LOCAL'][j] > 0:
                            # Getting the addressing array ibool
                            ibool = P.get_ibool(j)

                            ibool_ds[
                                :, :, :,
                                P.vars['CNSPEC'][j]:P.vars['CNSPEC'][j+1]] = ibool

                            del ibool

                    # Only get BUFFER_ELEMENTS if providided in the file

                    if P.vars["USE_BUFFER_ELEMENTS"]:
                        # Neighbor locations in neighbor array note that
                        # for slices
                        self.comm.Barrier()

                        with (
                                xyz_ds.collective,
                                adjacency_ds.collective):

                            k = (P.vars['NGLOB_LOCAL'] != 0).argmax(axis=0)

                            if P.vars['NGLOB_LOCAL'][j] > 0:

                                xadj, adjacency = P.get_adjacency(j)

                                if j == k:
                                    xadj_ds[
                                        P.vars['CNSPEC'][j]:
                                        P.vars['CNSPEC'][j+1]+1] = xadj
                                else:
                                    xadj_ds[
                                        P.vars['CNSPEC'][j] + 1:
                                        P.vars['CNSPEC'][j+1] + 1] = xadj[1:]

                                adjacency_ds[
                                    P.vars['CNEIGH'][j]:
                                    P.vars['CNEIGH'][j+1]] = adjacency

                self.comm.Barrier()

                with epsilon_ds.collective:
                    if P.vars['NGLOB_LOCAL'][j] > 0:
                        epsilon_ds[
                            :, :, :, :, P.vars['CNSPEC'][j]:P.vars['CNSPEC'][j+1], :
                        ] = P.get_epsilon(j, norm, self.precision)

                if self.rank == 0:
                    t111 = time.time()
                    print(72*'-')
                    print(
                        f'+ --> Writing arrays for slice {j} took {t111-t000:.1f} seconds')
                    print(72*'-')

            # self.comm.Barrier()

            if self.rank == 0:
                t11 = time.time()
                print(72*'+')
                print(
                    f'+ --> Writing arrays for component {_comp} took {t11-t00:.1f} seconds')
                print(72*'+')

        if self.rank == 0:
            t1 = time.time()
            print(72*'=')
            print(f'      All arrays took {t1-t0:.1f} seconds')
            print(72*'=')

    def open(self):
        self.DB = h5py.File(self.h5file, 'w', driver='mpio', comm=self.comm)
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
