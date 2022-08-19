import os
import subprocess
from shutil import copytree, ignore_patterns
import typing as tp
import numpy as np
from .source import FORCESOLUTION, CMTSOLUTION
from . import utils
from .logger import logger


class Simulation:

    specfemdir: str

    forward_test: bool = False

    # Read Par_file
    pardict: tp.OrderedDict

    def __init__(
            self, specfemdir,
            station_latitude: float,
            station_longitude: float,
            station_burial: float,
            network: str,
            station: str,
            target_latitude: float | np.ndarray | tp.Iterable,
            target_longitude: float | np.ndarray | tp.Iterable,
            target_depth: float | np.ndarray | tp.Iterable,
            force_factor: float = 1e14,
            t0: float = 0.0, tc: float = 0.0,
            duration_in_min: float = 20.0,
            nstep: None | int = None,
            ndt: None | float = None,
            forward_test: bool = False,
            broadcast_mesh_model: bool = False,
            simultaneous_runs: bool = False,
            cmtsolutionfile: str | None = None) -> None:
        """Makes specfem directory into SGT database simulator.

        Note that the specfem Par_file should be written in the same way you
        would do a forward simulation. That is, set flags such as

        .. literal::

            NEX_??, NPROC_?? NCHUNKS -> Only 6 has been tested so far. MODEL
            ROTATION TOPOGRAPHY ELLIPTICITY GRAVITY ATTENUATION USE_ADIOS ->
            TRUE !!!



        Before Running the manager, check the DT that the mesher outputs. If
        ``nstep`` is not set for the class, NSTEP will be computed from the
        ``duration``, ``t0``, ``tc``, and ``dt``. If ``ndt`` is not provided for
        downsampling during simulation, the value for the approximately accurate
        period will be used.

        What does the simulation manager do?

        1. Edits Par_file for SGT saving
        2. Edits constants.h.in -> reverses rotation if ROTATION=True in
           Par_file
        3. Adds rundirs 0, 1, 2; one for each force E, N, Z.
            a. at the station adds force solution
            b. write GF_LOCATIONS for each directory
            c.

        What does the simulation manager not do?

        - Postprocessing of the SGT files, such as, combine an simulated SGT set
          to a single SGT file for each station.
        - Querying of the SGT

        """

        self.specfemdir = specfemdir
        self.station_latitude = station_latitude   # degree
        self.station_longitude = station_longitude  # degree
        self.station_burial = station_burial       # km
        self.target_latitude = target_latitude     # degree
        self.target_longitude = target_longitude   # degree
        self.target_depth = target_depth           # km

        # Simulation parameters
        self.force_factor = force_factor
        self.t0 = t0
        self.tc = tc
        self.duration_in_min = duration_in_min
        self.nstep = nstep
        self.ndt_requested = ndt
        self.ndt = ndt

        # Parameters for a forward backward test
        self.forward_test = forward_test
        self.cmtsolutionfile = cmtsolutionfile
        self.network = network
        self.station = station

        # Submission specific options
        self.simultaneous_runs = simultaneous_runs
        self.broadcast_mesh_model = broadcast_mesh_model

        # Logger
        self.logger = logger

        # Run setup
        self.check_inputs()
        self.setup()

    def check_inputs(self):

        # Station Location
        utils.checktypes([
            self.station_latitude,
            self.station_longitude,
            self.station_burial], error='Station location')

        # Target Location
        utils.checktypes([
            self.target_latitude,
            self.target_longitude,
            self.target_depth], error='Target location')

        # Check forward test specific values
        if self.forward_test:
            if self.cmtsolutionfile is None:
                raise ValueError(
                    'For forward test CMTSOLUTION must be provided')

    def create(self):
        """Actually creates all necessary directories, after .setup() is run."""

        if self.forward_test:

            # ignore the rundirs
            ignorings = ignore_patterns('run00*')

            # Copy specfemdirectory
            copytreecmd = f"""
            rsync -av \
                --exclude='run00*' \
                --exclude='.*' \
                --exclude='EXAMPLES' \
                --exclude='tests' \
                --exclude='utils' \
                --exclude='doc' \
                --exclude='DATABASES_MPI/*' \
                --exclude='obj/*' \
                --exclude='bin/*' \
                --delete --ignore-existing \
                {self.specfemdir}/ {self.specfemdir_forward}"""

            subprocess.check_call(copytreecmd, shell=True)
            # copytree(self.specfemdir, self.specfemdir_forward, ignore=ignorings)


        # make rundirs
        for _comp, _compdict in self.compdict.items():

            # Remove pre-existing directory
            if os.path.exists(_compdict['dir']):
                subprocess.check_call(f'rm -rf {_compdict["dir"]}', shell=True)

            # Make dir
            os.makedirs(_compdict["dir"])

            # DATA DIR
            DATADIR = os.path.join(_compdict["dir"], 'DATA')
            os.makedirs(DATADIR)

            # OUTPUT DIR
            OUTPUT_DIR = os.path.join(_compdict["dir"], 'OUTPUT_FILES')
            os.makedirs(OUTPUT_DIR)

            if self.simultaneous_runs is False:

                # Link DATABASES
                DATABASES_MPI_SOURCE = os.path.join(
                    self.specfemdir, "DATABASES_MPI")
                DATABASES_MPI_TARGET = os.path.join(
                    _compdict["dir"], "DATABASES_MPI")

                os.symlink(DATABASES_MPI_SOURCE, DATABASES_MPI_TARGET)

                # Link BINs
                BINS_SOURCE = os.path.join(
                    self.specfemdir, "bin")
                BINS_TARGET = os.path.join(
                    _compdict["dir"], "bin")

                os.symlink(BINS_SOURCE, BINS_TARGET)

        # Create Write all the files
        self.write_Par_file()
        self.write_STATIONS()
        self.write_GF_LOCATIONS()
        self.write_CMT()

    def setup(self):
        """Setting up the directory structure for specfem."""
        # Mostly setting up paths
        # self.specfemdir = os.path.abspath(self.specfemdir)

        # Source Time Function file
        self.stf_file = os.path.join(self.specfemdir, 'DATA', 'stf')

        # Constants file has to be updated for reciprocal simulations
        self.constants_file = os.path.join(
            self.specfemdir, 'setup', 'constants.h.in')

        # Par_file of course has to be updated as well.
        self.par_file = os.path.join(
            self.specfemdir, 'DATA', 'Par_file')

        # Read Par_file into dictionary from (include comments for posterity)
        self.pardict = utils.get_par_file(self.par_file, savecomments=True, verbose=False)

        # Get simulation sampling rate and min period
        self.dt, self.T = utils.get_dt_from_mesh_header(self.specfemdir)

        # Print statement showing the setup
        self.compdict = dict(
            N=dict(dir=os.path.join(self.specfemdir, 'run0001')),
            E=dict(dir=os.path.join(self.specfemdir, 'run0002')),
            Z=dict(dir=os.path.join(self.specfemdir, 'run0003'))
        )

        if self.forward_test:
            self.setup_forward()

    def setup_forward(self):
        basename = os.path.basename(self.specfemdir)
        upper = os.path.dirname(self.specfemdir)
        self.specfemdir_forward = os.path.join(
            upper, basename + '_forward')

        # Constants file has to be updated for reciprocal simulations
        self.constants_file_forward = os.path.join(
            self.specfemdir_forward, 'setup', 'constants.h.in')

        # Par_file of course has to be updated as well.
        self.par_file_forward = os.path.join(
            self.specfemdir_forward, 'DATA', 'Par_file')

        # Stations file
        self.stations_file_forward = os.path.join(
            self.specfemdir_forward, 'DATA', 'STATIONS')

        # Stations file
        self.CMTSOLUTION_file = os.path.join(
            self.specfemdir_forward, 'DATA', 'STATIONS')

        # Read Par_file into dictionary from (include comments for posterity)
        self.pardict_forward = utils.get_par_file(
            self.par_file, savecomments=True, verbose=False)

    def update_rotation(self):
        """Updates the rotation value in the ``constants.h.in``
        """

        if self.pardict['ROTATION']:
            utils.update_constants(self.constants_file,
                                   self.constants_file, rotation='-')

        if self.forward_test:
            utils.update_constants(self.constants_file,
                                   self.constants_file, rotation='+')

    def write_GF_LOCATIONS(self):

        # If target is a single place make a list
        if isinstance(self.target_latitude, float):
            self.target_latitude = [self.target_latitude]
            self.target_longitude = [self.target_longitude]
            self.target_depth = [self.target_depth]

        # If targets are a 2D (or more D) array flatten it
        elif isinstance(self.target_latitude, np.ndarray):
            self.target_latitude = self.target_latitude.flatten()
            self.target_longitude = self.target_longitude.flatten()
            self.target_depth = self.target_depth.flatten()

        for _, _compdict in self.compdict.items():

            # GF_LOCATIONS file
            locations_file = os.path.join(
                _compdict['dir'], 'DATA', 'GF_LOCATIONS')

            # Open GF locations file for each compenent
            with open(locations_file, 'w') as f:

                # Loop over provided target locations
                for _lat, _lon, _dep in zip(
                        self.target_latitude,
                        self.target_longitude,
                        self.target_depth):
                    f.write(f'{_lat:7.4f}   {_lon:7.4f}   {_dep:7.4f}')

    def write_FORCES(self):

        for _comp, _compdict in self.compdict.items():

            # GF_LOCATIONS file
            force_file = os.path.join(
                _compdict['dir'], 'DATA', 'FORCESOLUTION')

            # Set only one component to 1.0
            N, E, Z = 0.0, 0.0, 0.0

            if _comp == 'Z':
                Z = 1.0
            elif _comp == 'E':
                E = 1.0
            elif _comp == 'N':
                N = 1.0
            else:
                raise ValueError('Component must be N, E, or Z')

            # Create force and
            force = FORCESOLUTION(
                time_shift=0.0, hdur=0.0,
                latitude=self.station_latitude,
                longitude=self.station_longitude,
                depth=self.station_burial, stf=2,
                forcefactor=self.force_factor,
                vector_E=E, vector_N=N, vector_Z_UP=Z, force_no=1)

            # write it to subdir
            force.write(force_file)

    def write_STATIONS(self):
        """Only for forward test. Otherwise it doesn't matter."""

        for _comp, _compdict in self.compdict.items():

            # GF_LOCATIONS file
            stations_file = os.path.join(
                _compdict['dir'], 'DATA', 'STATIONS')

            # write stations file with one line
            with open(stations_file, 'w') as f:
                # STATION    NETWORK   LATITUDE  LONGITUDE  ELEVATION    BURIAL
                f.write(
                    "%-9s %5s %15.4f %12.4f %10.1f %6.1f\n"
                    % (self.station, self.network,
                       self.station_latitude, self.station_longitude, 0.0, self.station_burial*1.0e3)
                )

        if self.forward_test:
            # write stations file with one line
            with open(self.stations_file_forward, 'w') as f:
                # STATION    NETWORK   LATITUDE  LONGITUDE  ELEVATION    BURIAL
                f.write(
                    "%-9s %5s %15.4f %12.4f %10.1f %6.1f\n"
                    % (self.station, self.network,
                       self.station_latitude, self.station_longitude, 0.0, self.station_burial*1.0e3)
                )

    def write_CMT(self):
        """Only for forward test. Otherwise it doesn't matter."""

        if self.forward_test:
            # Read test cmt
            cmt = CMTSOLUTION.read(self.cmtsolutionfile)  # type: ignore
            cmt.write(self.CMTSOLUTION_file)

    def write_Par_file(self):

        # modify Reciprocal Par_file
        pardict = self.pardict.copy()

        pardict['SAVE_GREEN_FUNCTIONS'] = True

        # IF runs have to be parallel
        if self.simultaneous_runs:
            pardict['NUMBER_OF_SIMULTANEOUS_RUNS'] = 3
            pardict['BROADCAST_SAME_MESH_AND_MODEL'] = True
        else:
            pardict['NUMBER_OF_SIMULTANEOUS_RUNS'] = 1
            pardict['BROADCAST_SAME_MESH_AND_MODEL'] = False

        # pardict['NSTEP'] = self.nstep
        # pardict['DT'] = self.dt
        # pardict['T0'] = self.T0

        for _, _compdict in self.compdict.items():

            # GF_LOCATIONS file
            par_file = os.path.join(
                _compdict['dir'], 'DATA', 'Par_file')

            # Write Par file for each sub dir
            utils.write_par_file(pardict, par_file)

        if self.forward_test:

            # modify Reciprocal Par_file
            pardict = self.pardict.copy()
            pardict['SAVE_GREEN_FUNCTIONS'] = False

            pardict['NUMBER_OF_SIMULTANEOUS_RUNS'] = 1
            pardict['BROADCAST_SAME_MESH_AND_MODEL'] = False

            # Write Par_file
            utils.write_par_file(pardict, self.par_file_forward)

            pardict['NUMBER_OF_SIMULTANEOUS_RUNS'] = 3
            pardict['BROADCAST_SAME_MESH_AND_MODEL'] = True
            # pardict['NSTEP'] = self.nstep
            # pardict['DT'] = self.dt
            # pardict['T0'] = self.T0

    def __str__(self) -> str:
        rstr = "\n"
        rstr += "Reciprocal Simulation Setup:\n"
        rstr += "-----------------------------\n"
        rstr += f"Specfem basedir:{self.specfemdir:.>56}\n"
        rstr += f"E:{self.compdict['N']['dir']:.>70}\n"
        rstr += f"N:{self.compdict['N']['dir']:.>70}\n"
        rstr += f"Z:{self.compdict['N']['dir']:.>70}\n"
        rstr += "\n"
        rstr += f"Force Factor:{self.force_factor:.>59.4g}\n"
        rstr += f"T0:{self.t0:.>69.4f}\n"
        rstr += f"TC:{self.tc:.>69.4f}\n"
        rstr += f"DT:{self.dt:.>69.4f}\n"
        rstr += f"NDT requested:{self.ndt_requested:.>58.4f}\n"
        rstr += f"NDT:{self.ndt:.>68.4f}\n"
        rstr += "\n"
        rstr += f"ROTATION:{self.pardict['ROTATION']!s:.>63}\n"
        rstr += f"SIMULTANEOUS_RUNS:{self.simultaneous_runs!s:.>54}\n"
        rstr += f"BROADCAST_SAME_MESH_AND_MODEL:{self.pardict['BROADCAST_SAME_MESH_AND_MODEL']:.>42}\n"


        if self.forward_test:
            rstr += "\n"
            rstr += "Forward Test Setup:\n"
            rstr += "-------------------\n"
            rstr += f"Specfem forward:{self.specfemdir_forward:.>56}\n"

        return rstr

    def __repr__(self) -> str:
        self.__str__()

    def write_STF(self):
        pass
