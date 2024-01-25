import os
import subprocess
import typing as tp
import numpy as np
from copy import deepcopy
from . import utils
from .stf import create_stf
from .source import FORCESOLUTION, CMTSOLUTION
from .logger import logger


class Simulation:

    specfemdir: str

    forward_test: bool = False

    # Read Par_file
    pardict: tp.OrderedDict

    def __init__(
            self,
            specfemdir: str,
            stationdir: str | None = None,
            station_latitude: float | None = None,
            station_longitude: float | None = None,
            station_burial: float | None = None,
            network: str | None = None,
            station: str | None = None,
            target_file: str | None = None,
            target_latitude: float | np.ndarray | tp.Iterable | None = None,
            target_longitude: float | np.ndarray | tp.Iterable | None = None,
            target_depth: float | np.ndarray | tp.Iterable | None = None,
            par_file: str | None = None,
            element_buffer: int | None = None,
            force_factor: float = 1e14,
            t0: float = 0.0,
            tc: float = 0.0,
            duration_in_min: float = 20.0,
            nstep: None | int = None,
            subsample: bool = True,
            ndt: None | float = None,
            lpfilter: str = 'butter',
            forward_test: bool = False,
            forwardspecfemdir: str | None = None,
            forwardoutdir: str | None = None,
            broadcast_mesh_model: bool = False,
            simultaneous_runs: bool = False,
            cmtsolutionfile: str | None = None,
            overwrite: bool = False) -> None:
        """

        Makes specfem directory into Green function database simulator.


        Parameters
        ----------
        specfemdir : _type_
            base specfem directory
        stationdir : str | None, optional
            base directory for station simulations and GF files, by default None
        station_latitude : float | None, optional
            station latitude, by default None
        station_longitude : float | None, optional
            station longitude, by default None
        station_burial : float | None, optional
            station burial, by default None
        network : str | None, optional
            network, by default None
        station : str | None, optional
            station, by default None
        target_file : str | None, optional
            target file ``GF_LOCATIONS``, by default None
        target_latitude : float | np.ndarray | tp.Iterable | None, optional
            target latitude, only relevant if ``target_file`` not set,
            by default None
        target_longitude : float | np.ndarray | tp.Iterable | None, optional
            target longitude, only relevant if ``target_file`` not set,
            by default None
        target_depth : float | np.ndarray | tp.Iterable | None, optional
            target depth, only relevant if ``target_file`` not set,
            by default None
        par_file : str | None, optional
            needed to set up specfem, by default None
        element_buffer : int | None, optional
            how many buffer elements to set around tagged elements, by default None
        force_factor : float, optional
            factor to simulate the reciprocal forces with, by default 1e14
        t0 : float, optional
            starttime, by default 0.0
        tc : float, optional
            center of green function centroid, by default 0.0
        duration_in_min : float, optional
            duration in minutes, by default 20.0
        nstep : None | int, optional
            number of timesteps if you want to hardset a number of timestep,
            by default None
        subsample : bool, optional
            flag to turn on/off subsampling during the simulation,
            by default True
        ndt : None | float, optional
            requested new sampling rate, by default None
        lpfilter : str, optional
            type of low-pass filter, by default 'butter'
        forward_test : bool, optional
            flag to check whether you want to perform a test with
            a separate specfem directory to compute a
            specific event to test the green function extraction,
            by default False
        forwardoutdir : str | None, optional
            forward specfem directory (auto-generated), by default None
        broadcast_mesh_model : bool, optional
            flag to set in specfem [not necessary, anymore], by default False
        simultaneous_runs : bool, optional
            flag to set in specfem [not necessary, anymore], by default False
        cmtsolutionfile : str | None, optional
            if forward test a cmtsolution is required, by default None
        overwrite : bool, optional
            flag to overwrite certain direcories, by default False


        Notes
        -----

        The `specfem3D_globe` `Par_file` should be written in the same way
        you would do a normal forward simulation. That is, set flags such as

        ::

            NEX_*
            NPROC_*
            NCHUNKS
            MODEL
            ROTATION
            TOPOGRAPHY
            ELLIPTICITY
            GRAVITY
            ATTENUATION

            # IMPORTANT
            USE_ADIOS = .true.


        Before Running the manager, check the DT that the mesher outputs. If
        `nstep` is not set for the class, NSTEP will be computed from the
        `duration`, `t0`, `tc`, and `dt`. If `ndt` is not provided for
        downsampling during simulation, the value for the approximately accurate
        period will be used.

        What does the simulation manager do?

        1. Edits ``Par_file`` for reciprocal simulations
        2. Edits ``constants.h.in`` -> reverses rotation if ``ROTATION = .true``
           in ``Par_file``
        3. Now, now mesh specfem (class not required)
        4. Creates simulation directories from the specfem directory
            a. Writes ``FORCESOLUTION`` to ``DATA`` directory
            b. Writes ``GF_LOCATIONS`` to ``DATA`` directory
            c. Writes source time function to ``DATA`` directory
            d. Writes ``Par_file`` with correct parameters to ``DATA`` directory
                - Length of simulation ``NT``
                - starttime ``T0``

        What does the simulation manager not do?

        * Postprocessing of the Green function files, such as, combine an
          simulated Green function set to a single SGT file for each station.
        * Querying the Green function database

        """

        self.specfemdir = specfemdir
        self.stationdir = stationdir
        self.forwardspecfemdir = forwardspecfemdir
        self.forwardoutdir = forwardoutdir
        self.station_latitude = station_latitude   # degree
        self.station_longitude = station_longitude # degree
        self.station_burial = station_burial       # km
        self.target_latitude = target_latitude     # degree
        self.target_longitude = target_longitude   # degree
        self.target_depth = target_depth           # km
        self.target_file = target_file

        # Simulation parameters
        self.force_factor = float(force_factor)

        # Signal parameters
        self.t0 = t0
        self.tc = tc
        self.duration_in_min = duration_in_min
        self.nstep = nstep
        self.subsample = subsample
        self.ndt_requested = ndt
        self.ndt = ndt
        self.dt = None
        self.xth_sample = None
        self.hdur = None
        self.cutoff = None
        self.lpfilter = lpfilter

        if element_buffer is not None and element_buffer > 0:
            self.use_element_buffer = True
            self.element_buffer = element_buffer
        else:
            self.use_element_buffer = False
            self.element_buffer = 0

        # Parameters for a forward backward test
        self.forward_test = forward_test
        self.cmtsolutionfile = cmtsolutionfile
        self.network = network
        self.station = station
        self.use_forward_stf = False

        # Submission specific options
        self.simultaneous_runs = simultaneous_runs
        self.broadcast_mesh_model = broadcast_mesh_model
        self.par_file = par_file
        self.overwrite = overwrite

        # Logger
        self.logger = logger

        # Run setup
        if self.target_file is not None:
            self.read_GF_LOCATIONS()

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

    def create_specfem(self):
        """Only sets up specfem with the provided settings. You can mesh specfem
        after running this function. Use
        :func:`gf3d.simulation.Simulation.create` to create actual database
        directories."""

        # Specfem not updated
        if self.forward_test:

            self.logger.debug('Creating Forward Test directory.')

            # Copy specfemdirectory
            copytreecmd = f"""
            rsync -av \
                --include='.git/logs' \
                --exclude='run00*' \
                --exclude='.*' \
                --exclude='EXAMPLES' \
                --exclude='tests' \
                --exclude='utils' \
                --exclude='doc' \
                --exclude='DATABASES_MPI/*' \
                --exclude='obj/*' \
                --exclude='bin/*' \
                --delete \
                {self.forwardspecfemdir}/ {self.specfemdir_forward}"""

            subprocess.check_call(copytreecmd, shell=True)

        # After syncing since these would get overwritten otherwise
        self.write_Par_file()
        self.update_constants()

    def create(self):
        # make rundirs
        for _comp, _compdict in self.compdict.items():

            # Hello
            self.logger.debug(
                f'Creating Simulation Directory for {_comp}: {_compdict["dir"]}')

            # Remove pre-existing directory
            if (os.path.exists(_compdict['dir']) and self.overwrite) or not os.path.exists(_compdict['dir']):

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
        if self.forward_test:

            # Remove pre-existing directory
            if (os.path.exists(self.specfemdir_forward) and self.overwrite) or \
                not os.path.exists(self.specfemdir_forward):

                subprocess.check_call(f'rm -rf {self.specfemdir_forward}', shell=True)

                # Make dir
                os.makedirs(self.specfemdir_forward)

                # DATA DIR
                DATADIR = os.path.join(self.specfemdir_forward, 'DATA')
                os.makedirs(DATADIR)

                # OUTPUT DIR
                OUTPUT_DIR = os.path.join(self.specfemdir_forward, 'OUTPUT_FILES')
                os.makedirs(OUTPUT_DIR)

                if self.simultaneous_runs is False:

                    # Link DATABASES
                    DATABASES_MPI_SOURCE = os.path.join(
                        self.forwardspecfemdir, "DATABASES_MPI")
                    DATABASES_MPI_TARGET = os.path.join(
                        self.specfemdir_forward, "DATABASES_MPI")

                    os.symlink(DATABASES_MPI_SOURCE, DATABASES_MPI_TARGET)

                    # Link BINs
                    BINS_SOURCE = os.path.join(
                        self.forwardspecfemdir, "bin")
                    BINS_TARGET = os.path.join(
                        self.specfemdir_forward, "bin")

                    os.symlink(BINS_SOURCE, BINS_TARGET)


        # Write Rotation files
        self.logger.debug('Updating constants.h.in ...')

        # Update forces and STATIONS
        self.update_forces_and_stations()

    def update_forces_and_stations(self):
        # Writing all simulation relevant files
        self.write_FORCES()
        self.write_GF_LOCATIONS()
        self.write_STATIONS()
        self.write_CMT()

        self.get_timestep_period()
        self.write_STF()
        self.write_Par_file()

    def setup(self):
        """Setting up the directory structure for specfem."""
        # Mostly setting up paths
        # self.specfemdir = os.path.abspath(self.specfemdir)

        # Constants file has to be updated for reciprocal simulations
        self.constants_file = os.path.join(
            self.specfemdir, 'setup', 'constants.h.in')

        # Par_file of course has to be updated as well.
        if self.par_file is None:
            self.par_file = os.path.join(
                self.specfemdir, 'DATA', 'Par_file')

            if os.path.exists(self.par_file) is False:
                self.par_file = os.path.join(
                    self.specfemdir, 'run0001', 'DATA', 'Par_file')

                if os.path.exists(self.par_file) is False:
                    raise ValueError(
                        'No Par_file found. Check ./DATA and ./run0001/DATA for Par_file')

        # Read Par_file into dictionary from (include comments for posterity)
        self.logger.debug('Reading Par_file')
        self.pardict = utils.get_par_file(
            self.par_file, savecomments=True, verbose=False)

        # Print statement showing the setup
        if self.station is not None:
            self.compdict = dict(
                N=dict(dir=os.path.join(self.stationdir, 'N', 'specfem')),
                E=dict(dir=os.path.join(self.stationdir, 'E', 'specfem')),
                Z=dict(dir=os.path.join(self.stationdir, 'Z', 'specfem'))
            )

        if self.forward_test:
            self.setup_forward()

    def setup_forward(self):

        # Use specfem base name to find outdir for forward simulation
        if self.forwardoutdir is None:
            basename = os.path.basename(self.specfemdir)
            upper = os.path.dirname(self.specfemdir)
            self.specfemdir_forward = os.path.join(
                upper, basename + '_forward')
        else:
            self.specfemdir_forward = self.forwardoutdir

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
            self.specfemdir_forward, 'DATA', 'CMTSOLUTION')

        # Read test cmt
        self.cmt = CMTSOLUTION.read(self.cmtsolutionfile)  # type: ignore

        # Read Par_file into dictionary from (include comments for posterity)
        self.pardict_forward = utils.get_par_file(
            self.par_file, savecomments=True, verbose=False)

    def update_constants(self):
        """Updates the rotation value in the ``constants.h.in``
        """

        if self.pardict['ROTATION']:
            rotation = '-'
        else:
            rotation = '+'

        if self.subsample is True:
            external_stf = True
        else:
            external_stf = False

        utils.update_constants(
            self.constants_file, self.constants_file,
            rotation=rotation, external_stf=external_stf)

        if self.forward_test:
            # Don't do this to check really whether we can recover the
            # Seismogram
            utils.update_constants(
                self.constants_file_forward, self.constants_file_forward,
                rotation='+', external_stf=self.use_forward_stf)

    def get_timestep_period(self):
        """Gets the actual timestep from the mesh header, then takes the
        requested timestep ndt and fixes it to a subsample. Should the requested
        timestep be smaller than the simulation timesstep, the timestep is
        unchanged and self.dt and self.ndt remain the same.
        """

        # Do nothing if ndt is unchanged.
        if self.subsample is False:
            return

        if self.ndt_requested is None:
            logger.debug(
                '    .write_STF was called but no new time step was requested.')
            logger.debug(
                '        This is ok when the mesher is being prepared.')
            return

        # Get simulation sampling rate and min period
        self.logger.debug('Getting Values from mesher')
        self.dt, self.T = utils.get_dt_from_mesh_header(self.specfemdir)

        # Get number of timesteps
        duration_in_s = self.duration_in_min * 60.0 + self.tc

        # Compute new number of time
        if self.nstep is None:
            self.nstep = int(np.ceil(duration_in_s/self.dt))

        # Get every x sample
        self.xth_sample = int(self.ndt_requested//self.dt)

        if self.xth_sample == 0:
            raise ValueError(
                'Requested dt is smaller than simulation dt, a larger dt\n'
                'should be chosen, or omit the input parameter.')

        self.logger.debug(
            f"Number of timesteps for simulation: {self.nstep:6d}")
        self.logger.debug(
            f"Number of timesteps subsampled:     {int(self.nstep//self.xth_sample):6d}")
        self.logger.debug(
            f"Saving every x-th sample:           {self.xth_sample:6d}")

        # Show True sampling rate:
        self.ndt = self.dt * self.xth_sample
        self.fs = 1/self.ndt

        # Apparent frequency and required sampling time
        self.f = 1/self.T
        self.fs_mesh = 2*self.f

        self.logger.debug(f"Mesh resolves period: {self.T:.4f} s")
        self.logger.debug(f"Sampling at period:   {self.ndt:.4f} s")

        # Following specfem3D_globe we set the the half duration of the STF to
        # very short 5*DT, where DT is the integration sampling time ``self.dt``
        # Playing around with other half durations does not make sense for now.
        self.hdur = 5.0*self.dt
        self.logger.debug(f"hdur of step:         {self.hdur:.4f} s")

        # The distance between t0 and tc should be larger than the
        # 1.5 the haf duration, to ensure no abrupt start
        if 1.5 * self.hdur > np.abs(self.tc-self.t0):
            raise ValueError(
                f't0 and tc too close. \n1.5*hdur > |tc-t0| \n[{1.5*self.hdur:.1f} >  {self.tc-self.t0:.1f}]')

        # Determine low pass filter dependent on ndt and corresponding nyquist
        # frequency fny or fcutoff = 1/(2*dt)
        # Reduce
        self.cutoff = 1.0/(2.0*self.ndt)

        # Create new STF using the
        self.t, self.stf = create_stf(
            self.t0, self.tc, self.nstep, self.dt, self.hdur, self.cutoff,
            gaussian=True, lpfilter=self.lpfilter)

        if self.forward_test:
            self.t_forward, self.stf_forward = create_stf(
                self.t0, self.tc, self.nstep, self.dt, self.hdur, self.cutoff,
                gaussian=True, lpfilter=self.lpfilter)

    def write_STF(self):

        # Do nothing if ndt is unchanged.
        if self.subsample is False:
            return

        if self.ndt_requested is None:
            logger.debug(
                '    .write_STF was called but no new time step was requested.')
            return

        self.logger.debug('Writing Source Time Functions ...')

        for _, _compdict in self.compdict.items():
            # STF file write
            stf_file = os.path.join(
                _compdict['dir'], 'DATA', 'stf')

            # Header for Source time function
            header = ''
            header += 'Source Time function\n'
            header += f'T0: {self.t0:f}\n'
            header += f'TC: {self.tc:f}\n'
            header += f'DT: {self.dt:f}\n'
            header += f'NT: {self.nstep:d}\n'
            header += 'format :  stf'

            # Concatenate
            F = np.ones_like(self.t) * self.force_factor
            # X = np.vstack((self.t, self.stf, F)).T

            # Write STF to file using numpy
            np.savetxt(stf_file, self.stf, fmt='% 30.15f',
                       header='', comments=' #')

        if self.forward_test:

            # Header
            header = ''
            header += 'Source Time function\n'
            header += f'T0: {self.t0:f}\n'
            header += f'TC: {self.tc:f}\n'
            header += f'DT: {self.dt:f}\n'
            header += f'NT: {self.nstep:d}\n'
            header += 'format :  stf'

            # Concatenate
            M0 = np.ones_like(self.t) * self.cmt.M0
            X = np.vstack((self.t, self.stf, M0)).T

            # Concatenate
            stf_forward_file = os.path.join(
                self.specfemdir_forward, 'DATA', 'stf')

            # Save STF
            if self.use_forward_stf:
                np.savetxt(stf_forward_file, self.stf_forward, fmt='%30.15f',
                           header="", comments=' #')

    def read_GF_LOCATIONS(self):
        """GF LOCATIONS must have the format """
        # Open GF locations file for each compenent
        # with open(self.target_file, 'w') as f:

        locmat = np.loadtxt(self.target_file)

        # Assign latitude
        if len(locmat.shape) > 1:
            self.target_latitude = locmat[:, 0]
            self.target_longitude = locmat[:, 1]
            self.target_depth = locmat[:, 2]
        else:
            self.target_latitude = locmat[0]
            self.target_longitude = locmat[1]
            self.target_depth = locmat[2]

        # print(f"{'Lat':<20}{'Lon':<20}{'Dep':<20}")
        # print(60 * "-")
        # for i in range(len(self.target_latitude)):
        #     print(
        #         f"{self.target_latitude[i]:<20f}"
        #         f"{self.target_longitude[i]:<20f}"
        #         f"{self.target_depth[i]:<20f}"
        #     )
        # print(60 * "-")

    def write_GF_LOCATIONS(self):

        self.logger.debug('Writing GF_LOCATIONS ...')

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
                    f.write(f'{_lat:9.4f}   {_lon:9.4f}   {_dep:9.4f}\n')

        if self.forward_test:

            # GF_LOCATIONS file
            locations_file = os.path.join(
                self.specfemdir_forward, 'DATA', 'GF_LOCATIONS')

            # Open GF locations file for each compenent
            with open(locations_file, 'w') as f:

                lat, lon, dep = (
                    self.station_latitude,
                    self.station_longitude,
                    self.station_burial)

                f.write(f'{lat:7.4f}   {lon:7.4f}   {dep:7.4f}')

    def write_FORCES(self):

        self.logger.debug('Writing FORCESOLUTIONS ...')

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

            hdur = 0.0
            time_shift = 0.0

            # Create force and
            force = FORCESOLUTION(
                time_shift=time_shift, hdur=hdur,
                latitude=self.station_latitude,
                longitude=self.station_longitude,
                depth=self.station_burial, stf=2,
                forcefactor=self.force_factor,
                vector_E=E, vector_N=N, vector_Z_UP=Z, force_no=1)

            # write it to subdir
            force.write(force_file)

    def write_STATIONS(self):
        """Only for forward test. Otherwise it doesn't matter."""

        self.logger.debug('Writing STATIONS ...')

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
                       self.station_latitude, self.station_longitude, 0.0,
                       self.station_burial*1.0e3)
                )

        if self.forward_test:
            # write stations file with one line
            with open(self.stations_file_forward, 'w') as f:
                # STATION    NETWORK   LATITUDE  LONGITUDE  ELEVATION    BURIAL
                f.write(
                    "%-9s %5s %15.4f %12.4f %10.1f %6.1f\n"
                    % (self.station, self.network,
                       self.station_latitude, self.station_longitude, 0.0,
                       self.station_burial*1.0e3)
                )

                f.write(
                    "%-9s %5s %15.4f %12.4f %10.1f %6.1f\n"
                    % ('SRC', 'EQ', self.cmt.latitude, self.cmt.longitude, 0.0,
                       self.cmt.depth*1.0e3)
                )

    def write_CMT(self):
        """Only for forward test. Otherwise it doesn't matter."""

        if self.forward_test:
            self.logger.debug('Writing CMTSOLUTIONS ...')
            self.cmt.write(self.CMTSOLUTION_file)

    def write_Par_file(self):

        #
        self.logger.debug("Writing Par_file's")

        # modify Reciprocal Par_file
        pardict = deepcopy(self.pardict)

        pardict['SAVE_GREEN_FUNCTIONS'] = True
        pardict['USE_FORCE_POINT_SOURCE'] = True

        pardict['USE_BUFFER_ELEMENTS'] = self.use_element_buffer
        pardict['NUMBER_OF_BUFFER_ELEMENTS'] = self.element_buffer

        if 'NSTEP' in pardict:
            pardict.pop('NSTEP')
        if 'T0' in pardict:
            pardict.pop('T0')

        # IF runs have to be parallel
        if self.simultaneous_runs:
            pardict['NUMBER_OF_SIMULTANEOUS_RUNS'] = 3
            pardict['BROADCAST_SAME_MESH_AND_MODEL'] = True
        else:
            pardict['NUMBER_OF_SIMULTANEOUS_RUNS'] = 1
            pardict['BROADCAST_SAME_MESH_AND_MODEL'] = False

        if self.subsample:
            pardict['PRINT_SOURCE_TIME_FUNCTION'] = False
            if (self.ndt_requested is not None) and (self.station is not None):

                pardict['NSTEP'] = self.nstep
                pardict['T0'] = self.t0
                pardict['NTSTEP_BETWEEN_FRAMES'] = self.xth_sample
        else:
            pardict['PRINT_SOURCE_TIME_FUNCTION'] = True
            pardict['NTSTEP_BETWEEN_FRAMES'] = 1
            pardict['RECORD_LENGTH_IN_MINUTES'] = self.duration_in_min

        if self.station is not None:
            # Write Par_file for each separate component directory
            for _comp, _compdict in self.compdict.items():

                # Don't write first Par_file since it is in the
                # Main directory
                par_file = os.path.join(_compdict['dir'], 'DATA', 'Par_file')

                # Write Par file for each sub dir
                utils.write_par_file(pardict, par_file)

        else:
            # Par_file location
            par_file = os.path.join(self.specfemdir, 'DATA', 'Par_file')

            # Write separate Par file for main specfem directory
            utils.write_par_file(pardict, par_file)

        # Write even more separate Par file for a forward test directory
        if self.forward_test:

            # modify Reciprocal Par_file
            pardict = deepcopy(self.pardict)
            pardict['SAVE_GREEN_FUNCTIONS'] = False
            pardict['USE_FORCE_POINT_SOURCE'] = False

            pardict['NUMBER_OF_SIMULTANEOUS_RUNS'] = 1
            pardict['BROADCAST_SAME_MESH_AND_MODEL'] = False

            pardict['USE_BUFFER_ELEMENTS'] = False
            pardict['NUMBER_OF_BUFFER_ELEMENTS'] = 0

            # For now so that specfem makes it's
            if 'NSTEP' in pardict:
                pardict.pop('NSTEP')
            if 'T0' in pardict:
                pardict.pop('T0')

            # Force STF print
            if self.subsample:
                pardict['PRINT_SOURCE_TIME_FUNCTION'] = False
            else:
                pardict['PRINT_SOURCE_TIME_FUNCTION'] = True

            if self.subsample:
                if (self.ndt_requested is not None) and (self.station is not None):
                    pardict['NTSTEP_BETWEEN_FRAMES'] = self.xth_sample

                if self.use_forward_stf:
                    pardict['NSTEP'] = self.nstep
                    pardict['T0'] = self.t0
                else:
                    pardict['RECORD_LENGTH_IN_MINUTES'] = self.duration_in_min

            else:
                pardict['NTSTEP_BETWEEN_FRAMES'] = 1
                pardict['RECORD_LENGTH_IN_MINUTES'] = self.duration_in_min

            # Write Par_file

            utils.write_par_file(pardict, self.par_file_forward)

    def __str__(self) -> str:

        rstr = "\n"
        rstr += "Reciprocal Simulation Setup:\n"
        rstr += "-----------------------------\n"
        rstr += f"Specfem basedir:{self.specfemdir:.>56}\n"
        if self.station is not None:
            rstr += f"E:{self.compdict['E']['dir']:.>70}\n"
            rstr += f"N:{self.compdict['N']['dir']:.>70}\n"
            rstr += f"Z:{self.compdict['Z']['dir']:.>70}\n"
            rstr += "\n"
        rstr += f"Force Factor:{self.force_factor:.>59.4g}\n"
        rstr += f"T0:{self.t0:.>69.4f}\n"
        rstr += f"TC:{self.tc:.>69.4f}\n"
        if self.dt is not None:
            rstr += f"DT:{self.dt:.>69.4f}\n"

        if self.subsample:

            if self.ndt_requested is not None:
                rstr += f"NDT requested:{self.ndt_requested:.>58.4f}\n"

            if self.ndt is not None:
                rstr += f"NDT:{self.ndt:.>68.4f}\n"

            if self.xth_sample is not None:
                rstr += f"X_TH_SAMPLE:{self.xth_sample:.>60d}\n"

            if self.hdur is not None:
                rstr += f"HDUR:{self.hdur:.>67.4f}\n"

            if self.cutoff is not None:
                rstr += f"CUTOFF:{self.cutoff:.>65.4f}\n"

        if self.nstep is not None:
            rstr += f"NT:{self.nstep:.>69d}\n"
            rstr += f"T:{self.nstep*self.dt/60.0:>70.4f} min\n"
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
