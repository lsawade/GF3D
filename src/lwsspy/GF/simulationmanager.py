import os
import typing as tp
import numpy as np
from .source import FORCESOLUTION, CMTSOLUTION
from .utils import get_par_file, write_par_file, get_dt_from_mesh_header


class SimMgr:

    specfemdir: str

    forward_test: bool = False

    # Read Par_file
    pardict: tp.OrderedDict

    def __init__(
            self, specfemdir,
            station_latitude: float,
            station_longitude: float,
            station_burial_in_m: float,
            target_latitude: float | np.ndarray | tp.Iterable,
            target_longitude: float | np.ndarray | tp.Iterable,
            target_depth_in_km: float | np.ndarray | tp.Iterable,
            t0: float = 0.0, tc: float = 0.0,
            duration_in_min: float = 20.0,
            nstep: None | int = None,
            ndt: None | float = None,
            forward_test: bool = False,
            broadcast_mesh_model: bool = False,
            simultaneous_runs: bool = False) -> None:
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
        self.station_latitude = station_latitude
        self.station_longitude = station_longitude
        self.station_burial_in_m = station_burial_in_m
        self.station_longitude = station_longitude
        self.target_latitude = target_latitude
        self.target_longitude = target_longitude
        self.target_burial_in_km = target_depth_in_km
        self.target_longitude = target_longitude

        self.t0 = t0
        self.tc = tc
        self.duration_in_min = duration_in_min
        self.nstep = nstep
        self.ndt = ndt
        self.forward_test = forward_test

        # Submission specific options
        self.simultaneous_runs = simultaneous_runs
        self.broadcast_mesh_model = broadcast_mesh_model

        # Run setup
        self.setup()

    def check_inputs(self):
        pass

    def setup(self):
        """Setting up the directory structure for specfem."""
        # Mostly setting up paths
        self.specfemdir = os.path.abspath(self.specfemdir)

        # Source Time Function file
        self.stf_file = os.path.join(self.specfemdir, 'DATA', 'stf')

        # Constants file has to be updated for reciprocal simulations
        self.constants_file = os.path.join(
            self.specfemdir, 'setup', 'constants.h.in')

        # Par_file of course has to be updated as well.
        self.par_file = os.path.join(
            self.specfemdir, 'DATA', 'Par_file')

        # Read Par_file into dictionary from (include comments for posterity)
        self.pardict = get_par_file(self.par_file, savecomments=True)

        # Get simulation sampling rate and min period
        self.dt, self.T = get_dt_from_mesh_header(self.specfemdir)

        # Print statement showing the setup
        self.compdict = dict(
            N=dict(dir=os.path.join(self.specfemdir, 'run0001')),
            E=dict(dir=os.path.join(self.specfemdir, 'run0002')),
            Z_UP=dict(dir=os.path.join(self.specfemdir, 'run0003'))
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

        # Read Par_file into dictionary from (include comments for posterity)
        self.pardict_forward = get_par_file(self.par_file, savecomments=True)

    def update_rotation(self):
        pass

    def write_GF_LOCATIONS(self):
        pass

    def write_FORCES(self):
        pass

    def write_STATIONS(self):
        # For Forward test
        pass

    def write_STF(self):
        pass
