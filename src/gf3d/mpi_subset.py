import os
import logging
import h5py
import contextlib
import typing as tp
from copy import deepcopy
from obspy import Trace, Stream, Inventory
from obspy.core.trace import Stats
from scipy.spatial import KDTree
import numpy as np
from joblib import parallel_backend, Parallel, delayed
from obspy.core.util.attribdict import AttribDict
from .lagrange import gll_nodes, lagrange_any
from .source2xyz import source2xyz, rotate_mt
from .locate_point import locate_point
from .source import CMTSOLUTION
from .utils import timeshift, next_power_of_2
from .stf import create_stf
from .logger import logger
from .signal.filter import butter_band_two_pass_filter
from scipy import fft
from time import time
from multiprocessing import Process, Lock



class MPISubset(object):
    """
    This class expects that your DB list is completely consistent. Meaning all
    station databases have been created using the exact same GF_LOCATIONS file,
    contain the same amount of elements etc.

    The classes goals/functionality are the following:

    * Read the header info of the database.
    * Given a source location, find the K closest elements,
      where K defaults to 10 due to standard element adjacency.
    * Load strains of the selected elements for all stations and restructure
      ibool, xyz, so that iterative retreival is fast due to a much small
      process the kdtree is much smaller than the global one.
    * Then, for inversion purposes use .get_seismograms(cmt) method to return
      an obspy.Stream with traces from all stations.

    """

    # Main variables
    db: list[str] | str       # list of station data base files.

    lat: float                # in degrees

    lon: float                # in degrees

    depth: float              # in km

    headerfile: str           # path to first file in dblist

    header: dict              # Dictionary with same values for all station

    midpoints: np.ndarray     # all midpoints

    NGLL: np.array            # NGLL points of subset

    fullkdtree: KDTree        # KDTree to get elements

    ispec: np.ndarray         # subset of elements

    xyz: np.ndarray           # subset of the coordinates

    ibool: np.ndarray         # index array for locations

    kdtree: KDTree         # Kdtree for the subset of elements.

    components: list[str] = ['N', 'E', 'Z']  # components

    do_adjacency_search: bool = True  # Whether to check neighbouring elements

    xadj: np.ndarray | None = None

    adjacency: np.ndarray | None = None

    def __init__(self, subset) -> None:
        """Initializes the gfm manager"""

        import mpi4py.MPI as MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # List of station files
        self.db = subset
        self.subset = True
        self.headerfile = self.db

        # Load header
        self.load_bcast_header()


    def load_bcast_header(self):

        if self.rank == 0:

            self.header = dict()

            with h5py.File(self.headerfile, 'r') as db:

                if 'fortran' in db:
                    fortran = True
                else:
                    fortran = False

                self.header['NGLLX'] = db['NGLLX'][()]
                self.header['NGLLY'] = db['NGLLY'][()]
                self.header['NGLLZ'] = db['NGLLZ'][()]
                self.NGLL = self.header['NGLLX']
                self.networks = db['Networks'][:].astype('U13').tolist()
                self.stations = db['Stations'][:].astype('U13').tolist()
                self.latitudes = db['latitudes'][:]
                self.longitudes = db['longitudes'][:]
                self.burials = db['burials'][:]
                self.ibool = db['ibool'][:]
                self.xyz = db['xyz'][:]
                self.kdtree = KDTree(self.xyz[self.ibool[
                    self.NGLL//2,
                    self.NGLL//2,
                    self.NGLL//2, :], :])

                self.do_adjacency_search = db['do_adjacency_search'][()]

                if self.do_adjacency_search:
                    self.xadj = db['xadj'][:]
                    self.adjacency = db['adjacency'][:]

                # Header Variables
                self.header['dt'] = db['DT'][()]
                self.header['tc'] = db['TC'][()]
                self.header['nsteps'] = db['NSTEPS'][()]
                self.header['factor'] = db['FACTOR'][()]
                self.header['hdur'] = db['HDUR'][()]
                self.header['topography'] = db['TOPOGRAPHY'][()]
                self.header['ellipticity'] = db['ELLIPTICITY'][()]

                if self.header['topography']:
                    self.header['itopo'] = db['BATHY'][:]
                    self.header['nx_topo'] = db['NX_BATHY'][()]
                    self.header['ny_topo'] = db['NY_BATHY'][()]
                    self.header['res_topo'] = db['RESOLUTION_TOPO_FILE'][()]

                if self.header['ellipticity']:
                    self.header['rspl'] = db['rspl'][:]
                    self.header['ellipticity_spline'] = db['ellipticity_spline'][:]
                    self.header['ellipticity_spline2'] = db['ellipticity_spline2'][:]

        else:
            self.header = None
            self.NGLL = None
            self.networks = None
            self.stations =None
            self.latitudes =None
            self.longitudes = None
            self.burials = None
            self.ibool = None
            self.xyz = None
            self.kdtree = None
            self.do_adjacency_search = None
            self.xadj = None
            self.adjacency = None


        # Broadcast all header variables
        self.header = self.comm.bcast(self.header, root=0)
        self.NGLL = self.comm.bcast(self.NGLL, root=0)
        self.networks = self.comm.bcast(self.networks, root=0)
        self.stations = self.comm.bcast(self.stations, root=0)
        self.latitudes = self.comm.bcast(self.latitudes, root=0)
        self.longitudes = self.comm.bcast(self.longitudes, root=0)
        self.burials = self.comm.bcast(self.burials, root=0)
        self.ibool = self.comm.bcast(self.ibool, root=0)
        self.xyz = self.comm.bcast(self.xyz, root=0)
        self.kdtree = self.comm.bcast(self.kdtree, root=0)
        self.do_adjacency_search = self.comm.bcast(self.do_adjacency_search, root=0)

        if self.do_adjacency_search:
            self.xadj = self.comm.bcast(self.xadj , root=0)
            self.adjacency = self.comm.bcast(self.adjacency , root=0)

        if self.header['topography']:
            if self.rank != 0:
                self.header['itopo'] = None
                self.header['nx_topo'] = None
                self.header['ny_topo'] = None
                self.header['res_topo'] = None

            self.header['itopo'] = self.comm.bcast(self.header['itopo'], root=0)
            self.header['nx_topo'] = self.comm.bcast(self.header['nx_topo'], root=0)
            self.header['ny_topo'] = self.comm.bcast(self.header['ny_topo'], root=0)
            self.header['res_topo'] = self.comm.bcast(self.header['res_topo'], root=0)

        if self.header['ellipticity']:
            if self.rank != 0:
                self.header['rspl'] = None
                self.header['ellipticity_spline'] = None
                self.header['ellipticity_spline2'] = None

            self.header['rspl'] = self.comm.bcast(self.header['rspl'], root = 0)
            self.header['ellipticity_spline'] = self.comm.bcast(self.header['ellipticity_spline'], root = 0)
            self.header['ellipticity_spline2'] = self.comm.bcast(self.header['ellipticity_spline2'], root = 0)


    def get_seismograms(self, cmt: CMTSOLUTION) -> Stream:

        # Get moment tensor
        x_target, y_target, z_target, Mx = source2xyz(
            cmt.latitude, cmt.longitude, cmt.depth, M=cmt.tensor,
            topography=self.header['topography'],
            ellipticity=self.header['ellipticity'],
            ibathy_topo=self.header['itopo'],
            NX_BATHY=self.header['nx_topo'],
            NY_BATHY=self.header['ny_topo'],
            RESOLUTION_TOPO_FILE=self.header['res_topo'],
            rspl=self.header['rspl'],
            ellipicity_spline=self.header['ellipticity_spline'],
            ellipicity_spline2=self.header['ellipticity_spline2'],
        )

        if self.do_adjacency_search:
            logger.debug("DOING ADJACENCY SEARCH")

        ispec_selected, xi, eta, gamma, xix, xiy, xiz, etax, etay, etaz, gammax, gammay, gammaz, _, _, _, _ = locate_point(
            x_target, y_target, z_target, cmt.latitude, cmt.longitude,
            self.xyz[self.ibool[self.NGLL//2, self.NGLL//2, self.NGLL//2, :],
                     :], self.xyz[:, 0], self.xyz[:, 1],
            self.xyz[:, 2], self.ibool,
            xadj=self.xadj, adjacency=self.adjacency,
            POINT_CAN_BE_BURIED=True, kdtree=self.kdtree,
            do_adjacent_search=self.do_adjacency_search, NGLL=self.NGLL)

        # Get global indeces.
        iglob = self.ibool[:, :, :, ispec_selected]

        # Flattening because fancy indexing (multi-dimensional) only works with
        # 1D arrays since we are grabbing from a very multidimensional array
        # this is faster.
        iglobf = iglob.flatten()

        # Sorting because multidimensional array indexing only works with
        # ascending order
        sglobf = np.argsort(iglobf)

        # Resorting the flattened array
        rsglobf = np.argsort(sglobf)

        # Reshaping the flattened array order to the original iglob shape
        rsglob = rsglobf.reshape(iglob.shape)

        # Finally just a list of ascneding indeces to grap from the database
        indeces = np.arange(len(iglobf))


        if True:
            # Read from the HDF5 file in serial
            for _rank in range(self.size):

                if self.rank == _rank:
                    print(f'Rank {_rank} reading', flush=True)
                    with h5py.File(self.headerfile, 'r') as db:
                        # Get displacement for interpolation
                        displacement = db['displacement'][:, :, :, iglobf[sglobf], :]
                    print(f'Rank {_rank} done', flush=True)
                self.comm.Barrier()
        else:
            print(f'Rank {self.rank} reading', flush=True)
            with h5py.File(self.headerfile, 'r') as db:
                # Get displacement for interpolation
                displacement = db['displacement'][:, :, :, iglobf[sglobf], :]
            print(f'Rank {self.rank} done', flush=True)


        # Loaded displacement is in weird order so we have to redo it
        displacement = displacement[:, :, :, indeces[rsglob], :]

        # GLL points and weights (degree)
        npol = self.NGLL - 1
        xigll, _, _ = gll_nodes(npol)

        # Get lagrange values at specific GLL poins
        hxi, hpxi = lagrange_any(xi, xigll, npol)
        heta, hpeta = lagrange_any(eta, xigll, npol)
        hgamma, hpgamma = lagrange_any(gamma, xigll, npol)

        # Initialize epsilon array
        epsilon = np.zeros((len(self.stations), 3, 6, self.header['nsteps']))

        for k in range(self.NGLL):
            for j in range(self.NGLL):
                for i in range(self.NGLL):

                    hlagrange_xi = hpxi[i] * heta[j] * hgamma[k]
                    hlagrange_eta = hxi[i] * hpeta[j] * hgamma[k]
                    hlagrange_gamma = hxi[i] * heta[j] * hpgamma[k]
                    hlagrange_x = hlagrange_xi * xix + hlagrange_eta * etax + hlagrange_gamma * gammax
                    hlagrange_y = hlagrange_xi * xiy + hlagrange_eta * etay + hlagrange_gamma * gammay
                    hlagrange_z = hlagrange_xi * xiz + hlagrange_eta * etaz + hlagrange_gamma * gammaz

                    for _s in range(len(self.stations)):
                        for _c in range(3):
                            epsilon[_s, _c, 0, :] += (
                                displacement[_s, _c, 0, i, j, k, :] * hlagrange_x)
                            epsilon[_s, _c, 1, :] += (
                                displacement[_s, _c, 1, i, j, k, :] * hlagrange_y)
                            epsilon[_s, _c, 2, :] += (
                                displacement[_s, _c, 2, i, j, k, :] * hlagrange_z)
                            epsilon[_s, _c, 3, :] += 0.5 * (
                                displacement[_s, _c, 1, i,
                                             j, k, :] * hlagrange_x
                                + displacement[_s, _c, 0, i, j, k, :] * hlagrange_y)
                            epsilon[_s, _c, 4, :] += 0.5 * (
                                displacement[_s, _c, 2, i,
                                             j, k, :] * hlagrange_x
                                + displacement[_s, _c, 0, i, j, k, :] * hlagrange_z)
                            epsilon[_s, _c, 5, :] += 0.5 * (
                                displacement[_s, _c, 2, i,
                                             j, k, :] * hlagrange_y
                                + displacement[_s, _c, 1, i, j, k, :] * hlagrange_z)

        # Get lagrange values at specific GLL points
        M = Mx * np.array([1., 1., 1., 2., 2., 2.])

        # Since the database is the same for all
        seismograms = np.einsum('hijo,j->hio', epsilon[:, :, :, :], M)

        logger.debug(f"SEISUM: {np.sum(seismograms)}")

        # For following FFTs
        NP2 = next_power_of_2(2 * self.header['nsteps'])

        # This computes the differential half duration for the new STF from
        # the cmt half duration and the half duration of the database that the
        # database was computed with
        if (cmt.hdur / 1.628)**2 <= self.header['hdur']**2:
            hdur_diff = 0.000001
            logger.warn(
                f"Requested half duration smaller than what was simulated.\n"
                f"Half duration set to {hdur_diff}s to simulate a Heaviside function.")
            hdur_conv = np.sqrt(self.header['hdur']**2 - (cmt.hdur / 1.628)**2)

            logger.warn(f"Try convolving your seismogram with a Gaussian with "
                        f"{hdur_conv}s standard deviation.")

        else:
            hdur_diff = np.sqrt((cmt.hdur / 1.628)**2 - self.header['hdur']**2)

        # Heaviside STF to reproduce SPECFEM stf
        _, stf_r = create_stf(0, 200.0, self.header['nsteps'],
                              self.header['dt'], hdur_diff, cutoff=None, gaussian=False, lpfilter='butter')

        # Fourier Transform the STF
        STF_R = fft.fft(stf_r, n=NP2)

        # Compute correctional phase shift
        shift = -200.0 + cmt.time_shift
        phshift = np.exp(-1.0j*shift*np.fft.fftfreq(NP2,
                                                    self.header['dt'])*2*np.pi)

        logger.debug(f"Lengths: {self.header['nsteps']}, {NP2}")

        # Add traces to the
        traces = []
        for _h in range(len(self.stations)):
            for _i, comp in enumerate(['N', 'E', 'Z']):

                # Convolution with the STF and correctional timeshift
                # data = butter_band_two_pass_filter(
                #     seismograms[_h, _i, :], [0.001, 1/self.header['dt']/2.1], 1/self.header['dt'])

                data = np.real(
                    fft.ifft(fft.fft(seismograms[_h, _i, :], n=NP2) * STF_R * phshift
                             )[:self.header['nsteps']]) * self.header['dt']

                stats = Stats()
                stats.delta = self.header['dt']
                stats.network = self.networks[_h]
                stats.station = self.stations[_h]
                stats.location = 'S3'
                stats.latitude = self.latitudes[_h]
                stats.longitude = self.longitudes[_h]
                stats.coordinates = AttribDict(
                    latitude=self.latitudes[_h], longitude=self.longitudes[_h])
                stats.channel = f'MX{comp}'
                stats.starttime = cmt.origin_time - self.header['tc']
                stats.npts = self.header['nsteps']
                tr = Trace(data=data, header=stats)

                traces.append(tr)

        logger.debug('Outputting traces')
        return Stream(traces)

    def get_mt_frechet(self, cmt: CMTSOLUTION) -> tp.Dict[str, Stream]:

        # Get moment tensor
        x_target, y_target, z_target = source2xyz(
            cmt.latitude, cmt.longitude, cmt.depth, M=None,
            topography=self.header['topography'],
            ellipticity=self.header['ellipticity'],
            ibathy_topo=self.header['itopo'],
            NX_BATHY=self.header['nx_topo'],
            NY_BATHY=self.header['ny_topo'],
            RESOLUTION_TOPO_FILE=self.header['res_topo'],
            rspl=self.header['rspl'],
            ellipicity_spline=self.header['ellipticity_spline'],
            ellipicity_spline2=self.header['ellipticity_spline2'],
        )

        # Get rotated moment tensors
        pertdict = dict(
            Mrr=1e23, Mtt=1e23, Mpp=1e23,
            Mrt=1e23, Mrp=1e23, Mtp=1e23,
        )
        mx = []
        for _i, (_key, pert) in enumerate(pertdict.items()):
            m = np.zeros(6)
            m[_i] = pert
            mx.append(rotate_mt(cmt.latitude, cmt.longitude, m))

        if self.do_adjacency_search:
            logger.debug("DOING ADJACENCY SEARCH")

        ispec_selected, xi, eta, gamma, xix, xiy, xiz, etax, etay, etaz, gammax, gammay, gammaz, _, _, _, _ = locate_point(
            x_target, y_target, z_target, cmt.latitude, cmt.longitude,
            self.xyz[self.ibool[self.NGLL//2, self.NGLL//2, self.NGLL//2, :],
                     :], self.xyz[:, 0], self.xyz[:, 1],
            self.xyz[:, 2], self.ibool,
            xadj=self.xadj, adjacency=self.adjacency,
            POINT_CAN_BE_BURIED=True, kdtree=self.kdtree,
            do_adjacent_search=self.do_adjacency_search, NGLL=self.NGLL)

        # Get global indeces.
        iglob = self.ibool[:, :, :, ispec_selected]

        # Get displacement for interpolation
        displacement = self.displacement[:, :, :, iglob, :]

        # GLL points and weights (degree)
        npol = self.NGLL - 1
        xigll, _, _ = gll_nodes(npol)

        # Get lagrange values at specific GLL poins
        hxi, hpxi = lagrange_any(xi, xigll, npol)
        heta, hpeta = lagrange_any(eta, xigll, npol)
        hgamma, hpgamma = lagrange_any(gamma, xigll, npol)

        # Initialize epsilon array
        epsilon = np.zeros((len(self.stations), 3, 6, self.header['nsteps']))

        for k in range(self.NGLL):
            for j in range(self.NGLL):
                for i in range(self.NGLL):

                    hlagrange_xi = hpxi[i] * heta[j] * hgamma[k]
                    hlagrange_eta = hxi[i] * hpeta[j] * hgamma[k]
                    hlagrange_gamma = hxi[i] * heta[j] * hpgamma[k]
                    hlagrange_x = hlagrange_xi * xix + hlagrange_eta * etax + hlagrange_gamma * gammax
                    hlagrange_y = hlagrange_xi * xiy + hlagrange_eta * etay + hlagrange_gamma * gammay
                    hlagrange_z = hlagrange_xi * xiz + hlagrange_eta * etaz + hlagrange_gamma * gammaz

                    for _s in range(len(self.stations)):
                        for _c in range(3):
                            epsilon[_s, _c, 0, :] += (
                                displacement[_s, _c, 0, i, j, k, :] * hlagrange_x)
                            epsilon[_s, _c, 1, :] += (
                                displacement[_s, _c, 1, i, j, k, :] * hlagrange_y)
                            epsilon[_s, _c, 2, :] += (
                                displacement[_s, _c, 2, i, j, k, :] * hlagrange_z)
                            epsilon[_s, _c, 3, :] += 0.5 * (
                                displacement[_s, _c, 1, i,
                                             j, k, :] * hlagrange_x
                                + displacement[_s, _c, 0, i, j, k, :] * hlagrange_y)
                            epsilon[_s, _c, 4, :] += 0.5 * (
                                displacement[_s, _c, 2, i,
                                             j, k, :] * hlagrange_x
                                + displacement[_s, _c, 0, i, j, k, :] * hlagrange_z)
                            epsilon[_s, _c, 5, :] += 0.5 * (
                                displacement[_s, _c, 2, i,
                                             j, k, :] * hlagrange_y
                                + displacement[_s, _c, 1, i, j, k, :] * hlagrange_z)

        # For following FFTs
        NP2 = next_power_of_2(2 * self.header['nsteps'])

        # This computes the differential half duration for the new STF from
        # the cmt half duration and the half duration of the database that the
        # database was computed with
        if (cmt.hdur / 1.628)**2 <= self.header['hdur']**2:
            hdur_diff = 0.000001
            logger.warn(
                f"Requested half duration smaller than what was simulated.\n"
                f"Half duration set to {hdur_diff}s to simulate a Heaviside function.")
        else:
            hdur_diff = np.sqrt((cmt.hdur / 1.628)**2 - self.header['hdur']**2)

        # Heaviside STF to reproduce SPECFEM stf
        _, stf_r = create_stf(0, 400.0, self.header['nsteps'],
                              self.header['dt'], hdur_diff, cutoff=None, gaussian=False, lpfilter='butter')

        STF_R = fft.fft(stf_r, n=NP2)

        shift = -400.0 + cmt.time_shift
        phshift = np.exp(-1.0j*shift*np.fft.fftfreq(NP2,
                                                    self.header['dt'])*2*np.pi)

        logger.debug(f"Lengths: {self.header['nsteps']}, {NP2}")

        # Add traces to the
        compdict = dict()
        for _j, mtc in enumerate(['Mrr', 'Mtt', 'Mpp', 'Mrt', 'Mrp', 'Mtp']):

            # Get lagrange values at specific GLL points
            M = mx[_j] * np.array([1., 1., 1., 2., 2., 2.])

            # Since the database is the same for all
            seismograms = np.einsum('hijo,j->hio', epsilon[:, :, :, :], M)

            logger.debug(f"SEISUM: {np.sum(seismograms)}")

            traces = []

            for _h in range(len(self.stations)):
                for _i, comp in enumerate(['N', 'E', 'Z']):

                    data = np.real(
                        fft.ifft(
                            STF_R * fft.fft(seismograms[_h, _i, :], n=NP2)
                            * phshift
                        ))[:self.header['nsteps']] * self.header['dt']/1e23

                    stats = Stats()
                    stats.delta = self.header['dt']
                    stats.network = self.networks[_h]
                    stats.station = self.stations[_h]
                    stats.location = 'S3'
                    stats.latitude = self.latitudes[_h]
                    stats.longitude = self.longitudes[_h]
                    stats.coordinates = AttribDict(
                        latitude=self.latitudes[_h], longitude=self.longitudes[_h])
                    stats.channel = f'MX{comp}'
                    stats.starttime = cmt.origin_time - self.header['tc']
                    stats.npts = self.header['nsteps']
                    tr = Trace(data=data, header=stats)

                    traces.append(tr)

            # Add traces to dictionary
            compdict[mtc] = Stream(traces)

        logger.debug('Outputting traces')

        return compdict

    def get_mt_frechet_station(
            self, cmt: CMTSOLUTION,
            network: str, station: str) -> tp.Dict[str, Stream]:

        idx = None
        for _i, (net, sta) in enumerate(zip(self.networks, self.stations)):
            if net == network and sta == station:
                idx = _i
                break

        if idx is None:
            raise ValueError(
                f'Station {network}.{station} not found in database.')

        # Get moment tensor
        x_target, y_target, z_target = source2xyz(
            cmt.latitude, cmt.longitude, cmt.depth, M=None,
            topography=self.header['topography'],
            ellipticity=self.header['ellipticity'],
            ibathy_topo=self.header['itopo'],
            NX_BATHY=self.header['nx_topo'],
            NY_BATHY=self.header['ny_topo'],
            RESOLUTION_TOPO_FILE=self.header['res_topo'],
            rspl=self.header['rspl'],
            ellipicity_spline=self.header['ellipticity_spline'],
            ellipicity_spline2=self.header['ellipticity_spline2'],
        )

        # Get rotated moment tensors
        pertdict = dict(
            Mrr=1e23, Mtt=1e23, Mpp=1e23,
            Mrt=1e23, Mrp=1e23, Mtp=1e23,
        )
        mx = []
        for _i, (_key, pert) in enumerate(pertdict.items()):
            m = np.zeros(6)
            m[_i] = pert
            mx.append(rotate_mt(cmt.latitude, cmt.longitude, m))

        if self.do_adjacency_search:
            logger.debug("DOING ADJACENCY SEARCH")

        ispec_selected, xi, eta, gamma, xix, xiy, xiz, etax, etay, etaz, gammax, gammay, gammaz, _, _, _, _ = locate_point(
            x_target, y_target, z_target, cmt.latitude, cmt.longitude,
            self.xyz[self.ibool[self.NGLL//2, self.NGLL//2, self.NGLL//2, :],
                     :], self.xyz[:, 0], self.xyz[:, 1],
            self.xyz[:, 2], self.ibool,
            xadj=self.xadj, adjacency=self.adjacency,
            POINT_CAN_BE_BURIED=True, kdtree=self.kdtree,
            do_adjacent_search=self.do_adjacency_search, NGLL=self.NGLL)

        # Get global indeces.
        iglob = self.ibool[:, :, :, ispec_selected]

        # Get displacement for interpolation
        displacement = self.displacement[:, :, :, iglob, :]

        # GLL points and weights (degree)
        npol = self.NGLL - 1
        xigll, _, _ = gll_nodes(npol)

        # Get lagrange values at specific GLL poins
        hxi, hpxi = lagrange_any(xi, xigll, npol)
        heta, hpeta = lagrange_any(eta, xigll, npol)
        hgamma, hpgamma = lagrange_any(gamma, xigll, npol)

        # Initialize epsilon array
        epsilon = np.zeros((3, 6, self.header['nsteps']))

        for k in range(self.NGLL):
            for j in range(self.NGLL):
                for i in range(self.NGLL):

                    hlagrange_xi = hpxi[i] * heta[j] * hgamma[k]
                    hlagrange_eta = hxi[i] * hpeta[j] * hgamma[k]
                    hlagrange_gamma = hxi[i] * heta[j] * hpgamma[k]
                    hlagrange_x = hlagrange_xi * xix + hlagrange_eta * etax + hlagrange_gamma * gammax
                    hlagrange_y = hlagrange_xi * xiy + hlagrange_eta * etay + hlagrange_gamma * gammay
                    hlagrange_z = hlagrange_xi * xiz + hlagrange_eta * etaz + hlagrange_gamma * gammaz

                    for _c in range(3):
                        epsilon[_c, 0, :] += (
                            displacement[idx, _c, 0, i, j, k, :] * hlagrange_x)
                        epsilon[_c, 1, :] += (
                            displacement[idx, _c, 1, i, j, k, :] * hlagrange_y)
                        epsilon[_c, 2, :] += (
                            displacement[idx, _c, 2, i, j, k, :] * hlagrange_z)
                        epsilon[_c, 3, :] += 0.5 * (
                            displacement[idx, _c, 1, i,
                                         j, k, :] * hlagrange_x
                            + displacement[idx, _c, 0, i, j, k, :] * hlagrange_y)
                        epsilon[_c, 4, :] += 0.5 * (
                            displacement[idx, _c, 2, i,
                                         j, k, :] * hlagrange_x
                            + displacement[idx, _c, 0, i, j, k, :] * hlagrange_z)
                        epsilon[_c, 5, :] += 0.5 * (
                            displacement[idx, _c, 2, i,
                                         j, k, :] * hlagrange_y
                            + displacement[idx, _c, 1, i, j, k, :] * hlagrange_z)

        # For following FFTs
        NP2 = next_power_of_2(2 * self.header['nsteps'])

        # This computes the differential half duration for the new STF from
        # the cmt half duration and the half duration of the database that the
        # database was computed with
        if (cmt.hdur / 1.628)**2 <= self.header['hdur']**2:
            hdur_diff = 0.000001
            logger.warn(
                f"Requested half duration smaller than what was simulated.\n"
                f"Half duration set to {hdur_diff}s to simulate a Heaviside function.")
        else:
            hdur_diff = np.sqrt((cmt.hdur / 1.628)**2 - self.header['hdur']**2)

        # Heaviside STF to reproduce SPECFEM stf
        _, stf_r = create_stf(0, 400.0, self.header['nsteps'],
                              self.header['dt'], hdur_diff, cutoff=None,
                              gaussian=False, lpfilter='butter')

        STF_R = fft.fft(stf_r, n=NP2)

        shift = -400.0 + cmt.time_shift
        phshift = np.exp(-1.0j*shift*np.fft.fftfreq(NP2,
                                                    self.header['dt'])*2*np.pi)

        logger.debug(f"Lengths: {self.header['nsteps']}, {NP2}")

        # Add traces to the
        compdict = dict()
        for _j, mtc in enumerate(['Mrr', 'Mtt', 'Mpp', 'Mrt', 'Mrp', 'Mtp']):

            # Get lagrange values at specific GLL points
            M = mx[_j] * np.array([1., 1., 1., 2., 2., 2.])

            # Since the database is the same for all
            seismograms = np.einsum('ijo,j->io', epsilon[:, :, :], M)

            logger.debug(f"SEISUM: {np.sum(seismograms)}")

            traces = []

            for _i, comp in enumerate(['N', 'E', 'Z']):

                data = np.real(
                    fft.ifft(
                        STF_R * fft.fft(seismograms[_i, :], n=NP2)
                        * phshift
                    ))[:self.header['nsteps']] * self.header['dt']/1e23

                stats = Stats()
                stats.delta = self.header['dt']
                stats.network = self.networks[idx]
                stats.station = self.stations[idx]
                stats.location = 'S3'
                stats.latitude = self.latitudes[idx]
                stats.longitude = self.longitudes[idx]
                stats.coordinates = AttribDict(
                    latitude=self.latitudes[idx], longitude=self.longitudes[idx])
                stats.channel = f'MX{comp}'
                stats.starttime = cmt.origin_time - self.header['tc']
                print(stats.starttime)
                stats.npts = self.header['nsteps']
                tr = Trace(data=data, header=stats)

                traces.append(tr)

            # Add traces to dictionary
            compdict[mtc] = Stream(traces)

        logger.debug('Outputting traces')

        return compdict

    def get_frechet(self, cmt: CMTSOLUTION, rtype=3) -> tp.Dict[str, Stream]:
        """Computes centered finite difference using 10m perturbations"""

        mtpar = ['Mrr', 'Mtt', 'Mpp', 'Mrt', 'Mrp', 'Mtp']

        # Define base frechet derivative dictionary
        pertdict = dict(
            Mrr=1e23,
            Mtt=1e23,
            Mpp=1e23,
            Mrt=1e23,
            Mrp=1e23,
            Mtp=1e23,
        )

        # Add centroid location
        if rtype >= 2:
            pertdict.update(
                latitude=0.0001,
                longitude=0.0001,
                depth=0.01,
                time_shift=-1.0,
            )

        # Add frechet derivative for half duration
        if rtype >= 3:
            pertdict.update(
                hdur=0.001
            )

        frechets = dict()
        mt_computed = False
        for par, pert in pertdict.items():

            if par in mtpar:
                if mt_computed is False:
                    compdict = self.get_mt_frechet(cmt)
                    mt_computed = True

                drp = compdict[par]

            elif par == 'time_shift':

                drp = self.get_seismograms(cmt)
                drp.differentiate()
                for tr in drp:
                    tr.data *= -1

            else:
                # create cmt copies
                pcmt = deepcopy(cmt)
                mcmt = deepcopy(cmt)

                # Get model values
                m = getattr(cmt, par)

                # Set vals
                setattr(pcmt, par, m + pert)
                setattr(mcmt, par, m - pert)

                # Get reciprocal synthetics
                prp = self.get_seismograms(pcmt)
                mrp = self.get_seismograms(mcmt)

                #
                for ptr, mpr in zip(prp, mrp):
                    ptr.data -= mpr.data
                    ptr.data /= 2 * pert

                # Reassign to match
                drp = prp

            frechets[par] = drp

        return frechets

    def write_subset(self, outfile, duration=None, fortran=False):
        """Given the files in the database, get a set of strains and write it
        into to a single file in a single epsilon array.
        Also write a list of stations and normal header info required for
        source location. This very same GFManager would be used to read the.
        Where """

        with h5py.File(outfile, 'w') as db:

            # Store fixed length strings for fortran
            if fortran:
                # NOTE THAT THE STRING LENGTH HERE IS HARD-CODED in the
                # fortran extraction codes according to the specfem3dglobe
                # definitions.
                db.create_dataset('Networks', data=self.networks, dtype="S8")
                db.create_dataset('Stations', data=self.stations, dtype="S32")
            else:
                # For python subsets this does not matter.
                db.create_dataset('Networks', data=self.networks)
                db.create_dataset('Stations', data=self.stations)

            db.create_dataset('latitudes', data=self.latitudes)
            db.create_dataset('longitudes', data=self.longitudes)
            db.create_dataset('burials', data=self.burials)

            if fortran:
                db.create_dataset('fortran', data=1)
                db.create_dataset(
                    'ibool', data=self.ibool.transpose((3, 2, 1, 0)))
                db.create_dataset('xyz', data=self.xyz.transpose((1, 0)))
            else:
                db.create_dataset('ibool', data=self.ibool)
                db.create_dataset('xyz', data=self.xyz)

            db.create_dataset('do_adjacency_search',
                              data=self.do_adjacency_search)

            if self.do_adjacency_search:
                db.create_dataset('xadj', data=self.xadj)
                db.create_dataset('adjacency', data=self.adjacency)

            # Header Variables
            if duration is not None:
                nsteps = int(
                    np.ceil((self.header['tc'] + duration)/self.header['dt']))

                # Can only store as many steps as we have...
                if nsteps > self.header['nsteps']:
                    nsteps = self.header['nsteps']
            else:
                nsteps = self.header['nsteps']

            db.create_dataset('NSTEPS', data=nsteps)
            db.create_dataset('DT', data=self.header['dt'])
            db.create_dataset('TC', data=self.header['tc'])
            db.create_dataset('FACTOR', data=self.header['factor'])
            db.create_dataset('HDUR', data=self.header['hdur'])
            db.create_dataset('TOPOGRAPHY', data=self.header['topography'])
            db.create_dataset('ELLIPTICITY', data=self.header['ellipticity'])

            if self.header['topography']:
                db.create_dataset('NX_BATHY', data=self.header['nx_topo'])
                db.create_dataset('NY_BATHY', data=self.header['ny_topo'])
                db.create_dataset('RESOLUTION_TOPO_FILE',
                                  data=self.header['res_topo'])

                if fortran:
                    db.create_dataset(
                        'BATHY', data=self.header['itopo'].transpose(1, 0))
                else:
                    db.create_dataset('BATHY', data=self.header['itopo'])

            if self.header['ellipticity']:
                db.create_dataset('rspl', data=self.header['rspl'])
                db.create_dataset('ellipticity_spline',
                                  data=self.header['ellipticity_spline'])
                db.create_dataset('ellipticity_spline2',
                                  data=self.header['ellipticity_spline2'])

            db.create_dataset('NGLOB', data=self.header['NGLOB'])
            db.create_dataset('NGLLX', data=self.NGLL)
            db.create_dataset('NGLLY', data=self.NGLL)
            db.create_dataset('NGLLZ', data=self.NGLL)

            # Use duration keyword to set the number of samples to save to file:
            print('shape', self.displacement.shape)
            print('nsteps', nsteps)

            if fortran:
                logger.info('Writing Fortran order displacement array')
                db.create_dataset(
                    'displacement',
                    data=self.displacement[:, :, :, :, :nsteps].transpose((4, 3, 2, 1, 0)))
            else:
                logger.info('Writing C order displacement array')
                db.create_dataset(
                    'displacement', data=self.displacement[:, :, :, :, :nsteps])  # ,
            # shuffle=True, compression='lzf')

            logger.info('Done.')

    def load(self):
        """Given the files in the database, get a set of strains and write it
        into to a single file in a single epsilon array.
        Also write a list of stations and normal header info required for
        source location. This very same GFManager would be used to read the.
        Where """

        self.header = dict()

        with h5py.File(self.headerfile, 'r') as db:

            if 'fortran' in db:
                fortran = True
            else:
                fortran = False
            self.header['NGLLX'] = db['NGLLX'][()]
            self.header['NGLLY'] = db['NGLLY'][()]
            self.header['NGLLZ'] = db['NGLLZ'][()]
            self.NGLL = self.header['NGLLX']
            self.networks = db['Networks'][:].astype('U13').tolist()
            self.stations = db['Stations'][:].astype('U13').tolist()
            self.latitudes = db['latitudes'][:]
            self.longitudes = db['longitudes'][:]
            self.burials = db['burials'][:]
            self.ibool = db['ibool'][:]
            self.xyz = db['xyz'][:]
            self.kdtree = KDTree(self.xyz[self.ibool[
                self.NGLL//2,
                self.NGLL//2,
                self.NGLL//2, :], :])

            self.do_adjacency_search = db['do_adjacency_search'][()]

            if self.do_adjacency_search:
                self.xadj = db['xadj'][:]
                self.adjacency = db['adjacency'][:]

            # Header Variables
            self.header['dt'] = db['DT'][()]
            self.header['tc'] = db['TC'][()]
            self.header['nsteps'] = db['NSTEPS'][()]
            self.header['factor'] = db['FACTOR'][()]
            self.header['hdur'] = db['HDUR'][()]
            self.header['topography'] = db['TOPOGRAPHY'][()]
            self.header['ellipticity'] = db['ELLIPTICITY'][()]

            if self.header['topography']:
                self.header['itopo'] = db['BATHY'][:]
                self.header['nx_topo'] = db['NX_BATHY'][()]
                self.header['ny_topo'] = db['NY_BATHY'][()]
                self.header['res_topo'] = db['RESOLUTION_TOPO_FILE'][()]

            if self.header['ellipticity']:
                self.header['rspl'] = db['rspl'][:]
                self.header['ellipticity_spline'] = db['ellipticity_spline'][:]
                self.header['ellipticity_spline2'] = db['ellipticity_spline2'][:]

            self.displacement = db['displacement'][:]

            if fortran:
                self.displacement = self.displacement.transpose(
                    (4, 3, 2, 1, 0))
                self.header['itopo'] = self.header['itopo'].transpose((1, 0))
                self.xyz = self.xyz.transpose((1, 0))
                self.ibool = self.ibool.transpose((3, 2, 1, 0))

    def load_subset_header_only(self):
        """Given the files in the database, get a set of strains and write it
        into to a single file in a single epsilon array.
        Also write a list of stations and normal header info required for
        source location. This very same GFManager would be used to read the.
        Where """

        self.header = dict()

        with h5py.File(self.headerfile, 'r') as db:

            if 'fortran' in db:
                fortran = True
            else:
                fortran = False
            self.header['NGLLX'] = db['NGLLX'][()]
            self.header['NGLLY'] = db['NGLLY'][()]
            self.header['NGLLZ'] = db['NGLLZ'][()]
            self.networks = db['Networks'][:].astype('U13').tolist()
            self.stations = db['Stations'][:].astype('U13').tolist()
            self.latitudes = db['latitudes'][:]
            self.longitudes = db['longitudes'][:]
            self.burials = db['burials'][:]
            self.header['NSTAT'] = len(self.networks)

            self.header['do_adjacency_search'] = db['do_adjacency_search'][()]

            # Header Variables
            self.header['dt'] = db['DT'][()]
            self.header['tc'] = db['TC'][()]
            self.header['nsteps'] = db['NSTEPS'][()]
            self.header['factor'] = db['FACTOR'][()]
            self.header['hdur'] = db['HDUR'][()]
            self.header['topography'] = db['TOPOGRAPHY'][()]
            self.header['ellipticity'] = db['ELLIPTICITY'][()]

            if self.header['topography']:
                self.header['nx_topo'] = db['NX_BATHY'][()]
                self.header['ny_topo'] = db['NY_BATHY'][()]
                self.header['res_topo'] = db['RESOLUTION_TOPO_FILE'][()]

            if self.header['ellipticity']:
                self.header['nspl'] = len(db['rspl'][:])

    def get_inventory(self) -> Inventory:
        """Generates inventory from stations in subset

        Returns
        -------
        obspy.Inventory
            Inventory containing the networks and stations and locations
            of the stations. Note that "elevation" is the burial used for the
            station and is given as negative down.

        Raises
        ------
        ValueError
            if trying to access from database.
        """

        if self.subset is False:
            raise ValueError('Must be subset. Not implemented for database')

        from obspy.core.inventory.inventory import Inventory
        from obspy.core.inventory.station import Station
        from obspy.core.inventory.network import Network

        # First get unique networks
        unique_networks = set(self.networks)

        # Init networks list
        networks = []

        for net in unique_networks:

            # Get all stations from the same network
            idxs = [_i for _i, _net in enumerate(self.networks) if _net == net]

            stations = []
            for _i in idxs:

                # Create obspy.Station
                station = Station(self.stations[_i], self.latitudes[_i],
                                  self.longitudes[_i], -self.burials[_i])

                # append to list of stations belonging to the network
                stations.append(station)

            # Create Network
            network = Network(net, stations=stations)

            # Append to list of networks
            networks.append(network)

        # Create inventory
        inv = Inventory(networks=networks)

        return inv
