import logging
import h5py
import contextlib
from copy import deepcopy
from obspy import Trace, Stream
from obspy.core.trace import Stats
from scipy.spatial import KDTree
import numpy as np
from joblib import parallel_backend, Parallel, delayed
from obspy.core.util.attribdict import AttribDict
from .lagrange import gll_nodes, lagrange_any
from .source2xyz import source2xyz
from .locate_point import locate_point
from .source import CMTSOLUTION
from .utils import timeshift, next_power_of_2
from .stf import create_stf
from .logger import logger
from scipy import fft


def get_frechet(cmt, stationfile):
    """Computes centered finite difference using 10m perturbations"""

    mtpar = ['Mrr', 'Mtt', 'Mpp', 'Mrt', 'Mrp', 'Mtp']
    pertdict = dict(
        Mrr=1e23,
        Mtt=1e23,
        Mpp=1e23,
        Mrt=1e23,
        Mrp=1e23,
        Mtp=1e23,
        latitude=0.0001,
        longitude=0.0001,
        depth=0.01,
        time_shift=-1.0,
    )

    frechets = dict()

    for par, pert in pertdict.items():

        if par in mtpar:

            dcmt = deepcopy(cmt)

            # Set all M... to zero
            for mpar in mtpar:
                setattr(dcmt, mpar, 0.0)

            # Set one to none-zero
            setattr(dcmt, par, pert)

            # Get reciprocal synthetics
            drp = get_seismograms(stationfile, dcmt)

            for tr in drp:
                tr.data /= pert

        elif par == 'time_shift':

            drp = get_seismograms(stationfile, cmt)
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
            prp = get_seismograms(stationfile, pcmt)
            mrp = get_seismograms(stationfile, mcmt)

            for ptr, mpr in zip(prp, mrp):
                ptr.data -= mpr.data
                ptr.data /= 2 * pert

            # Reassign to match
            drp = prp

        frechets[par] = drp

    return frechets


def get_seismograms(stationfile: str, cmt: CMTSOLUTION):
    """

    This function takes in the path to a station file and a CMTSOLUTION and
    subsequently returns an obspy Stream with the relevant seismograms.
    The function performs the following steps:

    * read the stationfiles header info (TOPO, ELLIP, etc.)
    * convert cmtsolution to mesh coordinates, rotate moment tensor
    * get the element from a kdtree that uses all the element midpoints
    * locate source location in the element
    * grabs element strains
    * interpolates the strain to the source point
    * dots the moment tensor with the strains of all 3 components
    * returns Stream() containing all three components

    Use this function ONLY if your goal is retrieveing a single seismogram for
    a single station. Otherwise, please use the GFManager. It makes more sense
    if your goal is to perform source inversion using a set of stations, and/or
    a single station, but the location may change.

    """

    with h5py.File(stationfile, 'r') as db:

        station = db['Station'][()].decode("utf-8")
        network = db['Network'][()].decode("utf-8")
        latitude = db['latitude'][()]
        longitude = db['longitude'][()]
        topography = db['TOPOGRAPHY'][()]
        ellipticity = db['ELLIPTICITY'][()]
        ibathy_topo = db['BATHY'][:]
        NX_BATHY = db['NX_BATHY'][()]
        NY_BATHY = db['NY_BATHY'][()]
        RESOLUTION_TOPO_FILE = db['RESOLUTION_TOPO_FILE'][()]
        rspl = db['rspl'][:]
        ellipticity_spline = db['ellipticity_spline'][:]
        ellipticity_spline2 = db['ellipticity_spline2'][:]
        NGLLX = db['NGLLX'][()]
        NGLLY = db['NGLLY'][()]
        NGLLZ = db['NGLLZ'][()]
        ibool = db['ibool'][:]
        xyz = db['xyz'][:]
        dt = db['DT'][()]
        tc = db['TC'][()]
        NT = db['NSTEPS'][()]
        hdur = db['HDUR'][()]

        # t0 = db['TC'][()]
        do_adjacency_search = db['USE_BUFFER_ELEMENTS'][()]

        if do_adjacency_search:
            logger.debug(f'adj {do_adjacency_search}')
            xadj = db['xadj'][:]
            adjacency = db['adjacency'][:]
        else:
            xadj = None
            adjacency = None

    # Create KDTree
    logger.debug('Building KDTree ...')
    kdtree = KDTree(xyz[ibool[2, 2, 2, :], :])
    logger.debug('... Done')

    # Get location in mesh
    logger.debug('Conversion geo -> xyz')

    x_target, y_target, z_target, Mx = source2xyz(
        cmt.latitude,
        cmt.longitude,
        cmt.depth,
        M=cmt.tensor,
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
    logger.debug('... Done')

    # logger.debug('xadj', np.min(xadj), np.max(xadj))
    # logger.debug('adjacency', np.min(adjacency), np.max(adjacency))

    # Locate the point in mesh
    logger.debug('Locating the point ...')
    ispec_selected, xi, eta, gamma, xix, xiy, xiz, etax, etay, etaz, gammax, gammay, gammaz, _, _, _, _ = locate_point(
        x_target, y_target, z_target, cmt.latitude, cmt.longitude,
        xyz[ibool[2, 2, 2, :], :], xyz[:, 0], xyz[:, 1], xyz[:, 2], ibool,
        xadj=xadj, adjacency=adjacency,
        POINT_CAN_BE_BURIED=True, kdtree=kdtree,
        do_adjacent_search=do_adjacency_search, NGLL=NGLLX)
    logger.debug('...Done')

    # Read strains from the file
    logger.debug('Loading strains ...')
    with h5py.File(stationfile, 'r') as db:

        factor = db['FACTOR'][()]
        displacementd = dict()
        for _i, comp in enumerate(['N', 'E', 'Z']):

            norm_disp = db[f'displacement/{comp}/norm'][()]

            # Get global indeces.
            iglob = ibool[:, :, :, ispec_selected].flatten()

            # HDF5 can only access indeces in incresing order. So, we have to
            # sort the globs, and after we retreive the array unsort it and
            # reshape it
            sglob = np.argsort(iglob)
            rsglob = np.argsort(sglob)

            displacementd[comp] = db[f'displacement/{comp}/array'][
                :, iglob[sglob], :].astype(np.float64)[:, rsglob, :].reshape(3, NGLLX, NGLLY, NGLLZ, NT) * norm_disp / factor

    logger.debug('... Done')

    # GLL points and weights (degree)
    npol = NGLLX-1
    xigll, _, _ = gll_nodes(npol)

    # Get lagrange values at specific GLL poins
    hxi, hpxi = lagrange_any(xi, xigll, npol)
    heta, hpeta = lagrange_any(eta, xigll, npol)
    hgamma, hpgamma = lagrange_any(gamma, xigll, npol)

    # Initialize epsilon array
    epsilon = np.zeros((3, 6, NT))
    for k in range(NGLLZ):
        for j in range(NGLLY):
            for i in range(NGLLX):

                hlagrange_xi = hpxi[i] * heta[j] * hgamma[k]
                hlagrange_eta = hxi[i] * hpeta[j] * hgamma[k]
                hlagrange_gamma = hxi[i] * heta[j] * hpgamma[k]
                hlagrange_x = hlagrange_xi * xix + hlagrange_eta * etax + hlagrange_gamma * gammax
                hlagrange_y = hlagrange_xi * xiy + hlagrange_eta * etay + hlagrange_gamma * gammay
                hlagrange_z = hlagrange_xi * xiz + hlagrange_eta * etaz + hlagrange_gamma * gammaz

                for _i, comp in enumerate(['N', 'E', 'Z']):

                    epsilon[_i, 0, :] += (
                        displacementd[comp][0, i, j, k, :] * hlagrange_x)
                    epsilon[_i, 1, :] += (
                        displacementd[comp][1, i, j, k, :] * hlagrange_y)
                    epsilon[_i, 2, :] += (
                        displacementd[comp][2, i, j, k, :] * hlagrange_z)
                    epsilon[_i, 3, :] += 0.5 * (
                        displacementd[comp][1, i, j, k, :] * hlagrange_x
                        + displacementd[comp][0, i, j, k, :] * hlagrange_y)
                    epsilon[_i, 4, :] += 0.5 * (
                        displacementd[comp][2, i, j, k, :] * hlagrange_x
                        + displacementd[comp][0, i, j, k, :] * hlagrange_z)
                    epsilon[_i, 5, :] += 0.5 * (
                        displacementd[comp][2, i, j, k, :] * hlagrange_y
                        + displacementd[comp][1, i, j, k, :] * hlagrange_z)

    # For following FFTs
    NP2 = next_power_of_2(2*NT)

    # This computes the half duration for the new STF from the
    if (cmt.hdur / 1.628) <= hdur:
        hdur_r = 0.000001
        logger.warn(
            f"Requested half duration smaller than what was simulated.\n"
            f"Half duration set to {hdur_r}s to simulate a Heaviside function.")
    else:
        hdur_r = np.sqrt((cmt.hdur / 1.628)**2 - hdur**2)

    logger.debug(
        f"CMT hdur: {cmt.hdur:.3f}, GF DB hdur: {hdur:.3f} -> Used hdur: {hdur_r}")

    # Heaviside STF to reproduce SPECFEM stf
    _, stf_r = create_stf(0, 200.0, NT, dt, hdur_r,
                          cutoff=None, gaussian=False, lpfilter='butter')
    STF_R = fft.fft(stf_r, n=NP2)
    shift = -200.0
    phshift = np.exp(-1.0j*shift*np.fft.fftfreq(NP2, dt)*2*np.pi)

    # Add traces to the
    traces = []

    for _i, comp in enumerate(['N', 'E', 'Z']):

        # Get displacement from strain
        data = np.sum(np.array([1., 1., 1., 2., 2., 2.])[:, None]
                      * Mx[:, None] * np.squeeze(epsilon[_i, :, :]), axis=0)

        # Convolution with Specfem Heaviside function
        data = np.real(
            fft.ifft(phshift * fft.fft(data, n=NP2) * STF_R))[:NT] * dt

        stats = Stats()
        stats.delta = dt
        stats.network = network
        stats.station = station
        stats.latitude = latitude
        stats.longitude = longitude
        stats.coordinates = AttribDict(
            latitude=latitude, longitude=longitude)
        stats.location = ''
        stats.channel = f'MX{comp}'
        stats.starttime = cmt.cmt_time - tc
        stats.npts = len(data)
        tr = Trace(data=data, header=stats)
        traces.append(tr)

    return Stream(traces)


def get_seismograms_sub(stationfile: str, cmt: CMTSOLUTION):
    """

    This function takes in the path to a station file and a CMTSOLUTION and
    subsequently returns an obspy Stream with the relevant seismograms.
    The function performs the following steps:

    * read the stationfiles header info (TOPO, ELLIP, etc.)
    * convert cmtsolution to mesh coordinates, rotate moment tensor
    * get the element from a kdtree that uses all the element midpoints
    * locate source location in the element
    * grabs element strains
    * interpolates the strain to the source point
    * dots the moment tensor with the strains of all 3 components
    * returns Stream() containing all three components

    Use this function ONLY if your goal is retrieveing a single seismogram for
    a single station. Otherwise, please use the GFManager. It makes more sense
    if your goal is to perform source inversion using a set of stations, and/or
    a single station, but the location may change.

    """

    with h5py.File(stationfile, 'r') as db:

        station = db['Station'][()].decode("utf-8")
        network = db['Network'][()].decode("utf-8")
        latitude = db['latitude'][()]
        longitude = db['longitude'][()]
        topography = db['TOPOGRAPHY'][()]
        ellipticity = db['ELLIPTICITY'][()]
        ibathy_topo = db['BATHY'][:]
        NX_BATHY = db['NX_BATHY'][()]
        NY_BATHY = db['NY_BATHY'][()]
        RESOLUTION_TOPO_FILE = db['RESOLUTION_TOPO_FILE'][()]
        rspl = db['rspl'][:]
        ellipticity_spline = db['ellipticity_spline'][:]
        ellipticity_spline2 = db['ellipticity_spline2'][:]
        NGLLX = db['NGLLX'][()]
        NGLLY = db['NGLLY'][()]
        NGLLZ = db['NGLLZ'][()]
        ibool = db['ibool'][:]
        xyz = db['xyz'][:]
        dt = db['DT'][()]
        tc = db['TC'][()]
        NT = db['NSTEPS'][()]
        hdur = db['HDUR'][()]

        # t0 = db['TC'][()]
        FACTOR = db['FACTOR'][()]
        do_adjacency_search = db['USE_BUFFER_ELEMENTS'][()]

        if do_adjacency_search:
            xadj = db['xadj'][:]
            adjacency = db['adjacency'][:]
        else:
            xadj = None
            adjacency = None

    # Create KDTree
    logger.debug('Building KDTree ...')
    kdtree = KDTree(xyz[ibool[2, 2, 2, :], :])
    logger.debug('... Done')

    # Get location in mesh
    logger.debug('Conversion geo -> xyz')
    x_target, y_target, z_target, Mx = source2xyz(
        cmt.latitude,
        cmt.longitude,
        cmt.depth,
        M=cmt.tensor,
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
    logger.debug('... Done')

    # logger.debug('xadj', np.min(xadj), np.max(xadj))
    # logger.debug('adjacency', np.min(adjacency), np.max(adjacency))

    # Locate the point in mesh
    logger.debug('Locating the point ...')
    ispec_selected, xi, eta, gamma, xix, xiy, xiz, etax, etay, etaz, gammax, gammay, gammaz, _, _, _, _ = locate_point(
        x_target, y_target, z_target, cmt.latitude, cmt.longitude,
        xyz[ibool[2, 2, 2, :], :], xyz[:, 0], xyz[:, 1], xyz[:, 2], ibool,
        xadj=xadj, adjacency=adjacency,
        POINT_CAN_BE_BURIED=True, kdtree=kdtree,
        do_adjacent_search=do_adjacency_search, NGLL=NGLLX)
    logger.debug('...Done')

    # Read strains from the file
    logger.debug('Loading strains ...')
    with h5py.File(stationfile, 'r') as db:

        factor = db['FACTOR'][()]
        displacementd = dict()
        for _i, comp in enumerate(['N', 'E', 'Z']):

            norm_disp = db[f'displacement/{comp}/norm'][()]

            # Get global indeces.
            iglob = ibool[::2, ::2, ::2, ispec_selected].flatten()

            # HDF5 can only access indeces in incresing order. So, we have to
            # sort the globs, and after we retreive the array unsort it and
            # reshape it
            sglob = np.argsort(iglob)
            rsglob = np.argsort(sglob)

            displacementd[comp] = \
                db[f'displacement/{comp}/array'][
                    :, iglob[sglob], :].astype(np.float64)[:, rsglob, :].reshape(3, NGLLX-2, NGLLY-2, NGLLZ-2, NT) * norm_disp / factor

    logger.debug('... Done')

    # GLL points and weights (degree)
    npol = (NGLLX-2)-1
    xigll, _, _ = gll_nodes(npol)

    # Get lagrange values at specific GLL poins
    hxi, hpxi = lagrange_any(xi, xigll, npol)
    heta, hpeta = lagrange_any(eta, xigll, npol)
    hgamma, hpgamma = lagrange_any(gamma, xigll, npol)

    # Initialize epsilon array
    epsilon = np.zeros((3, 6, NT))
    for k in range(NGLLZ-2):
        for j in range(NGLLY-2):
            for i in range(NGLLX-2):

                hlagrange_xi = hpxi[i] * heta[j] * hgamma[k]
                hlagrange_eta = hxi[i] * hpeta[j] * hgamma[k]
                hlagrange_gamma = hxi[i] * heta[j] * hpgamma[k]
                hlagrange_x = hlagrange_xi * xix + hlagrange_eta * etax + hlagrange_gamma * gammax
                hlagrange_y = hlagrange_xi * xiy + hlagrange_eta * etay + hlagrange_gamma * gammay
                hlagrange_z = hlagrange_xi * xiz + hlagrange_eta * etaz + hlagrange_gamma * gammaz

                for _i, comp in enumerate(['N', 'E', 'Z']):

                    epsilon[_i, 0, :] += (
                        displacementd[comp][0, i, j, k, :] * hlagrange_x)
                    epsilon[_i, 1, :] += (
                        displacementd[comp][1, i, j, k, :] * hlagrange_y)
                    epsilon[_i, 2, :] += (
                        displacementd[comp][2, i, j, k, :] * hlagrange_z)
                    epsilon[_i, 3, :] += 0.5 * (
                        displacementd[comp][1, i, j, k, :] * hlagrange_x
                        + displacementd[comp][0, i, j, k, :] * hlagrange_y)
                    epsilon[_i, 4, :] += 0.5 * (
                        displacementd[comp][2, i, j, k, :] * hlagrange_x
                        + displacementd[comp][0, i, j, k, :] * hlagrange_z)
                    epsilon[_i, 5, :] += 0.5 * (
                        displacementd[comp][2, i, j, k, :] * hlagrange_y
                        + displacementd[comp][1, i, j, k, :] * hlagrange_z)

    # For following FFTs
    NP2 = next_power_of_2(2*NT)

    # This computes the half duration for the new STF from the
    hdur_r = np.sqrt((cmt.hdur / 1.628)**2 - hdur**2)

    # Heaviside STF to reproduce SPECFEM stf
    _, stf_r = create_stf(0, 100.0, NT, dt, hdur_r,
                          cutoff=None, gaussian=False, lpfilter='butter')
    STF_R = fft.fft(stf_r, n=NP2)
    shift = -100.0
    phshift = np.exp(-1.0j*shift*np.fft.fftfreq(NP2, dt)*2*np.pi)

    # Add traces to the
    traces = []

    for _i, comp in enumerate(['N', 'E', 'Z']):

        # Get displacement from strain
        data = np.sum(np.array([1., 1., 1., 2., 2., 2.])[:, None]
                      * Mx[:, None] * np.squeeze(epsilon[_i, :, :]), axis=0)

        # Convolution with Specfem Heaviside function
        data = np.real(
            fft.ifft(phshift * fft.fft(data, n=NP2) * STF_R))[:NT] * dt

        stats = Stats()
        stats.delta = dt
        stats.network = network
        stats.station = station
        stats.latitude = latitude
        stats.longitude = longitude
        stats.coordinates = AttribDict(
            latitude=latitude, longitude=longitude)
        stats.location = ''
        stats.channel = f'MX{comp}'
        stats.starttime = cmt.cmt_time - tc
        stats.npts = len(data)
        tr = Trace(data=data, header=stats)
        traces.append(tr)

    return Stream(traces)


class GFManager(object):
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
    db: list[str] | str   # list of station data base files.

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

    def __init__(self, db: list[str] | str) -> None:
        """Initializes the gfm manager"""

        # List of station files
        self.db = db
        if isinstance(self.db, str):
            self.subset = True
            self.headerfile = self.db
        else:
            self.subset = False
            self.headerfile = self.db[0]

    def get_mesh_location(self):
        pass

    def load_header_variables(self):

        self.header = dict()

        with h5py.File(self.headerfile, 'r') as db:
            # station = db['Station'][()].decode("utf-8")
            # network = db['Network'][()].decode("utf-8")
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

            self.header['NGLLX'] = db['NGLLX'][()]
            self.header['NGLLY'] = db['NGLLY'][()]
            self.header['NGLLZ'] = db['NGLLZ'][()]

            # Only read midpoints for now
            ibool = db['ibool'][
                self.header['NGLLX']//2,
                self.header['NGLLY']//2,
                self.header['NGLLZ']//2,
                :
            ]

            self.header['midpoints'] = db['xyz'][ibool, :]
            self.header['dt'] = db['DT'][()]
            self.header['tc'] = db['TC'][()]
            self.header['nsteps'] = db['NSTEPS'][()]
            self.header['factor'] = db['FACTOR'][()]
            self.header['hdur'] = db['HDUR'][()]

            # Create KDTree
            self.fullkdtree = KDTree(self.header['midpoints'])

            # Now if this is a subset, we can already define the needed arrays
            # for source location, that are usually defined using the
            # .get_elements method
            if self.subset:
                self.ibool = db['ibool'][:]
                self.epsilon = db['epsilon'][:]
                self.stations = db['stations'][:]
                self.xyz = db['xyz']
                self.kdtree = self.fullkdtree
                self.displacement = db['displacement'][:]
                self.do_adjacency_search = db['do_adjacency_search'][()]

            # Possible overwrites
            if self.do_adjacency_search:

                if self.subset:
                    self.header['xadj'] = db['xadj'][:]
                    self.header['adjacency'] = db['adjacency'][:]
                else:
                    do_adjacency_search = db['USE_BUFFER_ELEMENTS'][()]

                    if do_adjacency_search:
                        self.header['xadj'] = db['xadj'][:]
                        self.header['adjacency'] = db['adjacency'][:]
                    else:
                        logger.debug(
                            'Dont have buffer elements for doing the ')
                        self.do_adjacency_search = False

    def get_elements(self, lat, lon, depth, dist_in_km=125.0, NGLL=5):

        if self.subset:
            logging.warning(
                'Note that you already loaded a subset of elements from file, '
                'so this may have 0 effect.')

        # source location to query
        self.lat = lat
        self.lon = lon
        self.depth = depth

        x_target, y_target, z_target = source2xyz(
            lat, lon, depth, M=None,
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

        # Check if KDtree is loaded
        if not self.header:
            self.load_header_variables()

        # Get normalized distance
        r = dist_in_km/6371.0

        # Get elements
        point_target = np.array([x_target, y_target, z_target])
        self.ispec_subset = self.fullkdtree.query_ball_point(point_target, r=r)

        logger.debug('queried the kdtree')

        # Sort for reading HDF5
        if isinstance(self.ispec_subset, np.int64):
            self.ispec_subset = np.array([self.ispec_subset], dtype='i')
        else:
            self.ispec_subset = np.sort(self.ispec_subset)

        if self.subset:

            self.ibool = self.ibool[:, :, :, self.ispec_subset]

            # Get unique elements
            uni, inv = np.unique(self.ibool, return_inverse=True)

            # Get new index array of length of the unique values
            indeces = np.arange(len(uni))

            # Get fixed ibool array for the interpolation and source location
            self.ibool = indeces[inv].reshape(self.ibool.shape)

            # Then finally get sub set of coordinates
            self.xyz = self.xyz[uni, :]

            # Mini kdtree
            self.kdtree = KDTree(
                self.xyz[
                    self.ibool[
                        self.header['NGLLX']//2,
                        self.header['NGLLY']//2,
                        self.header['NGLLZ']//2,
                        :]
                ]
            )

            # Finally redefine epsilon
            self.epsilon = self.epsilon[:, :, :, :, :, :, self.ispec_subset, :]

        else:

            # Get number of files

            with contextlib.ExitStack() as stack:

                # Open Stack of files
                dbs = [stack.enter_context(h5py.File(fname, 'r'))
                       for fname in self.db]

                # Number of database files
                self.Ndb = len(self.db)

                # Depending on choice here we can create a subset with 3 GLL point from
                # one with 5!
                if NGLL == 3:
                    if self.header['NGLLX'] == 3:
                        iboolslice = slice(0, NGLL)
                        self.NGLL = 3
                    else:
                        iboolslice = slice(0, 5, 2)
                        self.NGLL = 3
                elif NGLL == 5:
                    if self.header['NGLLX'] == 3:
                        iboolslice = slice(0, 3)
                        self.NGLL = 3
                        print(72*'=')
                        print(
                            'The original database was only save using 3 GLL points.\n'
                            'You requested 5. Setting GLL to 3.')
                        print(72*'=')
                    else:
                        iboolslice = slice(0, 5)
                        self.NGLL = 5
                else:
                    raise ValueError(
                        f'NGLL {NGLL} is not valid. Choose 3 or 5.')

                # Read ibool
                ibool = dbs[0]['ibool'][
                    iboolslice, iboolslice, iboolslice,
                    self.ispec_subset]

                # Get unique elements
                self.nglob2sub, inv = np.unique(ibool, return_inverse=True)
                logger.debug('Done uniqueing')

                # Get new index array of length of the unique values
                indeces = np.arange(len(self.nglob2sub))

                # Get fixed ibool array for the interpolation and source location
                self.ibool = indeces[inv].reshape(ibool.shape)

                # Then finally get sub set of coordinates
                self.xyz = dbs[0]['xyz'][self.nglob2sub, :]

                # Mini kdtree
                self.kdtree = KDTree(
                    self.xyz[
                        self.ibool[
                            self.header['NGLLX']//2,
                            self.header['NGLLY']//2,
                            self.header['NGLLZ']//2,
                            :]
                    ]
                )

                # Read strains into big array
                self.networks = []
                self.stations = []
                self.latitudes = []
                self.longitudes = []

                if self.do_adjacency_search:
                    self.xadj = np.zeros(len(self.ispec_subset) + 1, dtype='i')
                    MAX_NEIGHBORS = 50
                    tmp_adjacency = np.zeros(
                        MAX_NEIGHBORS*len(self.ispec_subset), dtype='i')

                    inum_neighbor = 0
                    for _j in range(len(self.ispec_subset)):

                        ispec = self.ispec_subset[_j]

                        # Get neighbors
                        num_neighbors = self.header['xadj'][ispec +
                                                            1] - self.header['xadj'][ispec]

                        # Loop over neighbors in full mesh
                        for i in range(num_neighbors):

                            # get neighbor from global adjacency
                            ispec_neighbor = self.header['adjacency'][self.header['xadj'][ispec] + i]

                            # Check whether global neighbor is also a local neighbor.
                            if ispec_neighbor in self.ispec_subset:

                                # If is neighbor increase total neighbor counter.
                                inum_neighbor = inum_neighbor + 1

                                # Get indeces of neighbors
                                idx = np.where(ispec_neighbor ==
                                               self.ispec_subset)[0]

                                # Add neighbor to adjacency vector
                                tmp_adjacency[inum_neighbor] = idx

                        # Add total event counter to adjacency vetor
                        self.xadj[_j+1] = inum_neighbor

                        # Define final adjacency array
                        self.adjacency = tmp_adjacency[:inum_neighbor]

                        # Define neighbor counter
                        num_neighbors_all_gf = inum_neighbor

                # Initialize big array for aaaalll the strains at aaall
                # the stations
                self.displacement = np.zeros((
                    self.Ndb,
                    len(self.components),
                    3,
                    len(self.nglob2sub),
                    self.header['nsteps']
                ))

                logger.debug("Arraylength: {len(self.nglob2sub)}")

                self.Ndb
                self.networks = self.Ndb * [None]
                self.stations = self.Ndb * [None]
                self.latitudes = self.Ndb * [None]
                self.longitudes = self.Ndb * [None]
                self.NGLL

                def read_stuff(args):
                    _i, db = args

                    logger.debug(f'{_i}: {db}')
                    network = db['Network'][()].decode("utf-8")
                    station = db['Station'][()].decode("utf-8")
                    self.networks[_i] = network
                    self.stations[_i] = station
                    self.latitudes[_i] = db['latitude'][()]
                    self.longitudes[_i] = db['longitude'][()]

                    logger.debug(f"Reading {network}.{station}")

                    # Get force factor specific to file
                    factor = db['FACTOR'][()]

                    for _j, comp in enumerate(self.components):

                        # Get norm and displacement
                        norm = db[f'displacement/{comp}/norm'][()]

                        # Get displacement
                        self.displacement[_i, _j, :, :, :] = db[f'displacement/{comp}/array'][:, self.nglob2sub, :].astype(
                            np.float32) * norm / factor

                with parallel_backend('threading', n_jobs=self.Ndb):
                    Parallel()(delayed(read_stuff)(i)
                               for i in zip(range(self.Ndb), dbs))

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
        else:
            hdur_diff = np.sqrt((cmt.hdur / 1.628)**2 - self.header['hdur']**2)

        # Heaviside STF to reproduce SPECFEM stf
        _, stf_r = create_stf(0, 400.0, self.header['nsteps'],
                              self.header['dt'], hdur_diff, cutoff=None, gaussian=False, lpfilter='butter')

        STF_R = fft.fft(stf_r, n=NP2)

        shift = -400.0
        phshift = np.exp(-1.0j*shift*np.fft.fftfreq(NP2,
                                                    self.header['dt'])*2*np.pi)

        logger.debug(f"Lengths: {self.header['nsteps']}, {NP2}")
        # Add traces to the
        traces = []
        for _h in range(len(self.stations)):
            for _i, comp in enumerate(['N', 'E', 'Z']):

                data = np.real(
                    fft.ifft(
                        STF_R * fft.fft(seismograms[_h, _i, :], n=NP2)
                        * phshift
                    ))[:self.header['nsteps']] * self.header['dt']

                stats = Stats()
                stats.delta = self.header['dt']
                stats.network = self.networks[_h]
                stats.station = self.stations[_h]
                stats.latitude = self.latitudes[_h]
                stats.longitude = self.longitudes[_h]
                stats.coordinates = AttribDict(
                    latitude=self.latitudes[_h], longitude=self.longitudes[_h])
                stats.location = ''
                stats.channel = f'MX{comp}'
                stats.starttime = cmt.cmt_time - self.header['tc']
                stats.npts = self.header['nsteps']
                tr = Trace(data=data, header=stats)

                traces.append(tr)

        logger.debug('Outputting traces')
        return Stream(traces)

    def get_frechet(self, cmt: CMTSOLUTION, rtype=3):
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

        for par, pert in pertdict.items():

            if par in mtpar:

                dcmt = deepcopy(cmt)

                # Set all M... to zero
                for mpar in mtpar:
                    setattr(dcmt, mpar, 0.0)

                # Set one to none-zero
                setattr(dcmt, par, pert)

                # Get reciprocal synthetics
                drp = self.get_seismograms(dcmt)

                for tr in drp:
                    tr.data /= pert

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

                for ptr, mpr in zip(prp, mrp):
                    ptr.data -= mpr.data
                    ptr.data /= 2 * pert

                # Reassign to match
                drp = prp

            frechets[par] = drp

        return frechets

    def write_subset(self, outfile, duration=None):
        """Given the files in the database, get a set of strains and write it
        into to a single file in a single epsilon array.
        Also write a list of stations and normal header info required for
        source location. This very same GFManager would be used to read the.
        Where """

        with h5py.File(outfile, 'w') as db:

            db.create_dataset('Networks', data=self.networks)
            db.create_dataset('Stations', data=self.stations)
            db.create_dataset('latitudes', data=self.latitudes)
            db.create_dataset('longitudes', data=self.longitudes)
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
                db.create_dataset('BATHY', data=self.header['itopo'])
                db.create_dataset('NX_BATHY', data=self.header['nx_topo'])
                db.create_dataset('NY_BATHY', data=self.header['ny_topo'])
                db.create_dataset('RESOLUTION_TOPO_FILE',
                                  data=self.header['res_topo'])

            if self.header['ellipticity']:
                db.create_dataset('rspl', data=self.header['rspl'])
                db.create_dataset('ellipticity_spline',
                                  data=self.header['ellipticity_spline'])
                db.create_dataset('ellipticity_spline2',
                                  data=self.header['ellipticity_spline2'])

            db.create_dataset('NGLLX', data=self.NGLL)
            db.create_dataset('NGLLY', data=self.NGLL)
            db.create_dataset('NGLLZ', data=self.NGLL)

            # Use duration keyword to set the number of samples to save to file:
            print('shape', self.displacement.shape)
            print('nsteps', nsteps)
            db.create_dataset(
                'displacement', data=self.displacement[:, :, :, :, :nsteps])  # ,
            # shuffle=True, compression='lzf')

    def load(self):
        """Given the files in the database, get a set of strains and write it
        into to a single file in a single epsilon array.
        Also write a list of stations and normal header info required for
        source location. This very same GFManager would be used to read the.
        Where """

        self.header = dict()

        with h5py.File(self.headerfile, 'r') as db:

            self.header['NGLLX'] = db['NGLLX'][()]
            self.header['NGLLY'] = db['NGLLY'][()]
            self.header['NGLLZ'] = db['NGLLZ'][()]
            self.NGLL = self.header['NGLLX']
            self.networks = db['Networks'][:].astype('U13').tolist()
            self.stations = db['Stations'][:].astype('U13').tolist()
            self.latitudes = db['latitudes'][:]
            self.longitudes = db['longitudes'][:]
            self.ibool = db['ibool'][:]
            self.xyz = db['xyz'][:]
            self.kdtree = KDTree(self.xyz[self.ibool[
                self.NGLL//2,
                self.NGLL//2,
                self.NGLL//2, :]])

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
