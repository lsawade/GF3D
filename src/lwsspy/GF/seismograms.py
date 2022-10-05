import h5py
import contextlib
from obspy import Trace, Stream
from obspy.core.trace import Stats
from scipy.spatial import KDTree
import numpy as np
from .lagrange import gll_nodes, lagrange_any
from .source2xyz import source2xyz
from .locate_point import locate_point
from .source import CMTSOLUTION


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
    a single station. Otherwise, please use the SGTManager. It makes more sense
    if your goal is to perform source inversion using a set of stations, and/or
    a single station, but the location may change.

    """

    with h5py.File(stationfile, 'r') as db:

        station = db['Station'][()].decode("utf-8")
        network = db['Network'][()].decode("utf-8")
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
        # t0 = db['TC'][()]
        FACTOR = db['FACTOR'][()]

    # Create KDTree
    kdtree = KDTree(xyz[ibool[2, 2, 2, :], :])

    # Get location in mesh
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

    # Locate the point in mesh
    ispec_selected, xi, eta, gamma, _, _, _, _ = locate_point(
        x_target, y_target, z_target, cmt.latitude, cmt.longitude,
        xyz[ibool[2, 2, 2, :], :], xyz[:, 0], xyz[:, 1], xyz[:, 2], ibool,
        POINT_CAN_BE_BURIED=True, kdtree=kdtree)

    # Read strains from the file
    with h5py.File(stationfile, 'r') as db:

        factor = db['FACTOR'][()]
        epsilond = dict()
        for comp in ['N', 'E', 'Z']:
            epsilond[comp] = \
                db[f'epsilon/{comp}/array'][
                    :, :, :, :, ispec_selected, :].astype(np.float64) / factor

    # GLL points and weights (degree)
    npol = 4
    xigll, _, _ = gll_nodes(npol)

    # Get lagrange values at specific GLL poins
    shxi, _ = lagrange_any(xi, xigll, npol)
    sheta, _ = lagrange_any(eta, xigll, npol)
    shgamma, _ = lagrange_any(gamma, xigll, npol)

    # Initialize epsilon array
    sepsilon = np.zeros((3, 6, epsilond['N'].shape[-1]))

    for k in range(NGLLZ):
        for j in range(NGLLY):
            for i in range(NGLLX):
                hlagrange = shxi[i] * sheta[j] * shgamma[k]

                for _i, comp in enumerate(['N', 'E', 'Z']):

                    sepsilon[_i, :, :] += hlagrange * \
                        epsilond[comp][:, i, j, k, :]

    # Add traces to the
    traces = []

    for _i, comp in enumerate(['N', 'E', 'Z']):
        data = np.sum(np.array([1., 1., 1., 2., 2., 2.])[:, None]
                      * Mx[:, None] * np.squeeze(sepsilon[_i, :, :]), axis=0)
        stats = Stats()
        stats.delta = dt
        stats.network = network
        stats.station = station
        stats.location = ''
        stats.channel = f'MX{comp}'
        stats.starttime = cmt.cmt_time - tc
        stats.npts = len(data)
        tr = Trace(data=data, header=stats)

        traces.append(tr)

    return Stream(traces)


class SGTManager(object):
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
    lat: float                # in degrees

    lon: float                # in degrees

    depth: float              # in km

    dblist: list[str]         # list of station data base files.

    headerfile: str           # path to first file in dblist

    header: dict              # Dictionary with same values for all station

    midpoints: np.ndarray     # all midpoints

    fullkdtree: KDTree    # KDTree to get elements

    ispec: np.ndarray         # subset of elements

    xyz: np.ndarray           # subset of the coordinates

    ibool: np.ndarray         # index array for locations

    kdtree: KDTree         # Kdtree for the subset of elements.

    components: list[str] = ['N', 'E', 'Z']  # components

    def __init__(self, dblist: list[str]) -> None:
        """Initializes the SGT manager"""

        # List of station files
        self.dblist = dblist
        self.headerfile = self.dblist[0]
        self.Ndb = len(self.dblist)

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
            # t0 = db['TC'][()]
            self.header['factor'] = db['FACTOR'][()]

        # Create KDTree
        self.fullkdtree = KDTree(self.header['midpoints'])

    def get_elements(self, lat, lon, depth, k=10):

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

        # Get elements
        point_target = np.array([x_target, y_target, z_target])
        _, self.ispec_subset = self.fullkdtree.query(point_target, k=k)

        # Sort for reading HDF%
        self.ispec_subset = np.sort(self.ispec_subset)

        # Get number of files
        with contextlib.ExitStack() as stack:

            # open stack of files.
            dbs = [stack.enter_context(h5py.File(fname, 'r'))
                   for fname in self.dblist]

            # Read ibool, xyz
            ibool = dbs[0]['ibool'][:, :, :, self.ispec_subset]

            # Get unique elements
            uni, inv = np.unique(ibool, return_inverse=True)

            # Get new index array of length of the unique values
            indeces = np.arange(len(uni))

            # Get fixed ibool array for the interpolation and source location
            self.ibool = indeces[inv].reshape(ibool.shape)

            # Then finally get sub set of coordinates
            self.xyz = dbs[0]['xyz'][uni, :]

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
            self.stations = []

            # Initialize big array for aaaalll the strains at aaall
            # the stations
            self.epsilon = np.zeros((
                self.Ndb,
                len(self.components),
                6,
                self.header['NGLLX'],
                self.header['NGLLY'],
                self.header['NGLLZ'],
                len(self.ispec_subset),
                self.header['nsteps']
            ))

            for _i, db in enumerate(dbs):

                # Get force factor specific to file
                factor = db['FACTOR'][()]

                for _j, comp in enumerate(self.components):
                    self.epsilon[_i, _j, :, :, :, :, :, :] = \
                        db[f'epsilon/{comp}/array'][
                            :, :, :, :, self.ispec_subset, :].astype(np.float64) / factor

    def get_seismogram(self, cmt: CMTSOLUTION):

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

        # Locate the point in mesh
        ispec_selected, xi, eta, gamma, _, _, _, _ = locate_point(
            x_target, y_target, z_target, cmt.latitude, cmt.longitude,
            self.xyz[self.ibool[2, 2, 2, :], :],
            self.xyz[:, 0], self.xyz[:, 1], self.xyz[:, 2],
            self.ibool, POINT_CAN_BE_BURIED=True, kdtree=self.kdtree)

        # GLL points and weights (degree)
        npol = 4
        xigll, _, _ = gll_nodes(self.header['NGLLX']-1)
        etagll, _, _ = gll_nodes(self.header['NGLLY']-1)
        gammagll, _, _ = gll_nodes(self.header['NGLLZ']-1)

        # Get lagrange values at specific GLL poins
        shxi, _ = lagrange_any(xi, xigll, self.header['NGLLX']-1)
        sheta, _ = lagrange_any(eta, etagll, self.header['NGLLY']-1)
        shgamma, _ = lagrange_any(gamma, gammagll, self.header['NGLLZ']-1)

        # # Initialize epsilon array
        # sepsilon = np.zeros((3, 6, epsilon['N'].shape[-1]))
        M = Mx * np.array([1., 1., 1., 2., 2., 2.])

        # h is stations, i is compenent, j is strain element (0-5),
        # klm are gll points, n are is element, o is time
        # print('eps', self.epsilon.sape)
        print('eps', self.epsilon[:, :, :, :, :, :, ispec_selected, :].shape)
        print('M', M.shape)
        print('xi', shxi.shape)
        print('et', sheta.shape)
        print('ga', shgamma.shape)

        seismograms = np.einsum(
            'hijklmo,j,k,l,m->hio',
            self.epsilon[:, :, :, :, :, :, ispec_selected, :], M, shxi, sheta, shgamma)

        print(seismograms.shape)

        return seismograms
        # for k in range(NGLLZ):
        #     for j in range(NGLLY):
        #         for i in range(NGLLX):
        #             hlagrange=shxi[i] * sheta[j] * shgamma[k]

        #             for _i, comp in enumerate(['N', 'E', 'Z']):

        #                 sepsilon[_i, :, :] += hlagrange * \
        #                     epsilond[comp][:, i, j, k, :]
