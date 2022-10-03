import h5py
from obspy import Trace, Stream
from scipy.spatial import KDTree
import numpy as np
from .lagrange import gll_nodes, lagrange_any
from .source2xyz import source2xyz
from .locate_point import locate_point


def get_seismogram(stationfile: str, cmt: CMTSOLUTION):

    with h5py.File(stationfile, 'r') as db:
        station = db['Station'][()]
        network = db['Network'][()]
        topography = db['TOPOGRAPHY'][()]
        ellipticity = db['ELLIPTICITY'][()]
        ibathy_topo = db['BATHY'][:]
        NX_BATHY = db['NX_BATHY'][()]
        NY_BATHY = db['NY_BATHY'][()]
        RESOLUTION_TOPO_FILE = db['RESOLUTION_TOPO_FILE'][()]
        rspl = db['rspl'][:]
        ellipticity_spline = db['ellipticity_spline'][:]
        ellipticity_spline2 = db['ellipticity_spline2'][:]
        NGLLX = db['NGLLX'][:]
        NGLLY = db['NGLLY'][:]
        NGLLZ = db['NGLLZ'][:]
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
        tr = Trace(data)
        tr.stats.delta = dt
        tr.stats.network = network
        tr.stats.station = station
        tr.stats.location = ''
        tr.stats.channel = f'MX{comp}'
        tr.stats.startime = cmt.origin_time - tc

        traces.append(tr)

    return Stream(traces)
