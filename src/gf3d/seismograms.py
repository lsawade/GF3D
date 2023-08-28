import os
import logging
import h5py
import contextlib
import typing as tp
from copy import deepcopy
from obspy import Trace, Stream
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


def stationIO(
        _i: int, lock: Lock, subsetfilename: str, stationfilename: str, nsteps: int,
        nglob2sub: np.ndarray, sglob: np.ndarray, rsglob: np.ndarray,
        fortran: bool = False):
    """Station IO function this function is executed on a worker and reads in
    the station information and subset displacement data and writes it to the
    subset file if no other worker is writing to the file.


    Parameters
    ----------
    _i : int
        station index
    q : Queue
        multiprocessing queue containing a writing lock
    subsetfilename : str
        subset file name
    stationfilename : str
        station filename
    nsteps : int
        number of time steps
    nglob2sub : np.ndarray
        index array mapping global to subset
    sglob : np.ndarray
        sorting the indeces to read in ascending order
    rsglob : np.ndarray
        reverse sorting the indeces to read in ascending order
    fortran : bool, optional
        write in Fortran order, by default False
    """

    print(f'{_i:03d}: {stationfilename} -> {subsetfilename}')

    with h5py.File(stationfilename, 'r') as db:

        # Load station info
        network = db['Network'][()].decode("utf-8")
        station = db['Station'][()].decode("utf-8")
        latitude = db['latitude'][()]
        longitude = db['longitude'][()]
        burials = db['burial'][()]

        # Get force factor specific to file
        factor = db['FACTOR'][()]

        # Nglob
        Nglob = len(nglob2sub)

        # Initialize displacement array
        displacement = np.zeros((3, 3, Nglob, nsteps))

        components = ['N', 'E', 'Z']

        # Get norm and displacement
        for _j, comp in enumerate(components):

            # Get norm and displacement
            norm = db[f'displacement/{comp}/norm'][()]

            # Get displacement getting the output in ascending order
            t0 = time()
            array = db[f'displacement/{comp}/array'][:,
                                                     nglob2sub[sglob], :]
            print(f"{_i: > 05d} reading: {time()-t0}")

            t0 = time()
            array = array.astype(np.float32)
            print(f"{_i: > 05d} convert: {time()-t0}")

            t0 = time()
            array = array[:, rsglob, :] * norm / factor
            print(f"{_i: > 05d} resort:  {time()-t0}")

            t0 = time()
            displacement[_j, :, :, :] = array
            print(f"{_i: > 05d} assign:  {time()-t0}")

    lock.acquire()

    with h5py.File(subsetfilename, 'r+') as db:

        # Store fixed length strings for fortran
        db['Networks'][_i] = network
        db['Stations'][_i] = station
        db['latitudes'][_i] = latitude
        db['longitudes'][_i] = longitude
        db['burials'][_i] = burials

        if fortran:
            db[f'displacement'][:, :, :, :, _i] = displacement[:,
                                                               :, :, :].transpose((3, 2, 1, 0))
        else:
            db[f'displacement'][_i, :, :, :, :] = displacement[:, :, :, :]

    lock.release()


def stationIO_DB(
        _i: int, stationfilename: str, outDBdir: str,
        nsteps: int, ibool: np.ndarray, xyz: np.ndarray, NGLL: int,
        nglob2sub: np.ndarray, sglob: np.ndarray, rsglob: np.ndarray,
        do_adjacency_search: bool, xadj, adjacency):
    """Station IO function this function is executed on a worker and reads in
    the station information and subset displacement data and writes it to the
    subset file if no other worker is writing to the file.


    Parameters
    ----------
    _i : int
        station index
    q : Queue
        multiprocessing queue containing a writing lock
    stationfilename : str
        subset file name
    outstationfilename : str
        station filename
    nsteps : int
        number of time steps
    ibool : np.ndarray
        index array
    xyz : np.ndarray
        coordinates
    NGLL : int
        number of GLL points in one direction
    nglob2sub : np.ndarray
        index array mapping global to subset
    sglob : np.ndarray
        sorting the indeces to read in ascending order
    rsglob : np.ndarray
        reverse sorting the indeces to read in ascending order
    """

    header = dict()
    # Just get network and station name to create both file handles
    # (less indentation)
    with h5py.File(stationfilename, 'r') as db:
        header['Network'] = db['Network'][()].decode("utf-8")
        header['Station'] = db['Station'][()].decode("utf-8")

    # Make filename
    outstationfilename = os.path.join(
        outDBdir, header['Network'], header['Station'],
        f"{header['Network']}.{header['Station']}.h5")

    print(f'{_i:03d}: {stationfilename} -> {outstationfilename}')

    if os.path.exists(outstationfilename) is False:
        os.makedirs(os.path.dirname(outstationfilename), exist_ok=True)

    with h5py.File(stationfilename, 'r') as db, h5py.File(outstationfilename, 'w') as dbout:

        # station = db['Station'][()].decode("utf-8")
        # network = db['Network'][()].decode("utf-8")
        header['TOPOGRAPHY'] = db['TOPOGRAPHY'][()]
        header['ELLIPTICITY'] = db['ELLIPTICITY'][()]

        if header['TOPOGRAPHY']:
            header['BATHY'] = db['BATHY'][:]
            header['NX_BATHY'] = db['NX_BATHY'][()]
            header['NY_BATHY'] = db['NY_BATHY'][()]
            header['RESOLUTION_TOPO_FILE'] = db['RESOLUTION_TOPO_FILE'][()]

        if header['ELLIPTICITY']:
            header['rspl'] = db['rspl'][:]
            header['ellipticity_spline'] = db['ellipticity_spline'][:]
            header['ellipticity_spline2'] = db['ellipticity_spline2'][:]

        # Only read midpoints for now
        header['DT'] = db['DT'][()]
        header['TC'] = db['TC'][()]
        header['FACTOR'] = db['FACTOR'][()]
        header['HDUR'] = db['HDUR'][()]

        # Load station info
        header['latitude'] = db['latitude'][()]
        header['longitude'] = db['longitude'][()]
        header['burial'] = db['burial'][()]

        # Get force factor specific to file
        header['FACTOR'] = db['FACTOR'][()]

        # Nglob
        header['NGLOB'] = len(nglob2sub)
        header['NGLLX'] = NGLL
        header['NGLLY'] = NGLL
        header['NGLLZ'] = NGLL
        header['NSTEPS'] = nsteps

        # Coordinates and index array
        header['ibool'] = ibool
        header['xyz'] = xyz

        # Add adjacency or not
        if do_adjacency_search:
            header['do_adjacency_search'] = 1
            header['xadj'] = xadj
            header['adjacency'] = adjacency
            header['USE_BUFFER_ELEMENTS'] = 1
        else:
            header['do_adjacency_search'] = 0
            header['USE_BUFFER_ELEMENTS'] = 0

        # Write everything to the file except the displacement
        for key, value in header.items():
            dbout.create_dataset(key, data=value)

        components = ['N', 'E', 'Z']

        # Get norm and displacement
        for comp in components:

            # Write norm to new file
            key = f'displacement/{comp}/norm'
            dbout.create_dataset(key, data=db[key][()])

            # Get displacement getting the output in ascending order
            t0 = time()
            key = f'displacement/{comp}/array'
            array = db[key][:, nglob2sub[sglob], :header['NSTEPS']]
            print(f"{_i: > 05d} reading: {time()-t0}")

            t0 = time()
            array = array.astype(np.float32)
            print(f"{_i: > 05d} convert: {time()-t0}")

            t0 = time()
            array = array[:, rsglob, :]
            print(f"{_i: > 05d} resort:  {time()-t0}")

            dbout.create_dataset(key, data=array)


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


def get_seismograms(stationfile: str, cmt: CMTSOLUTION, ispec: int | None = None):
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

    if ispec is not None:
        print(ibool.shape)

        ibool = ibool[:, :, :, ispec:ispec+1]
        ibool_orig = deepcopy(ibool)

        # Get unique elements
        uni, inv = np.unique(ibool, return_inverse=True)

        # Get new index array of length of the unique values
        indeces = np.arange(len(uni))

        # Get fixed ibool array for the interpolation and source location
        ibool = indeces[inv].reshape(ibool.shape)

        # Then finally get sub set of coordinates
        xyz = xyz[uni, :]

        # Mini kdtree
        kdtree = KDTree(xyz[ibool[NGLLX//2, NGLLY//2, NGLLZ//2, :]])

        do_adjacency_search = False

    else:

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
    logger.debug(f' xyz.shape: {xyz.shape}')

    # Locate the point in mesh
    logger.debug('Locating the point ...')
    ispec_selected, xi, eta, gamma, xix, xiy, xiz, etax, etay, etaz, gammax, gammay, gammaz, _, _, _, _ = locate_point(
        x_target, y_target, z_target, cmt.latitude, cmt.longitude,
        xyz[ibool[2, 2, 2, :], :], xyz[:, 0], xyz[:, 1], xyz[:, 2], ibool,
        xadj=xadj, adjacency=adjacency,
        POINT_CAN_BE_BURIED=True, kdtree=kdtree,
        do_adjacent_search=do_adjacency_search, NGLL=NGLLX)
    logger.debug('...Done')
    logger.debug(f'SELECTED ELEMENT: {ispec_selected}')

    # Read strains from the file
    logger.debug('Loading strains ...')
    with h5py.File(stationfile, 'r') as db:

        factor = db['FACTOR'][()]
        displacementd = dict()

        if ispec is not None:

            # Get global indeces.
            iglob = ibool_orig[:, :, :, ispec_selected].flatten()

        else:

            # Get global indeces.
            iglob = ibool[:, :, :, ispec_selected].flatten()

        # HDF5 can only access indeces in incresing order. So, we have to
        # sort the globs, and after we retreive the array unsort it and
        # reshape it
        sglob = np.argsort(iglob)
        rsglob = np.argsort(sglob)

        for _i, comp in enumerate(['N', 'E', 'Z']):

            norm_disp = db[f'displacement/{comp}/norm'][()]

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
        hdur_r = 1e-6
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

    # Fourier Transform the STF
    STF_R = fft.fft(stf_r, n=NP2)

    # Compute phase shift for the heaviside STF
    shift = -200.0 + cmt.time_shift
    phshift = np.exp(-1.0j*shift*np.fft.fftfreq(NP2, dt)*2*np.pi)

    # Add traces to the
    traces = []

    for _i, comp in enumerate(['N', 'E', 'Z']):

        # Get displacement from strain
        data = np.sum(np.array([1., 1., 1., 2., 2., 2.])[:, None]
                      * Mx[:, None] * np.squeeze(epsilon[_i, :, :]), axis=0)

        # Convolution with Specfem Heaviside function
        data = np.real(
            fft.ifft(phshift * fft.fft(data, n=NP2) * dt * STF_R))[:NT]

        stats = Stats()
        stats.delta = dt
        stats.network = network
        stats.station = station
        stats.location = 'S3'
        stats.latitude = latitude
        stats.longitude = longitude
        stats.coordinates = AttribDict(
            latitude=latitude, longitude=longitude)
        stats.channel = f'MX{comp}'
        stats.starttime = cmt.origin_time - tc
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

            displacementd[comp] = db[f'displacement/{comp}/array'][
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
    shift = -100.0 + cmt.time_shift
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
        stats.location = 'S3'
        stats.latitude = latitude
        stats.longitude = longitude
        stats.coordinates = AttribDict(
            latitude=latitude, longitude=longitude)
        stats.channel = f'MX{comp}'
        stats.starttime = cmt.origin_time - tc
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

    def load_scalar_header_parameters(self):
        """Small function that loads only the scalar header
        """
        header = dict()

        with h5py.File(self.headerfile, 'r') as db:
            header['topography'] = db['TOPOGRAPHY'][()]
            header['ellipticity'] = db['ELLIPTICITY'][()]

            if header['topography']:
                logger.info("Loading topography ...")
                header['nx_topo'] = db['NX_BATHY'][()]
                header['ny_topo'] = db['NY_BATHY'][()]
                header['res_topo'] = db['RESOLUTION_TOPO_FILE'][()]

            if header['ellipticity']:
                logger.info("Loading ellipticity ...")
                header['nspl'] = len(db['rspl'][:])

            # Grid
            header['NSPEC'] = db['NSPEC'][()]
            header['NGLOB'] = db['NGLOB'][()]
            header['NGLLX'] = db['NGLLX'][()]
            header['NGLLY'] = db['NGLLY'][()]
            header['NGLLZ'] = db['NGLLZ'][()]

            # Timing
            header['dt'] = db['DT'][()]
            header['tc'] = db['TC'][()]
            header['nsteps'] = db['NSTEPS'][()]
            header['factor'] = db['FACTOR'][()]
            header['hdur'] = db['HDUR'][()]

            # Buffer elements
            header['USE_BUFFER_ELEMENTS'] = db['USE_BUFFER_ELEMENTS'][()]

        return header

    def load_header_variables(self):

        self.header = dict()

        with h5py.File(self.headerfile, 'r') as db:
            # station = db['Station'][()].decode("utf-8")
            # network = db['Network'][()].decode("utf-8")
            self.header['topography'] = db['TOPOGRAPHY'][()]
            self.header['ellipticity'] = db['ELLIPTICITY'][()]

            if self.header['topography']:
                logger.info("Loading topography ...")
                self.header['itopo'] = db['BATHY'][:]
                self.header['nx_topo'] = db['NX_BATHY'][()]
                self.header['ny_topo'] = db['NY_BATHY'][()]
                self.header['res_topo'] = db['RESOLUTION_TOPO_FILE'][()]

            if self.header['ellipticity']:
                logger.info("Loading ellipticity ...")
                self.header['rspl'] = db['rspl'][:]
                self.header['ellipticity_spline'] = db['ellipticity_spline'][:]
                self.header['ellipticity_spline2'] = db['ellipticity_spline2'][:]

            self.header['NGLOB'] = db['NGLOB'][()]
            self.header['NGLLX'] = db['NGLLX'][()]
            self.header['NGLLY'] = db['NGLLY'][()]
            self.header['NGLLZ'] = db['NGLLZ'][()]

            # Only read midpoints for now
            logger.info("Loading ibool midpoints ...")
            self.ibool = db['ibool'][:]

            logger.info("Loading xyz midpoints ...")
            self.xyz = db['xyz'][:]
            self.header['midpoints'] = self.xyz[
                self.ibool[
                    self.header['NGLLX']//2,
                    self.header['NGLLY']//2,
                    self.header['NGLLZ']//2, :],
                :]
            logger.info("Loading scalars ...")
            self.header['dt'] = db['DT'][()]
            self.header['tc'] = db['TC'][()]
            self.header['nsteps'] = db['NSTEPS'][()]
            self.header['factor'] = db['FACTOR'][()]
            self.header['hdur'] = db['HDUR'][()]

            # Create KDTree
            logger.info("Making KDTree ...")
            self.fullkdtree = KDTree(self.header['midpoints'])

            # Now if this is a subset, we can already define the needed arrays
            # for source location, that are usually defined using the
            # .get_elements method
            if self.subset:
                self.ibool = db['ibool'][:]
                self.stations = db['Stations'][:]
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

    def get_stations(self):

        # Check if KDtree is loaded
        if not self.header:
            self.load_header_variables()

        if self.subset:
            networks = self.networks
            stations = self.stations
            latitudes = self.latitudes
            longitudes = self.longitudes
            burials = self.burials

        else:

            networks = []
            stations = []
            latitudes = []
            longitudes = []
            burials = []
            # Get number of files
            logger.info("Opening h5 files ...")
            with contextlib.ExitStack() as stack:
                # Open Stack of files
                dbs = [stack.enter_context(h5py.File(fname, 'r'))
                       for fname in self.db]

                for db in dbs:
                    networks.append(db['Network'][()].decode())
                    stations.append(db['Station'][()].decode())
                    latitudes.append(db['latitude'][()])
                    longitudes.append(db['longitude'][()])
                    burials.append(db['burial'][()])

        return [sta_tup for sta_tup in zip(
            networks, stations, latitudes, longitudes, burials)]

    def get_elements(self, lat, lon, depth, dist_in_km=125.0, NGLL=5, threading: bool = True):

        if self.subset:
            logging.warning(
                'Note that you already loaded a subset of elements from file, '
                'so this may have 0 effect.')

        # source location to query
        self.lat = lat
        self.lon = lon
        self.depth = depth

        logger.info("Locating source ...")

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
        logger.info("Querying KDTree ...")
        point_target = np.array([x_target, y_target, z_target])
        self.ispec_subset = self.fullkdtree.query_ball_point(point_target, r=r)

        # Catch single element query
        if isinstance(self.ispec_subset, np.int64):
            self.ispec_subset = np.array([self.ispec_subset], dtype='i')
        elif isinstance(self.ispec_subset, list):
            if len(self.ispec_subset) == 0:
                raise ValueError(
                    "Could not find any elements within the radius.")
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
            logger.info("Opening h5 files ...")
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
                logger.info("Getting ibool subset ...")
                ibool = self.ibool[
                    iboolslice, iboolslice, iboolslice,
                    self.ispec_subset]

                # Get unique elements
                logger.info("Uniqueing ibool ...")
                self.nglob2sub, inv = np.unique(ibool, return_inverse=True)

                # HDF5 can only access indeces in incresing order. So, we have to
                # sort the globs, and after we retreive the array unsort it and
                # reshape it
                # For later I'll need to test this before I can implement it.
                sglob = np.argsort(self.nglob2sub)
                rsglob = np.argsort(sglob)

                # Get new index array of length of the unique values
                indeces = np.arange(len(self.nglob2sub))

                # Get fixed ibool array for the interpolation and source location
                self.ibool = indeces[inv].reshape(ibool.shape)

                # Then finally get sub set of coordinates
                self.xyz = self.xyz[self.nglob2sub, :]

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
                    logger.info("Making Adjacency ...")

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
                logger.info("Initializing Displacement array ...")
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
                self.burials = self.Ndb * [None]
                self.NGLL

                logger.info(
                    "Getting displacement from the individual stations")

                def read_stuff(args):
                    _i, db = args

                    print(f'{_i}: {db}')
                    network = db['Network'][()].decode("utf-8")
                    station = db['Station'][()].decode("utf-8")
                    self.networks[_i] = network
                    self.stations[_i] = station
                    self.latitudes[_i] = db['latitude'][()]
                    self.longitudes[_i] = db['longitude'][()]
                    self.burials[_i] = db['burial'][()]

                    logger.debug(f"Reading {network}.{station}")

                    # Get force factor specific to file
                    factor = db['FACTOR'][()]

                    for _j, comp in enumerate(self.components):

                        # Get norm and displacement
                        norm = db[f'displacement/{comp}/norm'][()]

                        # Get displacement getting the output in ascending order
                        t0 = time()
                        array = db[f'displacement/{comp}/array'][:,
                                                                 self.nglob2sub[sglob], :]
                        print(f"{_i: > 05d} reading: {time()-t0}")
                        t0 = time()
                        array = array.astype(np.float32)
                        print(f"{_i: > 05d} convert: {time()-t0}")
                        t0 = time()
                        array = array[:, rsglob, :] * norm / factor
                        print(f"{_i: > 05d} resort:  {time()-t0}")
                        t0 = time()
                        self.displacement[_i, _j, :, :, :] = array
                        print(f"{_i: > 05d} assign:  {time()-t0}")

                        # self.displacement[_i, _j, :, :, :]  = db[f'displacement/{comp}/array'][:, self.nglob2sub[sglob], :].astype(
                        #     np.float32)[:, rsglob, :] * norm / factor

                        #  displacementd[comp] = db[f'displacement/{comp}/array'][
                        #         :, iglob[sglob], :].astype(np.float64)[:, rsglob, :].reshape(3, NGLLX, NGLLY, NGLLZ, NT) * norm_disp / factor

                if threading:
                    with parallel_backend('threading', n_jobs=10):
                        Parallel()(delayed(read_stuff)(i)
                                   for i in zip(range(self.Ndb), dbs))
                else:
                    for i in zip(range(self.Ndb), dbs):
                        read_stuff(i)

    def write_subset_directIO(
            self, subsetfilename, lat, lon, depth, dist_in_km=125.0, duration=None,
            NGLL=5, fortran: bool = True):

        if self.subset:
            logging.warning('Note that you already loaded a subset. Exiting')
            return

        # source location to query
        self.lat = lat
        self.lon = lon
        self.depth = depth

        # Check if KDtree is loaded
        if not self.header:
            self.load_header_variables()

        logger.info("Locating source ...")

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

        # Get normalized distance
        r = dist_in_km/6371.0

        # Get elements
        logger.info("Querying KDTree ...")
        point_target = np.array([x_target, y_target, z_target])
        self.ispec_subset = self.fullkdtree.query_ball_point(point_target, r=r)

        # Catch single element query
        if isinstance(self.ispec_subset, np.int64):
            self.ispec_subset = np.array([self.ispec_subset], dtype='i')
        elif isinstance(self.ispec_subset, list):
            if len(self.ispec_subset) == 0:
                raise ValueError(
                    "Could not find any elements within the radius.")
            else:
                self.ispec_subset = np.sort(self.ispec_subset)

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
        logger.info("Getting ibool subset ...")
        ibool = self.ibool[
            iboolslice, iboolslice, iboolslice,
            self.ispec_subset]

        # Get unique elements
        logger.info("Uniqueing ibool ...")
        self.nglob2sub, inv = np.unique(ibool, return_inverse=True)

        # HDF5 can only access indeces in incresing order. So, we have to
        # sort the globs, and after we retreive the array unsort it and
        # reshape it
        # For later I'll need to test this before I can implement it.
        sglob = np.argsort(self.nglob2sub)
        rsglob = np.argsort(sglob)

        # Get new index array of length of the unique values
        indeces = np.arange(len(self.nglob2sub))

        # Get fixed ibool array for the interpolation and source location
        self.ibool = indeces[inv].reshape(ibool.shape)

        # Then finally get sub set of coordinates
        self.xyz = self.xyz[self.nglob2sub, :]

        # Read strains into big array
        self.networks = []
        self.stations = []
        self.latitudes = []
        self.longitudes = []

        if self.do_adjacency_search:
            logger.info("Making Adjacency ...")

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
        logger.info("Initializing Displacement array ...")
        self.displacement = np.zeros((
            self.Ndb,
            len(self.components),
            3,
            len(self.nglob2sub),
            self.header['nsteps']
        ))

        logger.debug("Arraylength: {len(self.nglob2sub)}")

        with h5py.File(subsetfilename, 'w') as db:

            # Store fixed length strings for fortran
            if fortran:
                # NOTE THAT THE STRING LENGTH HERE IS HARD-CODED in the
                # fortran extraction codes according to the specfem3dglobe
                # definitions.
                db.create_dataset('Networks', (self.Ndb,), dtype="S8")
                db.create_dataset('Stations', (self.Ndb,), dtype="S32")
            else:
                # For python subsets this does not matter.
                db.create_dataset('Networks', (self.Ndb,), dtype="S32")
                db.create_dataset('Stations', (self.Ndb,), dtype="S32")

            db.create_dataset('latitudes', (self.Ndb,), dtype='f')
            db.create_dataset('longitudes', (self.Ndb,), dtype='f')
            db.create_dataset('burials', (self.Ndb,), dtype='f')

            # Ibool and coordinates
            if fortran:
                db.create_dataset('fortran', data=1)
                db.create_dataset(
                    'ibool', data=self.ibool.transpose((3, 2, 1, 0)))
                db.create_dataset('xyz', data=self.xyz.transpose((1, 0)))
            else:
                db.create_dataset('ibool', data=self.ibool)
                db.create_dataset('xyz', data=self.xyz)

            # Adjacency flag
            db.create_dataset('do_adjacency_search',
                              data=self.do_adjacency_search)

            # Adjacency arrays
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

            # Topography flags
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

            # Ellipticity flags
            if self.header['ellipticity']:
                db.create_dataset('rspl', data=self.header['rspl'])
                db.create_dataset('ellipticity_spline',
                                  data=self.header['ellipticity_spline'])
                db.create_dataset('ellipticity_spline2',
                                  data=self.header['ellipticity_spline2'])

            # General variable
            db.create_dataset('NGLOB', data=len(self.nglob2sub))
            db.create_dataset('NGLLX', data=self.NGLL)
            db.create_dataset('NGLLY', data=self.NGLL)
            db.create_dataset('NGLLZ', data=self.NGLL)

            # Using all the previously defined variables we can now create the
            # Main shape of the displacement array.
            dispshape = (
                self.Ndb, len(self.components), 3, len(self.nglob2sub), nsteps)

            if fortran:
                logger.info('Initializing Fortran order displacement array')
                db.create_dataset('displacement', dispshape[::-1], dtype='f')
            else:
                logger.info('Initializing C order displacement array')
                db.create_dataset(
                    'displacement', dispshape, dtype='f')
            # shuffle=True, compression='lzf')

            logger.info('Done.')
        logger.info(
            "Getting displacement from the individual stations")

        # Process lock
        lock = Lock()
        processes = []

        # Loop over stations and start I/O for each station.
        for i in range(self.Ndb):
            p = Process(target=stationIO, args=(
                i, lock, subsetfilename, self.db[i], nsteps,
                self.nglob2sub, sglob, rsglob,
                fortran))
            processes.append(p)
            p.start()

        for process in processes:
            process.join()

    def write_DB_directIO(
            self, DBdir, lat, lon, depth, dist_in_km=125.0, duration=None,
            NGLL=5):

        if self.subset:
            logging.warning('Note that you already loaded a subset. Exiting')
            return

        # Check if KDtree is loaded
        if not self.header:
            self.load_header_variables()

        logger.info("Locating source ...")

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

        # Get normalized distance
        r = dist_in_km/6371.0

        # Get elements
        logger.info("Querying KDTree ...")
        point_target = np.array([x_target, y_target, z_target])
        self.ispec_subset = self.fullkdtree.query_ball_point(point_target, r=r)

        # Catch single element query
        if isinstance(self.ispec_subset, np.int64):
            self.ispec_subset = np.array([self.ispec_subset], dtype='i')
        elif isinstance(self.ispec_subset, list):
            if len(self.ispec_subset) == 0:
                raise ValueError(
                    "Could not find any elements within the radius.")
            else:
                self.ispec_subset = np.sort(self.ispec_subset)

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

        print(self.ibool)
        print(iboolslice)
        print(self.ibool.shape)

        # Read ibool
        logger.info("Getting ibool subset ...")
        ibool = self.ibool[
            iboolslice, iboolslice, iboolslice,
            self.ispec_subset]

        # Get unique elements
        logger.info("Uniqueing ibool ...")
        self.nglob2sub, inv = np.unique(ibool, return_inverse=True)

        # HDF5 can only access indeces in incresing order. So, we have to
        # sort the globs, and after we retreive the array unsort it and
        # reshape it
        # For later I'll need to test this before I can implement it.
        sglob = np.argsort(self.nglob2sub)
        rsglob = np.argsort(sglob)

        print("N2S", self.nglob2sub)
        # Get new index array of length of the unique values
        indeces = np.arange(len(self.nglob2sub))

        # Get fixed ibool array for the interpolation and source location
        self.ibool = indeces[inv].reshape(ibool.shape)

        # Then finally get sub set of coordinates
        self.xyz = self.xyz[self.nglob2sub, :]

        if self.do_adjacency_search:
            logger.info("Making Adjacency ...")

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
        else:
            do_adjacency_search = False
            xadj = None
            adjacency = None

        # Header Variables
        if duration is not None:
            nsteps = int(
                np.ceil((self.header['tc'] + duration)/self.header['dt']))

            # Can only store as many steps as we have...
            if nsteps > self.header['nsteps']:
                nsteps = self.header['nsteps']
        else:
            nsteps = self.header['nsteps']

        # Process lock
        lock = Lock()
        processes = []

        # Loop over stations and start I/O for each station.
        for i in range(self.Ndb):

            p = Process(target=stationIO_DB, args=(
                i, self.db[i], DBdir,
                nsteps, self.ibool, self.xyz, NGLL, self.nglob2sub, sglob, rsglob,
                self.do_adjacency_search, xadj, adjacency))
            processes.append(p)
            p.start()

        for process in processes:
            process.join()

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
