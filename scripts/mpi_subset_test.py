
from copy import deepcopy
from gf3d.mpi_subset import MPISubset
from gf3d.source import CMTSOLUTION
import os, sys
from mpi4py import MPI
comm = MPI.COMM_WORLD

def mpiabort_excepthook(type, value, traceback):
    comm.Abort()
    sys.__excepthook__(type, value, traceback)



cmtfile = """ PDEW2018  1 23  9 31 40.90  56.0000 -149.1700  14.1 0.0 7.9 GULF OF ALASKA
event name:     201801230931A
time shift:     23.0700
half duration:  22.3000
latitude:       56.2200
longitude:    -149.1200
depth:          33.6100
Mrr:       2.360000e+27
Mtt:      -4.850000e+27
Mpp:       2.500000e+27
Mrt:       1.940000e+27
Mrp:      -3.620000e+27
Mtp:       7.910000e+27
"""

pertdict = dict(
    synt=dict(pert=None, type=None),
    Mrr=dict(pert=1e23, type='FFD'),
    Mtt=dict(pert=1e23, type='FFD'),
    Mpp=dict(pert=1e23, type='FFD'),
    Mrt=dict(pert=1e23, type='FFD'),
    Mrp=dict(pert=1e23, type='FFD'),
    Mtp=dict(pert=1e23, type='FFD'),
    latitude=dict(pert=0.0001, type='CFD'),
    longitude=dict(pert=0.0001, type='CFD'),
    depth=dict(pert=0.0001, type='CFD'),
    time_shift=dict(pert=-1.0, type='grad'),
)


def test_mpi_subset():

    import mpi4py.MPI as MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    MS = MPISubset('subset.h5')

    # Print the rank and the type of the kdtree
    print(f'{rank}/{size}', type(MS.kdtree))
    print(f'{rank}/{size}', MS.header['tc'])


    # Make CMTSOLUTIONS
    cmt = CMTSOLUTION.read(cmtfile)

    # Perturb them in latitude by rank
    cmt.latitude += rank/2

    # Get the seismograms
    st = MS.get_seismograms(cmt)


def test_mpi_perturb():

    outdir = 'test_mpi_perturb'



    import mpi4py.MPI as MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()


    if rank == 0:
        if os.path.exists(outdir) is False:
            os.mkdir(outdir)

        parameters = [
            'synt',
            'Mrr',
            # 'Mtt',
            # 'Mpp',
            # 'Mrt',
            # 'Mrp',
            # 'Mtp',
            # 'latitude',
            # 'longitude',
            'depth',
            # 'time_shift'
        ]

        # Make local version that includes only the wanted parameters
        local_pertdict = dict()
        for p in parameters:
            local_pertdict[p] = pertdict[p]

        # Compute needed size for the MPI COMM World
        needed_size = sum([1 if local_pertdict[key]['type'] != 'CFD' else 2
                           for key in local_pertdict.keys()])

        # Hello
        print(needed_size)

        if needed_size != size:
            print('ERROR: number of MPI processes must be equal to'
                  'the number offorward modelings', flush=True)
            raise ValueError()


        # Make CMTSOLUTIONS
        cmt = CMTSOLUTION.read(cmtfile)

        # Make the perturbations
        rankcounter = 0
        rankmap = []

        for par in local_pertdict.keys():

            print(f'{par}', flush=True)

            if par == 'synt':

                syntcmt = deepcopy(cmt)
                rankmap.append([syntcmt, 'synt', None, None])
                rankcounter += 1

            elif local_pertdict[par]['type'] == 'CFD':

                # Make positive perturbation
                poscmt = deepcopy(cmt)

                val = getattr(poscmt, par)
                setattr(poscmt, par, val + local_pertdict[par]['pert'])

                rankmap.append([poscmt, par, 1, rankcounter + 1])

                negcmt = deepcopy(cmt)
                val = getattr(negcmt, par)
                setattr(negcmt, par, val - local_pertdict[par]['pert'])

                rankmap.append([negcmt, par, -1, rankcounter])

                rankcounter += 2

            elif local_pertdict[par]['type'] == 'FFD':

                # Make positive perturbation
                poscmt = deepcopy(cmt)

                val = getattr(poscmt, par)
                setattr(poscmt, par, val + local_pertdict[par]['pert'])

                rankmap.append([poscmt, par, None, None])

                rankcounter += 1

            elif local_pertdict[par]['type'] == 'grad':

                # Make positive perturbation
                rankmap.append([deepcopy(cmt), par, None, None])

                rankcounter += 1

            else:
                print('ERROR: unknown perturbation type', flush=True)
                raise ValueError()

    else:
        rankmap = None
        local_pertdict = None

    # Broadcast the local_pertdict
    local_pertdict = comm.bcast(local_pertdict, root=0)

    # Scatter cmt solutions with the rankmap
    rankmap = comm.scatter(rankmap, root=0)
    cmt = rankmap[0]
    par = rankmap[1]
    pert = rankmap[2]
    sr_rank = rankmap[3]
    print(f"{rank}/{size}", par, pert, sr_rank, flush=True)

    # Get the seismograms
    MS = MPISubset('subset.h5')

    # Get the seismograms
    print(f"{rank}/{size} -- Getting seismograms", flush=True)
    st = MS.get_seismograms(cmt)

    print(f"{rank}/{size} --     Got seismograms", flush=True)
    # Compute final seismograms
    if par == 'synt':
        pass
        # station = st.select(network='II', station='ARU', component='Z')
        # station.filter('lowpass', freq=1.0/90.0)
        # station.plot(outfile='synthetics.png')

    elif local_pertdict[par]['type'] == 'CFD':


        if pert == 1:
            print(f"{rank}/{size} -- waiting to receive", flush=True)
            neg = comm.recv(source=sr_rank)
            print(f"{rank}/{size} -- received", flush=True)
        elif pert == -1:
            print(f"{rank}/{size} -- waiting to send", flush=True)
            comm.send(st, dest=sr_rank)
            print(f"{rank}/{size} -- sent", flush=True)
        else:
            raise ValueError('Only 1 and -1 can be used for pert since it '
                             'indicates sending or receiving streams')

        if pert == 1:
            for _postr, _negtr in zip(st, neg):

                # Subtract and divide by twice the perturbation
                _postr.data = (_postr.data - _negtr.data) \
                    / (2 * local_pertdict[par]['pert'])

    elif local_pertdict[par]['type'] == 'grad':

        # Take the gradient
        st.differentiate()

        # Multiply by -1
        for _tr in st:
            _tr.data *= -1

    elif local_pertdict[par]['type'] == 'FFD':

        # Divide the perturbed seismogram by the perturbation
        for tr in st:
            tr.data *= 1/local_pertdict[par]['pert']

    else:
        raise ValueError(f'Unknown perturbation type. Abort for par: {par}')


    # Write the seismograms
    if par == 'synt' or par == 'time_shift' or pert == 1 or local_pertdict[par]['type']:
        st.write(f'{outdir}/{par}.mseed', format='MSEED')


if __name__ == '__main__':
    # sys.excepthook = mpiabort_excepthook
    test_mpi_perturb()
    # sys.excepthook = sys.__excepthook__
