
import numpy as np
from mpi4py import MPI
import h5py


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

with h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD) as f:

    dset = f.create_dataset('test', (4, 1000), dtype='i',
                            chunks=(1, 1000), compression="gzip")

    with dset.collective:
        dset[rank] = np.full(1000, rank, dtype='i')
