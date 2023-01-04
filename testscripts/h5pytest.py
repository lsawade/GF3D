
import numpy as np
from mpi4py import MPI
import h5py


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = np.array([3, 4, 0, 2])
Ntot = np.sum(N)
offsets = np.hstack((np.array([0]), np.cumsum(N)))

if size != 4:
    raise ValueError('This was made to be run with 4 processors.')

precision = np.float16

if rank == 0:
    x = np.random.randn(3, 1000).astype(precision)
elif rank == 1:
    x = np.random.randn(4, 1000).astype(precision)
elif rank == 2:
    x = np.random.randn(0, 1000).astype(precision)
elif rank == 3:
    x = np.random.randn(2, 1000).astype(precision)

comm.Barrier()

# dt = h5py.string_dtype(encoding='utf-8', length=10)
with h5py.File('parallel_test.h5', 'w', driver='mpio', comm=comm) as db:

    dset = db.create_dataset('test', (Ntot, 1000), dtype=precision,
                             chunks=(1, 1000), compression="lzf")
    with dset.collective:
        if offsets[rank]-offsets[rank+1] > 0:
            dset[offsets[rank]:offsets[rank+1], :] = x

    # dset = db.create_dataset('test', (4, 1000), dtype='i',
    #                          chunks=(1, 1000), compression="gzip")

    # with dset.collective:
    #     dset[rank] = np.full(1000, rank, dtype='i')

    # dset2 = db.create_dataset('blub', (1), dtype=dt)
    # if rank == 0:
    #     dset2[rank] = str.encode('hello')


# comm.Barrier()

# with h5py.File('parallel_test.h5', 'r', driver='mpio', comm=comm) as db:

#     print(rank, db['test'][:])
#     print(rank, db['blub'][0].decode())
