#!/usr/bin/env python

from lwsspy.GF.postprocess_mpi import Adios2HDF5
import sys
from mpi4py.MPI import COMM_WORLD as comm


def global_except_hook(exctype, value, traceback):
    sys.stderr.write("except_hook. Calling MPI_Abort().\n")
    # NOTE: mpi4py must be imported inside exception handler, not globally.
    # In chainermn, mpi4py import is carefully delayed, because
    # mpi4py automatically call MPI_Init() and cause a crash on Infiniband environment.
    import mpi4py.MPI
    mpi4py.MPI.COMM_WORLD.Abort(1)
    sys.__excepthook__(exctype, value, traceback)


sys.excepthook = global_except_hook


def processadios(h5file, Nfile, Efile, Zfile, config_file, precision, compression):

    if comm.Get_rank() == 0:

        # Processing I/O
        print("Processing I/O and Params", flush=True)
        print("-------------------------", flush=True)
        print("    H5", h5file, flush=True)
        print("     N", Nfile, flush=True)
        print("     E", Efile, flush=True)
        print("     Z", Zfile, flush=True)
        print("   CFG", config_file, flush=True)
        print("     P", precision, flush=True)
        print("     C", compression, flush=True)

    else:
        pass

    print('I reached the processing.', flush=True)

    with Adios2HDF5(
            h5file,
            Nfile,
            Efile,
            Zfile,
            config_file, subspace=False,
            precision=precision,
            compression=compression,  # 'gzip',
            compression_opts=None,
            comm=comm) as A2H:

        A2H.write()


# Get input!
if len(sys.argv) != 8:
    raise ValueError

h5file, Nfile, Efile, Zfile, config_file, precision, compression = \
    sys.argv[1:]

processadios(h5file, Nfile, Efile, Zfile,
             config_file, precision, compression)
