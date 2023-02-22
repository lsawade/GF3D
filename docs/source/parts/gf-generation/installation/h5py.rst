
.. _h5py-install:

`h5py` Installation
-------------------



.. code-block:: bash

    # Summit
    export HDF5_DIR=/gpfs/alpine/geo111/scratch/lsawade/SGT/SpecfemMagic/packages/hdf5/build/

    # Traverse
    export HDF5_DIR=/scratch/gpfs/lsawade/SpecfemMagic/packages/hdf5/build/

    CC="mpicc" HDF5_MPI="ON" HDF5_DIR=${HDF5_DIR} pip install --no-binary=h5py --no-cache-dir h5py
