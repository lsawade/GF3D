
.. _custom-installation:

Custom Installation for Database Creation
=========================================

There are a requirements to create a Database

- A conda environment with working ``mpi4py``
- An installation of parallel ``ADIOS`` to store Green functions in ADIOS format
- An installation of parallel ``HDF5`` to post process the ADIOS files into
- An installation of this package (``GF3D``) installed using the parallel
  versions of ``HDF5``, ``h5py`` and ``ADIOS``.
- A working directory of ``specfem3d_globe`` (including your model) that was
  compiled with paralle ADIOS from above.

Most of the steps are explained below, but not everything is a one fits all.
Especially the installation of parallel `HDF5` can sometimes be ... annoying.

.. toctree::

    creating-environment
    mpi4py
    hdf5
    h5py
    adios


