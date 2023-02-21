# 3D Green Function generation and extraction software.

## 'Quick'-Install

***For extraction only***:

Make environment
```bash
conda create -n gf "python=3.10" numpy scipy matplotlib h5py
```
activate
```bash
conda activate gf
```
and install
```bash
cd path/to/GF3D
pip install -e .
```

***For generation***:

For post-processing the following pacakges need to be installed by hand after
`numpy`, `scipy` and `matplotlib` have been installed:


### `mpi4py`


```bash
export MPICC=$(which mpicc)
python -m pip install mpi4py
```

### `h5py`

you need to compile HDF5 with parallel i/o support.

```bash
# Summit
export HDF5_DIR=/gpfs/alpine/geo111/scratch/lsawade/SGT/SpecfemMagic/packages/hdf5/build/

# Traverse
export HDF5_DIR=/scratch/gpfs/lsawade/SpecfemMagic/packages/hdf5/build/

CC="mpicc" HDF5_MPI="ON" HDF5_DIR=${HDF5_DIR} pip install --no-binary=h5py --no-cache-dir h5py
```

### `adios2`

You need to compile adios as well for reading the Specfem output files.

conda develop `path/to/adios-install-dir/lib/python3.10/site-packages/adios2`