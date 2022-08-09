"""
Script that downloads and install ADIOS 2 from source using the MPI compiler
in the current environment.

There are a couple of requirements to make is work seemlessly. You need both
numpy (duh), but also openmpi compilers. Now usually these are preinstalled
by yourself. But we want to make it as easy as possible, and for macOS there
are mpi compilers available. So, on macOS:

# First,
# Install NumPy as needed for ADIOS
$ conda install -c conda-forge numpy

# Then,
# Installing mpi4py and MPI compilers using
$ conda install -c conda-forge openmpi-mpicc openmpi-mpicxx
$ conda install -c conda-forge mpi4py openmpi

"""
import os
from posixpath import islink
import sys
from shutil import which
from subprocess import check_call
from glob import glob


# Check whether cmake is available
if which('cmake3') is None:
    print('cmake is required to install ADIOS 2.')
    print('Please install before proceeding')
    sys.exit()

# Set reconfigure
if len(sys.argv) > 1:
    if sys.argv[1] == "0":
        reconfigure = True
        reinstall = False
    elif sys.argv[1] == "1":
        reconfigure = True
        reinstall = True
    elif sys.argv[1] == "B":
        reconfigure = False
        reinstall = False
    else:
        print(f"{sys.argv[1]} as command line option not supported")
        sys.exit()
else:
    reconfigure = True
    reinstall = False

# Set variables for download and compilation
# Root is at path/to/lwsspy.GF/dependencies/
ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'dependencies')
ADIOS_DIR = os.path.join(ROOT, 'adios')
ADIOS_DOWNLOAD = os.path.join(ADIOS_DIR, 'adios-download')
ADIOS_BUILD = os.path.join(ADIOS_DIR, 'adios-build')
ADIOS_INSTALL = os.path.join(ADIOS_DIR, 'adios-install')
ADIOS_GIT = "https://github.com/ornladios/ADIOS2.git"

# Create ADIOS directory if it doesnt exist
os.makedirs(ADIOS_DIR) if os.path.exists(ADIOS_DIR) is False else None

# Go into ADIOS
os.chdir(ADIOS_DIR)

# Download git directory if it doesn't exist or needs to be dowloaded
if os.path.exists(ADIOS_DOWNLOAD) is False or reinstall:
    # If also redownloading the package
    if reinstall:
        check_call(f'rm -rf {ADIOS_DOWNLOAD}', shell=True)
    check_call(f'git clone {ADIOS_GIT} {ADIOS_DOWNLOAD}', shell=True)
# sys.exit()
# If you just want to reconfigure just remove dowload and
if os.path.exists(ADIOS_DOWNLOAD) and reconfigure:
    check_call(f'rm -rf {ADIOS_BUILD}', shell=True)
    check_call(f'rm -rf {ADIOS_INSTALL}', shell=True)


# Create ADIOS build  if it doesnt exist
os.makedirs(ADIOS_BUILD) if os.path.exists(ADIOS_BUILD) is False else None

# Enter build directory
os.chdir(ADIOS_BUILD)

# Set environment compilers
os.environ['CC'] = str(which('mpicc'))
os.environ['CXX'] = str(which('mpicxx'))
os.environ['MPICC'] = str(which('mpicc'))

PYTHON=os.path.abspath(str(which("python3")))
PYTHON_INCLUDE_DIRS=os.path.join(os.path.dirname(os.path.dirname(PYTHON)),'include')
PYTHON_LIB=os.path.join(os.path.dirname(os.path.dirname(PYTHON)),'lib')
print(70*"=")
print("PYTHON:        ", PYTHON)
print("PYTHON INCLUDE:", PYTHON_INCLUDE_DIRS)
print("PYTHON LIBRARY:", PYTHON_LIB)
print(70*"=")

# Get python executable
# The capitalization of the '-DPython_EXECUTABLE={PYTHON} ' is important!!!
configuration_call = (
    f'cmake3 -DCMAKE_INSTALL_PREFIX={ADIOS_INSTALL} '
    '-DADIOS2_USE_MPI=OFF '
    '-DADIOS2_USE_Fortran=ON '
    '-DADIOS2_USE_HDF5=OFF '
    '-DADIOS2_USE_Python=ON '
    '-DADIOS2_USE_FFS=OFF '
    '-DBUILD_SHARED_LIBS=OFF '
    f'-DPython_EXECUTABLE={PYTHON} '
    f'../{os.path.basename(ADIOS_DOWNLOAD)}'
)

if reconfigure:

    # Configure
    check_call(configuration_call, shell=True)

    # Make -j <N> does not work in subprocess.check_call
    check_call('make -j 6', shell=True)

    # Install
    check_call('make install', shell=True)


# This below is very hacky..., bu it works
# Link python bindings for python >=3.10
globstr = os.path.join(ADIOS_INSTALL, 'lib',
                       'python*.*', 'site-packages', 'adios2')
BINDINGS = glob(globstr)[0]

# Link source
SOURCE = BINDINGS

# Link target
PYTHON_VERSION = SOURCE.split('/')[-3]
TARGET = os.path.join(os.path.dirname(os.path.dirname(
    str(which('python')))), 'lib', PYTHON_VERSION, "site-packages", "adios2")

# Remove old link
if os.path.islink(TARGET):
    check_call(f'rm {TARGET}', shell=True)

# Add new link
check_call(f'ln -s {SOURCE} {TARGET}', shell=True)


# Link the libraries
LIBDIR = os.path.join(ADIOS_INSTALL, 'lib')
PYTHON_DIRECTORY = os.path.join(os.path.dirname(
    os.path.dirname(str(which('python')))), 'lib')
for _file in os.listdir(LIBDIR):
    if _file == 'cmake':
        continue
    elif "python" in _file:
        continue

    TARGET = os.path.join(PYTHON_DIRECTORY, _file)
    SOURCE = os.path.join(LIBDIR, _file)
    # Remove old link
    if os.path.islink(TARGET):
        check_call(f'rm {TARGET}', shell=True)

    # Add new link
    check_call(f'ln -s {SOURCE} {TARGET}', shell=True)
