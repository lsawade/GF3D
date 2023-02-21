:orphan:

.. title:: GF3D Documentation

.. module:: gf-index

.. GF3D documentation master file, created by
   sphinx-quickstart on Mon Feb 20 15:43:45 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

########################################
Green Functions 3D Generator & Extractor
########################################

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tabs/gf-generation
   tabs/gf-extraction

*************
Quick-Install
*************

This is only interesting if you want to extract from an existing database.
for the generation of a database you will need to manually install
``specfem3d_globe``, ``Parallel HDF5`` and ``ADIOS``.


.. card:: Install only

    **Pypi**

    .. code-block:: bash

        pip install gf3d

    **Conda** (Coming soon):

    .. code-block:: bash

        conda install -c conda-forge gf3d

.. card:: From ``environment.yml``

    Download github repo

    .. code-block:: bash

        git clone https://github.com/lsawade/GF3D
        cd GF3D
        conda env create -n gf3d -f environment.yml


Installation

.. grid:: 1 1 2 2

    .. grid-item::

        Make environment

        .. code-block:: bash

            conda create -n gf "python=3.10" numpy scipy matplotlib h5py

        activate

        .. code-block:: bash

            conda activate gf

        and install

        .. code-block:: bash

            cd path/to/GF3D
            pip install -e .


    .. grid-item::

        Install using `pip <https://pypi.org/project/matplotlib>`__:

        .. code-block:: bash

            pip install gf3d

    .. grid-item::

        COMING SOON! Install using `conda <https://docs.continuum.io/anaconda/>`__:

        .. code-block:: bash

            conda install -c conda-forge gf3d

.. Further details are available in the :doc:`Installation Guide <users/installing/index>`.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
