:orphan:

.. title:: GF3D Documentation

.. module:: gf-index

.. GF3D documentation master file, created by
   sphinx-quickstart on Mon Feb 20 15:43:45 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Green Functions 3D Generator & Extractor
########################################

``GF3D`` is a library of function to create and query station-specific Green
Functions on the basis of `specfem3d_globe`_.

.. toctree::
    :hidden:

    parts/gf-generation/index
    parts/gf-extraction/index


Quick-Install
*************

.. card:: From Github & ``environment.yml``

    Current way to go!

    .. code-block:: bash

        git clone https://github.com/lsawade/GF3D
        cd GF3D
        conda env create -n gf3d -f environment.yml


.. card:: Package Managers (Coming soon)

    .. grid:: 1 1 2 2

        .. grid-item::

            **Pypi**

            .. code-block:: bash

                pip install gf3d

        .. grid-item::

            **Conda**:

            .. code-block:: bash

                conda install -c conda-forge gf3d



Custom Installation for Database Creation
*****************************************

This is only interesting if you want to extract from an existing database.
for the generation of a database you will need to manually install
``specfem3d_globe``, ``Parallel HDF5`` and ``ADIOS``. See
:ref:`custom-installation`.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _specfem3d_globe: https://github.com/SPECFEM/specfem3d_globe