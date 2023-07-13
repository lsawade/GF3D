.. title:: GF Extraction

.. module:: gf3d.seismograms

.. _green-function-extraction:

3D Green Function Extraction
-----------------------------

This page goes over some examples on how to extract seismograms from the
database as well as from subset.

The examples at the bottom show advanced use cases including some plotting
tools. The two main tools that you are going to need are the python ``Class``
:class:`gf3d.seismograms.GFManager`, which loads the subset file into memory,
and subsequently can be queried, for cmt locations. The use of the
:class:`gf3d.seismograms.GFManager` is extensively demonstrated in the gallery
below.

Command Line Interface (CLI)
============================

We implemented a command line interface to interface with the database, query
from a hosted interface, and extract seismograms from generated subsets.
Make sure that your environment is activated, and gf3d is installed.

Run the CLI like so:

.. code:: bash

  gf3d COMMAND [SUBCOMMAND [SUBSUBCOMMAND] ...] [OPTIONS] ARG1 ARG2

Maybe start with ``gf3d --help``. Which should output:

.. code:: bash

    Usage: gf3d [OPTIONS] COMMAND [ARGS]...

    Options:
      --help  Show this message and exit.

    Commands:
      database  Interface to a GF3D database
      subset    Interface to generated subset.

As you can see at the bottom of the output, there are sub commands. These
indicate a call structure as follows:

.. code:: bash

    gf3d
    - database
        - query
            - info
            - stations
            - subset
            - extract [NOT IMPLEMENTED]
        - extract [NOT IMPLEMENTED]
    - subset
        - info
        - stations
        - extract

So, if you want to query a subset and want to know how to make the query

.. code:: bash

    gf3d database query subset --help

which would print

.. code:: bash

    Usage: gf3d database query subset [OPTIONS] DATABASENAME SUBSETFILENAME
                                      LATITUDE LONGITUDE DEPTH_IN_KM RADIUS_IN_KM

      Query a subset from a hosted database server.

      IMPORTANT: For negative latitudes and longitudes use following setup:

          gf3d query subset [--option = value] -- DATABASENAME SUBSETFILENAME
          LATITUDE ...

    Options:
      --fortran BOOLEAN  number of greetings
      --ngll INTEGER     Number of GLL points 5 or 3
      --netsta TEXT      Station subselection. NOT IMPLEMENTED
      --help             Show this message and exit.

A normal query for a subset from a hosted database would look like this

.. code:: bash

    gf3d database query subset -- princeton testquery.h5 -31.1300 -72.0900 17.3500 28.0

.. warning::

    The ``--`` is important for entering negative numbers. It's hard for CLI to
    distinguish ``-option`` and a negative number. Hence, ``click`` implements the ``--`` which tells the CLI to take all following command line arguments are, in fact, arguments and not options. For details, please visit: `click: option-like-arguments <https://click.palletsprojects.com/en/8.1.x/arguments/#option-like-arguments>`_.



OLD command line tool
=====================

This is only here for posterity please use the tool above.
The other tool that is immensely powerful is the command-line tool to query a
specific subset file for a specific moment tensor. The command-line tool
is automatically installed when you install ``GF3D`` and can be called using
standard ``CMTSOLUTION`` parameters. The standard call signature is

::

    gf-get-synthetics <subsetfile> \
        <yyyy> <mm> <dd> <HH> <MM> <SS.SSSS> \
        <Mrr> <Mtt> <Mpp> <Mrt> <Mrp> <Mtp> \
        <latitude> <longitude> <depth> <tshift> <hdur> \
        <tfir> <spsam> <it1smp> <it2smp> \
        <pminswrays> <pmaxswrays> \
        <itypsokern> \
        <outdir>

The call signature is quite long, but every parameter is essential. The parameters are as follows:

::

    subset_file = file containing Green functions of a subset of elements
    yyyy        = origin year
    mm          = origin month
    dd          = origin day
    HH          = origin hour
    MM          = origin minute
    SS.S        = origin seconds
    Mrr         = Moment tensor element  [dyn * cm]
    Mtt         = ---//---
    Mpp         = ---//---
    Mrt         = ---//---
    Mrp         = ---//---
    Mtp         = ---//---
    latitude    = centroid latitude      [deg]
    longitude   = centroid longitude     [deg]
    depth       = centroid depth         [km]
    tshift      = centroid time shift    [s]
    hdur        = centroid half_duration [s]
    tfir        = start time of first sample w.r.t. origin time
    spsam       = seconds per sample (1/sample rate)
    it1smp      = index of first sample wanted [compute starttime for interpolation]
    it2smp      = index of last sample wanted  [compute npts]
    pminswrays  = minimum period wanted [band-pass]
    pmaxswrays  = maximum period wanted
    itypsokern  = code for type of kernels desired (see below)
    outdir      = directory to write synthetics to

Most of the parameters are self-explanatory with the description. If

- ``itypsokern=0``, the subroutine only computes the synthetic seismogram
- ``itypsokern=1``, the subroutine computes the synthetic seismogram, and the 6
  moment-tensor kernels
- ``itypsokern=2``, the subroutine computes the synthetic seismogram in synt, 6
  moment-tensor kernels and 4 centroid kernels (latitude, longitude, depth,
  tshift)

The synthetic seismograms are then stored in ``<outdir>``. The synthetics traces
are stored in ``<outdir>/synt``, and the kernels are stored in the
``<outdir>/[par]``


Examples
++++++++

.. toctree::

    ../../examples/extraction/database/index.rst
    ../../examples/extraction/subset/index.rst
    ../../examples/client/index.rst


