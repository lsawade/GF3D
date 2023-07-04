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


