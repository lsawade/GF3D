#!/usr/bin/env python

"""

USAGE:
------

    gf-get-synthetics <subsetfile> \\
        <yyyy> <mm> <dd> <HH> <MM> <SS.SSSS> \\
        <Mrr> <Mtt> <Mpp> <Mrt> <Mrp> <Mtp> \\
        <latitude> <longitude> <depth> <tshift> <hdur> \\
        <tfir> <spsam> <it1smp> <it2smp> \\
        <pminswrays> <pmaxswrays> \\
        <itypsokern> \\
        <outdir>

PARAMETERS:
-----------

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

OUTPUT:
-------

if itypsokern=0, the subroutine only returns the synthetic seismogram in synt
   itypsokern=1, the subroutine returns the synthetic seismogram in synt, and
the 6 moment-tensor kernels in pdarray
   itypsokern=2, the subroutine returns the synthetic seismogram in synt, the
6 moment-tensor kernels pdarray (index 1-6) and 4 centroid kernels in
pdarray (index 7-10)


:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2023.02.23 13.0


"""

from sys import argv, exit
from obspy import Stream


def process(st: Stream, bandpass, starttime, dt, npts):
    st.filter('bandpass', freqmin=1.0 /
              bandpass[1], freqmax=1.0/bandpass[0], zerophase=True)
    st.interpolate(sampling_rate=1/dt, method='weighted_average_slopes',
                   starttime=starttime, npts=npts)


def bin():

    # Check if argument is given
    if len(argv) != 27:

        print('')
        print(argv)
        print(len(argv))
        print('')
        print('Incorrect number of arguments check doc below!')
        print('==============================================')
        print(__doc__)
        exit()
    else:
        import os
        from obspy import UTCDateTime
        from gf3d.seismograms import GFManager
        from gf3d.source import CMTSOLUTION

        subsetfile = argv[1]
        yyyy = int(argv[2])
        mm = int(argv[3])
        dd = int(argv[4])
        HH = int(argv[5])
        MM = int(argv[6])
        SS = float(argv[7])
        Mrr = float(argv[8])
        Mtt = float(argv[9])
        Mpp = float(argv[10])
        Mrt = float(argv[11])
        Mrp = float(argv[12])
        Mtp = float(argv[13])
        latitude = float(argv[14])
        longitude = float(argv[15])
        depth = float(argv[16])
        time_shift = float(argv[17])
        hdur = float(argv[18])
        tfir = float(argv[19])
        spsam = float(argv[20])
        it1smp = int(argv[21])
        it2smp = int(argv[22])
        pminswrays = float(argv[23])
        pmaxswrays = float(argv[24])
        itypsokern = int(argv[25])
        outdir = str(argv[26])

        # Create CMT source
        cmt = CMTSOLUTION(
            origin_time=UTCDateTime(yyyy, mm, dd, HH, MM, SS),
            pde_lat=0.0, pde_lon=0.0, pde_depth=0.0, mb=0.0, ms=0.0,
            region_tag='n/a', eventname='n/a',
            time_shift=time_shift, hdur=hdur, latitude=latitude,
            longitude=longitude, depth=depth,
            Mrr=Mrr, Mtt=Mtt, Mpp=Mpp, Mrt=Mrt, Mrp=Mrp, Mtp=Mtp)

        # Copmute interpolation parameters
        npts = it2smp - it1smp
        starttime = cmt.origin_time + tfir
        dt = spsam

        # Bandpass
        bandpass = [pminswrays, pmaxswrays]

        GFM = GFManager(subsetfile)
        GFM.load()

        if itypsokern >= 0:
            # Get seismograms
            synt = GFM.get_seismograms(cmt)

            # Define synthetic output directory
            syntdir = os.path.join(outdir, 'synt')

            # Create synthetic output directory if it doesn't exist
            if os.path.exists(syntdir) == False:
                os.makedirs(syntdir)

            # Process synthetics
            process(synt, bandpass, starttime, dt, npts)

            for tr in synt:
                tr.write(os.path.join(syntdir,
                         tr.id + '.mseed'), format='MSEED')

        if itypsokern >= 1:
            pardict = GFM.get_frechet(cmt, rtype=itypsokern)

            for par, kern in pardict.items():

                # Define synthetic output directory
                pardir = os.path.join(outdir, f'{par}')

                # Create synthetic output directory if it doesn't exist
                if os.path.exists(pardir) == False:
                    os.makedirs(pardir)

                # Process Kernel
                process(kern, bandpass, starttime, dt, npts)

                for tr in kern:
                    tr.write(os.path.join(
                        pardir, tr.id + '.mseed'), format='MSEED')


if __name__ == "__main__":
    bin()
