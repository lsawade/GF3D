# Infrastructure Notes

Quite a few options to be made/were added to [``specfem3d_globe``][sf3d].

Main additions:
---
* ***Saving specific elements***:
    * optional file GF_LOCATIONS
        * this file contains locations of source locations (lat, lon, dep)
        * the file is used to pinpoint elements these locations
    * added file ``locate_green_locations.f90``
        * ended up being a mix between ``locate_receivers.f90`` and
          ``locate_sources.  f90``,
        * ``locate_green_locations.f90`` also creates readdressing of the
          elements and GLL points so that not all GLL points have to be saved at
          writing and can be queried as such.
    * added ADIOS writing routine ``save_forward_arrays_GF()`` based on the
      original ``save_forward_arrays_undo_attenuation()`` routines.
---
* ***Subsampling capabilities***:
    * using the parameter ``NTSTEP_BETWEEN_OUTPUT_WAVEFIELD``, the routine
      ``save_forward_arrays_GF()`` can be called at every x-th time step which
      is specified in the ``Par_file``; the default is ``1``, but that makes
      little sense due to storage
    * The x-th time step should be controlled hand-in-hand with the parameters
      ``T0`` and ``NSTEP`` and be based on the approximate resolve period in the
      by the mesh
    * There are a couple of new python routines that control ``T0``, ``NSTEP``,
      ``NSTEP_BETWEEN_OUTPUT_WAVEFIELD`` together and create a source time
      function that is low-pass filtered. This way we ensure that there is no
      aliasing
        * We filter the source time function using a Bessel filter to avoid ringing
          compared to e.g. Butter/Chebychev filters.
        * So, depending on the given ``DT`` a half-duration is chosen for the
          approximate Gaussian moment-rate function but then we low-pass/anti-aliasing filter the given source time function and create a
          new one.

Most of the things above are controllable Using the ``Simulation`` and the
``SimulationManager``


[sf3d]: https://specfem3d-globe.readthedocs.io/en/latest/