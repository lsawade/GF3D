# `nnodes` workflow for the creation of an SGT database

This directory is used to create an SGT database without hassle. Inputs are

* `STATIONS` file like in specfem --> One simulation per file
* `GF_LOCATIONS` file
* `reciprocal.toml` which describes the database setup including location for
  the database. Database strucutre is as follows

  ```
  path/to/database
  |--- reciprocal.toml
  |--- STATIONS
  |--- GF_LOCATIONS
  |--- Networks/
       |--- II/
            |--- BFO.h5
            |--- ...

       |--- IU/
            |--- HRV.h5
            |--- ...

  ```
