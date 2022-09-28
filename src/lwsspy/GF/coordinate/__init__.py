def xyz_2_latlon_minmax(nspec, nglob, ibool, xstore, ystore, zstore, &
                                 lat_min, lat_max, lon_min, lon_max)

"""
! returns minimum and maximum values of latitude/longitude of given mesh points
! latitude in degree between[-90, 90], longitude in degree between[0, 360]

  use constants, only: CUSTOM_REAL, NGLLX, NGLLY, NGLLZ, HUGEVAL

  implicit none

  integer, intent(in ) : : nspec, nglob

  integer, intent(in ) : : ibool(NGLLX, NGLLY, NGLLZ, nspec)

  ! arrays containing coordinates of the points
  real(kind=CUSTOM_REAL), dimension(nglob), intent( in ) : : xstore, ystore, zstore

  double precision, intent(out): : lat_min, lat_max, lon_min, lon_max

  ! local parameters
  double precision: : x, y, z
  double precision: : r, lat, lon
  !double precision:: r_min, r_max
  integer: : ispec, i, j, k, iglob
"""

  # initializes
  lat_min = HUGEVAL
  lat_max = -HUGEVAL
  lon_min = HUGEVAL
  lon_max = -HUGEVAL
  # r_min = HUGEVAL
  # r_max = -HUGEVAL

  # loops over all elements
  do ispec = 1, nspec

  # loops only over corners
   do k = 1, NGLLZ, NGLLZ-1
     do j = 1, NGLLY, NGLLY-1
       do i = 1, NGLLX, NGLLX-1

          gets x/y/z coordinates
          iglob = ibool(i, j, k, ispec)
          x = xstore(iglob)
          y = ystore(iglob)
          z = zstore(iglob)

          ! converts geocentric coordinates x/y/z to geographic radius/latitude/longitude (in degrees)
          call xyz_2_rlatlon_dble(x, y, z, r, lat, lon)

          ! stores min/max
          if (lat < lat_min) lat_min = lat
          if (lat > lat_max) lat_max = lat

          if (lon < lon_min) lon_min = lon
          if (lon > lon_max) lon_max = lon

          !if (r < r_min) r_min = r
          !if (r > r_max) r_max = r
        enddo
      enddo
    enddo
  enddo

  ! limits latitude to[-90.0, 90.0]
  if (lat_min < -90.d0) lat_min = -90.d0
  if (lat_max > 90.d0) lat_max = 90.d0

  ! limits longitude to[0.0, 360.0]
  if (lon_min < 0.d0) lon_min = lon_min + 360.d0
  if (lon_min > 360.d0) lon_min = lon_min - 360.d0
  if (lon_max < 0.d0) lon_max = lon_max + 360.d0
  if (lon_max > 360.d0) lon_max = lon_max - 360.d0

  end subroutine xyz_2_latlon_minmax
