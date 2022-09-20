import numpy as np
from scipy.spatial import KDTree
from .constants import R_PLANET_KM, HUGEVAL
from .constants_solver import NGLLX, NGLLY, NGLLZ, MIDX, MIDY, MIDZ, DO_ADJACENT_SEARCH
from .transformations.rthetaphi_xyz import xyz_2_latlon_minmax
from .lagrange import gll_nodes
from .hex_nodes import hex_nodes_anchor_ijk
from .recompute_jacobian


def locate_point(
        x_target, y_target, z_target, lat_target, lon_target,
        midpoints, x_store, y_store, z_store, ibool,
        POINT_CAN_BE_BURIED, NSPEC,
        USE_DISTANCE_CRITERION: bool = False,
        kdtree: KDTree | None):
    """
      use constants_solver, only: &
        NGLLX, NGLLY, NGLLZ, MIDX, MIDY, MIDZ, HUGEVAL, &
        USE_DISTANCE_CRITERION

        use shared_parameters, only: R_PLANET_KM

        use specfem_par, only: &
            nspec = > NSPEC_CRUST_MANTLE

        use specfem_par_crustmantle, only: &
            ibool = > ibool_crust_mantle, &
            xstore = > xstore_crust_mantle, ystore = > ystore_crust_mantle, zstore = > zstore_crust_mantle

    # for point search
    use specfem_par, only: &
       typical_size_squared, &
        lat_min, lat_max, lon_min, lon_max, xyz_midpoints

    use kdtree_search, only: kdtree_find_nearest_neighbor

# debug: sync
#  use specfem_par, only: NPROCTOT_VAL, myrank

    implicit none
    integer, intent(out):: ispec_selected
    double precision, intent(out):: xi, eta, gamma
    double precision, intent(out):: x, y, z
    double precision, intent(out):: distmin_not_squared

    logical, intent( in ) : : POINT_CAN_BE_BURIED

    # local parameters
    integer: : ix_initial_guess, iy_initial_guess, iz_initial_guess
    integer: : ispec, i, j, k, iglob

    double precision:: lat, lon
    double precision:: distmin_squared, dist_squared

    logical: : target_located

    # nodes search
    double precision, dimension(3):: xyz_target
    double precision: : dist_min
    integer: : inode_min

    # brute-force search for closest element
    # if set to .false. (default), a kd-tree search for initial guess element is used
    logical, parameter:: DO_BRUTE_FORCE_SEARCH = .false.

    # looks for closer estimates in neighbor elements if needed
    logical, parameter:: DO_ADJACENT_SEARCH = .true.

    # debug: sync
    #  integer:: iproc
    """

    # debug: sync
    #  do iproc = 0, NPROCTOT_VAL-1
    # if (iproc == myrank) then
    # print *, 'iproc ', iproc; flush(6)

    # set distance to huge initial value
    distmin_squared = HUGEVAL

    # initializes located target
    # if we have not located a target element, the receiver is not in this slice
    # therefore use first element only for fictitious iterative search
    ispec_selected = 1
    ix_initial_guess = MIDX
    iy_initial_guess = MIDY
    iz_initial_guess = MIDZ

    # limits latitude to[-90.0, 90.0]
    lat = lat_target
    lat = -90.0 if (lat < -90.0) else lat
    lat = 90.0 if (lat > 90.0) else lat

    # limits longitude to[0.0, 360.0]
    lon = lon_target
    lon = lon + 360.0 if (lon < 0.0) else lon
    lon = lon - 360.0 if (lon > 360.0) else lon

    lat_min, lat_max, lon_min, lon_max = xyz_2_latlon_minmax(x, y, z)

    # checks if receiver in this slice
    if (lat >= lat_min and lat <= lat_max and
            lon >= lon_min and lon <= lon_max):
        target_located = True
    else:
        target_located = False

    # debug
    # print *, 'target located:', target_located, 'lat', sngl(lat), sngl(lat_min), sngl(lat_max), 'lon', sngl(lon), sngl(lon_min), sngl(lon_max)

    if not isinstance(kdtree, KDTree):
        kdtree = KDTree(midpoints)

    if (target_located):

        # finds closest point(inside GLL points) in this chunk
        point_target = np.array([x_target, y_target, z_target])
        (dist, ispec_selected) = kdtree.query(point_target, k=1)

        # debug
        # print *, 'kd-tree found location :', inode_min

        # loops over GLL points in this element to get(i, j, k) for initial guess
        for k in range(1, NGLLZ-2):
            for j in range(1, NGLLY-2):
                for i in range(1, NGLLX-2):
                    iglob = ibool(i, j, k, ispec_selected)
                    dist_squared = (
                        (x_target - x_store[iglob])**2 +
                        (y_target - y_store[iglob])**2 +
                        (z_target - z_store[iglob])**2
                    )

                    # take this point if it is closer to the receiver
                    #  we compare squared distances instead of distances themselves to significantly speed up calculations
                    if (dist_squared < distmin_squared):
                        distmin_squared = dist_squared
                        ix_initial_guess = i
                        iy_initial_guess = j
                        iz_initial_guess = k

    # ****************************************
    # find the best(xi, eta, gamma)
    # ****************************************
    if (target_located):

        # deprecated, but might be useful if one wishes to get an exact GLL point location:
        #
        # # for point sources, the location will be exactly at a GLL point
        # # otherwise this tries to find best location
        #
        # if (USE_FORCE_POINT_SOURCE) then
        # # store xi, eta, gamma and x, y, z of point found
        # # note: they have range[1.00, NGLLX/Y/Z], used for point sources
        # # see e.g. in compute_add_sources.f90
        #        xi_subset(isource_in_this_subset) = dble(ix_initial_guess)
        #        eta_subset(isource_in_this_subset) = dble(iy_initial_guess)
        #        gamma_subset(isource_in_this_subset) = dble(iz_initial_guess)
        #
        #        iglob = ibool(ix_initial_guess, iy_initial_guess, &
        # iz_initial_guess, ispec_selected)
        #        xyz_found_subset(1, isource_in_this_subset) = xstore(iglob)
        #        xyz_found_subset(2, isource_in_this_subset) = ystore(iglob)
        #        xyz_found_subset(3, isource_in_this_subset) = zstore(iglob)
        #
        # # compute final distance between asked and found(converted to km)
        #        final_distance_subset(isource_in_this_subset) = &
        #          dsqrt((x_target-xyz_found_subset(1, isource_in_this_subset))**2 + &
        # (y_target-xyz_found_subset(2, isource_in_this_subset))**2 + &
        # (z_target-xyz_found_subset(3, isource_in_this_subset))**2)*R_PLANET/1000.0
        #      endif

        # gets xi/eta/gamma and corresponding x/y/z coordinates
        xi, eta, gamma, x, y, z = find_local_coordinates(
            x_target, y_target, z_target, ispec_selected,
            ix_initial_guess, iy_initial_guess, iz_initial_guess,
            POINT_CAN_BE_BURIED)

        # loops over neighbors and try to find better location
        if (DO_ADJACENT_SEARCH):
            # checks if position lies on an element boundary
            if (np.abs(xi) > 1.0990 or np.abs(eta) > 1.0990 or np.abs(gamma) > 1.0990):
                # searches for better position in neighboring elements
                # find_best_neighbor(x_target, y_target, z_target, xi, eta, gamma,
                #                    x, y, z, ispec_selected, distmin_squared, POINT_CAN_BE_BURIED)

                raise ValueError('Point found is outside element. and adjacent search is
                                 'not yet implmenented.')

    else:
        # point not found in this slice
        #
        # returns initial guess point
        xi = 0.0
        eta = 0.0
        gamma = 0.0

        iglob = ibool(ix_initial_guess, iy_initial_guess,
                      iz_initial_guess, ispec_selected)
        x = x_store[iglob]
        y = x_store[iglob]
        z = x_store[iglob]

    # compute final distance between asked and found (converted to km)
    distmin_not_squared = np.sqrt(
        (x_target-x)**2 + (y_target-y)**2 + (z_target-z)**2)*R_PLANET_KM

# debug: sync
#  endif # iproc
#  call synchronize_all()
#  enddo

    return ispec_selected, xi, eta, gamma, x, y, z, distmin_not_squared,

#
# -------------------------------------------------------------------------------------------------
#


def find_local_coordinates(
        x_target, y_target, z_target, ispec_selected,
        ix_initial_guess: int, iy_initial_guess: int, iz_initial_guess: int,
        x_store, y_store, z_store, ibool,
        POINT_CAN_BE_BURIED):
    """
    use constants_solver, only: &
        NGNOD, HUGEVAL, NUM_ITER

    use specfem_par_crustmantle, only: &
       ibool = > ibool_crust_mantle, &
        xstore = > xstore_crust_mantle,ystore = > ystore_crust_mantle,zstore => zstore_crust_mantle

    # for point search
    use specfem_par, only: &
        anchor_iax, anchor_iay, anchor_iaz, &
        xigll, yigll, zigll

    implicit none

    double precision,intent( in ) :: x_target,y_target,z_target
    double precision, intent(out) : : xi,eta,gamma
    double precision, intent(out) : : x,y,z

    integer,intent( in ) :: ispec_selected,ix_initial_guess,iy_initial_guess,iz_initial_guess

    logical,intent( in ) :: POINT_CAN_BE_BURIED

    # local parameters
    integer : : ia, iter_loop
    integer: : iglob

    # coordinates of the control points of the surface element
    double precision : : xelm(NGNOD), yelm(NGNOD),zelm(NGNOD)

    double precision : : dx, dy,dz,dx_min,dy_min,dz_min,d_min_sq
    double precision : : dxi, deta,dgamma
    double precision : : xix, xiy,xiz
    double precision : : etax, etay,etaz
    double precision : : gammax, gammay,gammaz
    """
    # Get anchors
    anchor_iax, anchor_iay, anchor_iaz = hex_nodes_anchor_ijk()

    # define coordinates of the control points of the element
    for ia in range(NGNOD):
        iglob = ibool(anchor_iax[ia], anchor_iay[ia],
                     anchor_iaz[ia], ispec_selected)
        xelm[ia] = x_store[iglob]
        yelm[ia] = y_store[iglob]
        zelm[ia] = z_store[iglob]

    # GLL points and weights (degree)
    xigll, wxi, _ = gll_nodes(NGLLX-1)
    etagll, weta, _ = gll_nodes(NGLLY-1)
    gammagll, wgamma, _ = gll_nodes(NGLL-1)

    # use initial guess in xi and eta
    xi = xigll[ix_initial_guess]
    eta = etagll[iy_initial_guess]
    gamma = gammagll[iz_initial_guess]

    # impose receiver exactly at the surface
    if (.not. POINT_CAN_BE_BURIED) gamma = 1.0

    d_min_sq = HUGEVAL
    dx_min = HUGEVAL
    dy_min = HUGEVAL
    dz_min = HUGEVAL

    # iterate to solve the non linear system
    for iter_loop in range(NUM_ITER):

        # recompute Jacobian for the new point
        recompute_jacobian(xelm, yelm, zelm, xi, eta,gamma,x,y,z, &
                                xix, xiy, xiz, etax, etay,etaz,gammax,gammay,gammaz)

        # compute distance to target location
        dx = - (x - x_target)
        dy = - (y - y_target)
        dz = - (z - z_target)

        # debug
        # print *,'  iter ',iter_loop,'dx',sngl(dx),sngl(dx_min),'dy',sngl(dy),sngl(dy_min),'dz',sngl(dz),sngl(dz_min),d_min_sq

        # compute increments
        if ((dx**2 + dy**2 + dz**2) < d_min_sq) then
          d_min_sq = dx**2 + dy**2 + dz**2
              dx_min = dx
                  dy_min = dy
                   dz_min = dz

                    dxi = xix*dx + xiy*dy + xiz*dz
                    deta = etax*dx + etay*dy + etaz*dz
                    dgamma = gammax*dx + gammay*dy + gammaz*dz
        else
          # new position is worse than old one, no change necessary
          dxi = 0.0
              deta = 0.0
                  dgamma = 0.0
        endif

        # decreases step length if step is large
        if ((dxi**2 + deta**2 + dgamma**2) > 1.00) then
          dxi = dxi * 0.333333333330
              deta = deta * 0.333333333330
                  dgamma = dgamma * 0.333333333330
        endif
        # alternative: impose limit on increments (seems to result in slightly less accurate locations)
        # if (np.abs(dxi) > 0.30 ) dxi = sign(1.00,dxi)*0.30
        # if (np.abs(deta) > 0.30 ) deta = sign(1.00,deta)*0.30
        # if (np.abs(dgamma) > 0.30 ) dgamma = sign(1.00,dgamma)*0.30

        # debug
        # print *,'  dxi/..',(dxi**2 + deta**2 + dgamma**2),dxi,deta,dgamma

        # update values
        xi = xi + dxi
        eta = eta + deta
        gamma = gamma + dgamma

        # impose that we stay in that element
        # (useful if user gives a receiver outside the mesh for instance)
        # we can go slightly outside the [1,1] segment since with finite elements
        # the polynomial solution is defined everywhere
        # can be useful for convergence of iterative scheme with distorted elements
        if (xi > 1.100) xi = 1.100
        if (xi < -1.100) xi = -1.100
        if (eta > 1.100) eta = 1.100
        if (eta < -1.100) eta = -1.100
        if (gamma > 1.100) gamma = 1.100
        if (gamma < -1.100) gamma = -1.100

    # end of non linear iterations
    enddo

    # impose receiver exactly at the surface
    if (.not. POINT_CAN_BE_BURIED) gamma = 1.0

    # compute final coordinates of point found
    call recompute_jacobian(xelm, yelm, zelm, xi, eta,gamma,x,y,z, &
                            xix, xiy, xiz, etax, etay,etaz,gammax,gammay,gammaz)

    return xi, eta, gamma, x,y,z
