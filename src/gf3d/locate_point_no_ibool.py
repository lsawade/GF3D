"""
Locate point within element mesh without using ibool array.

This version is designed for element-major database format where each element
has its own local coordinate array rather than a global ibool mapping.
"""

from scipy.spatial import KDTree
import numpy as np
from .constants import R_PLANET_KM, HUGEVAL
from .constants_solver import NGNOD, NUM_ITER
from .lagrange import gll_nodes
from .hex_nodes import hex_nodes_anchor_ijk
from .transformations.recompute_jacobian import recompute_jacobian
from .logger import logger


def locate_point_no_ibool(
        x_target, y_target, z_target,
        xyz_element,  # (NGLL, NGLL, NGLL, 3) - coordinates of the element
        POINT_CAN_BE_BURIED: bool = True,
        NGLL: int = 5):
    """
    Locate a point within a single element and compute local coordinates.
    
    This version does not use ibool and works directly with element coordinates.
    
    Parameters
    ----------
    x_target, y_target, z_target : float
        Target point in Cartesian coordinates
    xyz_element : ndarray
        Element coordinates with shape (NGLL, NGLL, NGLL, 3)
    POINT_CAN_BE_BURIED : bool
        If False, force gamma=1.0 (surface)
    NGLL : int
        Number of GLL points per dimension
    
    Returns
    -------
    xi, eta, gamma : float
        Local coordinates within element [-1, 1]
    x, y, z : float
        Final Cartesian coordinates
    xix, xiy, xiz : float
        Jacobian derivatives dx/dxi, dy/dxi, dz/dxi
    etax, etay, etaz : float
        Jacobian derivatives dx/deta, dy/deta, dz/deta
    gammax, gammay, gammaz : float
        Jacobian derivatives dx/dgamma, dy/dgamma, dz/dgamma
    distmin : float
        Distance to target in km
    """
    
    # Extract coordinate arrays
    x_store = xyz_element[:, :, :, 0]
    y_store = xyz_element[:, :, :, 1]
    z_store = xyz_element[:, :, :, 2]
    
    # Find initial guess by searching for closest GLL point
    distmin_squared = HUGEVAL
    ix_initial_guess = NGLL // 2
    iy_initial_guess = NGLL // 2
    iz_initial_guess = NGLL // 2
    
    for k in range(NGLL):
        for j in range(NGLL):
            for i in range(NGLL):
                dist_squared = (
                    (x_target - x_store[i, j, k])**2 +
                    (y_target - y_store[i, j, k])**2 +
                    (z_target - z_store[i, j, k])**2
                )
                
                if dist_squared < distmin_squared:
                    distmin_squared = dist_squared
                    ix_initial_guess = i
                    iy_initial_guess = j
                    iz_initial_guess = k
    
    # Get anchor nodes
    anchor_iax, anchor_iay, anchor_iaz = hex_nodes_anchor_ijk(NGLL)
    
    # Define coordinates of the control points of the element
    xelm = np.zeros(NGNOD)
    yelm = np.zeros(NGNOD)
    zelm = np.zeros(NGNOD)
    
    for ia in range(NGNOD):
        xelm[ia] = x_store[anchor_iax[ia], anchor_iay[ia], anchor_iaz[ia]]
        yelm[ia] = y_store[anchor_iax[ia], anchor_iay[ia], anchor_iaz[ia]]
        zelm[ia] = z_store[anchor_iax[ia], anchor_iay[ia], anchor_iaz[ia]]
    
    # GLL points
    xigll, _, _ = gll_nodes(NGLL - 1)
    etagll, _, _ = gll_nodes(NGLL - 1)
    gammagll, _, _ = gll_nodes(NGLL - 1)
    
    # Initial guess in xi, eta, gamma
    xi = xigll[ix_initial_guess]
    eta = etagll[iy_initial_guess]
    gamma = gammagll[iz_initial_guess]
    
    # Impose receiver exactly at the surface
    if not POINT_CAN_BE_BURIED:
        gamma = 1.0
    
    d_min_sq = HUGEVAL
    
    # Iterate to solve the non-linear system
    logger.debug(f"START: target=({x_target}, {y_target}, {z_target}), local=({xi}, {eta}, {gamma})")
    
    for iter_loop in range(NUM_ITER):
        # Recompute Jacobian for the new point
        x, y, z, xix, xiy, xiz, etax, etay, etaz, gammax, gammay, gammaz = \
            recompute_jacobian(xelm, yelm, zelm, xi, eta, gamma)
        
        # Compute distance to target location
        dx = -(x - x_target)
        dy = -(y - y_target)
        dz = -(z - z_target)
        
        # Compute increments
        if (dx**2 + dy**2 + dz**2) < d_min_sq:
            d_min_sq = dx**2 + dy**2 + dz**2
            
            dxi = xix * dx + xiy * dy + xiz * dz
            deta = etax * dx + etay * dy + etaz * dz
            dgamma = gammax * dx + gammay * dy + gammaz * dz
        else:
            # New position is worse than old one, no change necessary
            dxi = 0.0
            deta = 0.0
            dgamma = 0.0
        
        # Decrease step length if step is large
        if (dxi**2 + deta**2 + dgamma**2) > 1.0:
            dxi = dxi * 0.333333333330
            deta = deta * 0.333333333330
            dgamma = dgamma * 0.333333333330
        
        # Update values
        xi = xi + dxi
        eta = eta + deta
        gamma = gamma + dgamma
        
        logger.debug(f"  iter {iter_loop}: xyz=({x}, {y}, {z}), local=({xi}, {eta}, {gamma})")
        
        # Impose that we stay in that element
        xi = np.clip(xi, -1.100, 1.100)
        eta = np.clip(eta, -1.100, 1.100)
        gamma = np.clip(gamma, -1.100, 1.100)
    
    # Impose receiver exactly at the surface
    if not POINT_CAN_BE_BURIED:
        gamma = 1.0
    
    # Compute final coordinates of point found
    x, y, z, xix, xiy, xiz, etax, etay, etaz, gammax, gammay, gammaz = \
        recompute_jacobian(xelm, yelm, zelm, xi, eta, gamma)
    
    # Compute final distance between asked and found (converted to km)
    distmin = np.sqrt(
        (x_target - x)**2 + (y_target - y)**2 + (z_target - z)**2
    ) * R_PLANET_KM
    
    return xi, eta, gamma, x, y, z, xix, xiy, xiz, etax, etay, etaz, gammax, gammay, gammaz, distmin


def find_element_for_source(
        x_target, y_target, z_target,
        all_midpoints,
        kdtree: KDTree | None = None):
    """
    Find which element contains a source point.
    
    Parameters
    ----------
    x_target, y_target, z_target : float
        Target point in Cartesian coordinates
    all_midpoints : ndarray
        Midpoint coordinates for all elements, shape (NSPEC, 3)
    kdtree : KDTree, optional
        Pre-computed KDTree for midpoints
    
    Returns
    -------
    ispec_selected : int
        Index of the element containing the point
    """
    
    if not isinstance(kdtree, KDTree):
        kdtree = KDTree(all_midpoints)
    
    # Find nearest element
    point_target = np.array([x_target, y_target, z_target])
    dist, ispec_selected = kdtree.query(point_target, k=1)
    
    logger.debug(f'KDTree found nearest element: {ispec_selected} (distance: {dist * R_PLANET_KM:.3f} km)')
    
    return ispec_selected
