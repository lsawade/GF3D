#!/usr/bin/env python
"""
Find Deep Elements in Element-Major Database

This script identifies all elements below a specified depth threshold,
accounting for WGS84 ellipticity.

Author: Lucas Sawade
Date: January 2026
"""
import sys
import h5py
import numpy as np
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_element_coordinates(metadata_file: str) -> np.ndarray:
    """Load element coordinates from metadata file
    
    Parameters
    ----------
    metadata_file : str
        Path to metadata.hdf5
    
    Returns
    -------
    xyz : ndarray
        Array of shape (NSPEC, NGLLX, NGLLY, NGLLZ, 3) with element coordinates
    """
    with h5py.File(metadata_file, 'r') as f:
        xyz = f['xyz'][:]
        nspec = f['NSPEC'][()]
        ngllx = f['NGLLX'][()]
        nglly = f['NGLLY'][()]
        ngllz = f['NGLLZ'][()]
    
    print(f"Loaded element coordinates:")
    print(f"  Elements: {nspec}")
    print(f"  NGLL: {ngllx}x{nglly}x{ngllz}")
    
    return xyz


def calculate_element_depths(xyz: np.ndarray) -> tuple:
    """Calculate depth for all elements accounting for WGS84 ellipticity
    
    Parameters
    ----------
    xyz : ndarray
        Element coordinates array (NSPEC, NGLLX, NGLLY, NGLLZ, 3)
    
    Returns
    -------
    depths : ndarray
        Depth in km for each element (NSPEC,)
    latitudes : ndarray
        Latitude in degrees for each element (NSPEC,)
    longitudes : ndarray
        Longitude in degrees for each element (NSPEC,)
    """
    nspec = xyz.shape[0]
    ngllx, nglly, ngllz = xyz.shape[1:4]
    
    # Get midpoint indices
    mid_i = ngllx // 2
    mid_j = nglly // 2
    mid_k = ngllz // 2
    
    # Extract midpoints for all elements
    midpoints = xyz[:, mid_i, mid_j, mid_k, :]  # (NSPEC, 3)
    
    # WGS84 ellipsoid parameters
    a_wgs84 = 6378137.0  # Semi-major axis (equatorial radius) in meters
    b_wgs84 = 6356752.314245  # Semi-minor axis (polar radius) in meters
    f_wgs84 = (a_wgs84 - b_wgs84) / a_wgs84  # Flattening
    
    # Calculate radius from center for all elements
    r = np.sqrt(np.sum(midpoints**2, axis=1))  # (NSPEC,)
    
    # Calculate latitude and longitude
    latitudes = np.degrees(np.arcsin(midpoints[:, 2] / r))
    longitudes = np.degrees(np.arctan2(midpoints[:, 1], midpoints[:, 0]))
    
    # Calculate ellipsoidal radius at each latitude
    lat_rad = np.radians(latitudes)
    R_ellipsoid = a_wgs84 * (1.0 - f_wgs84 * np.sin(lat_rad)**2)
    
    # Calculate depths accounting for ellipticity
    # r is normalized in SPECFEM, so depth = (1 - r) * R_ellipsoid
    depths = (1.0 - r) * R_ellipsoid / 1000.0  # Convert to km
    
    return depths, latitudes, longitudes


def find_deep_elements(
    metadata_file: str,
    depth_threshold: float = 280.0,
    output_file: str = None
) -> np.ndarray:
    """Find all elements below a specified depth threshold
    
    Parameters
    ----------
    metadata_file : str
        Path to metadata.hdf5 file
    depth_threshold : float
        Depth threshold in km (default: 280.0)
    output_file : str, optional
        Path to save element indices as text file
    
    Returns
    -------
    deep_elements : ndarray
        Array of element indices that are deeper than threshold
    """
    print("="*80)
    print("Finding Deep Elements")
    print("="*80)
    
    # Load element coordinates
    print(f"\nLoading element coordinates from: {metadata_file}")
    xyz = load_element_coordinates(metadata_file)
    nspec = xyz.shape[0]
    
    # Calculate depths with ellipticity correction
    print(f"\nCalculating depths for {nspec} elements...")
    depths, latitudes, longitudes = calculate_element_depths(xyz)
    
    # Find elements below threshold
    deep_mask = depths >= depth_threshold
    deep_elements = np.where(deep_mask)[0]
    
    # Statistics
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Depth threshold: {depth_threshold:.1f} km")
    print(f"Total elements: {nspec}")
    print(f"Deep elements (>= threshold): {len(deep_elements)} ({100*len(deep_elements)/nspec:.2f}%)")
    print(f"\nDepth statistics (all elements):")
    print(f"  Min depth:  {np.min(depths):.2f} km")
    print(f"  Max depth:  {np.max(depths):.2f} km")
    print(f"  Mean depth: {np.mean(depths):.2f} km")
    
    if len(deep_elements) > 0:
        print(f"\nDepth statistics (deep elements only):")
        deep_depths = depths[deep_elements]
        print(f"  Min depth:  {np.min(deep_depths):.2f} km")
        print(f"  Max depth:  {np.max(deep_depths):.2f} km")
        print(f"  Mean depth: {np.mean(deep_depths):.2f} km")
        
        print(f"\nSample of deep elements:")
        n_samples = min(10, len(deep_elements))
        for i in range(n_samples):
            idx = deep_elements[i]
            print(f"  Element {idx:6d}: depth={depths[idx]:6.2f} km, "
                  f"lat={latitudes[idx]:7.2f}°, lon={longitudes[idx]:7.2f}°")
        if len(deep_elements) > n_samples:
            print(f"  ... and {len(deep_elements) - n_samples} more")
    
    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_file, deep_elements, fmt='%d')
        print(f"\nSaved {len(deep_elements)} element indices to: {output_file}")
    
    print(f"{'='*80}")
    
    return deep_elements


def main():
    parser = argparse.ArgumentParser(
        description='Find elements below specified depth threshold',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('metadata_file', type=str,
                       help='Path to metadata.hdf5 file')
    parser.add_argument('--depth', type=float, default=280.0,
                       help='Depth threshold in km (default: 280.0)')
    parser.add_argument('--output', type=str,
                       help='Output file to save element indices (optional)')
    
    args = parser.parse_args()
    
    deep_elements = find_deep_elements(
        args.metadata_file,
        args.depth,
        args.output
    )
    
    sys.exit(0)


if __name__ == '__main__':
    main()
