#!/usr/bin/env python
"""
Validate Database Reorganization via Seismogram Comparison

This script validates that the reorganized element-major database produces
identical seismograms to the original station-major database.

Strategy:
1. Load metadata from element-major DB to get element coordinates
2. Reverse-engineer a source location from a test element's midpoint
3. Compute seismograms using old station-major DB (via GFManager)
4. Compute seismograms using new element-major DB (direct element access)
5. Compare traces to verify they match within numerical tolerance

Author: Lucas Sawade
Date: January 2026
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gf3d.seismograms import GFManager
from gf3d.source import CMTSOLUTION
from gf3d.lagrange import gll_nodes, lagrange_any
from gf3d.source2xyz import source2xyz
from gf3d.locate_point import locate_point
from gf3d.locate_point_no_ibool import locate_point_no_ibool, find_element_for_source
from gf3d.stf import create_stf
from gf3d.utils import next_power_of_2

from obspy import Stream, Trace
from obspy.core.trace import Stats
from obspy.core.util.attribdict import AttribDict
from scipy.spatial import KDTree
from scipy import fft
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


class ElementMajorReader:
    """Simple reader for element-major database format"""
    
    def __init__(self, db_dir: str, station_indices: Optional[List[int]] = None):
        """Initialize reader
        
        Parameters
        ----------
        db_dir : str
            Path to reorganized database directory
        station_indices : list of int, optional
            List of station indices to process (e.g., [0, 5, 10])
            If None, process all stations
        """
        self.db_dir = Path(db_dir)
        self.metadata_file = self.db_dir / "metadata.hdf5"
        self.gf_dir = self.db_dir / "green_functions"
        self.station_indices = set(station_indices) if station_indices else None
        
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
        
        # Load metadata
        self._load_metadata()
    
    def _load_metadata(self):
        """Load metadata from metadata.hdf5"""
        with h5py.File(self.metadata_file, 'r') as f:
            # Station info
            self.networks = [n.decode('utf-8') if isinstance(n, bytes) else n 
                           for n in f['Networks'][:]]
            self.stations = [s.decode('utf-8') if isinstance(s, bytes) else s 
                           for s in f['Stations'][:]]
            self.latitudes = f['latitudes'][:]
            self.longitudes = f['longitudes'][:]
            self.burials = f['burials'][:]
            self.nstations = len(self.networks)
            
            # Simulation parameters
            self.dt = float(f['DT'][()])
            self.tc = float(f['TC'][()])
            self.hdur = float(f['HDUR'][()])
            self.nsteps = int(f['NSTEPS'][()])
            self.nspec = int(f['NSPEC'][()])
            self.ngllx = int(f['NGLLX'][()])
            self.nglly = int(f['NGLLY'][()])
            self.ngllz = int(f['NGLLZ'][()])
            
            # Topography and ellipticity
            self.topography = bool(f['TOPOGRAPHY'][()])
            self.ellipticity = bool(f['ELLIPTICITY'][()])
            
            if self.topography:
                self.ibathy_topo = f['BATHY'][:]
                self.nx_bathy = int(f['NX_BATHY'][()])
                self.ny_bathy = int(f['NY_BATHY'][()])
                self.resolution_topo = float(f['RESOLUTION_TOPO_FILE'][()])
            
            if self.ellipticity:
                self.rspl = f['rspl'][:]
                self.ellipticity_spline = f['ellipticity_spline'][:]
                self.ellipticity_spline2 = f['ellipticity_spline2'][:]
            
            # Element coordinates (element-major format)
            self.xyz = f['xyz'][:]  # Shape: (NSPEC, NGLLX, NGLLY, NGLLZ, 3)
            
            # Create ibool-like structure for element access
            # In element-major, each element has its own local coordinate array
            # We'll create a simple sequential ibool
            self.ibool = np.arange(self.ngllx * self.nglly * self.ngllz).reshape(
                self.ngllx, self.nglly, self.ngllz, 1
            )
        
        print(f"Loaded metadata:")
        print(f"  Stations: {self.nstations}")
        print(f"  Elements: {self.nspec}")
        print(f"  Time steps: {self.nsteps}")
        print(f"  NGLL: {self.ngllx}x{self.nglly}x{self.ngllz}")
    
    def get_element_midpoints(self) -> np.ndarray:
        """Get midpoint coordinates for all elements
        
        Returns
        -------
        midpoints : ndarray
            Array of shape (NSPEC, 3) with xyz coordinates
        """
        mid_i = self.ngllx // 2
        mid_j = self.nglly // 2
        mid_k = self.ngllz // 2
        return self.xyz[:, mid_i, mid_j, mid_k, :]
    
    def reverse_engineer_source(self, ispec: int, offset_km: float = 0.0, depth_correction: float = 0.0) -> Tuple[float, float, float]:
        """Reverse engineer a source location from element coordinates
        
        Parameters
        ----------
        ispec : int
            Element index
        offset_km : float
            Offset from element center in km (for testing interpolation)
        depth_correction : float
            Depth correction in km to add to computed depth
        
        Returns
        -------
        lat, lon, depth : float
            Source location in geographic coordinates
        """
        # Get element midpoint in Cartesian coordinates
        mid_i = self.ngllx // 2
        mid_j = self.nglly // 2
        mid_k = self.ngllz // 2
        
        x_elem, y_elem, z_elem = self.xyz[ispec, mid_i, mid_j, mid_k, :]
        
        # Add offset if requested
        if offset_km != 0.0:
            offset_m = offset_km * 1000.0
            x_elem += offset_m
        
        # Convert Cartesian to geographic with WGS84 ellipticity correction
        # WGS84 ellipsoid parameters
        a_wgs84 = 6378137.0  # Semi-major axis (equatorial radius) in meters
        b_wgs84 = 6356752.314245  # Semi-minor axis (polar radius) in meters
        f_wgs84 = (a_wgs84 - b_wgs84) / a_wgs84  # Flattening ≈ 1/298.257223563
        
        # Radius from center (xyz coordinates are normalized in SPECFEM, where 1.0 = surface)
        r = np.sqrt(x_elem**2 + y_elem**2 + z_elem**2)
        
        # Latitude and longitude (initial estimate using spherical)
        lat = np.degrees(np.arcsin(z_elem / r))
        lon = np.degrees(np.arctan2(y_elem, x_elem))
        
        # Ellipsoidal radius at this latitude (distance from center to surface)
        # R(φ) = a * (1 - f * sin²(φ))
        lat_rad = np.radians(lat)
        R_ellipsoid = a_wgs84 * (1.0 - f_wgs84 * np.sin(lat_rad)**2)
        
        # Depth accounting for ellipticity
        # r is normalized, so depth = (1 - r) * R_ellipsoid
        depth = (1.0 - r) * R_ellipsoid / 1000.0  # Convert to km
        
        # Apply depth correction if provided
        depth += depth_correction
        
        return lat, lon, depth
    
    def get_seismograms(self, cmt: CMTSOLUTION) -> Tuple[Stream, int, float, float, float]:
        """Get seismograms using element-major format
        
        Automatically finds the correct element containing the source and
        computes seismograms for all stations.
        
        Parameters
        ----------
        cmt : CMTSOLUTION
            Source description
        
        Returns
        -------
        stream : Stream
            Obspy stream with 3-component seismograms for all stations
        ispec_selected : int
            Element index found by kdtree search
        xi, eta, gamma : float
            Local coordinates within the element
        """
        # Convert source to mesh coordinates and rotate moment tensor
        x_target, y_target, z_target, Mx = source2xyz(
            cmt.latitude,
            cmt.longitude,
            cmt.depth,
            M=cmt.tensor,
            topography=self.topography,
            ellipticity=self.ellipticity,
            ibathy_topo=self.ibathy_topo if self.topography else None,
            NX_BATHY=self.nx_bathy if self.topography else None,
            NY_BATHY=self.ny_bathy if self.topography else None,
            RESOLUTION_TOPO_FILE=self.resolution_topo if self.topography else None,
            rspl=self.rspl if self.ellipticity else None,
            ellipicity_spline=self.ellipticity_spline if self.ellipticity else None,
            ellipicity_spline2=self.ellipticity_spline2 if self.ellipticity else None,
        )
        
        # Find which element contains the source using ALL element midpoints
        print(f"Finding which element contains the source...")
        all_midpoints = self.get_element_midpoints()  # (NSPEC, 3)
        kdtree = KDTree(all_midpoints)
        ispec_selected = find_element_for_source(
            x_target, y_target, z_target,
            all_midpoints,
            kdtree=kdtree
        )
        print(f"Selected element: {ispec_selected}")
        
        # Load element coordinates for the selected element
        xyz_element = self.xyz[ispec_selected, :, :, :, :]  # (NGLLX, NGLLY, NGLLZ, 3)
        
        # Now compute local coordinates (xi, eta, gamma) within the selected element
        print(f"Computing local coordinates within element {ispec_selected}...")
        (
            xi, eta, gamma,
            x, y, z,
            xix, xiy, xiz,
            etax, etay, etaz,
            gammax, gammay, gammaz,
            distmin
        ) = locate_point_no_ibool(
            x_target, y_target, z_target,
            xyz_element,
            POINT_CAN_BE_BURIED=True,
            NGLL=self.ngllx
        )
        
        print(f"Local coordinates: xi={xi:.4f}, eta={eta:.4f}, gamma={gamma:.4f}")
        print(f"Distance to target: {distmin:.3f} km")
        
        # Check if local coordinates are valid (should be in [-1, 1])
        coords_outside = False
        if abs(xi) > 1.01 or abs(eta) > 1.01 or abs(gamma) > 1.01:
            print(f"ERROR: Local coordinates outside element bounds!")
            print(f"  |xi|={abs(xi):.4f}, |eta|={abs(eta):.4f}, |gamma|={abs(gamma):.4f}")
            print(f"  Coordinates outside ±1.01 indicate source is not properly in this element.")
            print(f"  Old DB (GFManager) would search neighboring elements in this case,")
            print(f"  but new DB does not. This makes validation invalid.")
            coords_outside = True
        
        # Return local coordinates for validation check
        if coords_outside:
            return Stream(), ispec_selected, xi, eta, gamma
        
        # Check moment tensor
        print(f"\nMoment tensor Mx: {Mx}")
        Mx_max = np.max(np.abs(Mx))
        print(f"Max |Mx| value: {Mx_max:.6e}")
        if Mx_max == 0.0:
            print(f"ERROR: Moment tensor is ZERO!")
        
        # Load displacement data for the selected element
        element_file = self.gf_dir / f"element{ispec_selected:06d}.h5"
        print(f"Reading displacement from: {element_file}")
        with h5py.File(element_file, 'r') as f:
            # Shape: (Nstations, 3_comp, 3_forces, 5, 5, 5, NSTEPS)
            displacement = f['displacement'][:]
        
        # Check if data is non-zero
        max_val = np.max(np.abs(displacement))
        print(f"Max displacement value: {max_val:.6e}")
        if max_val == 0.0:
            print(f"WARNING: Element {ispec_selected} has ALL ZEROS!")
        
        # Check specific station G.AIS if it exists
        try:
            ais_idx = [i for i, (net, sta) in enumerate(zip(self.networks, self.stations)) 
                      if net == 'G' and sta == 'AIS'][0]
            ais_val = np.max(np.abs(displacement[ais_idx, :, :, :, :, :, :]))
            print(f"Station G.AIS (index {ais_idx}) max value: {ais_val:.6e}")
        except (IndexError, ValueError):
            print(f"Station G.AIS not found")
        
        # GLL nodes and Lagrange polynomials
        npol = self.ngllx - 1
        xigll, _, _ = gll_nodes(npol)
        hxi, hpxi = lagrange_any(xi, xigll, npol)
        heta, hpeta = lagrange_any(eta, xigll, npol)
        hgamma, hpgamma = lagrange_any(gamma, xigll, npol)
        
        # Diagnostic: Check Lagrange polynomial values
        print(f"\nLagrange polynomials:")
        print(f"  hxi    (values): max={np.max(np.abs(hxi)):.6e}, sum={np.sum(hxi):.6e}")
        print(f"  hpxi   (derivs): max={np.max(np.abs(hpxi)):.6e}, sum={np.sum(hpxi):.6e}")
        print(f"  heta   (values): max={np.max(np.abs(heta)):.6e}, sum={np.sum(heta):.6e}")
        print(f"  hpeta  (derivs): max={np.max(np.abs(hpeta)):.6e}, sum={np.sum(hpeta):.6e}")
        print(f"  hgamma (values): max={np.max(np.abs(hgamma)):.6e}, sum={np.sum(hgamma):.6e}")
        print(f"  hpgamma(derivs): max={np.max(np.abs(hpgamma)):.6e}, sum={np.sum(hpgamma):.6e}")
        
        if np.max(np.abs(hxi)) == 0.0 or np.max(np.abs(heta)) == 0.0 or np.max(np.abs(hgamma)) == 0.0:
            print(f"ERROR: Lagrange polynomials are ZERO!")
        
        # Prepare STF
        NP2 = next_power_of_2(2 * self.nsteps)
        
        if (cmt.hdur / 1.628) <= self.hdur:
            hdur_r = 1e-6
            print(f"Warning: Requested hdur smaller than simulated. Using {hdur_r}s")
        else:
            hdur_r = np.sqrt((cmt.hdur / 1.628) ** 2 - self.hdur**2)
        
        _, stf_r = create_stf(
            0, 200.0, self.nsteps, self.dt, hdur_r, 
            cutoff=None, gaussian=False, lpfilter="butter"
        )
        STF_R = fft.fft(stf_r, n=NP2)
        shift = -200.0 + cmt.time_shift
        phshift = np.exp(-1.0j * shift * np.fft.fftfreq(NP2, self.dt) * 2 * np.pi)
        
        # Process all stations
        all_traces = []
        
        for istation in range(self.nstations):
            # Apply station index filter if specified
            if self.station_indices is not None and istation not in self.station_indices:
                continue
            
            # Get displacement for this station
            disp_station = displacement[istation, :, :, :, :, :, :]
            # Shape: (3_comp, 3_forces, 5, 5, 5, NSTEPS)
            
            # Compute epsilon (strain tensor) via interpolation
            epsilon = np.zeros((3, 6, self.nsteps))
            
            for k in range(self.ngllz):
                for j in range(self.nglly):
                    for i in range(self.ngllx):
                        hlagrange_xi = hpxi[i] * heta[j] * hgamma[k]
                        hlagrange_eta = hxi[i] * hpeta[j] * hgamma[k]
                        hlagrange_gamma = hxi[i] * heta[j] * hpgamma[k]
                        hlagrange_x = (hlagrange_xi * xix + hlagrange_eta * etax + 
                                      hlagrange_gamma * gammax)
                        hlagrange_y = (hlagrange_xi * xiy + hlagrange_eta * etay + 
                                      hlagrange_gamma * gammay)
                        hlagrange_z = (hlagrange_xi * xiz + hlagrange_eta * etaz + 
                                      hlagrange_gamma * gammaz)
                        
                        for icomp in range(3):
                            epsilon[icomp, 0, :] += disp_station[icomp, 0, i, j, k, :] * hlagrange_x
                            epsilon[icomp, 1, :] += disp_station[icomp, 1, i, j, k, :] * hlagrange_y
                            epsilon[icomp, 2, :] += disp_station[icomp, 2, i, j, k, :] * hlagrange_z
                            epsilon[icomp, 3, :] += 0.5 * (
                                disp_station[icomp, 1, i, j, k, :] * hlagrange_x +
                                disp_station[icomp, 0, i, j, k, :] * hlagrange_y
                            )
                            epsilon[icomp, 4, :] += 0.5 * (
                                disp_station[icomp, 2, i, j, k, :] * hlagrange_x +
                                disp_station[icomp, 0, i, j, k, :] * hlagrange_z
                            )
                            epsilon[icomp, 5, :] += 0.5 * (
                                disp_station[icomp, 2, i, j, k, :] * hlagrange_y +
                                disp_station[icomp, 1, i, j, k, :] * hlagrange_z
                            )
            
            # Create traces for this station
            for icomp, comp in enumerate(['N', 'E', 'Z']):
                # Dot moment tensor with strain
                data = np.sum(
                    np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])[:, None] *
                    Mx[:, None] *
                    epsilon[icomp, :, :],
                    axis=0
                )
                
                # Convolve with STF
                data = np.real(fft.ifft(phshift * fft.fft(data, n=NP2) * self.dt * STF_R))[:self.nsteps]
                
                # Create trace
                stats = Stats()
                stats.delta = self.dt
                stats.network = self.networks[istation]
                stats.station = self.stations[istation]
                stats.location = "S3"
                stats.latitude = self.latitudes[istation]
                stats.longitude = self.longitudes[istation]
                stats.coordinates = AttribDict(
                    latitude=self.latitudes[istation],
                    longitude=self.longitudes[istation]
                )
                stats.channel = f"MX{comp}"
                stats.starttime = cmt.origin_time - self.tc
                stats.npts = len(data)
                
                tr = Trace(data=data, header=stats)
                all_traces.append(tr)
        
        return Stream(all_traces), ispec_selected, xi, eta, gamma


def plot_station_comparison(old_stream: Stream, new_stream: Stream, station_id: str, output_dir: Path):
    """Plot comparison of old vs new seismograms for one station
    
    Parameters
    ----------
    old_stream : Stream
        Stream from old database (3 components for one station)
    new_stream : Stream
        Stream from new database (3 components for one station)
    station_id : str
        Station identifier (e.g., 'G.AIS')
    output_dir : Path
        Directory to save plots
    """
    # Sort by component
    old_stream = old_stream.copy().sort(keys=['channel'])
    new_stream = new_stream.copy().sort(keys=['channel'])
    
    # Create figure with 2:1 width:height ratio
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(f'Station {station_id} - Old vs New Database', fontsize=14, fontweight='bold')
    
    components = ['N', 'E', 'Z']
    colors_old = '#1f77b4'  # Blue
    colors_new = '#ff7f0e'  # Orange
    
    for i, (comp, ax) in enumerate(zip(components, axes)):
        # Find traces for this component
        old_tr = [tr for tr in old_stream if tr.stats.channel.endswith(comp)][0]
        new_tr = [tr for tr in new_stream if tr.stats.channel.endswith(comp)][0]
        
        # Time array
        times = old_tr.times()
        
        # Plot both traces
        ax.plot(times, old_tr.data, color=colors_old, linewidth=1.0, label='Old DB', alpha=0.8)
        ax.plot(times, new_tr.data, color=colors_new, linewidth=1.0, label='New DB', alpha=0.8, linestyle='--')
        
        # Compute differences
        diff = np.abs(old_tr.data - new_tr.data)
        max_diff = np.max(diff)
        max_val = np.max(np.abs(old_tr.data))
        rel_diff = max_diff / max_val if max_val > 0 else 0
        
        # Labels and formatting
        ax.set_ylabel(f'{comp}\n[m]', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        
        # Add difference text
        ax.text(0.02, 0.95, f'Max diff: {max_diff:.2e}\nRel diff: {rel_diff:.2e}',
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[-1].set_xlabel('Time [s]', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{station_id.replace(".", "_")}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)


def compare_streams(stream1: Stream, stream2: Stream, station_to_index: Dict[str, int] = None, rtol: float = 1e-5) -> Tuple[bool, Dict[str, Tuple[Stream, Stream]]]:
    """Compare two streams for equality
    
    Parameters
    ----------
    stream1, stream2 : Stream
        Streams to compare
    station_to_index : dict, optional
        Mapping from station_id (e.g., 'G.AIS') to station index
    rtol : float
        Relative tolerance (validation controlled by relative difference only)
    
    Returns
    -------
    match : bool
        True if streams match within tolerance
    station_streams : dict
        Dictionary mapping station_id to (old_stream, new_stream) tuples
    """
    if len(stream1) != len(stream2):
        print(f"ERROR: Different number of traces: {len(stream1)} vs {len(stream2)}")
        return False
    
    # Sort both streams by network.station.channel
    stream1 = stream1.copy().sort(keys=['network', 'station', 'channel'])
    stream2 = stream2.copy().sort(keys=['network', 'station', 'channel'])
    
    all_match = True
    max_diff = 0.0
    max_rel_diff = 0.0
    
    matching_stations = []  # List of (station_id, station_idx, abs_diff, rel_diff)
    mismatching_stations = []  # List of (station_id, station_idx, reason, abs_diff, rel_diff)
    station_streams = {}  # Store streams for each station
    
    for tr1, tr2 in zip(stream1, stream2):
        id1 = f"{tr1.stats.network}.{tr1.stats.station}.{tr1.stats.channel}"
        id2 = f"{tr2.stats.network}.{tr2.stats.station}.{tr2.stats.channel}"
        station_id = f"{tr1.stats.network}.{tr1.stats.station}"
        
        # Get station index if mapping provided
        station_idx = station_to_index.get(station_id, -1) if station_to_index else -1
        
        # Collect traces by station
        if station_id not in station_streams:
            station_streams[station_id] = (Stream(), Stream())
        station_streams[station_id][0].append(tr1)
        station_streams[station_id][1].append(tr2)
        
        if id1 != id2:
            print(f"ERROR: Trace ID mismatch: {id1} vs {id2}")
            mismatching_stations.append((station_id, station_idx, "ID mismatch", 0, 0))
            all_match = False
            continue
        
        if len(tr1.data) != len(tr2.data):
            print(f"ERROR: {id1} - Different lengths: {len(tr1.data)} vs {len(tr2.data)}")
            mismatching_stations.append((station_id, station_idx, "Length mismatch", 0, 0))
            all_match = False
            continue
        
        diff = np.abs(tr1.data - tr2.data)
        max_abs_diff = np.max(diff)
        max_val = np.max(np.abs(tr1.data))
        rel_diff = max_abs_diff / max_val if max_val > 0 else 0
        max_diff = max(max_diff, max_abs_diff)
        max_rel_diff = max(max_rel_diff, rel_diff)
        
        # Only use relative tolerance for validation (absolute difference controlled by magnitude)
        if rel_diff > rtol:
            if station_id not in [s[0] for s in mismatching_stations]:
                mismatching_stations.append((station_id, station_idx, "Value mismatch", max_abs_diff, rel_diff))
            all_match = False
        else:
            if station_id not in [s[0] for s in matching_stations] and station_id not in [s[0] for s in mismatching_stations]:
                matching_stations.append((station_id, station_idx, max_abs_diff, rel_diff))
    
    print(f"\n{'='*80}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"\n✓ MATCHING STATIONS ({len(matching_stations)} stations):")
    for station_id, station_idx, abs_diff, rel_diff in matching_stations:
        idx_str = f"[{station_idx:3d}]" if station_idx >= 0 else "[---]"
        print(f"  {idx_str} {station_id:15s} - max_abs_diff: {abs_diff:.3e}, max_rel_diff: {rel_diff:.3e}")
    
    print(f"\n✗ MISMATCHING STATIONS ({len(mismatching_stations)} stations):")
    for station_id, station_idx, reason, abs_diff, rel_diff in mismatching_stations:
        idx_str = f"[{station_idx:3d}]" if station_idx >= 0 else "[---]"
        if reason == "Value mismatch":
            print(f"  {idx_str} {station_id:15s} - {reason:20s} max_abs_diff: {abs_diff:.3e}, max_rel_diff: {rel_diff:.3e}")
        else:
            print(f"  {idx_str} {station_id:15s} - {reason}")
    
    print(f"\n{'='*80}")
    if all_match:
        print(f"✓ All traces match within tolerance")
        print(f"  Max absolute difference across all traces: {max_diff:.3e}")
        print(f"  Max relative difference across all traces: {max_rel_diff:.3e}")
    else:
        print(f"✗ Some traces do not match")
        print(f"  Max absolute difference across all traces: {max_diff:.3e}")
        print(f"  Max relative difference across all traces: {max_rel_diff:.3e}")
    
    return all_match, station_streams


def validate_seismograms(
    old_db_dir: str,
    new_db_dir: str,
    test_element: int,
    moment_magnitude: float = 6.0,
    rtol: float = 1e-5,
    station_indices: Optional[List[int]] = None,
    plot_dir: Optional[str] = None,
    depth_correction: float = 0.0
) -> bool:
    """Validate seismograms from both database formats
    
    Parameters
    ----------
    old_db_dir : str
        Path to old station-major database
    new_db_dir : str
        Path to new element-major database
    test_element : int
        Element index to test
    moment_magnitude : float
        Moment magnitude for synthetic source
    rtol : float
        Relative tolerance for comparison (validation controlled by relative difference only)
    station_indices : list of int, optional
        List of station indices to compare (e.g., [0, 5, 10])
        If None, compare all stations
    plot_dir : str, optional
        Directory to save comparison plots. If None, no plots are generated
    depth_correction : float
        Depth correction in km to add to reverse-engineered depth (default: 0.0)
    
    Returns
    -------
    success : bool
        True if validation passed
    """
    print("="*80)
    print("Seismogram Validation")
    print("="*80)
    
    # Load element-major database
    print(f"\nLoading element-major database: {new_db_dir}")
    em_reader = ElementMajorReader(new_db_dir, station_indices=station_indices)
    
    # Reverse-engineer source location from element midpoint
    print(f"\nReverse-engineering source location from element {test_element}")
    if depth_correction != 0.0:
        print(f"  Applying depth correction: {depth_correction:+.2f} km")
    lat, lon, depth = em_reader.reverse_engineer_source(test_element, depth_correction=depth_correction)
    print(f"  Latitude:  {lat:.4f}°")
    print(f"  Longitude: {lon:.4f}°")
    print(f"  Depth:     {depth:.2f} km")
    
    # Create synthetic CMT solution
    print(f"\nCreating synthetic CMT (Mw={moment_magnitude})")
    M0 = 10**(1.5 * moment_magnitude + 9.1)  # dyne-cm
    cmt = CMTSOLUTION()
    cmt.latitude = lat
    cmt.longitude = lon
    cmt.depth = depth
    cmt.time_shift = 0.0
    cmt.hdur = 10.0  # 10 second half-duration
    
    # Simple moment tensor (strike-slip on vertical fault)
    cmt.Mrr = 0.0
    cmt.Mtt = M0
    cmt.Mpp = -M0
    cmt.Mrt = 0.0
    cmt.Mrp = 0.0
    cmt.Mtp = 0.0
    
    # Diagnostic: Check if CMT tensor property works
    print(f"\nCMT moment tensor components:")
    print(f"  Mrr: {cmt.Mrr:.6e}")
    print(f"  Mtt: {cmt.Mtt:.6e}")
    print(f"  Mpp: {cmt.Mpp:.6e}")
    print(f"  Mrt: {cmt.Mrt:.6e}")
    print(f"  Mrp: {cmt.Mrp:.6e}")
    print(f"  Mtp: {cmt.Mtp:.6e}")
    print(f"\nCMT tensor property: {cmt.tensor}")
    print(f"  Max |tensor|: {np.max(np.abs(cmt.tensor)):.6e}")
    print(f"  M0: {cmt.M0:.6e} dyne-cm")
    print(f"  Mw: {cmt.Mw:.2f}")
    
    # Compute seismograms using new format FIRST
    # This allows us to check if coordinates are within element bounds
    print(f"\nComputing seismograms using NEW format...")
    new_stream, ispec_found, xi, eta, gamma = em_reader.get_seismograms(cmt)
    print(f"  Generated {len(new_stream)} traces from new DB")
    print(f"  Source location: lat={lat:.4f}°, lon={lon:.4f}°, depth={depth:.2f} km")
    
    # Check if coordinates are within element bounds
    coords_outside = (abs(xi) > 1.01 or abs(eta) > 1.01 or abs(gamma) > 1.01)
    
    if coords_outside:
        print(f"\n{'='*80}")
        print(f"✗ VALIDATION ABORTED")
        print(f"{'='*80}")
        print(f"Reason: Local coordinates outside element bounds (±1.1)")
        print(f"  |xi|={abs(xi):.4f}, |eta|={abs(eta):.4f}, |gamma|={abs(gamma):.4f}")
        print(f"\nThe reverse-engineered source location (from element {test_element}) landed")
        print(f"outside the element bounds when transformed through source2xyz().")
        print(f"This occurs because:")
        print(f"  1. reverse_engineer_source() uses simple spherical coordinates")
        print(f"  2. source2xyz() applies topography/ellipticity corrections")
        print(f"  3. These transforms don't invert perfectly")
        print(f"\nGFManager.get_seismograms() searches neighboring elements when this happens,")
        print(f"but ElementMajorReader.get_seismograms() does not. This makes validation invalid.")
        print(f"\nSuggestions:")
        print(f"  1. Try a different test element (--element)")
        print(f"  2. Use element closer to spherically symmetric region")
        print(f"  3. Reduce magnitude to minimize interpolation errors")
        print(f"{'='*80}")
        return False
    
    # Get element midpoints for distance calculation
    all_midpoints = em_reader.get_element_midpoints()
    
    # Check new data for non-zero values
    new_max = max([np.max(np.abs(tr.data)) for tr in new_stream]) if len(new_stream) > 0 else 0.0
    print(f"  Max amplitude in new traces: {new_max:.6e}")
    if new_max == 0.0:
        print(f"  WARNING: ALL NEW TRACES ARE ZERO!")
    
    # Check which element was actually found
    print(f"\nElement selection:")
    print(f"  Input element (for reverse-engineering): {test_element}")
    print(f"  Found element (by kdtree search):        {ispec_found}")
    
    if ispec_found == test_element:
        print(f"  ✓ Source location maps back to original element")
    else:
        offset_km = np.linalg.norm(all_midpoints[test_element] - all_midpoints[ispec_found])
        print(f"  → Source maps to different element (offset: {offset_km:.3f} km)")
        print(f"  → This is expected due to topography/ellipticity in coordinate transform")
        print(f"  → Using element {ispec_found} for validation (both databases use same source)")
    
    # Now compute old DB seismograms for comparison
    # Get list of station files for old DB
    print(f"\nScanning old database: {old_db_dir}")
    old_db_path = Path(old_db_dir)
    station_files = []
    station_names = []
    for idx, (network_dir, station_dir) in enumerate(sorted(
        [(nd, sd) for nd in sorted(old_db_path.iterdir()) if nd.is_dir() 
         for sd in sorted(nd.iterdir()) if sd.is_dir()]
    )):
        network = network_dir.name
        station = station_dir.name
        
        # Apply station index filter if specified
        if station_indices is not None and idx not in station_indices:
            continue
        
        h5_file = station_dir / f"{network}.{station}.h5"
        if h5_file.exists():
            station_files.append(h5_file)
            station_names.append(f"{network}.{station}")
    
    if station_indices:
        print(f"Found {len(station_files)} station files (indices {sorted(station_indices)}):")
        for idx, name in zip(sorted(station_indices), station_names):
            print(f"  [{idx}] {name}")
    else:
        print(f"Found {len(station_files)} station files")
    
    # Create GFManager with station files
    print(f"\nInitializing GFManager with old format...")
    gfm = GFManager([str(f) for f in station_files])
    
    # Load header and get elements around the source location
    print(f"Loading header and getting elements around source...")
    gfm.load_header_variables()
    gfm.get_elements(lat, lon, depth, dist_in_km=25.0, NGLL=5, threading=True)
    
    print(f"Displacement", np.max(np.abs(gfm.displacement)))
    
    # Compute seismograms using old format
    print(f"\nComputing seismograms using OLD format (GFManager)...")
    old_stream = gfm.get_seismograms(cmt)
    print(f"  Generated {len(old_stream)} traces from old DB")
    
    # Check if old data has actual values
    old_max = max([np.max(np.abs(tr.data)) for tr in old_stream]) if len(old_stream) > 0 else 0.0
    print(f"  Max amplitude in old traces: {old_max:.6e}")
    if old_max == 0.0:
        print(f"  WARNING: ALL OLD TRACES ARE ZERO!")
    
    # Create station to index mapping
    station_to_index = {}
    for idx, (net, sta) in enumerate(zip(em_reader.networks, em_reader.stations)):
        station_id = f"{net}.{sta}"
        station_to_index[station_id] = idx
    
    # Compare streams
    print(f"\nComparing seismograms...")
    print(f"  Relative tolerance: {rtol}")
    success, station_streams = compare_streams(old_stream, new_stream, station_to_index=station_to_index, rtol=rtol)
    
    # Generate plots if requested
    if plot_dir is not None:
        output_path = Path(plot_dir)
        print(f"\nGenerating comparison plots...")
        print(f"  Output directory: {output_path}")
        for station_id, (old_st, new_st) in station_streams.items():
            plot_station_comparison(old_st, new_st, station_id, output_path)
        print(f"  Generated {len(station_streams)} plots")
    
    print("="*80)
    if success:
        print("✓ VALIDATION PASSED")
    else:
        print("✗ VALIDATION FAILED")
    print("="*80)
    
    return success


def parse_station_indices(indices_args: List[str]) -> List[int]:
    """Parse station index arguments supporting ranges
    
    Parameters
    ----------
    indices_args : list of str
        Arguments like ['0', '5', '10-15', '20']
    
    Returns
    -------
    indices : list of int
        Expanded list of indices [0, 5, 10, 11, 12, 13, 14, 15, 20]
    """
    indices = set()
    for arg in indices_args:
        if '-' in arg:
            start, end = arg.split('-')
            indices.update(range(int(start), int(end) + 1))
        else:
            indices.add(int(arg))
    return sorted(list(indices))


def main():
    parser = argparse.ArgumentParser(
        description='Validate database reorganization via seismogram comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('old_db', type=str, 
                       help='Path to old station-major database')
    parser.add_argument('new_db', type=str,
                       help='Path to new element-major database')
    parser.add_argument('--element', type=int, default=9032,
                       help='Element index to test (default: 9032)')
    parser.add_argument('--magnitude', type=float, default=6.0,
                       help='Moment magnitude for test (default: 6.0)')
    parser.add_argument('--rtol', type=float, default=1e-5,
                       help='Relative tolerance for validation - only relative difference is checked (default: 1e-5)')
    parser.add_argument('--stations', type=str, nargs='+',
                       help='Station indices or ranges to compare (e.g., 0 5 10-15). If not specified, compare all stations')
    parser.add_argument('--plot-dir', type=str,
                       help='Directory to save comparison plots (if not specified, no plots generated)')
    parser.add_argument('--depth-correction', type=float, default=0.0,
                       help='Depth correction in km to add to reverse-engineered depth (default: 0.0)')
    
    args = parser.parse_args()
    
    # Parse station indices
    station_indices = None
    if args.stations:
        station_indices = parse_station_indices(args.stations)
    
    success = validate_seismograms(
        args.old_db,
        args.new_db,
        args.element,
        args.magnitude,
        args.rtol,
        station_indices,
        args.plot_dir,
        args.depth_correction
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
