#!/usr/bin/env python
"""
Reorganize Green Function Database from Station-Major to Element-Major

This script transforms a large Green function database from:
  - Station-centric: One file per station containing all elements
  - To Element-centric: One file per element containing all stations

Input structure:
  DB_old/
    Network1/Station1/Network1.Station1.h5
    Network2/Station2/Network2.Station2.h5
    ...

Output structure:
  DB_reorg/
    metadata.hdf5
    green_functions/
      element000000.hdf5
      element000001.hdf5
      ...

Usage:
  # Single process (testing)
  python reorganize_database.py /path/to/DB_old /path/to/DB_reorg
  
  # MPI parallel (production)
  mpirun -np 100 python reorganize_database.py /path/to/DB_old /path/to/DB_reorg \\
    --compression lzf --chunks station
"""

import argparse
import h5py
import numpy as np
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    print("Warning: mpi4py not available, running in serial mode")


class DatabaseReorganizer:
    """Main class for reorganizing the Green function database"""
    
    def __init__(self, input_dir: str, output_dir: str, 
                 compression: Optional[str] = None,
                 chunks: str = 'station',
                 validate: bool = True,
                 checkpoint_interval: int = 100,
                 elements: Optional[List[int]] = None,
                 dry_run: bool = False):
        """
        Initialize the database reorganizer
        
        Parameters
        ----------
        input_dir : str
            Path to input database directory
        output_dir : str
            Path to output database directory
        compression : str, optional
            Compression type: 'lzf', 'gzip', or None
        chunks : str
            Chunking strategy: 'station', 'gll', 'auto', or 'none'
        validate : bool
            Whether to validate random samples
        checkpoint_interval : int
            How often to save checkpoint (number of elements)
        elements : list of int, optional
            Specific element indices to process (for testing)
        dry_run : bool
            If True, only simulate the reorganization without writing files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.compression = compression
        self.chunks = chunks
        self.validate = validate
        self.checkpoint_interval = checkpoint_interval
        self.elements = elements
        self.dry_run = dry_run
        
        # MPI setup
        if HAS_MPI:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1
        
        # Data structures (populated during scanning)
        self.station_files: List[Path] = []
        self.station_info: Dict = {}
        self.nstations: int = 0
        self.nspec: int = 0
        self.nglob: int = 0
        self.nsteps: int = 0
        self.ngllx: int = 5
        self.nglly: int = 5
        self.ngllz: int = 5
        
        # Progress tracking
        self.checkpoint_file = self.output_dir / f"checkpoint_rank{self.rank:04d}.txt"
        self.completed_elements = set()
        
    def log(self, message: str, rank_specific: bool = False):
        """Print log message (only from rank 0 unless rank_specific=True)"""
        if not rank_specific and self.rank != 0:
            return
        prefix = f"[Rank {self.rank}] " if rank_specific or self.size > 1 else ""
        print(f"{prefix}{message}", flush=True)
    
    def scan_database(self):
        """
        Scan the input database to inventory all station files and extract dimensions
        
        Only rank 0 performs the scan, then broadcasts to all ranks
        """
        if self.rank == 0:
            self.log("Scanning database...")
            start_time = time.time()
            
            # Find all HDF5 files following Network/Station/Network.Station.h5 pattern
            station_files = []
            for network_dir in sorted(self.input_dir.iterdir()):
                if not network_dir.is_dir():
                    continue
                for station_dir in sorted(network_dir.iterdir()):
                    if not station_dir.is_dir():
                        continue
                    # Look for Network.Station.h5 file
                    network = network_dir.name
                    station = station_dir.name
                    h5_file = station_dir / f"{network}.{station}.h5"
                    if h5_file.exists():
                        station_files.append(h5_file)
            
            if len(station_files) == 0:
                raise ValueError(f"No station files found in {self.input_dir}")
            
            self.log(f"Found {len(station_files)} station files")
            
            # Read dimensions from first file
            with h5py.File(station_files[0], 'r') as f:
                self.nsteps = int(f['NSTEPS'][()])
                self.nglob = int(f['NGLOB'][()])
                self.nspec = int(f['NSPEC'][()])
                self.ngllx = int(f['NGLLX'][()])
                self.nglly = int(f['NGLLY'][()])
                self.ngllz = int(f['NGLLZ'][()])
            
            self.log(f"Database dimensions:")
            self.log(f"  NSPEC (elements): {self.nspec}")
            self.log(f"  NGLOB (points): {self.nglob}")
            self.log(f"  NSTEPS (time): {self.nsteps}")
            self.log(f"  NGLL: {self.ngllx}x{self.nglly}x{self.ngllz}")
            
            # Build station info
            station_info = {}
            for idx, sfile in enumerate(station_files):
                with h5py.File(sfile, 'r') as f:
                    network = f['Network'][()].decode('utf-8') if isinstance(f['Network'][()], bytes) else f['Network'][()]
                    station = f['Station'][()].decode('utf-8') if isinstance(f['Station'][()], bytes) else f['Station'][()]
                    latitude = float(f['latitude'][()])
                    longitude = float(f['longitude'][()])
                    burial = float(f['burial'][()])
                    
                    station_info[idx] = {
                        'file': sfile,
                        'network': network,
                        'station': station,
                        'latitude': latitude,
                        'longitude': longitude,
                        'burial': burial
                    }
            
            self.nstations = len(station_files)
            self.station_files = station_files
            self.station_info = station_info
            
            elapsed = time.time() - start_time
            self.log(f"Scan complete in {elapsed:.2f}s")
            
            # Prepare data for broadcast
            scan_data = {
                'station_files': station_files,
                'station_info': station_info,
                'nstations': self.nstations,
                'nspec': self.nspec,
                'nglob': self.nglob,
                'nsteps': self.nsteps,
                'ngllx': self.ngllx,
                'nglly': self.nglly,
                'ngllz': self.ngllz
            }
        else:
            scan_data = None
        
        # Broadcast to all ranks
        if HAS_MPI:
            scan_data = self.comm.bcast(scan_data, root=0)
            if self.rank != 0:
                self.station_files = scan_data['station_files']
                self.station_info = scan_data['station_info']
                self.nstations = scan_data['nstations']
                self.nspec = scan_data['nspec']
                self.nglob = scan_data['nglob']
                self.nsteps = scan_data['nsteps']
                self.ngllx = scan_data['ngllx']
                self.nglly = scan_data['nglly']
                self.ngllz = scan_data['ngllz']
    
    def create_metadata_file(self):
        """
        Create the metadata.hdf5 file with station arrays and shared fields
        
        Only rank 0 creates this file
        """
        if self.rank != 0:
            return
        
        if self.dry_run:
            self.log("[DRY RUN] Would create metadata.hdf5...")
        else:
            self.log("Creating metadata.hdf5...")
        
        start_time = time.time()
        
        metadata_file = self.output_dir / "metadata.hdf5"
        
        if self.dry_run:
            # Just read and report what would be written
            with h5py.File(self.station_files[0], 'r') as f:
                self.log("  Station arrays:")
                self.log(f"    - Networks: {self.nstations} stations")
                self.log(f"    - Stations: {self.nstations} stations")
                self.log(f"    - latitudes: {self.nstations} values")
                self.log(f"    - longitudes: {self.nstations} values")
                self.log(f"    - burials: {self.nstations} values")
                
                self.log("  Simulation parameters:")
                for field in ['DT', 'TC', 'T0', 'HDUR', 'FACTOR', 'NSTEPS']:
                    if field in f:
                        self.log(f"    - {field}: {f[field][()]}")
                
                self.log("  Grid parameters:")
                for field in ['NSPEC', 'NGLLX', 'NGLLY', 'NGLLZ']:
                    if field in f:
                        self.log(f"    - {field}: {f[field][()]}")
                
                self.log("  Flags:")
                for field in ['TOPOGRAPHY', 'ELLIPTICITY']:
                    if field in f:
                        self.log(f"    - {field}: {f[field][()]}")
                
                if 'BATHY' in f:
                    self.log(f"  Bathymetry data: shape {f['BATHY'].shape}")
                
                for field in ['rspl', 'ellipticity_spline', 'ellipticity_spline2']:
                    if field in f:
                        self.log(f"  Ellipticity data: {field} shape {f[field].shape}")
                
                ibool = f['ibool'][:]
                xyz_global = f['xyz'][:]
                self.log(f"  xyz coordinates: would reorganize from {xyz_global.shape} to ({self.nspec}, {self.ngllx}, {self.nglly}, {self.ngllz}, 3)")
                
                # Estimate file size
                total_bytes = 0
                total_bytes += self.nstations * 50  # String data estimate
                total_bytes += self.nstations * 8 * 3  # lat/lon/burial
                total_bytes += self.nspec * self.ngllx * self.nglly * self.ngllz * 3 * 8  # xyz
                if 'BATHY' in f:
                    total_bytes += f['BATHY'].size * f['BATHY'].dtype.itemsize
                
                self.log(f"  Estimated metadata.hdf5 size: {total_bytes / (1024**3):.2f} GB")
            
            elapsed = time.time() - start_time
            self.log(f"[DRY RUN] Metadata scan complete in {elapsed:.2f}s")
            return
        
        with h5py.File(metadata_file, 'w') as meta:
            # Station arrays
            networks = []
            stations = []
            latitudes = []
            longitudes = []
            burials = []
            
            for idx in range(self.nstations):
                info = self.station_info[idx]
                networks.append(info['network'])
                stations.append(info['station'])
                latitudes.append(info['latitude'])
                longitudes.append(info['longitude'])
                burials.append(info['burial'])
            
            # Create string datasets with proper encoding
            dt_string = h5py.string_dtype(encoding='utf-8')
            meta.create_dataset('Networks', data=np.array(networks, dtype=object), dtype=dt_string)
            meta.create_dataset('Stations', data=np.array(stations, dtype=object), dtype=dt_string)
            meta.create_dataset('latitudes', data=np.array(latitudes, dtype=np.float64))
            meta.create_dataset('longitudes', data=np.array(longitudes, dtype=np.float64))
            meta.create_dataset('burials', data=np.array(burials, dtype=np.float64))
            
            # Read shared fields from first station file
            with h5py.File(self.station_files[0], 'r') as f:
                # Simulation parameters
                simulation_fields = ['DT', 'TC', 'T0', 'HDUR', 'FACTOR', 'NSTEPS']
                for field in simulation_fields:
                    if field in f:
                        meta.create_dataset(field, data=f[field][()])
                
                # Grid parameters
                grid_fields = ['NSPEC', 'NGLLX', 'NGLLY', 'NGLLZ']
                for field in grid_fields:
                    if field in f:
                        meta.create_dataset(field, data=f[field][()])
                
                # Flags
                flag_fields = ['TOPOGRAPHY', 'ELLIPTICITY']
                for field in flag_fields:
                    if field in f:
                        meta.create_dataset(field, data=f[field][()])
                
                # Topography/Bathymetry data
                if 'BATHY' in f:
                    bathy_data = f['BATHY'][:]
                    if self.compression:
                        meta.create_dataset('BATHY', data=bathy_data, compression=self.compression)
                    else:
                        meta.create_dataset('BATHY', data=bathy_data)
                
                if 'NX_BATHY' in f:
                    meta.create_dataset('NX_BATHY', data=f['NX_BATHY'][()])
                if 'NY_BATHY' in f:
                    meta.create_dataset('NY_BATHY', data=f['NY_BATHY'][()])
                if 'DX_BATHY' in f:
                    meta.create_dataset('DX_BATHY', data=f['DX_BATHY'][()])
                if 'RESOLUTION_TOPO_FILE' in f:
                    meta.create_dataset('RESOLUTION_TOPO_FILE', data=f['RESOLUTION_TOPO_FILE'][()])
                
                # Ellipticity data
                ellipticity_fields = ['rspl', 'ellipticity_spline', 'ellipticity_spline2']
                for field in ellipticity_fields:
                    if field in f:
                        ellip_data = f[field][:]
                        if self.compression:
                            meta.create_dataset(field, data=ellip_data, compression=self.compression)
                        else:
                            meta.create_dataset(field, data=ellip_data)
                
                # Reorganize xyz coordinates from (NGLOB, 3) to (NSPEC, 5, 5, 5, 3)
                self.log("Reorganizing xyz coordinates to element-major format...")
                ibool = f['ibool'][:]  # Shape: (NGLLX, NGLLY, NGLLZ, NSPEC)
                xyz_global = f['xyz'][:]  # Shape: (NGLOB, 3)
                
                # Allocate element-major coordinate array
                xyz_element = np.zeros((self.nspec, self.ngllx, self.nglly, self.ngllz, 3), 
                                      dtype=xyz_global.dtype)
                
                # Reorganize using ibool mapping
                for ispec in range(self.nspec):
                    iglob = ibool[:, :, :, ispec]  # Shape: (5, 5, 5)
                    xyz_element[ispec, :, :, :, :] = xyz_global[iglob, :]
                
                # Store reorganized coordinates
                if self.compression:
                    meta.create_dataset('xyz', data=xyz_element, compression=self.compression)
                else:
                    meta.create_dataset('xyz', data=xyz_element)
                
                self.log(f"xyz shape: {xyz_element.shape}")
        
        elapsed = time.time() - start_time
        self.log(f"Metadata file created in {elapsed:.2f}s")
    
    def get_chunk_shape(self, displacement_shape: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
        """
        Get the chunk shape for displacement dataset
        
        Parameters
        ----------
        displacement_shape : tuple
            Shape of displacement array (Nstations, 3, 3, 5, 5, 5, NSTEPS)
        
        Returns
        -------
        chunk_shape : tuple or None
            Chunk shape or None for no chunking
        """
        if self.chunks == 'none':
            return None
        elif self.chunks == 'station' or self.chunks == 'auto':
            # One station, all components/forces, all GLL, all time
            return (1, 3, 3, 5, 5, 5, self.nsteps)
        elif self.chunks == 'gll':
            # One station, one component, one force, all GLL, all time
            return (1, 1, 1, 5, 5, 5, self.nsteps)
        else:
            return True  # Let HDF5 auto-chunk
    
    def process_element(self, ispec: int, ibool: np.ndarray, 
                       station_handles: List[h5py.File]) -> np.ndarray:
        """
        Process a single element: extract displacement data from all stations
        
        Parameters
        ----------
        ispec : int
            Element index
        ibool : ndarray
            Index array (NGLLX, NGLLY, NGLLZ, NSPEC)
        station_handles : list
            Open HDF5 file handles for all stations
        
        Returns
        -------
        displacement : ndarray
            Displacement array (Nstations, 3, 3, 5, 5, 5, NSTEPS)
        """
        # Get iglob indices for this element
        iglob_element = ibool[:, :, :, ispec]  # Shape: (5, 5, 5)
        iglob_flat = iglob_element.flatten()  # Shape: (125,)
        
        # Sort indices for HDF5 access
        sorted_idx = np.argsort(iglob_flat)
        iglob_sorted = iglob_flat[sorted_idx]
        
        # Reverse sort to unsort later
        unsort_idx = np.argsort(sorted_idx)
        
        # Allocate output array
        displacement = np.zeros((self.nstations, 3, 3, 5, 5, 5, self.nsteps), dtype=np.float32)
        
        # Read from each station
        station_times = []
        for istation, fh in enumerate(station_handles):
            station_start = time.time()
            
            # Read each component (N, E, Z) and each force direction (3)
            for icomp, comp in enumerate(['N', 'E', 'Z']):
                disp_key = f'displacement/{comp}/array'
                
                # Read data at sorted iglob indices
                # displacement array shape: (3, NGLOB, NSTEPS)
                disp_data = fh[disp_key][:, iglob_sorted, :]  # Shape: (3, 125, NSTEPS)
                
                # Unsort
                disp_data = disp_data[:, unsort_idx, :]  # Shape: (3, 125, NSTEPS)
                
                # Reshape to GLL layout
                disp_data = disp_data.reshape(3, 5, 5, 5, self.nsteps)
                
                # Store in output array
                displacement[istation, icomp, :, :, :, :, :] = disp_data
            
            station_time = time.time() - station_start
            station_times.append(station_time)
            
            # Log every 25 stations or if it's taking unusually long
            if (istation + 1) % 25 == 0 or station_time > 5.0:
                avg_time = sum(station_times) / len(station_times)
                self.log(f"    Station {istation+1}/{self.nstations}: {station_time:.2f}s (avg: {avg_time:.2f}s)", 
                        rank_specific=True)
        
        total_station_time = sum(station_times)
        avg_station_time = total_station_time / self.nstations if self.nstations > 0 else 0
        self.log(f"    All stations read: total {total_station_time:.2f}s, avg {avg_station_time:.2f}s/station", 
                rank_specific=True)
        
        return displacement
    
    def write_element_file(self, ispec: int, displacement: np.ndarray):
        """
        Write element HDF5 file
        
        Parameters
        ----------
        ispec : int
            Element index
        displacement : ndarray
            Displacement array (Nstations, 3, 3, 5, 5, 5, NSTEPS)
        """
        if self.dry_run:
            # Just report what would be written
            return
        
        # Create green_functions directory if needed
        gf_dir = self.output_dir / "green_functions"
        gf_dir.mkdir(parents=True, exist_ok=True)
        
        # Element file path
        element_file = gf_dir / f"element{ispec:06d}.h5"
        
        # Write file
        with h5py.File(element_file, 'w') as f:
            # Determine chunking
            chunk_shape = self.get_chunk_shape(displacement.shape)
            
            # Create dataset
            if self.compression:
                f.create_dataset('displacement', data=displacement, 
                               compression=self.compression, chunks=chunk_shape)
            else:
                if chunk_shape is not None:
                    f.create_dataset('displacement', data=displacement, chunks=chunk_shape)
                else:
                    f.create_dataset('displacement', data=displacement)
            
            # Add element index as attribute
            f.attrs['element_index'] = ispec
    
    def load_checkpoint(self):
        """Load checkpoint file to resume processing"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                for line in f:
                    ispec = int(line.strip())
                    self.completed_elements.add(ispec)
            self.log(f"Loaded checkpoint: {len(self.completed_elements)} completed elements", 
                    rank_specific=True)
    
    def save_checkpoint(self, ispec: int):
        """Append element to checkpoint file"""
        with open(self.checkpoint_file, 'a') as f:
            f.write(f"{ispec}\n")
        self.completed_elements.add(ispec)
    
    def process_elements_parallel(self):
        """
        Process elements in parallel using MPI
        
        Each rank processes a strided subset of elements
        """
        self.log(f"Processing elements with {self.size} ranks...")
        
        # Load checkpoint
        self.load_checkpoint()
        
        # Synchronize after metadata creation
        if HAS_MPI:
            self.comm.Barrier()
        
        # Open all station files
        self.log("Opening all station files...", rank_specific=True)
        station_handles = []
        
        # Configure HDF5 file access with larger cache to handle many open files
        # rdcc_nbytes: Raw data chunk cache size (default 1MB, increase to 10MB per file)
        # rdcc_nslots: Number of chunk slots in cache (default 521, increase for better performance)
        # rdcc_w0: Preemption policy (0-1, higher = prefer to keep newly read chunks)
        fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
        fapl.set_cache(0, 997, 10*1024*1024, 0.75)  # (mdc_nelmts, rdcc_nslots, rdcc_nbytes, rdcc_w0)
        
        for idx, sfile in enumerate(self.station_files):
            fid = h5py.h5f.open(bytes(sfile), h5py.h5f.ACC_RDONLY, fapl=fapl)
            station_handles.append(h5py.File(fid))
            
            # Log progress for large station counts
            if (idx + 1) % 50 == 0:
                self.log(f"  Opened {idx+1}/{len(self.station_files)} station files", rank_specific=True)
        
        self.log(f"All {len(self.station_files)} station files opened", rank_specific=True)
        
        # Load ibool from first station file (it's the same for all stations)
        ibool = station_handles[0]['ibool'][:]
        
        # Determine elements for this rank
        if self.elements is not None:
            # User specified subset of elements
            all_elements = sorted(self.elements)
            self.log(f"Processing user-specified subset of {len(all_elements)} elements")
            # Distribute across ranks
            my_elements = [all_elements[i] for i in range(self.rank, len(all_elements), self.size)]
        else:
            # Process all elements
            my_elements = list(range(self.rank, self.nspec, self.size))
        
        # Filter out completed elements
        my_elements = [e for e in my_elements if e not in self.completed_elements]
        
        self.log(f"Processing {len(my_elements)} elements", rank_specific=True)
        
        if self.dry_run:
            # In dry-run mode, only process first element to demonstrate
            self.log("[DRY RUN] Processing first element to demonstrate...", rank_specific=True)
            my_elements = my_elements[:1] if my_elements else []
        
        # Process elements
        start_time = time.time()
        total_bytes = 0
        for idx, ispec in enumerate(my_elements):
            element_start = time.time()
            
            # Close and reopen files between elements to reset HDF5 cache
            # This prevents cache thrashing when accessing different iglob patterns
            if idx > 0:
                self.log(f"  Closing and reopening station files to reset cache...", rank_specific=True)
                for fh in station_handles:
                    fh.close()
                station_handles = []
                
                fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
                fapl.set_cache(0, 997, 10*1024*1024, 0.75)
                
                for sfile in self.station_files:
                    fid = h5py.h5f.open(bytes(sfile), h5py.h5f.ACC_RDONLY, fapl=fapl)
                    station_handles.append(h5py.File(fid))
            
            self.log(f"Processing element {ispec} ({idx+1}/{len(my_elements)})...", rank_specific=True)
            
            # Process element
            read_start = time.time()
            displacement = self.process_element(ispec, ibool, station_handles)
            read_time = time.time() - read_start
            
            # Calculate size
            element_bytes = displacement.nbytes
            total_bytes += element_bytes
            
            # Write element file (or skip in dry-run)
            write_start = time.time()
            self.write_element_file(ispec, displacement)
            write_time = time.time() - write_start
            
            element_time = time.time() - element_start
            
            self.log(f"  Element {ispec}: read {read_time:.2f}s, write {write_time:.2f}s, "
                    f"total {element_time:.2f}s, size {element_bytes / (1024**2):.2f} MB", 
                    rank_specific=True)
            
            if self.dry_run:
                self.log(f"[DRY RUN] Element {ispec}: shape {displacement.shape}", rank_specific=True)
            
            # Checkpoint (skip in dry-run)
            if not self.dry_run and (idx + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(ispec)
            
            # Progress report
            if (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                remaining = len(my_elements) - (idx + 1)
                eta = remaining / rate if rate > 0 else 0
                
                self.log(f"Progress: {idx+1}/{len(my_elements)} elements "
                        f"({rate:.2f} elem/s, ETA: {eta/60:.1f} min)", rank_specific=True)
        
        # Close all station files
        for fh in station_handles:
            fh.close()
        
        total_time = time.time() - start_time
        
        if self.dry_run:
            # Report dry-run statistics
            if len(my_elements) > 0:
                avg_bytes_per_element = total_bytes / len(my_elements)
                
                # Calculate full dataset estimates
                if self.elements is not None:
                    total_elements = len(self.elements)
                else:
                    total_elements = self.nspec
                
                estimated_total_gb = (avg_bytes_per_element * total_elements) / (1024**3)
                
                self.log(f"[DRY RUN] Element statistics:", rank_specific=True)
                self.log(f"  Average element size: {avg_bytes_per_element / (1024**2):.2f} MB", rank_specific=True)
                self.log(f"  Total elements to process: {total_elements}", rank_specific=True)
                self.log(f"  Estimated total size: {estimated_total_gb:.2f} GB", rank_specific=True)
                
                # Estimate processing time
                if total_time > 0:
                    rate = len(my_elements) / total_time
                    elements_per_rank = total_elements / self.size
                    estimated_hours = (elements_per_rank / rate) / 3600 if rate > 0 else 0
                    self.log(f"  Processing rate: {rate:.2f} elem/s", rank_specific=True)
                    self.log(f"  Estimated time per rank: {estimated_hours:.2f} hours", rank_specific=True)
        else:
            self.log(f"Completed {len(my_elements)} elements in {total_time/60:.1f} minutes", 
                    rank_specific=True)
    
    def validate_random_samples(self, n_samples: int = 10):
        """
        Validate random element/station pairs between old and new structure
        
        Parameters
        ----------
        n_samples : int
            Number of random samples to validate
        """
        if self.rank != 0:
            return
        
        # Determine which elements were actually processed
        if self.elements is not None:
            valid_elements = sorted(self.elements)
        else:
            valid_elements = list(range(self.nspec))
        
        # Limit samples to what we actually processed
        n_samples = min(n_samples, len(valid_elements))
        
        self.log(f"Validating {n_samples} random samples from {len(valid_elements)} elements...")
        
        # Load ibool from first station file
        with h5py.File(self.station_files[0], 'r') as f:
            ibool = f['ibool'][:]
        
        rng = np.random.RandomState(42)
        
        for i in range(n_samples):
            # Random element from those actually processed
            ispec = rng.choice(valid_elements)
            istation = rng.randint(0, self.nstations)
            
            # Random component and force
            icomp = rng.randint(0, 3)
            iforce = rng.randint(0, 3)
            comp_name = ['N', 'E', 'Z'][icomp]
            
            # Random time step
            istep = rng.randint(0, self.nsteps)
            
            # Read from old structure
            station_file = self.station_files[istation]
            iglob_element = ibool[:, :, :, ispec].flatten()
            sorted_idx = np.argsort(iglob_element)
            iglob_sorted = iglob_element[sorted_idx]
            unsort_idx = np.argsort(sorted_idx)
            
            with h5py.File(station_file, 'r') as f:
                old_data = f[f'displacement/{comp_name}/array'][iforce, iglob_sorted, istep]
                old_data = old_data[unsort_idx].reshape(5, 5, 5)
            
            # Read from new structure
            element_file = self.output_dir / "green_functions" / f"element{ispec:06d}.h5"
            with h5py.File(element_file, 'r') as f:
                new_data = f['displacement'][istation, icomp, iforce, :, :, :, istep]
            
            # Compare
            if not np.allclose(old_data, new_data, rtol=1e-6):
                self.log(f"VALIDATION FAILED for sample {i+1}")
                self.log(f"  Element: {ispec}, Station: {istation}, "
                        f"Component: {comp_name}, Force: {iforce}, Time: {istep}")
                self.log(f"  Max difference: {np.max(np.abs(old_data - new_data))}")
                return False
        
        self.log("Validation passed!")
        return True
    
    def run(self):
        """Main execution flow"""
        if self.rank == 0:
            self.log("="*80)
            mode_str = "[DRY RUN] " if self.dry_run else ""
            self.log(f"{mode_str}Green Function Database Reorganization")
            self.log("="*80)
            self.log(f"Input:  {self.input_dir}")
            self.log(f"Output: {self.output_dir}")
            self.log(f"Ranks:  {self.size}")
            self.log(f"Compression: {self.compression}")
            self.log(f"Chunks: {self.chunks}")
            if self.elements:
                self.log(f"Elements: {len(self.elements)} specified")
            if self.dry_run:
                self.log(f"Mode: DRY RUN (no files will be written)")
            self.log("="*80)
            
            # Create output directory
            if not self.dry_run:
                self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Scan database
        self.scan_database()
        
        # Step 2: Create metadata file (rank 0 only)
        self.create_metadata_file()
        
        # Synchronize before processing
        if HAS_MPI:
            self.comm.Barrier()
        
        # Step 3: Process elements in parallel
        self.process_elements_parallel()
        
        # Synchronize before validation
        if HAS_MPI:
            self.comm.Barrier()
        
        # Step 4: Validate (rank 0 only)
        if self.validate and not self.dry_run:
            self.validate_random_samples()
        
        if self.rank == 0:
            self.log("="*80)
            mode_str = "[DRY RUN] " if self.dry_run else ""
            self.log(f"{mode_str}Reorganization complete!")
            self.log("="*80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Reorganize Green function database from station-major to element-major",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Serial mode (testing)
  python reorganize_database.py /path/to/DB_old /path/to/DB_reorg

  # MPI parallel with 100 ranks
  mpirun -np 100 python reorganize_database.py /path/to/DB_old /path/to/DB_reorg

  # With compression and custom chunking
  mpirun -np 100 python reorganize_database.py /path/to/DB_old /path/to/DB_reorg \\
    --compression lzf --chunks station

  # Resume from checkpoint
  mpirun -np 100 python reorganize_database.py /path/to/DB_old /path/to/DB_reorg \\
    --compression lzf
        """
    )
    
    parser.add_argument('input_dir', type=str,
                       help='Input database directory')
    parser.add_argument('output_dir', type=str,
                       help='Output database directory')
    parser.add_argument('--compression', type=str, default=None,
                       choices=['lzf', 'gzip', 'none'],
                       help='Compression type (default: none)')
    parser.add_argument('--chunks', type=str, default='station',
                       choices=['station', 'gll', 'auto', 'none'],
                       help='Chunking strategy (default: station)')
    parser.add_argument('--no-validate', action='store_true',
                       help='Skip validation step')
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                       help='Checkpoint interval in elements (default: 100)')
    parser.add_argument('--elements', type=int, nargs='+', default=None,
                       help='Specific element indices to process (for testing), e.g., --elements 10 41 31 50')
    parser.add_argument('--dry-run', action='store_true',
                       help='Simulate reorganization without writing files (useful for testing)')
    
    args = parser.parse_args()
    
    # Convert 'none' string to None
    compression = None if args.compression == 'none' else args.compression
    
    # Create reorganizer
    reorganizer = DatabaseReorganizer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        compression=compression,
        chunks=args.chunks,
        validate=not args.no_validate,
        checkpoint_interval=args.checkpoint_interval,
        elements=args.elements,
        dry_run=args.dry_run
    )
    
    # Run
    try:
        reorganizer.run()
    except Exception as e:
        if reorganizer.rank == 0:
            print(f"\nERROR: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
