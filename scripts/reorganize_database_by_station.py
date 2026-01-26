#!/usr/bin/env python
"""
Reorganize Green Function Database: Station-at-a-Time Approach

This script reorganizes a Green function database from station-major format to 
element-major format using a station-at-a-time approach with MPI parallelization.

Strategy:
- Initialize all element files (distributed across ranks)
- Process stations sequentially (or distributed across ranks)
- For each station: extract all elements in parallel, write to element files
- Only 1 input file open at a time (much better I/O than element-at-a-time)
- Each rank writes to its subset of element files

Input structure:  Network/Station/Network.Station.h5 (one file per station)
Output structure: green_functions/element{ispec:06d}.h5 (one file per element)
                  metadata.hdf5 (shared metadata and station information)

Author: Lucas Sawade
Date: January 2026
"""

import argparse
import h5py
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    print("Warning: mpi4py not available, running in serial mode")


class DatabaseReorganizerByStation:
    """Reorganize database one station at a time"""
    
    def __init__(self, input_dir: str, output_dir: str, 
                 compression: Optional[str] = None,
                 chunks: str = 'station',
                 validate: bool = True,
                 elements: Optional[List[int]] = None,
                 stations: Optional[List[int]] = None,
                 dry_run: bool = False,
                 use_nvme: bool = False):
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
        elements : list of int, optional
            Specific element indices to process (for testing)
        stations : list of int, optional
            Specific station indices to process (for testing)
        dry_run : bool
            If True, only simulate the reorganization without writing files
        use_nvme : bool
            If True, copy each station to NVMe before processing
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.compression = compression
        self.chunks = chunks
        self.validate = validate
        self.elements = elements
        self.stations = stations
        self.dry_run = dry_run
        self.use_nvme = use_nvme
        
        # MPI setup
        if HAS_MPI:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1
        
        # NVMe setup (after rank is set)
        if self.use_nvme:
            import os
            # Use $NVMe env var if available, fallback to /mnt/bb
            nvme_base = Path(os.environ.get('NVMe', os.environ.get('MNTBB', '/mnt/bb')))
            # Use user's directory - only rank 0 copies
            self.nvme_dir = nvme_base / "lsawade"
        else:
            self.nvme_dir = None
        
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
        
    def log(self, message: str, rank_specific: bool = False):
        """Print log message (only from rank 0 unless rank_specific=True)"""
        if not rank_specific and self.rank != 0:
            return
        prefix = f"[Rank {self.rank}] " if rank_specific or self.size > 1 else ""
        print(f"{prefix}{message}", flush=True)
    
    def scan_database(self):
        """Scan input database and broadcast info to all ranks"""
        if self.rank == 0:
            self.log("Scanning database...")
            start_time = time.time()
            
            # Find all HDF5 files
            station_files = []
            for network_dir in sorted(self.input_dir.iterdir()):
                if not network_dir.is_dir():
                    continue
                for station_dir in sorted(network_dir.iterdir()):
                    if not station_dir.is_dir():
                        continue
                    network = network_dir.name
                    station = station_dir.name
                    h5_file = station_dir / f"{network}.{station}.h5"
                    if h5_file.exists():
                        station_files.append(h5_file)
            
            if len(station_files) == 0:
                raise ValueError(f"No station files found in {self.input_dir}")
            
            self.log(f"Found {len(station_files)} station files")
            
            # Read dimensions
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
        """Create metadata.hdf5 with station info and shared fields (rank 0 only, skip if exists)"""
        if self.rank != 0:
            return
        
        if self.dry_run:
            self.log("[DRY RUN] Would create metadata.hdf5...")
            with h5py.File(self.station_files[0], 'r') as f:
                ibool = f['ibool'][:]
                xyz_global = f['xyz'][:]
                self.log(f"  xyz: would reorganize from {xyz_global.shape} to "
                        f"({self.nspec}, {self.ngllx}, {self.nglly}, {self.ngllz}, 3)")
            return
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_file = self.output_dir / "metadata.hdf5"
        
        # Skip if metadata file already exists (for iterative station-by-station processing)
        if metadata_file.exists():
            self.log("Metadata file already exists, skipping creation")
            return
        
        self.log("Creating metadata.hdf5...")
        start_time = time.time()
        
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
            
            dt_string = h5py.string_dtype(encoding='utf-8')
            meta.create_dataset('Networks', data=np.array(networks, dtype=object), dtype=dt_string)
            meta.create_dataset('Stations', data=np.array(stations, dtype=object), dtype=dt_string)
            meta.create_dataset('latitudes', data=np.array(latitudes, dtype=np.float64))
            meta.create_dataset('longitudes', data=np.array(longitudes, dtype=np.float64))
            meta.create_dataset('burials', data=np.array(burials, dtype=np.float64))
            
            # Copy shared fields from first station
            with h5py.File(self.station_files[0], 'r') as f:
                for field in ['DT', 'TC', 'T0', 'HDUR', 'FACTOR', 'NSTEPS']:
                    if field in f:
                        meta.create_dataset(field, data=f[field][()])
                
                for field in ['NSPEC', 'NGLLX', 'NGLLY', 'NGLLZ']:
                    if field in f:
                        meta.create_dataset(field, data=f[field][()])
                
                for field in ['TOPOGRAPHY', 'ELLIPTICITY']:
                    if field in f:
                        meta.create_dataset(field, data=f[field][()])
                
                if 'BATHY' in f:
                    bathy_data = f['BATHY'][:]
                    if self.compression:
                        meta.create_dataset('BATHY', data=bathy_data, compression=self.compression)
                    else:
                        meta.create_dataset('BATHY', data=bathy_data)
                
                for field in ['NX_BATHY', 'NY_BATHY', 'DX_BATHY', 'RESOLUTION_TOPO_FILE']:
                    if field in f:
                        meta.create_dataset(field, data=f[field][()])
                
                for field in ['rspl', 'ellipticity_spline', 'ellipticity_spline2']:
                    if field in f:
                        data = f[field][:]
                        if self.compression:
                            meta.create_dataset(field, data=data, compression=self.compression)
                        else:
                            meta.create_dataset(field, data=data)
                
                # Reorganize xyz to element-major
                self.log("Reorganizing xyz coordinates...")
                ibool = f['ibool'][:]
                xyz_global = f['xyz'][:]
                xyz_element = np.zeros((self.nspec, self.ngllx, self.nglly, self.ngllz, 3), 
                                      dtype=xyz_global.dtype)
                
                for ispec in range(self.nspec):
                    iglob = ibool[:, :, :, ispec]
                    xyz_element[ispec, :, :, :, :] = xyz_global[iglob, :]
                
                if self.compression:
                    meta.create_dataset('xyz', data=xyz_element, compression=self.compression)
                else:
                    meta.create_dataset('xyz', data=xyz_element)
        
        elapsed = time.time() - start_time
        self.log(f"Metadata file created in {elapsed:.2f}s")
    
    def get_chunk_shape(self) -> Optional[Tuple[int, ...]]:
        """Get chunk shape for displacement dataset"""
        if self.chunks == 'none':
            return None
        elif self.chunks == 'station' or self.chunks == 'auto':
            return (1, 3, 3, 5, 5, 5, self.nsteps)
        elif self.chunks == 'gll':
            return (1, 1, 1, 5, 5, 5, self.nsteps)
        else:
            return True  # Auto-chunk
    
    def initialize_element_files(self):
        """
        Initialize all element HDF5 files with empty datasets
        Distributed across ranks for parallel initialization
        Skips files that already exist (for iterative station-by-station processing)
        """
        if self.dry_run:
            self.log("[DRY RUN] Would initialize element files...", rank_specific=True)
            return
        
        self.log("Initializing element files (skipping existing)...", rank_specific=True)
        start_time = time.time()
        
        # Create output directory
        gf_dir = self.output_dir / "green_functions"
        if self.rank == 0:
            gf_dir.mkdir(parents=True, exist_ok=True)
        
        if HAS_MPI:
            self.comm.Barrier()
        
        # Determine elements for this rank
        if self.elements is not None:
            all_elements = sorted(self.elements)
            my_elements = [all_elements[i] for i in range(self.rank, len(all_elements), self.size)]
        else:
            my_elements = list(range(self.rank, self.nspec, self.size))
        
        self.log(f"  Will check/initialize {len(my_elements)} elements: {my_elements[:10]}{'...' if len(my_elements) > 10 else ''}", 
                rank_specific=True)
        
        # Displacement shape (requires nstations from database scan)
        disp_shape = (self.nstations, 3, 3, 5, 5, 5, self.nsteps)
        chunk_shape = self.get_chunk_shape()
        
        # Create files (skip if exists)
        created_count = 0
        skipped_count = 0
        
        for idx, ispec in enumerate(my_elements):
            element_file = gf_dir / f"element{ispec:06d}.h5"
            
            # Skip if file already exists
            if element_file.exists():
                skipped_count += 1
                continue
            
            # Create new file with full displacement array shape
            with h5py.File(element_file, 'w') as f:
                if self.compression:
                    f.create_dataset('displacement', shape=disp_shape, dtype=np.float32,
                                   compression=self.compression, chunks=chunk_shape)
                else:
                    if chunk_shape is not None:
                        f.create_dataset('displacement', shape=disp_shape, dtype=np.float32,
                                       chunks=chunk_shape)
                    else:
                        f.create_dataset('displacement', shape=disp_shape, dtype=np.float32)
                
                f.attrs['element_index'] = ispec
            
            created_count += 1
            
            if (created_count + skipped_count) % 100 == 0:
                self.log(f"  Processed {created_count + skipped_count}/{len(my_elements)} "
                        f"(created: {created_count}, skipped: {skipped_count})", 
                        rank_specific=True)
        
        elapsed = time.time() - start_time
        self.log(f"Element file initialization complete in {elapsed:.2f}s: "
                f"created {created_count}, skipped {skipped_count} existing files", 
                rank_specific=True)
    
    def stage_station_to_nvme(self, station_path: Path) -> Path:
        """Copy one station file to NVMe (rank 0 only, others wait)"""
        if not self.use_nvme:
            return station_path
        
        import shutil
        
        nvme_path = self.nvme_dir / "current_station.h5"
        
        # Only rank 0 copies
        if self.rank == 0:
            self.nvme_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(station_path, nvme_path)
        
        # All ranks wait for copy to finish
        if HAS_MPI:
            self.comm.Barrier()
        
        return nvme_path
    
    def cleanup_station_nvme(self):
        """Remove current station file from NVMe (rank 0 only)"""
        if not self.use_nvme:
            return
        if not self.nvme_dir:
            return
        if self.rank != 0:
            return
            
        nvme_path = self.nvme_dir / "current_station.h5"
        if nvme_path.exists():
            self.log(f"Removing {nvme_path}...")
            try:
                nvme_path.unlink()
                self.log(f"Successfully removed {nvme_path}")
            except Exception as e:
                self.log(f"Warning: Failed to remove {nvme_path}: {e}")
        else:
            self.log(f"NVMe file {nvme_path} does not exist, nothing to clean up")
    
    def cleanup_nvme_dir(self):
        """Remove NVMe directory if empty (rank 0 only)"""
        if self.use_nvme and self.nvme_dir and self.nvme_dir.exists() and self.rank == 0:
            try:
                # Only remove if directory is empty
                self.nvme_dir.rmdir()
            except OSError:
                # Directory not empty or permission denied - that's OK
                pass
    
    def process_station(self, istation: int, ibool: np.ndarray):
        """
        Process one station: extract all elements and write to element files
        
        Parameters
        ----------
        istation : int
            Station index
        ibool : ndarray
            Index array (NGLLX, NGLLY, NGLLZ, NSPEC)
        """
        station_file = self.station_files[istation]
        station_info = self.station_info[istation]
        
        self.log(f"Processing station {istation+1}/{self.nstations}: "
                f"{station_info['network']}.{station_info['station']}", rank_specific=True)
        
        station_start = time.time()
        
        # Stage to NVMe if requested
        stage_start = time.time()
        nvme_path = self.stage_station_to_nvme(station_file)
        stage_time = time.time() - stage_start
        
        if self.use_nvme and self.rank == 0:
            self.log(f"  Staged to NVMe in {stage_time:.2f}s")
        
        # Determine elements for this rank
        if self.elements is not None:
            all_elements = sorted(self.elements)
            my_elements = [all_elements[i] for i in range(self.rank, len(all_elements), self.size)]
        else:
            my_elements = list(range(self.rank, self.nspec, self.size))
        
        self.log(f"  Station {istation+1}: will process {len(my_elements)} elements: "
                f"{my_elements[:10]}{'...' if len(my_elements) > 10 else ''}", 
                rank_specific=True)
        
        # Open station file (from NVMe if staged)
        with h5py.File(nvme_path, 'r') as sf:
            # Get normalization factor for this station
            factor = sf['FACTOR'][()]
            
            # Process each element assigned to this rank
            for idx, ispec in enumerate(my_elements):
                # Get iglob indices for this element
                iglob_element = ibool[:, :, :, ispec]  # Shape: (5, 5, 5)
                iglob_flat = iglob_element.flatten()  # Shape: (125,)
                
                # Sort for efficient HDF5 access
                sorted_idx = np.argsort(iglob_flat)
                iglob_sorted = iglob_flat[sorted_idx]
                unsort_idx = np.argsort(sorted_idx)
                
                # Allocate array for this element from this station
                # Shape: (3, 3, 5, 5, 5, NSTEPS) - component, force, GLL, time
                element_data = np.zeros((3, 3, 5, 5, 5, self.nsteps), dtype=np.float32)
                
                # Read each component
                for icomp, comp in enumerate(['N', 'E', 'Z']):
                    disp_key = f'displacement/{comp}/array'
                    norm_key = f'displacement/{comp}/norm'
                    
                    # Get normalization factor for this component
                    norm = sf[norm_key][()]
                    
                    # Read: (3 forces, NGLOB, NSTEPS) -> (3, 125, NSTEPS)
                    disp_data = sf[disp_key][:, iglob_sorted, :]
                    
                    # Apply normalization: displacement * norm / factor (keep as float32)
                    disp_data = disp_data * (norm / factor)
                    
                    # Unsort and reshape
                    disp_data = disp_data[:, unsort_idx, :]
                    disp_data = disp_data.reshape(3, 5, 5, 5, self.nsteps)
                    
                    element_data[icomp, :, :, :, :, :] = disp_data
                
                # Write to element file (if not dry-run)
                if not self.dry_run:
                    gf_dir = self.output_dir / "green_functions"
                    element_file = gf_dir / f"element{ispec:06d}.h5"
                    
                    with h5py.File(element_file, 'r+') as ef:
                        # Write this station's data
                        ef['displacement'][istation, :, :, :, :, :, :] = element_data
                
                # Progress for large element counts
                if (idx + 1) % 500 == 0:
                    self.log(f"    Processed {idx+1}/{len(my_elements)} elements", 
                            rank_specific=True)
        
        elapsed = time.time() - station_start
        self.log(f"  Station {istation+1} complete: {elapsed:.2f}s, "
                f"{len(my_elements)} elements written", rank_specific=True)
        
        # Cleanup NVMe after processing
        self.cleanup_station_nvme()
    
    def process_stations_parallel(self):
        """Process all stations, with elements distributed across ranks"""
        self.log(f"Processing stations with {self.size} ranks...")
        if self.use_nvme:
            self.log("NVMe staging enabled")
        
        # Synchronize after initialization
        if HAS_MPI:
            self.comm.Barrier()
        
        # Load ibool (same for all stations)
        with h5py.File(self.station_files[0], 'r') as f:
            ibool = f['ibool'][:]
        
        # Determine which stations to process
        if self.stations is not None:
            station_list = sorted(self.stations)
            self.log(f"Processing user-specified subset of {len(station_list)} stations")
        else:
            station_list = list(range(self.nstations))
        
        if self.dry_run:
            self.log("[DRY RUN] Processing first station only...")
            station_list = station_list[:1]
        
        # Process each station (all ranks process same station in parallel)
        start_time = time.time()
        
        for istation in station_list:
            self.process_station(istation, ibool)
            
            # Synchronize after each station
            if HAS_MPI:
                self.comm.Barrier()
        
        total_time = time.time() - start_time
        
        if self.dry_run:
            self.log(f"[DRY RUN] Processed {len(station_list)} station(s) in {total_time:.2f}s")
        else:
            self.log(f"All stations processed in {total_time/60:.1f} minutes")
    
    def validate_random_samples(self, n_samples: int = 10):
        """Validate random samples between old and new structures"""
        if self.rank != 0:
            return
        
        # Determine which elements were actually processed
        if self.elements is not None:
            valid_elements = sorted(self.elements)
        else:
            valid_elements = list(range(self.nspec))
        
        # Determine which stations were actually processed
        if self.stations is not None:
            valid_stations = sorted(self.stations)
        else:
            valid_stations = list(range(self.nstations))
        
        # Limit samples to what we actually processed
        n_samples = min(n_samples, len(valid_elements) * len(valid_stations))
        
        self.log(f"Validating {n_samples} random samples from {len(valid_elements)} elements "
                f"and {len(valid_stations)} stations...")
        
        with h5py.File(self.station_files[0], 'r') as f:
            ibool = f['ibool'][:]
        
        rng = np.random.RandomState(42)
        
        for i in range(n_samples):
            ispec = rng.choice(valid_elements)
            istation = rng.choice(valid_stations)
            icomp = rng.randint(0, 3)
            iforce = rng.randint(0, 3)
            comp_name = ['N', 'E', 'Z'][icomp]
            istep = rng.randint(0, self.nsteps)
            
            # Read from old structure
            station_file = self.station_files[istation]
            iglob_element = ibool[:, :, :, ispec].flatten()
            sorted_idx = np.argsort(iglob_element)
            iglob_sorted = iglob_element[sorted_idx]
            unsort_idx = np.argsort(sorted_idx)
            
            with h5py.File(station_file, 'r') as f:
                # Apply same normalization as during reorganization
                factor = f['FACTOR'][()]
                norm = f[f'displacement/{comp_name}/norm'][()]
                
                disp_key = f'displacement/{comp_name}/array'
                old_data = f[disp_key][iforce, iglob_sorted, istep]
                old_data = old_data[unsort_idx].reshape(5, 5, 5)
                
                # Normalize: displacement * norm / factor
                old_data = old_data * (norm / factor)
            
            # Read from new structure
            element_file = self.output_dir / "green_functions" / f"element{ispec:06d}.h5"
            with h5py.File(element_file, 'r') as f:
                new_data = f['displacement'][istation, icomp, iforce, :, :, :, istep]
            
            if not np.allclose(old_data, new_data, rtol=1e-6):
                self.log(f"ERROR: Mismatch at element {ispec}, station {istation}, "
                        f"comp {comp_name}, force {iforce}, time {istep}")
                self.log(f"  Max diff: {np.max(np.abs(old_data - new_data))}")
                return False
        
        self.log("Validation passed!")
        return True
    
    def run(self):
        """Main execution flow"""
        if self.rank == 0:
            self.log("="*80)
            mode_str = "[DRY RUN] " if self.dry_run else ""
            self.log(f"{mode_str}GF Database Reorganization (Station-at-a-Time)")
            self.log("="*80)
            self.log(f"Input:  {self.input_dir}")
            self.log(f"Output: {self.output_dir}")
            self.log(f"Compression: {self.compression or 'none'}")
            self.log(f"Chunking: {self.chunks}")
            self.log(f"MPI ranks: {self.size}")
            self.log("="*80)
        
        # Step 1: Scan database
        self.scan_database()
        
        # Step 2: Create metadata file (rank 0 only)
        self.create_metadata_file()
        
        if HAS_MPI:
            self.comm.Barrier()
        
        # Step 3: Initialize element files (parallel)
        self.initialize_element_files()
        
        if HAS_MPI:
            self.comm.Barrier()
        
        # Step 4: Process stations (parallel element extraction)
        self.process_stations_parallel()
        
        if HAS_MPI:
            self.comm.Barrier()
        
        # Cleanup NVMe directories
        self.cleanup_nvme_dir()
        
        # Step 5: Validate (rank 0 only)
        if self.validate and not self.dry_run:
            self.validate_random_samples()
        
        if self.rank == 0:
            self.log("="*80)
            self.log("Reorganization complete!")
            self.log("="*80)

        # This should make sure all ranks finish before exiting
        if HAS_MPI:
            self.comm.Barrier()
        

def main():
    parser = argparse.ArgumentParser(
        description='Reorganize GF database from station-major to element-major (station-at-a-time)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to test
  mpirun -np 64 python reorganize_database_by_station.py \\
    /input/DB /output/DB --dry-run
  
  # Test with subset
  mpirun -np 64 python reorganize_database_by_station.py \\
    /input/DB /output/DB --elements 0 100 200 --stations 0 5 10
  
  # Full run with compression
  mpirun -np 64 python reorganize_database_by_station.py \\
    /input/DB /output/DB --compression lzf --chunks station
        """
    )
    
    parser.add_argument('input_dir', type=str, help='Input database directory')
    parser.add_argument('output_dir', type=str, help='Output database directory')
    parser.add_argument('--compression', type=str, choices=['lzf', 'gzip', 'none'],
                       help='Compression type (default: none)')
    parser.add_argument('--chunks', type=str, choices=['station', 'gll', 'auto', 'none'],
                       default='station', help='Chunking strategy (default: station)')
    parser.add_argument('--no-validate', action='store_true',
                       help='Skip validation of random samples')
    parser.add_argument('--elements', type=int, nargs='+',
                       help='Specific element indices to process (for testing)')
    parser.add_argument('--stations', type=int, nargs='+',
                       help='Specific station indices to process (for testing)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Simulate without writing files')
    parser.add_argument('--use-nvme', action='store_true',
                       help='Copy each station to NVMe before processing (recommended for Frontier)')
    
    args = parser.parse_args()
    
    # Handle compression
    compression = None if args.compression == 'none' else args.compression
    
    # Create reorganizer and run
    reorganizer = DatabaseReorganizerByStation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        compression=compression,
        chunks=args.chunks,
        validate=not args.no_validate,
        elements=args.elements,
        stations=args.stations,
        dry_run=args.dry_run,
        use_nvme=args.use_nvme
    )
    
    reorganizer.run()


if __name__ == '__main__':
    main()
