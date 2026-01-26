#!/usr/bin/env python
"""
Reorganize Green Function Database: Coordinated Element Processing

This script reorganizes a Green function database from station-major format to 
element-major format using coordinated element-by-element processing.

Strategy:
- All ranks process the SAME element at the same time
- Each rank reads from a subset of stations (Nstations / world_size)
- Rank 0 broadcasts iglob indices (computed once per element)
- Data is gathered to rank 0
- Rank 0 writes complete element file
- Move to next element synchronously

This minimizes I/O contention:
- Only ~world_size station files open at once
- iglob computed once per element
- Only 1 write per element

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


class DatabaseReorganizerCoordinated:
    """Reorganize database using coordinated element processing"""
    
    def __init__(self, input_dir: str, output_dir: str, 
                 compression: Optional[str] = None,
                 chunks: str = 'station',
                 validate: bool = True,
                 elements: Optional[List[int]] = None,
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
        dry_run : bool
            If True, only simulate the reorganization without writing files
        use_nvme : bool
            If True, copy one station at a time to NVMe before processing
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.compression = compression
        self.chunks = chunks
        self.validate = validate
        self.elements = elements
        self.dry_run = dry_run
        self.use_nvme = use_nvme
        
        # NVMe setup
        if self.use_nvme:
            import os
            self.nvme_dir = Path(os.environ.get('NVM', '/mnt/bb')) / f"gf3d_station_rank{self.rank}"
        else:
            self.nvme_dir = None
        
        # MPI setup
        if HAS_MPI:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1
        
        # Data structures
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
        """Create metadata.hdf5 (rank 0 only)"""
        if self.rank != 0:
            return
        
        if self.dry_run:
            self.log("[DRY RUN] Would create metadata.hdf5...")
            return
        
        self.log("Creating metadata.hdf5...")
        start_time = time.time()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = self.output_dir / "metadata.hdf5"
        
        with h5py.File(metadata_file, 'w') as meta:
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
            
            with h5py.File(self.station_files[0], 'r') as f:
                for field in ['DT', 'TC', 'T0', 'HDUR', 'FACTOR', 'NSTEPS', 
                             'NSPEC', 'NGLLX', 'NGLLY', 'NGLLZ',
                             'TOPOGRAPHY', 'ELLIPTICITY',
                             'NX_BATHY', 'NY_BATHY', 'DX_BATHY', 'RESOLUTION_TOPO_FILE']:
                    if field in f:
                        meta.create_dataset(field, data=f[field][()])
                
                if 'BATHY' in f:
                    bathy_data = f['BATHY'][:]
                    if self.compression:
                        meta.create_dataset('BATHY', data=bathy_data, compression=self.compression)
                    else:
                        meta.create_dataset('BATHY', data=bathy_data)
                
                for field in ['rspl', 'ellipticity_spline', 'ellipticity_spline2']:
                    if field in f:
                        data = f[field][:]
                        if self.compression:
                            meta.create_dataset(field, data=data, compression=self.compression)
                        else:
                            meta.create_dataset(field, data=data)
                
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
            return True
    
    def open_my_station_files(self, my_station_start: int, my_station_end: int) -> Dict[int, h5py.File]:
        """Open this rank's station files and keep them open"""
        opened_files = {}
        for istation in range(my_station_start, my_station_end):
            station_file = self.station_files[istation]
            # Disable chunk cache - random iglob access has no reuse
            propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
            settings = list(propfaid.get_cache())
            settings[2] = 0  # Disable chunk cache (0 bytes)
            settings[1] = 0  # No hash slots
            settings[3] = 0.0  # w0
            propfaid.set_cache(*settings)
            
            fid = h5py.h5f.open(bytes(station_file), flags=h5py.h5f.ACC_RDONLY, fapl=propfaid)
            opened_files[istation] = h5py.File(fid)
        
        return opened_files
    
    def close_my_station_files(self, opened_files: Dict[int, h5py.File]):
        """Close all opened station files"""
        for f in opened_files.values():
            f.close()
    
    def stage_station_to_nvme(self, station_path: Path) -> Path:
        """Copy one station file to NVMe"""
        if not self.use_nvme:
            return station_path
        
        import shutil
        self.nvme_dir.mkdir(parents=True, exist_ok=True)
        
        nvme_path = self.nvme_dir / "current_station.h5"
        shutil.copy2(station_path, nvme_path)
        
        return nvme_path
    
    def cleanup_station_nvme(self):
        """Remove current station file from NVMe"""
        if self.use_nvme and self.nvme_dir:
            nvme_path = self.nvme_dir / "current_station.h5"
            if nvme_path.exists():
                nvme_path.unlink()
    
    def cleanup_nvme_dir(self):
        """Remove NVMe directory entirely"""
        if self.use_nvme and self.nvme_dir and self.nvme_dir.exists():
            import shutil
            shutil.rmtree(self.nvme_dir)
    
    def process_element_coordinated(self, ispec: int, ibool: np.ndarray, 
                                          opened_files: Dict[int, h5py.File],
                                          my_stations: List[int]) -> Optional[np.ndarray]:
        """
        Process one element with all ranks working together
        
        Parameters
        ----------
        ispec : int
            Element index to process
        ibool : ndarray
            Index array (NGLLX, NGLLY, NGLLZ, NSPEC)
        opened_files : dict
            Pre-opened station files for this rank
        my_stations : list of int
            Global station indices assigned to this rank
        
        Returns
        -------
        displacement : ndarray or None
            Full displacement array on rank 0, None on other ranks
            Shape: (Nstations, 3, 3, 5, 5, 5, NSTEPS)
        """
        # Rank 0 computes and broadcasts iglob indices
        iglob_start = time.time()
        if self.rank == 0:
            iglob_element = ibool[:, :, :, ispec]
            iglob_flat = iglob_element.flatten()
            sorted_idx = np.argsort(iglob_flat)
            iglob_sorted = iglob_flat[sorted_idx]
            unsort_idx = np.argsort(sorted_idx)
        else:
            iglob_sorted = None
            unsort_idx = None
        
        # Broadcast iglob indices to all ranks
        if HAS_MPI:
            iglob_sorted = self.comm.bcast(iglob_sorted, root=0)
            unsort_idx = self.comm.bcast(unsort_idx, root=0)
        iglob_time = time.time() - iglob_start
        
        # Allocate array for this rank's stations
        my_displacement = np.zeros((len(my_stations), 3, 3, 5, 5, 5, self.nsteps), dtype=np.float32)
        
        # Read from assigned stations using pre-opened files
        read_start = time.time()
        for local_idx, istation in enumerate(my_stations):
            sf = opened_files[istation]
            
            for icomp, comp in enumerate(['N', 'E', 'Z']):
                disp_key = f'displacement/{comp}/array'
                # Read only the 125 points we need
                data = sf[disp_key][:, iglob_sorted, :]  # (3, 125, NSTEPS)
                data = data[:, unsort_idx, :]
                data = data.reshape(3, 5, 5, 5, self.nsteps)
                my_displacement[local_idx, icomp, :, :, :, :, :] = data
        read_time = time.time() - read_start
        
        # Gather all data to rank 0
        gather_start = time.time()
        if HAS_MPI:
            # Gather sizes
            sendcount = my_displacement.size
            recvcounts = self.comm.gather(sendcount, root=0)
            
            if self.rank == 0:
                # Allocate receive buffer (flattened)
                total_size = sum(recvcounts)
                recvbuf = np.zeros(total_size, dtype=np.float32)
                
                # Compute displacements for gatherv
                displs = [0]
                for count in recvcounts[:-1]:
                    displs.append(displs[-1] + count)
                
                self.comm.Gatherv(my_displacement.flatten(), 
                                 [recvbuf, recvcounts, displs, MPI.FLOAT], root=0)
                
                # Reshape back to (nstations, 3, 3, 5, 5, 5, nsteps)
                displacement = recvbuf.reshape(self.nstations, 3, 3, 5, 5, 5, self.nsteps)
                gather_time = time.time() - gather_start
                return displacement, iglob_time, read_time, gather_time
            else:
                self.comm.Gatherv(my_displacement.flatten(), None, root=0)
                gather_time = time.time() - gather_start
                return None, iglob_time, read_time, gather_time
        else:
            gather_time = time.time() - gather_start
            return my_displacement, iglob_time, read_time, gather_time
    
    def write_element_file(self, ispec: int, displacement: np.ndarray):
        """Write element file (rank 0 only)"""
        if self.rank != 0:
            return
        
        if self.dry_run:
            return
        
        gf_dir = self.output_dir / "green_functions"
        gf_dir.mkdir(parents=True, exist_ok=True)
        
        element_file = gf_dir / f"element{ispec:06d}.h5"
        chunk_shape = self.get_chunk_shape()
        
        with h5py.File(element_file, 'w') as f:
            if self.compression:
                f.create_dataset('displacement', data=displacement,
                               compression=self.compression, chunks=chunk_shape)
            else:
                if chunk_shape is not None:
                    f.create_dataset('displacement', data=displacement, chunks=chunk_shape)
                else:
                    f.create_dataset('displacement', data=displacement)
            
            f.attrs['element_index'] = ispec
    
    def process_all_elements(self):
        """Process all elements station-by-station"""
        self.log("Processing elements station-by-station with NVMe staging..." if self.use_nvme 
                else "Processing elements station-by-station...")
        
        if HAS_MPI:
            self.comm.Barrier()
        
        # Load ibool (same on all ranks)
        with h5py.File(self.station_files[0], 'r') as f:
            ibool = f['ibool'][:]
        
        # Determine elements to process
        if self.elements is not None:
            element_list = sorted(self.elements)
            self.log(f"Processing {len(element_list)} user-specified elements")
        else:
            element_list = list(range(self.nspec))
            self.log(f"Processing all {self.nspec} elements")
        
        # Compute station distribution
        stations_per_rank = self.nstations // self.size
        my_station_start = self.rank * stations_per_rank
        my_station_end = (self.rank + 1) * stations_per_rank if self.rank < self.size - 1 else self.nstations
        my_nstations = my_station_end - my_station_start
        
        if self.rank == 0:
            self.log(f"Station distribution: {stations_per_rank} stations per rank")
            self.log(f"Each rank processes {len(element_list)} elements")
        
        self.log(f"My stations: {my_station_start}-{my_station_end-1} ({my_nstations} stations)", 
                rank_specific=True)
        
        # Create list of station indices for this rank
        my_stations = list(range(my_station_start, my_station_end))
        
        start_time = time.time()
        
        # Process each element
        for elem_idx, ispec in enumerate(element_list):
            element_start = time.time()
            
            # Open files fresh for this element
            open_start = time.time()
            opened_files = self.open_my_station_files(my_station_start, my_station_end)
            open_time = time.time() - open_start
            
            # All ranks process this element together
            result = self.process_element_coordinated(ispec, ibool, opened_files, my_stations)
            displacement, iglob_time, read_time, gather_time = result
            
            # Close files immediately
            close_start = time.time()
            self.close_my_station_files(opened_files)
            close_time = time.time() - close_start
            
            # Rank 0 writes
            if self.rank == 0:
                write_start = time.time()
                self.write_element_file(ispec, displacement)
                write_time = time.time() - write_start
                
                element_time = time.time() - element_start
                
                if (elem_idx + 1) % 10 == 0 or elem_idx < 5:
                    elapsed = time.time() - start_time
                    rate = (elem_idx + 1) / elapsed
                    remaining = len(element_list) - (elem_idx + 1)
                    eta = remaining / rate if rate > 0 else 0
                    
                    self.log(f"Element {ispec} ({elem_idx+1}/{len(element_list)}): {element_time:.2f}s total | "
                            f"open:{open_time:.2f}s iglob:{iglob_time:.3f}s read:{read_time:.2f}s "
                            f"gather:{gather_time:.2f}s write:{write_time:.2f}s close:{close_time:.2f}s | "
                            f"Rate: {rate:.2f} elem/s, ETA: {eta/60:.1f} min")
            
            # Synchronize before next element
            if HAS_MPI:
                self.comm.Barrier()
        
        total_time = time.time() - start_time
        
        if self.rank == 0:
            self.log(f"Completed all {len(element_list)} elements in {total_time/60:.1f} minutes")
            self.log(f"Average: {total_time/len(element_list):.2f}s per element")
    
    def validate_random_samples(self, n_samples: int = 10):
        """Validate random samples"""
        if self.rank != 0:
            return
        
        if self.elements is not None:
            valid_elements = sorted(self.elements)
        else:
            valid_elements = list(range(self.nspec))
        
        n_samples = min(n_samples, len(valid_elements) * self.nstations)
        
        self.log(f"Validating {n_samples} random samples...")
        
        with h5py.File(self.station_files[0], 'r') as f:
            ibool = f['ibool'][:]
        
        rng = np.random.RandomState(42)
        
        for i in range(n_samples):
            ispec = rng.choice(valid_elements)
            istation = rng.randint(0, self.nstations)
            icomp = rng.randint(0, 3)
            iforce = rng.randint(0, 3)
            comp_name = ['N', 'E', 'Z'][icomp]
            istep = rng.randint(0, self.nsteps)
            
            station_file = self.station_files[istation]
            iglob_element = ibool[:, :, :, ispec].flatten()
            sorted_idx = np.argsort(iglob_element)
            iglob_sorted = iglob_element[sorted_idx]
            unsort_idx = np.argsort(sorted_idx)
            
            with h5py.File(station_file, 'r') as f:
                disp_key = f'displacement/{comp_name}/array'
                old_data = f[disp_key][iforce, iglob_sorted, istep]
                old_data = old_data[unsort_idx].reshape(5, 5, 5)
            
            element_file = self.output_dir / "green_functions" / f"element{ispec:06d}.h5"
            with h5py.File(element_file, 'r') as f:
                new_data = f['displacement'][istation, icomp, iforce, :, :, :, istep]
            
            if not np.allclose(old_data, new_data, rtol=1e-6):
                self.log(f"ERROR: Mismatch at element {ispec}, station {istation}")
                return False
        
        self.log("Validation passed!")
        return True
    
    def run(self):
        """Main execution flow"""
        if self.rank == 0:
            self.log("="*80)
            self.log("GF Database Reorganization (Coordinated)")
            self.log("="*80)
            self.log(f"Input:  {self.input_dir}")
            self.log(f"Output: {self.output_dir}")
            self.log(f"Compression: {self.compression or 'none'}")
            self.log(f"Chunking: {self.chunks}")
            self.log(f"MPI ranks: {self.size}")
            self.log("="*80)
        
        self.scan_database()
        self.create_metadata_file()
        
        if HAS_MPI:
            self.comm.Barrier()
        
        self.process_all_elements()
        
        if HAS_MPI:
            self.comm.Barrier()
        
        if self.validate and not self.dry_run:
            self.validate_random_samples()
        
        if self.rank == 0:
            self.log("="*80)
            self.log("Reorganization complete!")
            self.log("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Reorganize GF database (coordinated element processing)',
        formatter_class=argparse.RawDescriptionHelpFormatter
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
    parser.add_argument('--dry-run', action='store_true',
                       help='Simulate without writing files')
    parser.add_argument('--use-nvme', action='store_true',
                       help='Copy one station at a time to NVMe (recommended for Frontier)')
    
    args = parser.parse_args()
    
    compression = None if args.compression == 'none' else args.compression
    
    reorganizer = DatabaseReorganizerCoordinated(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        compression=compression,
        chunks=args.chunks,
        validate=not args.no_validate,
        elements=args.elements,
        dry_run=args.dry_run,
        use_nvme=args.use_nvme
    )
    
    reorganizer.run()


if __name__ == '__main__':
    main()
