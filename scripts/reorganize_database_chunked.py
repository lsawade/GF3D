#!/usr/bin/env python
"""
Reorganize Green Function Database: Chunked I/O Approach

This script reorganizes a Green function database from station-major format to 
element-major format using a chunked I/O approach to minimize concurrent file operations.

Strategy:
- Process stations in chunks (e.g., 20 stations at a time)
- For each chunk: read all stations, reorganize, write to element files
- Minimizes concurrent file operations (only chunk_size stations open)
- Should scale better with rank count since I/O is bounded by chunk size

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


class DatabaseReorganizerChunked:
    """Reorganize database using chunked I/O approach"""
    
    def __init__(self, input_dir: str, output_dir: str, 
                 compression: Optional[str] = None,
                 chunks: str = 'station',
                 validate: bool = True,
                 elements: Optional[List[int]] = None,
                 stations: Optional[List[int]] = None,
                 station_chunk_size: int = 20,
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
        elements : list of int, optional
            Specific element indices to process (for testing)
        stations : list of int, optional
            Specific station indices to process (for testing)
        station_chunk_size : int
            Number of stations to process in each chunk (default: 20)
        dry_run : bool
            If True, only simulate the reorganization without writing files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.compression = compression
        self.chunks = chunks
        self.validate = validate
        self.elements = elements
        self.stations = stations
        self.station_chunk_size = station_chunk_size
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
    
    def initialize_element_files(self):
        """Initialize element files (distributed across ranks)"""
        if self.dry_run:
            return
        
        self.log("Initializing element files...", rank_specific=True)
        start_time = time.time()
        
        gf_dir = self.output_dir / "green_functions"
        if self.rank == 0:
            gf_dir.mkdir(parents=True, exist_ok=True)
        
        if HAS_MPI:
            self.comm.Barrier()
        
        if self.elements is not None:
            all_elements = sorted(self.elements)
            my_elements = [all_elements[i] for i in range(self.rank, len(all_elements), self.size)]
        else:
            my_elements = list(range(self.rank, self.nspec, self.size))
        
        disp_shape = (self.nstations, 3, 3, 5, 5, 5, self.nsteps)
        chunk_shape = self.get_chunk_shape()
        
        for idx, ispec in enumerate(my_elements):
            element_file = gf_dir / f"element{ispec:06d}.h5"
            
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
            
            if (idx + 1) % 500 == 0:
                self.log(f"  Initialized {idx+1}/{len(my_elements)} element files", 
                        rank_specific=True)
        
        elapsed = time.time() - start_time
        self.log(f"Initialized {len(my_elements)} element files in {elapsed:.2f}s", 
                rank_specific=True)
    
    def process_station_chunk(self, station_chunk: List[int], ibool: np.ndarray):
        """
        Process a chunk of stations
        
        Parameters
        ----------
        station_chunk : list of int
            Station indices in this chunk
        ibool : ndarray
            Index array (NGLLX, NGLLY, NGLLZ, NSPEC)
        """
        chunk_start = time.time()
        
        self.log(f"Processing station chunk: stations {station_chunk[0]}-{station_chunk[-1]} "
                f"({len(station_chunk)} stations)", rank_specific=(self.rank == 0))
        
        # Determine elements for this rank
        if self.elements is not None:
            all_elements = sorted(self.elements)
            my_elements = [all_elements[i] for i in range(self.rank, len(all_elements), self.size)]
        else:
            my_elements = list(range(self.rank, self.nspec, self.size))
        
        # Process each element for this rank
        process_start = time.time()
        gf_dir = self.output_dir / "green_functions"
        
        # Process each station in the chunk
        for istation in station_chunk:
            station_file = self.station_files[istation]
            station_start = time.time()
            
            # Process elements in batches to reduce file handle pressure
            batch_size = 50  # Close/reopen every 50 elements
            num_batches = (len(my_elements) + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                batch_start_idx = batch_idx * batch_size
                batch_end_idx = min((batch_idx + 1) * batch_size, len(my_elements))
                batch_elements = my_elements[batch_start_idx:batch_end_idx]
                
                # Open station file for this batch
                with h5py.File(station_file, 'r') as sf:
                    for local_idx, ispec in enumerate(batch_elements):
                        idx = batch_start_idx + local_idx
                        
                        # Get iglob indices for this element
                        iglob_element = ibool[:, :, :, ispec]
                        iglob_flat = iglob_element.flatten()
                        sorted_idx = np.argsort(iglob_flat)
                        iglob_sorted = iglob_flat[sorted_idx]
                        unsort_idx = np.argsort(sorted_idx)
                        
                        # Allocate array for this element from this station
                        element_data = np.zeros((3, 3, 5, 5, 5, self.nsteps), dtype=np.float32)
                        
                        # Process each component - read ONLY the needed iglob indices
                        for icomp, comp in enumerate(['N', 'E', 'Z']):
                            disp_key = f'displacement/{comp}/array'
                            # Read only the 125 points we need, not all NGLOB
                            data = sf[disp_key][:, iglob_sorted, :]  # (3, 125, NSTEPS)
                            data = data[:, unsort_idx, :]
                            data = data.reshape(3, 5, 5, 5, self.nsteps)
                            element_data[icomp, :, :, :, :, :] = data
                        
                        # Write to element file
                        if not self.dry_run:
                            element_file = gf_dir / f"element{ispec:06d}.h5"
                            with h5py.File(element_file, 'r+') as ef:
                                ef['displacement'][istation, :, :, :, :, :, :] = element_data
                        
                        # Progress logging (rank 0, every 100 elements)
                        if self.rank == 0 and (idx + 1) % 100 == 0:
                            elapsed = time.time() - station_start
                            rate = (idx + 1) / elapsed if elapsed > 0 else 0
                            self.log(f"    Station {istation}: {idx+1}/{len(my_elements)} elements "
                                    f"({rate:.1f} elem/s, batch {batch_idx+1}/{num_batches})")

        
        process_time = time.time() - process_start
        chunk_time = time.time() - chunk_start
        
        if self.rank == 0:
            self.log(f"  Chunk complete: process {process_time:.2f}s, total {chunk_time:.2f}s")
    
    def process_chunked(self):
        """Process all stations in chunks"""
        self.log(f"Processing with chunk size = {self.station_chunk_size} stations")
        
        if HAS_MPI:
            self.comm.Barrier()
        
        # Load ibool
        with h5py.File(self.station_files[0], 'r') as f:
            ibool = f['ibool'][:]
        
        # Determine stations to process
        if self.stations is not None:
            station_list = sorted(self.stations)
        else:
            station_list = list(range(self.nstations))
        
        # Divide into chunks
        chunks = []
        for i in range(0, len(station_list), self.station_chunk_size):
            chunks.append(station_list[i:i+self.station_chunk_size])
        
        self.log(f"Processing {len(station_list)} stations in {len(chunks)} chunks")
        
        start_time = time.time()
        
        for chunk_idx, station_chunk in enumerate(chunks):
            self.log(f"Chunk {chunk_idx+1}/{len(chunks)}")
            self.process_station_chunk(station_chunk, ibool)
            
            if HAS_MPI:
                self.comm.Barrier()
            
            # Progress estimate
            if self.rank == 0:
                elapsed = time.time() - start_time
                rate = (chunk_idx + 1) / elapsed
                remaining = len(chunks) - (chunk_idx + 1)
                eta = remaining / rate if rate > 0 else 0
                self.log(f"Progress: {chunk_idx+1}/{len(chunks)} chunks, "
                        f"ETA: {eta/60:.1f} min")
        
        total_time = time.time() - start_time
        self.log(f"All chunks processed in {total_time/60:.1f} minutes")
    
    def validate_random_samples(self, n_samples: int = 10):
        """Validate random samples"""
        if self.rank != 0:
            return
        
        if self.elements is not None:
            valid_elements = sorted(self.elements)
        else:
            valid_elements = list(range(self.nspec))
        
        if self.stations is not None:
            valid_stations = sorted(self.stations)
        else:
            valid_stations = list(range(self.nstations))
        
        n_samples = min(n_samples, len(valid_elements) * len(valid_stations))
        
        self.log(f"Validating {n_samples} random samples...")
        
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
            self.log("GF Database Reorganization (Chunked I/O)")
            self.log("="*80)
            self.log(f"Input:  {self.input_dir}")
            self.log(f"Output: {self.output_dir}")
            self.log(f"Compression: {self.compression or 'none'}")
            self.log(f"Chunking: {self.chunks}")
            self.log(f"Station chunk size: {self.station_chunk_size}")
            self.log(f"MPI ranks: {self.size}")
            self.log("="*80)
        
        self.scan_database()
        self.create_metadata_file()
        
        if HAS_MPI:
            self.comm.Barrier()
        
        self.initialize_element_files()
        
        if HAS_MPI:
            self.comm.Barrier()
        
        self.process_chunked()
        
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
        description='Reorganize GF database (chunked I/O approach)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input_dir', type=str, help='Input database directory')
    parser.add_argument('output_dir', type=str, help='Output database directory')
    parser.add_argument('--compression', type=str, choices=['lzf', 'gzip', 'none'],
                       help='Compression type (default: none)')
    parser.add_argument('--chunks', type=str, choices=['station', 'gll', 'auto', 'none'],
                       default='station', help='Chunking strategy (default: station)')
    parser.add_argument('--station-chunk-size', type=int, default=20,
                       help='Number of stations to process per chunk (default: 20)')
    parser.add_argument('--no-validate', action='store_true',
                       help='Skip validation of random samples')
    parser.add_argument('--elements', type=int, nargs='+',
                       help='Specific element indices to process (for testing)')
    parser.add_argument('--stations', type=int, nargs='+',
                       help='Specific station indices to process (for testing)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Simulate without writing files')
    
    args = parser.parse_args()
    
    compression = None if args.compression == 'none' else args.compression
    
    reorganizer = DatabaseReorganizerChunked(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        compression=compression,
        chunks=args.chunks,
        validate=not args.no_validate,
        elements=args.elements,
        stations=args.stations,
        station_chunk_size=args.station_chunk_size,
        dry_run=args.dry_run
    )
    
    reorganizer.run()


if __name__ == '__main__':
    main()
