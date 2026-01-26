#!/bin/bash
#SBATCH --job-name=reorg_gf_db
#SBATCH --account=GEO111
#SBATCH --nodes=1
#SBATCH --ntasks=56
#SBATCH --time=00:45:00
#SBATCH --constraint=nvme
#SBATCH --array=26,106,116,152,154,175%1
#SBATCH -p batch
#SBATCH --output=logs/reorg_station_%a.out
#SBATCH --error=logs/reorg_station_%a.err

# Array job to reorganize GF database one station at a time
# Each job processes a single station and writes to element files
# Element files are created on first run (station 0) and reused

module purge
module load PrgEnv-cray amd-mixed cray-mpich craype-accel-amd-gfx90a
module load core-personal hdf5-personal/1.12.3
module unload darshan-runtime

export MPLCONFIGDIR=${LUSTRE}/.matplotlib
export OMP_NUM_THREADS=1
export MPICH_GPU_SUPPORT_ENABLED=0

source ~/miniconda3/bin/activate gf

# Change directory to gf3d repo
cd /ccs/home/lsawade/gcmt/gf3d

# Create log directory if it doesn't exist
mkdir -p logs

# Input/output directories
INPUT_DIR="/lustre/orion/geo111/scratch/lsawade/DB_new/GLAD-M25/128_single"
OUTPUT_DIR="/lustre/orion/geo111/scratch/lsawade/DB_reorg"

# Station index from array task ID
STATION_ID=${SLURM_ARRAY_TASK_ID}

echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Processing station: ${STATION_ID}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "========================================"

# Run reorganization for this station
srun python scripts/reorganize_database_by_station.py \
  ${INPUT_DIR} \
  ${OUTPUT_DIR} \
  --use-nvme \
  --stations ${STATION_ID} \
  --no-validate

echo "========================================"
echo "Station ${STATION_ID} complete at $(date)"
echo "========================================"
