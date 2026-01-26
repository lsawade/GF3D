#!/bin/bash
# Script to get comma-separated list of incomplete station indices

LOG_DIR="${1:-/ccs/home/lsawade/gcmt/gf3d/logs}"

cd "$LOG_DIR" || exit 1

# Find incomplete stations and extract indices
incomplete_stations=$(
  for f in reorg_station_*.out; do
    if ! grep -q "Station.*complete at" "$f" 2>/dev/null; then
      # Extract the number from filename like reorg_station_47.out
      echo "$f" | sed 's/reorg_station_\([0-9]*\)\.out/\1/'
    fi
  done | sort -n | paste -sd,
)

if [ -z "$incomplete_stations" ]; then
  echo "All stations complete!"
  exit 0
else
  echo "$incomplete_stations"
fi
