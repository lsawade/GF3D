import os
import sys
import numpy as np

def next_power_of_2(x):
    return int(1) if x == 0 else int(2**np.ceil(np.log2(x)))


def get_dt_from_mesh_header(specfemdir: str):

    # Header file in specfem directory
    header_file = os.path.join(
        specfemdir, 'OUTPUT_FILES', 'values_from_mesher.h')

    with open(header_file, 'r') as f:

        # Read all lines
        lines = f.readlines()
        lines = [line.strip() for line in lines]

        # Get DT
        for line in lines:

            if 'the time step of the solver will be DT' in line:
                dt = float(line.split('=')[1].strip('(s)'))

            if 'the (approximate) minimum period resolved will be' in line:
                minT = float(line.split('=')[1].strip('(s)'))

    return dt, minT
