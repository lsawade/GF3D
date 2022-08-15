#!/usr/bin/env python

"""

Usage:

    gf-get-dt <filename1> <filename2> ... <filenameN>

This file contains a script that if located on princeton servers copies from
tigress/temp folder, and if not on Princeton servers secure copies from the same
folder. This only works if used with VPN or tigressgateway. Otherwise, Duo
access is required. Works for any number of files.


:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2021.08.15 21.09


"""

from ..utils import get_dt_from_mesh_header

from sys import argv


def print_dt_minT():

    # Check if argument is given
    if len(argv) == 1 or len(argv) > 2:
        print(__doc__)
        exit()
    else:
        filename = argv[1]

    dt, minT = get_dt_from_mesh_header(filename)

    print('Sampling values from mesher:')
    print('----------------------------')
    print('dt:   ', dt)
    print('minT: ', minT)
