#!/usr/bin/env python

"""

Usage:

    fpull <filename1> <filename2> ... <filenameN>

This file contains a script that if located on princeton servers copies from
tigress/temp folder, and if not on Princeton servers secure copies from the same
folder. This only works if used with VPN or tigressgateway. Otherwise, Duo
access is required. Works for any number of files.


:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2021.01.17 11.45


"""

from subprocess import check_call
import socket
from sys import argv, exit


def bin():
    # Server Location
    username = "lsawade"
    hostname = "tigressdata.princeton.edu"
    tempfolder = "/tigress/lsawade/temp"

    # Current Host
    current_host = socket.gethostname()

    # Check if argument is given
    if len(argv) == 1:
        print(__doc__)
        exit()
    else:
        filename = argv[1]

    check_call(
        f'scp -r {username}@{hostname}:{tempfolder}/{filename} ./',
        shell=True)


if __name__ == "__main__":
    bin()
