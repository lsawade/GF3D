from asyncio import constants
from collections import OrderedDict
from curses.has_key import has_key
from multiprocessing.sharedctypes import Value
import os
import sys
from turtle import write
import numpy as np
import typing as tp


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


def par2par_file(parameter: float | str | int | bool) -> str:
    """Converts a value to a specfem dictionary parameter

    Parameters
    ----------
    parameter : float | str | int | bool
        parameter value

    Returns
    -------
    str
        outputstring for writing the parameter file

    Raises
    ------
    ValueError
        raised if parameter conversion not implemented
    """

    if isinstance(parameter, float):

        # Check whether g formatting removes decimal points
        out = f'{parameter:g}'
        if '.' in out:
            out += 'd0'
        else:
            out += '.d0'

        return out

    elif isinstance(parameter, str):
        return parameter

    elif isinstance(parameter, bool):
        if parameter:
            return '.true.'
        else:
            return '.false.'

    elif isinstance(parameter, int):
        return f'{parameter:d}'

    else:
        raise ValueError(
            f'Parameter conversion not implemented for\n'
            f'P: {parameter} of type {type(parameter)}')


def checkInt(str):
    if str[0] in ('-', '+'):
        return str[1:].isdigit()
    else:
        return str.isdigit()


def par_file2par(value: str, verbose: bool = False) -> float | str | int | bool:

    if value == ".true.":
        rvalue = True

    elif value == ".false.":
        rvalue = False

    elif "d0" in value:
        rvalue = float(value.replace("d0", "0"))

    elif checkInt(value):
        rvalue = int(value)

    else:
        rvalue = value

    if verbose:
        print(f'converting {value} to {rvalue}')

    return rvalue


def get_par_file(parfile, savecomments: bool = False, verbose: bool = True) -> OrderedDict:

    pardict = OrderedDict()
    cmtcounter = 0
    noncounter = 0
    cmtblock = []
    nonblock = []

    with open(parfile, 'r') as f:
        for line in f.readlines():

            if verbose:
                print(line.strip())

            # Check for comment by removing leading (all) spaces
            if '#' == line.replace(' ', '')[0]:
                if savecomments:

                    if len(nonblock) > 0:
                        pardict[f'space-{noncounter}'] = nonblock
                        nonblock = []
                        noncounter += 1

                    cmtblock.append(line)

            # Check for empty line by stripping all spaces except '\n'
            elif '\n' == line.replace(' ', ''):

                if savecomments:
                    if len(cmtblock) > 0:
                        pardict[f'comment-{cmtcounter}'] = cmtblock
                        cmtblock = []
                        cmtcounter += 1

                    nonblock.append(line)

            elif '=' in line:

                if savecomments:

                    if len(cmtblock) > 0:
                        pardict[f'comment-{cmtcounter}'] = cmtblock
                        cmtblock = []
                        cmtcounter += 1

                    if len(nonblock) > 0:
                        pardict[f'space-{noncounter}'] = nonblock
                        nonblock = []
                        noncounter += 1

                # Get key and value
                key, val = line.split('=')[:2]
                key = key.replace(' ', '')

                # Save guard if someone puts a comment behind a value
                if '#' in val:
                    val, cmt = val.split('#')
                else:
                    cmt = None

                val = val.strip()

                # Add key and value to dictionary
                pardict[key] = par_file2par(val, verbose=verbose)

                # save comment behind value
                if savecomments:
                    if cmt is not None:
                        pardict[f'{key}-comment'] = cmt.strip()

            else:
                raise ValueError(f'Conversion of\n{line}\n not implemented.')

    return pardict


def write_par_file(pardict: OrderedDict, par_file: str | None = None, write_comments: bool = True):

    # If output file is provided open a file to write
    if par_file is not None:
        f = open(par_file, 'w')
    else:
        f = None

    for key, value in pardict.items():

        if 'comment-' in key or 'space-' in key:

            if write_comments:
                if f is not None:
                    f.writelines(value)
                    # f.write('\n')
                else:
                    for line in value:
                        print(line.strip())

        else:

            # Skip value comments
            if f'-comment' in key:
                continue

            # Fix the parfile print depending on whether value has comment
            if f'{key}-comment' in pardict and write_comments:
                outstr = f'{key:31s} = {par2par_file(value):<s}   # {pardict[f"{key}-comment"]:<s}\n'
            else:
                outstr = f"{key:31s} = {par2par_file(value):<s}\n"

            # Print string or write to file
            if f is not None:
                f.write(outstr)
            else:
                print(outstr.strip())

    # Close file if defined
    if f is not None:
        f.close()


def update_constants(
        infile: str, outfile: str | None = None,
        rotation: str = '+',
        external_stf: bool = True):

    if rotation not in ['+', '-']:
        raise ValueError('rotation must be "+", or "-".')

    # Read constants.h.in
    with open(infile, 'r') as inconstants:
        lines = inconstants.readlines()

    # Replace the set the rotation value to either + or
    newlines = []
    for line in lines:

        if 'EARTH_HOURS_PER_DAY' in line:
            if rotation == '+':
                newlines.append(
                    '  double precision, parameter :: EARTH_HOURS_PER_DAY = 24.d0\n')
            else:
                newlines.append(
                    '  double precision, parameter :: EARTH_HOURS_PER_DAY = -24.d0\n')
        elif 'EXTERNAL_SOURCE_TIME_FUNCTION' in line:
            if external_stf:
                newlines.append(
                    '  logical, parameter :: EXTERNAL_SOURCE_TIME_FUNCTION = .true.\n')
            else:
                newlines.append(
                    '  logical, parameter :: EXTERNAL_SOURCE_TIME_FUNCTION = .false.\n')
        else:
            newlines.append(line)

    # Write to out
    if outfile is not None:
        with open(outfile, 'w') as outconstants:
            outconstants.writelines(newlines)
    else:
        for line in newlines:
            print(line.rstrip())


def checktypes(checklist: list, error: str):

    # Get the type of the first thing in the list
    firsttype = type(checklist[0])

    # Check if types match
    if all([isinstance(_c, firsttype) for _c in checklist]) is False:
        raise ValueError(f'All values have to match for {error}.')

    # If array check if shapes match
    if firsttype == np.ndarray:
        shape = np.shape(checklist[0])

        if all([shape == np.shape(_c) for _c in checklist]) is False:
            raise ValueError(f'All values have to match for {error}.')
