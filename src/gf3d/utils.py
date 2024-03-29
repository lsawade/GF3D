from __future__ import print_function
import requests
from urllib.request import urlopen
import zipfile
from typing import List
import urllib.request
from collections import deque
from itertools import chain
from sys import getsizeof, stderr
from collections import OrderedDict
import os
import sys
import numpy as np
import typing as tp
import toml

try:
    from tqdm import tqdm

    class DownloadProgressBar(tqdm):

        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    def downloadfile_progress(url: str, floc: str, desc: str | None = None):
        """Downloads file to location but with a progress bar
        Parameters
        ----------
        url : str
            Source URL
        floc : str
            Destination
        """

        if desc is None:
            desc = url.split('/')[-1]
        try:
            with DownloadProgressBar(unit='B', unit_scale=True,
                                     miniters=1, desc=desc) as t:

                urllib.request.urlretrieve(
                    url, filename=floc, reporthook=t.update_to)

        except Exception as e:
            print(f"Error when downloading {url}: {e}")
            raise(e)

except ImportError:
    pass

try:
    from reprlib import repr
except ImportError:
    pass


def filedir(file):
    return os.path.dirname(os.path.abspath(file))


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore


def enablePrint():
    sys.stdout = sys.__stdout__


def next_power_of_2(x):
    return int(1) if x == 0 else int(2**np.ceil(np.log2(x)))

# Read input file dict


def read_toml(file: str):
    return toml.load(file)


def timeshift(s: np.ndarray, dt: float, shift: float) -> np.ndarray:
    """ shift a signal by a given time-shift in the frequency domain
    Parameters
    ----------
    s : Arraylike
        signal
    N2 : int
        length of the signal in f-domain (usually equals the next pof 2)
    dt : float
        sampling interval
    shift : float
        the time shift in seconds
    Returns
    -------
    Timeshifted signal"""

    S = np.fft.fft(s)

    # Omega
    phshift = np.exp(-1.0j*shift*np.fft.fftfreq(len(s), dt)*2*np.pi)
    s_out = np.real(np.fft.ifft(phshift*S))
    return s_out


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
            try:
                if f'{key}-comment' in pardict and write_comments:
                    outstr = f'{key:31s} = {par2par_file(value):<s}   # {pardict[f"{key}-comment"]:<s}\n'
                else:
                    outstr = f"{key:31s} = {par2par_file(value):<s}\n"
            except ValueError as e:
                raise ValueError(
                    f'Key: {key} has value {value}. Par2parfile is not implemented for that type.')

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


def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    def dict_handler(d): return chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                    }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    # estimate sizeof object without __sizeof__
    default_size = getsizeof(0)

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


def unzip(zipfilename, directory_to_extract_to):
    with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)


def downloadfile(url: str, floc: str, *args, **kwargs):
    """Downloads file to location
    Parameters
    ----------
    url : str
        Source URL
    floc : str
        Destination
    """
    try:
        urllib.request.urlretrieve(url, floc)
    except Exception as e:
        print(f"Error when downloading {url}: {e}")
        raise(e)


def get_url_content(url) -> str:
    with urlopen(url) as response:
        body = response.read()
    return body


def sec2hhmmss(seconds: float, roundsecs: bool = True) \
        -> (int, int, float | int):
    """Turns seconds into tuple of (hours, minutes, seconds)

    Parameters
    ----------
    seconds : float
        seconds

    Returns
    -------
    Tuple
        (hours, minutes, seconds)

    Notes
    -----
    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.03.05 18.44

    """

    # Get hours
    hh = int(seconds // 3600)

    # Get minutes
    mm = int((seconds - hh * 3600) // 60)

    # Get seconds
    ss = (seconds - hh * 3600 - mm * 60)

    if roundsecs:
        ss = round(ss)

    return (hh, mm, ss)


def sec2timestamp(seconds: float) -> str:
    """Gets time stamp from seconds in format "hh h mm m ss s"

    Parameters
    ----------
    seconds : float
        Seconds to get string from

    Returns
    -------
    str
        output timestamp
    """

    hh, mm, ss = sec2hhmmss(seconds)
    return f"{int(hh):02} h {int(mm):02} m {int(ss):02} s"
