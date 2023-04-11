from __future__ import annotations
from typing import TYPE_CHECKING
import time
import os
from ..utils import sec2hhmmss

if TYPE_CHECKING:
    from .cmt import CMTCatalog

def cmts2dir(C: CMTCatalog, outdir: str = "./newcatalog"):
    """Write CMTSOLUTION files to a directory all separately.

    Parameters
    ----------
    C : CMTCatalog
        Catalog to be written
    outdir : str, optional
        Directory to write files to, by default "./newcatalog"
    """

    # Create dir if doesn't exist.
    if os.path.exists(outdir) is False:
        os.mkdir(outdir)

    # Start print
    print(f"---> Writing cmts to {outdir}/")
    t0 = time.time()

    # Writing
    for _cmt in C:
        outfilename = os.path.join(outdir, _cmt.eventname)
        _cmt.write(outfilename)

    # End print
    t1 = time.time()
    print(f"     Done. Elapsed Time: {sec2hhmmss(t1-t0)[-1]}")


def cmts2file(C: CMTCatalog, outfile: str = "./catalog.txt"):
    """Writes the entire catalog of CMTSOLUTIONS to a single file, interesting
    for finite fault solutions.`

    Parameters
    ----------
    C: CMTCatalog
        Catalog to write
    outfile : str, optional
        File to write to, by default "./catalog.txt"
    """


    # Start print
    print(f"---> Writing cmts to {outfile}")
    t0 = time.time()

    # Writing
    for _i, _cmt in enumerate(C):
        if _i == 0:
            _cmt.write(outfile)
        else:
            _cmt.write(outfile, mode="a")

    # End print
    t1 = time.time()
    print(f"     Done. Elapsed Time: {sec2hhmmss(t1-t0)[-1]}")

