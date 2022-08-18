import numpy as np
import typing as tp
from . import utils
from .signal import filter


def create_stf(t0, tc, nstep, dt, cutoff) -> tp.Tuple[np.ndarray, np.ndarray]:

    # Get mesh header
    pass

    # Compute STF
    t = np.arange(10)
    stf = np.arange(10)

    # Low pass the create_STF
    # filter.bessel_lowpass()

    return t, stf
