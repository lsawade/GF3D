import numpy as np
import typing as tp
from .signal import filter
from scipy import special


def create_stf(
        t0, tc, nstep, dt, hdur, cutoff: float | None = None, gaussian: bool = True,
        lpfilter: str = 'bessel') -> tp.Tuple[np.ndarray, np.ndarray]:
    """Computes a source time function wih optional low pass filter.

    Parameters
    ----------
    t0 : float
        zero time
    tc : float
        center of error function
    nstep : int
        number of timesteps
    dt : sampling time
        sampling time
    hdur : float
        half duration of the gaussian/error function
    cutoff : float | None, optional
        optional low pass filter parameter, by default None
    lpfilter: str
        specify lowpass filter type. Only used if cutoff is not None. possible
        values 'bessel' (default), 'butter', 'cheby1', 'cheby2'

    Returns
    -------
    tp.Tuple[np.ndarray, np.ndarray]
        corresponding [t, stf]
    """

    # Compute STF
    t = np.arange(t0, t0 + nstep*dt, dt)

    if gaussian:
        # Use Gaussian for FORCE source
        stf = gauss(t, tc, hdur)
    else:
        # Compute step function
        stf = erf(t, tc, hdur)

    # Filter the data, and plot both the original and filtered signals.
    if cutoff is not None:
        if lpfilter == 'bessel':
            bcutoff = 1.5*cutoff
            order = 7
            stf = filter.bessel_lowpass(stf, cutoff, 1/dt, order=order)
        elif lpfilter == 'butter':
            order = 6
            stf = filter.butter_low_two_pass_filter(
                stf, cutoff, 1/dt, order=order)
        elif lpfilter == 'cheby1':
            rp1 = 0.05
            corder = 10
            stf = filter.cheby1_lowpass(
                stf, cutoff, 1/dt, order=corder, rp=rp1)
        elif lpfilter == 'cheby2':
            rp2 = 20
            corder = 10
            stf = filter.cheby2_lowpass(
                stf, cutoff, 1/dt, order=corder, rp=rp2)
        else:
            raise ValueError(f'Filter "{lpfilter}" not supported.')

    # As correct as possible
    # stf = integrate.cumtrapz(stf, x=t, initial=0)
    # stf = stf/np.max(np.abs(stf))
    # print('STF SUM', integrate.trapz(stf, x=t)

    return t, stf


def gauss(t, tc, hdur):
    """Computes specfem style Gaussian"""
    return np.exp(-(t-tc)**2/hdur**2) / (np.sqrt(np.pi) * hdur)


def erf(t, tc, hdur):
    """Computes specfem style error function."""
    return 0.5*special.erf((t-tc)/hdur)+0.5
