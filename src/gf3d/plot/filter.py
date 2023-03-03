from scipy.signal import sosfreqz, resample
# from scipy import signal
from scipy.interpolate import interp1d
from scipy.fft import fftfreq, fft, next_fast_len
import matplotlib.pyplot as plt
import numpy as np
from ..signal.filter import butter_lowpass, butter_lowpass_filter, \
    butter_low_two_pass_filter,  cheby1_lowpass, cheby1_low_sos, \
    cheby2_lowpass, cheby2_low_sos, bessel_low_sos, bessel_lowpass


def resample(t, s, tn):
    return interp1d(t, s)(tn)


def plot_frequency_response(t, data, cutoff, nfs=None, order=None):
    """Plot frequency response of a lowpass filter."""

    # Sample spacing
    dt = np.diff(t)[0]

    # Sampling frequency
    fs = 1.0/dt

    # Associated frequency vector
    fvec = fftfreq(len(data), d=dt)

    # Get the filter coefficients so we can check its frequency response.
    sos = butter_lowpass(cutoff, fs, order=order)

    # Filter the data, and plot both the original and filtered signals.
    fdata = butter_lowpass_filter(data, cutoff, fs, order=order)
    fdata2 = butter_low_two_pass_filter(data, cutoff, fs, order=order)

    # Plot the frequency response.
    w, h = sosfreqz(sos, fs=fs, worN=8000)

    # Resample if nfs is none
    if nfs is not None:
        ndt = 1/nfs
        tn = np.arange(t[0], t[-1], ndt)

        ndata = resample(t, data, tn)
        nfdata = resample(t, fdata, tn)
        nfdata2 = resample(t, fdata2, tn)
    else:
        ndt = dt
        tn = t
        ndata = data
        nfdata = fdata
        nfdata2 = fdata2
        pass

    # Associated frequency vector
    fvec = fftfreq(len(ndata), d=ndt)

    # For FFT
    N = len(ndata)
    NFL = next_fast_len(N)  # N  #

    # FFT of the data and filtered signals
    # Next power of 2 gives very weird results
    Fdata = np.abs(fft(ndata, n=NFL)[:N])
    Ffdata = np.abs(fft(nfdata, n=NFL)[:N])
    Ffdata2 = np.abs(fft(nfdata2, n=NFL)[:N])

    fig = plt.figure(figsize=(10, 10))
    axes = []
    axes.append(plt.subplot(2, 1, 1))
    plt.plot(w, np.abs(h), 'k--', label='Response', alpha=0.75)
    # plt.plot(fvec, Fdata/np.max(Fdata), 'k', label='SF out')
    # plt.plot(fvec, Ffdata/np.max(Ffdata), 'b', label='Filt. Spectrum')
    plt.plot(fvec, Ffdata2/np.max(Ffdata2), 'g', label='SF filt.')
    plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
    plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5*fs)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()
    plt.legend(frameon=False)

    axes.append(plt.subplot(2, 1, 2))
    plt.plot(tn, ndata, 'k-', label='SF out')
    # plt.plot(t, fdata, 'b-', linewidth=2, label='Filtered data')
    plt.plot(tn, nfdata2, 'g-', linewidth=2, label='SF filt.')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend(frameon=False, loc='lower left')

    plt.subplots_adjust(hspace=0.35)

    return (fig, axes)


def compare_lowpass_filters(
        t, data, cutoff, bcutoff,
        ndt=None, border=6, corder=10, bessorder=5, rp1=1, rp2=10):
    """Plot frequency response of a lowpass filter."""

    # Sample spacing
    dt = np.diff(t)[0]

    # Sampling frequency
    fs = 1.0/dt

    # Get the filter coefficients so we can check its frequency response.
    butter_sos = butter_lowpass(cutoff, fs, order=border)
    cheby1_sos = cheby1_low_sos(cutoff, fs, order=corder, rp=rp1)
    cheby2_sos = cheby2_low_sos(cutoff, fs, order=corder, rp=rp2)
    bessel_sos = bessel_low_sos(bcutoff, fs, order=bessorder)

    # Filter the data, and plot both the original and filtered signals.
    fbutter = butter_low_two_pass_filter(data, cutoff, fs, order=border)
    fcheby1 = cheby1_lowpass(data, cutoff, fs, order=corder, rp=rp1)
    fcheby2 = cheby2_lowpass(data, cutoff, fs, order=corder, rp=rp2)
    fbessel = bessel_lowpass(data, bcutoff, fs, order=bessorder)

    # Create some other filters
    # numtaps = 1547
    # bands = np.array([0, cutoff, cutoff*1.01, fs/2])
    # desired = np.array([1, 1, 0, 0])
    # weights = np.array([100, 1])
    # fir_firls = signal.firls(numtaps, bands, desired, fs=fs, weight=weights)
    # rweights = np.array([10, 1])
    # fir_remez = signal.remez(
    #     numtaps, bands, desired[::2], fs=fs, weight=rweights)
    # fir_firwin2 = signal.firwin2(numtaps, bands, desired, fs=fs)

    # Finite impulse response filters
    # flabels = ['LS', 'Remez', 'FIRwin2']
    # firs = [fir_firls, fir_remez, fir_firwin2]
    # firfreqs = [signal.freqz(_fir, fs=fs, worN=8000) for _fir in firs]
    # firfdata = [signal.filtfilt(_fir, 1, data) for _fir in firs]
    # firFdata = [np.abs(fft(_firfdata, n=NFL)[:N]) for _firfdata in firfdata]
    # firgdata = [np.gradient(_firfdata, t) for _firfdata in firfdata]

    # FFT of the data and filtered signals
    # Next power of 2 gives very weird results
    if ndt is not None:
        tn = np.arange(t[0], t[-1], ndt)
        fbutter = resample(t, fbutter, tn)
        fcheby1 = resample(t, fcheby1, tn)
        fcheby2 = resample(t, fcheby2, tn)
        fbessel = resample(t, fbessel, tn)
    else:
        tn = t
        ndt = dt

    # For FFT
    N = len(fbutter)
    NFL = next_fast_len(N)  # N  #

    # Associated frequency vector
    fvec = fftfreq(len(fbutter), d=ndt)

    Fbutter = np.abs(fft(fbutter, n=NFL)[:N])
    Fcheby1 = np.abs(fft(fcheby1, n=NFL)[:N])
    Fcheby2 = np.abs(fft(fcheby2, n=NFL)[:N])
    Fbessel = np.abs(fft(fbessel, n=NFL)[:N])

    # Plot the frequency response.
    butterw, butterh = sosfreqz(butter_sos, fs=fs, worN=8000)
    cheby1w, cheby1h = sosfreqz(cheby1_sos, fs=fs, worN=8000)
    cheby2w, cheby2h = sosfreqz(cheby2_sos, fs=fs, worN=8000)
    besselw, besselh = sosfreqz(bessel_sos, fs=fs, worN=8000)

    fig = plt.figure(figsize=(10, 10))
    axes = []
    axes.append(plt.subplot(3, 1, 1))
    plt.plot(butterw, np.abs(butterh), 'r--', label='Butter Resp.', alpha=0.75)
    plt.plot(cheby1w, np.abs(cheby1h), 'g--', label='Cheby1 Resp.', alpha=0.75)
    plt.plot(cheby2w, np.abs(cheby2h), 'b--', label='Cheby2 Resp.', alpha=0.75)
    plt.plot(besselw, np.abs(besselh), 'c--', label='Bessel Resp.', alpha=0.75)
    plt.plot(fvec, Fbutter/np.max(Fbutter), 'r', label='Butter')
    plt.plot(fvec, Fcheby1/np.max(Fcheby1), 'g', label='Cheby1')
    plt.plot(fvec, Fcheby2/np.max(Fcheby2), 'b', label='Cheby2')
    plt.plot(fvec, Fbessel/np.max(Fbessel), 'c', label='Bessel')
    # for _labl, _firfreq in zip(flabels, firfreqs):
    #     plt.plot(_firfreq[0],  np.abs(_firfreq[1]) /
    #              np.max(np.abs(_firfreq[1])), label=f'{_labl} Resp.')
    # for _labl, _firFdata in zip(flabels, firFdata):
    #     plt.plot(fvec,  np.abs(_firFdata) /
    #              np.max(np.abs(_firFdata)), label=f'{_labl}')
    plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.xscale('log')
    plt.grid()
    plt.legend(frameon=False)

    axes.append(plt.subplot(3, 1, 2))
    plt.plot(t, data, 'k-', label='SF out')
    plt.plot(tn, fbutter, 'r-', linewidth=2, label='Butter.')
    plt.plot(tn, fcheby1, 'g-', linewidth=2, label='Cheby1')
    plt.plot(tn, fcheby2, 'b-', linewidth=2, label='Cheby2')
    plt.plot(tn, fbessel, 'c-', linewidth=2, label='Bessel')
    # for _labl, _firfdata in zip(flabels, firfdata):
    #     plt.plot(t, _firfdata/np.max(_firfdata), label=f'{_labl}')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend(frameon=False, loc='lower left')

    axes.append(plt.subplot(3, 1, 3))
    gdata = np.gradient(data, t)
    gbutter = np.gradient(fbutter, tn)
    gcheby1 = np.gradient(fcheby1, tn)
    gcheby2 = np.gradient(fcheby2, tn)
    gbessel = np.gradient(fbessel, tn)
    plt.plot(t, gdata/np.max(gdata), 'k-', label='Data')
    plt.plot(tn, gbutter/np.max(gbutter), 'r-', linewidth=2, label='Butter.')
    plt.plot(tn, gcheby1/np.max(gcheby1), 'g-', linewidth=2, label='Cheby1')
    plt.plot(tn, gcheby2/np.max(gcheby2), 'b-', linewidth=2, label='Cheby2')
    plt.plot(tn, gbessel/np.max(gbessel), 'c-', linewidth=2, label='Bessel')
    # for _labl, _firgdata in zip(flabels, firgdata):
    #     plt.plot(t, _firgdata/np.max(_firgdata), label=f'{_labl}')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend(frameon=False, loc='lower left')

    plt.subplots_adjust(hspace=0.35)

    return (fig, axes)
