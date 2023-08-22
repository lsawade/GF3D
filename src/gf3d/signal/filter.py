from scipy.signal import butter, bessel, cheby1, cheby2, sosfilt, sosfiltfilt


def butter_bandpass(cutoffs, fs, order=5):
    return butter(order, cutoffs, fs=fs, btype='bandpass', output='sos')


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', output='sos')


def butter_lowpass_filter(data, cutoff, fs, order: int | None = 5):
    sos = butter_lowpass(cutoff, fs, order=order)
    y = sosfilt(sos, data)
    return y


def butter_low_two_pass_filter(data, cutoff, fs, order: int | None = 5):
    sos = butter_lowpass(cutoff, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y


def butter_band_two_pass_filter(data, cutoffs, fs, order: int | None = 5):
    sos = butter_bandpass(cutoffs, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y


def cheby1_low_sos(cutoff, fs, order=10, rp: float = 1):
    return cheby1(order, rp, cutoff, 'lowpass', fs=fs, output='sos')


def cheby1_lowpass(data, cutoff, fs, order: int = 10, rp: float = 10):
    sos = cheby1_low_sos(cutoff, fs, order=order, rp=rp)
    y = sosfiltfilt(sos, data)
    return y


def cheby2_low_sos(cutoff, fs, order=10, rp=10):
    return cheby2(order, rp, cutoff, 'lowpass', fs=fs, output='sos')


def cheby2_lowpass(data, cutoff, fs, order=10, rp=1):
    sos = cheby2_low_sos(cutoff, fs, order=order, rp=rp)
    y = sosfiltfilt(sos, data)
    return y


def bessel_low_sos(cutoff, fs, order=4):
    sos = bessel(order, cutoff, btype='lowpass',
                 output='sos', fs=fs, analog=False)
    return sos


def bessel_lowpass(data, cutoff, fs, order=4):
    sos = bessel_low_sos(cutoff, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y
