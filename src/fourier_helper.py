import numpy as np


import params


def dft_shift(X):
    """
    With k=0 as the center point, odd-length vectors will produce symmetric data sets with (N-1)/2 points left and
    right of the origin, whereas even-length vectors will be asymmetric, with one more point on the positive axis;
    indeed, the highest positive frequency for even-length signals will be equal to omega_{N/2} = pi. Since the
    frequencies of pi and -pi are identical, we can copy the top frequency data point to the negative axis and
    obtain a symmetric vector also for even-length signals. Here is a function that does that.
    (credits: Prandoni P. - Signal processing for Communication course at EPFL)

    :param X: the fourier transform of our signal x
    :return: a shifted version of the fourier transform (to be around 0) for even and odd length signals
    :
    """
    N = len(X)
    if N % 2 == 0:
        # even-length: return N+1 values
        return np.arange(-int(N/2), int(N/2) + 1), np.concatenate((X[int(N/2):], X[:int(N/2)+1]))
    else:
        # odd-length: return N values
        return np.arange(-int((N-1)/2), int((N-1)/2) + 1), np.concatenate((X[int((N+1)/2):], X[:int((N+1)/2)]))


def dft_map(X, Fs, shift=True):
    """
    In order to look at the spectrum of the sound file with a DFT we need to map the digital frequency "bins" of the
    DFT to real-world frequencies. The k-th basis function over C_N completes k periods over N samples.
    If the time between samples is 1/Fs, then the real-world frequency of the k-th basis function is periods over time,
    namely k(F_s/N). Let's remap the DFT coefficients using the sampling rate.
    (credits: Prandoni P. - Signal processing for Communication course at EPFL)

    :param X: the fourier transform of our signal x
    :param Fs: the sampling frequency
    :param shift: rather we want to shift the fft or not
    :return: a real-world-frequency DFT
    """
    resolution = float(Fs) / len(X)
    if shift:
        n, Y = dft_shift(X)
    else:
        Y = X
        n = np.arange(0, len(Y))
    f = n * resolution
    return f, Y


# TODO additional checks on the certainty of the decision on the removed freq. range
def find_removed_freq_range(X):
    """
    Checks which range of frequencies has been removed by the channel (among 1-3kHz, 3-5kHz, 5-7kHz, 7-9kHz)
    :param X: a fourier transform
    :return: the index in params.FREQ_RANGES corresponding to the removed frequency range
    """
    mean_1 = np.mean(X[params.FREQ_RANGES[0][0], params.FREQ_RANGES[0][1]])
    mean_2 = np.mean(X[params.FREQ_RANGES[1][0], params.FREQ_RANGES[1][1]])
    mean_3 = np.mean(X[params.FREQ_RANGES[2][0], params.FREQ_RANGES[2][1]])
    mean_4 = np.mean(X[params.FREQ_RANGES[3][0], params.FREQ_RANGES[3][1]])

    means = [mean_1, mean_2, mean_3, mean_4]
    return np.argmin(means)
