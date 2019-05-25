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
        return np.arange(-int(N / 2), int(N / 2) + 1), np.concatenate((X[int(N / 2):], X[:int(N / 2) + 1]))
    else:
        # odd-length: return N values
        return np.arange(-int((N - 1) / 2), int((N - 1) / 2) + 1), np.concatenate(
            (X[int((N + 1) / 2):], X[:int((N + 1) / 2)]))


def dft_map(X, Fs=params.Fs, shift=True):
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


# # TODO additional checks on the certainty of the decision on the removed freq. range
# def find_removed_freq_range(X):
#     """
#     Checks which range of frequencies has been removed by the channel (among 1-3kHz, 3-5kHz, 5-7kHz, 7-9kHz)
#     :param X: the fourier transform of the signal
#     :return: the index in params.FREQ_RANGES corresponding to the removed frequency range
#     """
#     n_frequencies = len(params.FREQ_RANGES)
#     means = np.zeros(n_frequencies)
#     for i in range(n_frequencies):
#         means[i] = np.mean(abs(
#             np.real(
#                 X[params.FREQ_RANGES[i][0]:params.FREQ_RANGES[i][1]])) +
#                    1j * np.mean(
#             np.imag(
#                 X[params.FREQ_RANGES[i][0]:params.FREQ_RANGES[i][1]])))
#     if params.verbose:
#         print("4 frequency ranges means: {}".format(means))
#     return np.argmin(means)


# TODO awful code, change that ASAP
def find_removed_freq_range_2(samples):
    """
        Checks which range of frequencies has been removed by the channel (among 1-3kHz, 3-5kHz, 5-7kHz, 7-9kHz)
        :param samples: the samples received from the server
        :return: the index in params.FREQ_RANGES corresponding to the removed frequency range
    """
    # import matplotlib.pyplot as plt
    X = np.fft.fft(samples)
    f, Y = dft_map(X)
    # plt.plot(f, abs(Y))

    range_indices = []
    for i in range(len(params.FREQ_RANGES)):
        j = 0
        if i == 0:
            while f[j] < params.FREQ_RANGES[i][0]:
                j += 1
            range_indices.append(j)
            print("FREQUENCY {}\nIndex: {}\nValue: {}\n".format(params.FREQ_RANGES[i][0], j, f[j]))
        while f[j] < params.FREQ_RANGES[i][1]:
            j += 1
        range_indices.append(j)
        print("FREQUENCY {}\nIndex: {}\nValue: {}\n".format(params.FREQ_RANGES[i][1], j, f[j]))

    means = [np.mean(abs(Y[range_indices[0]:range_indices[1]])), np.mean(abs(Y[range_indices[1]:range_indices[2]])),
             np.mean(abs(Y[range_indices[2]:range_indices[3]])), np.mean(abs(Y[range_indices[3]:range_indices[4]]))]
    return range_indices, np.argmin(means)


def modulate_complex_samples(samples, frequencies):
    """
    Modulate the signal by shifting duplicates of it in the given frequencies
    :param samples: the signal to modulate
    :param frequencies: the frequencies we want the signal to be duplicated and shifted in
    :return: the modulated signals
    """
    n_sample = len(samples)
    time_indices = np.arange(n_sample) / params.Fs
    re_samples = np.real(samples)
    im_samples = np.imag(samples)
    new_samples = np.zeros(n_sample)
    for n in range(n_sample):
        for f in frequencies:
            new_samples[n] += re_samples[n] * np.sqrt(2) * np.cos(2 * np.pi * f * time_indices[n]) - \
                              im_samples[n] * np.sqrt(2) * np.sin(2 * np.pi * f * time_indices[n])
    return new_samples.real


def demodulate(samples, f):
    """
    Demodulate the signal
    :param samples: the signal to demodulate
    :param f: the frequency we want the signal to be modulated with
    :return: the demodulated signal
    """
    n_sample = len(samples)
    time_indices = np.arange(n_sample) / params.Fs
    new_samples = []
    for n in range(n_sample):
        new_samples.append(samples[n] * np.sqrt(2) * np.cos(2 * np.pi * f * time_indices[n]) -
                           1j * samples[n] * np.sqrt(2) * np.sin(2 * np.pi * f * time_indices[n]))
    return new_samples
