import matplotlib.pyplot as plt
import numpy as np
import time

import mappings
import params


def string2bits(s=''):
    """
    :param s: the string to be converted
    :return: the corresponding array of bits
    """
    return [bin(ord(x))[2:].zfill(8) for x in s]


def bits2string(b=None):
    """
    :param b: array of bits to be converted
    :return: the corresponding string
    """
    return ''.join([chr(int(x, 2)) for x in b])


def choose_mapping():
    """
    :return: The mapping corresponding to the given modulation type
    """
    if params.MOD_TYPE == "qam":
        mapping = mappings.qam_map(params.M)
    elif params.MOD_TYPE == "psk":
        mapping = mappings.psk_map(params.M)
    else:
        raise ValueError('No modulation of this type is defined')

    if params.verbose:
        print("Chosen mapping: {}".format(mapping))
        plot_complex_symbols(mapping, "Chosen mapping", "red")

    return mapping


# TODO manage to plot without waiting for closing
def plot_complex_symbols(complex_values, title, color="black"):
    """
    :param complex_values: array of complex values to plot
    :param title: title of the plot
    :param color: color of the points
    :return: None (plot the graph)
    """
    re = [x.real for x in complex_values]
    im = [x.imag for x in complex_values]

    plt.scatter(re, im, color=color)
    plt.legend(['Symbols'])
    plt.title(title)
    plt.xlabel("Re")
    plt.ylabel("Im")
    ax = plt.gca()
    if params.MOD_TYPE == "psk":
        disk1 = plt.Circle((0, 0), 1, color='k', fill=False)
        ax.add_artist(disk1)
    plt.axvline(linewidth=1, color="black")
    plt.axhline(linewidth=1, color="black")
    plt.show()
    return None


def plot_complex_function(complex_values, title):
    """
    :param complex_values: complex values (e.g at the output of the pulse-shaping)
    :param title: title of the plot
    :return: None (plot the graph)
    """
    re = [x.real for x in complex_values]
    im = [x.imag for x in complex_values]
    indices = range(len(complex_values))
    plt.subplot(2, 1, 1)
    plt.title("Real part of the samples")
    plt.plot(indices, re)
    plt.subplot(2, 1, 2)
    plt.title("Imaginary part of the samples")
    plt.plot(indices, im)
    plt.show()
    return None


def root_raised_cosine(N, beta=params.BETA, T=params.T, Fs=params.Fs):
    """
    :param N: number of samples in output
    :param beta: rolloff factor (0<=beta<1)
    :param T: symbol period (in seconds)
    :param Fs: sampling frequency (in Hz)
    :return: time indices, and 1-dimensional FIR (finite-impulse response) filter coefficients
    """

    if T <= 0 or N <= 0 or Fs <= 0 or beta < 0 or beta > 1:
        raise AttributeError("Be careful, we must have T>0, N>0, Fs>0, 0<=beta<=1!")

    Ts = 1 / Fs  # time between each sample
    rrc = np.zeros(N)
    time_indices = (np.arange(N) - N / 2) * Ts
    sample_numbers = np.arange(N)

    def rrc_helper(t):
        if t == T / (4 * beta):
           return (beta / np.sqrt(2)) * (((1 + 2 / np.pi) * (np.sin(np.pi / (4 * beta)))) + (
                        (1 - 2 / np.pi) * (np.cos(np.pi / (4 * beta)))))
        elif t == -T / (4 * beta):
           return (beta / np.sqrt(2)) * (((1 + 2 / np.pi) * (np.sin(np.pi / (4 * beta)))) + (
                        (1 - 2 / np.pi) * (np.cos(np.pi / (4 * beta)))))
        else:
            return (np.sin(np.pi * t * (1 - beta) / T) + 4 * beta * (t / T) * np.cos(np.pi * t * (1 + beta) / T)) / (
                                         np.pi * t * (1 - (4 * beta * t / T) * (4 * beta * t / T)) / T)

    for n in sample_numbers:
        t = time_indices[n]
        if t == 0.0:
            rrc[n] = 1 - beta + (4 * beta / np.pi)
        elif beta != 0.0:
            rrc[n] = rrc_helper(t)
        else:
            rrc[n] = (np.sin(np.pi * t * (1 - beta) / T) + 4 * beta * (t / T) * np.cos(np.pi * t * (1 + beta) / T)) / (np.pi * t * (1 - (4 * beta * t / T) * (4 * beta * t / T)) / T)
    if params.verbose:
        print("Root-raised-cosine: N = {} samples, beta = {}, T = {} seconds, Fs = {} "
              "samples per second (Hz)".format(N, beta, T, Fs))
        print()
        plt.plot(time_indices, rrc)
        plt.title("Root-raised-cosine")
        plt.xlabel("Time (in seconds)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()
    return time_indices, rrc


# Commpy implementation (just to test the speed of ours)
def rrcosfilter(N, alpha=params.BETA, Ts=params.T, Fs=params.Fs):
    """
    Generates a root raised cosine (RRC) filter (FIR) impulse response.
    Parameters
    ----------
    N : int
        Length of the filter in samples.
    alpha : float
        Roll off factor (Valid values are [0, 1]).
    Ts : float
        Symbol period in seconds.
    Fs : float
        Sampling Rate in Hz.
    Returns
    ---------
    time_idx : 1-D ndarray of floats
        Array containing the time indices, in seconds, for
        the impulse response.
    h_rrc : 1-D ndarray of floats
        Impulse response of the root raised cosine filter.
    """

    T_delta = 1 / float(Fs)
    time_idx = (np.arange(N) - N / 2) * T_delta
    sample_num = np.arange(N)
    h_rrc = np.zeros(N, dtype=float)

    for x in sample_num:
        t = (x - N / 2) * T_delta
        if t == 0.0:
            h_rrc[x] = 1.0 - alpha + (4 * alpha / np.pi)
        elif alpha != 0 and t == Ts / (4 * alpha):
            h_rrc[x] = (alpha / np.sqrt(2)) * (((1 + 2 / np.pi) *
                                                (np.sin(np.pi / (4 * alpha)))) + (
                                                           (1 - 2 / np.pi) * (np.cos(np.pi / (4 * alpha)))))
        elif alpha != 0 and t == -Ts / (4 * alpha):
            h_rrc[x] = (alpha / np.sqrt(2)) * (((1 + 2 / np.pi) *
                                                (np.sin(np.pi / (4 * alpha)))) + (
                                                           (1 - 2 / np.pi) * (np.cos(np.pi / (4 * alpha)))))
        else:
            h_rrc[x] = (np.sin(np.pi * t * (1 - alpha) / Ts) +
                        4 * alpha * (t / Ts) * np.cos(np.pi * t * (1 + alpha) / Ts)) / \
                       (np.pi * t * (1 - (4 * alpha * t / Ts) * (4 * alpha * t / Ts)) / Ts)

    if params.verbose:
        print("Root-raised-cosine: N = {} samples, beta = {}, T = {} seconds, Fs = {} "
              "samples per second (Hz)".format(N, alpha, Ts, Fs))
        print()
        plt.plot(time_idx, h_rrc)
        plt.title("Root-raised-cosine")
        plt.xlabel("Time (in seconds)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()

    return time_idx, h_rrc


def maximum_likelihood_sync(received_signal, training_sequence=params.PREAMBLE):
    """
    Synchronizes the received signal, i.e returns the number of samples after which the data signal begins.\n

    - We first check which range of frequencies has been removed by the channel (among 1-3kHz, 3-5kHz, 5-7kHz,
    7-9kHz) thanks to a Fourier-transform on the received signal.

    - Then we remove the corresponding frequency components from our original training sequence and correlate the
    received signal with the modified training sequence to aim for the highest scalar product, which will correspond
    to the delay.\n

    :param received_signal: signal received from the server
    :param training_sequence: real-valued sequence used to synchronize the received signal
    :return: delay in number of samples
    """

    return None

mapping = choose_mapping()

# TODO speed up our RRC
if __name__ == "__main__":
    start = time.time()
    _, a = root_raised_cosine(100000)
    intermediate = time.time()
    _, b =rrcosfilter(100000)
    end = time.time()
    print("My rrc: {}\nCommpy's rrc: {}".format(intermediate - start, end - intermediate))
    print((a == b).all())
