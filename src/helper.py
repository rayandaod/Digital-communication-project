import matplotlib.pyplot as plt
import numpy as np
import math

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


mapping = choose_mapping()


def root_raised_cosine(N, beta=params.BETA, T=params.T, Fs=params.Fs):
    """
    :param N: number of samples in output
    :param beta: rolloff factor (0<=beta<1)
    :param T: symbol period (in seconds)
    :param Fs: sampling frequency (in Hz)
    :return: 1-dimensional FIR (finite-impulse response) filter coefficients
    """

    if T <= 0 or N < 0 or Fs < 0 or beta < 0 or beta >= 1:
        raise AttributeError("Be careful, we must have T>0, N>0, Fs>0, 0<=beta<1!")

    Ts = 1 / Fs  # time between each sample
    rrc = np.zeros(N)
    time_indices = (np.arange(N) - N / 2) * Ts
    sample_numbers = np.arange(N)
    for n in sample_numbers:
        t = time_indices[n]
        rrc[n] = (4 * beta / (np.pi * np.sqrt(T))) * (
                np.cos((1 + beta) * np.pi * t / T) + (1 - beta) * (np.pi / (4 * beta)) * np.sinc(
            (1 - beta) * t / T)) / (1 - (4 * beta * t / T) ** 2)
        # if math.isnan(rrc[n]):
        #     print("Nan")
    if params.verbose:
        print("Root-raised-cosine: N = {} samples, beta = {}, T = {} seconds, Fs = {} "
              "samples per second (Hz)".format(N, beta, T, Fs))
        print()
        plt.plot(time_indices, rrc)
        plt.plot(time_indices-T, rrc)
        plt.title("Root-raised-cosine")
        plt.xlabel("Time (in seconds)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()
    return time_indices, rrc


if __name__ == "__main__":
    root_raised_cosine(22050)
