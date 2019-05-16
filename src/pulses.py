import numpy as np
import matplotlib.pyplot as plt

import params


def root_raised_cosine(SPAN, beta=params.BETA, T=params.T, Fs=params.Fs):
    """
    :param SPAN: number of samples in output
    :param beta: rolloff factor (0<=beta<1)
    :param T: symbol period (in seconds)
    :param Fs: sampling frequency (in Hz)
    :return: time indices, and 1-dimensional FIR (finite-impulse response) filter coefficients
    """

    if T <= 0 or SPAN <= 0 or Fs <= 0 or beta < 0 or beta > 1:
        raise AttributeError("Be careful, we must have T>0, N>0, Fs>0, 0<=beta<=1!")

    Ts = 1 / Fs  # time between each sample
    rrc = np.zeros(SPAN)
    time_indices = (np.arange(SPAN) - SPAN / 2) * Ts
    sample_numbers = np.arange(SPAN)

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
            rrc[n] = (np.sin(np.pi * t * (1 - beta) / T) + 4 * beta * (t / T) * np.cos(np.pi * t * (1 + beta) / T)) / \
                     (np.pi * t * (1 - (4 * beta * t / T) * (4 * beta * t / T)) / T)
    if params.verbose:
        print("Root-raised-cosine: N = {} samples, beta = {}, T = {} seconds, Fs = {} "
              "samples per second (Hz)".format(SPAN, beta, T, Fs))
        print()
        plt.plot(time_indices, rrc)
        plt.title("Root-raised-cosine")
        plt.xlabel("Time (in seconds)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()
    return time_indices, rrc
