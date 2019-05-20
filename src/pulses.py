import numpy as np
import time

import params
import plot_helper


def root_raised_cosine2(SPAN, beta=params.BETA, T=params.T, Fs=params.Fs):
    """
    :param SPAN: number of samples in output
    :param beta: rolloff factor (0<=beta<=1)
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
        print("Root-raised-cosine:\nN = {} samples, beta = {}, T = {} seconds, Fs = {} "
              "samples per second (Hz)".format(SPAN, beta, T, Fs))
        print("Highest value = {}".format(1 - beta + (4 * beta / np.pi)))
        print("--------------------------------------------------------")
        plot_helper.simple_plot(time_indices, rrc, "Root-raised-cosine", "Time (in seconds)", "Amplitude")
    return time_indices, rrc


def root_raised_cosine(SPAN=params.SPAN, beta=params.BETA, T=params.T, Fs=params.Fs):
    """
    :param SPAN: number of samples in output
    :param beta: rolloff factor (0<=beta<=1)
    :param T: symbol period (in seconds)
    :param Fs: sampling frequency (in Hz)
    :return: time indices, and 1-dimensional FIR (finite-impulse response) filter coefficients
    """

    if T <= 0 or SPAN <= 0 or Fs <= 0 or beta < 0 or beta > 1:
        raise AttributeError("Be careful, we must have T>0, N>0, Fs>0, 0<=beta<=1!")

    Ts = 1/Fs
    rrc = np.zeros(SPAN)
    time_indices = (np.arange(SPAN) - SPAN/2)*Ts
    sample_numbers = np.arange(SPAN)

    if beta == 0.0:
        for n in sample_numbers:
            t = time_indices[n]
            rrc[n] = (1/np.sqrt(T)) * np.sinc(t/T)
    else:
        forbidden_value_t = T / (4 * beta)
        rrc_beta_forbidden_value = (beta/(np.pi * np.sqrt(2.0 * T))) * ((np.pi + 2.0)* np.sin(np.pi/(4.0*beta)) + (np.pi - 2.0)* np.cos(np.pi/(4.0*beta)))
        first_term = (4.0 * beta/(np.pi * np.sqrt(T)))
        second_term = (1.0 - beta) * np.pi / (4.0 * beta)
        third_term = (4.0 * beta / T) ** 2
        pi = np.pi

        for n in sample_numbers:
            t = time_indices[n]
            if abs(t) == forbidden_value_t:
                rrc[n] = rrc_beta_forbidden_value
            elif beta == 1.0 or t == 0.0:
                rrc[n] = first_term * (np.cos((1.0 + beta) * np.pi * t / T) + second_term) / (1.0 - third_term*t*t)
            else:
                rrc[n] = first_term * \
                         (np.cos((1+beta) * pi * t/T) + second_term * (np.sin(pi*(1-beta)*t/T)/(np.pi*(1-beta)*t/T))) /\
                         (1 - third_term*t**2)

    if params.verbose:
        print("Root-raised-cosine:\nN = {} samples, beta = {}, T = {} seconds, Fs = {} samples per second (Hz)"
              .format(SPAN, beta, T, Fs))
        print("Highest value = {}".format(max(rrc)))
        print("--------------------------------------------------------")
        plot_helper.simple_plot(time_indices, rrc, "Root-raised-cosine", "Time (in seconds)", "Amplitude")
    return time_indices, rrc


if __name__ == "__main__":
    _, pulse = root_raised_cosine(1000)
