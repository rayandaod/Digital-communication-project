import numpy as np

import params
import plot_helper


def root_raised_cosine(SPAN=params.SPAN, beta=params.BETA, T=params.T, Fs=params.Fs, normalize=params.NORMALIZE_PULSE):
    """
    :param SPAN: number of samples in output
    :param beta: rolloff factor (0<=beta<=1)
    :param T: symbol period (in seconds)
    :param Fs: sampling frequency (in Hz)
    :param normalize: rather we normalize the rrc or not
    :return: time indices, and 1-dimensional FIR (finite-impulse response) filter coefficients
    """

    if T <= 0 or SPAN <= 0 or Fs <= 0 or beta < 0 or beta > 1:
        raise AttributeError("Be careful, we must have T>0, N>0, Fs>0, 0<=beta<=1!")

    Ts = 1 / Fs
    rrc = np.zeros(SPAN)
    time_indices = (np.arange(SPAN) - SPAN / 2) * Ts
    sample_numbers = np.arange(SPAN)

    if beta == 0:
        for n in sample_numbers:
            t = time_indices[n]
            if t != 0:
                rrc[n] = (1 / np.sqrt(T)) * (np.sin(np.pi * t / T) / (np.pi * t / T))
            else:
                rrc[n] = (1 / np.sqrt(T))

    else:
        pi = np.pi
        forbidden_value_t = T / (4 * beta)
        rrc_beta_forbidden_value = (beta / (pi * np.sqrt(2 * T))) * \
                                   ((pi + 2) * np.sin(pi / (4 * beta)) + (pi - 2) * np.cos(pi / (4 * beta)))
        first_term = (4 * beta / (pi * np.sqrt(T)))
        second_term = (1 - beta) * pi / (4 * beta)
        third_term = (4 * beta / T) ** 2

        for n in sample_numbers:
            t = time_indices[n]
            if abs(t) == forbidden_value_t:
                rrc[n] = rrc_beta_forbidden_value
            elif beta == 1 or t == 0:
                rrc[n] = first_term * (np.cos((1 + beta) * pi * t / T) + second_term) / (1 - third_term * t ** 2)
            else:
                rrc[n] = first_term * \
                         (np.cos((1 + beta) * np.pi * t / T) + second_term *
                          (np.sin(np.pi * (1 - beta) * t / T) / (np.pi * (1 - beta) * t / T))) / (
                                     1 - third_term * t ** 2)

    if normalize:
        rrc = rrc / np.linalg.norm(rrc)

    if params.verbose:
        print("Root-raised-cosine:\nSPAN = {} samples, beta = {}, T = {} seconds, Fs = {} samples per second (Hz)"
              .format(SPAN, beta, T, Fs))
        print("Highest value = {}".format(max(rrc)))
        print("--------------------------------------------------------")
        plot_helper.simple_and_fft_plots(time_indices, rrc,
                                         "Root-raised-cosine, normalized={}".format(normalize),
                                         shift=True)
    return time_indices, rrc


# Intended for testing (to run the program, run main.py)
if __name__ == "__main__":
    _, pulse = root_raised_cosine(1000)
