import numpy as np

import params
import plot_helper


def qam_map(M):
    """
    Compute the array of symbols that correspond to a M-QAM mapping

    :param M:   The size of our mapping, i.e the number of symbols we can send
    :return:    The array of symbols of the M-QAM (Quadrature Amplitude Modulation)
    """
    log_sqrt_m = np.log2(np.sqrt(M))
    if log_sqrt_m != np.ceil(log_sqrt_m):
        raise ValueError("Parameter[M] is not of the form 2^2K, K a positive integer.")
    # Implement Gray code
    if M == 4:
        return [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]
    elif M == 16:
        return [-3 - 3j, -3 - 1j, -3 + 3j, -3 + 1j, -1 - 3j, -1 - 1j, -1 + 3j, -1 + 1j, 3 - 3j, 3 - 1j, 3 + 3j, 3 + 1j,
                1 - 3j, 1 - 1j, 1 + 3j, 1 + 1j]
    else:
        raise ValueError("This mapping does not exist yet... He he he")


def psk_map(M):
    """
    Compute the array of symbols that correspond to a M-PSK mapping

    :param M:   The size of our mapping, i.e the number of symbols we can send
    :return:    The array of symbols of the M-PSK (Pulse Shift Keying)
    """
    return np.exp(1j * 2 * np.pi * np.arange(0, M) / M)


def pam_map(M):
    """
    Compute the array of symbols that correspond to a M-PAM mapping

    :param M:   The size of our mapping, i.e the number of symbols we can send
    :return:    The array of symbols of the M-PAM (Pulse Amplitude Modulation)
    """
    if M % 2 != 0:
        raise ValueError('Parameter[M] is not even.')
    N = M - 1
    return np.arange(-N, N + 1, 2)


def choose_mapping(normalize=params.NORMALIZE_MAPPING):
    """
    Choose the mapping according to the parameters in the file params.py

    :param normalize:   Rather we normalize the mapping or not
    :return:            The corresponding array of symbols
    """
    if params.MAPPING == "qam":
        chosen_mapping = qam_map(params.M)
    elif params.MAPPING == "psk":
        chosen_mapping = psk_map(params.M)
    elif params.MAPPING == "pam":
        chosen_mapping = pam_map(params.M)
    else:
        raise ValueError('No modulation of this type is defined')

    if normalize:
        chosen_mapping = chosen_mapping / np.sqrt(np.mean(np.abs(chosen_mapping) ** 2))

    if params.logs:
        print("Choosing the mapping...")
        print("Mapping: {}".format(chosen_mapping))
        print("Normalized = {}".format(params.NORMALIZE_MAPPING))
        print("--------------------------------------------------------")
    if params.plots:
        plot_helper.plot_complex_symbols(chosen_mapping, title="Chosen mapping, normalized={}".format(normalize),
                                         color="red", annotate=True)

    return chosen_mapping
