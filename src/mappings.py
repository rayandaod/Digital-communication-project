import numpy as np

import params
import plot_helper


def qam_map(M):
    """
    :param M: the size of our mapping, i.e the number of symbols we can send
    :return: the array of symbols of the M-QAM (Quadrature Amplitude Modulation)
    """
    log_sqrt_m = np.log2(np.sqrt(M))
    if log_sqrt_m != np.ceil(log_sqrt_m):
        raise ValueError('Parameter[M] is not of the form 2^2K, K a positive integer.')
    N = np.sqrt(M) - 1
    aux = np.arange(-N, N + 1, 2)
    x, y = np.meshgrid(aux, aux[::-1])
    return (x + 1j * y).T.flatten()


def psk_map(M):
    """
    :param M: the size of our mapping, i.e the number of symbols we can send
    :return: the array of symbols of the M-PSK (Pulse Shift Keying)
    """
    return np.exp(1j * 2 * np.pi * np.arange(0, M) / M)


def pam_map(M):
    """
    :param M: the size of our mapping, i.e the number of symbols we can send
    :return: the array of symbols of the M-PAM (Pulse Amplitude Modulation)
    """
    if M % 2 != 0:
        raise ValueError('Parameter[M] is not even.')
    N = M - 1
    return np.arange(-N, N + 1, 2)


def choose_mapping(normalize=False):
    """
    :return: The mapping corresponding to the given mapping
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

    if params.verbose:
        print("Chosen mapping:\n{}".format(chosen_mapping))
        print("--------------------------------------------------------")
        plot_helper.plot_complex_symbols(chosen_mapping, "Chosen mapping", "red")

    return chosen_mapping


# TODO why does this work
mapping = choose_mapping(normalize=True)

# TODO make qam_map output counter-clockwise or clockwise, starting with 1+j
