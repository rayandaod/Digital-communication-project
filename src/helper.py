import numpy as np

import mappings
import params
import plot_helper


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


def choose_mapping(normalize=False):
    """
    :return: The mapping corresponding to the given mapping
    """
    if params.MAPPING == "qam":
        chosen_mapping = mappings.qam_map(params.M)
    elif params.MAPPING == "psk":
        chosen_mapping = mappings.psk_map(params.M)
    elif params.MAPPING == "pam":
        chosen_mapping = mappings.pam_map(params.M)
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


if __name__ == "__main__":
    print("helper.py")
