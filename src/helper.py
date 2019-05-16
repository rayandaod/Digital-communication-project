import matplotlib.pyplot as plt
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


def choose_mapping():
    """
    :return: The mapping corresponding to the given modulation type
    """
    if params.MOD_TYPE == "qam":
        mapping = mappings.qam_map(params.M)
    elif params.MOD_TYPE == "psk":
        mapping = mappings.psk_map(params.M)
    elif params.MOD_TYPE == "pam":
        mapping = mappings.pam_map(params.M)
    else:
        raise ValueError('No modulation of this type is defined')

    if params.verbose:
        print("Chosen mapping: {}".format(mapping))
        plot_helper.plot_complex_symbols(mapping, "Chosen mapping", "red")

    return mapping


mapping = choose_mapping()


if __name__ == "__main__":
    print("helper.py")
