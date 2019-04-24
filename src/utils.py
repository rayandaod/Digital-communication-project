import numpy as np
import matplotlib.pyplot as plt

import mappings

# General variables
verbose = True
message_file_path = "../data/input_lorem_ipsum.txt"
output_file_path = "../data/output_lorem_ipsum.txt"
server_hostname = "iscsrv72.epfl.ch"
server_port = 80

# Communication parameters
M = 4
bits_per_symbol = int(np.log2(M))
modulation_type = "qam"
noise_variance = 0.1
sampling_rate = 22050  # samples per second
abs_sample_interval = 1  # samples amplitude must be between -1 and 1


def string2bits(s=''):
    """
    :param s:
    :return:
    """
    return [bin(ord(x))[2:].zfill(8) for x in s]


def bits2string(b=None):
    """
    :param b: array of bits
    :return:
    """
    return ''.join([chr(int(x, 2)) for x in b])


def choose_mapping():
    """
    :return: The mapping corresponding to the given modulation type
    """
    if modulation_type == "qam":
        mapping = mappings.qam_map(M)
    elif modulation_type == "psk":
        mapping = mappings.psk_map(M)
    else:
        raise ValueError('No modulation of this type was found')

    if verbose:
        print("Chosen mapping: {}".format(mapping))
        plot_complex(mapping, "Chosen mapping", "red")

    return mapping


def plot_complex(complex_values, title, color):
    """
    :param complex_values: array of complex values to plot
    :param title: title of the plot
    :param color: color of the points
    :return: None (plot the graph)
    """
    X = [x.real for x in complex_values]
    Y = [x.imag for x in complex_values]
    plt.scatter(X, Y, color=color)
    plt.legend(['Symbols'])
    plt.title(title)
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.grid()
    plt.show()

# TODO manage to plot without waiting for closing
