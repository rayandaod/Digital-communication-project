import matplotlib.pyplot as plt

import mappings
import params


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
    if params.MOD_TYPE == "qam":
        mapping = mappings.qam_map(params.M)
    elif params.MOD_TYPE == "psk":
        mapping = mappings.psk_map(params.M)
    else:
        raise ValueError('No modulation of this type was found')

    if params.verbose:
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


# TODO manage to plot without waiting for closing
