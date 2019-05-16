import matplotlib.pyplot as plt

import params


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
    return None


def plot_complex_function(complex_values, title):
    """
    :param complex_values: complex values (e.g at the output of the pulse-shaping)
    :param title: title of the plot
    :return: None (plot the graph)
    """
    re = [x.real for x in complex_values]
    im = [x.imag for x in complex_values]
    indices = range(len(complex_values))
    plt.subplot(2, 1, 1)
    plt.title("Real part of the samples")
    plt.plot(indices, re)
    plt.subplot(2, 1, 2)
    plt.title("Imaginary part of the samples")
    plt.plot(indices, im)
    plt.show()
    return None
