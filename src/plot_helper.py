import matplotlib.pyplot as plt
import numpy as np

import params
import fourier_helper


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
    if params.MAPPING == "psk":
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
    plt.suptitle(title)
    plt.ylabel("Re")
    plt.plot(indices, re)
    plt.subplot(2, 1, 2)
    plt.ylabel("Im")
    plt.plot(indices, im)
    plt.show()
    return None


def two_fft_plots(samples_1, samples_2, title, y_label_1, y_label_2):
    X = np.fft.fft(samples_1)
    Y = np.fft.fft(samples_2)

    f_x, y_x = fourier_helper.dft_map(X, shift=False)
    f_y, y_y = fourier_helper.dft_map(Y, shift=False)

    _, axs = plt.subplots(2, 1)
    fig = plt.figure(2)
    fig.suptitle(title)

    axs[0].plot(f_x, abs(y_x))
    axs[0].set_ylabel(y_label_1)
    fourier_helper.vertical_lines_frequency_ranges(axs[0])
    axs[0].set_xlim(params.FREQ_RANGES[0][0] - 1000, params.FREQ_RANGES[3][1] + 1000)

    axs[1].plot(f_y, abs(y_y))
    axs[1].set_xlabel("Frequency (in Hz)")
    axs[1].set_ylabel(y_label_2)
    fourier_helper.vertical_lines_frequency_ranges(axs[1])
    axs[1].set_xlim(params.FREQ_RANGES[0][0] - 1000, params.FREQ_RANGES[3][1] + 1000)

    plt.interactive(False)
    plt.show()


def two_simple_plots(samples_1, samples_2, title, y_label_1, y_label_2):
    _, axs = plt.subplots(2, 1)
    plt.figure(1).suptitle(title)

    axs[0].plot(range(len(samples_1)), samples_1)
    axs[0].set_ylabel(y_label_1)

    axs[1].plot(range(len(samples_2)), samples_2)
    axs[1].set_xlabel("Samples")
    axs[1].set_ylabel(y_label_2)

    axs[0].grid(True)
    axs[1].grid(True)


def fft_plot(samples, title):
    X = np.fft.fft(samples)
    f_x, y_x = fourier_helper.dft_map(X, shift=False)

    fig = plt.figure()
    fig.suptitle(title)

    plt.plot(f_x, abs(y_x))
    plt.xlabel("Frequency (in Hz)")
    plt.ylabel("Amplitude")
    fourier_helper.vertical_lines_frequency_ranges(plt)
    plt.xlim(params.FREQ_RANGES[0][0] - 1000, params.FREQ_RANGES[3][1] + 1000)

    plt.interactive(False)
    plt.show()


def simple_plot(x_axis, y_axis, title, x_label, y_label):
    plt.plot(x_axis, y_axis)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.show()
