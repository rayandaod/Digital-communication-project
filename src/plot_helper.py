import matplotlib.pyplot as plt
import numpy as np

import fourier_helper
import params


def plot_complex_symbols(complex_values, title, color="black", annotate=False):
    """
    :param complex_values: array of complex values to plot
    :param title: title of the plot
    :param color: color of the points
    :param annotate: rather we annotate the symbols or not
    :return: None (plot the graph)
    """
    re = [x.real for x in complex_values]
    im = [x.imag for x in complex_values]

    plt.scatter(re, im, color=color)
    plt.title(title)
    plt.xlabel("Re")
    plt.ylabel("Im")
    ax = plt.gca()

    if params.MAPPING == "psk":
        disk1 = plt.Circle((0, 0), 1, color='k', fill=False)
        ax.add_artist(disk1)

    if params.MAPPING == "pam":
        for c in complex_values:
            ax.annotate('({0:.2f}'.format(c), xy=(c, 0.001))
    elif annotate:
        for c in complex_values:
            ax.annotate('({0: .2f} {1} {2:.2f}j)'
                        .format(c.real, '+-'[c.imag < 0], abs(c.imag)), xy=(np.real(c), np.imag(c) + 0.001))

    plt.axvline(linewidth=1, color="black")
    plt.axhline(linewidth=1, color="black")

    plt.interactive(False)
    plt.show()
    return None


def plot_complex_function(complex_values, title, dots=False):
    """
    :param complex_values: complex values (e.g at the output of the pulse-shaping)
    :param title: title of the plot
    :param dots: rather we want dots or not
    :return: None (plot the graph)
    """
    re = [x.real for x in complex_values]
    im = [x.imag for x in complex_values]
    indices = range(len(complex_values))
    plt.subplot(2, 1, 1)
    plt.suptitle(title)
    plt.ylabel("Re")
    if dots:
        plt.plot(indices, re, 'o')
    else:
        plt.plot(indices, re)
    plt.subplot(2, 1, 2)
    plt.ylabel("Im")
    plt.plot(indices, im)
    plt.interactive(False)
    plt.show()
    return None


def vertical_lines_frequency_ranges(plot):
    """
    Plots 4 vertical red lines showing the frequency ranges we are interested in
    :param plot: the current plot
    :return:
    """
    for i in np.arange(len(params.FREQ_RANGES)):
        if i == 0:
            plot.axvline(x=-params.FREQ_RANGES[i][0], color='r')
            plot.axvline(x=params.FREQ_RANGES[i][0], color='r')
        plot.axvline(x=params.FREQ_RANGES[i][1], color='r')
        plot.axvline(x=-params.FREQ_RANGES[i][1], color='r')


def two_fft_plots(samples_1, samples_2, title, y_label_1, y_label_2):
    """
    Plots 2 fft plots, one above the other
    :param samples_1: the first sample array
    :param samples_2: the second sample array
    :param title: the title for both plots
    :param y_label_1: the first y axis label
    :param y_label_2: the second y axis label
    :return: None
    """
    X = np.fft.fft(samples_1)
    Y = np.fft.fft(samples_2)

    f_x, y_x = fourier_helper.dft_map(X, shift=False)
    f_y, y_y = fourier_helper.dft_map(Y, shift=False)

    fig, axs = plt.subplots(2, 1)
    fig.suptitle(title)

    axs[0].plot(f_x, abs(y_x))
    axs[0].set_ylabel(y_label_1)
    vertical_lines_frequency_ranges(axs[0])
    axs[0].set_xlim(params.FREQ_RANGES[0][0] - 1000, params.FREQ_RANGES[3][1] + 1000)

    axs[1].plot(f_y, abs(y_y))
    axs[1].set_xlabel("Frequency (in Hz)")
    axs[1].set_ylabel(y_label_2)
    vertical_lines_frequency_ranges(axs[1])
    axs[1].set_xlim(params.FREQ_RANGES[0][0] - 1000, params.FREQ_RANGES[3][1] + 1000)

    plt.interactive(False)
    plt.show()
    return None


def two_simple_plots(samples_1, samples_2, title, y_label_1, y_label_2):
    """
    Plots 2 simple plots, one above the other
    :param samples_1: the first sample array
    :param samples_2: the second sample array
    :param title: the title for both plots
    :param y_label_1: the first y axis label
    :param y_label_2: the second y axis label
    :return: None
    """
    fig, axs = plt.subplots(2, 1)
    fig.suptitle(title)

    axs[0].plot(range(len(samples_1)), samples_1)
    axs[0].set_ylabel(y_label_1)

    axs[1].plot(range(len(samples_2)), samples_2)
    axs[1].set_xlabel("Samples")
    axs[1].set_ylabel(y_label_2)

    axs[0].grid(True)
    axs[1].grid(True)

    plt.interactive(False)
    plt.show()
    return None


def fft_plot(samples, title, shift=False):
    """
    Plots a simple fft plot
    :param shift: rather we shift the fft or not
    :param samples: the sample array
    :param title: the title for the plot
    :return: None
    """
    X = np.fft.fft(samples)
    f_x, y_x = fourier_helper.dft_map(X, shift=shift)

    fig, axs = plt.subplots(2, 1)
    fig.suptitle(title)

    axs[0].plot(f_x, abs(np.real(y_x)))
    axs[0].set_xlabel("Frequency (in Hz)")
    axs[0].set_ylabel("Real")

    axs[1].plot(f_x, abs(np.imag(y_x)))
    axs[1].set_xlabel("Frequency (in Hz)")
    axs[1].set_ylabel("Imaginary")

    vertical_lines_frequency_ranges(axs[0])
    vertical_lines_frequency_ranges(axs[1])

    plt.subplots_adjust(hspace=0.5)
    plt.interactive(False)
    plt.show()
    return None


def simple_plot(x_axis, y_axis, title):
    """
    Plots a simple plot
    :param x_axis: the x axis
    :param y_axis: the y axis
    :param title: the title of the plot
    :return: None
    """
    plt.plot(x_axis, y_axis)

    plt.title(title)
    plt.xlabel("Amplitude")
    plt.ylabel("Time (in seconds)")

    plt.grid()
    plt.interactive(False)
    plt.show()
    return None


def simple_and_fft_plots(time_indices, samples, title, shift=False):
    fig, axs = plt.subplots(2, 1)
    fig.suptitle(title)

    axs[0].plot(time_indices, samples)
    axs[0].set_ylabel("Amplitude")
    axs[0].set_xlabel("Time (in seconds)")

    X = np.fft.fft(samples)
    f_x, y_x = fourier_helper.dft_map(X, shift=shift)

    axs[1].plot(f_x, abs(y_x))
    axs[1].set_xlabel("Frequency (in Hertz)")
    axs[1].set_ylabel("Amplitude")

    axs[0].grid(True)
    axs[1].grid(True)

    plt.subplots_adjust(hspace=0.5)
    plt.interactive(False)
    plt.show()
    return None
