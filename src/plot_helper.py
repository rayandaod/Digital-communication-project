import matplotlib.pyplot as plt
import numpy as np

import fourier_helper
import params


def plot_complex_symbols(complex_values, title, color="black", annotate=False):
    """
    Plot the complex values as black circles in a 2-D (complex) space

    :param complex_values:  The array of complex values to plot
    :param title:           The title of the plot
    :param color:           The color of the points
    :param annotate:        Rather we annotate the symbols or not
    :return:                None
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
    Plot a function of complex values

    :param complex_values:  The complex values to plot (e.g at the output of the pulse-shaping)
    :param title:           The title of the plot
    :param dots:            Rather we want dots or not
    :return:                None
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

    :param plot:    The current plot
    :return:        None
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

    :param samples_1:   The first sample array
    :param samples_2:   The second sample array
    :param title:       The title for both plots
    :param y_label_1:   The first y axis label
    :param y_label_2:   The second y axis label
    :return:            None
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
    Plot 2 simple plots, one above the other

    :param samples_1:   The first sample array
    :param samples_2:   The second sample array
    :param title:       The title for both plots
    :param y_label_1:   The first y axis label
    :param y_label_2:   The second y axis label
    :return:            None
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

    :param samples: The sample array
    :param title:   The title for the plot
    :param shift:   Rather we shift the fft or not
    :return:        None
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

    :param x_axis:  The x axis
    :param y_axis:  The y axis
    :param title:   The title of the plot
    :return:        None
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
    """
    Plot a simple and an fft plot of the samples

    :param time_indices:    The time array for the samples
    :param samples:         The samples to plot
    :param title:           The title of the plot
    :param shift:           Rather we shift the fft or not
    :return:                None
    """
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


def samples_fft_plots(samples, title, shift=False, time=False, is_complex=True):
    """
    Plot the samples (2 plots if is_complex=True) and the Fourier transform

    :param samples:     The samples to plot
    :param title:       The title of the plot
    :param shift:       Rather we shift the Fourier transform plot or not
    :param time:        Rather we put the time in seconds or the number of samples
    :param is_complex:  Rather the samples are complex or not
    :return:            None
    """
    num_plots = 2 if not is_complex else 3

    fig, axs = plt.subplots(num_plots, 1)
    fig.suptitle(title)
    x_axis = np.arange(len(samples))
    x_label = "Samples"

    if time:
        x_axis = x_axis / params.Fs
        x_label = "Time (in seconds)"

    axs[0].plot(x_axis, np.real(samples))
    axs[0].set_xlabel(x_label)
    axs[0].set_ylabel("Real")

    if num_plots == 3:
        axs[1].plot(x_axis, np.imag(samples))
        axs[1].set_xlabel(x_label)
        axs[1].set_ylabel("Imaginary")

    X = np.fft.fft(samples)
    f_x, y_x = fourier_helper.dft_map(X, shift=shift)

    axs[num_plots - 1].plot(f_x, abs(y_x))
    axs[num_plots - 1].set_xlabel("Frequency (in Hertz)")
    axs[num_plots - 1].set_ylabel("|X(f)|^2")

    for i in range(num_plots):
        axs[i].grid(True)

    vertical_lines_frequency_ranges(axs[num_plots - 1])

    if not shift:
        axs[num_plots - 1].set_xlim(params.FREQ_RANGES[0][0] - 1000, params.FREQ_RANGES[3][1] + 1000)

    plt.subplots_adjust(hspace=0.5)
    plt.interactive(False)
    plt.show()
    return None


def delay_plots(samples, delay, title):
    """
    Plot the samples and a vertical black line at the position of the estimated delay

    :param samples: The samples to plot
    :param delay:   The estimated delay
    :param title:   The title of the plot
    :return:        None
    """
    fig, axs = plt.subplots(3, 1)
    fig.suptitle(title)
    x_axis = np.arange(len(samples[0])) / params.Fs
    x_label = "Time (in seconds)"

    for i in range(len(samples)):
        axs[i].plot(x_axis, np.real(samples[i]))
        axs[i].set_ylabel("Samples {}".format(i))
        axs[i].axvline(x=delay / params.Fs, color='black')
        axs[i].grid(True)
    axs[len(samples) - 1].set_xlabel(x_label)

    plt.subplots_adjust(hspace=0.5)
    plt.interactive(False)
    plt.show()
    return None


def compare_preambles(preamble_received, preamble_sent, title):
    """
    Compare the preamble received and the preamble sent in a plot

    :param preamble_received:   The preamble received
    :param preamble_sent:       The preamble sent
    :param title:               The title of the plot
    :return:                    None
    """
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(title)
    x_axis = np.arange(max(len(preamble_received), len(preamble_sent)))

    axs[0][0].plot(x_axis, np.real(preamble_received))
    axs[0][0].set_ylabel("Real")
    axs[1][0].plot(x_axis, np.imag(preamble_received))
    axs[1][0].set_ylabel("Imaginary")

    axs[0][1].plot(x_axis, np.real(preamble_sent))
    axs[0][1].set_ylabel("Real")
    axs[1][1].plot(x_axis, np.imag(preamble_sent))
    axs[1][1].set_ylabel("Imaginary")
    return None
