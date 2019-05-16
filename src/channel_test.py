import numpy as np
import matplotlib.pyplot as plt

import params
import launcher
import fourier_helper


def vertical_lines_frequency_ranges(plot):
    plot.axvline(x=1000, color='r')
    plot.axvline(x=3000, color='r')
    plot.axvline(x=5000, color='r')
    plot.axvline(x=7000, color='r')
    plot.axvline(x=9000, color='r')


if __name__ == "__main__":

    """
    Send the samples from the input file to the server, and get the output samples in the output file
    """
    launcher.launch()

    """
    Plot the input and output samples in Time domain
    """
    input = np.loadtxt(params.message_sample_path)
    output = np.loadtxt(params.output_sample_path)

    _, axs = plt.subplots(2, 1)
    plt.figure(1).suptitle("Input and output in Time domain")

    axs[0].plot(range(len(input)), input)
    axs[0].set_ylabel('Input')

    # output = output[13000:]
    axs[1].plot(range(len(output)), output)
    axs[1].set_xlabel('Samples')
    axs[1].set_ylabel('Output')

    axs[0].grid(True)
    axs[1].grid(True)

    """
    Plot the output samples in the Frequency domain
    """
    X = np.fft.fft(input)
    Y = np.fft.fft(output)

    f_x, y_x = fourier_helper.dft_map(X, params.Fs, shift=False)
    f_y, y_y = fourier_helper.dft_map(Y, params.Fs, shift=False)

    _, axs = plt.subplots(2, 1)
    fig = plt.figure(2)
    fig.suptitle("Input and output in Frequency domain")

    axs[0].plot(f_x, abs(y_x))
    axs[0].set_ylabel('Input')
    vertical_lines_frequency_ranges(axs[0])
    axs[0].set_xlim(params.FREQ_RANGES[0][0] - 1000, params.FREQ_RANGES[3][1] + 1000)

    axs[1].plot(f_y, abs(y_y))
    axs[1].set_xlabel('Frequency (in Hz)')
    axs[1].set_ylabel('Output')
    vertical_lines_frequency_ranges(axs[1])
    axs[1].set_xlim(params.FREQ_RANGES[0][0] - 1000, params.FREQ_RANGES[3][1] + 1000)

    plt.interactive(False)
    plt.show()
