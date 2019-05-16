import numpy as np
import matplotlib.pyplot as plt

import params
import launcher


def dft_shift(X):
    """
    With k=0 as the center point, odd-length vectors will produce symmetric data sets with (N-1)/2 points left and
    right of the origin, whereas even-length vectors will be asymmetric, with one more point on the positive axis;
    indeed, the highest positive frequency for even-length signals will be equal to omega_{N/2} = pi. Since the
    frequencies of pi and -pi are identical, we can copy the top frequency data point to the negative axis and
    obtain a symmetric vector also for even-length signals. Here is a function that does that.
    (credits: Prandoni P. - Signal processing for Communication course at EPFL)

    :param X: the fourier transform of our signal x
    :return: a shifted version of the fourier transform (to be around 0) for even and odd length signals
    :
    """
    N = len(X)
    if N % 2 == 0:
        # even-length: return N+1 values
        return np.arange(-int(N/2), int(N/2) + 1), np.concatenate((X[int(N/2):], X[:int(N/2)+1]))
    else:
        # odd-length: return N values
        return np.arange(-int((N-1)/2), int((N-1)/2) + 1), np.concatenate((X[int((N+1)/2):], X[:int((N+1)/2)]))


def dft_map(X, Fs, shift=True):
    """
    In order to look at the spectrum of the sound file with a DFT we need to map the digital frequency "bins" of the
    DFT to real-world frequencies. The k-th basis function over C_N completes k periods over N samples.
    If the time between samples is 1/Fs, then the real-world frequency of the k-th basis function is periods over time,
    namely k(F_s/N). Let's remap the DFT coefficients using the sampling rate.
    (credits: Prandoni P. - Signal processing for Communication course at EPFL)

    :param X: the fourier transform of our signal x
    :param Fs: the sampling frequency
    :param shift: rather we want to shift the fft or not
    :return: a real-world-frequency DFT
    """
    resolution = float(Fs) / len(X)
    if shift:
        n, Y = dft_shift(X)
    else:
        Y = X
        n = np.arange(0, len(Y))
    f = n * resolution
    return f, Y


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

    f_x, y_x = dft_map(X, params.Fs, shift=False)
    f_y, y_y = dft_map(Y, params.Fs, shift=False)

    _, axs = plt.subplots(2, 1)
    fig = plt.figure(2)
    fig.suptitle("Input and output in Frequency domain")

    axs[0].plot(f_x, abs(y_x))
    axs[0].set_ylabel('Input')
    vertical_lines_frequency_ranges(axs[0])
    axs[0].set_xlim(params.MIN_FREQ - 1000, params.MAX_FREQ + 1000)

    axs[1].plot(f_y, abs(y_y))
    axs[1].set_xlabel('Frequency (in Hz)')
    axs[1].set_ylabel('Output')
    vertical_lines_frequency_ranges(axs[1])
    axs[1].set_xlim(params.MIN_FREQ - 1000, params.MAX_FREQ + 1000)

    plt.interactive(False)
    plt.show()
