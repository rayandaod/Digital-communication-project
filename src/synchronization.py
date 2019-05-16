import numpy as np
import scipy.signal as sc

import params
import fourier_helper


def maximum_likelihood_sync(received_signal, training_sequence=params.PREAMBLE):
    """
    Synchronizes the received signal, i.e returns the number of samples after which the data signal begins.\n

    - We first check which range of frequencies has been removed by the channel (among 1-3kHz, 3-5kHz, 5-7kHz, 7-9kHz)
    thanks to a Fourier-transform on the received signal.

    - Then we remove the corresponding frequency components from our original training sequence and correlate the
    received signal with the modified training sequence to aim for the highest scalar product, which will correspond to
    the delay.\n

    :param received_signal: signal received from the server
    :param training_sequence: real-valued sequence used to synchronize the received signal
    :return: delay in number of samples
    """
    n = len(received_signal)

    # Fourier transform
    # TODO Do we need to compute the fourier of the whole signal?
    RX = np.fft.fft(received_signal)
    frequencies_mapped, RX_mapped = fourier_helper.dft_map(RX, params.Fs, shift=False)
    removed_freq_range = fourier_helper.find_removed_freq_range(RX_mapped)

    # Correlation
    padded_training_sequence = np.pad(training_sequence, (0, n - len(training_sequence)), 'constant')
    correlation_array = sc.correlate(received_signal, padded_training_sequence)

    # TODO Should we put abs here? In which case is it useful?
    index = np.argmax(abs(correlation_array))

    M = max(len(received_signal), len(training_sequence))
    delay_range = np.arange(-M+1, M-1)

    return delay_range[index]


if __name__ == "__main__":
    print(maximum_likelihood_sync([-1, 1, 1, 1, 1, 1, 1, -1,
            -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1, # starts here
            -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1,
            1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1,
            -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1,
            1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, 1,
            -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1,
            1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 1, -1,                    # ends here
                             -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1, # starts here
            -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1,
            1, -1, -1, 1, 1, 1, -1], params.PREAMBLE))
