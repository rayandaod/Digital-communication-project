import numpy as np

import src.params


def maximum_likelihood_sync(received_signal, training_sequence=src.params.PREAMBLE):
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
    padded_training_sequence = np.pad(training_sequence, (0, n - len(training_sequence)), 'constant')
    correlation_array = np.correlate(training_sequence, received_signal)
    print(correlation_array)
    print(padded_training_sequence)
    print(received_signal)
    return np.argmax(correlation_array)


if __name__ == "__main__":
    print(maximum_likelihood_sync([0, 0, 0, 1, 2, 3, 1, 3, 2, 1], [1, 2, 3]))