import numpy as np
import scipy.signal as sc

import params


def ML_theta_estimation(received_signal, preamble_samples):
    """
    Synchronizes the received signal, i.e returns the number of samples after which the first preamble is detected,
    and the number of samples after which the second preamble is detected.

    :param received_signal: signal received from the server
    :param preamble_samples: real-valued sequence used to synchronize the received signal
    :return: delay in number of samples
    """
    # Correlation between the received signal and the preamble to find the delay
    if len(received_signal) >= len(preamble_samples):
        n = len(received_signal)
        preamble_samples = np.pad(preamble_samples, (0, n - len(preamble_samples)), 'constant')
    else:
        n = len(preamble_samples)
        received_signal = np.pad(received_signal, (0, n - len(received_signal)), 'constant')
    correlation_array = sc.correlate(received_signal, preamble_samples)

    # TODO Should we put abs here? Useful only if the channel multiplies the training sequence by -1
    index = np.argmax(abs(correlation_array))

    M = max(len(received_signal), len(preamble_samples))
    delay_range = np.arange(-M + 1, M - 1)

    return delay_range[index]


def ML_phase_scaling_estim(preamble_samples, preamble_samples_received):
    # TODO: why dot works and not vdot (supposed to conjugate the first term in the formula)
    dot_product = np.dot(preamble_samples, preamble_samples_received)
    preamble_energy = 0
    for i in range(len(preamble_samples) - int(params.SPAN / 2)):
        preamble_energy += np.absolute(preamble_samples[i]) ** 2
    return np.angle(dot_product), abs(dot_product) / preamble_energy
