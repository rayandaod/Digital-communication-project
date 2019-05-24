import numpy as np
import scipy.signal as sc

import params
import mappings
import writers


def generate_preamble_symbols(n_symbols_to_send):
    if params.PREAMBLE_TYPE == "random":
        preamble_symbols = generate_random_preamble_symbols(n_symbols_to_send)
    elif params.PREAMBLE_TYPE == "barker":
        preamble_symbols = generate_barker_preamble_symbols()
    else:
        raise ValueError('This preamble type does not exist yet... Hehehe')

    if params.MAPPING == "qam" and not params.NORMALIZE_MAPPING:
        # TODO: improve that
        if params.M == 16:
            writers.write_preamble_symbols(preamble_symbols*3)
        elif params.M == 4:
            writers.write_preamble_symbols(preamble_symbols)
    else:
        raise ValueError('TODO: automate the scaling of the barker sequence')

    return None


def generate_random_preamble_symbols(n_symbols_to_send):
    preamble_symbols = np.random.choice(mappings.mapping,
                                        size=int(np.ceil(n_symbols_to_send * params.PREAMBLE_LENGTH_RATIO)))
    if params.verbose:
        print("Synchronization sequence:\n{}".format(preamble_symbols))
        print("--------------------------------------------------------")
    return preamble_symbols


# TODO see if 4 copies are enough/too much
def generate_barker_preamble_symbols():
    barker_code_13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
    # preamble_symbols = np.hstack((barker_code_13, barker_code_13[::-1]))
    # preamble_symbols = np.hstack((preamble_symbols, preamble_symbols))
    preamble_symbols = np.repeat(barker_code_13, 4)
    return preamble_symbols + 1j*preamble_symbols


# # Estimate the frequency offset
# def moose_algorithm(samples):
#     n_samples = samples.size
#     print("N_samples: {}".format(n_samples))
#     n_samples_half = int(n_samples/2)
#     first_half = np.matrix(samples[:n_samples_half])
#     second_half = np.matrix(samples[n_samples_half:])
#     print("Dims: {}, {}".format(first_half.shape, second_half.shape))
#     phase_offset, _, _, _ = np.linalg.lstsq(first_half.transpose(), second_half.transpose())
#
#     # Use the phase offset to find the frequency
#     freq_shift = np.angle(phase_offset)/2/np.pi/(1/params.Fs*n_samples_half)
#     return freq_shift


def maximum_likelihood_sync(received_signal, preamble_samples):
    """
    Synchronizes the received signal, i.e returns the number of samples after which the first preamble is detected,
    and the number of samples after which the second preamble is detected.

    :param received_signal: signal received from the server
    :param preamble_samples: real-valued sequence used to synchronize the received signal
    :param beginning: rater we are synchronizing for the beginning of the signal or not (the end)
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
    delay_range = np.arange(-M+1, M-1)

    return delay_range[index]

