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
    preamble_symbols = np.hstack((barker_code_13, barker_code_13[::-1]))
    preamble_symbols = np.hstack((preamble_symbols, preamble_symbols))
    return preamble_symbols + 1j*preamble_symbols


# Estimate the frequency offset
def moose_algorithm(samples):
    n_samples = samples.size
    print("N_samples: {}".format(n_samples))
    n_samples_half = int(n_samples/2)
    first_half = np.matrix(samples[:n_samples_half])
    second_half = np.matrix(samples[n_samples_half:])
    print("Dims: {}, {}".format(first_half.shape, second_half.shape))
    phase_offset, _, _, _ = np.linalg.lstsq(first_half.transpose(), second_half.transpose())

    # Use the phase offset to find the frequency
    freq_shift = np.angle(phase_offset)/2/np.pi/(1/params.Fs*n_samples_half)
    return freq_shift


def maximum_likelihood_sync(received_signal, synchronization_sequence):
    """
    Synchronizes the received signal, i.e returns the number of samples after which the data signal begins.\n

    - We first check which range of frequencies has been removed by the channel (among 1-3kHz, 3-5kHz, 5-7kHz, 7-9kHz)
    thanks to a Fourier-transform on the received signal.

    - Then we remove the corresponding frequency components from our original training sequence and correlate the
    received signal with the modified training sequence to aim for the highest scalar product, which will correspond to
    the delay.\n

    :param received_signal: signal received from the server
    :param synchronization_sequence: real-valued sequence used to synchronize the received signal
    :return: delay in number of samples
    """
    n = len(received_signal)

    # Identify which range of frequencies has been removed
    # TODO Do we need to compute the fourier of the whole signal? only the [preamble + data] part is relevant in freq.
    # RX = np.fft.fft(received_signal)
    # frequencies_mapped, RX_mapped = fourier_helper.dft_map(RX, shift=True)
    # removed_freq_range = fourier_helper.find_removed_freq_range(RX_mapped)
    # if params.verbose:
        # print(plot_helper.fft_plot(received_signal, "", shift=True))
        # print("Frequency range that has been removed: {}".format(removed_freq_range))

    # TODO According to Prandoni, it should work without that (seems like it's true)
    # # Remove it from the training sequence
    # S = np.fft.fft(synchronization_sequence)
    # frequencies_mapped, S_mapped = fourier_helper.dft_map(S, shift=False)
    # S_mapped[params.FREQ_RANGES[removed_freq_range][0]:params.FREQ_RANGES[removed_freq_range][1]] = 0
    # synchronization_sequence = np.fft.ifft(S_mapped)

    # Correlation between the received signal and the sync sequence to find the delay
    padded_new_training_sequence = np.pad(synchronization_sequence, (0, n - len(synchronization_sequence)), 'constant')
    correlation_array = sc.correlate(received_signal, padded_new_training_sequence)
    print(correlation_array)

    # TODO Should we put abs here? Useful only if the channel multiplies the training sequence by -1
    index = np.argmax(abs(correlation_array))

    M = max(len(received_signal), len(synchronization_sequence))
    delay_range = np.arange(-M+1, M-1)

    return delay_range[index]


if __name__ == "__main__":
    print("synchronization.py")
