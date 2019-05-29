import time
import sys

import params
import mappings
import receiver_helper


def n_tuple_former():
    # Prepare the data
    samples_received, preamble_samples_sent = receiver_helper.prepare_data()

    # Find the frequency range that has been removed
    removed_freq_range, frequency_ranges_available, indices_available = receiver_helper.find_removed_frequency(
        samples_received)

    # Demodulation
    demodulated_samples = receiver_helper.demodulate(samples_received, removed_freq_range, frequency_ranges_available,
                                                     indices_available)

    # Low pass
    y = receiver_helper.low_pass(demodulated_samples, indices_available)

    # Find the delay
    delay = receiver_helper.find_delay(y, preamble_samples_sent, frequency_ranges_available)

    # Extract the preamble samples
    preamble_samples_received = receiver_helper.extract_preamble_samples(y, delay, preamble_samples_sent,
                                                                         frequency_ranges_available, indices_available)

    # Compute the phase shift and the scaling factor
    phase_shift_estim, scaling_factor_estim = receiver_helper.estimate_parameters(preamble_samples_sent,
                                                                                  preamble_samples_received,
                                                                                  indices_available)

    # Crop the samples (remove the delay, the preamble, and adjust to the first relevant sample of data)
    data_samples = receiver_helper.crop_samples_1(y, delay, len(preamble_samples_sent), indices_available)

    # Find the second_preamble_index
    second_preamble_index = receiver_helper.find_second_preamble_index(data_samples, preamble_samples_sent)

    # Crop the samples (remove the garbage at the end)
    data_samples = receiver_helper.crop_samples_2(data_samples, second_preamble_index)

    # Correct the phase shift on the data samples
    data_samples = receiver_helper.correct_params(data_samples, phase_shift_estim)

    # Down-sample the samples to obtain the symbols
    data_symbols = receiver_helper.downsample(data_samples)

    return data_symbols, removed_freq_range


def decoder(symbols, removed_freq_range):
    """
    Map the received symbols to the closest symbols of our mapping
    :param removed_freq_range: index of the range that was removed by the server
    :param symbols: the observation vector, i.e the received symbols
    :return: integers between 0 and M-1, i.e integers corresponding to the bits sent
    """
    # Choose the mapping according to the params file
    mapping = mappings.choose_mapping()

    # Associate the received symbols to integers (indices of our mapping)
    ints = receiver_helper.symbols_to_ints(symbols, mapping)

    # Deduce the received message
    message_received = receiver_helper.ints_to_message(ints, removed_freq_range)
    return message_received


# Intended for testing (to run the program, run main.py)
if __name__ == "__main__":
    if params.logs:
        moment = time.strftime("%Y-%b-%d__%H_%M_%S", time.localtime())
        log_file = open("../logs/" + moment + ".log", "w+")
        sys.stdout = log_file
    data_symbols, removed_freq_range = n_tuple_former()
    decoder(data_symbols, removed_freq_range)
