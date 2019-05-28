import numpy as np
import time
import sys

import helper
import params
import read_write
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
    data_symbols = receiver_helper.down_sample(data_samples)

    return data_symbols


def decoder(symbols, mapping):
    """
    Map the received symbols to the closest symbols of our mapping
    :param symbols: the observation vector, i.e the received symbols
    :param mapping: the chosen mapping for the communication
    :return: integers between 0 and M-1, i.e integers corresponding to the bits sent
    """
    if params.logs:
        print("Decoding the symbols...")
    if params.MOD == 1 or params.MOD == 2:
        ints = receiver_helper.symbols_to_ints(symbols, mapping)

        if params.logs:
            print("Equivalent integers:\n{}".format(ints))

        # Convert the ints to BITS_PER_SYMBOL bits
        bits = ["{0:0{bits_per_symbol}b}".format(i, bits_per_symbol=params.BITS_PER_SYMBOL) for i in ints]
        if params.logs:
            print("Groups of BITS_PER_SYMBOL bits representing each integer:\n{}".format(bits))

        # Make a new string with it
        bits = ''.join(bits)
        if params.logs:
            print("Bits grouped all together:\n{}".format(bits))

        # Slice the string into substrings of 7 characters
        bits = [bits[i:i + 7] for i in range(0, len(bits), 7)]
        if params.logs:
            print("Groups of 7 bits:\n{}".format(bits))

        # Add a zero at the beginning of each substring (cf transmitter)
        new_bits = []
        for sub_string in bits:
            new_bits.append('0' + sub_string)
        if params.logs:
            print("Groups of 8 bits (0 added at the beginning, cf. transmitter):\n{}".format(new_bits))

        # Convert from array of bytes to string
        message_received = ''.join(helper.bits2string(new_bits))

        message_sent = read_write.read_message_sent()
        print("Message sent:     {}".format(message_sent))
        print("Message received: {}".format(message_received))
        helper.compare_messages(message_sent, message_received)
    elif params.MOD == 3:
        return None
    else:
        raise ValueError("This modulation type does not exist yet... He he he")
    if params.logs:
        print("--------------------------------------------------------")
    return message_received


# Intended for testing (to run the program, run main.py)
if __name__ == "__main__":
    if params.logs:
        moment = time.strftime("%Y-%b-%d__%H_%M_%S", time.localtime())
        log_file = open("../logs/" + moment + ".log", "w+")
        sys.stdout = log_file

    symbols = n_tuple_former()
    message = decoder(symbols, mappings.choose_mapping())
    read_write.write_message_received(message)
