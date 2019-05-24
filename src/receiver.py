import numpy as np

import params
import helper
import plot_helper
import synchronization
import fourier_helper
import pulses
import mappings


def decoder(y, mapping):
    """
    :param y: the observation vector, i.e the received symbols
    :param mapping: the chosen mapping for the communication
    :return: integers between 0 and M-1, i.e integers corresponding to the bits sent
    """

    # Make sure y and mapping have less or equal than 2 dimensions
    if len(y.shape) > 2 or len(mapping.shape) > 2:
        raise AttributeError("One of the vectors y and mapping has more than 2 dimensions!")

    # If y is a column vector, make it a row vector
    n_elems_axis_0_y = np.size(y, 0)
    if n_elems_axis_0_y != 1:
        y = y.reshape(1, n_elems_axis_0_y)
    else:
        y = y.reshape(1, np.size(y, 1))

    # If mapping is a row vector, make it a column vector
    if np.size(mapping, 0) == 1:
        mapping = mapping.reshape(np.size(mapping, 1), 1)
    else:
        mapping = mapping.reshape(np.size(mapping, 0), 1)

    if params.verbose:
        print("y: {}\n{}".format(np.shape(y), y))
        print("mapping: {} \n{}".format(np.shape(mapping), mapping))

    # Number of symbols in the mapping
    M = np.size(mapping, 0)
    # Number of symbols received
    S = np.size(y, 1)

    distances = np.transpose(abs(np.tile(y, (M, 1)) - np.tile(mapping, (1, S))))
    ints = np.argmin(distances, 1)
    if params.verbose:
        print("Equivalent integers:\n{}".format(ints))
    return ints


def ints_to_message(ints):
    """
    :param ints: integers between 0 and M-1, i.e integers corresponding to the bits sent
    :return: the corresponding guessed message as a string
    """

    # Convert the ints to BITS_PER_SYMBOL bits
    bits = ["{0:0{bits_per_symbol}b}".format(i, bits_per_symbol=params.BITS_PER_SYMBOL) for i in ints]
    if params.verbose:
        print("Groups of BITS_PER_SYMBOL bits representing each integer:\n{}".format(bits))

    # Make a new string with it
    bits = ''.join(bits)
    if params.verbose:
        print("Bits grouped all together:\n{}".format(bits))

    # Slice the string into substrings of 7 characters
    bits = [bits[i:i+7] for i in range(0, len(bits), 7)]
    if params.verbose:
        print("Groups of 7 bits:\n{}".format(bits))

    # Add a zero at the beginning of each substring (cf transmitter)
    new_bits = []
    for sub_string in bits:
        new_bits.append('0' + sub_string)
    if params.verbose:
        print("Groups of 8 bits (0 added at the beginning, cf. transmitter):\n{}".format(new_bits))

    # Convert from array of bytes to string
    message = ''.join(helper.bits2string(new_bits))
    print("Message received:\n{}".format(message))

    return message


def received_from_server():
    # Read the received samples from the server
    output_sample_file = open(params.output_sample_file_path, "r")
    received_samples = [float(line) for line in output_sample_file.readlines()]
    output_sample_file.close()

    # Plot the received samples
    plot_helper.plot_complex_function(received_samples, "Received samples in time domain")
    plot_helper.fft_plot(received_samples, "Received samples in frequency domain", shift=True)

    # Read the preamble samples saved previously
    preamble_samples_file = open(params.preamble_sample_file_path, "r")
    preamble_samples = [complex(line) for line in preamble_samples_file.readlines()]
    preamble_samples_file.close()
    len_preamble_samples = len(preamble_samples)

    # Read the preamble symbols saved previously
    preamble_symbol_file = open(params.preamble_symbol_file_path, "r")
    preamble_symbols = [complex(line) for line in preamble_symbol_file.readlines()]
    preamble_symbol_file.close()

    plot_helper.plot_complex_function(preamble_samples, "Preamble samples in time domain")

    # Find the frequency range that has been removed
    range_indices, removed_freq_range = fourier_helper.find_removed_freq_range_2(received_samples)
    print("Removed frequency range: {}".format(removed_freq_range))

    # Choose a frequency among the 3 available frequency ranges / the only available frequency range
    if params.MODULATION_TYPE == 1:
        if removed_freq_range == 0:
            fc = np.mean(params.FREQ_RANGES[1])
        else:
            fc = np.mean(params.FREQ_RANGES[0])
    elif params.MODULATION_TYPE == 2:
        if removed_freq_range == 0 or removed_freq_range == 1:
            fc = 7000
        else:
            fc = 3000
    else:
        raise ValueError('This modulation type does not exist yet... Hehehe')

    # Demodulate the samples with the appropriate frequency fc
    demodulated_samples = fourier_helper.demodulate(received_samples, fc)
    plot_helper.plot_complex_function(demodulated_samples, "Demodulated samples in Time domain")
    plot_helper.fft_plot(demodulated_samples, "Demodulated samples in Time domain", shift=True)

    # Match filter
    _, h = pulses.root_raised_cosine()
    half_span_h = int(params.SPAN/2)
    h_matched = np.conjugate(h[::-1])
    y = np.convolve(demodulated_samples, h_matched)
    plot_helper.plot_complex_function(y, "y in Time domain")
    plot_helper.fft_plot(y, "y in Frequency domain", shift=True)

    # Find the delay
    delay = synchronization.maximum_likelihood_sync(demodulated_samples, preamble_samples=preamble_samples)
    print("Delay: {} samples".format(delay))
    print("--------------------------------------------------------")

    # Extract the preamble samples
    preamble_samples_received = y[half_span_h + delay - 1:half_span_h + delay + len_preamble_samples - 1]
    plot_helper.two_simple_plots(preamble_samples_received, preamble_samples,
                                 "Comparison between preamble samples received and preamble samples sent",
                                 "received", "expected")
    print("Number of samples for the actual preamble: {}".format(len_preamble_samples))
    print("Number of samples for the received preamble: {}".format(len(preamble_samples_received)))

    # Compute the frequency offset, and the scaling factor
    # TODO: why dot works and not vdot (supposed to conjugate the first term in the formula)
    dot_product = np.dot(preamble_samples[:len_preamble_samples - half_span_h],
                         preamble_samples_received[:len(preamble_samples_received) - half_span_h])
    print("Dot product: {}".format(dot_product))

    frequency_offset_estim = np.angle(dot_product)
    print("Frequency offset: {}".format(frequency_offset_estim))

    # Crop the samples (remove the delay, and the ramp-up/ramp-down)
    data_samples = y[delay + params.SPAN - 1:len(y) - params.SPAN + 1]
    plot_helper.plot_complex_function(data_samples, "y after putting the right sampling time")

    # TODO: why frequency_offset - pi/2 works ?
    data_samples = data_samples * np.exp(-1j * (frequency_offset_estim - np.pi / 2))

    # Down-sample
    symbols_received = data_samples[::params.USF]
    print("Symbols received:\n{}", format(symbols_received))

    # Remove the preamble symbols at the beginning
    data_symbols = symbols_received[len(preamble_symbols):]

    plot_helper.plot_complex_function(data_symbols, "y without preamble")
    plot_helper.plot_complex_symbols(data_symbols, "Data symbols received", annotate=False)

    # Decode the symbols
    ints = decoder(symbols_received, mappings.mapping)
    ints_to_message(ints)


# Intended for testing (to run the program, run main.py)
if __name__ == "__main__":

    received_from_server()
