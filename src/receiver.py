import numpy as np

import fourier_helper
import helper
import mappings
import parameter_estim
import params
import plot_helper
import pulses
import read_write


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

    if params.logs:
        print("y: {}\n{}".format(np.shape(y), y))
        print("mapping: {} \n{}".format(np.shape(mapping), mapping))

    # Number of symbols in the mapping
    M = np.size(mapping, 0)
    # Number of symbols received
    S = np.size(y, 1)

    distances = np.transpose(abs(np.tile(y, (M, 1)) - np.tile(mapping, (1, S))))
    ints = np.argmin(distances, 1)
    if params.logs:
        print("Equivalent integers:\n{}".format(ints))
    return ints


def ints_to_message(ints):
    """
    :param ints: integers between 0 and M-1, i.e integers corresponding to the bits sent
    :return: the corresponding guessed message as a string
    """

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
    message = ''.join(helper.bits2string(new_bits))
    print("Message received:\n{}".format(message))

    return message


def received_from_server():
    # Load the input and output samples from their respective files
    input_samples = np.loadtxt(params.input_sample_file_path)
    received_samples = np.loadtxt(params.output_sample_file_path)

    # Plot the input and output samples in Time domain and Frequency domain
    if params.plots:
        plot_helper.two_simple_plots(input_samples, received_samples, "Input and output in Time domain", "Input", "Output")
        plot_helper.two_fft_plots(input_samples, received_samples, "Input and output in Frequency domain", "Input", "Output")

    # Read the preamble samples saved previously
    preamble_samples = read_write.read_preamble_samples()
    len_preamble_samples = len(preamble_samples)

    if params.plots:
        plot_helper.plot_complex_function(received_samples, "Received samples in time domain")
        plot_helper.fft_plot(received_samples, "Received samples in frequency domain", shift=True)

    # Find the frequency range that has been removed
    range_indices, removed_freq_range = fourier_helper.find_removed_freq_range(received_samples)
    if params.logs:
        print("Removed frequency range: {} (range {})".format(removed_freq_range, removed_freq_range + 1))

    # Choose a frequency for demodulation
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
        raise ValueError('This modulation type does not exist yet... He he he')

    # Demodulate the received samples to base-band
    demodulated_samples = fourier_helper.demodulate(received_samples, fc)
    if params.plots:
        plot_helper.plot_complex_function(demodulated_samples, "Demodulated samples in Time domain")
        plot_helper.fft_plot(demodulated_samples, "Demodulated samples in Frequency domain", shift=True)

    # Match filter (i.e Low-pass)
    _, h = pulses.root_raised_cosine()
    half_span_h = int(params.SPAN / 2)
    h_matched = np.conjugate(h[::-1])
    y = np.convolve(demodulated_samples, h_matched)
    if params.plots:
        plot_helper.plot_complex_function(y, "y in Time domain")
        plot_helper.fft_plot(y, "y in Frequency domain", shift=True)

    # Find the delay
    delay = parameter_estim.ML_theta_estimation(demodulated_samples, preamble_samples=preamble_samples)
    if params.logs:
        print("Delay: {} samples".format(delay))
        print("--------------------------------------------------------")

    # Extract the preamble samples
    preamble_samples_received = y[half_span_h + delay - 1:half_span_h + delay + len_preamble_samples - 1]
    if params.plots:
        plot_helper.two_simple_plots(preamble_samples_received.real, preamble_samples.real,
                                     "Comparison between preamble samples received and preamble samples sent",
                                     "received", "expected")
    if params.logs:
        print("Number of samples for the actual preamble: {}".format(len_preamble_samples))
        print("Number of samples for the received preamble: {}".format(len(preamble_samples_received)))

    # Compute the phase shift
    phase_shift_estim, scaling_factor = parameter_estim.ML_phase_scaling_estim(
        preamble_samples[:len_preamble_samples - half_span_h],
        preamble_samples_received[:len(preamble_samples_received) - half_span_h])
    if params.logs:
        print("Frequency offset: {}".format(phase_shift_estim))
        print("Scaling factor: {}".format(scaling_factor))

    # Crop the samples (remove the delay, the preamble, and the ramp-up)
    data_samples = y[half_span_h + delay + len_preamble_samples - half_span_h + params.USF - 1 - 1:]

    # Find the second_preamble_index
    second_preamble_index = parameter_estim.ML_theta_estimation(data_samples, preamble_samples=preamble_samples[::-1])
    if params.logs:
        print("Second preamble index: {} samples".format(second_preamble_index))
        print("--------------------------------------------------------")

    # Crop the samples (remove the garbage at the end)
    data_samples = data_samples[:second_preamble_index + half_span_h - params.USF + 1]
    if params.plots:
        plot_helper.plot_complex_function(data_samples, "y after removing the delay, the preamble, and the ramp-up")

    # Correct the phase shift on the data samples
    data_samples = data_samples * np.exp(-1j * (phase_shift_estim - np.pi / 2))

    # Down-sample the samples to obtain the symbols
    data_symbols = data_samples[::params.USF]
    if params.logs:
        print("Number of symbols received: {}".format(len(data_symbols)))
    if params.plots:
        plot_helper.plot_complex_function(data_symbols, "y without preamble")
        plot_helper.plot_complex_symbols(data_symbols, "Symbols received", annotate=False)

    # Decode the symbols
    ints = decoder(data_symbols, mappings.choose_mapping())
    message_received = ints_to_message(ints)
    read_write.write_message_received(message_received)

    message_sent = read_write.read_message_sent()
    print("Message sent:\n{}".format(message_sent))


# Intended for testing (to run the program, run main.py)
if __name__ == "__main__":
    received_from_server()
