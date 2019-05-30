import numpy as np

import fourier_helper
import helper
import parameter_estim
import params
import plot_helper
import pulses
import read_write


# TODO: Code redundancy to manage


def prepare_data():
    """
    Prepare the data used to receive the signal, i.e retrieve the received samples and the original preamble

    :return: The received samples and the original preamble
    """
    if params.logs:
        print("Preparing the data...")

    # Load the input and output samples from their respective files
    input_samples = np.loadtxt(params.input_sample_file_path)
    samples_received = np.loadtxt(params.output_sample_file_path)

    # Plot the input and output samples in Time domain and Frequency domain
    if params.plots:
        plot_helper.samples_fft_plots(input_samples, "Sent samples", is_complex=False)
        plot_helper.samples_fft_plots(samples_received, "Received samples", is_complex=False)

    # Read the preamble samples saved previously
    preamble_samples_sent = read_write.read_preamble_samples()

    if params.logs:
        print("--------------------------------------------------------")
    return samples_received, preamble_samples_sent


def find_removed_frequency(samples_received):
    """
    Find the frequency range that has been removed by the server and return the index to the range in params.FREQ_RANGES

    :param samples_received:    The samples received from the server
    :return:                    The removed frequency range as an index of params.FREQ_RANGES, and two helping arrays,
                                    i.e an array of booleans where True indicates that the corresponding frequency range
                                    is available, and False that the corresponding frequency range is removed, and an
                                    array of the indices of the available frequency ranges
    """
    if params.logs:
        print("Finding the frequency range that has been removed...")
    _, removed_freq_range = fourier_helper.find_removed_freq_range(samples_received)
    if params.logs:
        print("Removed frequency range: {} (range {})".format(removed_freq_range, removed_freq_range + 1))

    frequency_ranges_available = [True if i != removed_freq_range else False for i in range(len(params.FREQ_RANGES))]
    indices_available = []
    for i in range(len(frequency_ranges_available)):
        if frequency_ranges_available[i]:
            indices_available.append(i)

    if params.logs:
        # print("Frequency ranges available boolean array: {}".format(frequency_ranges_available))
        # print("Available indices array: {}".format(indices_available))
        print("--------------------------------------------------------")
    return removed_freq_range, frequency_ranges_available, indices_available


def demodulate(samples_received, removed_freq_range, frequency_ranges_available, indices_available):
    """
    Perform a demodulation of the signal from pass-band to base-band.

    :param samples_received:            The filtered samples received from the server
    :param removed_freq_range:          The index of the removed frequency range
    :param frequency_ranges_available:  An array of booleans where True indicates that the corresponding frequency range
                                            is available
    :param indices_available:           An array of the indices of the available frequency ranges
    :return:                            The demodulated samples
    """
    if params.logs:
        print("Demodulation...")
    if params.MOD == 1 or params.MOD == 2:
        if params.MOD == 1:
            fc = np.mean(params.FREQ_RANGES[frequency_ranges_available.index(True)])

        # TODO: improve this in term of the frequency_ranges_available array above
        else:
            if removed_freq_range == 0 or removed_freq_range == 1:
                fc = np.mean([params.FREQ_RANGES[2][0], params.FREQ_RANGES[3][1]])
            else:
                fc = np.mean([params.FREQ_RANGES[0][0], params.FREQ_RANGES[2][1]])
        if params.logs:
            print("Chosen fc for demodulation: {}".format(fc))

        demodulated_samples = fourier_helper.demodulate(samples_received, fc)
        if params.plots:
            plot_helper.samples_fft_plots(demodulated_samples, "Demodulated received samples", shift=True)

    elif params.MOD == 3:
        demodulated_samples = []
        demodulation_frequencies = np.mean(params.FREQ_RANGES, axis=1)
        for i in range(len(params.FREQ_RANGES)):
            if frequency_ranges_available[i]:
                demodulated_samples.append(fourier_helper.demodulate(samples_received, demodulation_frequencies[i]))
        if params.logs:
            print("Demodulation frequencies: {}".format(demodulation_frequencies))
        if params.plots:
            for i in range(len(indices_available)):
                plot_helper.samples_fft_plots(
                    demodulated_samples[i],
                    "Demodulated received samples {}".format(indices_available[i]), shift=True)

    else:
        raise ValueError('This modulation type does not exist yet... He he he')
    if params.logs:
        print("--------------------------------------------------------")
    return demodulated_samples


def low_pass(demodulated_samples, indices_available):
    """
    Low pass the demodulated samples to get rid of the noise and the other frequencies

    :param demodulated_samples: The samples received from the server, demodulated (i.e base-band)
    :param indices_available:   An array of the indices of the available frequency ranges
    :return:                    The demodulated samples low-passed, i.e the only frequencies of interest for the samples
    """
    if params.logs:
        print("Low passing...")
    _, h = pulses.root_raised_cosine()
    h_matched = np.conjugate(h[::-1])

    if params.MOD == 1 or params.MOD == 2:
        y = np.convolve(demodulated_samples, h_matched)
        if params.plots:
            plot_helper.samples_fft_plots(y, "Low-passed samples", shift=True)
        if params.logs:
            print("Length of y: {}".format(len(y)))
    elif params.MOD == 3:
        y = []
        for i in range(len(demodulated_samples)):
            y.append(np.convolve(demodulated_samples[i], h_matched))
        if params.plots:
            for i in range(len(y)):
                plot_helper.samples_fft_plots(
                    y[i], "Low-passed samples {}".format(indices_available[i]), shift=True)
        if params.logs:
            print("Y shape:")
            for i in range(len(y)):
                print(np.shape(y[i]))
    else:
        raise ValueError('This modulation type does not exist yet... He he he')
    if params.logs:
        print("--------------------------------------------------------")
    return y


def find_delay(samples, preamble_samples_sent, frequency_ranges_available):
    """
    Find the delay

    :param samples:                     The received samples form the server, demodulated and low-passed
    :param preamble_samples_sent:       The preamble samples used as a synchronization sequence, and previously stored
    :param frequency_ranges_available:  An array of booleans where True indicates that the corresponding frequency range
                                            is available
    :return:                            The delay found in number of samples
    """
    if params.logs:
        print("Finding the delay...")
    if params.MOD == 1 or params.MOD == 2:
        delay = parameter_estim.ML_theta_estimation(samples, preamble_samples_sent)
    elif params.MOD == 3:
        delays = []
        for i in range(len(samples)):
            if frequency_ranges_available[i]:
                delays.append(parameter_estim.ML_theta_estimation(samples[i], preamble_samples_sent))
        delay = int(np.round(np.mean(delays)))
        if params.plots:
            plot_helper.delay_plots(samples, delay, "Delays estimated (only real part is shown)")
    else:
        raise ValueError('This modulation type does not exist yet... He he he')
    if params.logs:
        print("Delay: {} samples".format(delay))
        print("--------------------------------------------------------")
    return delay


def extract_preamble_samples(samples, delay, preamble_samples_sent, frequency_ranges_available, indices_available):
    """
    Extract the preamble samples from the received samples (demodulated and low-passed)

    :param samples:                     The received samples (previously demodulated and low-passed)
    :param delay:                       The delay found previously in number of samples
    :param preamble_samples_sent:       The preamble samples used as a synchronization sequence, and previously stored
    :param frequency_ranges_available:  An array of booleans where True indicates that the corresponding frequency range
                                            is available
    :param indices_available:           An array of the indices of the available frequency ranges
    :return:                            The preamble samples received
    """
    if params.logs:
        print("Extracting the preamble samples...")
    len_preamble_samples_sent = len(preamble_samples_sent)
    if params.MOD == 1 or params.MOD == 2:
        preamble_samples_received = samples[delay:delay + len_preamble_samples_sent]
        if params.plots:
            plot_helper.two_simple_plots(np.real(preamble_samples_received), np.real(preamble_samples_sent),
                                         "Preamble samples received vs preamble samples sent", "received", "expected")
        if params.logs:
            print("Number of samples for the actual preamble: {} samples".format(len_preamble_samples_sent))
            print("Number of samples for the received preamble: {} samples".format(len(preamble_samples_received)))
    elif params.MOD == 3:
        preamble_samples_received = []
        for i in range(len(samples)):
            preamble_samples_received.append(samples[i][delay:delay + len_preamble_samples_sent])
        # if params.plots:
        #     for i in range(len(preamble_samples_received)):
        #         plot_helper.compare_preambles(preamble_samples_received[i], preamble_samples_sent,
        #                                       "{}: Preamble samples received vs preamble samples sent"
        #                                       .format(indices_available[i]))
    else:
        raise ValueError('This modulation type does not exist yet... He he he')

    if params.logs:
        print("--------------------------------------------------------")
    return preamble_samples_received


def estimate_parameters(preamble_samples_sent, preamble_samples_received, indices_available):
    """
    Estimate the phase shift and the scaling factor by comparing the preamble samples received and the preamble samples
    sent

    :param preamble_samples_sent:       The preamble samples originally sent by the transmitter
    :param preamble_samples_received:   The preamble samples received from the server, extracted previously
    :param indices_available:           An array of the indices of the available frequency ranges
    :return:                            The estimations of the phase shift and the scaling factor
    """
    if params.logs:
        print("Computing the phase shift and the scaling factor...")
    # Remove SPAN/2 samples in the end because there is still data there for the received preamble
    half_span_h = int(params.SPAN / 2)
    len_preamble_samples_sent = len(preamble_samples_sent)
    if params.MOD == 1 or params.MOD == 2:
        phase_shift_estim, scaling_factor_estim = parameter_estim.ML_phase_scaling_estim(
            preamble_samples_sent[:len_preamble_samples_sent - half_span_h],
            preamble_samples_received[:len(preamble_samples_received) - half_span_h])
        if params.logs:
            print("Phase shift: {}".format(phase_shift_estim))
            # print("Scaling factor: {}".format(scaling_factor_estim))
    elif params.MOD == 3:
        phase_shift_estim = []
        scaling_factor_estim = []
        for i in range(len(preamble_samples_received)):
            phase_shift_estim_in_range, scaling_factor_estim_in_range = parameter_estim.ML_phase_scaling_estim(
                preamble_samples_sent[:len_preamble_samples_sent - half_span_h],
                preamble_samples_received[i][:len(preamble_samples_received[i]) - half_span_h])
            phase_shift_estim.append(phase_shift_estim_in_range)
            scaling_factor_estim.append(scaling_factor_estim_in_range)
        if params.logs:
            for i in range(len(phase_shift_estim)):
                print("Phase shift {}: {}".format(indices_available[i], phase_shift_estim[i]))
                # print("Scaling factor {}: {}\n".format(indices_available[i], scaling_factor_estim[i]))
    else:
        raise ValueError('This modulation type does not exist yet... He he he')
    if params.logs:
        print("--------------------------------------------------------")
    return phase_shift_estim, scaling_factor_estim


def crop_samples_1(samples, delay, len_preamble_samples_sent, indices_available):
    """
    Crop the samples by removing the delay, the preamble, and adjusting to the first symbol of data

    :param samples:                     The received samples (previously demodulated and low-passed)
    :param delay:                       The delay computed previously (representing the number of samples before the
                                            apparition of the first preamble)
    :param len_preamble_samples_sent:   The length in number of samples of the preamble originally sent
    :param indices_available:           An array of the indices of the available frequency ranges
    :return:                            The samples cropped, i.e the samples representing the data, the ending preamble,
                                            and the noise at the end
    """
    if params.logs:
        print("Cropping the samples (removing the delay, the preamble, "
              "and adjusting to the first relevant sample of data)...")
    half_span_h = int(params.SPAN / 2)
    if params.MOD == 1 or params.MOD == 2:
        data_samples = samples[delay + len_preamble_samples_sent - half_span_h + params.USF:]
        if params.plots:
            plot_helper.plot_complex_function(samples, "y before removing anything")
            plot_helper.plot_complex_function(data_samples, "y after removing the delay, the preamble, and adjusting")
    elif params.MOD == 3:
        data_samples = []
        for i in range(len(samples)):
            data_samples.append(samples[i][delay + len_preamble_samples_sent - 1 - half_span_h + 1 + params.USF:])
        # if params.plots:
        #     for i in range(len(data_samples)):
        #         plot_helper.plot_complex_function(data_samples[i],
        #                                           "Samples {} after removing the delay, the preamble, and adjusting".
        #                                           format(indices_available[i]))
    else:
        raise ValueError('This modulation type does not exist yet... He he he')
    if params.logs:
        print("--------------------------------------------------------")
    return data_samples


def find_second_preamble_index(data_samples, preamble_samples_sent):
    """
    Find the number of samples before the beginning of the preamble at the end of the samples originally sent

    :param data_samples:            The samples containing the data, the preamble at the end, and the noise at the very
                                        end
    :param preamble_samples_sent:   The preamble samples originally sent by the transmitter
    :return:                        The number of samples before the beginning of the preamble at the end
    """
    if params.logs:
        print("Finding the second preamble index...")
    if params.MOD == 1 or params.MOD == 2:
        second_preamble_index = parameter_estim.ML_theta_estimation(data_samples,
                                                                    preamble_samples=preamble_samples_sent[::-1])
    elif params.MOD == 3:
        second_preamble_index = []
        for i in range(len(data_samples)):
            second_preamble_index.append(parameter_estim.ML_theta_estimation(
                data_samples[i], preamble_samples=preamble_samples_sent[::-1]))
        second_preamble_index = int(np.round(np.mean(second_preamble_index)))
    else:
        raise ValueError('This modulation type does not exist yet... He he he')
    if params.logs:
        print("Second preamble index: {} samples".format(second_preamble_index))
        print("--------------------------------------------------------")
    return second_preamble_index


def crop_samples_2(data_samples, second_preamble_index):
    """
    Crop the data samples to remove the preamble at the end (and the noise at the end by the same occasion), and adjust
    to the last sample of data

    :param data_samples:            The samples containing the data, the preamble at the end, and the noise at the very
                                        end
    :param second_preamble_index:   The number of samples before the beginning of the preamble at the end
    :return:                        The data samples
    """
    if params.logs:
        print("Cropping the samples (removing the garbage at the end)...")
    half_span_h = int(params.SPAN / 2)
    if params.MOD == 1 or params.MOD == 2:
        data_samples = data_samples[:second_preamble_index + half_span_h - params.USF + 1]
        if params.plots:
            plot_helper.plot_complex_function(data_samples, "y (only data)")
    elif params.MOD == 3:
        for i in range(len(data_samples)):
            data_samples[i] = data_samples[i][:second_preamble_index + half_span_h - params.USF + 1]
        # if params.plots:
        #     for i in range(len(data_samples)):
        #         plot_helper.plot_complex_function(data_samples[i], "y (only data)")
    else:
        raise ValueError("This modulation type does not exist yet... He he he")
    if params.logs:
        print("--------------------------------------------------------")
    return data_samples


def correct_params(data_samples, phase_shift_estim):
    """
    Correct the phase shift and the scaling on the data samples

    :param data_samples:        The data samples
    :param phase_shift_estim:   The phase shift estimation computed previously
    :return:                    The data samples phase shift corrected
    """
    if params.logs:
        print("Correcting the phase shift and the scaling factor (not yet) on the data samples...")
    if params.MOD == 1 or params.MOD == 2:
        data_samples = data_samples * np.exp(-1j * (phase_shift_estim - np.pi / 2))
    elif params.MOD == 3:
        for i in range(len(data_samples)):
            data_samples[i] = data_samples[i] * np.exp(-1j * (phase_shift_estim[i] - np.pi / 2))
    else:
        raise ValueError("This modulation type does not exist yet... He he he")
    if params.logs:
        print("--------------------------------------------------------")
    return data_samples


def downsample(data_samples):
    """
    Down-sample the data samples to obtain the received symbols

    :param data_samples:    The data samples
    :return:                The data symbols received
    """
    if params.logs:
        print("Down-sampling...")
    if params.MOD == 1 or params.MOD == 2:
        data_symbols = data_samples[::params.USF]
        if params.logs:
            print("Number of symbols received: {}".format(len(data_symbols)))
        if params.plots:
            plot_helper.plot_complex_function(data_symbols, "y without preamble")
            plot_helper.plot_complex_symbols(data_symbols, "Symbols received", annotate=False)
    elif params.MOD == 3:
        data_symbols = []
        for i in range(len(data_samples)):
            data_symbols.append(data_samples[i][::params.USF])
        if params.logs:
            print("Shape of the received symbols: {}".format(np.shape(data_symbols)))
        if params.plots:
            for i in range(len(data_symbols)):
                plot_helper.plot_complex_function(data_symbols[i], "Samples without preamble")
                plot_helper.plot_complex_symbols(data_symbols[i], "Symbols received", annotate=False)
    else:
        raise ValueError("This modulation type does not exist yet... He he he")
    if params.logs:
        print("--------------------------------------------------------")
    return data_symbols


def symbols_to_ints(symbols, mapping):
    """
    Map the received symbols to the indices of the closest symbol in the chosen mapping

    :param symbols: The received symbols from the server
    :param mapping: The chosen mapping for the transmission
    :return:        The corresponding indices to the received symbols
    """
    if params.logs:
        print("Mapping symbols to to the mapping indices...")
    if params.MOD == 1 or params.MOD == 2:
        ints = symbols_to_ints_helper(symbols, mapping)
    elif params.MOD == 3:
        ints = []
        for i in range(len(symbols)):
            ints.append(symbols_to_ints_helper(symbols[i], mapping))
    else:
        raise ValueError("This modulation type does not exist yet... He he he")
    if params.logs:
        print("--------------------------------------------------------")
    return ints


def symbols_to_ints_helper(symbols, mapping):
    """
    Help to perform the mapping from the received symbols to the indices of the closest symbol in the chosen mapping
    (to avoid code redundancy)

    :param symbols: The received symbols from the server
    :param mapping: The chosen mapping for the transmission
    :return:        The corresponding indices to the received symbols
    """
    # Make sure symbols and mapping have less or equal than 2 dimensions
    if len(np.shape(symbols)) > 2 or len(np.shape(mapping)) > 2:
        raise AttributeError("One of the vectors symbols and mapping has more than 2 dimensions!")

    # If symbols is a column vector, make it a row vector
    n_elems_axis_0_y = np.size(symbols, 0)
    if n_elems_axis_0_y != 1:
        symbols = np.reshape(symbols, (1, n_elems_axis_0_y))
    else:
        symbols = np.reshape(symbols, (1, np.size(symbols, 1)))

    # If mapping is a row vector, make it a column vector
    if np.size(mapping, 0) == 1:
        mapping = np.reshape(mapping, (np.size(mapping, 1), 1))
    else:
        mapping = np.reshape(mapping, (np.size(mapping, 0), 1))

    # Number of symbols in the mapping
    M = len(mapping)
    # Number of symbols received
    S = len(symbols)

    distances = np.transpose(abs(np.tile(symbols, (M, 1)) - np.tile(mapping, (1, S))))
    ints = np.argmin(distances, 1)
    return ints


def ints_to_message(ints, removed_freq_range):
    """
    Convert the integers (indices of the chosen mapping) to the final received message

    :param ints:                The indices of the symbols that are the closest to the received symbols
    :param removed_freq_range:  The removed frequency range by the server (to know if we use the parity check or not)
    :return:                    The received message
    """
    if params.MOD == 1 or params.MOD == 2:
        # Convert the ints to BITS_PER_SYMBOL bits
        bits_separated = ["{0:0{bits_per_symbol}b}".format(i, bits_per_symbol=params.BITS_PER_SYMBOL) for i in ints]
        if params.logs:
            print("Groups of BITS_PER_SYMBOL bits representing each integer:\n{}".format(bits_separated))

        # Make a new string with it
        bits_separated = ''.join(bits_separated)
        if params.logs:
            print("Bits grouped all together:\n{}".format(bits_separated))

        # Slice the string into substrings of 7 characters
        bits_separated = [bits_separated[i:i + 7] for i in range(0, len(bits_separated), 7)]
        if params.logs:
            print("Groups of 7 bits:\n{}".format(bits_separated))

        # Add a zero at the beginning of each substring (cf transmitter)
        new_bits = []
        for sub_string in bits_separated:
            new_bits.append('0' + sub_string)
        if params.logs:
            print("Groups of 8 bits (0 added at the beginning, cf. transmitter):\n{}".format(new_bits))

    elif params.MOD == 3:
        bits_grouped_by_bits_per_symbol = []
        for j in range(len(ints)):
            bits_grouped_by_bits_per_symbol.append(
                ["{0:0{bits_per_symbol}b}".format(i, bits_per_symbol=params.BITS_PER_SYMBOL) for i in ints[j]])
        if params.logs:
            print("Bits grouped by groups of BITS_PER_SYMBOL bits: {}".format(
                np.shape(bits_grouped_by_bits_per_symbol)))
            for i in range(len(bits_grouped_by_bits_per_symbol)):
                print(bits_grouped_by_bits_per_symbol[i])
            print()

        # Make an array of strings with it
        bits_grouped = []
        for j in range(len(bits_grouped_by_bits_per_symbol)):
            bits_grouped.append(''.join(bits_grouped_by_bits_per_symbol[j]))
        if params.logs:
            print("Bits grouped: {}".format(np.shape(bits_grouped)))
            for i in range(len(bits_grouped)):
                print(bits_grouped[i])
            print()

        # Separate bits
        bits_separated = []
        for i in range(len(bits_grouped)):
            bits_alone = []
            for j in range(len(bits_grouped[i])):
                bits_alone.append(int(bits_grouped[i][j]))
            bits_separated.append(bits_alone)
        if params.logs:
            print("Bits separated: {}".format(np.shape(bits_separated)))
            for i in range(len(bits_separated)):
                print(bits_separated[i])
            print()

        if params.logs:
            print("Removed frequency range: {}".format(removed_freq_range))
        if removed_freq_range != 3:
            if params.logs:
                print("--> We have to reconstruct the missing bit stream")
            parity_check = np.sum(bits_separated, axis=0)

            reconstructed_bit_stream = []
            for j in range(len(parity_check)):
                reconstructed_bit_stream.append(0 if parity_check[j] % 2 == 0 else 1)
            if params.logs:
                print(
                    "Missing bit stream: {}\n{}\n".format(np.shape(reconstructed_bit_stream), reconstructed_bit_stream))

            bit_streams_to_use = bits_separated[:len(bits_separated) - 1]
            bit_streams_to_use.insert(removed_freq_range, reconstructed_bit_stream)
        else:
            if params.logs:
                print("--> We can use the received bit streams\n")
            bit_streams_to_use = bits_separated

        if params.logs:
            print("Bit streams to use: {}".format(np.shape(bits_separated)))
            for i in range(len(bit_streams_to_use)):
                print(bit_streams_to_use[i])
            print()

        # Isolate the bits
        bits_separated = []
        for i in range(len(bit_streams_to_use[0])):
            for j in range(len(bit_streams_to_use)):
                bits_separated.append(int(bit_streams_to_use[j][i]))
        if params.logs:
            print("Bits in order: {}\n{}\n".format(np.shape(np.shape(bits_separated)), bits_separated))

        # Slice the string into substrings of 7 characters
        bits_separated = [bits_separated[i:i + 7] for i in range(0, len(bits_separated), 7)]
        if params.logs:
            print("Bits in groups of 7: {}\n{}\n".format(np.shape(bits_separated), bits_separated))

        # Remove the last element if it is not composed of 7 bits (comes from the rounding)
        if len(bits_separated[len(bits_separated) - 1]) != 7:
            bits_separated = bits_separated[:len(bits_separated) - 1]

        # Re-put the zero at the beginning (c.f transmitter)
        strings = []
        for i in range(len(bits_separated)):
            strings.append(''.join(str(x) for x in bits_separated[i]))
        new_bits = []
        for sub_string in strings:
            new_bits.append('0' + str(sub_string))
        if params.logs:
            print("Final bytes: {}\n{}\n".format(np.shape(new_bits), new_bits))
    else:
        raise ValueError("This modulation type does not exist yet... He he he")

    # Convert from array of bytes to string
    message_received = ''.join(helper.bits2string(new_bits))

    message_sent = read_write.read_message_sent()
    if params.logs:
        print("Message sent ({} characters):     {}".format(len(message_sent), message_sent))
        print("Message received ({} characters): {}".format(len(message_received), message_received))
        helper.compare_messages(message_sent, message_received)
    else:
        print("Message received ({} characters):\n{}".format(len(message_received), message_received))
    read_write.write_message_received(message_received)
    if params.logs:
        print("--------------------------------------------------------")
    return message_received
