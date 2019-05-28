import numpy as np
import time
import sys

import fourier_helper
import helper
import mappings
import parameter_estim
import params
import plot_helper
import pulses
import read_write


# TODO: merge both methods
def decoder(y, mapping):
    """
    Map the received symbols to the closest symbols of our mapping
    :param y: the observation vector, i.e the received symbols
    :param mapping: the chosen mapping for the communication
    :return: integers between 0 and M-1, i.e integers corresponding to the bits sent
    """
    # Number of symbols in the mapping
    M = len(mapping)
    # Number of symbols received
    S = len(y)

    distances = np.transpose(abs(np.tile(y, (M, 1)) - np.tile(mapping, (1, S))))
    ints = np.argmin(distances, 1)
    if params.logs:
        print("Equivalent integers:\n{}".format(ints))
    return ints


def ints_to_message(ints):
    """
    Map the integers (i.e indices in our mapping) to the received message
    :param ints: integers between 0 and M-1, i.e integers corresponding to the bits sent
    :return: the corresponding received message
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
    message_received = ''.join(helper.bits2string(new_bits))

    message_sent = read_write.read_message_sent()
    print("Message sent:     {}".format(message_sent))
    print("Message received: {}".format(message_received))
    helper.compare_messages(message_sent, message_received)

    return message_received


# TODO: Slice it in smaller methods
def received_from_server():
    # Prepare the data -------------------------------------------------------------------------------------------------
    if params.logs:
        print("Preparing the data...")
    # Load the input and output samples from their respective files
    input_samples = np.loadtxt(params.input_sample_file_path)
    # TODO: put output again
    received_samples = np.loadtxt(params.input_sample_file_path)

    # Plot the input and output samples in Time domain and Frequency domain
    if params.plots:
        plot_helper.samples_fft_plots(input_samples, "Sent samples", complex=False)
        plot_helper.samples_fft_plots(received_samples, "Received samples", complex=False)

    # Read the preamble samples saved previously
    preamble_samples = read_write.read_preamble_samples()
    len_preamble_samples = len(preamble_samples)
    if params.logs:
        print("--------------------------------------------------------")
    # ------------------------------------------------------------------------------------------------------------------

    # Find the frequency range that has been removed -------------------------------------------------------------------
    if params.logs:
        print("Finding the frequency range that has been removed...")
    range_indices, removed_freq_range = fourier_helper.find_removed_freq_range(received_samples)
    if params.logs:
        print("Removed frequency range: {} (range {})".format(removed_freq_range, removed_freq_range + 1))

    # Array of the form [True, True, False, True], where False means that the 3rd frequency range is removes here
    frequency_ranges_available = [True if i != removed_freq_range else False for i in range(len(params.FREQ_RANGES))]
    indices_available = []
    for i in range(len(frequency_ranges_available)):
        if frequency_ranges_available[i]:
            indices_available.append(i)
    if params.logs:
        print("Frequency ranges available boolean array: {}".format(frequency_ranges_available))
        print("Available indices array: {}".format(indices_available))
        print("--------------------------------------------------------")
    # ------------------------------------------------------------------------------------------------------------------

    # Demodulation -----------------------------------------------------------------------------------------------------
    if params.logs:
        print("Demodulation...")
    if params.MODULATION_TYPE == 1 or params.MODULATION_TYPE == 2:
        if params.MODULATION_TYPE == 1:
            fc = np.mean(params.FREQ_RANGES[frequency_ranges_available.index(True)])

        # TODO: improve this in term of the frequency_ranges_available array above
        else:
            if removed_freq_range == 0 or removed_freq_range == 1:
                fc = np.mean([params.FREQ_RANGES[2][0], params.FREQ_RANGES[3][1]])
            else:
                fc = np.mean([params.FREQ_RANGES[0][0], params.FREQ_RANGES[2][1]])
        if params.logs:
            print("Chosen fc for demodulation: {}".format(fc))

        demodulated_samples = fourier_helper.demodulate(received_samples, fc)
        if params.plots:
            plot_helper.samples_fft_plots(demodulated_samples, "Demodulated received samples", shift=True)

    elif params.MODULATION_TYPE == 3:
        demodulated_samples = []
        demodulation_frequencies = np.mean(params.FREQ_RANGES, axis=1)
        for i in range(len(params.FREQ_RANGES)):
            if frequency_ranges_available[i]:
                demodulated_samples.append(fourier_helper.demodulate(received_samples, demodulation_frequencies[i]))
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
    # ------------------------------------------------------------------------------------------------------------------

    # Low pass ---------------------------------------------------------------------------------------------------------
    if params.logs:
        print("Low passing...")
    _, h = pulses.root_raised_cosine()
    half_span_h = int(params.SPAN / 2)
    h_matched = np.conjugate(h[::-1])

    if params.MODULATION_TYPE == 1 or params.MODULATION_TYPE == 2:
        y = np.convolve(demodulated_samples, h_matched)
        if params.plots:
            plot_helper.samples_fft_plots(y, "Low-passed samples", shift=True)
        if params.logs:
            print("Length of y: {}".format(len(y)))
    elif params.MODULATION_TYPE == 3:
        y = []
        for i in range(len(demodulated_samples)):
            y.append(np.convolve(demodulated_samples[i], h_matched))
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
    # ------------------------------------------------------------------------------------------------------------------

    # Find the delay ---------------------------------------------------------------------------------------------------
    if params.logs:
        print("Finding the delay...")
    if params.MODULATION_TYPE == 1 or params.MODULATION_TYPE == 2:
        delay = parameter_estim.ML_theta_estimation(y, preamble_samples)
    elif params.MODULATION_TYPE == 3:
        delays = []
        for i in range(len(y)):
            if frequency_ranges_available[i]:
                delays.append(parameter_estim.ML_theta_estimation(y[i], preamble_samples))
        delay = int(np.round(np.mean(delays)))
        if params.plots:
            plot_helper.delay_plots(y, delay, "Delays estimated (only real part is shown)")
    else:
        raise ValueError('This modulation type does not exist yet... He he he')
    if params.logs:
        print("Delay: {} samples".format(delay))
        print("--------------------------------------------------------")
    # ------------------------------------------------------------------------------------------------------------------

    # Extract the preamble samples -------------------------------------------------------------------------------------
    if params.logs:
        print("Extracting the preamble samples...")
    if params.MODULATION_TYPE == 1 or params.MODULATION_TYPE == 2:
        preamble_samples_received = y[delay:delay + len_preamble_samples]
        if params.plots:
            plot_helper.two_simple_plots(np.real(preamble_samples_received), np.real(preamble_samples),
                                         "Preamble samples received vs preamble samples sent", "received", "expected")
        if params.logs:
            print("Number of samples for the actual preamble: {} samples".format(len_preamble_samples))
            print("Number of samples for the received preamble: {} samples".format(len(preamble_samples_received)))
    elif params.MODULATION_TYPE == 3:
        preamble_samples_received = []
        for i in range(len(y)):
            preamble_samples_received.append(y[i][delay:delay + len_preamble_samples])
        if params.plots:
            for i in range(len(preamble_samples_received)):
                if frequency_ranges_available[i]:
                    plot_helper.compare_preambles(preamble_samples_received[i], preamble_samples,
                                                  "Preamble samples received {} vs preamble samples sent"
                                                  .format(indices_available[i]))
    else:
        raise ValueError('This modulation type does not exist yet... He he he')

    if params.logs:
        print("--------------------------------------------------------")
    # ------------------------------------------------------------------------------------------------------------------

    # Compute the phase shift and the scaling factor -------------------------------------------------------------------
    if params.logs:
        print("Computing the phase shift and the scaling factor...")
    # Remove SPAN/2 samples in the end because there is still data there for the received preamble
    if params.MODULATION_TYPE == 1 or params.MODULATION_TYPE == 2:
        phase_shift_estim, scaling_factor_estim = parameter_estim.ML_phase_scaling_estim(
            preamble_samples[:len_preamble_samples - half_span_h],
            preamble_samples_received[:len(preamble_samples_received) - half_span_h])
        if params.logs:
            print("Phase shift: {}".format(phase_shift_estim))
            print("Scaling factor: {}".format(scaling_factor_estim))
    elif params.MODULATION_TYPE == 3:
        phase_shift_estim_array = []
        scaling_factor_estim_array = []
        for i in range(len(preamble_samples_received)):
            phase_shift_estim_in_range, scaling_factor_estim_in_range = parameter_estim.ML_phase_scaling_estim(
                preamble_samples[:len_preamble_samples - half_span_h],
                preamble_samples_received[i][:len(preamble_samples_received[i]) - half_span_h])
            phase_shift_estim_array.append(phase_shift_estim_in_range)
            scaling_factor_estim_array.append(scaling_factor_estim_in_range)
        if params.logs:
            for i in range(len(phase_shift_estim_array)):
                print("Phase shift {}: {}".format(indices_available[i], phase_shift_estim_array[i]))
                print("Scaling factor {}: {}".format(indices_available[i], scaling_factor_estim_array[i]))
    else:
        raise ValueError('This modulation type does not exist yet... He he he')
    if params.logs:
        print("--------------------------------------------------------")
    # ------------------------------------------------------------------------------------------------------------------

    # Crop the samples (remove the delay, the preamble, and adjust to the first relevant sample of data) ---------------
    if params.logs:
        print("Cropping the samples (removing the delay, the preamble, "
              "and adjusting to the first relevant sample of data)...")
    if params.MODULATION_TYPE == 1 or params.MODULATION_TYPE == 2:
        data_samples = y[delay + len_preamble_samples - half_span_h + params.USF:]
        if params.plots:
            plot_helper.plot_complex_function(y, "y before removing anything")
            plot_helper.plot_complex_function(data_samples, "y after removing the delay, the preamble, and adjusting")
    elif params.MODULATION_TYPE == 3:
        data_samples = []
        for i in range(len(y)):
            data_samples.append(y[i][delay + len_preamble_samples - 1 - half_span_h + 1 + params.USF:])
        if params.plots:
            for i in range(len(data_samples)):
                plot_helper.plot_complex_function(data_samples[i],
                                                  "y[{}] after removing the delay, the preamble, and adjusting".
                                                  format(indices_available[i]))
    else:
        raise ValueError('This modulation type does not exist yet... He he he')
    if params.logs:
        print(params.USF)
        print("--------------------------------------------------------")
    # ------------------------------------------------------------------------------------------------------------------

    # Find the second_preamble_index -----------------------------------------------------------------------------------
    if params.logs:
        print("Finding the second preamble index...")
    if params.MODULATION_TYPE == 1 or params.MODULATION_TYPE == 2:
        second_preamble_index = parameter_estim.ML_theta_estimation(data_samples,
                                                                    preamble_samples=preamble_samples[::-1])
    elif params.MODULATION_TYPE == 3:
        second_preamble_index = []
        for i in range(len(data_samples)):
                second_preamble_index.append(parameter_estim.ML_theta_estimation(
                    data_samples[i], preamble_samples=preamble_samples[::-1]))
        second_preamble_index = int(np.round(np.mean(second_preamble_index)))
    else:
        raise ValueError('This modulation type does not exist yet... He he he')
    if params.logs:
        print("Second preamble index: {} samples".format(second_preamble_index))
        print("--------------------------------------------------------")
    # ------------------------------------------------------------------------------------------------------------------

    # Crop the samples (remove the garbage at the end) -----------------------------------------------------------------
    if params.logs:
        print("Cropping the samples (removing the garbage at the end)...")
    if params.MODULATION_TYPE == 1 or params.MODULATION_TYPE == 2:
        data_samples = data_samples[:second_preamble_index + half_span_h - params.USF + 1]
        if params.plots:
            plot_helper.plot_complex_function(data_samples, "y (only data)")
    elif params.MODULATION_TYPE == 3:
        for i in range(len(data_samples)):
            data_samples[i] = data_samples[i][:second_preamble_index + half_span_h - params.USF + 1]
        if params.plots:
            for i in range(len(data_samples)):
                plot_helper.plot_complex_function(data_samples[i], "y (only data)")
    else:
        raise ValueError("This modulation type does not exist yet... He he he")
    if params.logs:
        print("--------------------------------------------------------")
    # ------------------------------------------------------------------------------------------------------------------

    # Correct the phase shift on the data samples ----------------------------------------------------------------------
    if params.logs:
        print("Correcting the phase shift and the scaling factor (not yet) on the data samples...")
    if params.MODULATION_TYPE == 1 or params.MODULATION_TYPE == 2:
        data_samples = data_samples * np.exp(-1j * (phase_shift_estim - np.pi / 2))
    elif params.MODULATION_TYPE == 3:
        print(len(data_samples))
        for i in range(len(data_samples)):
            data_samples[i] = data_samples[i] * np.exp(-1j * (phase_shift_estim_array[i] - np.pi / 2))
    else:
        raise ValueError("This modulation type does not exist yet... He he he")
    if params.logs:
        print("--------------------------------------------------------")
    # ------------------------------------------------------------------------------------------------------------------

    # Down-sample the samples to obtain the symbols --------------------------------------------------------------------
    if params.logs:
        print("Down-sampling...")
    if params.MODULATION_TYPE == 1 or params.MODULATION_TYPE == 2:
        data_symbols = data_samples[::params.USF]
        if params.logs:
            print("Number of symbols received: {}".format(len(data_symbols)))
        if params.plots:
            plot_helper.plot_complex_function(data_symbols, "y without preamble")
            plot_helper.plot_complex_symbols(data_symbols, "Symbols received", annotate=False)
    elif params.MODULATION_TYPE == 3:
        data_symbols = []
        for i in range(len(data_samples)):
            data_symbols.append(data_samples[i][::params.USF])
        if params.logs:
            print("Shape of the received symbols: {}".format(np.shape(data_symbols)))
        if params.plots:
            for i in range(len(data_symbols)):
                plot_helper.plot_complex_function(data_symbols[i], "y without preamble")
                plot_helper.plot_complex_symbols(data_symbols[i], "Symbols received", annotate=False)
    else:
        raise ValueError("This modulation type does not exist yet... He he he")
    if params.logs:
        print("--------------------------------------------------------")
    # ------------------------------------------------------------------------------------------------------------------

    # Decode the symbols -----------------------------------------------------------------------------------------------
    if params.logs:
        print("Decoding the symbols...")
    if params.MODULATION_TYPE == 1 or params.MODULATION_TYPE == 2:
        ints = decoder(data_symbols, np.asarray(mappings.choose_mapping()))
        message_received = ints_to_message(ints)
        read_write.write_message_received(message_received)
    # elif params.MODULATION_TYPE == 3:
        # TODO
    else:
        raise ValueError("This modulation type does not exist yet... He he he")
    if params.logs:
        print("--------------------------------------------------------")


# Intended for testing (to run the program, run main.py)
if __name__ == "__main__":
    if params.logs:
        moment = time.strftime("%Y-%b-%d__%H_%M_%S", time.localtime())
        log_file = open("../logs/" + moment + ".log", "w+")
        sys.stdout = log_file
    received_from_server()

# TODO: make sure the scaling factor, delay, phase shift must be the same for 3 freq ranges
