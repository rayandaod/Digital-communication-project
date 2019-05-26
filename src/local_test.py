import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfilt, sosfreqz
from scipy.signal import upfirdn

import fourier_helper
import mappings
import parameter_estim
import params
import plot_helper
import preambles
import pulses
import read_write
import receiver
import transmitter

"""
Local test, with a homemade simulation of the real server
"""

FILTER_ORDER = 30


def butter_bandpass(low_cut_freq, high_cut_freq, Fs=params.Fs, order=5):
    """
    :param low_cut_freq: the low cut frequency
    :param high_cut_freq: the high cut frequency
    :param Fs: the sampling frequency
    :param order: the order of the filter
    :return:
    """
    nyq = 0.5 * Fs
    low = low_cut_freq / nyq
    high = high_cut_freq / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, low_cut_freq, high_cut_freq, Fs=params.Fs, order=5):
    """
    :param data: the samples to filter
    :param low_cut_freq: the low cut frequency
    :param high_cut_freq: the high cut frequency
    :param Fs: the sampling frequency
    :param order: the order of the filter
    :return: the filtered samples
    """
    sos = butter_bandpass(low_cut_freq, high_cut_freq, Fs, order=order)
    y = sosfilt(sos, data)
    if params.plots:
        w, h = sosfreqz(sos, worN=2000)
        plt.plot((Fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
        plt.title("Output of butter bandpass")
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.grid(True)
        plt.legend(loc='best')
        plt.show()
    return y


def server_simulation(samples, clip=True, filter_freq=True, delay_start=True, delay_end=True, noise=True, scale=True):
    """
    Simulate a server that clips the data to [-1, 1] adds delay, AWGN(0, params.NOISE_VAR), and some garbage at the end
    :param scale: rather we scale the samples or not
    :param noise: rather we noise the signal or not
    :param delay_end: rather we add delay at the end or not
    :param delay_start: rather we add delay at the beginning
    :param filter_freq: rather we remove one of the 4 frequency ranges
    :param clip: rather we clip the samples to [-1, 1]
    :param samples: the samples to filter
    :return: the filtered samples
    """
    print("Channel simulation...")

    # Clip the data to [-1, 1]
    if clip:
        samples = np.clip(samples, -1, 1)
        print("Samples clipped to [-1, 1]")

    # # Remove 1 frequency range among the 4 authorized ranges
    if filter_freq:
        range_to_remove = np.random.randint(4)
        if range_to_remove == 0:
            low_cut_freq = params.FREQ_RANGES[1][0]
            high_cut_freq = params.FREQ_RANGES[3][1]
            samples = butter_bandpass_filter(samples, low_cut_freq, high_cut_freq, order=FILTER_ORDER)
        elif range_to_remove == 1:
            low_cut_freq_1 = params.FREQ_RANGES[0][0]
            high_cut_freq_1 = params.FREQ_RANGES[0][1]
            low_cut_freq_2 = params.FREQ_RANGES[2][0]
            high_cut_freq_2 = params.FREQ_RANGES[3][1]
            samples_1 = butter_bandpass_filter(samples, low_cut_freq_1, high_cut_freq_1, order=FILTER_ORDER)
            samples_2 = butter_bandpass_filter(samples, low_cut_freq_2, high_cut_freq_2, order=FILTER_ORDER)
            samples = (samples_1 + samples_2) / 2
        elif range_to_remove == 2:
            low_cut_freq_1 = params.FREQ_RANGES[0][0]
            high_cut_freq_1 = params.FREQ_RANGES[1][1]
            low_cut_freq_2 = params.FREQ_RANGES[3][0]
            high_cut_freq_2 = params.FREQ_RANGES[3][1]
            samples_1 = butter_bandpass_filter(samples, low_cut_freq_1, high_cut_freq_1, order=FILTER_ORDER)
            samples_2 = butter_bandpass_filter(samples, low_cut_freq_2, high_cut_freq_2, order=FILTER_ORDER)
            samples = (samples_1 + samples_2) / 2
        else:
            low_cut_freq = params.FREQ_RANGES[0][0]
            high_cut_freq = params.FREQ_RANGES[3][0]
            samples = butter_bandpass_filter(samples, low_cut_freq, high_cut_freq, order=FILTER_ORDER)
        print("Frequency range removed: {}".format(range_to_remove))

    # Introduce a delay at the beginning
    if delay_start:
        delay_start = np.zeros(np.random.randint(params.Fs))
        samples = np.concatenate((delay_start, samples))
        print("Delay introduced at the beginning: {} samples".format(len(delay_start)))

    # Introduce a delay at the end
    if delay_end:
        delay_end = np.zeros(np.random.randint(params.Fs / 5))
        print("Delay introduced at the end: {} samples".format(len(delay_end)))
        samples = np.concatenate((samples, delay_end))

    # Introduce AWGN(0, params.NOISE_VAR)
    if noise:
        samples += np.random.normal(0, np.sqrt(params.NOISE_VAR), size=len(samples))

    # Scale the samples down
    if scale:
        channel_scaling = 1 / (np.random.randint(5) + 1)
        samples = channel_scaling * samples
        print("Scaling introduced: {}".format(channel_scaling))

    # Clip the data to [-1, 1]
    if clip:
        samples = np.clip(samples, -1, 1)
        print("Samples clipped to [-1, 1]")
        print("--------------------------------------------------------")

    return samples


def local_test():
    """
    Test the design locally with modulation and demodulation
    :return: None
    """
    mapping = mappings.choose_mapping()
    ints = transmitter.message_to_ints()
    symbols = transmitter.encoder(ints, mapping)

    # Generate the pulse h
    _, h = pulses.root_raised_cosine()
    half_span_h = int(params.SPAN / 2)

    # Generate the preamble symbols and read it from the corresponding file
    preambles.generate_preamble_symbols(len(symbols))
    preamble_symbols = read_write.read_preamble_symbols()

    # Generate the preamble samples
    preamble_samples = upfirdn(h, preamble_symbols, params.USF)
    len_preamble_samples = len(preamble_samples)

    # Concatenate the preamble symbols with the data symbols
    total_symbols = np.concatenate((preamble_symbols, symbols, preamble_symbols[::-1]))

    # Shape the signal with the pulse h
    total_samples = upfirdn(h, total_symbols, params.USF)

    print("Shaping the preamble...")
    print("Number of symbols for the preamble: {}".format(len(preamble_symbols)))
    print("Number of samples for the preamble: {}".format(len(preamble_samples)))
    print("--------------------------------------------------------")
    plot_helper.plot_complex_function(preamble_samples, "Synchronization sequence shaped, in Time domain")
    plot_helper.fft_plot(preamble_samples, "Synchronization sequence shaped, in Frequency domain", shift=True)

    print("Shaping the preamble-data-preamble...")
    print("Up-sampling factor: {}".format(params.USF))
    print("Number of samples: {}".format(len(total_samples)))
    print("Minimum sample: {}".format(min(total_samples)))
    print("Maximum sample: {}".format(max(total_samples)))
    print("--------------------------------------------------------")
    plot_helper.plot_complex_function(total_samples, "Total samples in Time domain")
    plot_helper.fft_plot(total_samples, "Total samples in Frequency domain", shift=True)

    # Modulate the total_samples
    if params.MODULATION_TYPE == 1:
        samples = fourier_helper.modulate_complex_samples(total_samples, params.np.mean(params.FREQ_RANGES, axis=1))
    elif params.MODULATION_TYPE == 2:
        samples = fourier_helper.modulate_complex_samples(total_samples,
                                                          [params.FREQ_RANGES[0][1], params.FREQ_RANGES[2][1]])
    else:
        raise ValueError('This modulation type does not exist yet... He he he')

    print("Modulation of the signal...")
    print("Number of samples: {}".format(len(samples)))
    print("Minimum sample after modulation: {}".format(min(samples)))
    print("Maximum sample after modulation: {}".format(max(samples)))
    print("--------------------------------------------------------")
    plot_helper.plot_complex_function(samples, "Input samples after modulation, in Time domain")
    plot_helper.fft_plot(samples, "Input samples after modulation, in Frequency domain", shift=True)

    # Scale the signal to the range [-1, 1] (with a bit of uncertainty margin, according to params.ABS_SAMPLE_RANGE)
    samples = (samples / (max(abs(samples))) * params.ABS_SAMPLE_RANGE)
    print("Scaling the signal...")
    print("Minimum sample after scaling: {}".format(min(samples)))
    print("Maximum sample after scaling: {}".format(max(samples)))
    print("--------------------------------------------------------")

    # ----------------------------------------------------------------------------------------------------------------
    # Channel simulation ---------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------
    samples = server_simulation(samples, filter_freq=False)
    # ----------------------------------------------------------------------------------------------------------------
    # Channel simulation's end ---------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    # Supposed to retrieve the preamble symbols and samples from the appropriate files, but here we got it above
    plot_helper.plot_complex_function(samples, "Samples received from the simulated channel, time domain")
    plot_helper.fft_plot(samples, "Samples received from the simulated channel, frequency domain")

    # Find the frequency range that has been removed
    range_indices, removed_freq_range = fourier_helper.find_removed_freq_range(samples)
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

    demodulated_samples = fourier_helper.demodulate(samples, fc)
    plot_helper.plot_complex_function(demodulated_samples, "Demodulated samples in Time domain")
    plot_helper.fft_plot(demodulated_samples, "Demodulated samples in Frequency domain", shift=True)

    # Match filter (i.e Low-pass)
    h_matched = np.conjugate(h[::-1])
    y = np.convolve(demodulated_samples, h_matched)
    plot_helper.plot_complex_function(y, "y in Time domain")
    plot_helper.fft_plot(y, "y in Frequency domain", shift=True)

    # Find the delay
    delay = parameter_estim.ML_theta_estimation(demodulated_samples, preamble_samples=preamble_samples)
    print("Delay: {} samples".format(delay))
    print("--------------------------------------------------------")

    # Extract the preamble samples
    preamble_samples_received = y[half_span_h + delay - 1:half_span_h + delay + len_preamble_samples - 1]
    plot_helper.two_simple_plots(preamble_samples_received.real, preamble_samples.real,
                                 "Comparison between preamble samples received and preamble samples sent",
                                 "received", "expected")
    print("Number of samples for the actual preamble: {}".format(len_preamble_samples))
    print("Number of samples for the received preamble: {}".format(len(preamble_samples_received)))

    # Compute the phase offset
    # We remove the rrc-equivalent-tail because there is data on the tail otherwise
    # TODO: why dot works and not vdot (supposed to conjugate the first term in the formula)
    dot_product = np.dot(preamble_samples[:len_preamble_samples - half_span_h],
                         preamble_samples_received[:len(preamble_samples_received) - half_span_h])
    print("Dot product: {}".format(dot_product))

    preamble_energy = 0
    for i in range(len_preamble_samples - half_span_h):
        preamble_energy += np.absolute(preamble_samples[i]) ** 2
    print("Energy of the preamble: {}".format(preamble_energy))

    frequency_offset_estim = np.angle(dot_product)
    print("Frequency offset: {}".format(frequency_offset_estim))

    scaling_factor = abs(dot_product) / preamble_energy
    print("Scaling factor: {}".format(scaling_factor))

    # Crop the samples (remove the delay, the preamble, and the ramp-up)
    data_samples = y[half_span_h + delay + len_preamble_samples - half_span_h + params.USF - 1 - 1:]

    # Find the second_preamble_index
    second_preamble_index = parameter_estim.ML_theta_estimation(data_samples, preamble_samples=preamble_samples[::-1])
    print("Second preamble index: {} samples".format(second_preamble_index))
    print("--------------------------------------------------------")

    # Crop the samples (remove the preamble, and the garbage at the end)
    data_samples = data_samples[:second_preamble_index + half_span_h - params.USF + 1]
    plot_helper.plot_complex_function(data_samples, "y after removing the delay, the preamble, and the ramp-up")

    # TODO: why frequency_offset - pi/2 works ?
    data_samples = data_samples * np.exp(-1j * (frequency_offset_estim - np.pi / 2))

    # Down-sample the samples to obtain the symbols
    data_symbols = data_samples[::params.USF]
    print("Number of symbols received: {}".format(len(data_symbols)))

    plot_helper.plot_complex_function(data_symbols, "y without preamble")
    plot_helper.plot_complex_symbols(data_symbols, "Symbols received", annotate=False)

    # Decode the symbols
    ints = receiver.decoder(data_symbols, mapping)
    message_received = receiver.ints_to_message(ints)

    message_file = open(params.input_message_file_path)
    message_sent = message_file.readline()
    print(message_received == message_sent)


# Intended for testing (to run the program, run main.py)
if __name__ == "__main__":
    local_test()
