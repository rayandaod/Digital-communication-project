import pulses
from scipy.signal import upfirdn
import numpy as np
import plot_helper
import mappings
import synchronization
import params
import transmitter
import receiver
import fourier_helper
import helper


"""
Testing file
"""

# np.random.seed(30)


def generate_awgn(mean, std):
    return np.random.normal(mean, std, size=np.random.randint(params.Fs))


def local_test():
    """
    Test the design locally with modulation and demodulation
    :return: None
    """
    ints = transmitter.message_to_ints()
    symbols = transmitter.encoder(ints, mappings.mapping)

    # Generate the pulse h
    _, h = pulses.root_raised_cosine()
    half_span_h = int(params.SPAN/2)

    # Generate the preamble symbols and read it from the corresponding file
    synchronization.generate_preamble_symbols(len(symbols))
    preamble_symbols = helper.read_preamble_symbols()

    # Generate the preamble samples
    preamble_samples = upfirdn(h, preamble_symbols, params.USF)
    len_preamble_samples = len(preamble_samples)

    # Concatenate the preamble symbols with the data symbols
    total_symbols = np.concatenate((preamble_symbols, symbols))

    # Shape the signal with the pulse h
    total_samples = upfirdn(h, total_symbols, params.USF)

    print("Shaping the preamble...")
    print("Number of symbols for the preamble: {}".format(len(preamble_symbols)))
    print("Number of samples for the preamble: {}".format(len(preamble_samples)))
    print("--------------------------------------------------------")
    plot_helper.plot_complex_function(preamble_samples, "Synchronization sequence shaped, in Time domain")
    plot_helper.fft_plot(preamble_samples, "Synchronization sequence shaped, in Frequency domain", shift=True)

    print("Shaping the preamble and the data...")
    print("Shaping the data...")
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
        samples = fourier_helper.modulate_complex_samples(total_samples, [params.FREQ_RANGES[0][1], params.FREQ_RANGES[2][1]])
    else:
        raise ValueError('This modulation type does not exist yet... He he he')

    maximum = max(samples)

    if params.verbose:
        print("Modulation of the signal...")
        print("Number of samples: {}".format(len(samples)))
        print("Minimum sample after modulation: {}".format(min(samples)))
        print("Maximum sample after modulation: {}".format(maximum))
        print("--------------------------------------------------------")
        plot_helper.plot_complex_function(samples, "Input samples after modulation, in Time domain")
        plot_helper.fft_plot(samples, "Input samples after modulation, in Frequency domain", shift=True)

    # ----------------------------------------------------------------------------------------------------------------
    # Channel simulation (delay (-> phase shift) and scaling)---------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------
    samples = np.concatenate((generate_awgn(0, 0.05), samples))
    # samples = samples/2
    plot_helper.plot_complex_function(samples, "Samples received from the simulated channel")
    # ----------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    # Supposed to retrieve the preamble symbols and samples from the appropriate files, but here we got it above

    # Demodulate the samples with the appropriate frequency fc
    if params.MODULATION_TYPE == 1:
        fc = 2000
    elif params.MODULATION_TYPE == 2:
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
    delay = synchronization.maximum_likelihood_sync(demodulated_samples, synchronization_sequence=preamble_samples)
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

    preamble_energy = 0
    for i in range(len_preamble_samples - half_span_h):
        preamble_energy += np.absolute(preamble_samples[i])**2
    print("Energy of the preamble: {}".format(preamble_energy))

    frequency_offset_estim = np.angle(dot_product)
    print("Frequency offset: {}".format(frequency_offset_estim))

    scaling_factor = abs(dot_product)/preamble_energy
    print("Scaling factor: {}".format(scaling_factor))

    # Crop the samples (remove the delay, and the ramp-up/ramp-down)
    data_samples = y[delay + params.SPAN - 1:len(y) - params.SPAN + 1]
    plot_helper.plot_complex_function(data_samples, "y after putting the right sampling time")

    # TODO: why frequency_offset - pi/2 works ?
    data_samples = data_samples * np.exp(-1j*(frequency_offset_estim-np.pi/2))

    # Down-sample the samples to obtain the symbols
    symbols_received = data_samples[::params.USF]
    print("Symbols received:\n{}", format(symbols_received))

    # Remove the preamble symbols at the beginning
    data_symbols = symbols_received[len(preamble_symbols):]

    plot_helper.plot_complex_function(data_symbols, "y without preamble")
    plot_helper.plot_complex_symbols(data_symbols, "Symbols received", annotate=False)

    # Decode the symbols
    ints = receiver.decoder(data_symbols, mappings.mapping)
    receiver.ints_to_message(ints)


if __name__ == "__main__":
    local_test()
