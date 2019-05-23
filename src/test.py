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


"""
Testing file
"""


def local_test_without_modulation():
    """
       Test the design locally without modulation nor demodulation
       :return: None
    """
    ints = transmitter.message_to_ints()
    symbols = transmitter.encoder(ints, mappings.mapping)

    # Generate the pulse h
    _, h = pulses.root_raised_cosine()

    preamble = synchronization.generate_sync_sequence(len(symbols))
    preamble_samples = upfirdn(h, preamble, params.USF)

    total_symbols = np.concatenate((preamble, symbols))

    # Shape the signal with the pulse h
    total_samples = upfirdn(h, total_symbols, params.USF)

    print("Shaping the preamble...")
    print("Number of symbols for the preamble: {}".format(len(preamble)))
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

    delay = synchronization.maximum_likelihood_sync(total_samples, synchronization_sequence=preamble_samples)
    print("Delay: {} samples".format(delay))
    print("--------------------------------------------------------")

    h_matched = np.conjugate(h[::-1])
    y = np.convolve(total_samples, h_matched)
    plot_helper.plot_complex_function(y, "y in Time domain")
    plot_helper.fft_plot(y, "y in Frequency domain", shift=True)

    data_samples = y[delay + len(h) - 1:len(y) - len(h) + 1]
    plot_helper.plot_complex_function(data_samples, "y after puting the right sampling time")

    symbols_received = data_samples[::params.USF]
    print("Symbols received:\n{}", format(symbols_received))

    data_symbols = symbols_received[len(preamble):]

    plot_helper.plot_complex_function(data_symbols, "y after puting the right sampling time")
    plot_helper.plot_complex_symbols(data_symbols, "Symbols received", annotate=False)

    ints = receiver.decoder(data_symbols, mappings.mapping)
    receiver.ints_to_message(ints)


def local_test():
    """
    Test the design locally with modulation and demodulation
    :return: None
    """
    ints = transmitter.message_to_ints()
    symbols = transmitter.encoder(ints, mappings.mapping)

    # Generate the pulse h
    _, h = pulses.root_raised_cosine()

    # Generate the preamble symbols and the preamble (shaped) samples
    preamble_symbols = synchronization.generate_sync_sequence(len(symbols))
    preamble_samples = upfirdn(h, preamble_symbols, params.USF)

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
        samples = fourier_helper.modulate(total_samples, params.np.mean(params.FREQ_RANGES, axis=1))
    elif params.MODULATION_TYPE == 2:
        samples = fourier_helper.modulate(total_samples, [params.FREQ_RANGES[0][1], params.FREQ_RANGES[2][1]])
    else:
        raise ValueError('This modulation type does not exist yet... Hehehe')

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
    # Channel simulation
    # ----------------------------------------------------------------------------------------------------------------

    # Supposed to retrieve the preamble symbols and samples from the appropriate files, but here we got it above

    # Demodulate the samples with the appropriate frequency fc
    demodulated_samples = fourier_helper.demodulate(samples, 2000)
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

    # Crop the samples (remove the delay, and the ramp-up/ramp-down)
    data_samples = y[delay + len(h) - 1:len(y) - len(h) + 1]
    plot_helper.plot_complex_function(data_samples, "y after puting the right sampling time")

    # Down-sample the samples to obtain the symbols
    symbols_received = data_samples[::params.USF]
    print("Symbols received:\n{}", format(symbols_received))

    # Remove the preamble symbols at the beginning
    data_symbols = symbols_received[len(preamble_symbols):]

    plot_helper.plot_complex_function(data_symbols, "y after puting the right sampling time")
    plot_helper.plot_complex_symbols(data_symbols, "Symbols received", annotate=False)

    # Decode the symbols
    ints = receiver.decoder(data_symbols, mappings.mapping)
    receiver.ints_to_message(ints)


if __name__ == "__main__":
    local_test()
