import pulses
from scipy.signal import upfirdn
import numpy as np
import plot_helper
import mappings
import synchronization
import params
import transmitter
import receiver


"""
Testing file
"""
if __name__ == "__main__":
    print(params.T*params.Fs)
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

    data_samples = y[delay + len(h)-1:len(y)-len(h)+1]
    plot_helper.plot_complex_function(data_samples, "y after puting the right sampling time")

    symbols_received = data_samples[::params.USF]
    print("Symbols received:\n{}",format(symbols_received))

    data_symbols = symbols_received[len(preamble):]

    plot_helper.plot_complex_function(data_symbols, "y after puting the right sampling time")
    plot_helper.plot_complex_symbols(data_symbols, "Symbols received", annotate=False)

    ints = receiver.decoder(data_symbols, mappings.mapping)
    receiver.ints_to_message(ints)
