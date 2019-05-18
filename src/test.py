import pulses
from scipy.signal import upfirdn
import enc_dec_helper
import numpy as np
import plot_helper
import fourier_helper


"""
Testing file
"""
if __name__ == "__main__":
    synchronization_sequence = np.random.choice(enc_dec_helper.mapping, size=4)
    _, h = pulses.root_raised_cosine()
    samples = upfirdn(h, synchronization_sequence)
    plot_helper.plot_complex_function(samples, "Synchronization sequence shaped")
    samples = fourier_helper.modulate(samples, [2000])
    plot_helper.plot_complex_function(samples, "Synchronization sequence modulated")
    plot_helper.fft_plot(samples, "Synchronization sequence fft after modulation")
