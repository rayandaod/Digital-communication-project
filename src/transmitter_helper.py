import numpy as np
from scipy.signal import upfirdn

import params
import read_write
import preambles
import plot_helper
import fourier_helper


def generate_preamble_to_transmit(len_data_symbols):
    if params.logs:
        print("Generating the preamble...")

    preambles.generate_preamble_symbols(len_data_symbols)
    preamble_symbols = read_write.read_preamble_symbols()

    if params.plots:
        plot_helper.plot_complex_symbols(preamble_symbols, "Preamble symbols")
    if params.logs:
        print("Preamble symbols:\n{}".format(preamble_symbols))
        print("--------------------------------------------------------")
    return preamble_symbols


def concatenate_symbols(preamble_symbols, data_symbols):
    if params.logs:
        print("Concatenating everything together (preamble-data-flipped preamble)...")

    if params.MODULATION_TYPE == 1 or params.MODULATION_TYPE == 2:
        p_data_p_symbols = np.concatenate((preamble_symbols, data_symbols, preamble_symbols[::-1]))
        if params.logs:
            print("Total symbols: {}".format(p_data_p_symbols))
            print("Number of total symbols: {}".format(np.shape(p_data_p_symbols)))
    elif params.MODULATION_TYPE == 3:
        p_data_p_symbols = []
        for i in range(len(data_symbols)):
            p_data_p_symbols.append(np.concatenate((preamble_symbols, data_symbols[i], preamble_symbols[::-1])))
        if params.logs:
            for i in range(len(p_data_p_symbols)):
                print("Total symbols {}: {}".format(i, p_data_p_symbols))
                print("Number of total symbols {}: {}".format(i, np.shape(p_data_p_symbols)))
                if params.plots:
                    plot_helper.plot_complex_symbols(p_data_p_symbols[i], "Symbols {}".format(i))
    else:
        raise ValueError("This mapping type does not exist yet... He he he")
    return p_data_p_symbols


def shape_symbols(h, p_data_p_symbols, USF):
    if params.logs:
        print("Pulse shaping the symbols...")

    if params.MODULATION_TYPE == 1 or params.MODULATION_TYPE == 2:
        p_data_p_samples = upfirdn(h, p_data_p_symbols, USF)
        if params.logs:
            print("Samples: {}".format(p_data_p_samples))
            print("Up-sampling factor: {}".format(params.USF))
            print("Number of samples: {}".format(len(p_data_p_samples)))
        if params.plots:
            plot_helper.samples_fft_plots(p_data_p_samples, "Samples after the pulse shaping", shift=True)
    elif params.MODULATION_TYPE == 3:
        p_data_p_samples = []
        for i in range(len(p_data_p_symbols)):
            p_data_p_samples.append(upfirdn(h, p_data_p_symbols[i], USF))
        if params.plots:
            for i in range(len(p_data_p_samples)):
                plot_helper.samples_fft_plots(p_data_p_samples[i], "Samples {} after the pulse shaping".format(i),
                                              shift=True)
    else:
        raise ValueError("This mapping type does not exist yet... He he he")

    if params.logs:
        print("--------------------------------------------------------")
    return p_data_p_samples


def shape_preamble_samples(h, preamble_symbols, USF):
    if params.logs:
        print("Shaping the preamble...")

    preamble_samples = upfirdn(h, preamble_symbols, USF)
    read_write.write_preamble_samples(preamble_samples)

    if params.logs:
        print("Number of samples for the preamble: {}".format(len(preamble_samples)))
    if params.plots:
        plot_helper.samples_fft_plots(preamble_samples, "Preamble samples", shift=True)
    if params.logs:
        print("--------------------------------------------------------")
    return None


def modulate_samples(p_data_p_samples):
    if params.logs:
        print("Choosing the modulation frequencies and modulating the samples...")
    if params.MODULATION_TYPE == 1 or params.MODULATION_TYPE == 3:
        modulating_frequencies = params.np.mean(params.FREQ_RANGES, axis=1)
    elif params.MODULATION_TYPE == 2:
        modulating_frequencies = [params.FREQ_RANGES[0][1], params.FREQ_RANGES[2][1]]
    else:
        raise ValueError("This mapping type does not exist yet... He he he")

    # Modulate the samples to fit in the required bands
    if np.any(np.iscomplex(p_data_p_samples)):
        if params.MODULATION_TYPE == 1 or params.MODULATION_TYPE == 2:
            p_data_p_samples = [item for sublist in p_data_p_samples for item in sublist]
            p_data_p_modulated_samples = fourier_helper.modulate_complex_samples(p_data_p_samples,
                                                                                 modulating_frequencies)
            if params.logs:
                print("Min and max sample after modulation: ({}, {})".format(min(p_data_p_samples),
                                                                             max(p_data_p_samples)))
            if params.plots:
                plot_helper.samples_fft_plots(p_data_p_samples, "Samples to send", time=True, complex=True, shift=True)
        elif params.MODULATION_TYPE == 3:
            modulated_samples = []
            for i in range(len(p_data_p_samples)):
                modulated_samples.append(fourier_helper.modulate_complex_samples(p_data_p_samples[i],
                                                                                 [modulating_frequencies[i]]))
            p_data_p_modulated_samples = np.sum(modulated_samples, axis=0).flatten()
        else:
            raise ValueError("This mapping type does not exist yet... He he he")
    else:
        raise ValueError("TODO: handle real samples (e.g SSB)")
    if params.logs:
        print("--------------------------------------------------------")
    return p_data_p_modulated_samples


def scale_samples(p_data_p_modulated_samples):
    if params.logs:
        print("Scaling the samples to the server constraints...")
    samples_to_send = p_data_p_modulated_samples / (np.max(np.abs(p_data_p_modulated_samples))) * params.\
        ABS_SAMPLE_RANGE

    if params.logs:
        print("Scaling the signal...")
        print("Number of samples: {}".format(len(samples_to_send)))
        print("Minimum sample after scaling: {}".format(min(samples_to_send)))
        print("Maximum sample after scaling: {}".format(max(samples_to_send)))
        print("--------------------------------------------------------")
    return samples_to_send
