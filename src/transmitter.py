import numpy as np
from scipy.signal import upfirdn

import helper
import params
import read_write
import plot_helper
import pulses
import fourier_helper
import mappings
import preambles


def message_to_ints():
    """
    :return: the mapping indices corresponding to our message
    """
    # Retrieve the message from file
    message_file = open(params.message_file_path)
    message = message_file.readline()
    print("Sent message:\n{}".format(message))
    print("Length: {} characters".format(len(message)))

    # TODO Tried to compress message
    # compressed_message = zlib.compress(message_encoded)
    message_encoded = message.encode('ascii')

    # Retrieve the message as a sequences of binary bytes
    string_bytes = helper.string2bits(message)

    # Next step is to re-arrange string_bytes in agreement with M. Indeed, with a symbol constellation of M points,
    # we can only represent BITS_PER_SYMBOL=log2(M) bits per symbol. Thus, we want to re-structure string_bytes
    # with BITS_PER_SYMBOL=log2(M) bits by row.

    # Remove the most significant bit (0) as it is useless in ASCII (do not forget to put it again in the receiver!)
    new_bits = [b[1:] for b in string_bytes]

    # Make a new string with these cropped bytes
    new_bits = ''.join(new_bits)

    # New structure with bits_per_symbol bits by row
    new_bits = [new_bits[i:i + params.BITS_PER_SYMBOL] for i in range(0, len(new_bits), params.BITS_PER_SYMBOL)]

    # Convert this new bits sequence to an integer sequence
    ints = [int(b, 2) for b in new_bits]

    if params.verbose:
        print("Encoded message:\n{}".format(message_encoded))
        print("Corresponding bytes:\n{}".format(string_bytes))
        # print("Size (in bytes) of encoded message:\n{}".format(sys.getsizeof(message_encoded)))
        # print("Compressed message: {}".format(compressed_message))
        # print("Size (in bytes) of compressed message:\n{}".format(sys.getsizeof(compressed_message)))
        print("Cropped and re-structured bits:\n{}".format(new_bits))
        print("Equivalent integers (indices for our mapping):\n{}".format(ints))
        print("--------------------------------------------------------")

    return ints


def encoder(indices, mapping):
    """
    :param indices: the mapping indices corresponding to our message
    :param mapping: the mapping corresponding to the given modulation type
    :return: the symbols/n-tuples
    """
    corresponding_symbols = [mapping[i] for i in indices]

    if params.verbose:
        print("Symbols/n-tuples to be sent:\n{}".format(corresponding_symbols))
        print("Average symbol energy: {}".format(np.mean(np.abs(corresponding_symbols)**2)))
        print("Number of symbols: {}".format(len(corresponding_symbols)))
        print("Minimum symbol: {}".format(min(corresponding_symbols)))
        print("Maximum symbol: {}".format(max(corresponding_symbols)))
        print("--------------------------------------------------------")
        plot_helper.plot_complex_symbols(corresponding_symbols, "{} data symbols to send"
                                         .format(len(corresponding_symbols)), "blue")

    return np.asarray(corresponding_symbols)


def symbols_to_samples(h, data_symbols, USF=params.USF):
    """
    :param h: the sampled pulse
    :param data_symbols: the symbols modulating the pulse
    :param USF: the up-sampling factor (number of samples per symbols)
    :return: the samples of a modulated pulse train to send to the server
    """
    #
    # # If symbols is not a column vector, make it one
    # if np.size(symbols, 0) == 1:
    #     symbols = symbols.reshape(np.size(symbols, 1), 1)
    # else:
    #     symbols = symbols.reshape(np.size(symbols, 0), 1)

    # Generate the preamble_symbols and write them in the appropriate file
    preambles.generate_preamble_symbols(len(data_symbols))
    preamble_symbols = read_write.read_preamble_symbols()
    if params.verbose:
        print("Preamble symbols:\n{}".format(preamble_symbols))
        print("--------------------------------------------------------")
    plot_helper.plot_complex_symbols(preamble_symbols, "Preamble symbols")

    # Concatenate the synchronization sequence with the symbols to send
    total_symbols = np.concatenate((preamble_symbols, data_symbols))
    plot_helper.plot_complex_symbols(total_symbols, "Total symbols to send")

    # TODO can/should I remove the ramp-up and ramp_down? (then less samples to send)
    # Shape the signal with the pulse h
    total_samples = upfirdn(h, total_symbols, USF)

    if params.verbose:
        print("Shaping the preamble and the data...")
        print("Samples to be sent:\n{}".format(total_samples))
        print("Up-sampling factor: {}".format(params.USF))
        print("Number of samples: {}".format(len(total_samples)))
        print("Minimum sample: {}".format(min(total_samples)))
        print("Maximum sample: {}".format(max(total_samples)))
        print("--------------------------------------------------------")
        plot_helper.plot_complex_function(total_samples, "Input samples in Time domain")
        plot_helper.fft_plot(total_samples, "Input samples in Frequency domain", shift=True)

    # Write the preamble samples (base-band, so might be complex) in the preamble_samples file
    preamble_samples = upfirdn(h, preamble_symbols, USF)
    read_write.write_preamble_samples(preamble_samples)

    if params.verbose:
        print("Shaping the preamble...")
        print("Number of samples for the preamble: {}".format(len(preamble_samples)))
        plot_helper.plot_complex_function(preamble_samples, "Synchronization sequence shaped, in Time domain")
        plot_helper.fft_plot(preamble_samples, "Synchronization sequence shaped, in Frequency domain", shift=True)
        print("--------------------------------------------------------")

    # Modulate the samples to fit in the required bands
    if np.any(np.iscomplex(total_samples)):
        if params.MODULATION_TYPE == 1:
            total_samples = fourier_helper.modulate_complex_samples(total_samples,
                                                                    params.np.mean(params.FREQ_RANGES, axis=1))
        elif params.MODULATION_TYPE == 2:
            total_samples = fourier_helper.modulate_complex_samples(total_samples, [params.FREQ_RANGES[0][1],
                                                                                    params.FREQ_RANGES[2][1]])
        else:
            raise ValueError('This modulation type does not exist yet... Hehehe')

        if params.verbose:
            print("Modulation of the signal...")
            print("Number of samples: {}".format(len(total_samples)))
            print("Minimum sample after modulation: {}".format(min(total_samples)))
            print("Maximum sample after modulation: {}".format(max(total_samples)))
            print("--------------------------------------------------------")
            plot_helper.plot_complex_function(total_samples, "Input samples after modulation, in Time domain")
            plot_helper.fft_plot(total_samples, "Input samples after modulation, in Frequency domain", shift=True)
    else:
        raise ValueError("TODO: handle real samples (e.g SSB)")

    # Scale the signal to the range [-1, 1] (with a bit of uncertainty margin, according to params.ABS_SAMPLE_RANGE)
    total_samples = (total_samples/(max(total_samples))*params.ABS_SAMPLE_RANGE)

    if params.verbose:
        print("Scaling the signal...")
        print("Minimum sample after scaling: {}".format(min(total_samples)))
        print("Maximum sample after scaling: {}".format(max(total_samples)))
        print("--------------------------------------------------------")

    return total_samples


# Intended for testing (to run the program, run main.py)
if __name__ == '__main__':
    # Encode the message
    symbols = encoder(message_to_ints(), mappings.mapping)

    # Generate the root-raised_cosine
    _, h_pulse = pulses.root_raised_cosine()

    # Construct the samples to send
    input_samples = symbols_to_samples(h_pulse, symbols)

    # Write the samples in the input file
    read_write.write_samples(input_samples)

# TODO Add checks everywhere on the sizes of the arrays etc
# TODO Try with a longer/shorter message
# TODO Try with different M
# TODO Add prints if verbose for debugging
# TODO Try to make it work with text compression (?). Idea : first remove the useless zero,
# TODO      then back to string, then compression
