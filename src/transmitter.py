import zlib
import sys
import numpy as np
from scipy.signal import upfirdn

import enc_dec_helper
import params
import writers
import plot_helper
import pulses
import fourier_helper
import synchronization


def message_to_ints():
    """
    :return: the mapping indices corresponding to our message
    """
    # Retrieve the message from file
    message_file = open(params.message_file_path)
    message = message_file.readline()
    print("Sent message:\n{}".format(message))

    # Tried to compress message
    message_encoded = message.encode('ascii')
    compressed_message = zlib.compress(message_encoded)

    # Retrieve the message as a sequences of binary bytes
    string_bytes = enc_dec_helper.string2bits(message)

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
        print("Size (in bytes) of encoded message:\n{}".format(sys.getsizeof(message_encoded)))
        print("Compressed message: {}".format(compressed_message))
        print("Size (in bytes) of compressed message:\n{}".format(sys.getsizeof(compressed_message)))
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
    symbols = [mapping[i] for i in indices]

    if params.verbose:
        print("Symbols/n-tuples to be sent:\n{}".format(symbols))
        print("Average symbol energy: {}".format(np.mean(np.abs(symbols)**2)))
        print("Number of symbols: {}".format(len(symbols)))
        print("Minimum symbol: {}".format(min(symbols)))
        print("Maximum symbol: {}".format(max(symbols)))
        print("--------------------------------------------------------")
        plot_helper.plot_complex_symbols(symbols, "{} transmitted symbols".format(len(symbols)), "blue")

    return np.asarray(symbols)


def symbols_to_samples(h, symbols_to_send, USF=params.USF):
    """
    :param h: the sampled pulse
    :param symbols_to_send: the symbols modulating the pulse
    :param USF: the up-sampling factor (number of samples per symbols)
    :return: the samples of a modulated pulse train to send to the server
    """
    #
    # # If symbols is not a column vector, make it a column vector
    # if np.size(symbols, 0) == 1:
    #     symbols = symbols.reshape(np.size(symbols, 1), 1)
    # else:
    #     symbols = symbols.reshape(np.size(symbols, 0), 1)

    # Insert the synchronization sequence
    synchronization.PREAMBLE = np.random.choice(enc_dec_helper.mapping,
                                                size=int(np.ceil(len(symbols_to_send) * params.PREAMBLE_LENGTH_RATIO)))
    symbols_to_send = np.concatenate((synchronization.PREAMBLE, symbols_to_send))
    if params.verbose:
        print("Synchronization sequence:\n{}".format(synchronization.PREAMBLE))
        print("--------------------------------------------------------")

    # TODO can/should I remove the ramp-up and ramp_down?
    # Shape the signal with the pulse h
    samples = upfirdn(h, symbols_to_send, USF)
    maximum = max(samples)

    if params.verbose:
        print("Shaping the preamble and the data...")
        print("Samples to be sent:\n{}".format(samples))
        print("Up-sampling factor: {}".format(params.USF))
        print("Number of samples: {}".format(len(samples)))
        print("Minimum sample: {}".format(min(samples)))
        print("Maximum sample: {}".format(maximum))
        print("--------------------------------------------------------")
        plot_helper.plot_complex_function(samples, "Input samples")

    synchronization.preamble_shaped = upfirdn(h, synchronization.PREAMBLE, USF)
    if params.verbose:
        print("Shaping the preamble...")
        print("Synchronization sequence shaped:\n{}".format(synchronization.preamble_shaped))
        print("Number of samples for the preamble: {}".format(len(synchronization.preamble_shaped)))
        plot_helper.plot_complex_function(synchronization.preamble_shaped, "Synchronization sequence shaped")
        print("--------------------------------------------------------")

    if np.any(np.iscomplex(samples)):
        if params.MODULATION_TYPE == 1:
            samples = fourier_helper.modulate(samples, params.np.mean(params.FREQ_RANGES, axis=1))
        elif params.MODULATION_TYPE == 2:
            samples = fourier_helper.modulate(samples, [params.FREQ_RANGES[0][1], params.FREQ_RANGES[2][1]])
        else:
            raise ValueError('This modulation type does not exist yet... Hehehe')

        maximum = max(samples)

        if params.verbose:
            print("Modulation of the signal...")
            print("Number of samples: {}".format(len(samples)))
            print("Minimum sample after modulation: {}".format(min(samples)))
            print("Maximum sample after modulation: {}".format(maximum))
            print("--------------------------------------------------------")
            plot_helper.plot_complex_function(samples, "Input samples after modulation")

    # Scale the signal to the range [-1, 1] (with a bit of uncertainty margin, according to params.ABS_SAMPLE_RANGE)
    samples = samples/(maximum*(2-params.ABS_SAMPLE_RANGE))
    maximum = max(samples)

    if params.verbose:
        print("Scaling the signal...")
        print("Minimum sample after scaling: {}".format(min(samples)))
        print("Maximum sample after scaling: {}".format(maximum))
        print("--------------------------------------------------------")

    return maximum, samples


# Intended for testing (to run the program, run main.py)
if __name__ == '__main__':
    # Encode the message
    symbols = encoder(message_to_ints(), enc_dec_helper.mapping)

    # Generate the root-raised_cosine
    _, h_pulse = pulses.root_raised_cosine()

    # Construct the signal to send
    maximum, input_samples = symbols_to_samples(h_pulse, symbols)

    # Write the samples in the input file
    writers.write_samples(input_samples)

# TODO Add checks everywhere on the sizes of the arrays etc
# TODO Try with a longer/shorter message
# TODO Try with different M
# TODO Add prints if verbose for debugging
# TODO Try to make it work with text compression (?). Idea : first remove the useless zero,
# TODO      then back to string, then compression
