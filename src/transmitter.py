import zlib
import sys
import numpy as np
from scipy.signal import upfirdn

import helper
import params


def message_to_ints():
    """
    :return: the mapping indices corresponding to our message
    """
    # Retrieve the message from file
    message_file = open(params.message_file_path)
    message = message_file.readline()
    print("Message to be sent:\n{}".format(message))

    # Tried to compress message
    message_encoded = message.encode('ascii')
    compressed_message = zlib.compress(message_encoded)

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
        print("Original message: {}".format(message))
        print("Encoded message: {}".format(message_encoded))
        print("Size (in bytes) of encoded message: {}".format(sys.getsizeof(message_encoded)))
        print("Compressed message: {}".format(compressed_message))
        print("Size (in bytes) of compressed message {}".format(sys.getsizeof(compressed_message)))
        print("Cropped and re-structured bits: {}".format(new_bits))
        print("Equivalent integers (indices for our mapping): {}".format(ints))

    return ints


def encoder(indices, mapping):
    """
    :param indices: the mapping indices corresponding to our message
    :param mapping: the mapping corresponding to the given modulation type
    :return: the symbols/n-tuples
    """
    symbols = [mapping[i] for i in indices]

    if params.verbose:
        print("Average symbol energy: {}".format(np.mean(np.abs(symbols)**2)))
        print("Symbols/n-tuples to be sent: {}".format(symbols))
        helper.plot_complex(symbols, "{} transmitted symbols".format(len(symbols)), "blue")

    return np.asarray(symbols)


# TODO
def symbols_to_samples(symbols, h, USF):
    """
    :param symbols: the symbols modulating the pulse
    :param h: the sampled pulse
    :param USF: the up-sampling factor (number of samples per symbols)
    :return: the samples of a modulated pulse train to send to the server
    """

    # If symbols is not a column vector, make it a column vector
    if np.size(symbols, 0) == 1:
        symbols = symbols.reshape(np.size(symbols, 1), 1)
    else:
        symbols = symbols.reshape(np.size(symbols, 0), 1)

    # TODO Ask if ok to use that
    return upfirdn(h, symbols, USF)


# Intended for testing (to run the program, run main.py)
if __name__ == '__main__':
    print("Transmitter:")
    # symbols = encoder(message_to_ints(), helper.mapping)

    # TODO How do we choose the USF?
    print(symbols_to_samples(np.array([1+2j, -1-0.5j, -1+0.5j, 1+0.1j, 1-2j, 1+2j, -1-0.5j]),
                             None, 5))

# TODO Add checks everywhere on the sizes of the arrays etc
# TODO Try with a longer/shorter message
# TODO Try with different M
# TODO Add prints if verbose for debugging
# TODO Try to make it work with text compression (?). Idea : first remove the useless zero,
# TODO      then back to string, then compression
