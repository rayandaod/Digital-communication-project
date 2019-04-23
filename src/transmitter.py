import zlib
import sys
import numpy as np
import matplotlib.pyplot as plt

import mappings

verbose = True

message_file_name = "../data/input_lorem_ipsum.txt"

M = 4
bits_per_symbol = int(np.log2(M))
modulation_type = "qam"


def string2bits(s=''):
    return [bin(ord(x))[2:].zfill(8) for x in s]


# Outputs the mapping indices corresponding to our message
def mapping_indices():
    # Retrieve the message from file
    message_file = open(message_file_name)
    message = message_file.readline()

    # Tried to compress message
    message_encoded = message.encode('ascii')
    compressed_message = zlib.compress(message_encoded)

    # Retrieve the message as a sequences of binary bytes
    string_bytes = string2bits(message)

    # Next step is to re-arrange string_bytes in agreement with M. Indeed, with a symbol constellation of M points,
    # we can only represent log2(M) bits per symbol. Thus, we want to re-structure string_bytes with log2(M)
    # bits by row.

    # Remove the most significant bit (0) as it is useless in ASCII (do not forget to put it again in the receiver!)
    new_bits = [b[1:] for b in string_bytes]
    # Make a new string with these cropped bytes
    new_bits = ''.join(new_bits)
    # New structure with bits_per_symbol bits by row
    new_bits = [new_bits[i:i + bits_per_symbol] for i in range(0, len(new_bits), bits_per_symbol)]
    # Convert this new bits sequence to an integer sequence
    ints = [int(b, 2) for b in new_bits]

    if verbose:
        print("Original message: {}".format(message))
        print("Encoded message: {}".format(message_encoded))
        print("Size (in bytes) of encoded message: {}".format(sys.getsizeof(message_encoded)))
        print("Compressed message: {}".format(compressed_message))
        print("Size (in bytes) of compressed message {}".format(sys.getsizeof(compressed_message)))
        print("Cropped and re-structured bits: {}".format(new_bits))
        print("Equivalent integers (indices for our mapping): {}".format(ints))

    return ints


# Chooses the mapping according to the given modulation_type
def choose_mapping():
    if modulation_type == "qam":
        return mappings.qam_map(M)
    elif modulation_type == "psk":
        return mappings.psk_map(M)
    else:
        raise ValueError('No modulation of this type was found')


# Forms our n-tuples
def encoder(indices, mapping):
    symbols = [mapping[i] for i in indices]

    if verbose:
        print("Average symbol energy: {}".format(np.mean(np.abs(symbols)**2)))
        X = [x.real for x in symbols]
        Y = [x.imag for x in symbols]
        plt.scatter(X, Y, color='red')
        plt.legend(['Symbols'])
        plt.title("{} transmitted symbols".format(len(symbols)))
        plt.xlabel("Re")
        plt.ylabel("Im")
        plt.grid()
        plt.show()

    return symbols


if __name__ == '__main__':
    mapping = choose_mapping()
    mapping = mapping/np.sqrt(np.mean(np.abs(mapping)**2))

    indices = mapping_indices()
    symbols = encoder(indices, mapping)

# TODO Add checks everywhere on the sizes of the arrays etc
# TODO Try with a longer/shorter message
# TODO Try with different M
# TODO Add prints if verbose for debugging
