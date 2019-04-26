import numpy as np

import params
import helper


def decoder(y, mapping):
    """
    :param y: the observation vector, i.e the received symbols
    :param mapping: the chosen mapping for the communication
    :return: integers between 0 and M-1, i.e integers corresponding to the bits sent
    """

    # Make sure y and mapping have less or equal than 2 dimensions
    if len(y.shape) > 2 or len(mapping.shape) > 2:
        raise AttributeError("One of the vectors y and mapping has more than 2 dimensions!")

    # If y is a column vector, make it a row vector
    n_elems_axis_0_y = np.size(y, 0)
    if n_elems_axis_0_y != 1:
        y = y.reshape(1, n_elems_axis_0_y)
    else:
        y = y.reshape(1, np.size(y, 1))

    # If mapping is a row vector, make it a column vector
    if np.size(mapping, 0) == 1:
        mapping = mapping.reshape(np.size(mapping, 1), 1)
    else:
        mapping = mapping.reshape(np.size(mapping, 0), 1)

    if params.verbose:
        print("y: {}\n{}".format(np.shape(y), y))
        print("mapping: {} \n{}".format(np.shape(mapping), mapping))

    # Number of symbols in the mapping
    M = np.size(mapping, 0)
    # Number of symbols received
    S = np.size(y, 1)

    distances = np.transpose(abs(np.tile(y, (M, 1)) - np.tile(mapping, (1, S))))
    return np.argmin(distances, 1)


def ints_to_message(ints):
    """
    :param ints: integers between 0 and M-1, i.e integers corresponding to the bits sent
    :return: the corresponding message
    """

    # TODO make it work for any M
    # Convert the ints to BITS_PER_SYMBOL bits (2 for now because M=4)
    bits = ["{0:02b}".format(i) for i in ints]
    if params.verbose:
        print(bits)

    # Make a new string with it
    bits = ''.join(bits)
    if params.verbose:
        print(bits)

    # Slice the string into substrings of 7 characters
    bits = [bits[i:i+7] for i in range(0, len(bits), 7)]
    if params.verbose:
        print(bits)

    # Add a zero at the beginning of each substring (cf transmitter)
    new_bits = []
    for sub_string in bits:
        new_bits.append('0' + sub_string)
    if params.verbose:
        print(new_bits)

    message = ''.join(helper.bits2string(new_bits))
    print("Message received:\n{}".format(message))

    return message


if __name__ == "__main__":
    print("Receiver:")
    observation_test = np.array([1+2j, -1-0.5j, -1+0.5j, 1+0.1j, 1-2j, 1+2j, -1-0.5j])
    helper.plot_complex(observation_test, "observation", "red")
    ints = decoder(observation_test, helper.mapping)
    print(ints)
    print(ints_to_message(ints))
