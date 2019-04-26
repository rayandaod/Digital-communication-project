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
        print("y: {}".format(y))
        print(np.shape(y))
        print("mapping: {}".format(mapping))
        print(np.shape(mapping))

    # Number of symbols in the mapping
    M = np.size(mapping, 0)
    # Number of symbols received
    S = np.size(y, 1)

    distances = np.transpose(abs(np.tile(y, (M, 1)) - np.tile(mapping, (1, S))))
    return np.argmin(distances, 1)

def ints_to_bits():
    return None


if __name__ == "__main__":
    print("Receiver:")
    observation_test = np.array([1+2j, -1-0.5j, -1+0.5j, 1+0.1j, 1-2j])
    helper.plot_complex(observation_test, "observation", "red")
    print(decoder(observation_test, params.mapping))

# TODO Do not forget to put back the most significant bit (0) in the receiver
# TODO sequence of bits
