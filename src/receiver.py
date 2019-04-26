import numpy as np

import params
import helper


def decoder(y, mapping):
    """
    :param y: the observation vector
    :param mapping: the chosen mapping for the communication
    :return: the corresponding M-ary symbols, i.e integers between 0 and [M=length(mapping)]-1
    """
    M = len(mapping)
    L = len(y)

    repmat1 = np.tile(y, (M, 1))
    repmat2 = np.tile(mapping, (1, L))

    if params.verbose:
        print(y)
        print(mapping)
        print(M)
        print(L)
        print(repmat1)
        print(repmat2)

    distances = abs(repmat1 - repmat2)
    return np.argmin(distances)


if __name__ == "__main__":
    print("Receiver:")
    observation_test = [1+2j, -1-0.5j, -1+0.5j, 1+0.1j, 1-2j]
    helper.plot_complex(observation_test, "observation", "red")
    print(decoder(observation_test, params.mapping))

# TODO Do not forget to put back the most significant bit (0) in the receiver
# TODO sequence of bits
