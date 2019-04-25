import numpy as np
from numpy import matlib

import utils


def decoder(y, mapping):
    """
    :param y: the observation vector
    :param mapping: the chosen mapping for the communication
    :return: the corresponding M-ary symbols, i.e integers between 0 and [M=length(mapping)]-1
    """
    M = len(mapping)
    L = len(y)
    distances = abs(matlib.repmat(y, M, 1) - matlib.repmat(mapping, 1, L))
    return np.argmin(distances)


if __name__ == "__main__":
    print("Receiver:")


# TODO Do not forget to put back the most significant bit (0) in the receiver
# TODO sequence of bits
