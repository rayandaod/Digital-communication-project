import numpy as np

import src.params


def write_gaussian_noise(num_samples, mean, std):
    """
    Write noise in input file for testing purpose
    """
    f = open(src.params.message_sample_path, "w")
    samples = np.random.normal(mean, std, size=num_samples)
    for i in range(num_samples):
        f.write(str(samples[i]) + '\n')
    return None


def write_samples(samples):
    """
    Write samples in the input sample file
    :param samples: samples array to write in the file
    :return: None
    """
    f = open(src.params.message_sample_path, "w")
    for i in range(len(samples)):
        f.write(str(samples[i]) + '\n')
    return None
