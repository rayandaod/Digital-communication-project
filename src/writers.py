import numpy as np

import params

f = open(params.message_sample_path, "w")


def write_gaussian_noise(duration, mean, std):
    """
    Write a gaussian noise with the given parameters in the input file
    :param duration: duration of the noise (in seconds)
    :param mean: mean of the gaussian noise
    :param std: standard deviation of thr gaussian noise
    :return: None
    """
    n_samples = duration*params.Fs
    samples = np.random.normal(mean, std, n_samples)
    for i in range(n_samples):
        f.write(str(samples[i]) + '\n')
    return None


def write_samples(samples):
    """
    Write samples in the input sample file
    :param samples: samples array to write in the file
    :return: None
    """
    for i in range(len(samples)):
        f.write(str(samples[i]) + '\n')
    return None


def write_sinus(duration, freq, scaling_factor=1):
    """
    Write a sinus in the input sample file, at the given frequency
    :param scaling_factor:
    :param duration: duration of the sinus (in seconds)
    :param freq: given frequency for the sinus
    :return: None
    """
    n_samples = params.Fs * duration
    t = np.linspace(0, duration, n_samples)
    samples = np.sin(freq * 2 * np.pi * t)
    for i in range(n_samples):
        f.write(str(samples[i]*scaling_factor) + '\n')
    return None
