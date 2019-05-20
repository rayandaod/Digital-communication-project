import numpy as np

import params

f = open(params.input_sample_file_path, "w")


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
    f.close()
    return None


def write_samples(samples, preamble=False):
    """
    Write samples in the input sample file
    :param samples: samples array to write in the file
    :param preamble: boolean to decide if we write in the preamble_samples file
    :return: None
    """
    if preamble:
        preamble_sample_file = open(params.preamble_sample_file_path, "w")
        for i in range(len(samples)):
            preamble_sample_file.write(str(samples[i]) + '\n')
        preamble_sample_file.close()
    else:
        for i in range(len(samples)):
            f.write(str(samples[i]) + '\n')
        f.close()
    return None


def write_sinus(duration, freqs, scaling_factor=1.):
    """
    Write a sinus in the input sample file, at the given frequency
    :param scaling_factor:
    :param duration: duration of the sinus (in seconds)
    :param freqs: array of frequencies for the sum of sinus
    :return: None
    """
    n_samples = int(np.ceil(duration*params.Fs))
    t = np.arange(n_samples)/params.Fs
    samples = np.zeros(n_samples)
    for freq in freqs:
        samples += np.sin(freq * 2 * np.pi * t)
    for i in range(n_samples):
        f.write(str(samples[i]*scaling_factor) + '\n')
    f.close()
    return None


if __name__ == "__main__":
    # write_samples(input_samples)
    write_gaussian_noise(1, mean=0, std=1/4)
    # write_sinus(1, [2000, 4000, 6000, 8000], scaling_factor=1/8)
