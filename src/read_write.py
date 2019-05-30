import numpy as np

import params

"""
Reading methods
"""


def read_preamble_samples():
    """
    Read the samples from the preamble sample file

    :return: The preamble samples
    """
    preamble_samples_file = open(params.preamble_sample_file_path, "r")
    preamble_samples = np.asarray([complex(line) for line in preamble_samples_file.readlines()])
    preamble_samples_file.close()
    return preamble_samples


def read_preamble_symbols():
    """
    Read the symbols from the preamble symbols file

    :return: The preamble symbols
    """
    preamble_symbol_file = open(params.preamble_symbol_file_path, "r")
    preamble_symbols = [complex(line) for line in preamble_symbol_file.readlines()]
    preamble_symbol_file.close()
    return preamble_symbols


def read_message_sent():
    """
    Read the message that was initially sent

    :return: The sent message
    """
    input_message_file = open(params.input_message_file_path)
    message_sent = input_message_file.readline()
    input_message_file.close()
    return message_sent


"""
Writing methods
"""


def write_samples(samples):
    """
    Write samples in the input sample file

    :param samples: The samples array to write in the file
    :return:        None
    """
    f = open(params.input_sample_file_path, "w")
    for i in range(len(samples)):
        f.write(str(samples[i]) + '\n')
    f.close()
    return None


def write_preamble_symbols(preamble_symbols):
    """
    Write preamble samples in the preamble sample file

    :param preamble_symbols:    The preamble samples array to write in the file
    :return:                    None
    """
    preamble_symbol_file = open(params.preamble_symbol_file_path, "w")
    for i in range(len(preamble_symbols)):
        preamble_symbol_file.write(str(preamble_symbols[i]) + '\n')
    preamble_symbol_file.close()


def write_preamble_samples(preamble_samples):
    """
    Write preamble samples in the preamble sample file

    :param preamble_samples:    The preamble samples array to write in the file
    :return:                    None
    """
    preamble_sample_file = open(params.preamble_sample_file_path, "w")
    for i in range(len(preamble_samples)):
        preamble_sample_file.write(str(preamble_samples[i]) + '\n')
    preamble_sample_file.close()


def write_message_received(message):
    """
    Write the message received in the appropriate file

    :param message: The message to be stored
    :return:        None
    """
    output_message_file = open(params.output_message_file_path, "w")
    output_message_file.write(message)
    output_message_file.close()
    return None


def write_gaussian_noise(duration, mean, std):
    """
    Write a gaussian noise with the given parameters in the input file

    :param duration:    The duration of the noise (in seconds)
    :param mean:        The mean of the gaussian noise
    :param std:         The standard deviation of the gaussian noise
    :return:            None
    """
    f = open(params.input_sample_file_path, "w")
    n_samples = duration * params.Fs
    samples = np.random.normal(mean, std, n_samples)
    for i in range(n_samples):
        f.write(str(samples[i]) + '\n')
    f.close()
    return None


def write_sinus(duration, frequencies, scaling_factor=1.):
    """
    Write a sinus in the input sample file, at the given frequency

    :param scaling_factor:  The scaling factor wo multiply the sinus with
    :param duration:        The duration of the sinus (in seconds)
    :param frequencies:     The array of frequencies for the sum of sinus
    :return:                None
    """
    f = open(params.input_sample_file_path, "w")
    n_samples = int(np.ceil(duration * params.Fs))
    t = np.arange(n_samples) / params.Fs
    samples = np.zeros(n_samples)
    for freq in frequencies:
        samples += np.sin(freq * 2 * np.pi * t)
    for i in range(n_samples):
        f.write(str(samples[i] * scaling_factor) + '\n')
    f.close()
    return None
