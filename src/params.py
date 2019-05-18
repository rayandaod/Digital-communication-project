import numpy as np


def choose_symbol_period():
    if MODULATION_TYPE == 1:
        return (1+BETA)/1900
    elif MODULATION_TYPE == 2:
        return (1+BETA)/3900
    else:
        raise ValueError('This modulation type does not exist yet... Hehehe')


# General variables
verbose = True
message_file_path = "../data/input_text.txt"
output_file_path = "../data/output_text.txt"

input_sample_file_path = "../data/input_samples.txt"
output_sample_file_path = "../data/output_samples.txt"
preamble_sample_file_path = "../data/preamble_samples.txt"

server_hostname = "iscsrv72.epfl.ch"
server_port = 80

# /!\ DO NOT CHANGE THE FOLLOWING VARIABLES /!\
NOISE_VAR = 0.05  # channel noise variance
Fs = 22050  # sampling frequency / sampling rate / samples per second (in Hz)
FREQ_RANGES = [[1000, 3000],  # frequency ranges authorized by the channel
               [3000, 5000],  # one of them (at random) is set to 0 by the channel
               [5000, 7000],
               [7000, 9000]]
# /!\ -------------------------------------- /!\
FREQUENCY_MARGIN = 50  # Hz - You can tweak this one but be careful

# Communication parameters (you should only tweak the first 5 ones for this project)
M = 16  # length of the mapping (must be of the form 2^2k if QAM is chosen)
MAPPING = "qam"  # mapping: qam or psk or pam for now
BITS_PER_SYMBOL = int(np.log2(M))  # number of bits we transmit per symbol
MODULATION_TYPE = 1  # 1 = naive approach (duplicate 4 times)
# 2 = less naive approach (duplicate 2 times, (care about covering 4000Hz with the rrc --> choose T accordingly))

BETA = 0.5  # rolloff factor of our root-raised-cosine pulse
T = choose_symbol_period()  # symbol period (in seconds), i.e time before we can repeat the pulse while satisfying
# Nyquist crit.

USF = int(np.ceil(T * Fs))  # up-sampling factor, i.e the number of zeros to add between any 2 symbols before
# pulse shaping
SPAN = 20 * USF  # size of our pulse in number of samples

ABS_SAMPLE_RANGE = 0.8  # samples amplitude must be between -1 and 1, but we keep a little margin for the noise

# TODO Check that this ratio is right (test with other ratios)
PREAMBLE_LENGTH_RATIO = 0.15  # Ratio of synchronization symbol sequence compared to the number of symbols to send
