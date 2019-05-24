import numpy as np


def choose_symbol_period():
    if MODULATION_TYPE == 1:
        return np.floor(((1+BETA)/1900)*Fs)/Fs
    elif MODULATION_TYPE == 2:
        return np.round(((1+BETA)/3900)*Fs)/Fs
    else:
        raise ValueError('This modulation type does not exist yet... Hehehe')


# General variables
verbose = True
message_file_path = "../data/input_text.txt"
output_file_path = "../data/output_text.txt"

input_sample_file_path = "../data/input_samples.txt"
output_sample_file_path = "../data/output_samples.txt"
preamble_symbol_file_path = "../data/preamble_symbols.txt"
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

# Communication parameters (you should only tweak the first 5 ones for this project)
MAPPING = "qam"  # mapping: qam or psk or pam for now
NORMALIZE_MAPPING = False  # rather we normalize the mapping or not
M = 16  # length of the mapping (must be of the form 2^2k if QAM is chosen)
BITS_PER_SYMBOL = int(np.log2(M))  # number of bits we transmit per symbol

MODULATION_TYPE = 1
# 1 = naive approach (duplicate 4 times)
# 2 = less naive approach (duplicate 2 times, (care about covering 4000Hz with the rrc --> choose T accordingly))

BETA = 0.22  # rolloff factor of our root-raised-cosine pulse (usually between 0.2 and 0.3 (said Prandoni))
T = choose_symbol_period()  # symbol period (in seconds), i.e time before we can repeat the pulse while satisfying
# Nyquist crit.
NORMALIZE_PULSE = True  # rather we normalize the pulse or not

USF = int(np.ceil(T * Fs))  # up-sampling factor, i.e the number of zeros to add between any 2 symbols before
# pulse shaping
SPAN = 20 * USF  # size of our pulse in number of samples

PREAMBLE_TYPE = "barker"  # Type of preamble to generate (barker or random for now)
PREAMBLE_LENGTH_RATIO = 0.36  # Ratio of random preamble symbols compared to the number of symbols of the data

ABS_SAMPLE_RANGE = 0.8  # samples amplitude must be between -1 and 1, but we keep a little margin for the noise

# TODO test with different values for all parameters
