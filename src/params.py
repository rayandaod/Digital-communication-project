import numpy as np


def choose_T():
    if MODULATION_TYPE == 1:
        return (1+BETA)/((FREQ_RANGES[0][1] - FREQ_RANGES[0][0])/2 - FREQUENCY_MARGIN)
    elif MODULATION_TYPE == 2:
        return (1+BETA)/(2*(FREQ_RANGES[0][1] - FREQ_RANGES[0][0] - FREQUENCY_MARGIN))
    else:
        raise ValueError('This modulation type does not exist yet... Hehehe')


# General variables
verbose = True
message_file_path = "../data/input_text.txt"
output_file_path = "../data/output_text.txt"

message_sample_path = "../data/input_samples.txt"
output_sample_path = "../data/output_samples.txt"

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
MODULATION_TYPE = 2  # 1 = naive approach (duplicate 4 times)
# 2 = less naive approach (duplicate 2 times, (care about covering 4000Hz with the rrc -->
# choose T accordingly))
BETA = 0.5  # rolloff factor of our root-raised-cosine pulse
T = choose_T()  # symbol period (in seconds), i.e time before we can repeat the pulse while satisfying Nyquist crit.
USF = int(np.ceil(T * Fs))  # up-sampling factor, i.e the number of zeros to add between any 2 samples before
# pulse shaping
SPAN = 20 * USF  # size of our pulse in number of samples
ABS_SAMPLE_RANGE = 0.8  # samples amplitude must be between -1 and 1, but we keep a little margin for the noise
PREAMBLE = [-1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1,  # training sequence
            -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1,
            1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1,
            -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1,
            1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, 1,
            -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1,
            1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1]

# TODO check that the length of the pn-sequence is optimal

