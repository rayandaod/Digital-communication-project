import numpy as np


def choose_symbol_period():
    if MODULATION_TYPE == 1:
        return np.floor(((1 + BETA) / MODULATION_TYPE_1_BANDWIDTH) * Fs) / Fs
    elif MODULATION_TYPE == 2:
        return np.round(((1 + BETA) / MODULATION_TYPE_2_BANDWIDTH) * Fs) / Fs
    else:
        raise ValueError('This modulation type does not exist yet... He he he')


# General variables
logs = True
plots = False
input_message_file_path = "../data/input_text.txt"
output_message_file_path = "../data/output_text.txt"

input_sample_file_path = "../data/input_samples.txt"
output_sample_file_path = "../data/output_samples.txt"
preamble_symbol_file_path = "../data/preamble_symbols.txt"
preamble_sample_file_path = "../data/preamble_samples.txt"

server_hostname = "iscsrv72.epfl.ch"
server_port = 80

# ---------------------------------------------
# /!\ DO NOT CHANGE THE FOLLOWING VARIABLES /!\
# ---------------------------------------------
NOISE_VAR = 0.0025  # channel noise variance
Fs = 22050  # sampling frequency / sampling rate / samples per second (in Hz)
FREQ_RANGES = [[1000, 3000],  # frequency ranges authorized by the channel
               [3000, 5000],  # one of them (at random) is set to 0 by the channel
               [5000, 7000],
               [7000, 9000]]
# ---------------------------------------------
# /!\ -------------------------------------- /!\
# ---------------------------------------------

# Communication parameters
MAPPING = "qam"  # mapping: qam or psk or pam for now
NORMALIZE_MAPPING = False  # rather we normalize the mapping or not
M = 4  # length of the mapping (must be of the form 2^2k if QAM is chosen)
BITS_PER_SYMBOL = int(np.log2(M))  # number of bits we transmit per symbol

MODULATION_TYPE = 1
# 1 = naive approach (duplicate 4 times)
# 2 = less naive approach (duplicate 2 times, (care about covering 4000Hz with the rrc --> choose T accordingly))
MODULATION_TYPE_1_BANDWIDTH = 2000
MODULATION_TYPE_2_BANDWIDTH = 4000

BETA = 0.2  # rolloff factor of our root-raised-cosine pulse (usually between 0.2 and 0.3 (said Prandoni))
T = choose_symbol_period()  # symbol period (in seconds), i.e time before we can repeat the pulse while satisfying
# Nyquist crit.
NORMALIZE_PULSE = True  # rather we normalize the pulse or not

USF = int(np.ceil(T * Fs))  # up-sampling factor, i.e the number of zeros to add between any 2 symbols before
# pulse shaping
SPAN = 4 * USF  # size of our pulse in number of samples

PREAMBLE_TYPE = "barker"  # Type of preamble to generate (barker or random for now)
BARKER_SEQUENCE_REPETITION = 1  # Number of repetitions of the barker sequence
PREAMBLE_LENGTH_RATIO = 0.36  # Ratio of random preamble symbols compared to the number of symbols of the data

ABS_SAMPLE_RANGE = 0.85  # samples amplitude must be between -1 and 1, but we keep a little margin for the noise


def params_log():
    print("--------------------------------------------------------")
    print("-----------------------PARAMETERS-----------------------")
    print("--------------------------------------------------------")
    print("Mapping: {}".format(MAPPING))
    print("M = {}".format(M))
    print("Normalized mapping: {}\n".format(NORMALIZE_MAPPING))

    print("Modulation type: {}".format(MODULATION_TYPE))
    print("Bandwidth of the pulse: {} Hz\n".format(
        MODULATION_TYPE_1_BANDWIDTH if MODULATION_TYPE == 1 else MODULATION_TYPE_2_BANDWIDTH))

    print("Root-raised-cosine:")
    print("Beta = {}".format(BETA))
    print("Normalized pulse: {}".format(NORMALIZE_PULSE))
    print("USF: {} samples".format(USF))
    print("SPAN: {} samples\n".format(SPAN))

    print("Preamble type: {}".format(PREAMBLE_TYPE))
    if PREAMBLE_TYPE == "barker":
        print("Number of repetitions: {}\n".format(BARKER_SEQUENCE_REPETITION))
    else:
        print("Preamble length ratio = {}\n".format(PREAMBLE_LENGTH_RATIO))

    print("Maximum absolute value after scaling: {}".format(ABS_SAMPLE_RANGE))
    print("--------------------------------------------------------")
    print("--------------------------------------------------------")
    print("--------------------------------------------------------")
    print()
