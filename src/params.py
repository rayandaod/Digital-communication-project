import numpy as np

# General variables
verbose = False
message_file_path = "../data/input_lorem_ipsum.txt"
output_file_path = "../data/output_lorem_ipsum.txt"
server_hostname = "iscsrv72.epfl.ch"
server_port = 80

# Communication parameters (you should change only the first 2)
M = 4
MOD_TYPE = "qam"
BITS_PER_SYMBOL = int(np.log2(M))
NOISE_VAR = 0.1
SAMPLING_RATE = 22050  # samples per second
ABS_SAMPLE_INTERVAL = 1  # samples amplitude must be between -1 and 1

# mapping = mapping/np.sqrt(np.mean(np.abs(mapping)**2))
