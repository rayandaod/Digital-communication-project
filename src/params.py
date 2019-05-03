import numpy as np

# General variables
verbose = True
message_file_path = "../data/input_lorem_ipsum.txt"
output_file_path = "../data/output_lorem_ipsum.txt"
server_hostname = "iscsrv72.epfl.ch"
server_port = 80

# Communication parameters (you should only tweak the first 5 ones for this project)
M = 4  # length of the mapping
MOD_TYPE = "qam"  # modulation type
BITS_PER_SYMBOL = int(np.log2(M))  # number of bits we transmit per symbol
BETA = 0.22  # rolloff factor of our root-raised-cosine pulse
T = 10  # symbol period (in number of samples)

# DO NOT CHANGE THE FOLLOWING VARIABLES
NOISE_VAR = 0.1  # noise variance of the channel
SAMPLING_RATE = 22050  # samples per second
Fs = SAMPLING_RATE/T  # sampling frequency (in Hz)
ABS_SAMPLE_INTERVAL = 1  # samples amplitude must be between -1 and 1
PREAMBLE = [-1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1,  # Preamble sequence
            -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1,
            1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1,
            -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1,
            1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, 1,
            -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1,
            1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1]

# mapping = mapping/np.sqrt(np.mean(np.abs(mapping)**2))

# TODO check that the length of the pn-sequence is optimal
