import numpy as np

# General variables
verbose = False
message_file_path = "../data/input_lorem_ipsum.txt"
output_file_path = "../data/output_lorem_ipsum.txt"
server_hostname = "iscsrv72.epfl.ch"
server_port = 80

# Communication parameters (you should only tweak the first 2 for this project)
M = 4  # length of the mapping
MOD_TYPE = "qam"  # modulation type
BITS_PER_SYMBOL = int(np.log2(M))
NOISE_VAR = 0.1
T = 1  # seconds
SAMPLING_RATE = 22050  # samples per second
Fs = SAMPLING_RATE/T  # sampling frequency
ABS_SAMPLE_INTERVAL = 1  # samples amplitude must be between -1 and 1

pn_seq = [-1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1,
          -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1,
          1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1,
          -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1,
          1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, 1,
          -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1,
          1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1]

# mapping = mapping/np.sqrt(np.mean(np.abs(mapping)**2))
