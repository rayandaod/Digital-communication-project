import subprocess
import time
import sys
import numpy as np

import helper
import mappings
import params
import plot_helper
import pulses
import read_write
import transmitter_helper


# TODO: refactor and make modular (PAM not handled yet)
# TODO: make it simple with transmitter helper
def encoder(mapping):
    """
    Encode a message into a sequence of symbols according to the given mapping
    :param mapping: the mapping used for transmission
    :return: the corresponding symbols for the message
    """
    # Retrieve the message from file
    message_file = open(params.input_message_file_path)
    message = message_file.readline()
    print("Sent message:\n{}".format(message))
    if params.logs:
        print("Length: {} characters".format(len(message)))
        print("--------------------------------------------------------")

    # Retrieve the message as a sequences of binary bytes
    string_bytes = helper.string2bits(message)

    if params.logs:
        print("Corresponding bytes:\n{}".format(string_bytes))
        print("--------------------------------------------------------")

    # Remove the most significant bit (0) as it is useless in ASCII (do not forget to put it again in the receiver!)
    new_bits = [b[1:] for b in string_bytes]

    # Make a new string with these cropped bytes
    new_bits = ''.join(new_bits)

    if params.MODULATION_TYPE == 1 or params.MODULATION_TYPE == 2:
        # New structure with bits_per_symbol bits by row
        new_bits = [new_bits[i:i + params.BITS_PER_SYMBOL] for i in range(0, len(new_bits), params.BITS_PER_SYMBOL)]

        # Convert this new bits sequence to an integer sequence
        ints = [[int(b, 2) for b in new_bits]]

        if params.logs:
            print("Cropped and re-structured bits:\n{}".format(new_bits))
            print("Equivalent integers (indices for our mapping):\n{}".format(ints))
            print("--------------------------------------------------------")
    elif params.MODULATION_TYPE == 3:
        # Choose the number of bit streams (depends on the number of frequency ranges)
        n_bit_streams = len(params.FREQ_RANGES)

        # Choose the length of our bit streams
        len_bit_streams = int(np.ceil(len(new_bits) / (n_bit_streams - 1)))

        # Make it even
        if len_bit_streams % 2 != 0:
            len_bit_streams = len_bit_streams + 1

        # Construct the bit streams array with zeros
        bit_streams = np.zeros((n_bit_streams, len_bit_streams), dtype=int)

        # Fill the bit streams arrays
        for i in range(len(new_bits)):
            bit_streams[i % (n_bit_streams - 1)][int(np.ceil(i / (n_bit_streams - 1)))] = new_bits[i]

        # Construct the parity check bit stream and insert it in the bit streams array
        pc_bit_stream = np.sum(bit_streams[:n_bit_streams - 1], axis=0)
        for i in range(len_bit_streams):
            pc_bit_stream[i] = 0 if pc_bit_stream[i] % 2 == 0 else 1
        bit_streams[n_bit_streams - 1] = pc_bit_stream

        if params.logs:
            print(" ")
            print("Bit stream {}\n: {}".format(bit_streams.shape, bit_streams))
            print("--------------------------------------------------------")

        # Group them by groups of BITS_PER_SYMBOL bits
        ints = np.zeros((n_bit_streams, int(len_bit_streams / 2)), dtype=str)
        for i in range(n_bit_streams):
            for j in range(int(len_bit_streams / params.BITS_PER_SYMBOL)):
                grouped_bits = str(bit_streams[i][j]) + str(bit_streams[i][j + params.BITS_PER_SYMBOL - 1])
                mapping_index = int(grouped_bits, base=2)
                ints[i][j] = mapping_index

        if params.logs:
            print("Ints bits stream {}\n: {}".format(ints.shape, ints))
            print("--------------------------------------------------------")
    else:
        raise ValueError("This modulation type does not exist yet... He he he")

    corresponding_symbols = np.zeros(np.shape(ints), dtype=complex)
    for i in range(len(ints)):
        print(np.shape(ints))
        corresponding_symbols[i] = [mapping[int(j)] for j in ints[i]]

    if params.logs:
        print("Mapping the integers to the symbols in the mapping...")
        print("Symbols/n-tuples to be sent:\n{}".format(corresponding_symbols))
        print("Shape of the symbols: {}".format(np.shape(corresponding_symbols)))
        print("--------------------------------------------------------")
    if params.plots:
        plot_helper.plot_complex_symbols(corresponding_symbols, "{} data symbols to send"
                                         .format(np.shape(corresponding_symbols)), "blue")

    return np.asarray(corresponding_symbols)


def waveform_former(h, data_symbols, USF=params.USF):
    """
    :param h: the sampled pulse
    :param data_symbols: the data symbols modulating the pulse
    :param USF: the up-sampling factor, i.e the number of samples per symbols, also called SPS
    :return: the samples of a modulated pulse train to send to the server
    """
    # Generate the preamble_symbols and write them in the appropriate file ---------------------------------------------
    preamble_symbols = transmitter_helper.generate_preamble_to_transmit(len(data_symbols))
    # ------------------------------------------------------------------------------------------------------------------

    # Shape the preamble symbols and write the preamble samples in the preamble_samples file ---------------------------
    transmitter_helper.shape_preamble_samples(h, preamble_symbols, USF)
    # ------------------------------------------------------------------------------------------------------------------

    # Concatenate the data symbols with the preamble symbols at the beginning and at the end ---------------------------
    p_data_p_symbols = transmitter_helper.concatenate_symbols(preamble_symbols, data_symbols)
    # ------------------------------------------------------------------------------------------------------------------

    # Shape each of the symbols array ----------------------------------------------------------------------------------
    p_data_p_samples = transmitter_helper.shape_symbols(h, p_data_p_symbols, USF)
    # ------------------------------------------------------------------------------------------------------------------

    # Choose the modulation frequencies and modulate the samples -----------------------------------------------------
    p_data_p_modulated_samples = transmitter_helper.modulate_samples(p_data_p_samples)
    # ------------------------------------------------------------------------------------------------------------------

    # Scale the signal to the range [-1, 1] (with a bit of uncertainty margin, according to params.ABS_SAMPLE_RANGE)
    samples_to_send = transmitter_helper.scale_samples(p_data_p_modulated_samples)

    return samples_to_send


def send_samples():
    """
    Launch the client.py file with the correct arguments according to the parameters in the param file
    :return: None
    """
    subprocess.call(["python3 client.py" +
                     " --input_file=" + params.input_sample_file_path +
                     " --output_file=" + params.output_sample_file_path +
                     " --srv_hostname=" + params.server_hostname +
                     " --srv_port=" + str(params.server_port)],
                    shell=True)
    return None


# Intended for testing (to run the program, run main.py)
if __name__ == '__main__':
    if params.logs:
        moment = time.strftime("%Y-%b-%d__%H_%M_%S", time.localtime())
        log_file = open("../logs/" + moment + ".log", "w+")
        sys.stdout = log_file
        params.params_log()

    # Encode the message
    symbols = encoder(mappings.choose_mapping())

    # Generate the root-raised_cosine
    _, h_pulse = pulses.root_raised_cosine()

    # Construct the samples to send
    samples = waveform_former(h_pulse, symbols)

    # Write the samples in the input file
    read_write.write_samples(samples)

    # Send the samples to the server
    send_samples()

