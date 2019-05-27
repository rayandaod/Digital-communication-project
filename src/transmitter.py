import subprocess
import time
import sys
import numpy as np
from scipy.signal import upfirdn

import fourier_helper
import helper
import mappings
import params
import plot_helper
import preambles
import pulses
import read_write


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

    # TODO: refactor and make modular (PAM not handled yet)
    if params.MODULATION_TYPE == 3:
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

    elif params.MODULATION_TYPE == 1 or params.MODULATION_TYPE == 2:
        # New structure with bits_per_symbol bits by row
        new_bits = [new_bits[i:i + params.BITS_PER_SYMBOL] for i in range(0, len(new_bits), params.BITS_PER_SYMBOL)]

        # Convert this new bits sequence to an integer sequence
        ints = [[int(b, 2) for b in new_bits]]

        if params.logs:
            print("Cropped and re-structured bits:\n{}".format(new_bits))
            print("Equivalent integers (indices for our mapping):\n{}".format(ints))
            print("--------------------------------------------------------")

    else:
        raise ValueError("This modulation type does not exist yet... He he he")

    if params.MAPPING == "qam" or params.MAPPING == "psk":
        corresponding_symbols = np.zeros(np.shape(ints), dtype=complex)
    elif params.MAPPING == "pam":
        corresponding_symbols = np.zeros(np.shape(ints), dtype=int)
    else:
        raise ValueError("This mapping type does not exist yet... He he he")

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
    :param data_symbols: the symbols modulating the pulse
    :param USF: the up-sampling factor (number of samples per symbols)
    :return: the samples of a modulated pulse train to send to the server
    """

    # Generate the preamble_symbols and write them in the appropriate file
    preambles.generate_preamble_symbols(len(data_symbols))
    preamble_symbols = read_write.read_preamble_symbols()
    if params.logs:
        print("Preamble symbols:\n{}".format(preamble_symbols))
        print("--------------------------------------------------------")
    if params.plots:
        plot_helper.plot_complex_symbols(preamble_symbols, "Preamble symbols")

    # Concatenate the data symbols with the preamble symbols at the beginning and at the end
    total_symbols = np.zeros((data_symbols.shape[0], data_symbols.shape[1] + 2 * len(preamble_symbols)), dtype=complex)
    for i in range(len(total_symbols)):
        total_symbols[i] = np.concatenate((preamble_symbols, data_symbols[i], preamble_symbols[::-1]))

    print("Total symbols: {}".format(total_symbols))
    print("Shape of the total symbols: {}".format(np.shape(total_symbols)))

    # Shape each of the symbols array
    samples = []
    for i in range(len(total_symbols)):
        samples.append(upfirdn(h, total_symbols[i], USF))

    # # Remove the ramp-up and ramp-down of the samples
    # cropped_samples = []
    # for i in range(len(samples)):
    #     # TODO: why + and - 3? Might be wrong
    #     cropped_samples.append(samples[i][int(params.SPAN/2) + 3:len(samples[i]) - int(params.SPAN/2) - 3])
    # samples = cropped_samples

    if params.logs:
        print("Shaping the preamble and the data...")
        print("Samples: {}".format(samples))
        print("Up-sampling factor: {}".format(params.USF))
        print("Shape of the total samples: {}".format(np.shape(samples)))
        print("--------------------------------------------------------")
    if params.plots:
        for i in range(len(samples)):
            plot_helper.samples_fft_plots(samples[i], "Samples {}".format(i), shift=True)

    # Write the preamble samples (base-band, so might be complex) cropped in the preamble_samples file
    preamble_samples = upfirdn(h, preamble_symbols, USF)
    # # TODO: why +3 and - 2? Might be wrong
    # preamble_samples = preamble_samples[int(params.SPAN/2) + 3:len(preamble_samples) - int(params.SPAN/2) + 2]
    read_write.write_preamble_samples(preamble_samples)

    if params.logs:
        print("Shaping the preamble...")
        print("Number of samples for the preamble: {}".format(len(preamble_samples)))
        print("--------------------------------------------------------")
    if params.plots:
        plot_helper.samples_fft_plots(preamble_samples, "Preamble samples", shift=True)

    # Choose the modulation frequencies
    if params.MODULATION_TYPE == 1 or params.MODULATION_TYPE == 3:
        modulating_frequencies = params.np.mean(params.FREQ_RANGES, axis=1)
    elif params.MODULATION_TYPE == 2:
        modulating_frequencies = [params.FREQ_RANGES[0][1], params.FREQ_RANGES[2][1]]
    else:
        raise ValueError("This mapping type does not exist yet... He he he")

    # Modulate the samples to fit in the required bands
    if np.any(np.iscomplex(samples)):
        if params.MODULATION_TYPE == 1 or params.MODULATION_TYPE == 2:
            samples = fourier_helper.modulate_complex_samples(samples[0], modulating_frequencies)
        elif params.MODULATION_TYPE == 3:
            modulated_samples = []
            for i in range(len(samples)):
                modulated_samples.append(fourier_helper.modulate_complex_samples(samples[i], [modulating_frequencies[i]]))
            samples = np.sum(modulated_samples, axis=0).flatten()

        if params.logs:
            print("Modulation of the signal...")
            print("Minimum sample after modulation: {}".format(min(samples)))
            print("Maximum sample after modulation: {}".format(max(samples)))
            print("--------------------------------------------------------")
        if params.plots:
            plot_helper.samples_fft_plots(samples, "Samples to send", time=True)
    else:
        raise ValueError("TODO: handle real samples (e.g SSB)")

    # Scale the signal to the range [-1, 1] (with a bit of uncertainty margin, according to params.ABS_SAMPLE_RANGE)
    samples = samples / (np.max(np.abs(samples))) * params.ABS_SAMPLE_RANGE

    if params.logs:
        print("Scaling the signal...")
        print("Number of samples: {}".format(len(samples)))
        print("Minimum sample after scaling: {}".format(min(samples)))
        print("Maximum sample after scaling: {}".format(max(samples)))
        print("--------------------------------------------------------")

    return samples


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
    moment = time.strftime("%Y-%b-%d__%H_%M_%S", time.localtime())
    log_file = open("../logs/" + moment + ".log", "w+")
    if params.logs:
        sys.stdout = log_file

    # Encode the message
    symbols = encoder(mappings.choose_mapping())

    # Generate the root-raised_cosine
    _, h_pulse = pulses.root_raised_cosine()

    # Construct the samples to send
    input_samples = waveform_former(h_pulse, symbols)

    # Write the samples in the input file
    read_write.write_samples(input_samples)

    # Send the samples to the server
    # send_samples()

