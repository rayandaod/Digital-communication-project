import numpy as np

import read_write
import transmitter_helper


# TODO: Make modular (PAM not handled yet)
def encoder():
    """
    Encode a message into a sequence of symbols according to the given mapping

    :return: The corresponding symbols for the message
    """
    # Retrieve the message from the file as bytes
    message_bytes = transmitter_helper.retrieve_message_as_bytes()

    # Associate the message bytes to the corresponding symbols
    corresponding_symbols = transmitter_helper.grouped_bytes_to_symbols(message_bytes)

    return np.asarray(corresponding_symbols)


def waveform_former(h, data_symbols):
    """
    Shape the data symbols with the pulse h

    :param h:               The pulse used to shape the symbols
    :param data_symbols:    The data symbols modulating the pulse
    :return:                The samples of a modulated pulse train to send to the server
    """
    # Generate the preamble_symbols and write them in the appropriate file
    preamble_symbols = transmitter_helper.generate_preamble_to_transmit(len(data_symbols))

    # Shape the preamble symbols and write the preamble samples in the preamble_samples file
    transmitter_helper.shape_preamble_samples(h, preamble_symbols)

    # Concatenate the data symbols with the preamble symbols at the beginning and at the end
    p_data_p_symbols = transmitter_helper.concatenate_symbols(preamble_symbols, data_symbols)

    # Shape each of the symbols array
    p_data_p_samples = transmitter_helper.shape_symbols(h, p_data_p_symbols)

    # Choose the modulation frequencies and modulate the samples
    p_data_p_modulated_samples = transmitter_helper.modulate_samples(p_data_p_samples)

    # Scale the signal to the range [-1, 1] (with a bit of uncertainty margin, according to params.ABS_SAMPLE_RANGE)
    samples_to_send = transmitter_helper.scale_samples(p_data_p_modulated_samples)

    # Write the samples to send in the appropriate file
    read_write.write_samples(samples_to_send)

    return samples_to_send


def send_samples():
    """
    Send the samples to the server, and received the output samples in the corresponding file

    :return: None
    """
    transmitter_helper.send_samples()
    return None
