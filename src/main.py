import sys
import time

import mappings
import receiver
import transmitter
import pulses
import params
import read_write

if __name__ == "__main__":
    if params.logs:
        moment = time.strftime("%Y-%b-%d__%H_%M_%S", time.localtime())
        log_file = open("../logs/" + moment + ".log", "w")
        sys.stdout = log_file

    # Transmitter
    symbols = transmitter.encoder(transmitter.message_to_ints(), mappings.choose_mapping())
    _, h = pulses.root_raised_cosine()
    samples_to_send = transmitter.symbols_to_samples(h, symbols)
    read_write.write_samples(samples_to_send)
    transmitter.send_samples()

    # Wait for the user to press enter (in case the server is down for example)
    if params.logs:
        sys.stdout = sys.__stdout__
        input("Press Enter to continue...\n")
        sys.stdout = log_file

    # Receiver
    receiver.received_from_server()
