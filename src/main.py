import sys
import time

import mappings
import receiver
import transmitter
import pulses
import params

if __name__ == "__main__":
    if params.logs:
        moment = time.strftime("%Y-%b-%d__%H_%M_%S", time.localtime())
        log_file = open("../logs/" + moment + ".log", "w")
        sys.stdout = log_file

    # Transmitter
    symbols = transmitter.encoder(transmitter.message_to_ints(), mappings.choose_mapping())
    _, h = pulses.root_raised_cosine()
    transmitter.symbols_to_samples(h, symbols)
    transmitter.send_samples()

    # Wait for the user to press enter (in case the server is down for example)
    input("Press Enter to continue...\n")

    # Receiver
    receiver.received_from_server()
