import sys
import time

import mappings
import params
import pulses
import read_write
import receiver
import transmitter

if __name__ == "__main__":
    moment = time.strftime("%Y-%b-%d__%H_%M_%S", time.localtime())
    log_file = open("../logs/" + moment + ".log", "w+")
    if params.logs:
        sys.stdout = log_file

    # Write the parameters in the log file
    if params.logs:
        params.params_log()

    # Transmitter
    symbols = transmitter.encoder(mappings.choose_mapping())
    _, h = pulses.root_raised_cosine()
    samples_to_send = transmitter.waveform_former(h, symbols)
    read_write.write_samples(samples_to_send)
    transmitter.send_samples()

    # Receiver
    receiver.received_from_server()
