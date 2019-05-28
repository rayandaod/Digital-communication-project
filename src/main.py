import sys
import time

import params
import pulses
import read_write
import receiver
import transmitter

if __name__ == "__main__":
    if params.logs:
        moment = time.strftime("%Y-%b-%d__%H_%M_%S", time.localtime())
        log_file = open("../logs/" + moment + ".log", "w+")
        sys.stdout = log_file

    # Write the parameters in the log file
    if params.logs:
        params.params_log()

    # Transmitter
    symbols = transmitter.encoder()
    _, h = pulses.root_raised_cosine()
    samples_to_send = transmitter.waveform_former(h, symbols)
    read_write.write_samples(samples_to_send)
    transmitter.send_samples()

    # Receiver
    received_symbols = receiver.n_tuple_former()
