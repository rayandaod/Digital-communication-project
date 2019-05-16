import transmitter
import receiver
from src.helpers import enc_dec_helper

if __name__ == "__main__":
    symbols = transmitter.encoder(transmitter.message_to_ints(), enc_dec_helper.mapping)
    ints = receiver.decoder(symbols, enc_dec_helper.mapping)
    guessed_message = receiver.ints_to_message(ints)
