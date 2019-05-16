import transmitter
import receiver
import helper

if __name__ == "__main__":
    symbols = transmitter.encoder(transmitter.message_to_ints(), helper.mapping)
    ints = receiver.decoder(symbols, helper.mapping)
    guessed_message = receiver.ints_to_message(ints)
