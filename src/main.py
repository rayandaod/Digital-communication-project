import transmitter
import receiver
import enc_dec_helper

if __name__ == "__main__":
    # Encoder
    ints = transmitter.message_to_ints()
    mapping = enc_dec_helper.mapping
    symbols = transmitter.encoder(ints, mapping)

    # Receiver
    ints = receiver.decoder(symbols, mapping)
    guessed_message = receiver.ints_to_message(ints)
