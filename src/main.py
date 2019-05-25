import mappings
import receiver
import transmitter

if __name__ == "__main__":
    # Encoder
    ints = transmitter.message_to_ints()
    mapping = mappings.mapping
    symbols = transmitter.encoder(ints, mapping)

    # Receiver
    ints = receiver.decoder(symbols, mapping)
    guessed_message = receiver.ints_to_message(ints)
