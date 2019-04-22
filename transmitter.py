import zlib
import sys

verbose = True

message_file_name = "input_outputs/input_lorem_ipsum.txt"


def string2bits(s=''):
    return [bin(ord(x))[2:].zfill(8) for x in s]


def encoder():
    return None


if __name__ == '__main__':
    message_file = open(message_file_name)
    message = message_file.readline()
    message_encoded = message.encode('ascii')
    compressed_message = zlib.compress(message_encoded)

    if verbose:
        print("Original message: {}".format(message))
        print("Encoded message: {}".format(message_encoded))
        print("Size (in bytes) of encoded message: {}".format(sys.getsizeof(message_encoded)))
        print("Compressed message: {}".format(compressed_message))
        print("Size (in bytes) of compressed message {}".format(sys.getsizeof(compressed_message)))