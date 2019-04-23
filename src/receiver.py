output_file = "../data/output_lorem_ipsum.txt"

def bits2string(b=None):
    return ''.join([chr(int(x, 2)) for x in b])

# TODO Do not forget to put back the most significant bit (0) in the received
# TODO sequence of bits
