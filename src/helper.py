import params

def string2bits(s=''):
    """
    :param s: the string to be converted
    :return: the corresponding array of bits
    """
    return [bin(ord(x))[2:].zfill(8) for x in s]


def bits2string(b=None):
    """
    :param b: array of bits to be converted
    :return: the corresponding string
    """
    return ''.join([chr(int(x, 2)) for x in b])


def read_preamble_symbols():
    preamble_symbol_file = open(params.preamble_symbol_file_path, "r")
    preamble_symbols = [complex(line) for line in preamble_symbol_file.readlines()]
    preamble_symbol_file.close()
    return preamble_symbols


if __name__ == "__main__":
    print("helper.py")
