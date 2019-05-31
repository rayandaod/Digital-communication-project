def string2bits(s=''):
    """
    Converts a string to an array of strings of bytes

    :param s:   The string to be converted
    :return:    The corresponding array of bits
    """
    return [bin(ord(x))[2:].zfill(8) for x in s]


def bits2string(b=None):
    """
    Convert an array of strings of bytes to a single string

    :param b:   The array of bits to be converted
    :return:    The corresponding string
    """
    return ''.join([chr(int(x, 2)) for x in b])


def compare_messages(message_sent, message):
    """
    Compare 2 messages and print the differences

    :param message_sent:    The first message to compare
    :param message:         The second message to compare
    :return:                None
    """
    same_length = len(message_sent) == len(message)
    n_errors = 0
    comparison = ""
    for i in range(min(len(message), len(message_sent))):
        if message_sent[i] == message[i]:
            comparison += " "
        else:
            n_errors += 1
            comparison += '!'

    print("Errors:           {}".format(comparison))
    if not same_length:
        print("/!\\ MESSAGE DO NOT HAVE THE SAME LENGTH /!\\")
    else:
        print("({} error(s))".format(n_errors))
    return None
