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


def compare_messages(message_sent, message):
    """
    Compare 2 messages and print the differences
    :param message_sent: the first message to compare
    :param message: the second message to compare
    :return: None
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


# Intended for testing (to run the program, run main.py)
if __name__ == "__main__":
    print("helper.py")
