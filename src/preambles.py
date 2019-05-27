import numpy as np

import mappings
import params
import read_write


def generate_preamble_symbols(n_symbols_to_send):
    """
    Generate preamble symbols
    :param n_symbols_to_send:
    :return:
    """
    if params.PREAMBLE_TYPE == "random":
        preamble_symbols = generate_random_preamble_symbols(n_symbols_to_send)
    elif params.PREAMBLE_TYPE == "barker":
        preamble_symbols = generate_barker_preamble_symbols()
    else:
        raise ValueError('This preamble type does not exist yet... Hehehe')

    if params.MAPPING == "qam" and not params.NORMALIZE_MAPPING:
        number_of_points_on_side = np.sqrt(params.M)/2.0
        test_value = np.log2(params.M)/2.0
        if test_value != int(test_value):
            raise ValueError('Not a valid Qam value')
        read_write.write_preamble_symbols(preamble_symbols * (2.0*number_of_points_on_side - 1))

    return None


def generate_random_preamble_symbols(n_symbols_to_send):
    preamble_symbols = np.random.choice(mappings.choose_mapping(),
                                        size=int(np.ceil(n_symbols_to_send * params.PREAMBLE_LENGTH_RATIO)))
    return preamble_symbols


def generate_barker_preamble_symbols():
    barker_code_13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
    preamble_symbols = np.repeat(barker_code_13, params.BARKER_SEQUENCE_REPETITION)
    return preamble_symbols + 1j * preamble_symbols
