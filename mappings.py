import numpy as np


def qam_map(M):
    check_mappings(M)
    N = np.sqrt(M) - 1
    aux = np.arange(-N, N+1, 2)
    x, y = np.meshgrid(aux, aux[::-1])
    return x + 1j*y

def psk_map(M):
    check_mappings(M)
    return np.exp(1j*2*np.pi*np.arange(0, M)/M)

def check_mappings(M):
    log_sqrt_m = np.log2(np.sqrt(M))
    if log_sqrt_m != np.ceil(log_sqrt_m):
        raise ValueError('Parameter[M] is not of the form 2^2K, K a positive integer.')

if __name__ == '__main__':
    M = 16
    print(qam_map(M))
    print(psk_map(M))