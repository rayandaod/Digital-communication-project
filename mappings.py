import numpy as np


# Returns a M-QAM constellation
def qam_map(M):
    log_sqrt_m = np.log2(np.sqrt(M))
    if log_sqrt_m != np.ceil(log_sqrt_m):
        raise ValueError('Parameter[M] is not of the form 2^2K, K a positive integer.')
    N = np.sqrt(M) - 1
    aux = np.arange(-N, N+1, 2)
    x, y = np.meshgrid(aux, aux[::-1])
    return x + 1j*y

# Returns a M-PSK constellation
def psk_map(M):
    return np.exp(1j*2*np.pi*np.arange(0, M)/M)


if __name__ == '__main__':
    M = 3
    print(qam_map(M))
    print(psk_map(M))
