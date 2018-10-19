import time
import math
import numpy as np
from numba import jit


@jit("f8(f8[:])", cache=False, nopython=True, nogil=True, parallel=True)
def esum(z):
    return np.sum(np.exp(z))


@jit("f8[:](f8[:])", cache=False, nopython=True, nogil=True, parallel=True)
def softmax_optimized(z):
    num = np.exp(z)
    s = num / esum(z)
    return s


@jit("f8[:](f8[:])", cache=False, nopython=True, nogil=True, parallel=True)
def softmax_python(z):
    s = []

    exp_sum = 0
    for i in range(len(z)):
        exp_sum += math.exp(z[i])

    for i in range(len(z)):
        s += [math.exp(z[i]) / exp_sum]

    return s


def main():
    np.random.seed(0)
    z = np.random.uniform(0, 10, 10 ** 8)   # generate random floats in the range [0,10)

    start = time.time()
    softmax_python(z.tolist())          # run pure python version of softmax
    elapsed = time.time() - start
    print('Ran pure python softmax calculations in {} seconds'.format(elapsed))

    softmax_optimized(z)                     # cache jit compilation
    start = time.time()
    softmax_optimized(z)                     # run optimzed version of softmax
    elapsed = time.time() - start
    print('\nRan optimized softmax calculations in {} seconds'.format(elapsed))


if __name__ == '__main__':
    main()
