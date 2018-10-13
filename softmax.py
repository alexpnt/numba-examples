import time
import numpy as np
from numba import jit


# @jit("f8[:](f8[:])", nopython=True, nogil=True, parallel=True)
def softmax(z):
	s = np.empty(z.shape)
	for j in range(z.shape[0]):
		s[j] = np.exp(z[j]) / np.sum(np.exp(z))
	return s


def main():
	np.random.seed(0)
	z = np.random.rand(2 ** 16)

	start = time.time()
	s = softmax(z)
	elapsed = time.time() - start

	print(s, '\nRan softmax calculations in {} seconds'.format(elapsed))


if __name__ == '__main__':
	main()
