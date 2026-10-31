import numpy as np

def orthogonal_projection(v, L):
	"""
	Compute the orthogonal projection of vector v onto line L.

	:param v: The vector to be projected
	:param L: The line vector defining the direction of projection
	:return: List representing the projection of v onto L
	"""
	if all(x == 0 for x in L):
		return [0] * len(L)

	v = np.asarray(v)
	L = np.asarray(L)

	scalar = np.dot(v, L) / np.dot(L, L)
	return scalar * L
