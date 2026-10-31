import numpy as np

def orthogonal_projection(v, L):
	"""
	Compute the orthogonal projection of vector v onto line L.

	:param v: The vector to be projected
	:param L: The line vector defining the direction of projection
	:return: List representing the projection of v onto L
	"""
	v = np.asarray(v)
	L = np.asarray(L)

	denom = np.dot(L, L)
	if np.isclose(denom, 0):
		return np.zeros_like(v, dtype=float)

	scalar = np.dot(v, L) / denom
	return scalar * L
