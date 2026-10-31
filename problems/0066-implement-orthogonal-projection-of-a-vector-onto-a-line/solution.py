from functools import reduce

def orthogonal_projection(v, L):
	"""
	Compute the orthogonal projection of vector v onto line L.

	:param v: The vector to be projected
	:param L: The line vector defining the direction of projection
	:return: List representing the projection of v onto L
	"""
	if all(x == 0 for x in L):
		return [0] * len(L)
	
	vL = sum(x * y for x, y in zip(v, L))
	L_norm_sq = sum(x ** 2 for x in L)

	scalar = vL / L_norm_sq

	return [scalar * x for x in L]
