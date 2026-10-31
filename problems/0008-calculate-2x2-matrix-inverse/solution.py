def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    """
    Calculate the inverse of a 2x2 matrix.
    
    Args:
        matrix: A 2x2 matrix represented as [[a, b], [c, d]]
    
    Returns:
        The inverse matrix as a 2x2 list, or None if the matrix is singular
        (i.e., determinant equals zero)
    """
    # Your code here
    a = matrix
    det = (a[0][0] * a[1][1]) - (a[0][1] * a[1][0])
    eps = 1e-15
    if abs(det) <= eps:
        return None

    adj = [[a[1][1], -a[0][1]], [-a[1][0], a[0][0]]]

    a_inv = [[round(el * (1 / det), 2) for el in row] for row in adj]

    return a_inv

