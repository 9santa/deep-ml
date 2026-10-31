def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
              
    ra, ca, rb, cb = len(a), len(a[0]), len(b), len(b[0])
    if ca != rb:
        return -1

    c = [[0.0 for _ in range(cb)] for _ in range(ra)]

    for i in range(ra):
        for k in range(ca):
            for j in range(cb):
                c[i][j] += a[i][k] * b[k][j]

    return c