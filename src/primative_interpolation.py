import numpy as np


def get_x(x):
    return 0

def bilinear_interpolation(M, pos):

    """
    Inputs:
        M: 2D np.ndarray for raw data
        S: [height, width] of the mapped data
        pos: [x, y] of the target point
    """

    assert isinstance(M, np.ndarray)
    m, n = M.shape
    m -= 1
    n -= 1
    x, y = pos
    if x < 0 or y < 0 or x > n or y > m:
        raise Exception("Point out of bound")
    
    a, b = int(x), int(y)
    da, db = x - int(x), y - int(y)

    if x == a and y == b:
        return M[b][a]
    elif x == a:
        return M[b][a] + (M[b + 1][a] - M[b][a]) * db
    elif y == b:
        return M[b][a] + (M[b][a + 1] - M[b][a]) * da
    else:
        p1, p2, p3, p4 = M[b][a], M[b][a + 1], M[b + 1][a], M[b + 1][a + 1]
        p5 = p1 + (p2 - p1) * da
        p6 = p3 + (p4 - p3) * da
        return p5 + (p6 - p5) * db
    
def bilinear_helper(M, scale=100):

    assert isinstance(M, np.ndarray)
    m, n = M.shape
    m -= 1
    n -= 1
    ret = np.zeros([scale * m + 1, scale * n + 1])
    for i in range(scale * m + 1):
        for j in range(scale * n + 1):
            ret[i][j] = bilinear_interpolation(M, [j/scale, i/scale])
    
    return ret

def nearest_neighbor_helper(M, scale=100):

    assert isinstance(M, np.ndarray)
    m, n = M.shape
    m -= 1
    n -= 1
    ret = np.zeros([scale * m + 1, scale * n + 1])
    for i in range(scale * m + 1):
        for j in range(scale * n + 1):

            ret[i][j] = M[int(j/scale + 0.5)][int(i/scale + 0.5)]
    
    return ret