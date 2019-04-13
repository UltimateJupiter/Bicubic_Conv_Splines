import numpy as np
import scipy
from general_arithmetic import piecewise_1d_to_2d


def divided_difference_coefficient(x, y):

    m = len(x)
    x = np.copy(x)
    a = np.copy(y)
    for k in range(1,m):
        a[k: m] = (a[k: m] - a[k - 1])/(x[k: m] - x[k - 1])

    return a

def global_polynomial_interpolation(x, y, tar):

    coeff = np.polyfit(x, y, deg=len(x)-1)
    func = np.poly1d(coeff)
    return func(tar)

def global_polynomial_interpolation_scaled(L, scale):

    x = np.arange(len(L))
    y = L
    target_x = np.linspace(0, len(L) - 1, (len(L) - 1) * scale + 1)
    return global_polynomial_interpolation(x, y, target_x)

def polyfit2d_from1d_helper(M, scale):

    return piecewise_1d_to_2d(M, global_polynomial_interpolation_scaled, scale)


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