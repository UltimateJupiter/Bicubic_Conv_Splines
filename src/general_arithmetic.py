import numpy as np
import scipy
from tqdm import tqdm

def tridiagonal_solver(A, b, vis=False):

    """
    Inputs:
        A       : a size 3*N np.ndarray (representing a N*N tridiagonal matrix A')
                  Note that A[0][0] and a[-1][2] are constant 0
        b       : a size N np.array where A'x = b
    
    Outputs:
        x: a size N np.array that A'x = b
    """
    
    A = A.astype("float64")
    b = b.astype("float64")
    iters = len(A)

    assert iters >= 3, "Too few points to perform the calculation"
    
    U = np.zeros((iters, 2), dtype="float64")
    x = np.zeros(iters, dtype="float64")
    
    U[0] = A[0][1:]
    U[:, 1] = A[:, 2]

    if vis == True:
        iterations = tqdm(range(1, iters))

    else:
        iterations = range(1, iters)
    
    for i in iterations:
        
        w = A[i][0] / U[i - 1][0]
        b[i] = b[i] - w * b[i - 1]
        U[i][0] = A[i][1] - w * U[i - 1][1]
        
    
    x[-1] = b[-1] / U[-1][0]
    for i in range(iters - 2, -1, -1):
        x[i] = (b[i] - U[i][1] * x[i + 1]) / U[i][0]
    
    return x


def piecewise_1d_to_2d(M, func, scale):

    # Edge cases should be well handled by func itself

    assert isinstance(M, np.ndarray)
    m, n = M.shape
    
    tmp_1 = np.zeros([m, (n - 1) * scale + 1])
    for i in range(m):
        tmp_1[i] = func(M[i], scale)
    tmp_1 = np.transpose(tmp_1)

    tmp_2 = np.zeros([(m - 1) * scale + 1, (n - 1) * scale + 1])
    for i in range((n - 1) * scale + 1):
        tmp_2[i] = func(tmp_1[i], scale)
    ret = np.transpose(tmp_2)

    return ret
        

def array_padding_1D(L, blank=False):

    X = np.zeros(len(L) + 2)
    X[1: -1] = L
    X[0], X[-1] = 2 * X[1] - X[2], 2 * X[-2] - X[-3]

    return X

def runge_2d(x_range, y_range):

    cx, cy = (x_range - 1) / 2, (y_range - 1) / 2
    norm = min(cx, cy)
    ret = np.zeros([x_range, y_range])
    for x in range(x_range):
        for y in range(y_range):
            h = np.sqrt((x - cx)**2 + (y - cy)**2)/norm
            ret[x][y] = 1 / (1 + 25 * h**2)
    return ret

def fourth_order_finitedif_derivative_approx(x, y, not_a_knot=True):

    assert isinstance(x, np.ndarray)
    assert len(x) >= 4, "Too few points"

    ret = np.zeros_like(x)

    if not not_a_knot:
        ret[0] = (-22 * y[0] + 36 * y[1] - 18 * y[2] + 4 * y[3]) / (-22 * x[0] + 36 * x[1] - 18 * x[2] + 4 * x[3])
        ret[1] = (-2 * y[0] - 3 * y[1] + 6 * y[2] - y[3]) / (-2 * y[0] - 3 * y[1] + 6 * y[2] - y[3])
        ret[-2] = (-2 * y[-1] - 3 * y[-2] + 6 * y[-3] - y[-4]) / (-2 * y[-1] - 3 * y[-2] + 6 * y[-3] - y[-4])
        ret[-1] = (-22 * y[-1] + 36 * y[-2] - 18 * y[-3] + 4 * y[-4]) / (-22 * x[-1] + 36 * x[-2] - 18 * x[-3] + 4 * x[-4])
    for i in range(2, len(x) - 2):
        ret[i] = (-y[i - 2] + 8 * y[i - 1] - 8 * y[i + 1] + y[i + 2]) / (-x[i - 2] + 8 * x[i - 1] - 8 * x[i + 1] + x[i + 2])
    
    return ret

