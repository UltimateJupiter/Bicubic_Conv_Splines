import numpy as np
import scipy
from tqdm import tqdm


def sinh(x):
    e = np.e
    return (e ** x - e ** (-x)) / 2

def cosh(x):
    e = np.e
    return (e ** x + e ** (-x)) / 2

def tridiagonal_solver(A, b, vis=False):

    """
    Inputs:
        A       : a size 3*N np.ndarray (representing a N*N tridiagonal matrix A')
                  Note that A[0][0] and A[-1][2] are constant 0
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

    """
    Code for Debug
    ==
    mat_a = np.zeros((len(A), len(A)), "float64")
    mat_a[0][:2] = A[0][1:]
    mat_a[-1][-2:] = A[-1][:-1]
    for i in range(1, len(A) - 1):
        mat_a[i][i-1:i+2] = A[i]

    res = ga.tridiagonal_solver(A, b)
    print(mat_a)
    print(np.matmul(mat_a, res) - b)
    """


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
        

def array_padding_1D(L, blank=False, size=1):

    X = np.zeros(len(L) + size * 2)
    X[size: -size] = L

    for i in range(-size, 0):
        X[i] = 2 * X[-i - 1] - X[-i - 2]

    for i in range(size - 1, -1, -1):
        X[i] = 2 * X[i + 1] - X[i + 2]

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

def second_order_finitedif_derivative_approx(x, y, boundary_uncentered=True):

    """
    O(h^2) derivitive calculation (boundaries are O(h^2) for non_centered & equally spaced)
    If boundary_uncentered is False, the ghost values are processed by padding
    """

    assert isinstance(x, np.ndarray)
    assert len(x) >= 2, "Too few points"

    ret = np.zeros_like(x, dtype="float64")

    if boundary_uncentered:
        ret[0] = (y[2] - 4 * y[1] + 3 * y[0]) / (x[0] - x[2])
        ret[-1] = (y[-3] - 4 * y[-2] + 3 * y[-1]) / (x[-1] - x[-3])
        ret[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    
    if not boundary_uncentered:
        X = array_padding_1D(x, size=1)
        Y = array_padding_1D(Y, size=1)
        ret = (Y[1:] - Y[:-1]) / (X[1:] - X[:-1])
    
    return ret

def fourth_order_finitedif_derivative_approx(x, y, boundary_uncentered=True):

    """
    O(h^4) derivitive calculation (boundaries are O(h^3))
    If boundary_uncentered is False, the ghost values are processed by padding
    """
    
    assert isinstance(x, np.ndarray)
    assert len(x) >= 4, "Too few points"

    ret = np.zeros_like(x, dtype="float64")

    if boundary_uncentered:
        ret[0] = (-22 * y[0] + 36 * y[1] - 18 * y[2] + 4 * y[3]) / (-22 * x[0] + 36 * x[1] - 18 * x[2] + 4 * x[3])
        ret[1] = (-2 * y[0] - 3 * y[1] + 6 * y[2] - y[3]) / (-2 * x[0] - 3 * x[1] + 6 * x[2] - x[3])
        ret[-2] = (-2 * y[-1] - 3 * y[-2] + 6 * y[-3] - y[-4]) / (-2 * x[-1] - 3 * x[-2] + 6 * x[-3] - x[-4])
        ret[-1] = (-22 * y[-1] + 36 * y[-2] - 18 * y[-3] + 4 * y[-4]) / (-22 * x[-1] + 36 * x[-2] - 18 * x[-3] + 4 * x[-4])
        ret[2:-2] = (-y[:-4] + 8 * y[1:-3] - 8 * y[3:-1] + y[4:]) / (-x[:-4] + 8 * x[1:-3] - 8 * x[3:-1] + x[4:])
    
    if not boundary_uncentered:
        X = array_padding_1D(x, size=2)
        Y = array_padding_1D(Y, size=2)
        ret = (-Y[:-4] + 8 * Y[1:-3] - 8 * Y[3:-1] + Y[4:]) / (-X[:-4] + 8 * X[1:-3] - 8 * X[3:-1] + X[4:])

    return ret