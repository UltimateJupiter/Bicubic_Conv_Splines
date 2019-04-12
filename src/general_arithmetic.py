import numpy as np
import scipy
from tqdm import tqdm

def tridiagonal_solver(A, b):

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
    
    for i in tqdm(range(1, iters)):
        
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