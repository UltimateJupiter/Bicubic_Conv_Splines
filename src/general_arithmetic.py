import numpy as np
import scipy

def tridiagonal_solver(A, b):

    """
    Inputs:
        A: a size 3*N np.ndarray (representing a N*N tridiagonal matrix A')
            Note that A[0][0] and a[-1][2] are constant 0
        b: a size N np.array where A'x = b
    
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
    
    for i in range(1, iters):
        
        w = A[i][0] / U[i - 1][0]
        b[i] = b[i] - w * b[i - 1]
        U[i][0] = A[i][1] - w * U[i - 1][1]
    
    x[-1] = b[-1] / U[-1][0]
    for i in range(iters - 2, -1, -1):
        x[i] = (b[i] - U[i][1] * x[i + 1]) / U[i][0]
    
    return x
