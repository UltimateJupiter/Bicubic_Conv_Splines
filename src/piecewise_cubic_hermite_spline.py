import numpy as np
from general_arithmetic import piecewise_1d_to_2d, array_padding_1D
import matplotlib.pyplot as plt

solve_mat = np.linalg.inv(np.array(
    [
        [1, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 2, 3]
    ]
))

def pchip_base(L, scale):
    
    assert isinstance(L, np.ndarray)
    assert scale >= 2
    assert len(L) >= 2

    X = array_padding_1D(L)

    ret = np.zeros((len(L) - 1) * scale + 1)

    coeff = np.zeros([4, len(L) - 1])
    coeff[0] = L[:-1]
    coeff[1] = L[1:]
    coeff[2] = (X[2: -1] - X[:-3]) / 2
    coeff[3] = (X[3:] - X[1:-2]) / 2

    coeff = np.matmul(solve_mat, coeff)
    coeff = np.transpose(coeff)

    ret = np.zeros([(len(L) - 1) * scale + 1])
    ret[-1] = L[-1]
    for i in range(0, (len(L) - 1) * scale):

        val = i / scale
        dx = val - int(val)
        ret[i] = np.dot(coeff[int(val)], np.array([1, dx, dx**2, dx**3]))

    return ret


def pchip_base_helper(M, scale):

    return piecewise_1d_to_2d(M, pchip_base, scale)
    

# print(pchip_base_helper(np.random.random([5,5]), 10))