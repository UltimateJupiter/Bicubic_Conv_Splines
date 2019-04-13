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

def pchip_coeff_generator(x, y, sort_x = False):

    """x and y should be equally spaced data points"""
    
    if sort_x:
        sorted(x)

    assert len(x) == len(y)
    coeff = np.zeros([4, len(x) - 1])
    Y = array_padding_1D(y)
    h = x[1] - x[0]
    
    coeff = np.zeros([4, len(x) - 1])
    coeff[0] = y[:-1]
    coeff[1] = y[1:]
    coeff[2] = (Y[2: -1] - Y[:-3]) / (2 * h)
    coeff[3] = (Y[3:] - Y[1:-2]) / (2 * h)

    coeff = np.matmul(solve_mat, coeff)
    coeff = np.transpose(coeff)
    return coeff

def pchip_1d_generator(x, y, tar_xs, plotting=0):

    plt_x = []
    plt_y = []
    sorted(x)
    sorted(y)
    coeff = pchip_coeff_generator(x, y)
    # print(coeff)

    if plotting == 1:
        for i in range(len(x) - 1):
            for m in np.linspace(x[i], x[i + 1], 20)[:-1]:
                h = m - x[i]
                val = coeff[i][0] + coeff[i][1] * h + coeff[i][2] * h**2 + coeff[i][3] * h**3
                plt_x.append(x[i] + h)
                plt_y.append(val)
        plt.scatter(x, y)
        plt.plot(plt_x, plt_y)
        plt.show()
    
    ret_ys = np.zeros_like(tar_xs)
    start_base = 0
    for i in range(len(tar_xs)):
        tar_x = tar_xs[i]
        assert tar_x >= x[0] and tar_x <= x[-1]
        for j in range(start_base, len(x) - 1):
            if tar_x >= x[j] and tar_x <= x[j + 1]:
                start_base = j
                h = tar_x - x[j]
                ret_ys[i] = coeff[j][0] + coeff[j][1] * h + coeff[j][2] * h**2 + coeff[j][3] * h**3
                break

    if plotting == 2:
        plt.scatter(x, y)
        plt.plot(tar_xs, ret_ys)
        plt.show()

    return ret_ys

def pchip_scaled(L, scale):
    
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

    return piecewise_1d_to_2d(M, pchip_scaled, scale)
    

# print(pchip_base_helper(np.random.random([5,5]), 10))