import numpy as np
from general_arithmetic import piecewise_1d_to_2d, array_padding_1D
import matplotlib.pyplot as plt
import general_arithmetic

solve_mat = np.linalg.inv(np.array(
    [
        [1, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 2, 3]
    ]
))

def local_monotone_mask(dydx, x, y):

    assert isinstance(dydx, np.ndarray)
    print(y)
    print(dydx)
    
    ret = np.zeros_like(dydx)
    ret[0] = dydx[0]
    ret[-1] = dydx[-1]
    
    slope = np.zeros(len(ret) - 1)
    slope = (y[1:] - y[:-1]) / (x[1:] - x[:-1])

    zero_base = np.zeros(len(ret) - 2)
    S_min = np.minimum(slope[1:], slope[:-1])
    S_max = np.maximum(slope[1:], slope[:-1])
    print(slope)
    print(S_max)
    print(S_min)

    min_cond = np.minimum(np.maximum(zero_base, dydx[1:-1]), 3 * S_min)
    max_cond = np.maximum(np.minimum(zero_base, dydx[1:-1]), 3 * S_max)

    # print((S_min < 0), (S_max > 0), slope)

    ret[1:-1] = min_cond * (S_min < 0) + max_cond * (S_max > 0)
    ret = dydx
    ret[1:-1] = ret[1:-1] * (1 - (S_min <= 0) * (S_max >= 0))
    print(ret)
    return ret

def mc_pchip_coeff_generator(x, y, sort_x=False, derives="2"):

    """x and y should be equally spaced data points"""
    
    if sort_x:
        sorted(x)

    assert len(x) == len(y)
    coeff = np.zeros([4, len(x) - 1])
    Y = array_padding_1D(y)
    h = x[1] - x[0]

    if derives == "2" or derives is None:
        dydx = general_arithmetic.second_order_finitedif_derivative_approx(x, y)
    if derives == "4":
        dydx = general_arithmetic.fourth_order_finitedif_derivative_approx(x, y)

    dydx = local_monotone_mask(dydx, x, y)

    coeff = np.zeros([4, len(x) - 1])
    coeff[0] = y[:-1]
    coeff[1] = y[1:]
    # coeff[2] = (Y[2: -1] - Y[:-3]) / (2 * h)
    # coeff[3] = (Y[3:] - Y[1:-2]) / (2 * h)
    # print(dydx)
    coeff[2] = dydx[:-1]
    coeff[3] = dydx[1:]

    coeff = np.matmul(solve_mat, coeff)
    coeff = np.transpose(coeff)
    return coeff

def mc_pchip_1d_generator(x, y, tar_xs, plotting=0, derives="2"):

    plt_x = []
    plt_y = []
    sorted(x)
    sorted(y)
    coeff = mc_pchip_coeff_generator(x, y, derives=derives)
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

def mc_pchip_1d_scaled(L, scale):

    x = np.arange(len(L))
    y = L
    target_x = np.linspace(0, len(L) - 1, (len(L) - 1) * scale + 1)
    return mc_pchip_1d_generator(x, y, target_x)

def mc_pchip_base_helper(M, scale):

    return piecewise_1d_to_2d(M, mc_pchip_1d_scaled, scale)