import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from general_arithmetic import tridiagonal_solver

def cubic_spline_coeff_generator(x, y, sort_x = False):

    assert len(x) == len(y)
    assert len(x) >= 3
    
    h = [x[i + 1] - x[i] for i in range(len(x) - 1)]
    A = np.zeros((len(x) - 2, 3))
    B = np.zeros(len(x) - 2)
    
    for i in range(len(x) - 3):
        tmp = h[i + 1]
        A[i:i + 2, 1] += tmp * 2
        A[i][2] += tmp
        A[i + 1][0] += tmp

    A[0][1] += 2 * h[0]
    A[-1][1] += 2 * h[len(x) - 2]

    f_interval = [(y[i + 1] - y[i]) / h[i] for i in range(len(x) - 1)]
    
    for i in range(1, len(x) - 1):
        B[i - 1] = 3 * (f_interval[i] - f_interval[i - 1])
    
    c_tmp = tridiagonal_solver(A, B)
    c = np.zeros(len(x))
    c[1:-1] = c_tmp

    b = [f_interval[i] - h[i] * (2 * c[i] + c[i + 1]) / 3 for i in range(len(x) - 1)]
    d = [(c[i + 1] - c[i]) / 3 / h[i] for i in range(len(x) - 1)]
    coeff = np.zeros((len(x) - 1, 4))
    coeff[:, 0] = y[:-1]
    coeff[:, 1] = b
    coeff[:, 2] = c[:-1]
    coeff[:, 3] = d

    return coeff

def cubic_spline_generator(x, y, tar_xs, plotting=0):
    
    plt_x = []
    plt_y = []
    sorted(x)
    sorted(y)
    coeff = cubic_spline_coeff_generator(x, y)

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

def bicubic_main():

    start, end, points = 0, 10, 50
    x = np.linspace(start, end, points)
    y = np.sin(10 * x)
    tar = np.linspace(start, end, points * 5)
    ret_y = cubic_spline_generator(x, y, tar, plotting=0)

# bicubic_main()


