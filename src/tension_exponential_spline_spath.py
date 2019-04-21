"""


TODO: THIS CODE DOES NOT BEHAVE NORMALLY, NEED MODIFICATION


"""
import numpy as np
import general_arithmetic as ga
import matplotlib.pyplot as plt
import cubic_spline_interpolation

def tension_spline_coeff_generator(x, y, p):

    """
    Inputs:
        x: x data
        y: y data
        p: tension factors
    """
    assert x.shape == y.shape

    dt_x = x[1:] - x[:-1]
    dt_y = y[1:] - y[:-1] 

    dy = ga.second_order_finitedif_derivative_approx(x, y)
    print(dy)
    dy_head, dy_end = dy[0], dy[-1]

    z = ga.sinh(p * dt_x) / dt_x
    w = z / (p - z)

    alpha = y[1:] / dt_x
    beta = y[:-1] / dt_x
    # print(alpha, beta)
    v = (p * ga.cosh(p * dt_x) - z) / (z - p)

    t = (p * p * z * dt_x) / (v * v - 1)
    A = np.asarray([t[:-1], (t[:-1] * v[:-1] + t[1:] * v[1:]), t[1:]])
    A = np.transpose(A)
    A[0][0], A[-1][-1] = 0, 0
    print(A)
    
    b = t[:-1] * (v[:-1] + 1) * (dt_y[:-1] / dt_x[:-1]) + t[1:] * (v[1:] + 1) * (dt_y[1:] / dt_x[1:])
    b[0] = b[0] - t[0] * dy_head
    b[-1] = b[-1] - t[-1] * dy_end
    # print(A, b)

    res = ga.tridiagonal_solver(A, b)
    dy[1:-1] = res
    # print(dy.shape)

    print("dy:", dy)
    for i in range(len(x)):
        plt.arrow(x[i], y[i], 0.3, 0.3*dy[i])

    gamma = (dy[:-1] + v * dy[1:] - (v + 1) * (dt_y / dt_x)) / (v * v - 1)
    delta = (v * dy[:-1] + dy[1:] - (v + 1) * (dt_y / dt_x)) / (v * v - 1)
    

    print(delta - gamma)

    A = dt_x * (beta + w * delta)
    B = alpha - beta + w * (gamma - delta)
    C = (gamma - delta * np.exp(-p * dt_x)) / (2 * (z - p))
    D = (delta * np.exp(p * dt_x) - gamma) / (2 * (z - p))

    return np.transpose(np.asarray([A, B, C, D]))

    # return np.transpose(np.asarray([alpha, beta, gamma, delta, p, z]))

"""
def uniform_tension_spline_generator(x, y, tar_xs, p, plotting=0, derives="2"):

    plt_x = []
    plt_y = []
    sorted(x)
    sorted(y)

    
    coeff = tension_spline_coeff_generator(x, y, p)
    print(coeff.shape)
    
    ret_ys = np.zeros_like(tar_xs)
    start_base = 0
    for i in range(len(tar_xs)):
        tar_x = tar_xs[i]
        assert tar_x >= x[0] and tar_x <= x[-1]
        for j in range(start_base, len(x) - 1):
            if tar_x >= x[j] and tar_x <= x[j + 1]:

                start_base = j
                dp = tar_x - x[j]
                dn = x[j + 1] - tar_x

                ret_ys[i] = dp * coeff[j][0] + dn * coeff[j][1]
                ret_ys[i] += coeff[j][2] * psi_func(dp, coeff[j][4], coeff[j][5])
                ret_ys[i] += coeff[j][3] * psi_func(dn, coeff[j][4], coeff[j][5])
                break

    return ret_ys
"""

def tension_spline_generator(x, y, tar_xs, p, plotting=0, derives="2"):

    plt_x = []
    plt_y = []
    sorted(x)
    sorted(y)

    
    coeff = tension_spline_coeff_generator(x, y, p)
    print(coeff.shape)
    
    ret_ys = np.zeros_like(tar_xs)
    start_base = 0
    for i in range(len(tar_xs)):
        tar_x = tar_xs[i]
        assert tar_x >= x[0] and tar_x <= x[-1]
        for j in range(start_base, len(x) - 1):
            if tar_x >= x[j] and tar_x <= x[j + 1]:

                start_base = j
                d = tar_x - x[j]

                ret_ys[i] = coeff[j][0] + coeff[j][1] * d + coeff[j][2] * np.exp(p[j] * d) + coeff[j][3] * np.exp(-p[j] * d)
                break

    return ret_ys

def test():

    x = np.array([0.0, 1.0, 1.5, 2.5, 4.0, 4.5, 5.5, 6.0, 8.0, 10.0])
    y = np.array([10.0, 8.0, 5.0, 4.0, 3.5, 3.41, 6.0, 7.1, 8.0, 8.5])

    # x = np.arange(10)
    # y = np.array([10,8,2,3,5,1,2,4,5,1])
    # y = x

    p = np.zeros(len(x) - 1, dtype="float64")
    p += 0.1
    # p = np.array([4, 4, 6, 12, 12, 0.001, 1, 4, 1, 1])

    tar_x = np.linspace(x[0], x[-1], 5000)
    tar_y = tension_spline_generator(x, y, tar_x, p)
    cubic_y = cubic_spline_interpolation.cubic_spline_generator(x, y, tar_x)
    # print(tar_y)

    plt.plot(tar_x, cubic_y)
    plt.plot(tar_x, tar_y)

    # plt.arrow(x, y, )
    plt.scatter(x, y)
    plt.show()
    plt.plot((tar_y[1:] - tar_y[:-1]) / (tar_x[1:] - tar_x[:-1]))
    plt.plot((cubic_y[1:] - cubic_y[:-1]) / (tar_x[1:] - tar_x[:-1]))
    plt.show()

test()

