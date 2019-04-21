import numpy as np
import general_arithmetic as ga
import matplotlib.pyplot as plt
import cubic_spline_interpolation

def oneshot_tension_spline_coeff_generator(x, y, p):

    """
    Inputs:
        x: x data
        y: y data
        p: tension factors
    """

    N = len(x)

    assert x.shape == y.shape
    assert len(x) - 1 == len(p)

    h = x[1:] - x[:-1]
    f = y
    dy = ga.second_order_finitedif_derivative_approx(x, y)
    dy_head, dy_end = dy[0], dy[-1]
    slp = (f[1:] - f[:-1]) / h

    b = np.zeros_like(x, "float64")
    b[0] = (f[1] - f[0]) / h[0] - dy_head
    b[-1] = dy_end - (f[-1] - f[-2]) / h[-1]
    b[1: -1] = slp[1:] - slp[:-1]

    s = ga.sinh(p * h)
    c = ga.cosh(p * h)

    e = ((1 / h) - (p / s)) / (p * p)
    d = ((p * c / s) - (1 / h)) / (p * p)
        
    A = np.zeros((3, len(x)))

    g_e = e / (d ** 2 - e ** 2)
    g_d = d / (d ** 2 - e ** 2)

    A[0][1:] = g_e
    A[2][:-1] = g_e
    A[1][:-1] += g_d
    A[1][1:] += g_d

    b_part = (1 / (d - e)) * slp
    b_solve = np.zeros_like(b, "float64")
    b_solve[1:] += b_part
    b_solve[:-1] += b_part

    A = np.transpose(A)
    res = ga.tridiagonal_solver(A, b_solve)

    B, C, D = None, None, None

    return [res, g_d, g_e, b_part, h, s, p, B, C, D]

def tension_spline_generator(x, y, tar_xs, p=-1, plotting=0, derives="2", iters=100):

    plt_x = []
    plt_y = []
    sorted(x)
    sorted(y)

    if not isinstance(p, np.ndarray):
        coeff = iterative_tension_spline_coeff_generator(x, y, iters)

    else:
        coeff = oneshot_tension_spline_coeff_generator(x, y, p)

    t, g_d, g_e, b_part, h, s, p, B, C, D = coeff

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

                val = y[j] * dn / h[j]
                val += y[j + 1] * dp / h[j]

                val -= b_part[j] * ((ga.sinh(p[j] * dp) - ga.sinh(p[j] * dn)) / (p[j] ** 2 * s[j]) + (dn - dp) / (p[j] ** 2 * h[j]))
                val += t[j] * (g_e[j] * (ga.sinh(p[j] * dp) / (p[j] ** 2 * s[j]) - dp / (p[j] ** 2 * h[j])) - g_d[j] * (ga.sinh(p[j] * dn) / (p[j] ** 2 * s[j]) - dn / (p[j] ** 2 * h[j])))
                val += t[j + 1] * (g_d[j] * (ga.sinh(p[j] * dp) / (p[j] ** 2 * s[j]) - dp / (p[j] ** 2 * h[j])) - g_e[j] * (ga.sinh(p[j] * dn) / (p[j] ** 2 * s[j]) - dn / (p[j] ** 2 * h[j])))

                ret_ys[i] = val
                break
    
    return ret_ys

def iterative_tension_spline_coeff_generator(x, y, iters=100):

    """
    Inputs:
        x: x data
        y: y data
        p: tension factors
    """

    N = len(x)
    assert x.shape == y.shape

    h = x[1:] - x[:-1]
    f = y
    dy = ga.fourth_order_finitedif_derivative_approx(x, y)
    dy_head, dy_end = dy[0], dy[-1]
    slp = (f[1:] - f[:-1]) / h

    b = np.zeros_like(x, "float64")
    b[0] = (f[1] - f[0]) / h[0] - dy_head
    b[-1] = dy_end - (f[-1] - f[-2]) / h[-1]
    b[1: -1] = slp[1:] - slp[:-1]

    m = np.zeros(len(x) + 1, "float64")
    m[0], m[-1] = dy_head, dy_end
    m[1: -1] = slp

    p_raw = np.zeros(len(x) - 1)
    p_raw += 0.4
    p = p_raw

    s = ga.sinh(p * h)
    c = ga.cosh(p * h)
    e = ((1 / h) - (p / s)) / (p * p)
    d = ((p * c / s) - (1 / h)) / (p * p)

    B, C, D = None, None, None

    for it in range(iters):
        
        s = ga.sinh(p * h)
        c = ga.cosh(p * h)
        e = ((1 / h) - (p / s)) / (p * p)
        d = ((p * c / s) - (1 / h)) / (p * p)
        A = np.zeros((3, len(x)))

        g_e = e / (d ** 2 - e ** 2)
        g_d = d / (d ** 2 - e ** 2)

        A[0][1:] = g_e
        A[2][:-1] = g_e
        A[1][:-1] += g_d
        A[1][1:] += g_d

        b_part = (1 / (d - e)) * slp
        b_solve = np.zeros_like(b, "float64")
        b_solve[1:] += b_part
        b_solve[:-1] += b_part

        A = np.transpose(A)
        t = ga.tridiagonal_solver(A, b_solve, vis=False)

        B, C, D = t2BCD(y, h, s, p, t, e, d, x)
        dt1 = t
        dt2 = BCD2ddt(B, C, D, p, x)

        p = tension_monotone_update(p, B, C, D, m, h, dt1, dt2)
        
    
    s = ga.sinh(p * h)
    c = ga.cosh(p * h)
    A = np.zeros((3, len(x)))

    print("Size:{}".format(len(x)))

    g_e = e / (d ** 2 - e ** 2)
    g_d = d / (d ** 2 - e ** 2)

    A[0][1:] = g_e
    A[2][:-1] = g_e
    A[1][:-1] += g_d
    A[1][1:] += g_d

    b_part = (1 / (d - e)) * slp
    b_solve = np.zeros_like(b, "float64")
    b_solve[1:] += b_part
    b_solve[:-1] += b_part

    A = np.transpose(A)
    res = ga.tridiagonal_solver(A, b_solve, vis=True)

    return [res, g_d, g_e, b_part, h, s, p, B, C, D]

def tension_monotone_update(p, B, C, D, m, h, dt1, dt2, stepsize=0.0005):

    """
    Form constraint to p on monotonic intervals
    """
    eps = 1
    lim = 0
    p_new = np.copy(p)
    p_update = np.zeros_like(p, "float64")
    print(dt1, dt2)

    for i in range(1, len(B)-1):
        k = i + 1 # index for m
        if m[k - 1] * m[k] < 0 or m[k + 1] * m[k] < 0:
            print("Pass:{}".format(i))
            continue
        # Monotone at i
        if dt1[i] * m[k] < 0:
            print("1:{}".format(i))

            if p_new[i - 1] > lim:
                tmp = (1 + p[i - 1] * h[i - 1]) / (p[i - 1] * h[i - 1]) * 2 * (max(abs(dt2[i - 1]), abs(dt2[i]), abs(dt2[i + 1]))) / abs(m[k - 1] + m[k]) + eps
                p_new[i - 1] = max(tmp, p_new[i - 1])
            
            if p_new[i] > lim:
                tmp = (1 + p[i] * h[i]) / (p[i] * h[i]) * 2 * (max(abs(dt2[i - 1]), abs(dt2[i]), abs(dt2[i + 1]))) / abs(m[k - 1] + m[k]) + eps
                p_new[i] = max(tmp, p_new[i])
            
        if dt1[i + 1] * m[k + 1] < 0:
            print("2:{}".format(i))
            if p_new[i] > lim:
                tmp = (1 + p[i] * h[i]) / (p[i] * h[i]) * 2 * (max(abs(dt2[i]), abs(dt2[i + 1]), abs(dt2[i + 2]))) / abs(m[k + 1] + m[k]) + eps
                p_new[i] = max(tmp, p_new[i])

            if p_new[i + 1] > lim:
                tmp = (1 + p[i + 1] * h[i + 1]) / (p[i + 1] * h[i + 1]) * 2 * (max(abs(dt2[i]), abs(dt2[i + 1]), abs(dt2[i + 2]))) / abs(m[k + 1] + m[k]) + eps
                p_new[i + 1] = max(tmp, p_new[i + 1])

        if dt1[i + 1] * m[k + 1] >= 0 and dt1[i] * m[k] >= 0:
            print("3:{}".format(i))
            if C[i] > 0 and D[i] < 0:
                if m[k] <= 0:
                    p_new[i] = max(p[i], p_new[i])
                else:
                    tmp = -B[i] / (2 * np.sqrt(-C[i] * D[i])) + eps
                    print(tmp)
                    p_new[i] = max(p_new[i], tmp)
            
            if C[i] < 0 and D[i] > 0:
                if m[k] >= 0:
                    p_new[i] = max(p[i], p_new[i])
                else:
                    tmp = B[i] / (2 * np.sqrt(-C[i] * D[i])) + eps
                    print(tmp)
                    p_new[i] = max(p_new[i], tmp)

    for i in range(len(p)):
        p_update[i] = p[i] + stepsize * (p_new[i] - p[i])
    
    print("P new:", p_new)
    # print("B:{}\nC:{}\nD:{}\ndt2:{}\n".format(B, C, D, dt2))
    return p_update
    

def t2BCD(y, h, s, p, t, e, d, x):

    B = np.zeros_like(h, "float64")
    C = np.zeros_like(h, "float64")
    D = np.zeros_like(h, "float64")

    for i in range(len(h)):

        hp2e_d = h[i] * p[i] ** 2 * (e[i] - d[i])
        e2_d2p2 = (p[i] ** 2) * (e[i] ** 2 - d[i] ** 2)
        base_1 = ((y[i + 1] - y[i]) / (hp2e_d * s[i]) - (t[i] * e[i] + t[i + 1] * d[i]) / (e2_d2p2 * s[i]))
        base_2 = ((y[i] - y[i + 1]) / (hp2e_d * s[i]) + (t[i] * d[i] + t[i + 1] * e[i]) / (e2_d2p2 * s[i]))

        B[i] = (y[i + 1] - y[i]) / h[i] - (2 * (y[i + 1] - y[i]) / (hp2e_d * h[i])) + ((t[i + 1] + t[i]) * (d[i] + e[i]) / (e2_d2p2 * h[i]))
        C[i] = (1 / (2 * np.exp(p[i] * x[i]))) * base_1 - (1 / (2 * np.exp(p[i] * x[i + 1]))) * base_2
        D[i] = -(np.exp(p[i] * x[i]) / 2) * base_1 + (np.exp(p[i] * x[i + 1]) / 2) * base_2

    return B, C, D

def BCD2ddt(B, C, D, p, x):

    t2 = np.zeros(len(B) + 1, "float64")
    for i in range(len(B)):
        t2[i] = C[i] * np.exp(p[i] * x[i]) * (p[i] ** 2) + D[i] * np.exp(- p[i] * x[i]) * (p[i] ** 2)
    return t2

def test():

    x = np.array([0.0, 1.0, 1.5, 2.5, 4.0, 4.5, 5.5, 6.0, 8.0, 10.0])
    y = np.array([10.0, 8.0, 5.0, 4.0, 3.5, 3.41, 6.0, 7.1, 8.0, 8.5])

    x = np.array([0.0, 1.0, 1.5, 2.5, 4.0, 4.5, 5.5, 6.0, 8.0, 10.0, 12.0, 12.5, 13.5, 14.5])
    y = np.array([10.0, 8.0, 5.0, 4.0, 3.5, 3.41, 6.0, 7.1, 8.0, 8.5, 8.2, 6.2, 4.1, 2.4])

    x = np.arange(10)
    y = np.array([0,0,0.01,0.03,0.07,0.93,0.97,0.99,1,1])

    # x = np.arange(10)
    # y = np.array([10,8,2,3,5,1,2,4,5,1])
    # y = x

    p = np.array([4, 4, 6, 12, 12, 0.001, 1, 4, 1], dtype="float64")
    p = np.zeros(len(x) - 1, dtype="float64")
    p += 0.001

    p = -1

    tar_x = np.linspace(x[0], x[-1], 5000)
    tar_y = tension_spline_generator(x, y, tar_x, p, iters=5)
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
    return

# test()
