import numpy as np
import matplotlib.pyplot as plt
import cubic_spline_interpolation
from tqdm import tqdm

def error_analysis(start, end, func, h, eval_number):
    
    x = np.arange(start, end, h)
    y = func(x)

    target_x = np.linspace(x[0], x[-1], eval_number)
    ground_truth = func(target_x)
    ret_y = cubic_spline_interpolation.cubic_spline_generator(x, y, target_x)
    return np.mean(np.abs(ground_truth - ret_y)), {"err_array": np.abs(ground_truth - ret_y), "x": x, "y": y, "tx": target_x, "ty": ret_y, "gt": ground_truth}

def test_sin_func(x):
    return x + np.sin(x)

def test_4rd_polynomial(x):
    return 2 * x + x ** 4

def test_runge_function(x):
    return 1 / (1 + 25 * x**2)

def error_for_cubic():

    start = -1
    end = 1
    err_runge, err_polynomial, err_sin, h_ls = [], [], [] ,[]
    
    for i in range(2, 25):
        
        h = 1 / (1.5 ** i)
        print("h={}".format(h))
        h_ls.append(h)
        err, d = error_analysis(start, end, test_runge_function, h, 1000)
        err_runge.append(err)
        err, d = error_analysis(start, end, test_4rd_polynomial, h, 1000)
        err_polynomial.append(err)
        err, d = error_analysis(start, end, test_sin_func, h, 1000)
        err_sin.append(err)
        
    plt.loglog(h_ls, err_runge,
               h_ls, err_polynomial,
               h_ls, err_sin,
               h_ls, [x**4 for x in h_ls], 'k-.')
    plt.legend(["Runge Function", "Polynomial", "Sin", "h^4"], loc="upper left")
    plt.ylabel("Err")
    plt.xlabel("h")
    # plt.loglog(h_ls, [x**4 for x in h_ls], c="red")
    plt.show()

error_for_cubic()