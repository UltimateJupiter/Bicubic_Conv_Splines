import numpy as np
import matplotlib.pyplot as plt
import cubic_spline_interpolation
import piecewise_cubic_hermite_spline
import monotonicity_preserving_pchip
from tqdm import tqdm

def error_analysis(start, end, func, interp_func, h, eval_number, derives=None):
    
    x = np.arange(start, end, h)
    y = func(x)

    target_x = np.linspace(x[0], x[-1], eval_number)
    ground_truth = func(target_x)
    ret_y = interp_func(x, y, target_x, derives=derives)
    return np.mean(np.abs(ground_truth - ret_y)), {"err_array": np.abs(ground_truth - ret_y), "x": x, "y": y, "tx": target_x, "ty": ret_y, "gt": ground_truth}

def test_sin_func(x):
    return x + np.sin(x)

def test_4rd_polynomial(x):
    return 2 * x + x ** 4

def test_runge_function(x):
    return 1 / (1 + 25 * x**2)

def baseline_4rd(x):
    return x ** 4

def baseline_2rd(x):
    return x ** 2

def error_convergence(interpolation_func, start, end, h_seq, test_funcs, test_names, name, derives="2"):

    baseline = [test_funcs[-1](i) for i in h_seq]
    err_ls = [[] for i in range(len(test_funcs) - 1)]

    case = 0
    for h in h_seq:
        case += 1
        print("h={} case={}".format(h, case))
        for i in range(len(test_funcs) - 1):
            err, d = error_analysis(start, end, test_funcs[i], interpolation_func, h, 1000, derives=derives)
            err_ls[i].append(err)
    
    plt.figure(figsize=(10,4))
    plt.title("Convergence Rate for {}".format(name))

    for i in range(len(test_funcs) - 1):
        plt.loglog(h_seq, err_ls[i], label=test_names[i])
    
    plt.loglog(h_seq, baseline, 'k-.', label=test_names[-1])
    plt.legend(loc="upper left")
    plt.ylabel("Err")
    plt.xlabel("h")
    # plt.loglog(h_ls, [x**4 for x in h_ls], c="red")
    plt.show()


def visualization_plot():

    x = np.arange(10)
    y = np.array([1,2,3,5,13,12,11,40,39,35])
    tar_x = np.linspace(x[0], x[-1], 100)

    pchip_1d = piecewise_cubic_hermite_spline.pchip_1d_generator(x, y, tar_x, derives='2')
    mc_pchip_1d = monotonicity_preserving_pchip.mc_pchip_1d_generator(x, y, tar_x, derives='4')
    cubic_1d = cubic_spline_interpolation.cubic_spline_generator(x, y, tar_x)

    plt.figure(figsize=(12,6))
    plt.plot(x, y, ".k",
             tar_x, pchip_1d,
             tar_x, mc_pchip_1d,
             tar_x, cubic_1d)
    
    plt.legend(["data", "pchip", "mc_pchip", "natural cubic"])
    plt.show()


def convergence_analysis():

    h_sequence_spline = [1 / (1.5 ** i) for i in range(2, 25)]
    test_funcs_spline = [test_sin_func, test_4rd_polynomial, test_runge_function, baseline_4rd]
    test_names_spline = ["Runge Function", "Order-4 Polynomial", "Sin", "h^4 (baseline)"]
    # error_convergence(cubic_spline_interpolation.cubic_spline_generator, -1, 1, h_sequence_spline, test_funcs_spline, test_names_spline, "Spline")

    h_sequence_spline = [1 / (2 ** i) for i in range(2, 25)]
    test_funcs_pchip = [test_sin_func, test_4rd_polynomial, test_runge_function, baseline_2rd]
    test_names_pchip = ["Runge Function", "Order-4 Polynomial", "Sin", "h^2 (baseline)"]
    error_convergence(piecewise_cubic_hermite_spline.pchip_1d_generator, -1, 1, h_sequence_spline, test_funcs_pchip, test_names_pchip, "Pchip", derives="2")

visualization_plot()

#convergence_analysis()