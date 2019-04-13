import numpy as np
import matplotlib.pyplot as plt
import general_arithmetic
import primative_interpolation
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

def baseline_3rd(x):
    return x ** 3

def baseline_2rd(x):
    return x ** 2

def error_convergence(interpolation_func, start, end, h_seq, funcs, names, name, derives="2", save=False):

    test_funcs = funcs[0]
    test_names = names[0]

    err_ls = [[] for i in range(len(test_funcs))]

    case = 0
    for h in h_seq:
        case += 1
        print("h={} case={}".format(h, case))
        for i in range(len(test_funcs)):
            err, d = error_analysis(start, end, test_funcs[i], interpolation_func, h, 1000, derives=derives)
            err_ls[i].append(err)
    
    plt.figure(figsize=(10,4))
    plt.title("Error Convergence for {}".format(name))

    for i in range(len(test_funcs)):
        plt.loglog(h_seq, err_ls[i], label=test_names[i])
    
    baseline_funcs = funcs[1]
    baseline_names = names[1]
    for i in range(len(baseline_funcs)):
        baseline = [baseline_funcs[i](j) for j in h_seq]
        plt.loglog(h_seq, baseline, 'k-.', label=baseline_names[i])

    plt.legend(loc="upper left")
    plt.ylabel("Err")
    plt.xlabel("h")
    if save:
        print("save to{}".format("./plots/Error Convergence for {}.png".format(name)))
        plt.savefig("./plots/Error Convergence for {}.png".format(name), dpi=200)
    # plt.loglog(h_ls, [x**4 for x in h_ls], c="red")
    plt.show()


def demo_gen():

    x = np.linspace(-1,1,20)
    y = 1 / (1 + 25 * x ** 2)

    tar_x = np.linspace(-1, 1, 400)
    mc_pchip_1d = monotonicity_preserving_pchip.mc_pchip_1d_generator(x, y, tar_x, derives='2')
    pchip_1d = piecewise_cubic_hermite_spline.pchip_1d_generator(x, y, tar_x, derives='2')
    global_interpolation = primative_interpolation.global_polynomial_interpolation(x, y, tar_x)
    gt = 1 / (1 + 25 * tar_x ** 2)

    plt.plot(
        x, y, ".k",
        tar_x, gt, "-.k",
        tar_x, global_interpolation,
        tar_x, pchip_1d, markersize=10
    )

    plt.legend(["Interpolant", "Runge Function", "PCHIP"])
    plt.show()

def visualization_plot():

    x = np.arange(10)
    y = np.array([2, 2, 2, 2, 10, 10, 10, 12, 6, 1])
    # x = np.linspace(-1,1,20)
    # y = 1 / (1 + 25 * x ** 2)
    tar_x = np.linspace(x[0], x[-1], 100)

    pchip_1d = piecewise_cubic_hermite_spline.pchip_1d_generator(x, y, tar_x, derives='2')
    mc_pchip_1d = monotonicity_preserving_pchip.mc_pchip_1d_generator(x, y, tar_x, derives='2')
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
    test_funcs_spline = [[test_sin_func, test_4rd_polynomial, test_runge_function], [baseline_4rd]]
    test_names_spline = [["Runge Function", "Order-4 Polynomial", "Sin"], ["h^4 (baseline)"]]
    error_convergence(cubic_spline_interpolation.cubic_spline_generator, -1, 1, h_sequence_spline, test_funcs_spline, test_names_spline, "Spline", save=True)

    h_sequence_pchip = [1 / (1.5 ** i) for i in range(2, 30)]
    test_funcs_pchip = [[test_sin_func, test_4rd_polynomial, test_runge_function], [baseline_3rd]]
    test_names_pchip = [["Runge Function", "Order-4 Polynomial", "Sin"], ["h^3 (baseline)"]]
    # error_convergence(piecewise_cubic_hermite_spline.pchip_1d_generator, -1, 1, h_sequence_pchip, test_funcs_pchip, test_names_pchip, "Pchip (Derivative Approx order 4)", derives="4", save=True)

    h_sequence_mc_pchip = [1 / (1.5 ** i) for i in range(2, 30)]
    test_funcs_mc_pchip = [[test_sin_func, test_4rd_polynomial, test_runge_function], [baseline_3rd]]#, baseline_3rd]]
    test_names_mc_pchip = [["Runge Function", "Order-4 Polynomial", "Sin"], ["h^3 (baseline)"]]#, "h^3 (baseline)"]]
    # error_convergence(monotonicity_preserving_pchip.mc_pchip_1d_generator, -1, 1, h_sequence_mc_pchip, test_funcs_mc_pchip, test_names_mc_pchip, "Monotonicity preserving Pchip (Derivative Approx order 4)", derives="4", save=True)

#visualization_plot()

convergence_analysis()

#demo_gen()