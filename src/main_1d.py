import numpy as np
import matplotlib.pyplot as plt
import general_arithmetic
import primative_interpolation
import cubic_spline_interpolation
import piecewise_cubic_hermite_spline
import monotonicity_preserving_pchip
import tension_exponential_spline
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
        plt.savefig("../plots/Error Convergence for {}.png".format(name), dpi=200)
    # plt.loglog(h_ls, [x**4 for x in h_ls], c="red")
    plt.show()


def demo_gen():

    x = np.linspace(-1,1,16)
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
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(["Interpolant", "Runge Function", "Global Polynomial Interpolation", "PCHIP"], fontsize=14)
    plt.show()

def visualization_auto_tension():

    #Spath Data
    y = np.array([8,7.5,2.6,2.5,2.3,2.7,2.8,7.6,8,8.1])
    x = np.array([1,2,3,3.5,3.8,6,7,8,9,10])
    #Jupiter Data
    y = np.array([1, 0.05, 0.01, 0, 0.2, 0.8, 0.9, 0.7, 0.8])
    x = np.array([0.42, 0.45, 0.55, 0.65,0.74,0.8,0.87,0.95,1.05])
    # Flux Data
    x = np.array([1,2,3,4,4.5,5.5,6.5,7.5])
    y = np.array([0, 0.005, 0.011, 0.017, 0.983, 0.994, 0.999, 1])
    tar_x = np.linspace(x[0], x[-1], 4000)

    c_i = 0
    p = -1

    plt.figure(figsize=(12,6))
    plt.rcParams.update({"axes.facecolor": "#bbbbbb"})
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    # plt.style.use('dark_background')
    iters = [1,2,4,8,16,32,64]
    iters = list(range(0, 20, 2))
    iters = [0,1,2,4,8,16,32,64,128]
    iters = [0,1,2,3,5,8,11,15,20,30]
    cnt = 0

    cmap = plt.cm.jet_r

    for it in iters:
        color_t = cmap(cnt / len(iters) * 0.7 + 0.2)
        cnt += 1
        print("Color:", color_t)
        tension_spline = tension_exponential_spline.tension_spline_generator(x, y, tar_x, p, iters=it)
        plt.plot(tar_x, tension_spline, lw=2, label="Iter {}".format(it), color=color_t)
    
    plt.scatter(x, y, c="w", s=20, label="Data", zorder=100)
    leg = plt.legend(fontsize=14, facecolor="#666666", loc="lower right")
    for text in leg.get_texts():
        plt.setp(text, color = 'w')
    
    plt.savefig("../plots/tension_iter_flux.png", bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()

def visualization_universal_tension():

    #Spath Data
    y = np.array([8,7.5,2.6,2.5,2.3,2.7,2.8,7.6,8,8.1])
    x = np.array([1,2,3,3.5,3.8,6,7,8,9,10])
    # Flux Data
    x = np.array([1,2,3,4,4.5,5.5,6.5,7.5])
    y = np.array([0, 0.005, 0.011, 0.017, 0.983, 0.994, 0.999, 1])
    #Jupiter Data
    y = np.array([1, 0.05, 0.01, 0, 0.2, 0.8, 0.9, 0.7, 0.8])
    x = np.array([0.42, 0.45, 0.55, 0.65,0.74,0.8,0.87,0.95,1.05])

    x = np.array([1,2,3,4,5,6,7])
    y = np.array([2,3,8,7,-1,2,1])

    tar_x = np.linspace(x[0], x[-1], 4000)

    c_i = 0
    p = -1

    plt.figure(figsize=(12,6))
    plt.rcParams.update({"axes.facecolor": "#bbbbbb"})
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    # plt.style.use('dark_background')

    tensions = [0.001, 16, 32, 64, 128, 256, 512]
    tensions = [0.001,2,4,8,16,32,128]
    cnt = 0

    cmap = plt.cm.jet_r

    for tension in tensions:
        color_t = cmap(cnt / len(tensions) * 0.7 + 0.2)
        cnt += 1
        
        p = np.zeros(len(x) - 1)
        p += tension

        print("Color:", color_t)
        tension_spline = tension_exponential_spline.tension_spline_generator(x, y, tar_x, p, iters=1)
        if tension < 0.01:
            plt.plot(tar_x, tension_spline, lw=2, label="P=0", color=color_t)
        else:
            plt.plot(tar_x, tension_spline, lw=2, label="P={}".format(tension), color=color_t)
    
    plt.scatter(x, y, c="w", s=20, label="Data", zorder=100)
    leg = plt.legend(fontsize=14, facecolor="#666666", loc="upper right")
    for text in leg.get_texts():
        plt.setp(text, color = 'w')
    
    plt.savefig("../plots/tension_universal_demo.png", bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()

def visualization_plot():


    #Spath Data
    y = np.array([8,7.5,2.6,2.5,2.3,2.7,2.8,7.6,8,8.1])
    x = np.array([1,2,3,3.5,3.8,6,7,8,9,10])
    #Jupiter Data
    y = np.array([1, 0.05, 0.01, 0, 0.2, 0.8, 0.9, 0.7, 0.8])
    x = np.array([0.42, 0.45, 0.55, 0.65,0.74,0.8,0.87,0.95,1.05])

    x = np.array([1,2,3,4,5,6,7])
    y = np.array([2,3,8,7,-1,2,1])
    # Flux Data
    x = np.array([1,2,3,4,4.5,5.5,6.5,7.5])
    y = np.array([0, 0.005, 0.011, 0.017, 0.983, 0.994, 0.999, 1])
    #x = np.array([0.0, 1.0, 1.5, 2.5, 4.0, 4.5, 5.5, 6.0, 8.0, 10.0])
    #y = np.array([10.0, 8.0, 5.0, 4.0, 3.5, 3.41, 6.0, 7.1, 8.0, 8.5])
    
    # x = np.linspace(-1,1,20)
    # y = 1 / (1 + 25 * x ** 2)
    tar_x = np.linspace(x[0], x[-1], 4000)
    p = np.zeros(len(x) - 1)
    p += 10
    p = -1

    pchip_1d_2rd = piecewise_cubic_hermite_spline.pchip_1d_generator(x, y, tar_x, derives='2')
    pchip_1d_4rd = piecewise_cubic_hermite_spline.pchip_1d_generator(x, y, tar_x, derives='4')
    mc_pchip_1d_2rd = monotonicity_preserving_pchip.mc_pchip_1d_generator(x, y, tar_x, derives='2')
    mc_pchip_1d_4rd = monotonicity_preserving_pchip.mc_pchip_1d_generator(x, y, tar_x, derives='4')
    cubic_1d = cubic_spline_interpolation.cubic_spline_generator(x, y, tar_x)
    tension_spline = tension_exponential_spline.tension_spline_generator(x, y, tar_x, p, iters=10)

    plt.figure(figsize=(12,6))
    plt.rcParams.update({"axes.facecolor": "#bbbbbb"})
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    cmap = plt.cm.hsv

    plt.plot(x, y, lw=3, linestyle="-.", label="Linear Spline", color="b")
    # plt.plot(tar_x, pchip_1d_2rd, lw=3, label="PCHIP-2", c=cmap(0.02))
    # plt.plot(tar_x, pchip_1d_4rd, lw=3, label="PCHIP-4", c=cmap(0.2))
    plt.plot(tar_x, cubic_1d, lw=3, label="Cubic Spline", c=cmap(0.4))
    #plt.plot(tar_x, mc_pchip_1d_2rd, lw=3, label="MC PCHIP-2", c="orange")
    # plt.plot(tar_x, mc_pchip_1d_4rd, lw=2, label="MC PCHIP-4")
    #plt.plot(tar_x, tension_spline, lw=2, label="tension")
    plt.scatter(x, y, c="w", s=20, label="Data", zorder=100)
    # plt.legend(["data", "pchip", "mc_pchip", "natural cubic"])
    leg = plt.legend(fontsize=14, facecolor="#666666", loc="lower right")
    for text in leg.get_texts():
        plt.setp(text, color = 'w')
    plt.savefig("../plots/spline_flux.png", bbox_inches='tight', dpi=300)

    plt.close()


    """
    plt.figure(figsize=(12,6))
    plt.rcParams.update({"axes.facecolor": "#bbbbbb"})
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.subplot(211)
    
    dpchip_2 = (pchip_1d_2rd[1:] - pchip_1d_2rd[:-1]) / (tar_x[1:] - tar_x[:-1]) 
    dpchip_4 = (pchip_1d_4rd[1:] - pchip_1d_4rd[:-1]) / (tar_x[1:] - tar_x[:-1])
    dcubic = (cubic_1d[1:] - cubic_1d[:-1]) / (tar_x[1:] - tar_x[:-1])
    
    plt.plot(tar_x[:-1], dpchip_2, lw=3, label="PCHIP-2'", c=cmap(0.02))
    # plt.plot(tar_x[:-1], dpchip_4, lw=3, label="PCHIP-4'", c=cmap(0.2))
    plt.plot(tar_x[:-1], dcubic, lw=3, label="Cubic Spline'", c=cmap(0.4))

    leg = plt.legend(fontsize=14, facecolor="#666666", loc="upper right")
    for text in leg.get_texts():
        plt.setp(text, color = 'w')

    plt.subplot(212)
    
    ddpchip_2 = (dpchip_2[1:] - dpchip_2[:-1]) / (tar_x[1:-1] - tar_x[:-2]) 
    ddpchip_4 = (dpchip_4[1:] - dpchip_4[:-1]) / (tar_x[1:-1] - tar_x[:-2])
    ddcubic = (dcubic[1:] - dcubic[:-1]) / (tar_x[1:-1] - tar_x[:-2])
    
    plt.plot(tar_x[1:-1], ddpchip_2, lw=3, label="PCHIP-2''", c=cmap(0.02))
    # plt.plot(tar_x[1:-1], ddpchip_4, lw=3, label="PCHIP-4''", c=cmap(0.2))
    plt.plot(tar_x[1:-1], ddcubic, lw=3, label="Cubic Spline''", c=cmap(0.4))

    leg = plt.legend(fontsize=14, facecolor="#666666", loc="upper right")
    for text in leg.get_texts():
        plt.setp(text, color = 'w')
    plt.savefig("../plots/d_cspline_demo.png", bbox_inches='tight', dpi=300)
    plt.close()
    """



def convergence_analysis():

    h_sequence_spline = [1 / (1.5 ** i) for i in range(2, 25)]
    test_funcs_spline = [[test_sin_func, test_4rd_polynomial, test_runge_function], [baseline_4rd]]
    test_names_spline = [["Runge Function", "Order-4 Polynomial", "Sin"], ["h^4 (baseline)"]]
    # error_convergence(cubic_spline_interpolation.cubic_spline_generator, -1, 1, h_sequence_spline, test_funcs_spline, test_names_spline, "Spline", save=True)

    h_sequence_pchip = [1 / (1.5 ** i) for i in range(2, 32)]
    test_funcs_pchip = [[test_sin_func, test_4rd_polynomial, test_runge_function], [baseline_3rd]]
    test_names_pchip = [["Runge Function", "Order-4 Polynomial", "Sin"], ["h^3 (baseline)"]]
    error_convergence(piecewise_cubic_hermite_spline.pchip_1d_generator, -1, 1, h_sequence_pchip, test_funcs_pchip, test_names_pchip, "Pchip (Derivative Approx order 2)", derives="2", save=True)

    h_sequence_mc_pchip = [1 / (1.5 ** i) for i in range(2, 30)]
    test_funcs_mc_pchip = [[test_sin_func, test_4rd_polynomial, test_runge_function], [baseline_3rd]]#, baseline_4rd]]
    test_names_mc_pchip = [["Runge Function", "Order-4 Polynomial", "Sin"], ["h^3 (baseline)"]]#, "h^4 (baseline)"]]
    # error_convergence(monotonicity_preserving_pchip.mc_pchip_1d_generator, -1, 1, h_sequence_mc_pchip, test_funcs_mc_pchip, test_names_mc_pchip, "Monotonicity preserving Pchip (Derivative Approx order 2)", derives="2", save=True)

visualization_plot()
# visualization_auto_tension()
# visualization_universal_tension()
# convergence_analysis()

# demo_gen()