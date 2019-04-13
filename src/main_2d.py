import bicubic_spline
import cubic_conv_interpolation
import primative_interpolation
import piecewise_cubic_hermite_spline
import monotonicity_preserving_pchip
import general_arithmetic
import cubic_spline_interpolation
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_1(fig, p1, p2, mat, name, color_map, scale, imx, imy, mat1_override=None):

    if not isinstance(mat1_override, np.ndarray):
        mat1 = mat
        ax = fig.add_subplot(p1)
        ax.set_title(name)
        ax.imshow(mat1, origin="lower", cmap=color_map, extent=[0, imx-1, 0, imy-1])

    else:
        mat1 = mat1_override
        ax = fig.add_subplot(p1)
        ax.imshow(mat1, origin="lower", cmap=color_map) 
        ax.set_title(name)
        ax.set_xlim(0, imx-1)
        ax.set_ylim(0, imy-1)
    

    ax = fig.add_subplot(p2, projection="3d")
    X = np.linspace(0, imx-1, (imx-1) * scale + 1)
    Y = np.linspace(0, imy-1, (imy-1) * scale + 1)
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, mat, rstride=1, cstride=1, cmap=color_map, edgecolor='black', linewidth=0.3)
    # ax.plot_wireframe(X, Y, bilinear_10 , cmap=color_map)
    ax.invert_xaxis()
    ax.set_title(name)

def plot_diff_1(mat1, mat2, name1, name2, color_map):
    
    plt.figure(figsize=(15,4))
    plt.subplot(131)
    plt.imshow(mat1, cmap=color_map)
    plt.title(name1)
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(mat1, cmap=color_map)
    plt.title(name2)
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(mat1 - mat2, cmap=color_map)
    plt.title("{} - {}".format(name1, name2))
    plt.colorbar()
    plt.show()

def general_comparison():

    imx, imy = 8, 8
    color_map = "jet"
    scale = 8
    rand_mat = np.random.random((imx, imy))
    rand_mat_slope = np.random.random((imx, imy)) + np.concatenate([np.expand_dims(np.linspace(i, 0.8 + i, imx), axis=1) for i in np.linspace(0, 0.8, imy)], axis=1)
    checker_board = np.array(([0,1]*(int(imx/2))+[1,0]*(int(imx/2)))*(int(imx/2))).reshape((imx,imx))
    runge_2d = general_arithmetic.runge_2d(imx, imy)
    
    imx, imy = 8, 8
    square_peak = np.zeros((imx, imy))
    square_peak[2:-2,2:-2] = np.ones((imx - 4, imy - 4))
    
    mat = runge_2d
    print(mat)

    
    # fig = plt.figure()
    fig = plt.figure(figsize=(17,6))

    # Nearest Neighbor
    # nearest_neighbor_10 = primative_interpolation.nearest_neighbor_helper(mat, scale)
    # plot_1(fig, 241, 245, nearest_neighbor_10, "Nearest Neighbor", color_map, scale, imx, imy, mat1_override=mat)

    # Bilinear
    bilinear_10 = primative_interpolation.bilinear_helper(mat, scale)
    plot_1(fig, 241, 245, bilinear_10, "Bilinear", color_map, scale, imx, imy)

    # Bicubic Hermite
    # bicubic_10 = bicubic_spline.bicubic_spline_helper(mat, scale)
    # plot_1(fig, 243, 247, bicubic_10, "Bicubic", color_map, scale, imx, imy)

    # 2d mc_pchip Spline
    mc_pchip_2d = monotonicity_preserving_pchip.mc_pchip_base_helper(mat, scale)
    plot_1(fig, 242, 246, mc_pchip_2d, "Monotonicity Preserving Pchip", color_map, scale, imx, imy)


    # 2d pchip Spline
    pchip_2d = piecewise_cubic_hermite_spline.pchip_base_helper(mat, scale)
    plot_1(fig, 243, 247, pchip_2d, "2D Pchip", color_map, scale, imx, imy)
    
    # 2d cubic spline
    cubic_2d = cubic_spline_interpolation.cubic_1d_to_2d(mat, scale)
    plot_1(fig, 244, 248, cubic_2d, "2D cubic", color_map, scale, imx, imy)

    plt.savefig('test.png', dpi=100)
    plt.show()


if __name__ == "__main__":    
    general_comparison()
    

    
