import bicubic_spline
import bicubic_conv_interpolation
import primative_interpolation
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


if __name__ == "__main__":
    
    imx, imy = 8, 8
    color_map = "jet"
    scale = 8
    rand_mat = np.random.random((imx, imy))
    rand_mat = np.random.random((imx, imy))*3 + np.concatenate([np.expand_dims(np.linspace(i, 0.8 + i, imx), axis=1) for i in np.linspace(0, 0.8, imy)], axis=1)
    print(rand_mat)
    
    # fig = plt.figure()
    fig = plt.figure(figsize=(13,7))
    
    ax = fig.add_subplot(231)
    ax.imshow(rand_mat, origin="lower", cmap=color_map) 
    ax.set_title("Nearest Neighbor")
    ax.set_xlim(0, imx-1)
    ax.set_ylim(0, imy-1)

    nearest_neighbor_10 = primative_interpolation.nearest_neighbor_helper(rand_mat, scale)
    ax = fig.add_subplot(234, projection="3d")
    X = np.linspace(0, imx-1, (imx-1) * scale + 1)
    Y = np.linspace(0, imy-1, (imy-1) * scale + 1)
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, nearest_neighbor_10, rstride=1, cstride=1, cmap=color_map, edgecolor='black', linewidth=0.3)
    # ax.plot_wireframe(X, Y, nearest_neighbor_10, cmap=color_map)
    ax.invert_xaxis()
    ax.set_title('Nearest Neighbor')


    # Bilinear
    bilinear_10 = primative_interpolation.bilinear_helper(rand_mat, scale)

    ax = fig.add_subplot(232)
    ax.set_title("Bilinear")
    ax.imshow(bilinear_10, origin="lower", cmap=color_map, extent=[0, imx-1, 0, imy-1])

    ax = fig.add_subplot(235, projection="3d")
    X = np.linspace(0, imx-1, (imx-1) * scale + 1)
    Y = np.linspace(0, imy-1, (imy-1) * scale + 1)
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, bilinear_10, rstride=1, cstride=1, cmap=color_map, edgecolor='black', linewidth=0.3)
    # ax.plot_wireframe(X, Y, bilinear_10 , cmap=color_map)
    ax.invert_xaxis()
    ax.set_title('Bilinear')

    # Bicubic Spline
    bicubic_10 = bicubic_spline.bicubic_spline_helper(rand_mat, scale)

    ax = fig.add_subplot(233)
    ax.set_title("Bicubic")
    ax.imshow(bicubic_10, origin="lower", cmap=color_map, extent=[0, imx-1, 0, imy-1])

    ax = fig.add_subplot(236, projection="3d")
    X = np.linspace(0, imx-1, (imx-1) * scale + 1)
    Y = np.linspace(0, imy-1, (imy-1) * scale + 1)
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, bicubic_10, rstride=1, cstride=1, cmap=color_map, edgecolor='black', linewidth=0.3)
    # ax.plot_wireframe(X, Y, bicubic_10, cmap=color_map)
    ax.invert_xaxis()
    ax.set_title('Bicubic')

    plt.savefig('test.png', dpi=100)
    plt.show()