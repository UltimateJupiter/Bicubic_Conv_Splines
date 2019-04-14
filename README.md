# Bicubic Splines Like Methods

by Xingyu Zhu

<xingyu.zhu@duke.edu>

This repo contains a set of codes for Splines, PCHIPs (Piecewise Cubic Hermite Interpolation Polynomial), Monotonicity preserving PCHIPS, tension splines, and their 2D derivatives.

Such algorithms are useful in performing numerical approximations in a 2-D large data grid, and are important in computational graphics such as smooth 2-D image resampling.

This project is done for the final project for MATH361S

### Progress so far

Finished Bilinear & Nearest as well as a basic main.py (19/04/03)

Finished Bicubic (19/04/04)

Finished 1D pchip & framework to convert 1D interpolation to 2D (19/04/05)

Update the main plotting program for 2D demos

Finished 1D cubic's convergence test, working on monotonic cubic (19/04/11)

Finished monotonicity preserving pchip, working on tension based cubic spline(19/04/13)

---
Convergence test for natural cubic spline
![Image0](https://raw.githubusercontent.com/UltimateJupiter/Bicubic_Conv_Splines/master/plots/Error%20Convergence%20for%20Spline.png)

Convergence test for pchip (centered difference 2nd-order)
![Image01](https://raw.githubusercontent.com/UltimateJupiter/Bicubic_Conv_Splines/master/plots/Error%20Convergence%20for%20Pchip%20(Derivative%20Approx%20order%202).png)

Convergence test for pchip (centered difference 4nd-order)
![Image01](https://raw.githubusercontent.com/UltimateJupiter/Bicubic_Conv_Splines/master/plots/Error%20Convergence%20for%20Pchip%20(Derivative%20Approx%20order%204).png)

Convergence test for mc_pchip (centered difference 2nd-order)
![Image01](https://raw.githubusercontent.com/UltimateJupiter/Bicubic_Conv_Splines/master/plots/Error%20Convergence%20for%20Monotonicity%20preserving%20Pchip%20(Derivative%20Approx%20order%202).png)

Convergence test for mc_pchip (centered difference 4nd-order)
![Image01](https://raw.githubusercontent.com/UltimateJupiter/Bicubic_Conv_Splines/master/plots/Error%20Convergence%20for%20Monotonicity%20preserving%20Pchip%20(Derivative%20Approx%20order%204).png)

2D Interpolation for random matrix
![Image1](https://i.ibb.co/gjLVRyd/overall.png)

2D Interpolation for a 2D Runge Function
![Image2](https://i.ibb.co/fxhVH3R/runge.png)

---

