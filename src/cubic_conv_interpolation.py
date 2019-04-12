import numpy as np
import matplotlib.pyplot as plt
from general_arithmetic import array_padding_1D

def cubic_conv_1d(L, scale):

    X = array_padding_1D(L)
    print(X)

    coeffs = np.zeros([4, len(L) - 1])
    for i in range(len(L) - 1):

        l = X[i: i+4]
        print(l)
        


text = np.random.random([8])
print(text)
cubic_conv_1d(text, 10)