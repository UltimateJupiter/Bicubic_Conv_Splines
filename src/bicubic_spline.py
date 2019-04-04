import numpy as np
import json

with open("bicubic_solve_mat.json") as j:
    data = json.load(j)
    Q = np.linalg.inv(np.asarray(data))

def get_4x4_mat(M, x, y):

    assert isinstance(x, int)
    m, n = M.shape

    if 1 <= x <= n-3 and 1 <= y <= m-3:
        return M[y-1:y+3, x-1:x+3]
    
    else:
        E = np.zeros([m + 2, n + 2])
        E[1:-1, 1:-1] = M
        E[-1] = 2 * E[-2] - E[-3]
        E[0] = 2 * E[1] - E[2]
        E[:, -1] = 2 * E[:, -2] - E[:, -3]
        E[:, 0] = 2 * E[:, 1] - E[:, 2]
        return E[y:y+4, x:x+4]

def bicubic_spline_coeff(M, padding=None):

    assert isinstance(M, np.ndarray)
    m, n = M.shape
    m -= 1
    n -= 1

    coeffs = np.zeros([m, n, 4, 4])

    for y in range(m):
        for x in range(n):
            A = get_4x4_mat(M, x, y)

            f_x = np.concatenate([np.expand_dims((A[:, 2] - A[:, 0]) / 2, axis=1),
                                  np.expand_dims((A[:, 3] - A[:, 1]) / 2, axis=1)],
                                  axis=1)

            f_y = np.concatenate([np.expand_dims((A[2] - A[0]) / 2, axis=0),
                                  np.expand_dims((A[3] - A[1]) / 2, axis=0)],
                                  axis=0)

            f_xy = np.concatenate([np.expand_dims((f_x[2] - f_x[0]) / 2, axis=0),
                                   np.expand_dims((f_x[3] - f_x[1]) / 2, axis=0)],
                                   axis=0)
            
            f = np.concatenate([A[1:3, 1:3].flatten(),
                                f_x[1:3].flatten(),
                                f_y[:, 1:3].flatten(),
                                f_xy.flatten()], axis=0)
            
            
            f = np.transpose(np.expand_dims(f, axis=0))

            coeffs[y, x] = np.matmul(Q, f).reshape(4, 4)
        
    return coeffs

def bicubic_spline_interpolation(M, coeffs, pos):
    
    m, n = M.shape
    m -= 1
    n -= 1
    x, y = pos
    a, b = int(x), int(y)
    dx, dy = x - a, y - b

    if x == n and y == m:
        coeff = coeffs[m-1][n-1]
        dx, dy = 1, 1
    elif x == n:
        coeff = coeffs[b][n-1]
        dx = 1
    elif y == m:
        coeff = coeffs[m-1][a]
        dy = 1
    else:
        coeff = coeffs[b][a]
    
    vdx = np.array([[1, dx, dx**2, dx**3]])
    vdy = np.array([[1], [dy], [dy**2], [dy**3]])
    ret = np.matmul(np.matmul(vdx, coeff), vdy)

    return ret


def bicubic_spline_helper(M, scale=100):

    coeffs = bicubic_spline_coeff(M)

    assert isinstance(M, np.ndarray)
    m, n = M.shape
    m -= 1
    n -= 1
    ret = np.zeros([scale * m + 1, scale * n + 1])
    for i in range(scale * m + 1):
        for j in range(scale * n + 1):
            ret[i][j] = bicubic_spline_interpolation(M, coeffs, [j/scale, i/scale])
    
    return ret
    