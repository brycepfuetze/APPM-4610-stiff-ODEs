import numpy as np

def RK4(a, b, h, ya, eval_f, params):

     N = int((b - a) / h)
     
     yapp = np.zeros((N + 1, 2, len(ya), 3))
     t = np.zeros(N + 1)
     
     yapp[[0], :, :, :] = ya
     t[0] = a

     for jj in range(1, N + 1):
        tj = a + (jj - 1) * h
        t[jj] = tj + h
        rk = yapp[[jj - 1], :, :, :]
        k1 = h * eval_f(tj, rk, params)
        k2 = h * eval_f(tj + h / 2, rk + 0.5 * k1, params)
        k3 = h * eval_f(tj + h / 2, rk + 1 / 2 * k2, params)
        k4 = h * eval_f(tj + h, rk + k3, params)
        yapp[[jj], :, :, :] = rk + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

     return t, yapp
