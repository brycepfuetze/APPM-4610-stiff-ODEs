import numpy as np

from RungeKutta import RK4
from EOM import nbody_EOM_rk4

M = np.diag([1, 2]) # Object masses

r = np.array([[1, 2, 3], [4, 5, 6]]) # Initial positions
rp = np.array([[0, 0, 0], [0, 0, 0]]) # Initial velocities

R0 = np.array([r, rp])

G = 1 # Gravitational constant

params = {'M': M, 'G': G}

yapp, t = RK4(0, 1, 0.1, R0, nbody_EOM_rk4, params)

yapp = yapp[:, 0, :, :]

print(yapp)
