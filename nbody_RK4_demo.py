import numpy as np
from matplotlib import pyplot as plt

from RungeKutta import RK4
from EOM import nbody_EOM_rk4

mu = 32930E9

G = 6.6743E-11 # Gravitational constant

M = np.diag([mu / G, 100]) # Object masses

r = np.array([[0, 0, 0], [6558.9E3, 3660.8E3, -10282.0E3]]) # Initial positions
rp = np.array([[0, 0, 0], [1.5525E3, -0.1746E3, 0.5038E3]]) # Initial velocities

R0 = np.array([r, rp])

params = {'M': M, 'G': G}

yapp, t = RK4(0, 53054, 1E3, R0, nbody_EOM_rk4, params)

yapp = yapp[:, 0, :, :]

r1 = yapp[:, 0, :]
r2 = yapp[:, 1, :]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', aspect='equal')

ax.plot(r1[:, 0], r1[:, 1], r1[:, 2])
ax.plot(r2[:, 0], r2[:, 1], r2[:, 2])

plt.show()
