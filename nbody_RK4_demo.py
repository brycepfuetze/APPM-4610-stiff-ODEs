import numpy as np
from matplotlib import pyplot as plt

from System import System
from Object import Object

G = 6.6743E-11 # Gravitational constant
mu = 32930E9 # Gravtitational parameter for planet

system = System('sys', G)

planet = Object('planet', mu / G, [0, 0, 0], [0, 0, 0])
satellite = Object('satellite', 100, [6558.9E3, 3660.8E3, -10282.0E3], [1.5525E3, -0.1746E3, 0.5038E3])

system.add_object(planet)
system.add_object(satellite)

_, yapp = system.propagate_system_rk4(0, 53054, 1E3)

r1 = yapp[:, 0, :]
r2 = yapp[:, 1, :]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', aspect='equal')

ax.plot(r1[:, 0], r1[:, 1], r1[:, 2])
ax.plot(r2[:, 0], r2[:, 1], r2[:, 2])

plt.show()
