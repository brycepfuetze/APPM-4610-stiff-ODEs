from matplotlib import pyplot as plt

from System import System

from HorizonsInterface import instantiate_object_horizons

G = 6.6743E-11 # Gravitational constant

solar_system = System('Solar System', G)

sun = instantiate_object_horizons('Sun', 1.9891E30, '10', '500@sun', '2025-01-01')
earth = instantiate_object_horizons('Earth', 5.9722E24, '399', '500@sun', '2025-01-01')
moon = instantiate_object_horizons('Moon', 7.34767309E22, '301', '500@sun', '2025-01-01')

solar_system.add_object(sun)
solar_system.add_object(earth)
solar_system.add_object(moon)

_, yapp = solar_system.propagate_system_rk4(0, 3.154e+7, 1E4)

sun_pos = yapp[:, 0, :]
earth_pos = yapp[:, 1, :]
moon_pos = yapp[:, 2, :]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(sun_pos[:, 0], sun_pos[:, 1], sun_pos[:, 2])
ax.plot(earth_pos[:, 0], earth_pos[:, 1], earth_pos[:, 2])
ax.plot(moon_pos[:, 0], moon_pos[:, 1], moon_pos[:, 2])

plt.show()
