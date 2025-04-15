from matplotlib import pyplot as plt
import astropy.units as u

from System import System

from HorizonsInterface import instantiate_object_horizons

G = 6.6743E-11 # Gravitational constant

solar_system = System('Solar System', G)

sun = instantiate_object_horizons('Sun', 1.9891E30, '10', '@10', '2025-01-01')
earth = instantiate_object_horizons('Earth', 5.9722E24, '399', '@10', '2025-01-01')
moon = instantiate_object_horizons('Moon', 7.34767309E22, '301', '@10', '2025-01-01')

solar_system.add_object(sun)
solar_system.add_object(earth)
solar_system.add_object(moon)

_, yapp = solar_system.propagate_system_rk4(0, 3.154e+7, 1E4)

earth_pos = (yapp[:, 1, :] * u.m).to(u.AU)
moon_pos = (yapp[:, 2, :] * u.m).to(u.AU)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(0, 0, 0, s=100, c='#fff200')
ax.plot(earth_pos[:, 0], earth_pos[:, 1], earth_pos[:, 2], c='#006eff', linewidth=3.0)
ax.plot(moon_pos[:, 0], moon_pos[:, 1], moon_pos[:, 2], c='#4e5154', linewidth=3.0)

ax.set_title('One-Year Propagation of Sun-Earth-Moon System')

ax.set_xlabel('$x$-distance [AU]')
ax.set_ylabel('$y$-distance [AU]')
ax.set_zlabel('$z$-distance [AU]')

plt.show()
