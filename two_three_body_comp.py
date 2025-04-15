from matplotlib import pyplot as plt
import astropy.units as u

from System import System

from HorizonsInterface import instantiate_object_horizons

G = 6.6743E-11 # Gravitational constant

solar_system_two_body = System('Solar System Two-Body', G)
solar_system_three_body = System('Solar System Three-Body', G)

sun = instantiate_object_horizons('Sun', 1.9891E30, '10', '@10', '2025-01-01')
earth = instantiate_object_horizons('Earth', 5.9722E24, '399', '@10', '2025-01-01')
moon = instantiate_object_horizons('Moon', 7.34767309E22, '301', '@10', '2025-01-01')

solar_system_two_body.add_object(sun)
solar_system_two_body.add_object(earth)

solar_system_three_body.add_object(sun)
solar_system_three_body.add_object(earth)
solar_system_three_body.add_object(moon)

_, yapp_two_body = solar_system_two_body.propagate_system_rk4(0, 3.154e+7, 1E4)
_, yapp_three_body = solar_system_three_body.propagate_system_rk4(0, 3.154e+7, 1E4)

earth_pos_two_body = (yapp_two_body[:, 1, :] * u.m).to(u.AU)
earth_pos_three_body = (yapp_three_body[:, 1, :] * u.m).to(u.AU)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(0, 0, 0, s=100, c='#fff200', label=None)
ax.plot(earth_pos_two_body[:, 0], earth_pos_two_body[:, 1], earth_pos_two_body[:, 2], c='#00bd09', linewidth=3.0, label='Two-Body')
ax.plot(earth_pos_three_body[:, 0], earth_pos_three_body[:, 1], earth_pos_three_body[:, 2], c='#006eff', linewidth=3.0, label='Three-Body')

ax.legend()

ax.set_title('Comparison of Two- and Three-Body Dynamics')

ax.set_xlabel('$x$-distance [AU]')
ax.set_ylabel('$y$-distance [AU]')
ax.set_zlabel('$z$-distance [AU]')

ax.view_init(elev=25, azim=-30, roll=5)

plt.show()
