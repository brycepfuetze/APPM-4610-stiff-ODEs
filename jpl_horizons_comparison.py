import numpy as np
from matplotlib import pyplot as plt
import astropy.units as u

from System import System

from HorizonsInterface import instantiate_object_horizons, get_data_horizons

G = 6.674E-11 # Gravitational constant

solar_system = System('Solar System', G)

sun = instantiate_object_horizons('Sun', 1.9891E30, '10', '@10', '2025-01-01')
earth = instantiate_object_horizons('Earth', 5.9722E24, '399', '@10', '2025-01-01')
moon = instantiate_object_horizons('Moon', 7.34767309E22, '301', '@10', '2025-01-01')

solar_system.add_object(sun)
solar_system.add_object(earth)
solar_system.add_object(moon)

t, yapp_rk4 = solar_system.propagate_system_rk4(0, (100 * u.year).to(u.s).value, int((20 * u.d).to(u.s).value))
_, yapp_verlet = solar_system.propagate_system_verlet(0, (100 * u.year).to(u.s).value, int((20 * u.d).to(u.s).value))
yapp_horizons, _ = get_data_horizons('301', '@10', {'start': '2025-01-01', 'stop': '2125-01-01', 'step': '20d'})

moon_pos_rk4 = (yapp_rk4[:, 2, :] * u.m).to(u.AU)
moon_pos_verlet = (yapp_verlet[:, 2, :] * u.m).to(u.AU)
moon_pos_horizons = (yapp_horizons * u.m).to(u.AU)

rk4_diff = np.linalg.norm(moon_pos_rk4 - moon_pos_horizons, axis=1)
verlet_diff = np.linalg.norm(moon_pos_verlet - moon_pos_horizons, axis=1)

t_years = (t * u.s).to(u.year)

fig, ax = plt.subplots()

ax.semilogy(t_years, verlet_diff, label='Verlet', color='tab:blue')
ax.semilogy(t_years, rk4_diff, label='RK4', color='tab:red')

ax.legend()

ax.set_title('Deviation Between Propagation Methods and JPL Horizons Data')

ax.set_xlabel('Years Since Propagation Start [y]')
ax.set_ylabel('Deviation from JPL Horizons Data [AU]')

ax.grid('both')

plt.show()
