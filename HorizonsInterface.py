import numpy as np
from astroquery.jplhorizons import Horizons
from astropy.time import Time
import astropy.units as u

from Object import Object

def get_initial_data_horizons(id, location, epoch):
    epoch = Time(epoch).jd

    obj = Horizons(id=id, location=location, epochs=epoch)

    x = obj.vectors().columns['x'].to(u.m).value[0]
    y = obj.vectors().columns['y'].to(u.m).value[0]
    z = obj.vectors().columns['z'].to(u.m).value[0]

    vx = obj.vectors().columns['vx'].to(u.m / u.s).value[0]
    vy = obj.vectors().columns['vy'].to(u.m / u.s).value[0]
    vz = obj.vectors().columns['vz'].to(u.m / u.s).value[0]

    r0 = np.array([x, y, z])
    v0 = np.array([vx, vy, vz])

    return r0, v0

def instantiate_object_horizons(name, m, id, location, epoch):
    r0, v0 = get_initial_data_horizons(id, location, epoch)

    return Object(name, m, r0, v0)
