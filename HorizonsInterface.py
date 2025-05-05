import numpy as np
from astroquery.jplhorizons import Horizons
from astropy.time import Time
import astropy.units as u

from Object import Object

def get_data_horizons(id, location, epoch):
    if isinstance(epoch, str):
        epoch = Time(epoch).jd

    obj = Horizons(id=id, location=location, epochs=epoch)

    vec = obj.vectors().columns

    x = vec['x'].to(u.m).value[:]
    y = vec['y'].to(u.m).value[:]
    z = vec['z'].to(u.m).value[:]

    vx = vec['vx'].to(u.m / u.s).value[:]
    vy = vec['vy'].to(u.m / u.s).value[:]
    vz = vec['vz'].to(u.m / u.s).value[:]

    r = np.array([x, y, z]).T
    v = np.array([vx, vy, vz]).T

    return r, v

def instantiate_object_horizons(name, m, id, location, epoch):
    r0, v0 = get_data_horizons(id, location, epoch)

    return Object(name, m, r0, v0)
