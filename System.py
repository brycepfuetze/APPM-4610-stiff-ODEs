import numpy as np

from RungeKutta import RK4
from EOM import nbody_EOM_rk4
from verlet import verlet, rk4

class System:
    def __init__(self, name, G):
        self.name = name
        self.G = G
        self.objects = []
        self.obj_count = 0

    def add_object(self, object):
        self.objects.append(object)
        self.obj_count += 1

    def gen_init_cond(self):
        N = self.obj_count

        r = np.zeros((N, 3))
        rp = np.zeros((N, 3))

        for i in range(N):
            r[i, :] = self.objects[i].get_r0()
            rp[i, :] = self.objects[i].get_v0()
            
        R0 = np.array([r, rp])
        
        return R0
    
    def gen_mass_mat(self):
        N = self.obj_count

        M = np.zeros(N)

        for i in range(N):
            M[i] = self.objects[i].get_m()
        
        return np.diag(M)

    def propagate_system_rk4(self, a, b, h):
        M = self.gen_mass_mat()
        R0 = self.gen_init_cond()

        params = {'M': M, 'G': self.G}

        t, yapp = RK4(a, b, h, R0, nbody_EOM_rk4, params)

        yapp = yapp[:, 0, :, :]

        return t, yapp
    
    def propagate_system_verlet(self, a, b, h):
        M = self.gen_mass_mat()
        R0 = self.gen_init_cond()

        m = np.diag(M)
        x0 = R0[0, :, :]
        v0 = R0[1, :, :]

        t = np.arange(a, b, h)
        
        yapp, _ = verlet(m, x0, v0, b - a, h, 1)

        return t, yapp
