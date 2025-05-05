#! /usr/bin/env python3
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# gravitational constant (m^3/kg/s^2)
G = 6.674e-11

np.seterr('raise')

# masses (Sun - Earth)
m = np.array([1.9885e30, 5.972e24])
# initial conditions
x0 = np.array([[0,0], [1.49e11,0]])
v0 = np.array([[0,0], [0,2.97e4]])

# compute forces/kg for the given x
def f(m, x):
    f = np.zeros(x.shape)
    # compute force on i-th body
    for i in range(x.shape[0]):
        index = np.arange(x.shape[0])
        # distance to all other bodies
        dx = x - x[i]
        r = la.norm(dx, axis=1)
        r_cb = r**3
        # remove self-force
        r_cb = np.where(index == i, np.inf, r_cb)
        # force from all other bodies
        F = G*m / r_cb * dx.T
        # total force on body
        f[i,:] = np.sum(F, axis=1)
    
    return f

# take a verlet step and return [x_{n+1}, v_{n+1/2}]
def verlet_step(m, x, v, h):
    v = v + h*f(m, x)
    x = x + h*v

    return [v, x]

# perform verlet timestepping
def verlet(m, x0, v0, tend, h, sampling):
    N = int(np.ceil(tend/h/sampling))
    x = np.zeros((N, x0.shape[0], x0.shape[1]))
    v = np.zeros((N, x0.shape[0], x0.shape[1]))

    for i in range(N*sampling):
        t = i*h
        if i%sampling == 0:
            x[int(i/sampling), :, :] = x0
            v[int(i/sampling), :, :] = v0
        [v0, x0] = verlet_step(m, x0, v0, h)
    
    return [x, v]

# perform Euler timestepping on the first order system
def euler(m, x0, v0, tend, h, sampling):
    N = int(tend/h/sampling)
    x = np.zeros((N, x0.shape[0], x0.shape[1]))
    v = np.zeros((N, x0.shape[0], x0.shape[1]))

    for i in range(N*sampling):
        t = i*h
        if i%sampling == 0:
            x[int(i/sampling), :, :] = x0
            v[int(i/sampling), :, :] = v0
        [v0, x0] = [v0 + h*f(m, x0), x0+h*v0]
    
    return [x, v]

# perform RK2 (midpoint) on the first order system
def rk2(m, x0, v0, tend, h, sampling):
    N = int(tend/h/sampling)
    x = np.zeros((N, x0.shape[0], x0.shape[1]))
    v = np.zeros((N, x0.shape[0], x0.shape[1]))

    for i in range(N*sampling):
        t = i*h
        if i%sampling == 0:
            x[int(i/sampling), :, :] = x0
            v[int(i/sampling), :, :] = v0
        # update
        [v0, x0] = [v0+h*f(m, x0+1/2*h*v0), x0+h*(v0+1/2*h*f(m, x0))]
    
    return [x, v]

# perform RK4 on the first order system
def rk4(m, x0, v0, tend, h, sampling):
    N = int(tend/h/sampling)
    x = np.zeros((N, x0.shape[0], x0.shape[1]))
    v = np.zeros((N, x0.shape[0], x0.shape[1]))

    for i in range(N*sampling):
        t = i*h
        if i%sampling == 0:
            x[int(i/sampling), :, :] = x0
            v[int(i/sampling), :, :] = v0
        # intermediate points
        [v1, x1] = [f(m, x0), v0]
        [v2, x2] = [f(m, x0+h*x1/2), v0+h*v1/2]
        [v3, x3] = [f(m, x0+h*x2/2), v0+h*v2/2]
        [v4, x4] = [f(m, x0+h*x3), v0+h*v3]
        # update
        [v0, x0] = [v0 + h/6 * (v1+2*v2+2*v3+v4), x0+h/6*(x1+2*x2+2*x3+x4)]
    
    return [x, v]

# calculate the total energy contained in the system
def energy(m, xs, vs):
    E = np.zeros(xs.shape[0])
    for i in range(E.size):
        x = xs[i,:,:]
        v = vs[i,:,:]
        # kinetic energy of all particles
        T = 0.5*m*la.norm(v, axis=1)**2
        # potential energy between all other particles
        U = 0
        for j in range(x.shape[0]):
            index = np.arange(x.shape[0])
            # distance to all other bodies
            dx = x - x[j]
            r = la.norm(dx, axis=1)
            # remove self-potential
            r = np.where(index == j, np.inf, r)
            # potential between particles
            U = U + np.sum(-G*m*m[j] / r)

        E[i] = np.sum(T) + U
    
    return E

if __name__ == "__main__":
    day_s = 60*60*24
    year_s = day_s * 365
    tend = day_s*365*10000
    tstep = day_s
    sampling = 512
    AU = 149578706600
    t = np.arange(0, int(tend/(tstep*sampling))*tstep*sampling, tstep*sampling)

    fig = plt.figure()
    gs0 = gridspec.GridSpec(1, 2, figure=fig)
    gs00 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs0[1])

    ax0 = fig.add_subplot(gs0[0])

    methods = [verlet, euler, rk2, rk4]
    method_names = ["Verlet", "Euler", "RK2", "RK4"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for i in range(len(methods)):
        method = methods[i]
        name = method_names[i]
        print(name)

        [x, v] = method(m, x0, v0, tend, tstep, sampling)
        ax = fig.add_subplot(gs00[i])

        ax.plot(x[:,1,0]/AU, x[:,1,1]/AU, color=colors[i])
        ax.grid()
        if i > 1:
            ax.set_xlabel("x [AU]")
        if i == 0 or i == 2:
            ax.set_ylabel("y [AU]")

        E = energy(m, x, v)

        ax0.semilogy(t/year_s, np.abs(E-E[0]), label=name, color=colors[i])

    ax0.set_xlabel("Time [yr]")
    ax0.set_ylabel("Energy Error [J]")
    ax0.legend()
    ax0.grid()
    #plt.ylim([1e31, 1e34])
    plt.show()
