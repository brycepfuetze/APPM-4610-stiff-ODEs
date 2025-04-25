import plotly.graph_objects as go
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
    N = int(tend/h/sampling)
    x = np.zeros((N, x0.shape[0], x0.shape[1]))
    v = np.zeros((N, x0.shape[0], x0.shape[1]))

    for i in range(N*sampling):
        t = i*h
        if i%sampling == 0:
            x[int(i/sampling), :, :] = x0
            v[int(i/sampling), :, :] = v0
        [v0, x0] = verlet_step(m, x0, v0, h)
    
    return [x, v]

day_s = 60*60*24
year_s = day_s * 365
tend = day_s*365*20
tstep = day_s
sampling = 2
AU = 149578706600
t = np.arange(0, int(tend/(tstep*sampling))*tstep*sampling, tstep*sampling)
[x, v] = verlet(m, x0, v0, tend, tstep, sampling)

N = len(x)

#xm = np.min(x[:,1,0]/AU) - 0.1 * np.min(x[:,1,0]/AU)
#xM = np.max(x[:,1,0]/AU) + 0.1 * np.max(x[:,1,0]/AU)
#ym = np.min(x[:,1,1]/AU) - 0.1 * np.min(x[:,1,1]/AU)
#yM = np.max(x[:,1,1]/AU) + 0.1 * np.max(x[:,1,1]/AU)

# Create figure , 
fig = go.Figure(
    data=[go.Scatter(x=x[:,1,0]/AU, y=x[:,1,1]/AU,
                     mode="lines",
                     line=dict(width=2, color="blue"),
                     name="Earth Orbit"),
          go.Scatter(x=[x[0,1,0]/AU], y=[x[0,1,1]],
                     mode="markers",
                     marker=dict(color="green", size=20),
                     name="Earth"),
          go.Scatter(x=[x[0,0,0]], y=[x[0,0,1]],
                   mode="markers",
                   marker=dict(color="orange", size=50),
                   name="Sun")])
fig.update_layout(width=1000, height=900,
        xaxis=dict(range=[-1.1, 1.1], autorange=False, zeroline=False),
        yaxis=dict(range=[-1.1, 1.1], autorange=False, zeroline=False),
        title_text="Earth Orbit of the Sun", title_x=0.5,
        updatemenus = [dict(type = "buttons",
        buttons = [
            dict(
                args = [None, {"frame": {"duration": 10, "redraw": False},
                                "fromcurrent": True, "transition": {"duration": 10}}],
                label = "Play",
                method = "animate",

                )])])

fig.update(frames=[go.Frame(
                        data=[go.Scatter(
                                   x=[x[k,1,0]/AU],
                                   y=[x[k,1,1]/AU])],
                        traces=[1]) # fig.data[1] is updated by each frame
        for k in range(N)])

fig.show()