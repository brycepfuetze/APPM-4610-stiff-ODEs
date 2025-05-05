import plotly.graph_objects as go
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.seterr('raise')

# Parameters based on this paper:
# https://arxiv.org/pdf/astro-ph/0009058.pdf
# The parameter provided assume G = 1, all m = 1, and speicfies starting locations and velocities

G = 1.0
# masses
m = np.array([1, 1, 1])
# initial conditions
v1 =    0.3962186234
v2 =   0.5086826315

x1init = -1
x2init = 1

perturbation_factor = 0.05
np.random.seed(40)

# lets create some random perturbations
v1 = v1 * np.random.uniform(1 - perturbation_factor, 1 + perturbation_factor)
v2 = v2 * np.random.uniform(1 - perturbation_factor, 1 + perturbation_factor)
x1init = x1init * np.random.uniform(1 - perturbation_factor, 1 + perturbation_factor)
x2init = x2init * np.random.uniform(1 - perturbation_factor, 1 + perturbation_factor)

T =  96.4358796119

x0 = np.array([[x1init,0], [x2init,0], [0,0]])
v0 = np.array([[v1,v2], [v1,v2], [-2*v1, -2*v2]])


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

# these you need to play with to make the integration stable
N = 10000
tend = T/8
tstep = T / N
sampling = 1

t = np.arange(0, int(tend/(tstep*sampling))*tstep*sampling, tstep*sampling)
[x, v] = verlet(m, x0, v0, tend, tstep, sampling)

N = len(x)

#xm = np.min(x[:,1,0]/AU) - 0.1 * np.min(x[:,1,0]/AU)
#xM = np.max(x[:,1,0]/AU) + 0.1 * np.max(x[:,1,0]/AU)
#ym = np.min(x[:,1,1]/AU) - 0.1 * np.min(x[:,1,1]/AU)
#yM = np.max(x[:,1,1]/AU) + 0.1 * np.max(x[:,1,1]/AU)

# Create figure
fig = go.Figure(
    data=[
        # Object 1
        go.Scatter(x=x[:,0,0], y=x[:,0,1],
                   mode="lines",
                   line=dict(width=2, color="red"),
                   name="Object 1 Path"),
        go.Scatter(x=[x[0,0,0]], y=[x[0,0,1]],
                   mode="markers",
                   marker=dict(color="red", size=20),
                   name="Object 1"),
        # Object 2
        go.Scatter(x=x[:,1,0], y=x[:,1,1],
                   mode="lines",
                   line=dict(width=2, color="blue"),
                   name="Object 2 Path"),
        go.Scatter(x=[x[0,1,0]], y=[x[0,1,1]],
                   mode="markers",
                   marker=dict(color="blue", size=20),
                   name="Object 2"),
        # Object 3
        go.Scatter(x=x[:,2,0], y=x[:,2,1],
                   mode="lines",
                   line=dict(width=2, color="green"),
                   name="Object 3 Path"),
        go.Scatter(x=[x[0,2,0]], y=[x[0,2,1]],
                   mode="markers",
                   marker=dict(color="green", size=20),
                   name="Object 3")
    ]
)

fig.update_layout(
    width=1200, height=700,
    title_text="Orbit Animation with 5% Perturbation", title_x=0.5,
    updatemenus=[
        dict(
            type="buttons",
            buttons=[
                dict(
                    args=[None, {"frame": {"duration": 10, "redraw": False},
                                 "fromcurrent": True, "transition": {"duration": 10}}],
                    label="Play",
                    method="animate"
                )
            ]
        )
    ]
)

fig.update(frames=[
    go.Frame(
        data=[
            go.Scatter(x=[x[k,0,0]], y=[x[k,0,1]]),  # Object 1
            go.Scatter(x=[x[k,1,0]], y=[x[k,1,1]]),  # Object 2
            go.Scatter(x=[x[k,2,0]], y=[x[k,2,1]])   # Object 3
        ],
        traces=[1, 3, 5]  # Update traces for Object 1, 2, and 3
    )
    for k in range(N)
])

fig.show()