import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Create a grid in the complex plane
x = np.linspace(-3, 3, 600)
y = np.linspace(-3, 3, 600)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Compute masks for stability regions
euler_region = np.abs(1 + Z) < 1
trap_region = np.abs((2 + Z) / (2 - Z)) < 1

# Helper function to plot filled region
def plot_region(region_mask, title):
    fig = px.imshow(region_mask.astype(int),
                    origin='lower',
                    x=x,
                    y=y,
                    color_continuous_scale=[[0, "white"], [1, "blue"]],
                    labels={'x': 'Re(λh)', 'y': 'Im(λh)'},
                    title=title,
                    aspect="equal")
    fig.update_layout(width=600, height=500,
                      coloraxis_showscale=False)
    fig.add_shape(type="line", x0=-3, x1=3, y0=0, y1=0,
                  line=dict(color="black", width=1))
    fig.add_shape(type="line", x0=0, x1=0, y0=-3, y1=3,
                  line=dict(color="black", width=1))
    return fig

# Plot each region
fig_euler = plot_region(euler_region, "Stability Region: Euler's Method")
fig_trap = plot_region(trap_region, "Stability Region: Implicit Trapezoidal Method")

fig_euler.show()
fig_trap.show()