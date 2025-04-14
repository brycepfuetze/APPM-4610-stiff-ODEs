import numpy as np
import pandas as pd
import plotly.express as px

def log_map(A, x0):
    x = np.zeros(101)
    x[0] = x0
    for i in range(100):
        x[i + 1] = A * x[i] * (1 - x[i])
    return x

# set ICs, can make A a vector if you want!
A = [3.9]
x0 = [0.6875, 0.6874, 0.6876]

# loop over A and x0 vectors
data = []
for a in A:
    for initial_x in x0:
        x_values = log_map(a, initial_x)
        data.extend({
            "Iteration": i,
            "Value": x_values[i],
            "A": a,
            "x0": initial_x
        } for i in range(101))

# send to DF for easy plotly
df = pd.DataFrame(data)

# plot!!!
fig = px.line(df, x="Iteration", y="Value", color="x0", facet_row="A", labels={"x0": "Initial x0"}, title="Logistic Map")

fig.update_layout(height=400 * len(A))
fig.show()