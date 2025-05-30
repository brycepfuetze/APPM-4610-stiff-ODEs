import numpy as np

def mag_cube(u, v):
    return np.power(np.linalg.norm(u - v), -3)

def nbody_EOM_rk4(_, R, params):
    M = params['M']
    G = params['G']

    sh = np.shape(M)
    N = sh[1]

    r = R[0, 0, :, :]

    rp = np.ones_like(r) * R[0, 1, :, :]

    rinv = np.zeros((N, N))

    for i in range(N):
        for j in range(i):
            rinv[i, j] = mag_cube(r[i, :], r[j, :])

    S = rinv + rinv.T

    P = S @ M

    X = np.tile(r[:, 0], (N, 1))
    Y = np.tile(r[:, 1], (N, 1))
    Z = np.tile(r[:, 2], (N, 1))

    dx = X.T - X
    dy = Y.T - Y
    dz = Z.T - Z

    AX = G * np.einsum('ij,ji->i', P, dx)
    AY = G * np.einsum('ij,ji->i', P, dy)
    AZ = G * np.einsum('ij,ji->i', P, dz)

    rpp = np.vstack((AX, AY, AZ)).T

    return np.array([rp, rpp])