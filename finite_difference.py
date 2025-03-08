import numpy as np

def amput_fd(S0, K, T, r, sigma, n, m):
    assert m % 2 == 0 # ensure S0 appears in grid   
    assert n >= T * sigma**2 * m**2, 'Unstable: n too small compared to m'
    S_max = 2 * S0
    dt = T / n 
    S_vals = np.linspace(0, S_max, m+1)

    V = np.zeros((n+1, m+1))
    V[n, :] = np.maximum(K - S_vals, 0)
    V[:, 0] = K

    js = np.arange(1, m)
    aj = (1/2 * sigma**2 * js**2 * dt - 1/2 * r * js * dt)
    bj = (1 - sigma**2 * js**2 * dt)
    gj = (1/2 * r * js * dt + 1/2 * sigma**2 * js**2 * dt)

    for i in range(n-1, -1, -1):
        V[i, 1:m] = np.maximum(
            1/(1 + r*dt) * (aj * V[i+1, 0:m-1] + bj * V[i+1, 1:m] + gj * V[i+1, 2:]),
            K - S_vals[1:m]
        )

    return V[0, m//2]