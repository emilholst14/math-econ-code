import numpy as np

def amput_binom(S0, K, T, r, sigma, n):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    disc = np.exp(-r * dt)
    q = (1 / disc - d) / (u - d)

    P = np.maximum(K - S0 * u ** (n - 2*np.arange(n+1)), 0)
    for i in range(n-1, -1, -1):
        P[:i+1] = np.maximum(
            disc * (q * P[:i+1] + (1-q) * P[1:i+2]),
            K - S0 * u**(i - 2*np.arange(i+1))
        )
    return P[0]