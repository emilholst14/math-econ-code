import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

S0 = 36
r = 0.06
sigma = 0.2
d= 50
dt = 1 / d
N = 10000

W = np.random.normal(0, np.sqrt(dt), size=(d, N))
S = np.exp((r - sigma**2 / 2) * dt + sigma * W)
S = S0 * np.vstack([np.ones(N), S]).cumprod(axis=0)

print(max(S[:, 1]))