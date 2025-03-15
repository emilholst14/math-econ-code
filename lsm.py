import numpy as np
from bsm import bsm

def amput_lsm(S0, K, T, r, sigma, N, d, M, av=False, cv=False):
    dt = T / d
    disc = np.exp(-r * dt)

    # Antithetical variables
    if av:
        W = np.random.normal(0, np.sqrt(dt), size=(d, N // 2))
        W = np.concatenate((W, -W), axis=1)
    else:
        W = np.random.normal(0, np.sqrt(dt), size=(d, N))
        
    S = np.exp((r - sigma**2 / 2) * dt + sigma * W)
    S = S0 * np.vstack([np.ones(N), S]).cumprod(axis=0)

    payoff = np.maximum(K - S, 0)
    vals = payoff[-1, :]
    for t in range(d-1, 0, -1):
        itm = payoff[t, :] > 0
        if np.any(itm):
            X = S[t, itm]
            Y = vals[itm] * disc
            reg = np.polyfit(X, Y, M)
            cont = np.polyval(reg, X)

            exercise = payoff[t, itm] > cont
            vals[itm] = np.where(exercise, payoff[t, itm], Y)
    
    V = vals * disc 

    # EU option as control variate
    if cv:
        eu_lsm = np.maximum(K - S[-1], 0) * np.exp(-r * T)
        eu_bsm = bsm(S0, K, T, r, sigma, 'put')
        beta = np.cov(V, eu_lsm)[0, 1] / np.var(eu_lsm)
        V = V - beta * (eu_lsm - eu_bsm) 
    sd = np.std(V, ddof=1)/np.sqrt(N)
    return np.mean(V), sd, [np.mean(V) - 1.96*sd, np.mean(V) + 1.96*sd]

# np.random.seed(1203)
# print(amput_lsm(100,100,2,0.05,0.2,100000,50,3,av=True, cv=True))
