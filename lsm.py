# American option with LSM

import numpy as np
from bsm import bsm

def amput_lsm(S0, K, T, r, sigma, N, d, M, kind, av=False, cv=False, seed=42):
    np.random.seed(seed)
    dt = T / d
    disc = np.exp(-r * dt)

    # Antithetical variables
    if av:
        W = np.random.standard_normal((d, N // 2))
        W = np.concatenate((W, -W), axis=1)
    else:
        W = np.random.standard_normal((d, N))
        
    S = np.exp((r - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * W)
    S = S0 * np.vstack([np.ones(N), np.cumprod(S, axis=0)])

    vals = np.maximum(K - S[-1, :], 0)
    for t in range(d-1, 0, -1):
        intrinsic = np.maximum(K - S[t, :], 0)
        itm = intrinsic > 0
        if np.any(itm):
            X = S[t, itm]
            Y = vals[itm] * disc
            reg = np.polyfit(X, Y, M)
            cont = np.zeros(N)
            cont[itm] = np.polyval(reg, X)
            exercise = intrinsic > cont
            vals = np.where(exercise, intrinsic, vals * disc) 

    # EU put as control
    if cv:
        eu_lsm = np.maximum(K - S[-1, :], 0) * np.exp(-r * T)
        eu_bsm = bsm(S0, K, T, r, sigma, 'put')
        beta = np.cov(vals, eu_lsm)[0, 1] / np.var(eu_lsm)
        vals = vals - beta * (eu_lsm - eu_bsm) 
        
    sd = np.std(vals, ddof=1)/np.sqrt(N)
    return np.mean(vals), sd, [np.mean(vals) - 1.96*sd, np.mean(vals) + 1.96*sd]
