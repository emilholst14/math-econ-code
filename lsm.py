import numpy as np

def amput_lsm(S0, K, T, r, sigma, d, N, M):
    dt = T / d
    disc = np.exp(-r * dt)

    W = np.random.normal(0, np.sqrt(dt), size=(d, N))
    S = np.exp((r - sigma**2 / 2) * dt + sigma * W)
    S = S0 * np.vstack([np.ones(N), S]).cumprod(axis=0)

    payoff = np.maximum(K - S, 0)
    vals = np.zeros_like(payoff)
    vals[-1, :] = payoff[-1, :]
    for t in range(d-1, 0, -1):
        itm = payoff[t, :] > 0
        if np.any(itm):
            reg = np.polyfit(S[t, :], vals[t+1, :] * disc, M)
            cont = np.polyval(reg, S[t, :])
        else:
            cont = np.zeros_like(vals[t+1, :])
        vals[t, :] = np.where(payoff[t, :] > cont, payoff[t, :], vals[t+1, :])

    V = vals[1, :] * disc
    sd = np.std(V, ddof=1)/np.sqrt(N)
    return np.mean(V), sd, [np.mean(V) - 1.96*sd, np.mean(V) + 1.96*sd]

def amput_lsm_av(S0, K, T, r, sigma, d, N, M):
    dt = T / d
    disc = np.exp(-r * dt)

    W = np.random.normal(0, np.sqrt(dt), size=(d, N // 2))
    W = np.concatenate((W, -W), axis=1)
    S = np.exp((r - sigma**2 / 2) * dt + sigma * W)
    S = S0 * np.vstack([np.ones(N), S]).cumprod(axis=0)

    payoff = np.maximum(K - S, 0)
    vals = np.zeros_like(payoff)
    vals[-1, :] = payoff[-1, :]
    for t in range(d-1, 0, -1):
        itm = payoff[t, :] > 0
        if np.any(itm):
            reg = np.polyfit(S[t, :], vals[t+1, :], M)
            cont = np.polyval(reg, S[t, :])
        else:
            cont = np.zeros_like(vals[t+1, :])
        vals[t, :] = np.where(payoff[t, :] > cont, payoff[t, :], vals[t+1, :])

    V = vals[1, :] * disc 
    sd = np.std(V, ddof=1)/np.sqrt(N)
    return np.mean(V), sd, [np.mean(V) - 1.96*sd, np.mean(V) + 1.96*sd]