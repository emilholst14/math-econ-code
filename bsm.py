import numpy as np 
from scipy.stats import norm

# Option on non-dividend paying stock

class BSM:
    def __init__(self, S, K, T, r, sigma, kind):
        assert kind == 'call' or kind == 'put', 'Wrong option type'
        self.S = S 
        self.K = K 
        self.T = T
        self.r = r 
        self.sigma = sigma
        self.kind = kind
        self.d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        self.d2 = self.d1 - sigma * np.sqrt(T)

    def price(self):
        if self.kind == 'call':
            return (self.S * norm.cdf(self.d1) 
                    - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2))
        if self.kind == 'put':
            return (self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) 
                    - self.S * norm.cdf(-self.d1))
        
    # Greeks
    def delta(self):
        if self.kind == 'call':
            return norm.cdf(self.d1)
        if self.kind == 'put':
            return norm.cdf(-self.d1)

    def gamma(self):
        return norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))

    def theta(self):
        if self.kind == 'call':
            return (self.S * norm.pdf(self.d1) * self.sigma / (2 * np.sqrt(self.T))
                    - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2))
        if self.kind == 'put':
            return (self.S * norm.pdf(self.d1) * self.sigma / (2 * np.sqrt(self.T))
                    + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2))
    
    def vega(self):
        return self.S * norm.pdf(self.d1) * np.sqrt(self.T)
    
    def greeks(self):
        return self.delta(), self.gamma(), self.theta(), self.vega()
    
# Implied volatility
def implied_vol(obs, S, K, T, r, kind, max_iter=200, tol=1e-5):
    assert kind == 'call' or kind == 'put', 'Wrong option type'
    vol = 0.5
    for i in range(max_iter):
        opt = BSM(S, K, T, r, vol, kind)
        diff = obs - opt.price()
        if abs(diff) < tol:
            return vol
        vol = vol + diff / opt.vega()
    return vol 
