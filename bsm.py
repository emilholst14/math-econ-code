import numpy as np 
from scipy.stats import norm

# Option on non-dividend paying stock

class BSM:
    def __init__(self, S0, K, T, r, sigma, kind):
        assert kind == 'call' or kind == 'put', 'Wrong option type'
        self.S0 = S0 
        self.K = K 
        self.T = T
        self.r = r 
        self.sigma = sigma
        self.kind = kind
        self.d1 = (np.log(S0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        self.d2 = self.d1 - sigma * np.sqrt(T)

    def price(self):
        if self.kind == 'call':
            return (self.S0 * norm.cdf(self.d1) 
                    - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2))
        if self.kind == 'put':
            return (self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) 
                    - self.S0 * norm.cdf(-self.d1))
        
    # Greeks
    def delta(self):
        if self.kind == 'call':
            return norm.cdf(self.d1)
        if self.kind == 'put':
            return norm.cdf(-self.d1)

    def gamma(self):
        return norm.pdf(self.d1) / (self.S0 * self.sigma * np.sqrt(self.T))

    def theta(self):
        if self.kind == 'call':
            return (self.S0 * norm.pdf(self.d1) * self.sigma / (2 * np.sqrt(self.T))
                    - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2))
        if self.kind == 'put':
            return (self.S0 * norm.pdf(self.d1) * self.sigma / (2 * np.sqrt(self.T))
                    + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2))
    
    def vega(self):
        return self.S0 * norm.pdf(self.d1) * np.sqrt(self.T)
    
    def greeks(self):
        return self.delta(), self.gamma(), self.theta(), self.vega()