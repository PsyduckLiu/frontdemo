import numpy as np
from scipy import stats

np.random.seed(42)


class Option:

    def __init__(self, S0, K, r, sigma, T, N):
        """
        :param  S0: initial price
        :param  K: strike price
        :param  r: risk-free interest rate
        :param  sigma: volatility 
        :param  T: time to maturity of the option
        :param  N: number of timesteps
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.N = N

    def contract(self, stock_prices):
        pass

    def simulate(self, n_sim):
        """Calculates the price of an option using Monte Carlo simulation.

        :param int n_sim: number of Monte Carlo simulations
        :return float: estimated price, upper confidence bound, lower confidence bound
        """

        dt = self.T / self.N

        # Generate stock price path
        Z = np.random.normal(size=(n_sim, self.N))
        S = self.S0 * np.exp(np.cumsum((self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z, axis=1))

        # Calculate option payoff
        payoff = self.contract(stock_prices=S)

        # Calculate option price
        option_price = np.exp(-self.r * self.T) * payoff

        C = np.mean(option_price)

        # Calculate confidence interval
        # s = option_price.std()
        # Cl = C - 1.96 * s / np.sqrt(n_sim)
        # Cu = C + 1.96 * s / np.sqrt(n_sim)

        return C


class VanillaOption(Option):

    def __init__(self, S0, K, r, sigma, T, N, OPtype):
        """
        :param  OPtype: 'call' or 'put'
        """
        super().__init__(S0=S0, K=K, r=r, sigma=sigma, T=T, N=N)
        self.OPtype = OPtype

    def contract(self, stock_prices):
        price = stock_prices[:, -1]

        if self.OPtype == 'call':
            return np.maximum(price - self.K, 0)
        elif self.OPtype == 'put':
            return np.maximum(self.K - price, 0)


class AsianPriceOption(Option):

    def __init__(self, S0, K, r, sigma, T, N, OPtype, metric='geometric'):
        """
        :param  OPtype: 'call' or 'put'
        :param  metric: 'arithmetic' or 'geometric'
        """
        super().__init__(S0=S0, K=K, r=r, sigma=sigma, T=T, N=N)
        self.OPtype = OPtype
        self.metric = metric

    def contract(self, stock_prices):
        # Calculate average price over time horizon
        if self.metric == 'arithmetic':
            avg_price = np.mean(stock_prices, axis=1)
        else:
            avg_price = stats.gmean(stock_prices, axis=1)

        if self.OPtype == 'call':
            return np.maximum(avg_price - self.K, 0)
        elif self.OPtype == 'put':
            return np.maximum(self.K - avg_price, 0)


class AsianStrikeOption(Option):

    def __init__(self, S0, r, sigma, T, N, OPtype, metric='geometric'):
        """
        Exercise price (strike price) is not provided at the beginning.

        :param  OPtype: 'call' or 'put'
        :param  metric: 'arithmetic' or 'geometric'
        """
        super().__init__(S0=S0, K=0, r=r, sigma=sigma, T=T, N=N)
        self.OPtype = OPtype
        self.metric = metric

    def contract(self, stock_prices):
        # Calculate average price over time horizon
        if self.metric == 'arithmetic':
            avg_price = np.mean(stock_prices, axis=1)
        else:
            avg_price = stats.gmean(stock_prices, axis=1)

        if self.OPtype == 'call':
            return np.maximum(stock_prices[:, self.N - 1] - avg_price, 0)
        elif self.OPtype == 'put':
            return np.maximum(avg_price - stock_prices[:, self.N - 1], 0)


class BarrierOption(Option):

    def __init__(self, S0, K, r, sigma, T, N, barrier, BarrierSide, KnockType, OPtype):
        """
        :param  barrier:     barrier price
        :param  BarrierSide: 'up' or 'down'
        :param  KnockType:   'in' or 'out'
        :param  OPtype:      'call' or 'put'
        """
        super().__init__(S0=S0, K=K, r=r, sigma=sigma, T=T, N=N)
        self.barrier = barrier
        self.BarrierSide = BarrierSide
        self.KnockType = KnockType
        self.OPtype = OPtype

    def contract(self, stock_prices):
        if self.KnockType == 'out':
            if self.BarrierSide == 'up':
                valid = (stock_prices < self.barrier).all(axis=1)
            else:
                valid = (stock_prices > self.barrier).all(axis=1)
        else:
            if self.BarrierSide == 'up':
                valid = (stock_prices > self.barrier).any(axis=1)
            else:
                valid = (stock_prices < self.barrier).any(axis=1)

        if self.OPtype == 'call':
            profit = np.maximum(stock_prices[:, -1] - self.K, 0)
        else:
            profit = np.maximum(self.K - stock_prices[:, -1], 0)
        return profit * valid


def Calculate_IV(S0, K, r, T, N, OPtype, price):
    """
    :param  price: observed price
    """
    # Calculate price when volatility = 0
    sigma_min = 0.0
    vanilla_option = VanillaOption(S0=S0, K=K, r=r, sigma=sigma_min, T=T, N=N, OPtype=OPtype)
    C_min, Cl, Cu = vanilla_option.simulate(n_sim=1000000)
    # Calculate price when volatility = 3
    # Simulation will not converge with volatility > 3 and steps = 1e6
    sigma_max = 3
    vanilla_option = VanillaOption(S0=S0, K=K, r=r, sigma=sigma_max, T=T, N=N, OPtype=OPtype)
    C_max, Cl, Cu = vanilla_option.simulate(n_sim=1000000)

    # Inproper price
    if price < C_min:
        return sigma_min
    if price > C_max:
        return sigma_max

    # Binary search
    while abs(sigma_max - sigma_min) > 1e-6:
        sigma = (sigma_min + sigma_max) / 2
        vanilla_option = VanillaOption(S0=S0, K=K, r=r, sigma=sigma, T=T, N=N, OPtype=OPtype)
        C, Cl, Cu = vanilla_option.simulate(n_sim=1000000)
        if C < price:
            sigma_min = sigma
            C_min = C
        else:
            sigma_max = sigma
            C_max = C
        print("sigma_min:%f, price_min:%f, sigma_max:%f, price_max:%f" % (sigma_min, C_min, sigma_max, C_max))
    return sigma


if __name__ == "__main__":
    # for i in range ( 10 ):
    #     vanilla_option = VanillaOption(S0=11, K=10, r=0.02, sigma=0.3, T=1, N=100, OPtype='call')
    #     C, Cl, Cu = vanilla_option.simulate(n_sim=1000000)
    #     print(C, Cl, Cu)

    # asian_option = AsianPriceOption(S0=11, K=10, r=0.02, sigma=0.3, T=1, N=100, OPtype='call', metric='geometric')
    # C, Cl, Cu = asian_option.simulate(n_sim=100000)
    # print(C, Cl, Cu)

    # asian_option = AsianStrikeOption(S0=11, r=0.02, sigma=0.3, T=1, N=100, OPtype='call', metric='geometric')
    # C, Cl, Cu = asian_option.simulate(n_sim=100000)
    # print(C, Cl, Cu)

    # up_out_call_option = BarrierOption(S0=11, K=10, r=0.02, sigma=0.3, T=1, N=100, barrier=12,BarrierSide='up',KnockType='out',OPtype='call')
    # C, Cl, Cu = up_out_call_option.simulate(n_sim=100000)
    # print(C, Cl, Cu)

    # down_out_call_option = BarrierOption(S0=11, K=10, r=0.02, sigma=0.3, T=1, N=100, barrier=9,BarrierSide='down',KnockType='out',OPtype='call')
    # C, Cl, Cu = down_out_call_option.simulate(n_sim=100000)
    # print(C, Cl, Cu)

    IV = Calculate_IV(S0=11, K=10, r=0.02, T=1, N=100, OPtype='call', price=1.92)
    print(IV)
