
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import beta

def fitter(xs, Fs):
    def betas_cdf(xs, Fs):
        xs = np.asarray(xs, dtype=float)
        Fs = np.asarray(Fs, dtype=float)

        xmin, xmax = np.min(xs), np.max(xs)
        x_scaled = (xs - xmin) / (xmax - xmin)

        def loss(params):
            a, b = params
            if a <= 0 or b <= 0:
                return 1.e6
            modelF = beta.cdf(x_scaled, a, b)
            return np.mean((modelF - Fs)**2)
        res = minimize(loss, x0=[1.0,1.0], bounds=[(1e-3, None), (1e-3, None)])
        a, b = res.x

        return a, b, xmin, (xmax - xmin)
    
    params = np.array(betas_cdf(xs, Fs))
    support = np.linspace(min(xs),max(xs),100)

    fig, ax = plt.subplots()
    ax.plot(xs, Fs, 'xb', label="Données")
    ax.plot(support, beta.cdf(support, params[0], params[1], 
                              loc=params[2], scale=params[3]), '-r',
            label="Beta CDF")
    ax.legend()
    ax.grid('on')
    params[3] = params[3] + params[2]
    return params, fig, ax
