import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import beta
from scipy import stats

def fitter(xs, Fs, name, x_label=None, y_label=None, plot_error=True, log_scale=False, minmax=False):
    """
    Function called by the "Generating plot" button. Allows to get the plot and the parameters of the plot.

    Args:
        - xs: 1-D array with current dataset intensity values
        - Fs: 1-D array with current dataset vulnerability values
        - x-label/y-label: string with x/y-axis labels
        - plot_error: boolean that decides if the plot contains the errors curves

    Return:
        - 1-D array with the 4 parameters of the fit
        - fig the matplotlib figure object
        - ax the matplotlib ax object
    """
    xs = np.array(xs, dtype=float)
    Fs = np.array(Fs, dtype=float)
    a, b, xmin, xmax = fitting(xs, Fs)

    # Tracé
    support = np.linspace(xmin, xmax, 200)
    x_scaled = (support - xmin) / (xmax - xmin)
    x_scaled = np.clip(x_scaled, 0, 1)
    modelF = beta.cdf(x_scaled, a, b)
    varup, vardown = get_vars(modelF, xs, Fs, params=[a,b,xmin,xmax])

    #params_high, params_low = bootstrap(xs, Fs, x_scaled, modelF, n_boot=10000)
    #params_high, params_low = bootstrap_param([a,b,xmin,xmax], modelF, n_boot=10000)
    #x_scaled_low = (support - params_low[2]) / (params_low[3] - params_low[2])
    #x_scaled_high = (support - params_high[2]) / (params_high[3] - params_high[2])

    fig, ax = plt.subplots()
    ax.plot(xs, Fs, 'xb', label="Data")
    ax.plot(support, modelF, '-r', label="Beta CDF fit")
    if plot_error == True:
    #    ax.fill_between(support, beta.cdf(x_scaled_high, params_low[0], params_low[1]), beta.cdf(x_scaled_low, params_high[0], params_high[1]), alpha=0.25)
        ax.fill_between(support, varup, vardown, alpha=0.25)
        ax.plot(support, varup, '--b', alpha=0.5, label='Boundaries')
        ax.plot(support, vardown, '--b', alpha=0.5)
    if log_scale == True:
        ax.set_xscale('log')
    if minmax == True:
        [support_max, modelF_max], params_max, [support_min, modelF_min], params_min= fit_minmax(a,b,xmin,xmax)
        ax.plot(support_max, modelF_max, '--r', label='Maximum and Minimum')
        ax.plot(support_min, modelF_min, '--r')
    ax.set_title(name)
    ax.legend()
    ax.grid(True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if minmax == True:
        return np.array([a, b, xmin, xmax]), np.array(params_max), np.array(params_min), fig, ax
    else:
        return np.array([a, b, xmin, xmax]), fig, ax

def manual_fig(params, xs, Fs, name, x_label = None, y_label = None, plot_error=True, log_scale=True, minmax=False):
    a, b, xmin, xmax = params[0], params[1], params[2], params[3]
    support = np.linspace(xmin, xmax, 200)
    x_scaled = (support - xmin) / (xmax - xmin)
    x_scaled = np.clip(x_scaled, 0, 1)
    modelF = beta.cdf(x_scaled, a, b)
    varup, vardown = get_vars(modelF, xs, Fs, params=[a,b,xmin,xmax])
    
    fig, ax = plt.subplots()
    ax.plot(xs, Fs, 'xb', label="Data")
    ax.plot(support, modelF, '-r', label="Beta CDF fit")
    if plot_error==True:
        ax.fill_between(support, varup, vardown, alpha=0.25)
        ax.plot(support, varup, '--b', alpha=0.5, label='Boundaries')
        ax.plot(support, vardown, '--b', alpha=0.5)
    if log_scale == True:
        ax.set_xscale('log')
    if minmax == True:
        [support_max, modelF_max], _ , [support_min, modelF_min], _ = fit_minmax(a,b,xmin,xmax)
        ax.plot(support_max, modelF_max, '--r', label='Maximum and Minimum')
        ax.plot(support_min, modelF_min, '--r')
    ax.set_title(name)
    ax.legend()
    ax.grid(True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    return fig

def get_vars(modelF, xs, Fs, params):
    a, b, xmin, xmax = params[0], params[1], params[2], params[3]
    xs = np.asarray(xs, dtype=float)
    Fs = np.asarray(Fs, dtype=float)

    x_scaled = (xs - xmin) / (xmax - xmin)
    x_scaled = np.clip(x_scaled, 0, 1)

    distances = (Fs - beta.cdf(x_scaled, a, b))
    norm_params = stats.norm.fit(distances)
    vars = (stats.norm.ppf(0.99, norm_params[0], norm_params[1]))*np.ones(len(modelF))

    varup = modelF + vars
    varup[varup>1] = 1
    vardown = modelF - vars
    vardown[vardown<0] = 0
    return varup, vardown

def bootstrap(xs,Fs, Fmodel, n_boot=10000):
    params_all = np.zeros(shape=[n_boot,4])
    mses = np.zeros(shape=n_boot)
    for i in range(n_boot):
        indexes = np.random.choice(len(xs), size=len(xs), replace=True)
        xtemp = xs[indexes]
        Ftemp = Fs[indexes]
        params_all[i] = fitting(xtemp, Ftemp)
        mses[i] = deviation(params_boot=params_all[i], Fmodel=Fmodel)
    #q95 = np.quantile(mses, 0.95)
    #q05 = np.quantile(mses, 0.05)
    #print(np.argmin(np.abs(mses - q95)))
    params_high = params_all[np.argmax(mses), :]
    params_low = params_all[np.argmin(mses), :]
    return params_high, params_low


def cumul(data):
    n = len(data)
    return 1-np.linspace(0,1,n)

def bootstrap_param(params_fit, Fmodel, n_boot=10000):
    params_all = np.zeros(shape=[n_boot,4])
    mses = np.zeros(shape=n_boot)
    for i in range(n_boot):
        beta_samples = beta.ppf(np.random.uniform(0,1,200), params_fit[0], params_fit[1])*params_fit[3]+params_fit[2]
        F_sampled = 1 - cumul(np.sort(beta_samples))
        params_all[i] = fitting(np.sort(beta_samples), F_sampled)
        mses[i] = deviation(params_boot=params_all[i], Fmodel=Fmodel)
    params_high = params_all[np.argmax(mses), :]
    params_low = params_all[np.argmin(mses), :]
    return params_high, params_low

def fitting(xs,Fs):
    # Détection des bornes connues
    has_xmin = np.any(Fs == 0)
    has_xmax = np.any(Fs == 1)

    xmin_known = xs[Fs == 0].min() if has_xmin else None
    xmax_known = xs[Fs == 1].max() if has_xmax else None

    xmin_data, xmax_data = np.min(xs), np.max(xs)

    # Fonction de perte
    def loss(params):
        if has_xmin and has_xmax:
            a, b = params
            xmin, xmax = xmin_known, xmax_known
        elif has_xmin and not has_xmax:
            a, b, xmax = params
            xmin = xmin_known
        elif not has_xmin and has_xmax:
            a, b, xmin = params
            xmax = xmax_known
        else:
            a, b, xmin, xmax = params

        # Contraintes physiques
        if a <= 0 or b <= 0 or xmax <= xmin:
            return 1e6

        # Transformation
        x_scaled = (xs - xmin) / (xmax - xmin)
        x_scaled = np.clip(x_scaled, 0, 1)
        modelF = beta.cdf(x_scaled, a, b)
        mse = np.mean((modelF - Fs) ** 2)

        # Pénalités si la CDF ne couvre pas bien 0-1
        pen_left = (beta.cdf(0, a, b))**2
        pen_right = (1 - beta.cdf(1, a, b))**2  
        penalty = 0.1 * (pen_left + pen_right)

        return mse + penalty

    # Initial guess et bornes
    xmin0, xmax0 = xmin_data, xmax_data
    init = [1.0, 1.0]
    bounds = [(1e-3, None), (1e-3, None)]

    #margin = 0.1 * (xmax_data - xmin_data)  # marge 10%
    margin_down = 0.1 * xmin_data
    margin_up = 0.1 * xmax_data

    if has_xmin and not has_xmax:
        init += [xmax_data + margin_up]
        bounds += [(xmax_data, xmax_data + 10*margin_up)]
    elif not has_xmin and has_xmax:
        init += [xmin_data - margin_down]
        if xmin_data - 10 * margin_down > 0:
            bounds += [(xmin_data - 10 * margin_down, xmin_data)]
        else:
            bounds += [(0, xmin_data)]
    elif not has_xmin and not has_xmax:
        init += [xmin_data - margin_down, xmax_data + margin_up]
        if xmin_data - 10 * margin_down > 0:
            bounds += [
                (xmin_data - 10 * margin_down, xmin_data),
                (xmax_data, xmax_data + 10*margin_up),
            ]
        else: 
            bounds += [
                (0, xmin_data),
                (xmax_data, xmax_data + 10*margin_up),
            ]

    res = minimize(loss, x0=init, bounds=bounds, method='L-BFGS-B')
    p = res.x

    if has_xmin and has_xmax:
        a, b = p
        xmin, xmax = xmin_known, xmax_known
    elif has_xmin and not has_xmax:
        a, b, xmax = p
        xmin = xmin_known
    elif not has_xmin and has_xmax:
        a, b, xmin = p
        xmax = xmax_known
    else:
        a, b, xmin, xmax = p
    return np.array([a, b, xmin, xmax])

def deviation(params_boot, Fmodel):
    support = np.linspace(params_boot[2], params_boot[3], 200)
    x_scaled_boot = (support - params_boot[2])/(params_boot[3]-params_boot[2])
    Fboot = beta.cdf(x_scaled_boot, params_boot[0], params_boot[1])
    mse = 1/len(Fmodel)*np.mean((Fboot-Fmodel))
    return mse

def fit_minmax(a,b,xmin,xmax):
    quantiles = beta.ppf(np.array([0,0.25,0.5,0.75,1]),a,b,loc=xmin,scale=(xmax-xmin))
    a_max, b_max, xmin_max, xmax_max = fitting(quantiles[:-1],np.array([0.25,0.5,0.75,1]))
    a_min, b_min, xmin_min, xmax_min = fitting(quantiles[1:],np.array([0,0.25,0.5,0.75]))

    support_max = np.linspace(xmin_max, xmax_max, 200)
    x_scaled_max = (support_max - xmin_max) / (xmax_max - xmin_max)
    x_scaled_max = np.clip(x_scaled_max, 0, 1)
    modelF_max = beta.cdf(x_scaled_max, a_max, b_max)

    support_min = np.linspace(xmin_min, xmax_min, 200)
    x_scaled_min = (support_min - xmin_min) / (xmax_min - xmin_min)
    x_scaled_min = np.clip(x_scaled_min, 0, 1)
    modelF_min = beta.cdf(x_scaled_min, a_min, b_min)
    return [support_max, modelF_max], [a_max, b_max, xmin_max, xmax_max], [support_min, modelF_min], [a_min, b_min, xmin_min, xmax_min]