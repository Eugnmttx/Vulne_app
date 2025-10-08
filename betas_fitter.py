import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import beta

def fitter(xs, Fs, name):
    xs = np.asarray(xs, dtype=float)
    Fs = np.asarray(Fs, dtype=float)

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
        pen_left = (beta.cdf(0, a, b))**2       # veut Fs≈0 à gauche
        pen_right = (1 - beta.cdf(1, a, b))**2  # veut Fs≈1 à droite
        penalty = 0.1 * (pen_left + pen_right)

        return mse + penalty

    # Initial guess et bornes
    xmin0, xmax0 = xmin_data, xmax_data
    init = [1.0, 1.0]
    bounds = [(1e-3, None), (1e-3, None)]

    margin = 0.1 * (xmax_data - xmin_data)  # marge 10%

    if has_xmin and not has_xmax:
        init += [xmax_data + margin]
        bounds += [(xmax_data, xmax_data + 10*margin)]
    elif not has_xmin and has_xmax:
        init += [xmin_data - margin]
        if xmin_data - 10 * margin > 0:
            bounds += [(xmin_data - 10 * margin, xmin_data)]
        else:
            bounds += [(0, xmin_data)]
    elif not has_xmin and not has_xmax:
        init += [xmin_data - margin, xmax_data + margin]
        if xmin_data - 10 * margin > 0:
            bounds += [
                (xmin_data - 10 * margin, xmin_data),
                (xmax_data, xmax_data + 10*margin),
            ]
        else: 
            bounds += [
                (0, xmin_data),
                (xmax_data, xmax_data + 10*margin),
            ]
    # Optimisation
    res = minimize(loss, x0=init, bounds=bounds, method='L-BFGS-B')
    p = res.x

    # Extraction des résultats
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

    # Tracé
    support = np.linspace(xmin, xmax, 200)
    x_scaled = (support - xmin) / (xmax - xmin)
    x_scaled = np.clip(x_scaled, 0, 1)
    modelF = beta.cdf(x_scaled, a, b)

    fig, ax = plt.subplots()
    ax.plot(xs, Fs, 'xb', label="Données")
    ax.plot(support, modelF, '-r', label="Beta CDF fit")
    ax.set_title(name)
    ax.legend()
    ax.grid(True)

    return np.array([a, b, xmin, xmax]), fig, ax

def manual_fig(params, xs, Fs, name):
    a, b, xmin, xmax = params[0], params[1], params[2], params[3]
    support = np.linspace(xmin, xmax, 200)
    x_scaled = (support - xmin) / (xmax - xmin)
    x_scaled = np.clip(x_scaled, 0, 1)
    modelF = beta.cdf(x_scaled, a, b)

    fig, ax = plt.subplots()
    ax.plot(xs, Fs, 'xb', label="Données")
    ax.plot(support, modelF, '-r', label="Beta CDF fit")
    ax.set_title(name)
    ax.legend()
    ax.grid(True)

    return fig