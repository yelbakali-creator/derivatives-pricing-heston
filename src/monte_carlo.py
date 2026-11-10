"""Monte Carlo simulation of the Heston model.

* Euler with full truncation (Lord, Koekkoek & van Dijk, 2010): v+ = max(v, 0) in drift and diffusion.
* Quadratic-Exponential (QE) scheme of Andersen (2008) for the variance, with his discretisation of
  ln S (gamma1 = gamma2 = 1/2). QE matches the first two moments of the exact non-central chi-square
  transition and has much smaller bias for a given time step.
Variance reduction: antithetic variates, and a control variate on the discounted terminal price,
whose expectation S0 exp(-qT) is known exactly.
"""
import numpy as np


def simulate_euler(S0, T, r, q, p, n_steps, n_paths, rng, antithetic=True):
    dt = T / n_steps
    m = n_paths // 2 if antithetic else n_paths
    lnS = np.full(n_paths, np.log(S0)); v = np.full(n_paths, p.v0)
    paths = np.empty((n_paths, n_steps + 1)); paths[:, 0] = S0
    for k in range(n_steps):
        z1 = rng.standard_normal(m); z2 = rng.standard_normal(m)
        if antithetic:
            z1, z2 = np.r_[z1, -z1], np.r_[z2, -z2]
        zv = p.rho * z1 + np.sqrt(1 - p.rho ** 2) * z2
        vp = np.maximum(v, 0.0)
        lnS += (r - q - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1
        v += p.kappa * (p.theta - vp) * dt + p.sigma * np.sqrt(vp * dt) * zv
        paths[:, k + 1] = np.exp(lnS)
    return paths


def simulate_qe(S0, T, r, q, p, n_steps, n_paths, rng, antithetic=True, psi_c=1.5, return_var=False):
    dt = T / n_steps
    m = n_paths // 2 if antithetic else n_paths
    k, th, s, rho = p.kappa, p.theta, p.sigma, p.rho
    e = np.exp(-k * dt)
    g1 = g2 = 0.5
    K0 = -rho * k * th / s * dt
    K1 = g1 * dt * (k * rho / s - 0.5) - rho / s
    K2 = g2 * dt * (k * rho / s - 0.5) + rho / s
    K3 = g1 * dt * (1 - rho ** 2)
    K4 = g2 * dt * (1 - rho ** 2)
    lnS = np.full(n_paths, np.log(S0)); v = np.full(n_paths, p.v0)
    paths = np.empty((n_paths, n_steps + 1)); paths[:, 0] = S0
    vars_ = np.empty((n_paths, n_steps + 1)) if return_var else None
    if return_var:
        vars_[:, 0] = p.v0
    for step in range(n_steps):
        u = rng.random(m); z = rng.standard_normal(m)
        if antithetic:
            u, z = np.r_[u, 1 - u], np.r_[z, -z]
        mean = th + (v - th) * e
        var = v * s ** 2 * e / k * (1 - e) + th * s ** 2 / (2 * k) * (1 - e) ** 2
        psi = var / mean ** 2
        v_new = np.empty(n_paths)
        quad = psi <= psi_c
        if quad.any():                                   # quadratic branch: v' = a (b + Zv)^2
            ps = psi[quad]
            b2 = 2 / ps - 1 + np.sqrt(2 / ps) * np.sqrt(2 / ps - 1)
            a = mean[quad] / (1 + b2)
            zv = np.sqrt(2) * _erfinv(2 * u[quad] - 1)
            v_new[quad] = a * (np.sqrt(b2) + zv) ** 2
        ex = ~quad
        if ex.any():                                     # exponential branch: point mass at 0 + exponential
            ps = psi[ex]
            pp = (ps - 1) / (ps + 1)
            beta = (1 - pp) / mean[ex]
            uu = u[ex]
            v_new[ex] = np.where(uu <= pp, 0.0, np.log((1 - pp) / np.maximum(1 - uu, 1e-300)) / beta)
        lnS += (r - q) * dt + K0 + K1 * v + K2 * v_new + np.sqrt(K3 * v + K4 * v_new) * z
        v = v_new
        paths[:, step + 1] = np.exp(lnS)
        if return_var:
            vars_[:, step + 1] = v
    return (paths, vars_) if return_var else paths


def _erfinv(x):
    from scipy.special import erfinv
    return erfinv(np.clip(x, -1 + 1e-16, 1 - 1e-16))


def price_european(paths, K, T, r, q, S0, call=True, control_variate=True, antithetic=True):
    """Discounted payoff mean and standard error, optionally with the terminal-price control variate.

    With antithetic paths (first half and second half are mirror images), the standard error is
    computed on pair averages, since the two members of a pair are not independent."""
    ST = paths[:, -1]
    pay = np.exp(-r * T) * (np.maximum(ST - K, 0) if call else np.maximum(K - ST, 0))
    if control_variate:
        X = np.exp(-r * T) * ST
        c = np.cov(pay, X)
        b = c[0, 1] / c[1, 1]
        pay = pay - b * (X - S0 * np.exp(-q * T))
    if antithetic:
        half = len(pay) // 2
        pay = 0.5 * (pay[:half] + pay[half:])
    return pay.mean(), pay.std(ddof=1) / np.sqrt(len(pay))
