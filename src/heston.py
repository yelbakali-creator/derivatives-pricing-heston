"""Heston (1993) stochastic volatility model: semi-analytic European option prices.

    dS = (r - q) S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + sigma sqrt(v) dW2,   corr(dW1, dW2) = rho

Two independent pricing methods:
* Fourier inversion of the probabilities P1, P2 (Heston's formula), using the "little Heston
  trap" form of the characteristic function (Albrecher et al., 2007), which avoids branch-cut
  discontinuities of the complex logarithm;
* the COS method (Fang & Oosterlee, 2009): a Fourier-cosine expansion of the density.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import quad


@dataclass
class HestonParams:
    kappa: float
    theta: float
    sigma: float
    rho: float
    v0: float


def _cf_j(phi, j, S, T, r, q, p: HestonParams):
    """Characteristic functions f_1, f_2 of Heston's formula (little-trap form)."""
    u = 0.5 if j == 1 else -0.5
    b = p.kappa - p.rho * p.sigma if j == 1 else p.kappa
    a = p.kappa * p.theta
    rsi = p.rho * p.sigma * 1j * phi
    d = np.sqrt((rsi - b) ** 2 - p.sigma ** 2 * (2 * u * 1j * phi - phi ** 2))
    g = (b - rsi - d) / (b - rsi + d)
    e = np.exp(-d * T)
    C = (r - q) * 1j * phi * T + a / p.sigma ** 2 * ((b - rsi - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    D = (b - rsi - d) / p.sigma ** 2 * (1 - e) / (1 - g * e)
    return np.exp(C + D * p.v0 + 1j * phi * np.log(S))


def probabilities(S, K, T, r, q, p):
    """P1 and P2 by adaptive quadrature over successive finite pieces: integrating directly to
    infinity can stop too early when the integrand decays slowly (short maturity, low variance)."""
    edges = [1e-10, 10, 50, 100, 300, 1000, 3000, np.inf]
    out = []
    for j in (1, 2):
        integrand = lambda phi: (np.exp(-1j * phi * np.log(K)) * _cf_j(phi, j, S, T, r, q, p) / (1j * phi)).real
        val = sum(quad(integrand, a, b, limit=500, epsabs=1e-14, epsrel=1e-12)[0] for a, b in zip(edges[:-1], edges[1:]))
        out.append(0.5 + val / np.pi)
    return out


def call_fourier(S, K, T, r, q, p):
    P1, P2 = probabilities(S, K, T, r, q, p)
    return S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2


def heston_delta(S, K, T, r, q, p):
    """Heston call delta = exp(-qT) P1 (exact)."""
    return np.exp(-q * T) * probabilities(S, K, T, r, q, p)[0]


# ---------- COS method ----------
def cf_logreturn(u, T, r, q, p):
    """Characteristic function of ln(S_T / S_0)."""
    rsi = p.rho * p.sigma * 1j * u
    d = np.sqrt((rsi - p.kappa) ** 2 + p.sigma ** 2 * (1j * u + u ** 2))
    g = (p.kappa - rsi - d) / (p.kappa - rsi + d)
    e = np.exp(-d * T)
    C = (r - q) * 1j * u * T + p.kappa * p.theta / p.sigma ** 2 * ((p.kappa - rsi - d) * T
                                                                  - 2 * np.log((1 - g * e) / (1 - g)))
    D = (p.kappa - rsi - d) / p.sigma ** 2 * (1 - e) / (1 - g * e)
    return np.exp(C + D * p.v0)


def _cumulants(T, r, q, p, h=1e-4, h4=2e-2):
    """Cumulants c1, c2, c4 of ln(S_T/S_0) from finite differences of the log characteristic function
    (k-th cumulant = k-th derivative of ln phi at 0, divided by i^k)."""
    lc = lambda u: np.log(cf_logreturn(np.array([u]), T, r, q, p))[0]
    c1 = ((lc(h) - lc(-h)) / (2j * h)).real
    c2 = (-(lc(h) - 2 * lc(0.0) + lc(-h)) / h ** 2).real
    d4 = (lc(2 * h4) - 4 * lc(h4) + 6 * lc(0.0) - 4 * lc(-h4) + lc(-2 * h4)) / h4 ** 4
    c4 = d4.real                                   # i^4 = 1
    return c1, c2, c4


def call_cos(S, K, T, r, q, p, N=192, L=12):
    """European call via the COS method: put priced by the cosine expansion, then put-call parity."""
    K = np.atleast_1d(np.asarray(K, float))
    c1, c2, c4 = _cumulants(T, r, q, p)
    width = L * np.sqrt(abs(c2) + np.sqrt(abs(c4)))  # truncation range of Fang & Oosterlee (2009)
    a, b = c1 - width, c1 + width
    k = np.arange(N)
    w = k * np.pi / (b - a)

    def chi(c, d):
        return (np.cos(w * (d - a)) * np.exp(d) - np.cos(w * (c - a)) * np.exp(c)
                + w * np.sin(w * (d - a)) * np.exp(d) - w * np.sin(w * (c - a)) * np.exp(c)) / (1 + w ** 2)

    def psi(c, d):
        out = np.empty(N)
        out[0] = d - c
        out[1:] = (np.sin(w[1:] * (d - a)) - np.sin(w[1:] * (c - a))) / w[1:]
        return out

    U_put = 2 / (b - a) * (-chi(a, 0.0) + psi(a, 0.0))
    cf = cf_logreturn(w, T, r, q, p)
    x = np.log(S / K)                                   # log-moneyness, one per strike
    terms = (cf[None, :] * np.exp(1j * np.outer(x - a, w))).real * U_put[None, :]
    terms[:, 0] *= 0.5
    put = K * np.exp(-r * T) * terms.sum(axis=1)
    call = put + S * np.exp(-q * T) - K * np.exp(-r * T)
    return call if len(call) > 1 else float(call[0])


def delta_vectorised(S, v, K, tau, r, q, p, upper=None, n_nodes=None):
    """Heston call delta exp(-q tau) P1 for arrays of spot S and variance v (same shape), by
    Gauss-Legendre quadrature of the P1 integral. Used for path-wise delta hedging."""
    S, v = np.asarray(S, float), np.asarray(v, float)
    # The integrand can decay slowly (low variance, short maturity): use a wide range, wider near expiry
    upper = 1000.0 * max(1.0, 1.0 / np.sqrt(10 * tau)) if upper is None else upper
    n_nodes = int(np.ceil(min(6000, 2 * upper) / 250) * 250) if n_nodes is None else n_nodes
    x, w = _gauss_legendre(n_nodes)
    phi = 0.5 * upper * (x + 1) + 1e-8
    w = 0.5 * upper * w
    b = p.kappa - p.rho * p.sigma
    rsi = p.rho * p.sigma * 1j * phi
    d = np.sqrt((rsi - b) ** 2 - p.sigma ** 2 * (1j * phi - phi ** 2))
    g = (b - rsi - d) / (b - rsi + d)
    e = np.exp(-d * tau)
    C = (r - q) * 1j * phi * tau + p.kappa * p.theta / p.sigma ** 2 * ((b - rsi - d) * tau - 2 * np.log((1 - g * e) / (1 - g)))
    D = (b - rsi - d) / p.sigma ** 2 * (1 - e) / (1 - g * e)
    lm = np.log(S / K).ravel()[:, None]
    f = np.exp(C[None, :] + D[None, :] * v.ravel()[:, None] + 1j * phi[None, :] * lm)
    P1 = 0.5 + ((f / (1j * phi[None, :])).real @ w) / np.pi
    return (np.exp(-q * tau) * np.clip(P1, 0, 1)).reshape(S.shape)


@lru_cache(maxsize=64)
def _gauss_legendre(n):
    """Gauss-Legendre nodes and weights on [-1, 1], cached: computing them costs O(n^3)."""
    return np.polynomial.legendre.leggauss(n)


def _integration_upper(v_min, tau, r, q, p, tol=1e-13):
    """Smallest phi beyond which |f_1(phi)| < tol, for the lowest variance in the batch."""
    grid = np.geomspace(50, 2e5, 120)
    pp = HestonParams(p.kappa, p.theta, p.sigma, p.rho, max(v_min, 0.0))
    mag = np.abs(np.array([_cf_j(g, 1, 1.0, tau, r, q, pp) for g in grid]))
    below = np.nonzero(mag < tol)[0]
    return float(grid[below[0]]) if len(below) else float(grid[-1])


def greeks_vectorised(S, v, K, tau, r, q, p, group_size=250):
    """Model delta, dC/dv and minimum-variance delta for arrays of (S, v); see _greeks_block.
    Paths are processed in groups sorted by variance, so that only low-variance paths (whose
    characteristic function decays slowly) pay for a wide integration range."""
    S, v = np.asarray(S, float), np.asarray(v, float)
    flat_S, flat_v = S.ravel(), v.ravel()
    order = np.argsort(flat_v)
    out = [np.empty(flat_S.shape) for _ in range(3)]
    for start in range(0, len(order), group_size):
        idx = order[start:start + group_size]
        res = _greeks_block(flat_S[idx], flat_v[idx], K, tau, r, q, p)
        for o, rr in zip(out, res):
            o[idx] = rr
    return tuple(o.reshape(S.shape) for o in out)


def _greeks_block(S, v, K, tau, r, q, p):
    """Model delta, variance sensitivity dC/dv and minimum-variance delta for arrays of (S, v).

    Minimum-variance delta (e.g. Bakshi, Cao & Chen, 1997): since dv is correlated with dS,
    the hedge that minimises the variance of the hedged position over dt is
        Delta_MV = Delta + (rho * sigma / S) * dC/dv.
    """
    S, v = np.asarray(S, float), np.asarray(v, float)
    # The Heston characteristic function can decay slowly (low variance, short maturity), so the range
    # is set adaptively: up to where |f_1| falls below 1e-13 for the slowest-decaying path (lowest v).
    upper = _integration_upper(np.min(v), tau, r, q, p)
    # Nodes: enough to resolve the oscillation exp(i phi ln(S/K)) across the range
    osc = max(1.0, np.max(np.abs(np.log(S / K))))
    n_nodes = int(np.ceil(np.clip(upper * osc / 1.5, 400, 5000) / 250) * 250)   # rounded for caching
    x, w = _gauss_legendre(n_nodes)
    phi = 0.5 * upper * (x + 1) + 1e-8
    w = 0.5 * upper * w
    lm = np.log(S / K).ravel()[:, None]
    vv = v.ravel()[:, None]
    dP = []
    for j in (1, 2):
        u = 0.5 if j == 1 else -0.5
        b = p.kappa - p.rho * p.sigma if j == 1 else p.kappa
        rsi = p.rho * p.sigma * 1j * phi
        d = np.sqrt((rsi - b) ** 2 - p.sigma ** 2 * (2 * u * 1j * phi - phi ** 2))
        g = (b - rsi - d) / (b - rsi + d)
        e = np.exp(-d * tau)
        C = (r - q) * 1j * phi * tau + p.kappa * p.theta / p.sigma ** 2 * ((b - rsi - d) * tau - 2 * np.log((1 - g * e) / (1 - g)))
        D = (b - rsi - d) / p.sigma ** 2 * (1 - e) / (1 - g * e)
        f = np.exp(C[None, :] + D[None, :] * vv + 1j * phi[None, :] * lm)
        integ = f / (1j * phi[None, :])
        if j == 1:
            P1 = 0.5 + (integ.real @ w) / np.pi
        dP.append(((integ * D[None, :]).real @ w) / np.pi)        # dP_j / dv
    Sf = S.ravel()
    delta = np.exp(-q * tau) * np.clip(P1, 0, 1)
    dCdv = Sf * np.exp(-q * tau) * dP[0] - K * np.exp(-r * tau) * dP[1]
    mv = delta + p.rho * p.sigma / Sf * dCdv
    return delta.reshape(S.shape), dCdv.reshape(S.shape), mv.reshape(S.shape)
