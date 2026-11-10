"""Black-Scholes prices, Greeks and implied volatility (continuous dividend yield q)."""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1d2(S, K, T, r, q, vol):
    d1 = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    return d1, d1 - vol * np.sqrt(T)


def price(S, K, T, r, q, vol, call=True):
    d1, d2 = _d1d2(S, K, T, r, q, vol)
    if call:
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def delta(S, K, T, r, q, vol, call=True):
    d1, _ = _d1d2(S, K, T, r, q, vol)
    return np.exp(-q * T) * (norm.cdf(d1) if call else norm.cdf(d1) - 1)


def gamma(S, K, T, r, q, vol):
    d1, _ = _d1d2(S, K, T, r, q, vol)
    return np.exp(-q * T) * norm.pdf(d1) / (S * vol * np.sqrt(T))


def vega(S, K, T, r, q, vol):
    d1, _ = _d1d2(S, K, T, r, q, vol)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def implied_vol(p, S, K, T, r, q, call=True):
    """Implied volatility by Brent's method; NaN if the price is outside no-arbitrage bounds."""
    intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) if call else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
    if p <= intrinsic + 1e-12:
        return np.nan
    return brentq(lambda v: price(S, K, T, r, q, v, call) - p, 1e-6, 5.0, xtol=1e-12)


def down_and_out_call(S, K, B, T, r, q, vol):
    """Continuously monitored down-and-out call with barrier B <= K (Reiner & Rubinstein, 1991)."""
    lam = (r - q + 0.5 * vol ** 2) / vol ** 2
    y = np.log(B ** 2 / (S * K)) / (vol * np.sqrt(T)) + lam * vol * np.sqrt(T)
    down_in = (S * np.exp(-q * T) * (B / S) ** (2 * lam) * norm.cdf(y)
               - K * np.exp(-r * T) * (B / S) ** (2 * lam - 2) * norm.cdf(y - vol * np.sqrt(T)))
    return price(S, K, T, r, q, vol, True) - down_in


def discrete_barrier_shift(B, vol, dt, down=True):
    """Broadie-Glasserman-Kou (1997) continuity correction for discretely monitored barriers."""
    return B * np.exp((-1 if down else 1) * 0.5826 * vol * np.sqrt(dt))
