"""Validation tests: Heston prices against Fang & Oosterlee (2009) and NAG Greeks; COS vs Fourier; Black-Scholes."""
import numpy as np

from src import black_scholes as bs
from src.heston import HestonParams, call_cos, call_fourier, greeks_vectorised, heston_delta

P = HestonParams(kappa=1.5768, theta=0.0398, sigma=0.5751, rho=-0.5711, v0=0.0175)


def test_fourier_matches_fang_oosterlee():
    assert abs(call_fourier(100, 100, 1.0, 0, 0, P) - 5.785155450) < 1e-7
    assert abs(call_fourier(100, 100, 10.0, 0, 0, P) - 22.318945791) < 1e-7


def test_cos_matches_fourier():
    for T in (1.0, 10.0):
        assert abs(call_cos(100, 100, T, 0, 0, P, N=256) - call_fourier(100, 100, T, 0, 0, P)) < 1e-6


def test_greeks_match_nag():
    r = 0.025
    price = call_fourier(100, 100, 1.0, r, 0, P)
    assert round(price, 4) == 7.2743
    assert round(heston_delta(100, 100, 1.0, r, 0, P), 4) == 0.6945
    _, dcdv, _ = greeks_vectorised(np.array([100.0]), np.array([P.v0]), 100, 1.0, r, 0, P)
    assert round(float(dcdv[0]), 4) == 52.5461


def test_vectorised_delta_matches_reference_on_hard_case():
    pp = HestonParams(P.kappa, P.theta, P.sigma, P.rho, 0.01)          # deep ITM, low variance, short maturity
    d, _, _ = greeks_vectorised(np.array([110.0]), np.array([0.01]), 100, 0.1, 0.025, 0, pp)
    assert abs(float(d[0]) - heston_delta(110, 100, 0.1, 0.025, 0, pp)) < 1e-8


def test_black_scholes_parity_and_implied_vol():
    c, p = bs.price(100, 90, 1, 0.025, 0, 0.2), bs.price(100, 90, 1, 0.025, 0, 0.2, call=False)
    assert abs(c - p - (100 - 90 * np.exp(-0.025))) < 1e-12
    assert abs(bs.implied_vol(c, 100, 90, 1, 0.025, 0) - 0.2) < 1e-10
