"""Derivatives pricing under the Heston model: validation, Monte Carlo, smile, calibration, exotics.

The delta-hedging experiment is in run_hedging.py."""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src import black_scholes as bs
from src.heston import HestonParams, call_cos, call_fourier, greeks_vectorised, heston_delta
from src.monte_carlo import price_european, simulate_euler, simulate_qe

ROOT = Path(__file__).parent
TABLES, FIGS = ROOT / "results" / "tables", ROOT / "results" / "figures"
for folder in (TABLES, FIGS):
    folder.mkdir(parents=True, exist_ok=True)
NAVY, RED, GREY, GOLD, GREEN = "#1F3864", "#C0392B", "#7F8C8D", "#D4A017", "#2E7D32"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

P = HestonParams(kappa=1.5768, theta=0.0398, sigma=0.5751, rho=-0.5711, v0=0.0175)   # Fang & Oosterlee set
S0, K0 = 100.0, 100.0


def main():
    rng = np.random.default_rng(2026)

    # 1. Semi-analytic prices and Greeks vs published references
    refs = []
    for T, ref in [(1.0, 5.785155450), (10.0, 22.318945791)]:
        refs.append({"case": f"Call, T = {T:g}, r = 0 (Fang & Oosterlee, 2009)", "reference": ref,
                     "fourier": call_fourier(S0, K0, T, 0, 0, P), "cos_N256": call_cos(S0, K0, T, 0, 0, P, N=256)})
    r_nag = 0.025
    price_nag = call_fourier(S0, K0, 1, r_nag, 0, P)
    h = 0.01
    gamma_nag = (call_fourier(S0 + h, K0, 1, r_nag, 0, P) - 2 * price_nag + call_fourier(S0 - h, K0, 1, r_nag, 0, P)) / h ** 2
    rho_nag = (call_fourier(S0, K0, 1, r_nag + 1e-5, 0, P) - call_fourier(S0, K0, 1, r_nag - 1e-5, 0, P)) / 2e-5
    nag = pd.DataFrame([["Price", price_nag, 7.2743], ["Delta", heston_delta(S0, K0, 1, r_nag, 0, P), 0.6945],
                        ["Gamma", gamma_nag, 0.0251], ["Rho", rho_nag, 62.1735]],
                       columns=["quantity", "this_project", "NAG_library_r_2_5pct"])
    refs = pd.DataFrame(refs)
    refs["fourier_minus_reference"] = refs["fourier"] - refs["reference"]
    refs["cos_minus_fourier"] = refs["cos_N256"] - refs["fourier"]
    refs.to_csv(TABLES / "validation_semi_analytic.csv", index=False)
    nag.to_csv(TABLES / "validation_greeks_nag.csv", index=False)
    cos_conv = pd.DataFrame([{"N": N, **{f"error_T{T:g}": call_cos(S0, K0, T, 0, 0, P, N=N) - call_fourier(S0, K0, T, 0, 0, P)
                                         for T in (1.0, 10.0)}} for N in (64, 96, 128, 160, 192, 256)])
    cos_conv.to_csv(TABLES / "cos_convergence.csv", index=False)

    # 2. Monte Carlo: discretisation bias by scheme and time step, and variance reduction
    ref1 = call_fourier(S0, K0, 1, 0, 0, P)
    bias = []
    for steps in (12, 26, 52, 252):
        for name, sim in [("Euler full truncation", simulate_euler), ("Andersen QE", simulate_qe)]:
            paths = sim(S0, 1.0, 0, 0, P, steps, 200_000, rng)
            m, se = price_european(paths, K0, 1.0, 0, 0, S0)
            bias.append({"scheme": name, "steps_per_year": steps, "price": m, "bias": m - ref1, "std_error": se,
                         "bias_in_std_errors": (m - ref1) / se})
    bias = pd.DataFrame(bias)
    bias.to_csv(TABLES / "mc_discretisation_bias.csv", index=False)
    vr = []
    for label, anti, cv in [("Plain Monte Carlo", False, False), ("Antithetic", True, False),
                            ("Antithetic + control variate", True, True)]:
        paths = simulate_qe(S0, 1.0, 0, 0, P, 52, 200_000, rng, antithetic=anti)
        m, se = price_european(paths, K0, 1.0, 0, 0, S0, control_variate=cv, antithetic=anti)
        vr.append({"method": label, "price": m, "std_error": se})
    vr = pd.DataFrame(vr)
    vr["variance_reduction_factor"] = (vr["std_error"].iloc[0] / vr["std_error"]) ** 2
    vr.to_csv(TABLES / "mc_variance_reduction.csv", index=False)

    # Barrier pricing check under Black-Scholes: Monte Carlo vs closed form with the BGK correction
    vol_bs, B = 0.15, 80.0
    z = rng.standard_normal((100_000, 252))
    gbm = S0 * np.exp(np.cumsum((r_nag - 0.5 * vol_bs ** 2) / 252 + vol_bs * np.sqrt(1 / 252) * z, axis=1))
    alive = gbm.min(axis=1) > B
    pay = np.exp(-r_nag) * np.maximum(gbm[:, -1] - K0, 0) * alive
    barrier_mc, barrier_se = pay.mean(), pay.std(ddof=1) / np.sqrt(len(pay))
    barrier_cf = bs.down_and_out_call(S0, K0, bs.discrete_barrier_shift(B, vol_bs, 1 / 252), 1.0, r_nag, 0, vol_bs)

    # 3. Implied volatility smile and surface
    strikes = np.arange(70, 131, 2.5)
    maturities = [0.25, 0.5, 1.0, 2.0]
    # N = 512: deep out-of-the-money prices are tiny, so implied volatilities need very accurate prices
    surf = {T: [bs.implied_vol(c, S0, k, T, r_nag, 0) for k, c in zip(strikes, call_cos(S0, strikes, T, r_nag, 0, P, N=512))]
            for T in maturities}
    pd.DataFrame(surf, index=strikes).to_csv(TABLES / "implied_vol_surface.csv")
    sens = {}
    for label, pp in [("rho = -0.9", HestonParams(P.kappa, P.theta, P.sigma, -0.9, P.v0)),
                      ("rho = 0", HestonParams(P.kappa, P.theta, P.sigma, 0.0, P.v0)),
                      ("rho = +0.5", HestonParams(P.kappa, P.theta, P.sigma, 0.5, P.v0)),
                      ("vol of vol = 0.2", HestonParams(P.kappa, P.theta, 0.2, P.rho, P.v0)),
                      ("vol of vol = 1.0", HestonParams(P.kappa, P.theta, 1.0, P.rho, P.v0))]:
        sens[label] = [bs.implied_vol(c, S0, k, 1.0, r_nag, 0) for k, c in zip(strikes, call_cos(S0, strikes, 1.0, r_nag, 0, pp, N=512))]

    # 4. Calibration: recover known parameters from a noisy implied-volatility surface
    cal_T = [0.25, 0.5, 1.0, 2.0, 3.0]
    cal_K = np.arange(80, 121, 5.0)
    true_iv = np.array([[bs.implied_vol(c, S0, k, T, r_nag, 0) for k, c in zip(cal_K, call_cos(S0, cal_K, T, r_nag, 0, P))]
                        for T in cal_T])
    market_iv = true_iv + rng.normal(0, 0.002, true_iv.shape)        # 0.2 vol point of quote noise
    market_px = np.array([[bs.price(S0, k, T, r_nag, 0, iv) for k, iv in zip(cal_K, row)] for T, row in zip(cal_T, market_iv)])
    vegas = np.array([[bs.vega(S0, k, T, r_nag, 0, iv) for k, iv in zip(cal_K, row)] for T, row in zip(cal_T, market_iv)])

    def resid(x):
        pp = HestonParams(*x)
        model = np.array([call_cos(S0, cal_K, T, r_nag, 0, pp, N=160) for T in cal_T])
        return ((model - market_px) / vegas).ravel()                  # approximately implied-vol errors

    cal_rows = []
    for start in ([2.5, 0.06, 0.3, -0.2, 0.03], [0.8, 0.03, 0.8, -0.8, 0.01], [4.0, 0.05, 0.5, -0.5, 0.02]):
        t0 = time.time()
        fit = least_squares(resid, start, bounds=([0.05, 0.005, 0.05, -0.99, 0.002], [10, 0.3, 2.0, 0.99, 0.3]),
                            x_scale=[1, 0.01, 0.1, 0.1, 0.01])
        cal_rows.append({"start": str(start), **dict(zip(["kappa", "theta", "sigma", "rho", "v0"], fit.x)),
                         "rmse_vol_points": np.sqrt(np.mean(fit.fun ** 2)) * 100, "seconds": time.time() - t0})
    cal = pd.DataFrame(cal_rows)
    cal.loc[len(cal)] = {"start": "TRUE PARAMETERS", "kappa": P.kappa, "theta": P.theta, "sigma": P.sigma,
                         "rho": P.rho, "v0": P.v0, "rmse_vol_points": np.nan, "seconds": np.nan}
    cal.to_csv(TABLES / "calibration_recovery.csv", index=False)

    # 5. Model risk on exotics: Heston vs Black-Scholes with the same 1-year ATM implied volatility
    atm_iv = bs.implied_vol(price_nag, S0, K0, 1.0, r_nag, 0)
    hp = simulate_qe(S0, 1.0, r_nag, 0, P, 252, 200_000, rng)
    bp = S0 * np.exp(np.cumsum((r_nag - 0.5 * atm_iv ** 2) / 252
                               + atm_iv * np.sqrt(1 / 252) * rng.standard_normal((200_000, 252)), axis=1))
    bp = np.c_[np.full(200_000, S0), bp]
    exotic_rows = []
    for name, payoff in [("European call (K = 100)", lambda x: np.maximum(x[:, -1] - K0, 0)),
                         ("Down-and-out call (K = 100, B = 80, daily)", lambda x: np.maximum(x[:, -1] - K0, 0) * (x.min(axis=1) > B)),
                         ("Down-and-in call (K = 100, B = 80, daily)", lambda x: np.maximum(x[:, -1] - K0, 0) * (x.min(axis=1) <= B)),
                         ("Asian call, daily arithmetic average", lambda x: np.maximum(x[:, 1:].mean(axis=1) - K0, 0)),
                         ("Out-of-the-money put (K = 80)", lambda x: np.maximum(80 - x[:, -1], 0))]:
        ph, pb = np.exp(-r_nag) * payoff(hp), np.exp(-r_nag) * payoff(bp)
        half = len(ph) // 2                                             # antithetic pairs
        pair = 0.5 * (ph[:half] + ph[half:])
        exotic_rows.append({"product": name, "heston": ph.mean(), "heston_se": pair.std(ddof=1) / np.sqrt(half),
                            "black_scholes_flat_vol": pb.mean(), "difference_pct": ph.mean() / pb.mean() - 1})
    exotics = pd.DataFrame(exotic_rows)
    exotics.to_csv(TABLES / "exotics_model_risk.csv", index=False)

    validation = pd.DataFrame([
        ["Fourier vs Fang-Oosterlee, T = 1", refs.loc[0, "fourier_minus_reference"]],
        ["Fourier vs Fang-Oosterlee, T = 10", refs.loc[1, "fourier_minus_reference"]],
        ["COS (N = 256) vs Fourier, T = 1", refs.loc[0, "cos_minus_fourier"]],
        ["COS (N = 256) vs Fourier, T = 10", refs.loc[1, "cos_minus_fourier"]],
        ["Put-call parity of Black-Scholes (max error)", abs(bs.price(S0, 90, 1, r_nag, 0, 0.2) - bs.price(S0, 90, 1, r_nag, 0, 0.2, False)
                                                          - (S0 - 90 * np.exp(-r_nag)))],
        ["Barrier under BS: Monte Carlo (daily) minus closed form with BGK correction", barrier_mc - barrier_cf],
        ["Barrier under BS: Monte Carlo standard error", barrier_se],
        ["Heston 1y ATM implied volatility (r = 2.5%)", atm_iv],
        ["dC/dv0 at the NAG point (NAG 'vega' 52.5461)", float(greeks_vectorised(np.array([S0]), np.array([P.v0]), K0, 1.0, r_nag, 0, P)[1][0])],
    ], columns=["check", "value"])
    validation.to_csv(TABLES / "validation_summary.csv", index=False)

    # 7. Charts
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for T, c in zip(maturities, [RED, GOLD, NAVY, GREEN]):
        axes[0].plot(strikes, np.array(surf[T]) * 100, color=c, label=f"T = {T:g}y")
    axes[0].set_xlabel("Strike"); axes[0].set_ylabel("Implied volatility (%)"); axes[0].set_title("Heston implied volatility smile by maturity")
    axes[0].legend(frameon=False, fontsize=8)
    for (label, vals), c in zip(sens.items(), [NAVY, GREY, RED, GOLD, GREEN]):
        axes[1].plot(strikes, np.array(vals) * 100, color=c, label=label)
    axes[1].plot(strikes, np.array(surf[1.0]) * 100, color="black", ls="--", label="base")
    axes[1].set_xlabel("Strike"); axes[1].set_title("1-year smile: correlation drives skew, vol of vol drives curvature")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "implied_vol_smile.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    for name, c in [("Euler full truncation", RED), ("Andersen QE", NAVY)]:
        b = bias[bias["scheme"] == name]
        ax.errorbar(b["steps_per_year"], b["bias"], yerr=1.96 * b["std_error"], fmt="o-", color=c, capsize=3, label=name)
    ax.axhline(0, color=GREY, lw=0.8); ax.set_xscale("log")
    ax.set_xlabel("Time steps per year (log scale)"); ax.set_ylabel("Price bias vs semi-analytic (with 95% CI)")
    ax.set_title("Monte Carlo discretisation bias: Euler vs Andersen QE")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "mc_bias.png", dpi=150); plt.close(fig)

    # Console summary
    pd.set_option("display.width", 220)
    print("Semi-analytic validation\n" + refs.to_string(index=False))
    print("\nGreeks vs NAG\n" + nag.round(4).to_string(index=False))
    print("\nCOS convergence\n" + cos_conv.to_string(index=False))
    print("\nMC bias\n" + bias.round(4).to_string(index=False))
    print("\nVariance reduction\n" + vr.round(4).to_string(index=False))
    print("\nValidation summary\n" + validation.to_string(index=False))
    print(f"Barrier MC {barrier_mc:.4f} vs closed form (BGK) {barrier_cf:.4f}")
    print("\nCalibration\n" + cal.round(4).to_string(index=False))
    print("\nExotics\n" + exotics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
