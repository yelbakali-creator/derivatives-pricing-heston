"""Delta hedging of a short call in a Heston world: Black-Scholes, Heston model and minimum-variance deltas."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import black_scholes as bs
from src.heston import HestonParams, call_fourier, greeks_vectorised
from src.monte_carlo import simulate_qe

ROOT = Path(__file__).parent
TABLES, FIGS = ROOT / "results" / "tables", ROOT / "results" / "figures"
for folder in (TABLES, FIGS):
    folder.mkdir(parents=True, exist_ok=True)
NAVY, GREY, GOLD = "#1F3864", "#7F8C8D", "#D4A017"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

P = HestonParams(kappa=1.5768, theta=0.0398, sigma=0.5751, rho=-0.5711, v0=0.0175)
S0, K0, R, T, N_PATHS, STEPS = 100.0, 100.0, 0.025, 1.0, 2000, 252


def main():
    rng = np.random.default_rng(2027)
    premium = call_fourier(S0, K0, T, R, 0, P)
    atm_iv = bs.implied_vol(premium, S0, K0, T, R, 0)
    paths, var = simulate_qe(S0, T, R, 0, P, STEPS, N_PATHS, rng, antithetic=False, return_var=True)
    dt = T / STEPS
    labels = ["Black-Scholes delta (1y ATM implied vol)", "Heston model delta", "Heston minimum-variance delta"]
    cash = {k: np.full(N_PATHS, premium) for k in labels}
    pos = {k: np.zeros(N_PATHS) for k in labels}
    for k in range(STEPS):                                  # rebalance daily, self-financing
        tau = T - k * dt
        S, v = paths[:, k], var[:, k]
        d_h, _, d_mv = greeks_vectorised(S, v, K0, tau, R, 0, P)
        deltas = {labels[0]: bs.delta(S, K0, tau, R, 0, atm_iv), labels[1]: d_h, labels[2]: d_mv}
        for lab in labels:
            cash[lab] -= (deltas[lab] - pos[lab]) * S
            pos[lab] = deltas[lab]
            cash[lab] *= np.exp(R * dt)
    pnl = {lab: cash[lab] + pos[lab] * paths[:, -1] - np.maximum(paths[:, -1] - K0, 0) for lab in labels}
    hedge = pd.DataFrame([{"hedge": k, "mean_pnl": v.mean(), "std_pnl": v.std(ddof=1),
                           "std_as_pct_of_premium": v.std(ddof=1) / premium,
                           "p5": np.quantile(v, 0.05), "p95": np.quantile(v, 0.95)} for k, v in pnl.items()])
    # Paired comparison on the same paths: variance reduction of each hedge vs the Black-Scholes hedge
    base = pnl[labels[0]]
    hedge["variance_vs_black_scholes"] = [np.var(pnl[k]) / np.var(base) for k in labels]
    # Paired bootstrap (same resampled paths for both hedges): 95% interval of each variance ratio
    boot_rng = np.random.default_rng(11)
    idx = boot_rng.integers(0, N_PATHS, (5000, N_PATHS))
    lo, hi = [], []
    for k in labels:
        ratios = np.var(pnl[k][idx], axis=1) / np.var(base[idx], axis=1)
        lo.append(np.quantile(ratios, 0.025)); hi.append(np.quantile(ratios, 0.975))
    hedge["ratio_ci_low"], hedge["ratio_ci_high"] = lo, hi
    hedge.to_csv(TABLES / "delta_hedging.csv", index=False)
    pd.DataFrame(pnl).to_csv(TABLES / "delta_hedging_pnl_by_path.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    bins = np.linspace(-8, 6, 80)
    for (label, v), c in zip(pnl.items(), [GREY, GOLD, NAVY]):
        ax.hist(v, bins=bins, alpha=0.55, color=c, label=f"{label}: SD {v.std():.2f}")
    ax.set_xlabel(f"Hedging P&L at maturity, short 1 ATM call (premium {premium:.2f})"); ax.set_yticks([])
    ax.set_title("Daily delta hedging in a Heston world (2,000 paths)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "delta_hedging.png", dpi=150); plt.close(fig)

    pd.set_option("display.width", 200)
    print(f"Premium {premium:.4f}, ATM implied vol {atm_iv:.4f}")
    print(hedge.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
