"""Compare all strategy presets with full quant metrics."""
import numpy as np
from src.backtest.backtester import Backtester, BacktestConfig
from src.backtest.quant_metrics import compute_quant_metrics
from src.strategies.presets import ALL_PRESETS, make_strategy_from_preset
from src.strategies.unified_strategy import unified_strategy

def run_preset(name, strategy_fn, n_sims=500):
    config = BacktestConfig(
        initial_capital=1000.0, kelly_fraction=0.25, min_ev=0.015,
        fee_rate=0.002, slippage_bps=10, n_simulations=n_sims, prob_sigma=0.05,
    )
    bt = Backtester(config)
    result = bt.run_monte_carlo_backtest(strategy_fn, initial_prob=0.5, n_steps=200)
    qm = compute_quant_metrics(result.equity_curve, result.trades)
    return result, qm

print("=" * 90)
print("STRATEGY PRESET COMPARISON — Full Quant Metrics")
print("=" * 90)
print()

# Run all presets
results = {}
for preset_name, preset in ALL_PRESETS.items():
    strategy_fn = make_strategy_from_preset(preset)
    result, qm = run_preset(preset_name, strategy_fn)
    results[preset_name] = (result, qm, preset)

# Also run the base unified strategy
result, qm = run_preset("unified (base)", unified_strategy)
results["unified"] = (result, qm, None)

# Print comparison table
header = f"{'Preset':<14} {'Return':>8} {'Sharpe':>8} {'Sortino':>8} {'Omega':>8} {'MaxDD':>8} {'CVaR99':>8} {'WR':>7} {'Trades':>7} {'PF':>7} {'Ulcer':>8} {'Grade':>6}"
print(header)
print("-" * len(header))

for name, (r, q, p) in results.items():
    desc = p.description[:30] if p else "Base strategy"
    pf_str = f"{q.profit_factor:.1f}" if q.profit_factor < 999 else "∞"
    print(
        f"{name:<14} {r.total_return:>7.1%} {r.sharpe:>8.1f} {q.sortino_ratio:>8.1f} "
        f"{q.omega_ratio:>8.2f} {r.max_dd:>7.2%} {q.cvar_99:>8.4f} "
        f"{r.win_rate:>6.1%} {r.n_trades:>7d} {pf_str:>7} {q.ulcer_index:>8.4f} {q.risk_grade:>6}"
    )

print()
print("=" * 90)
print("METRIC DEFINITIONS:")
print("  Sharpe:  (return - rf) / volatility         — standard risk-adjusted return")
print("  Sortino: (return - rf) / downside vol        — only penalizes losses (better than Sharpe)")
print("  Omega:   gains / losses (probability-weighted) — no distribution assumptions")
print("  MaxDD:   worst peak-to-trough decline")
print("  CVaR99:  expected loss in worst 1% of scenarios — Basel III standard")
print("  WR:      win rate                             — % profitable trades")
print("  PF:      profit factor (gross profit / loss)  — ∞ = no losing periods")
print("  Ulcer:   RMS of drawdowns                     — measures ongoing pain")
print("  Grade:   composite risk grade A+ to F")
print()

# Detailed report for each
for name, (r, q, p) in results.items():
    print(f"\n{'='*50}")
    print(f"  {name.upper()}" + (f" — {p.description}" if p else ""))
    print(f"{'='*50}")
    print(q.summary())
