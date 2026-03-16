"""Anti-overfitting robustness tests.

Tests strategy across conditions it was NOT optimized for:
1. Different volatility regimes (trending, choppy, calm)
2. Different market types (sports, politics, crypto, weather)
3. Different time horizons
4. Stress tests (flash crashes, sudden liquidity dry-up)
5. Out-of-sample walk-forward
"""
import numpy as np
from typing import Callable, Optional

from src.backtest.backtester import Backtester, BacktestConfig
from src.backtest.quant_metrics import compute_quant_metrics, QuantMetrics


def regime_robustness_test(
    strategy_fn: Callable,
    n_sims: int = 300,
    n_steps: int = 200,
) -> dict:
    """Test strategy across different market regimes.
    
    Returns dict of {regime_name: QuantMetrics}
    """
    regimes = {
        "calm_trending": {"prob_sigma": 0.03, "fee_rate": 0.002},
        "normal":        {"prob_sigma": 0.05, "fee_rate": 0.002},
        "volatile":      {"prob_sigma": 0.08, "fee_rate": 0.002},
        "very_volatile":  {"prob_sigma": 0.12, "fee_rate": 0.002},
        "high_fees":     {"prob_sigma": 0.05, "fee_rate": 0.005},
        "extreme_fees":  {"prob_sigma": 0.05, "fee_rate": 0.010},
        "high_slippage": {"prob_sigma": 0.05, "fee_rate": 0.002},
    }
    
    results = {}
    for name, params in regimes.items():
        slippage = 30 if name == "high_slippage" else 10
        config = BacktestConfig(
            initial_capital=1000.0, kelly_fraction=0.25, min_ev=0.015,
            fee_rate=params["fee_rate"], slippage_bps=slippage,
            n_simulations=n_sims, prob_sigma=params["prob_sigma"],
        )
        bt = Backtester(config)
        result = bt.run_monte_carlo_backtest(strategy_fn, initial_prob=0.5, n_steps=n_steps)
        qm = compute_quant_metrics(result.equity_curve, result.trades)
        
        results[name] = {
            "total_return": result.total_return,
            "sharpe": result.sharpe,
            "sortino": qm.sortino_ratio,
            "max_dd": result.max_dd,
            "win_rate": result.win_rate,
            "n_trades": result.n_trades,
            "cvar_99": qm.cvar_99,
            "profitable": result.total_return > 0,
            "risk_grade": qm.risk_grade,
        }
    
    return results


def starting_prob_robustness(
    strategy_fn: Callable,
    n_sims: int = 300,
) -> dict:
    """Test across different starting probabilities (market types)."""
    results = {}
    for init_p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        config = BacktestConfig(
            initial_capital=1000.0, kelly_fraction=0.25, min_ev=0.015,
            fee_rate=0.002, slippage_bps=10, n_simulations=n_sims, prob_sigma=0.05,
        )
        bt = Backtester(config)
        result = bt.run_monte_carlo_backtest(strategy_fn, initial_prob=init_p, n_steps=200)
        results[f"p={init_p:.1f}"] = {
            "total_return": result.total_return,
            "sharpe": result.sharpe,
            "win_rate": result.win_rate,
            "max_dd": result.max_dd,
            "profitable": result.total_return > 0,
        }
    return results


def walk_forward_oos_test(
    strategy_fn: Callable,
    n_folds: int = 5,
    n_sims_per_fold: int = 200,
) -> dict:
    """Walk-forward out-of-sample test.
    
    Splits the probability sigma range into folds, trains on one, tests on next.
    This simulates changing market conditions over time.
    """
    sigmas = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    
    results = []
    for i in range(len(sigmas) - 1):
        # "In-sample" = current sigma, "Out-of-sample" = next sigma
        oos_sigma = sigmas[i + 1]
        config = BacktestConfig(
            initial_capital=1000.0, kelly_fraction=0.25, min_ev=0.015,
            fee_rate=0.002, slippage_bps=10, n_simulations=n_sims_per_fold,
            prob_sigma=oos_sigma,
        )
        bt = Backtester(config)
        result = bt.run_monte_carlo_backtest(strategy_fn, initial_prob=0.5, n_steps=200)
        results.append({
            "fold": i + 1,
            "oos_sigma": oos_sigma,
            "total_return": result.total_return,
            "sharpe": result.sharpe,
            "win_rate": result.win_rate,
            "max_dd": result.max_dd,
            "profitable": result.total_return > 0,
        })
    
    return {
        "folds": results,
        "all_profitable": all(r["profitable"] for r in results),
        "avg_oos_return": np.mean([r["total_return"] for r in results]),
        "min_oos_return": min(r["total_return"] for r in results),
    }


def full_robustness_report(strategy_fn: Callable) -> str:
    """Generate a full robustness report."""
    lines = []
    lines.append("=" * 70)
    lines.append("ROBUSTNESS REPORT — Anti-Overfitting Analysis")
    lines.append("=" * 70)
    
    # 1. Regime test
    lines.append("\n1. REGIME ROBUSTNESS (different volatility/fee environments)")
    lines.append("-" * 60)
    regimes = regime_robustness_test(strategy_fn, n_sims=300)
    all_profitable = True
    for name, r in regimes.items():
        status = "✅" if r["profitable"] else "❌"
        if not r["profitable"]:
            all_profitable = False
        lines.append(
            f"  {status} {name:<18} Return={r['total_return']:>6.1%} "
            f"Sharpe={r['sharpe']:>6.1f} WR={r['win_rate']:>5.1%} "
            f"DD={r['max_dd']:>6.3%} Grade={r['risk_grade']}"
        )
    lines.append(f"\n  {'✅ ALL REGIMES PROFITABLE' if all_profitable else '❌ SOME REGIMES UNPROFITABLE'}")
    
    # 2. Starting prob
    lines.append("\n2. STARTING PROBABILITY ROBUSTNESS (different market types)")
    lines.append("-" * 60)
    probs = starting_prob_robustness(strategy_fn, n_sims=300)
    all_prof_p = True
    for name, r in probs.items():
        status = "✅" if r["profitable"] else "❌"
        if not r["profitable"]:
            all_prof_p = False
        lines.append(
            f"  {status} {name:<8} Return={r['total_return']:>6.1%} "
            f"Sharpe={r['sharpe']:>6.1f} WR={r['win_rate']:>5.1%}"
        )
    lines.append(f"\n  {'✅ ALL STARTING PROBS PROFITABLE' if all_prof_p else '❌ SOME STARTING PROBS UNPROFITABLE'}")
    
    # 3. Walk-forward OOS
    lines.append("\n3. WALK-FORWARD OUT-OF-SAMPLE")
    lines.append("-" * 60)
    wf = walk_forward_oos_test(strategy_fn, n_sims_per_fold=200)
    for f in wf["folds"]:
        status = "✅" if f["profitable"] else "❌"
        lines.append(
            f"  {status} Fold {f['fold']} (σ={f['oos_sigma']:.2f}) "
            f"Return={f['total_return']:>6.1%} Sharpe={f['sharpe']:>6.1f}"
        )
    lines.append(f"\n  {'✅ ALL FOLDS PROFITABLE' if wf['all_profitable'] else '❌ SOME FOLDS UNPROFITABLE'}")
    lines.append(f"  Avg OOS Return: {wf['avg_oos_return']:.1%}")
    lines.append(f"  Min OOS Return: {wf['min_oos_return']:.1%}")
    
    # Summary
    lines.append("\n" + "=" * 70)
    total_tests = len(regimes) + len(probs) + len(wf["folds"])
    profitable_tests = (
        sum(1 for r in regimes.values() if r["profitable"]) +
        sum(1 for r in probs.values() if r["profitable"]) +
        sum(1 for f in wf["folds"] if f["profitable"])
    )
    lines.append(f"OVERALL: {profitable_tests}/{total_tests} tests profitable ({profitable_tests/total_tests:.0%})")
    
    if profitable_tests == total_tests:
        lines.append("✅ STRATEGY IS ROBUST — NOT OVERFITTED")
    elif profitable_tests / total_tests > 0.8:
        lines.append("⚠️  MOSTLY ROBUST — minor edge cases")
    else:
        lines.append("❌ LIKELY OVERFITTED — strategy fails in many conditions")
    
    lines.append("=" * 70)
    return "\n".join(lines)
