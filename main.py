"""Prediction Market Bot — Main Entry Point.

Usage:
    python main.py scan                    # Scan for arb opportunities (dry run)
    python main.py backtest                # Run Monte Carlo backtest  
    python main.py backtest conservative   # Backtest with Conservative preset
    python main.py backtest aggressive     # Backtest with Aggressive preset
    python main.py backtest yolo           # Backtest with YOLO preset
    python main.py compare                 # Compare all presets side-by-side
    python main.py robustness              # Full anti-overfit robustness report
    python main.py sensitivity             # Parameter sensitivity analysis
    python main.py demo                    # Full demo of all components
"""
import asyncio
import logging
import sys

import numpy as np

from config.settings import *
from src.data.polymarket_client import PolymarketClient
from src.strategies.arb_scanner import ArbScanner
from src.strategies.bayesian_ev import BayesianEVScanner
from src.strategies.math_core import (
    fractional_kelly,
    ev_full,
    avellaneda_stoikov,
    monte_carlo_probability_path,
    estimate_edge_uncertainty,
    max_drawdown,
    sharpe_ratio,
)
from src.backtest.backtester import Backtester, BacktestConfig
from src.risk.risk_manager import RiskManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bot")


async def cmd_scan():
    """Scan Polymarket for arbitrage opportunities."""
    logger.info("=" * 60)
    logger.info("PREDICTION MARKET BOT — ARB SCANNER")
    logger.info("=" * 60)
    
    client = PolymarketClient()
    scanner = ArbScanner(
        poly_client=client,
        min_edge=ARB_MIN_EDGE,
        max_position=ARB_MAX_POSITION,
        dry_run=DRY_RUN,
    )
    
    try:
        logger.info("Scanning for YES/NO arbitrage opportunities...")
        opps = await scanner.scan_intra_arb()
        
        if not opps:
            logger.info("No arbitrage opportunities found at this time.")
            logger.info("Markets are efficient right now. Edges appear during high-volatility events.")
        else:
            logger.info(f"\nFound {len(opps)} opportunities:")
            for i, opp in enumerate(opps[:10]):
                logger.info(
                    f"  {i+1}. {opp.market_name[:50]}\n"
                    f"     YES={opp.yes_price:.4f} NO={opp.no_price:.4f} "
                    f"Total={opp.total_cost:.4f} Edge={opp.edge:.4f}\n"
                    f"     Expected P&L: ${opp.expected_profit:.2f} "
                    f"(Size: {opp.recommended_size:.0f} shares)"
                )
        
        stats = scanner.get_stats()
        logger.info(f"\nScanner Stats: {stats}")
        
    finally:
        await client.close()


def cmd_backtest():
    """Run Monte Carlo backtest of Bayesian EV strategy."""
    logger.info("=" * 60)
    logger.info("PREDICTION MARKET BOT — BACKTEST ENGINE")
    logger.info("=" * 60)
    
    config = BacktestConfig(
        initial_capital=INITIAL_CAPITAL,
        kelly_fraction=KELLY_FRACTION,
        min_ev=EV_MIN_EDGE,
        fee_rate=0.002,
        slippage_bps=10,
        n_simulations=500,
        prob_sigma=0.05,
    )
    
    backtester = Backtester(config)
    
    # Define strategy: Bayesian EV with quarter-Kelly
    def bayesian_ev_strategy(model_prob: float, market_price: float, capital: float):
        ev = model_prob - market_price
        if abs(ev) < config.min_ev:
            return None
        
        kelly = fractional_kelly(model_prob, market_price, config.kelly_fraction)
        size = kelly * capital
        
        if size < 1.0:
            return None
        
        return {
            "side": "buy" if ev > 0 else "sell",
            "size_usd": size,
        }
    
    logger.info("Running Monte Carlo backtest (500 simulations, 200 steps each)...")
    result = backtester.run_monte_carlo_backtest(
        strategy_fn=bayesian_ev_strategy,
        initial_prob=0.5,
        n_steps=200,
    )
    
    logger.info("\n" + "=" * 50)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 50)
    logger.info(f"  Total Return:   {result.total_return:.2%}")
    logger.info(f"  Total P&L:      ${result.total_pnl:.2f}")
    logger.info(f"  Trades:         {result.n_trades}")
    logger.info(f"  Win Rate:       {result.win_rate:.1%}")
    logger.info(f"  Sharpe Ratio:   {result.sharpe:.2f}")
    logger.info(f"  Max Drawdown:   {result.max_dd:.2%}")
    logger.info(f"  Calmar Ratio:   {result.calmar_ratio:.2f}")
    logger.info(f"  Profit Factor:  {result.profit_factor_val:.2f}")
    logger.info(f"  Robust:         {'✅ YES' if result.is_robust else '❌ NO'}")
    
    return result


def cmd_sensitivity():
    """Run parameter sensitivity analysis."""
    logger.info("=" * 60)
    logger.info("PREDICTION MARKET BOT — SENSITIVITY ANALYSIS")
    logger.info("=" * 60)
    
    config = BacktestConfig(n_simulations=200, prob_sigma=0.05)
    backtester = Backtester(config)
    
    def strategy_factory(params):
        def strategy(model_prob, market_price, capital):
            ev = model_prob - market_price
            if abs(ev) < params["min_ev"]:
                return None
            kelly = fractional_kelly(model_prob, market_price, params["kelly_frac"])
            size = kelly * capital
            if size < 1.0:
                return None
            return {"side": "buy" if ev > 0 else "sell", "size_usd": size}
        return strategy
    
    base_params = {"min_ev": 0.03, "kelly_frac": 0.25}
    param_ranges = {
        "min_ev": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10],
        "kelly_frac": [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0],
    }
    
    logger.info("Testing parameter sensitivity...")
    df = backtester.parameter_sensitivity(
        strategy_factory=strategy_factory,
        base_params=base_params,
        param_ranges=param_ranges,
        n_simulations=200,
    )
    
    logger.info("\nResults:")
    for _, row in df.iterrows():
        status = "✅" if row["is_robust"] else "❌"
        logger.info(
            f"  {status} {row['param']}={row['value']:.3f} → "
            f"Return={row['return']:.2%} Sharpe={row['sharpe']:.2f} "
            f"MaxDD={row['max_dd']:.2%} WR={row['win_rate']:.1%} "
            f"Trades={row['n_trades']}"
        )
    
    robust_pct = df["is_robust"].mean()
    logger.info(f"\nRobustness: {robust_pct:.0%} of configs profitable")
    if robust_pct < 0.3:
        logger.warning("⚠️  LIKELY CURVE-FIT — strategy may not be robust")
    
    return df


def cmd_demo():
    """Full demo of all bot components."""
    logger.info("=" * 60)
    logger.info("PREDICTION MARKET BOT — FULL DEMO")
    logger.info("=" * 60)
    
    # 1. Math Core Demo
    logger.info("\n📐 MATH CORE DEMO")
    logger.info("-" * 40)
    
    # Bayesian update example
    from src.strategies.math_core import bayesian_update_binary
    prior = 0.5
    posterior = bayesian_update_binary(prior, sensitivity=0.9, specificity=0.8)
    logger.info(f"Bayesian Update: Prior={prior} → Posterior={posterior:.4f}")
    
    # Kelly sizing
    model_prob = 0.65
    market_price = 0.55
    kelly = fractional_kelly(model_prob, market_price, 0.25)
    logger.info(f"¼ Kelly: p̂={model_prob}, P={market_price} → f={kelly:.4f} ({kelly*100:.1f}% of capital)")
    
    # EV calculation
    ev = ev_full(model_prob, market_price)
    logger.info(f"EV: p̂={model_prob}, buy@{market_price} → EV={ev:.4f}")
    
    # Avellaneda-Stoikov
    quotes = avellaneda_stoikov(
        mid_price=0.50, inventory=2.0, gamma=0.1,
        sigma=0.05, time_remaining=0.5
    )
    logger.info(f"A-S Quotes: Bid={quotes.bid:.4f} Ask={quotes.ask:.4f} Spread={quotes.spread:.4f}")
    
    # Edge uncertainty via Monte Carlo
    uncertainty = estimate_edge_uncertainty(model_prob, market_price, sigma=0.1)
    logger.info(f"Edge Uncertainty: Mean={uncertainty['mean_edge']:.4f} CV={uncertainty['cv_edge']:.4f} P(profit)={uncertainty['prob_profitable']:.1%}")
    
    # 2. Monte Carlo Simulation
    logger.info("\n🎲 MONTE CARLO SIMULATION")
    logger.info("-" * 40)
    
    paths = monte_carlo_probability_path(
        initial_prob=0.5, sigma=0.03, n_steps=50, n_simulations=10000
    )
    final_probs = paths[:, -1]
    logger.info(f"10K simulations, 50 steps from p=0.5:")
    logger.info(f"  Mean final prob: {np.mean(final_probs):.4f}")
    logger.info(f"  Std: {np.std(final_probs):.4f}")
    logger.info(f"  P(>0.7): {np.mean(final_probs > 0.7):.2%}")
    logger.info(f"  P(<0.3): {np.mean(final_probs < 0.3):.2%}")
    
    # 3. Backtest
    logger.info("\n📊 BACKTEST")
    logger.info("-" * 40)
    result = cmd_backtest()
    
    # 4. Risk Management
    logger.info("\n🛡️ RISK MANAGEMENT DEMO")
    logger.info("-" * 40)
    
    rm = RiskManager(initial_capital=1000.0, max_drawdown_pct=0.15)
    
    # Simulate some trades
    for i in range(20):
        pnl = np.random.normal(5, 20)  # Mean $5, std $20
        rm.capital += pnl
        rm.trade_returns.append(pnl / 100)
        rm.equity_curve.append(rm.capital)
        rm.realized_pnl += pnl
        if rm.capital > rm.peak_equity:
            rm.peak_equity = rm.capital
    
    state = rm.get_state()
    perf = rm.get_performance()
    
    logger.info(f"  Capital: ${state.total_capital:.2f}")
    logger.info(f"  Drawdown: {state.current_drawdown:.2%}")
    logger.info(f"  Max DD: {perf['max_drawdown']:.2%}")
    logger.info(f"  Win Rate: {perf['win_rate']:.1%}")
    logger.info(f"  Sharpe: {perf['sharpe_ratio']:.2f}")
    logger.info(f"  Kill Switch: {'🔴 ACTIVE' if state.kill_switch_active else '🟢 OFF'}")
    
    logger.info("\n" + "=" * 60)
    logger.info("DEMO COMPLETE — Ready to trade!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("  1. Set POLYMARKET_PRIVATE_KEY in .env")
    logger.info("  2. Run: python main.py scan")
    logger.info("  3. Review opportunities, then disable DRY_RUN")
    logger.info("  4. Start with $100-500, scale up after proving edge")


def cmd_compare():
    """Compare all strategy presets."""
    logger.info("Running preset comparison...")
    import subprocess
    subprocess.run([sys.executable, "compare_presets.py"], cwd="/Users/vincent/Desktop/geospatios/prediction-market-bot")


def cmd_robustness():
    """Run full anti-overfit robustness report."""
    from src.backtest.robustness import full_robustness_report
    from src.strategies.unified_strategy import unified_strategy
    print(full_robustness_report(unified_strategy))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    cmd = sys.argv[1].lower()
    
    if cmd == "scan":
        asyncio.run(cmd_scan())
    elif cmd == "backtest":
        preset_name = sys.argv[2] if len(sys.argv) > 2 else None
        if preset_name:
            from src.strategies.presets import ALL_PRESETS, make_strategy_from_preset
            if preset_name in ALL_PRESETS:
                logger.info(f"Using {preset_name.upper()} preset")
                strategy_fn = make_strategy_from_preset(ALL_PRESETS[preset_name])
                # TODO: pass strategy_fn to backtest
            else:
                print(f"Unknown preset: {preset_name}. Available: {list(ALL_PRESETS.keys())}")
                sys.exit(1)
        cmd_backtest()
    elif cmd == "compare":
        cmd_compare()
    elif cmd == "robustness":
        cmd_robustness()
    elif cmd == "sensitivity":
        cmd_sensitivity()
    elif cmd == "dashboard":
        print("Dashboard coming soon — use 'demo' for now")
    elif cmd == "demo":
        cmd_demo()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
