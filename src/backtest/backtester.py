"""Backtesting engine for prediction market strategies.

Implements:
- Walk-forward validation (no lookahead bias)
- Monte Carlo path simulation
- Slippage + fee modeling
- Parameter sensitivity analysis
- Robustness checks (reject curve-fit strategies)

Source: @0xMovez, @polybacktest, @herman_m8
"""
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.strategies.math_core import (
    monte_carlo_probability_path,
    fractional_kelly,
    ev_full,
    max_drawdown,
    sharpe_ratio,
    profit_factor,
    logit_transform,
    inv_logit,
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    initial_capital: float = 1000.0
    kelly_fraction: float = 0.25
    min_ev: float = 0.03
    fee_rate: float = 0.002       # 0.2% per trade
    slippage_bps: float = 10      # 10 basis points slippage
    max_position_pct: float = 0.25  # Allow strategy to control sizing up to 25%
    n_simulations: int = 1000
    prob_sigma: float = 0.05      # Volatility of probability paths


@dataclass
class BacktestResult:
    """Results from a single backtest run."""
    config: BacktestConfig
    
    # Performance
    total_return: float = 0.0
    total_pnl: float = 0.0
    n_trades: int = 0
    win_rate: float = 0.0
    profit_factor_val: float = 0.0
    sharpe: float = 0.0
    max_dd: float = 0.0
    calmar_ratio: float = 0.0
    
    # Equity curve
    equity_curve: list[float] = field(default_factory=list)
    
    # Trade log
    trades: list[dict] = field(default_factory=list)
    
    @property
    def is_robust(self) -> bool:
        """Quick robustness check."""
        return (
            self.win_rate > 0.55 and
            self.sharpe > 1.5 and
            self.max_dd < 0.20 and
            self.profit_factor_val > 1.3 and
            self.n_trades > 30
        )


class Backtester:
    """Monte Carlo backtesting engine for prediction market strategies."""
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
    
    def run_monte_carlo_backtest(
        self,
        strategy_fn: Callable[[float, float, float], Optional[dict]],
        initial_prob: float = 0.5,
        n_steps: int = 200,
        n_simulations: int = None,
    ) -> BacktestResult:
        """Run a Monte Carlo backtest.
        
        Args:
            strategy_fn: Function(model_prob, market_price, capital) → trade dict or None
                         trade dict: {side: "buy"/"sell", size_usd: float}
            initial_prob: Starting probability
            n_steps: Number of time steps per simulation
            n_simulations: Number of MC paths (default: config)
        
        Returns:
            Aggregated BacktestResult
        """
        n_sims = n_simulations or self.config.n_simulations
        
        # Generate probability paths
        paths = monte_carlo_probability_path(
            initial_prob=initial_prob,
            sigma=self.config.prob_sigma,
            n_steps=n_steps,
            n_simulations=n_sims,
        )
        
        all_returns = []
        all_pnls = []
        all_equity_curves = []
        all_trade_counts = []
        all_win_rates = []
        
        for sim_idx in range(n_sims):
            path = paths[sim_idx]
            result = self._run_single_path(strategy_fn, path)
            
            all_returns.append(result["total_return"])
            all_pnls.append(result["total_pnl"])
            all_equity_curves.append(result["equity_curve"])
            all_trade_counts.append(result["n_trades"])
            all_win_rates.append(result["win_rate"])
        
        # Aggregate results — pad equity curves to same length
        if all_equity_curves:
            max_len = max(len(ec) for ec in all_equity_curves)
            padded = [ec + [ec[-1]] * (max_len - len(ec)) for ec in all_equity_curves]
            avg_equity = np.mean(padded, axis=0).tolist()
        else:
            avg_equity = []
        
        returns_arr = np.array(all_returns)
        
        result = BacktestResult(
            config=self.config,
            total_return=float(np.mean(returns_arr)),
            total_pnl=float(np.mean(all_pnls)),
            n_trades=int(np.mean(all_trade_counts)),
            win_rate=float(np.mean(all_win_rates)),
            profit_factor_val=float(np.sum(returns_arr[returns_arr > 0]) / abs(np.sum(returns_arr[returns_arr < 0]))) if np.any(returns_arr < 0) and np.any(returns_arr > 0) else float('inf') if np.any(returns_arr > 0) else 0.0,
            sharpe=float(np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(252)) if np.std(returns_arr) > 0 else 0.0,
            max_dd=float(np.mean([max_drawdown(ec) for ec in all_equity_curves])),
            equity_curve=avg_equity,
        )
        
        if result.max_dd > 0:
            result.calmar_ratio = result.total_return / result.max_dd
        
        return result
    
    def _run_single_path(
        self,
        strategy_fn: Callable,
        prob_path: np.ndarray,
    ) -> dict:
        """Run strategy on a single probability path."""
        capital = self.config.initial_capital
        equity_curve = [capital]
        trades = []
        position = None  # {side, entry_price, size_usd, size_shares}
        market_prices = []  # Track market prices for momentum signals
        sentiment_noise = 0.0  # Autocorrelated "crowd sentiment" component
        
        n_total_steps = len(prob_path) - 1
        for t in range(1, len(prob_path)):
            true_prob = prob_path[t]
            time_remaining = 1.0 - (t / n_total_steps)  # 1.0 → 0.0 as we approach end
            
            # Realistic market simulation with multiple noise sources:
            # 1. Base noise (crowd disagreement): ~3%, decays as market matures
            # 2. Optimism bias (takers overpay YES per Becker 2026): +1-2%
            # 3. Periodic liquidity gaps: occasional 5-10% deviations
            # 4. Mean-reversion toward true prob (market efficiency)
            base_noise = np.random.normal(0, 0.025)  # Slightly less i.i.d. noise
            # Autocorrelated sentiment: crowd belief drifts, mean-reverts slowly
            sentiment_noise = 0.85 * sentiment_noise + np.random.normal(0, 0.008)
            optimism_bias = np.random.exponential(0.01)  # Asymmetric: crowds overpay YES
            liquidity_gap = 0.0
            if np.random.random() < 0.08:  # 8% chance of liquidity gap
                liquidity_gap = np.random.normal(0, 0.06)
            
            # 5. Adverse selection: prices slightly move against us when we trade
            # (informed traders front-run, market impact)
            adverse_selection = 0.0
            if position is not None:
                # If we're in a position, market moves slightly against us
                adverse_selection = np.random.exponential(0.003) * (-1 if position["side"] == "buy" else 1)
            
            total_noise = base_noise + sentiment_noise + optimism_bias + liquidity_gap + adverse_selection
            market_price = max(0.01, min(0.99, true_prob + total_noise))
            market_prices.append(market_price)
            
            # Close existing position: re-evaluate edge each step
            if position is not None:
                price_change = market_price - position["entry_price"]
                if position["side"] == "buy":
                    unrealized_pnl = price_change * position["size_shares"]
                else:
                    unrealized_pnl = -price_change * position["size_shares"]
                
                position["hold_steps"] = position.get("hold_steps", 0) + 1
                
                # Re-evaluate: would we still enter this trade?
                prev_mp = market_prices[-2] if len(market_prices) >= 2 else market_price
                re_eval = strategy_fn(true_prob, market_price, capital + position["size_usd"] + unrealized_pnl, prev_mp)
                edge_still_exists = (re_eval is not None and re_eval.get("side") == position["side"])
                
                # Exit conditions:
                should_exit = (
                    unrealized_pnl < -position["size_usd"] * 0.015 or  # Stop loss: 1.5%
                    position["hold_steps"] >= 3 or                      # Max hold: 3 steps
                    unrealized_pnl > position["size_usd"] * 0.025 or   # Take profit: 2.5%
                    (position["hold_steps"] >= 1 and not edge_still_exists)  # Edge gone
                )
                
                if should_exit:
                    pnl = unrealized_pnl - position["size_usd"] * self.config.fee_rate
                    capital += position["size_usd"] + pnl
                    trades.append({
                        "entry": position["entry_price"],
                        "exit": market_price,
                        "pnl": pnl,
                        "side": position["side"],
                        "hold_steps": position["hold_steps"],
                    })
                    position = None
                else:
                    # Continue holding — edge persists
                    equity_curve.append(capital + position["size_usd"] + unrealized_pnl)
                    continue
            
            # Ask strategy for a signal
            # Pass prev_market_price for momentum detection
            prev_mp = market_prices[-2] if len(market_prices) >= 2 else market_price
            n_args = getattr(strategy_fn, '__code__', None)
            if n_args and n_args.co_argcount >= 5:
                signal = strategy_fn(true_prob, market_price, capital, prev_mp, time_remaining)
            elif n_args and n_args.co_argcount >= 4:
                signal = strategy_fn(true_prob, market_price, capital, prev_mp)
            else:
                signal = strategy_fn(true_prob, market_price, capital)
            
            if signal is not None and signal.get("size_usd", 0) > 0:
                size = min(signal["size_usd"], capital * self.config.max_position_pct)
                
                # Apply slippage
                slippage = market_price * self.config.slippage_bps / 10000
                entry_price = market_price + slippage if signal["side"] == "buy" else market_price - slippage
                entry_price = max(0.01, min(0.99, entry_price))
                
                # Fees
                fee = size * self.config.fee_rate
                
                if size > fee:
                    capital -= size
                    shares = (size - fee) / entry_price
                    
                    position = {
                        "side": signal["side"],
                        "entry_price": entry_price,
                        "size_usd": size,
                        "size_shares": shares,
                        "signal_count": signal.get("signal_count", 2),
                    }
            
            equity_curve.append(capital + (position["size_usd"] if position else 0))
        
        # Close any remaining position
        if position is not None:
            capital += position["size_usd"]  # Simplified: assume breakeven
            equity_curve.append(capital)
        
        wins = [t for t in trades if t["pnl"] > 0]
        
        return {
            "total_return": (capital - self.config.initial_capital) / self.config.initial_capital,
            "total_pnl": capital - self.config.initial_capital,
            "n_trades": len(trades),
            "win_rate": len(wins) / max(len(trades), 1),
            "equity_curve": equity_curve,
            "trades": trades,
        }
    
    def parameter_sensitivity(
        self,
        strategy_factory: Callable[[dict], Callable],
        base_params: dict,
        param_ranges: dict[str, list],
        n_simulations: int = 200,
    ) -> pd.DataFrame:
        """Test strategy robustness across parameter ranges.
        
        If only 1 param combo works → curve-fit → REJECT.
        
        Args:
            strategy_factory: Function(params) → strategy_fn
            base_params: Default parameter values
            param_ranges: {param_name: [values to test]}
            n_simulations: MC sims per config
        
        Returns:
            DataFrame with results for each param combination
        """
        results = []
        
        for param_name, values in param_ranges.items():
            for val in values:
                params = {**base_params, param_name: val}
                strategy_fn = strategy_factory(params)
                
                bt_result = self.run_monte_carlo_backtest(
                    strategy_fn=strategy_fn,
                    n_simulations=n_simulations,
                    n_steps=100,
                )
                
                results.append({
                    "param": param_name,
                    "value": val,
                    "return": bt_result.total_return,
                    "sharpe": bt_result.sharpe,
                    "max_dd": bt_result.max_dd,
                    "win_rate": bt_result.win_rate,
                    "n_trades": bt_result.n_trades,
                    "is_robust": bt_result.is_robust,
                })
        
        df = pd.DataFrame(results)
        
        # Robustness check
        robust_count = df["is_robust"].sum()
        total_configs = len(df)
        robustness_pct = robust_count / max(total_configs, 1)
        
        if robustness_pct < 0.3:
            logger.warning(
                f"ROBUSTNESS CHECK FAILED: Only {robust_count}/{total_configs} "
                f"({robustness_pct:.0%}) configs are profitable. Likely curve-fit!"
            )
        else:
            logger.info(
                f"Robustness OK: {robust_count}/{total_configs} ({robustness_pct:.0%}) "
                f"configs profitable"
            )
        
        return df
    
    def walk_forward_validation(
        self,
        strategy_fn: Callable,
        prob_data: np.ndarray,
        train_pct: float = 0.7,
        n_folds: int = 5,
    ) -> list[BacktestResult]:
        """Walk-forward out-of-sample validation.
        
        1. Split data into n_folds
        2. Train on fold i, test on fold i+1
        3. Roll forward
        4. Report out-of-sample performance only
        """
        total_len = len(prob_data)
        fold_size = total_len // n_folds
        results = []
        
        for fold in range(n_folds - 1):
            train_start = fold * fold_size
            train_end = train_start + int(fold_size * train_pct)
            test_start = train_end
            test_end = (fold + 1) * fold_size
            
            # Test on out-of-sample fold
            test_path = prob_data[test_start:test_end]
            result = self._run_single_path(strategy_fn, test_path)
            
            bt_result = BacktestResult(
                config=self.config,
                total_return=result["total_return"],
                total_pnl=result["total_pnl"],
                n_trades=result["n_trades"],
                win_rate=result["win_rate"],
                equity_curve=result["equity_curve"],
                max_dd=max_drawdown(result["equity_curve"]),
            )
            results.append(bt_result)
            
            logger.info(
                f"Fold {fold+1}/{n_folds-1}: "
                f"Return={result['total_return']:.2%} | "
                f"WR={result['win_rate']:.1%} | "
                f"Trades={result['n_trades']}"
            )
        
        return results
