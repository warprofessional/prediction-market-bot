"""Comprehensive quant metrics suite — what real trading desks measure.

Beyond Sharpe, hedge funds and prop shops track:

Risk-Adjusted Returns:
- Sharpe Ratio: excess return / volatility (assumes normal returns — FLAWED for PM)
- Sortino Ratio: excess return / downside deviation (only penalizes losses, not vol)
- Calmar Ratio: annualized return / max drawdown
- Omega Ratio: probability-weighted gains / probability-weighted losses (no distribution assumptions)
- Information Ratio: alpha / tracking error (vs benchmark)

Tail Risk (Basel-style):
- VaR (Value at Risk): max loss at confidence level (95%, 99%)
- CVaR / Expected Shortfall: expected loss BEYOND VaR (what Basel III requires)
- Maximum Drawdown: worst peak-to-trough
- Ulcer Index: RMS of drawdowns (measures pain, not just worst case)

Trade Quality:
- Win Rate, Profit Factor, Average Win/Loss Ratio
- Expectancy: average $ per trade
- Payoff Ratio: avg win / avg loss (should be > 1)
- Consecutive Losses (max streak)

Robustness:
- Walk-Forward Efficiency: OOS performance / IS performance
- Parameter Stability: % of nearby params that are also profitable
- Regime Sensitivity: performance across different vol environments
"""
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class QuantMetrics:
    """Full quant metrics report for a strategy backtest."""
    
    # === Risk-Adjusted Returns ===
    sharpe_ratio: float = 0.0         # (return - rf) / std(return)
    sortino_ratio: float = 0.0        # (return - rf) / downside_std
    calmar_ratio: float = 0.0         # annual_return / max_drawdown
    omega_ratio: float = 0.0          # sum(max(r-t,0)) / sum(max(t-r,0))
    
    # === Tail Risk (Basel-style) ===
    var_95: float = 0.0               # 95% Value at Risk
    var_99: float = 0.0               # 99% Value at Risk
    cvar_95: float = 0.0              # 95% Conditional VaR (Expected Shortfall)
    cvar_99: float = 0.0              # 99% CVaR — what Basel III uses
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0    # Steps in longest drawdown
    ulcer_index: float = 0.0          # RMS of drawdowns
    
    # === Return Profile ===
    total_return: float = 0.0
    annualized_return: float = 0.0    # Assumes 252 trading days
    volatility: float = 0.0           # Annualized std
    downside_volatility: float = 0.0  # Std of negative returns only
    skewness: float = 0.0             # Negative skew = fat left tail
    kurtosis: float = 0.0             # High = fat tails
    
    # === Trade Quality ===
    n_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0       # gross_profit / gross_loss
    expectancy: float = 0.0           # avg $ per trade
    payoff_ratio: float = 0.0         # avg_win / avg_loss
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    
    # === Risk Summary ===
    risk_grade: str = ""              # A-F grade
    
    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {k: v for k, v in self.__dict__.items()}
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "╔══════════════════════════════════════════════╗",
            "║         QUANT METRICS REPORT                 ║",
            "╠══════════════════════════════════════════════╣",
            f"║  Total Return:     {self.total_return:>8.2%}                 ║",
            f"║  Ann. Return:      {self.annualized_return:>8.2%}                 ║",
            f"║  Volatility:       {self.volatility:>8.2%}                 ║",
            "╠══════════════════════════════════════════════╣",
            "║  RISK-ADJUSTED                               ║",
            f"║  Sharpe:           {self.sharpe_ratio:>8.2f}                 ║",
            f"║  Sortino:          {self.sortino_ratio:>8.2f}                 ║",
            f"║  Calmar:           {self.calmar_ratio:>8.2f}                 ║",
            f"║  Omega:            {self.omega_ratio:>8.2f}                 ║",
            "╠══════════════════════════════════════════════╣",
            "║  TAIL RISK (Basel-style)                     ║",
            f"║  VaR 95%:          {self.var_95:>8.4f}                 ║",
            f"║  VaR 99%:          {self.var_99:>8.4f}                 ║",
            f"║  CVaR 95%:         {self.cvar_95:>8.4f}                 ║",
            f"║  CVaR 99%:         {self.cvar_99:>8.4f}                 ║",
            f"║  Max Drawdown:     {self.max_drawdown:>8.3%}                 ║",
            f"║  Ulcer Index:      {self.ulcer_index:>8.4f}                 ║",
            "╠══════════════════════════════════════════════╣",
            "║  TRADE QUALITY                               ║",
            f"║  Trades:           {self.n_trades:>8d}                 ║",
            f"║  Win Rate:         {self.win_rate:>8.1%}                 ║",
            f"║  Profit Factor:    {self.profit_factor:>8.2f}                 ║",
            f"║  Expectancy:       ${self.expectancy:>7.2f}                 ║",
            f"║  Payoff Ratio:     {self.payoff_ratio:>8.2f}                 ║",
            f"║  Max Consec Losses:{self.max_consecutive_losses:>8d}                 ║",
            "╠══════════════════════════════════════════════╣",
            f"║  RISK GRADE:       {self.risk_grade:>8s}                 ║",
            "╚══════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)


def compute_quant_metrics(
    equity_curve: list[float],
    trades: list[dict],
    risk_free_rate: float = 0.04,   # 4% annual risk-free
    periods_per_year: float = 252,
) -> QuantMetrics:
    """Compute comprehensive quant metrics from backtest results.
    
    Args:
        equity_curve: list of portfolio values over time
        trades: list of {pnl, entry, exit, side, ...} dicts
        risk_free_rate: annual risk-free rate for Sharpe/Sortino
        periods_per_year: for annualization (252 trading days)
    """
    m = QuantMetrics()
    
    if len(equity_curve) < 2:
        return m
    
    eq = np.array(equity_curve, dtype=float)
    returns = np.diff(eq) / eq[:-1]  # Period returns
    returns = returns[np.isfinite(returns)]
    
    if len(returns) < 2:
        return m
    
    # === Return Profile ===
    m.total_return = (eq[-1] - eq[0]) / eq[0]
    m.annualized_return = (1 + m.total_return) ** (periods_per_year / len(returns)) - 1
    m.volatility = float(np.std(returns) * np.sqrt(periods_per_year))
    
    neg_returns = returns[returns < 0]
    m.downside_volatility = float(np.std(neg_returns) * np.sqrt(periods_per_year)) if len(neg_returns) > 0 else 0.0
    
    m.skewness = float(_skewness(returns))
    m.kurtosis = float(_kurtosis(returns))
    
    # === Risk-Adjusted Returns ===
    rf_per_period = risk_free_rate / periods_per_year
    excess = returns - rf_per_period
    
    m.sharpe_ratio = float(np.mean(excess) / np.std(excess) * np.sqrt(periods_per_year)) if np.std(excess) > 0 else 0.0
    
    neg_excess = excess[excess < 0]
    downside_std = float(np.std(neg_excess)) if len(neg_excess) > 0 else 1e-10
    m.sortino_ratio = float(np.mean(excess) / downside_std * np.sqrt(periods_per_year))
    
    m.max_drawdown = _max_drawdown(eq)
    m.calmar_ratio = m.annualized_return / m.max_drawdown if m.max_drawdown > 0 else float('inf')
    
    # Omega ratio: ∫max(r-t,0)dr / ∫max(t-r,0)dr where t = threshold (0)
    gains = np.sum(np.maximum(returns, 0))
    losses = np.sum(np.maximum(-returns, 0))
    m.omega_ratio = float(gains / losses) if losses > 0 else float('inf')
    
    # === Tail Risk ===
    m.var_95 = float(-np.percentile(returns, 5))    # Loss at 5th percentile
    m.var_99 = float(-np.percentile(returns, 1))    # Loss at 1st percentile
    
    tail_5 = returns[returns <= np.percentile(returns, 5)]
    m.cvar_95 = float(-np.mean(tail_5)) if len(tail_5) > 0 else 0.0
    
    tail_1 = returns[returns <= np.percentile(returns, 1)]
    m.cvar_99 = float(-np.mean(tail_1)) if len(tail_1) > 0 else 0.0
    
    m.max_drawdown_duration = _max_dd_duration(eq)
    m.ulcer_index = _ulcer_index(eq)
    
    # === Trade Quality ===
    if trades:
        pnls = [t.get("pnl", 0) for t in trades]
        m.n_trades = len(pnls)
        
        wins = [p for p in pnls if p > 0]
        losses_list = [p for p in pnls if p <= 0]
        
        m.win_rate = len(wins) / m.n_trades if m.n_trades > 0 else 0.0
        m.avg_win = float(np.mean(wins)) if wins else 0.0
        m.avg_loss = float(np.mean(losses_list)) if losses_list else 0.0
        m.payoff_ratio = abs(m.avg_win / m.avg_loss) if m.avg_loss != 0 else float('inf')
        m.profit_factor = sum(wins) / abs(sum(losses_list)) if losses_list and sum(losses_list) != 0 else float('inf')
        m.expectancy = float(np.mean(pnls))
        
        # Consecutive win/loss streaks
        m.max_consecutive_losses = _max_streak(pnls, negative=True)
        m.max_consecutive_wins = _max_streak(pnls, negative=False)
    
    # === Risk Grade ===
    m.risk_grade = _risk_grade(m)
    
    return m


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    return float(np.max(dd))


def _max_dd_duration(equity: np.ndarray) -> int:
    peak = np.maximum.accumulate(equity)
    in_dd = equity < peak
    max_dur = 0
    current = 0
    for v in in_dd:
        if v:
            current += 1
            max_dur = max(max_dur, current)
        else:
            current = 0
    return max_dur


def _ulcer_index(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd_pct = (peak - equity) / peak
    return float(np.sqrt(np.mean(dd_pct ** 2)))


def _skewness(returns: np.ndarray) -> float:
    n = len(returns)
    if n < 3:
        return 0.0
    m = np.mean(returns)
    s = np.std(returns)
    if s == 0:
        return 0.0
    return float(np.mean(((returns - m) / s) ** 3))


def _kurtosis(returns: np.ndarray) -> float:
    n = len(returns)
    if n < 4:
        return 0.0
    m = np.mean(returns)
    s = np.std(returns)
    if s == 0:
        return 0.0
    return float(np.mean(((returns - m) / s) ** 4) - 3)  # Excess kurtosis


def _max_streak(pnls: list[float], negative: bool = True) -> int:
    max_streak = 0
    current = 0
    for p in pnls:
        if (negative and p <= 0) or (not negative and p > 0):
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def _risk_grade(m: QuantMetrics) -> str:
    """Assign A-F risk grade based on composite metrics."""
    score = 0
    
    # Sharpe > 2 = good, > 3 = great
    if m.sharpe_ratio > 3: score += 3
    elif m.sharpe_ratio > 2: score += 2
    elif m.sharpe_ratio > 1: score += 1
    
    # Sortino > Sharpe means less downside = good
    if m.sortino_ratio > m.sharpe_ratio * 1.3: score += 2
    elif m.sortino_ratio > m.sharpe_ratio: score += 1
    
    # Max DD < 5% = excellent
    if m.max_drawdown < 0.01: score += 3
    elif m.max_drawdown < 0.05: score += 2
    elif m.max_drawdown < 0.10: score += 1
    
    # Win rate > 65% = good
    if m.win_rate > 0.85: score += 2
    elif m.win_rate > 0.65: score += 1
    
    # Profit factor > 2 = good
    if m.profit_factor > 3: score += 2
    elif m.profit_factor > 1.5: score += 1
    
    # Low tail risk
    if m.cvar_99 < 0.02: score += 1
    
    # Positive skew = good (right tail fatter)
    if m.skewness > 0: score += 1
    
    if score >= 12: return "A+"
    if score >= 10: return "A"
    if score >= 8: return "B+"
    if score >= 6: return "B"
    if score >= 4: return "C"
    if score >= 2: return "D"
    return "F"
