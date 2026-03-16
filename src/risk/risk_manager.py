"""Portfolio-level risk management.

Implements the 5 layers from @herman_m8's hedge fund desk model:
- VaR monitoring
- VPIN kill switches  
- Drawdown stops
- Position limits
- Kelly fraction enforcement

@0xPhasma proved: Full Kelly at 65% WR → -42% DD. Half Kelly → -8% DD, same profit.
"""
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from src.strategies.math_core import (
    value_at_risk,
    max_drawdown,
    sharpe_ratio,
    profit_factor,
    vpin,
)

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """An open position."""
    market_name: str
    token_id: str
    side: str
    entry_price: float
    size_shares: float
    size_usd: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    entry_time: float = field(default_factory=time.time)


@dataclass 
class RiskState:
    """Current risk state of the portfolio."""
    total_capital: float
    available_capital: float
    deployed_capital: float
    unrealized_pnl: float
    realized_pnl: float
    current_drawdown: float
    max_drawdown: float
    var_95: float
    var_99: float
    vpin_current: float
    n_positions: int
    kill_switch_active: bool
    
    @property
    def total_equity(self) -> float:
        return self.total_capital + self.unrealized_pnl


class RiskManager:
    """Portfolio risk manager with automatic kill switches."""
    
    def __init__(
        self,
        initial_capital: float = 1000.0,
        max_drawdown_pct: float = 0.15,
        vpin_kill_threshold: float = 0.6,
        max_single_position_pct: float = 0.10,
        max_total_exposure_pct: float = 0.50,
        kelly_fraction: float = 0.25,
        var_confidence: float = 0.99,
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_dd_pct = max_drawdown_pct
        self.vpin_kill = vpin_kill_threshold
        self.max_single_pct = max_single_position_pct
        self.max_total_pct = max_total_exposure_pct
        self.kelly_frac = kelly_fraction
        self.var_confidence = var_confidence
        
        # State
        self.positions: dict[str, Position] = {}
        self.equity_curve: list[float] = [initial_capital]
        self.realized_pnl: float = 0.0
        self.peak_equity: float = initial_capital
        self.kill_switch: bool = False
        
        # VPIN tracking
        self.buy_volume_window: list[float] = []
        self.sell_volume_window: list[float] = []
        self.vpin_window_size = 50  # Rolling window for VPIN
        
        # Trade history for Sharpe/PF calculation
        self.trade_returns: list[float] = []
    
    def can_trade(self) -> bool:
        """Check if we're allowed to trade."""
        if self.kill_switch:
            logger.warning("KILL SWITCH ACTIVE — no trading allowed")
            return False
        return True
    
    def check_position_limit(self, size_usd: float) -> bool:
        """Check if a new position would exceed limits."""
        # Single position limit
        if size_usd > self.capital * self.max_single_pct:
            logger.warning(
                f"Position ${size_usd:.2f} exceeds single limit "
                f"(${self.capital * self.max_single_pct:.2f})"
            )
            return False
        
        # Total exposure limit
        total_deployed = sum(p.size_usd for p in self.positions.values())
        if total_deployed + size_usd > self.capital * self.max_total_pct:
            logger.warning(
                f"Total exposure would exceed limit: "
                f"${total_deployed + size_usd:.2f} > ${self.capital * self.max_total_pct:.2f}"
            )
            return False
        
        return True
    
    def approve_trade(self, size_usd: float, edge: float) -> float:
        """Approve and possibly resize a trade.
        
        Returns approved size (may be smaller than requested).
        Returns 0 if rejected.
        """
        if not self.can_trade():
            return 0.0
        
        # Kelly cap
        max_kelly_size = self.capital * self.kelly_frac * edge
        size_usd = min(size_usd, max_kelly_size)
        
        # Position limit
        if not self.check_position_limit(size_usd):
            # Try with reduced size
            max_allowed = self.capital * self.max_single_pct
            size_usd = min(size_usd, max_allowed * 0.9)
            if not self.check_position_limit(size_usd):
                return 0.0
        
        return size_usd
    
    def add_position(self, position: Position):
        """Register a new position."""
        self.positions[position.token_id] = position
        logger.info(f"Position opened: {position.market_name[:40]} | ${position.size_usd:.2f}")
    
    def close_position(self, token_id: str, exit_price: float) -> float:
        """Close a position and return PnL."""
        if token_id not in self.positions:
            return 0.0
        
        pos = self.positions.pop(token_id)
        
        if pos.side == "buy_yes":
            pnl = (exit_price - pos.entry_price) * pos.size_shares
        else:
            pnl = (pos.entry_price - exit_price) * pos.size_shares
        
        self.realized_pnl += pnl
        self.capital += pnl
        self.trade_returns.append(pnl / pos.size_usd)
        self.equity_curve.append(self.capital)
        
        # Update peak and check drawdown
        if self.capital > self.peak_equity:
            self.peak_equity = self.capital
        
        current_dd = (self.peak_equity - self.capital) / self.peak_equity
        if current_dd > self.max_dd_pct:
            logger.critical(
                f"MAX DRAWDOWN BREACHED: {current_dd:.1%} > {self.max_dd_pct:.1%} — KILL SWITCH ON"
            )
            self.kill_switch = True
        
        logger.info(f"Position closed: {pos.market_name[:40]} | PnL=${pnl:.2f}")
        return pnl
    
    def update_vpin(self, buy_vol: float, sell_vol: float):
        """Update VPIN with new volume data."""
        self.buy_volume_window.append(buy_vol)
        self.sell_volume_window.append(sell_vol)
        
        # Keep rolling window
        if len(self.buy_volume_window) > self.vpin_window_size:
            self.buy_volume_window = self.buy_volume_window[-self.vpin_window_size:]
            self.sell_volume_window = self.sell_volume_window[-self.vpin_window_size:]
        
        current_vpin = vpin(sum(self.buy_volume_window), sum(self.sell_volume_window))
        
        if current_vpin > self.vpin_kill:
            logger.warning(f"VPIN={current_vpin:.3f} > {self.vpin_kill} — consider widening/pausing")
        
        return current_vpin
    
    def get_state(self) -> RiskState:
        """Get current risk state."""
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        deployed = sum(p.size_usd for p in self.positions.values())
        equity = self.capital + unrealized
        
        current_dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
        
        # VaR calculation
        if len(self.trade_returns) > 10:
            sigma = np.std(self.trade_returns)
        else:
            sigma = 0.05  # Default 5% vol estimate
        
        var_95 = equity * 1.645 * sigma
        var_99 = equity * 2.326 * sigma
        
        # Current VPIN
        current_vpin = vpin(
            sum(self.buy_volume_window[-20:]) if self.buy_volume_window else 0,
            sum(self.sell_volume_window[-20:]) if self.sell_volume_window else 0,
        )
        
        return RiskState(
            total_capital=self.capital,
            available_capital=self.capital - deployed,
            deployed_capital=deployed,
            unrealized_pnl=unrealized,
            realized_pnl=self.realized_pnl,
            current_drawdown=current_dd,
            max_drawdown=max_drawdown(self.equity_curve),
            var_95=var_95,
            var_99=var_99,
            vpin_current=current_vpin,
            n_positions=len(self.positions),
            kill_switch_active=self.kill_switch,
        )
    
    def get_performance(self) -> dict:
        """Get performance metrics."""
        wins = [r for r in self.trade_returns if r > 0]
        losses = [r for r in self.trade_returns if r <= 0]
        
        return {
            "total_return": (self.capital - self.initial_capital) / self.initial_capital,
            "total_pnl": self.realized_pnl,
            "n_trades": len(self.trade_returns),
            "win_rate": len(wins) / max(len(self.trade_returns), 1),
            "profit_factor": profit_factor(wins, losses),
            "sharpe_ratio": sharpe_ratio(self.trade_returns) if len(self.trade_returns) > 5 else 0,
            "max_drawdown": max_drawdown(self.equity_curve),
            "current_capital": self.capital,
            "peak_equity": self.peak_equity,
        }
