"""Strategy 4: Bayesian EV Scanner with Kelly Sizing.

Continuously updates probability estimates using Bayesian inference,
trades when model_prob diverges from market_price by > threshold.

Based on @dreyk0o0 ($1K → $3.3K, 72% WR) and @DaoMariowhales (+237% in 9 days).
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.data.polymarket_client import PolymarketClient
from src.strategies.math_core import (
    bayesian_update_binary,
    expected_value,
    ev_full,
    fractional_kelly,
    empirical_kelly,
    estimate_edge_uncertainty,
    vpin,
)

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """A trading signal from the Bayesian model."""
    market_name: str
    token_id: str
    side: str  # "buy_yes" or "buy_no"
    
    # Probabilities
    model_prob: float     # Our Bayesian estimate
    market_price: float   # Current market price
    ev: float             # Expected value
    
    # Sizing
    kelly_size: float     # Quarter-Kelly fraction
    position_usd: float   # Dollar amount to trade
    
    # Confidence
    confidence: float     # How confident we are (0-1)
    cv_edge: float        # Coefficient of variation of edge
    prob_profitable: float  # MC-estimated probability edge is real
    
    timestamp: float = field(default_factory=time.time)


class BayesianEVScanner:
    """Bayesian expected value scanner.
    
    Pipeline:
    1. Get market prices from Polymarket
    2. Build prior from multiple data sources
    3. Update with evidence (news, orderflow, related markets)
    4. Compute EV = model_prob - market_price
    5. If EV > threshold → size with Kelly → trade
    """
    
    def __init__(
        self,
        poly_client: PolymarketClient,
        min_ev: float = 0.03,
        confidence_threshold: float = 0.6,
        kelly_fraction_val: float = 0.25,
        capital: float = 1000.0,
        max_position_pct: float = 0.10,
        dry_run: bool = True,
    ):
        self.poly = poly_client
        self.min_ev = min_ev
        self.confidence_threshold = confidence_threshold
        self.kelly_frac = kelly_fraction_val
        self.capital = capital
        self.max_position_pct = max_position_pct
        self.dry_run = dry_run
        
        # State
        self.priors: dict[str, float] = {}  # token_id → prior probability
        self.positions: dict[str, float] = {}  # token_id → position size
        self.equity_curve: list[float] = [capital]
        self.signals: list[Signal] = []
        self.trades: list[dict] = []
    
    def set_prior(self, token_id: str, prior: float):
        """Set or update prior for a market."""
        self.priors[token_id] = prior
    
    def update_with_evidence(
        self, 
        token_id: str, 
        sensitivity: float = 0.8,
        specificity: float = 0.8,
    ) -> float:
        """Update probability with new evidence.
        
        Args:
            sensitivity: P(signal | event) — how good is our signal at detecting events
            specificity: P(no signal | no event) — false positive rate
        
        Returns:
            Updated posterior probability
        """
        prior = self.priors.get(token_id, 0.5)
        posterior = bayesian_update_binary(prior, sensitivity, specificity)
        self.priors[token_id] = posterior
        return posterior
    
    async def scan_for_signals(self) -> list[Signal]:
        """Scan all markets for EV signals."""
        markets = await self.poly.get_markets(limit=100)
        signals = []
        
        for market in markets:
            if len(market.tokens) < 2:
                continue
            
            try:
                yes_token = market.tokens[0]
                market_price = await self.poly.get_price(yes_token)
                
                # Get or initialize prior
                if yes_token not in self.priors:
                    # Default: start with market price as prior (efficient market hypothesis)
                    self.priors[yes_token] = market_price
                
                model_prob = self.priors[yes_token]
                
                # Calculate EV
                ev = expected_value(model_prob, market_price)
                
                if abs(ev) < self.min_ev:
                    continue
                
                # Estimate edge uncertainty via Monte Carlo
                uncertainty = estimate_edge_uncertainty(
                    model_prob, market_price, sigma=0.1, n_simulations=2000
                )
                
                if uncertainty["prob_profitable"] < self.confidence_threshold:
                    continue
                
                # Size with empirical Kelly
                kelly_size = empirical_kelly(
                    model_prob, market_price, 
                    uncertainty["cv_edge"],
                    fraction=self.kelly_frac,
                )
                
                # Cap at max position
                position_usd = min(
                    kelly_size * self.capital,
                    self.capital * self.max_position_pct,
                )
                
                if position_usd < 1.0:
                    continue
                
                side = "buy_yes" if ev > 0 else "buy_no"
                
                signal = Signal(
                    market_name=market.question[:80],
                    token_id=yes_token,
                    side=side,
                    model_prob=model_prob,
                    market_price=market_price,
                    ev=ev,
                    kelly_size=kelly_size,
                    position_usd=position_usd,
                    confidence=uncertainty["prob_profitable"],
                    cv_edge=uncertainty["cv_edge"],
                    prob_profitable=uncertainty["prob_profitable"],
                )
                signals.append(signal)
                
            except Exception as e:
                logger.debug(f"Error scanning {market.slug}: {e}")
                continue
            
            await asyncio.sleep(0.05)
        
        # Sort by EV (best first)
        signals.sort(key=lambda s: abs(s.ev), reverse=True)
        self.signals.extend(signals)
        
        return signals
    
    async def execute_signal(self, signal: Signal) -> dict:
        """Execute a trading signal."""
        trade = {
            "market": signal.market_name,
            "side": signal.side,
            "size_usd": signal.position_usd,
            "entry_price": signal.market_price,
            "model_prob": signal.model_prob,
            "ev": signal.ev,
            "kelly": signal.kelly_size,
            "timestamp": time.time(),
            "executed": False,
            "pnl": 0.0,
        }
        
        if self.dry_run:
            # Simulate: assume 65% of EV is realized (realistic after slippage/fees)
            simulated_pnl = signal.ev * signal.position_usd * 0.65
            trade["executed"] = True
            trade["pnl"] = simulated_pnl
            self.capital += simulated_pnl
            self.equity_curve.append(self.capital)
            
            logger.info(
                f"[DRY RUN] {signal.side} on '{signal.market_name[:40]}' | "
                f"EV={signal.ev:.4f} | Size=${signal.position_usd:.2f} | "
                f"P&L=${simulated_pnl:.2f} | Capital=${self.capital:.2f}"
            )
        
        self.trades.append(trade)
        return trade
    
    def get_stats(self) -> dict:
        """Get scanner statistics."""
        executed = [t for t in self.trades if t["executed"]]
        wins = [t for t in executed if t["pnl"] > 0]
        losses = [t for t in executed if t["pnl"] <= 0]
        
        return {
            "total_signals": len(self.signals),
            "total_trades": len(executed),
            "win_rate": len(wins) / max(len(executed), 1),
            "total_pnl": sum(t["pnl"] for t in executed),
            "avg_pnl": sum(t["pnl"] for t in executed) / max(len(executed), 1),
            "avg_ev": sum(t["ev"] for t in executed) / max(len(executed), 1),
            "capital": self.capital,
            "equity_curve_len": len(self.equity_curve),
        }
