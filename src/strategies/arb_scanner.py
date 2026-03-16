"""Strategy 1 & 2: Arbitrage Scanner.

Scans for:
1. Intra-platform: YES + NO < $1.00 (guaranteed profit)
2. Cross-platform: Polymarket vs Kalshi price divergence

Based on @Mnilax arb loop architecture:
  Scan → Verify (VWAP) → Execute (FOK) → Recover → Safety
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from src.data.polymarket_client import PolymarketClient, OrderBook
from src.strategies.math_core import vpin, kelly_fraction, fractional_kelly

logger = logging.getLogger(__name__)


@dataclass
class ArbOpportunity:
    """A detected arbitrage opportunity."""
    market_name: str
    strategy: str  # "intra_yesno" or "cross_platform"
    
    # Prices
    yes_price: float
    no_price: float
    total_cost: float
    edge: float  # 1.0 - total_cost for intra, price diff for cross
    
    # Sizing
    recommended_size: float  # In shares
    expected_profit: float
    
    # Metadata
    yes_token: str = ""
    no_token: str = ""
    platform: str = "polymarket"
    timestamp: float = field(default_factory=time.time)
    
    # Execution quality
    yes_depth: float = 0.0  # Available depth at this price
    no_depth: float = 0.0
    vwap_slippage: float = 0.0


@dataclass
class ArbResult:
    """Result of an arbitrage execution."""
    opportunity: ArbOpportunity
    executed: bool
    pnl: float
    fees: float
    slippage: float
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None


class ArbScanner:
    """Scans and executes arbitrage opportunities."""
    
    def __init__(
        self,
        poly_client: PolymarketClient,
        min_edge: float = 0.005,
        max_position: float = 500.0,
        kelly_fraction_val: float = 0.25,
        dry_run: bool = True,
    ):
        self.poly = poly_client
        self.min_edge = min_edge
        self.max_position = max_position
        self.kelly_frac = kelly_fraction_val
        self.dry_run = dry_run
        
        # Tracking
        self.total_scans = 0
        self.total_opps_found = 0
        self.total_trades = 0
        self.total_pnl = 0.0
        self.results: list[ArbResult] = []

    async def scan_intra_arb(self) -> list[ArbOpportunity]:
        """Scan for YES + NO < $1.00 opportunities.
        
        The Gabagool Ladder bot approach:
        1. Split collateral into YES/NO
        2. When combined < $1.00, buy both
        3. Guaranteed profit = $1.00 - total_cost per share
        """
        self.total_scans += 1
        opportunities = []
        
        raw_opps = await self.poly.scan_arb_opportunities()
        
        for opp in raw_opps:
            if opp["edge"] < self.min_edge:
                continue
            
            # Calculate VWAP and verify executability
            try:
                yes_book = await self.poly.get_orderbook(opp["yes_token"])
                no_book = await self.poly.get_orderbook(opp["no_token"])
                
                # Verify depth supports our position size
                max_shares = min(
                    self._executable_shares(yes_book, "buy"),
                    self._executable_shares(no_book, "buy"),
                    self.max_position / max(opp["total"], 0.01),
                )
                
                if max_shares < 1:
                    continue
                
                # Calculate VWAP cost for desired shares
                yes_vwap = self._vwap_cost(yes_book, max_shares, "buy")
                no_vwap = self._vwap_cost(no_book, max_shares, "buy")
                total_vwap = yes_vwap + no_vwap
                
                if total_vwap >= 1.0:
                    continue  # Edge disappeared at VWAP
                
                edge_after_slippage = 1.0 - total_vwap
                
                arb = ArbOpportunity(
                    market_name=opp["market"][:80],
                    strategy="intra_yesno",
                    yes_price=opp["yes_price"],
                    no_price=opp["no_price"],
                    total_cost=total_vwap,
                    edge=edge_after_slippage,
                    recommended_size=max_shares,
                    expected_profit=edge_after_slippage * max_shares,
                    yes_token=opp["yes_token"],
                    no_token=opp["no_token"],
                    yes_depth=yes_book.bid_depth,
                    no_depth=no_book.bid_depth,
                    vwap_slippage=total_vwap - opp["total"],
                )
                opportunities.append(arb)
                self.total_opps_found += 1
                
            except Exception as e:
                logger.debug(f"Error checking {opp['slug']}: {e}")
                continue
            
            await asyncio.sleep(0.1)
        
        # Sort by edge (best first)
        return sorted(opportunities, key=lambda x: x.edge, reverse=True)

    def _executable_shares(self, book: OrderBook, side: str) -> float:
        """Calculate max executable shares at reasonable slippage."""
        orders = book.asks if side == "buy" else book.bids
        total_shares = sum(o["size"] for o in orders[:5])  # Top 5 levels
        return total_shares

    def _vwap_cost(self, book: OrderBook, shares: float, side: str) -> float:
        """Calculate VWAP cost for desired shares.
        
        VWAP = Σ(price × volume) / Σ(volume)
        """
        orders = book.asks if side == "buy" else book.bids
        if side == "buy":
            orders = sorted(orders, key=lambda x: x["price"])  # Best ask first
        else:
            orders = sorted(orders, key=lambda x: x["price"], reverse=True)  # Best bid first
        
        remaining = shares
        total_cost = 0.0
        total_filled = 0.0
        
        for order in orders:
            fill = min(remaining, order["size"])
            total_cost += fill * order["price"]
            total_filled += fill
            remaining -= fill
            if remaining <= 0:
                break
        
        if total_filled == 0:
            return 1.0  # No liquidity
        
        return total_cost / total_filled

    async def execute_arb(self, opp: ArbOpportunity) -> ArbResult:
        """Execute an arbitrage opportunity.
        
        Uses FOK (Fill-or-Kill) orders for atomic execution.
        """
        if self.dry_run:
            logger.info(
                f"[DRY RUN] Would execute {opp.strategy}: "
                f"{opp.market_name[:40]} | "
                f"Edge={opp.edge:.4f} | "
                f"Size={opp.recommended_size:.0f} | "
                f"Expected P&L=${opp.expected_profit:.2f}"
            )
            
            # Simulate execution with some slippage
            simulated_slippage = opp.edge * 0.1  # 10% edge decay
            simulated_pnl = opp.expected_profit * 0.85  # 85% realization
            
            result = ArbResult(
                opportunity=opp,
                executed=True,
                pnl=simulated_pnl,
                fees=opp.recommended_size * 0.002,  # ~0.2% fees
                slippage=simulated_slippage,
            )
            self.total_trades += 1
            self.total_pnl += result.pnl
            self.results.append(result)
            return result
        
        # REAL EXECUTION (Phase 2)
        # TODO: Implement with py-clob-client
        # 1. Create market buy order for YES token (FOK)
        # 2. Create market buy order for NO token (FOK)
        # 3. If either fails, cancel and log
        # 4. If both succeed, profit = 1.0 - total_cost per share
        raise NotImplementedError("Live trading not yet implemented")

    async def run_scan_loop(self, interval: float = 2.0, max_iterations: Optional[int] = None):
        """Main scanning loop.
        
        Args:
            interval: Seconds between scans
            max_iterations: Stop after N iterations (None = infinite)
        """
        iteration = 0
        logger.info(f"Starting arb scanner (interval={interval}s, dry_run={self.dry_run})")
        
        while max_iterations is None or iteration < max_iterations:
            try:
                opps = await self.scan_intra_arb()
                
                for opp in opps[:3]:  # Execute top 3 opportunities
                    result = await self.execute_arb(opp)
                    if result.executed:
                        logger.info(
                            f"Trade #{self.total_trades}: {opp.strategy} | "
                            f"P&L=${result.pnl:.2f} | "
                            f"Running P&L=${self.total_pnl:.2f}"
                        )
                
                if not opps:
                    logger.debug(f"Scan #{self.total_scans}: No opportunities found")
                
            except Exception as e:
                logger.error(f"Scan error: {e}")
            
            iteration += 1
            await asyncio.sleep(interval)

    def get_stats(self) -> dict:
        """Get scanner statistics."""
        return {
            "total_scans": self.total_scans,
            "total_opps_found": self.total_opps_found,
            "total_trades": self.total_trades,
            "total_pnl": self.total_pnl,
            "avg_pnl_per_trade": self.total_pnl / max(self.total_trades, 1),
            "win_rate": sum(1 for r in self.results if r.pnl > 0) / max(len(self.results), 1),
        }
