"""CEX Lag Strategy — The #1 proven money maker.

How it works (from k9Q2 cluster, $1.5M+ all-time):
1. Watch Binance/Coinbase BTC/ETH price via API
2. When CEX price moves >0.3% in <1min, Polymarket 5-min up/down markets lag
3. Buy the direction on Polymarket before it adjusts (200ms-7s window)
4. Collect when market catches up

No ML needed. Just speed of data + the Kelly/signal framework we already built.

This module:
- Fetches real CEX price data
- Detects significant moves
- Generates signals for our unified strategy
- Can backtest against historical candle data
"""
import json
import math
import time
from dataclasses import dataclass
from typing import Optional

import httpx
import numpy as np


@dataclass
class CEXMove:
    """A detected significant move on CEX."""
    symbol: str
    direction: str  # "up" or "down"
    magnitude: float  # % move
    price_before: float
    price_after: float
    timestamp: float


def detect_moves_from_candles(candles: list, threshold: float = 0.003) -> list[CEXMove]:
    """Detect significant moves from 1-min candle data.
    
    Args:
        candles: Binance kline format [[ts, o, h, l, c, ...], ...]
        threshold: minimum move to detect (0.003 = 0.3%)
    """
    moves = []
    for i in range(1, len(candles)):
        prev_close = float(candles[i-1][4])
        curr_close = float(candles[i][4])
        ret = (curr_close - prev_close) / prev_close
        
        if abs(ret) >= threshold:
            moves.append(CEXMove(
                symbol="BTC",
                direction="up" if ret > 0 else "down",
                magnitude=abs(ret),
                price_before=prev_close,
                price_after=curr_close,
                timestamp=candles[i][0] / 1000,
            ))
    return moves


def simulate_polymarket_5min(btc_candles: list, move: CEXMove, 
                              lag_seconds: float = 3.0) -> dict:
    """Simulate what a 5-min up/down market would look like during a CEX move.
    
    When BTC moves on Binance, the Polymarket 5-min market should adjust.
    But there's a lag — the market takes 1-7 seconds to catch up.
    During that lag, we buy the correct direction at the old price.
    
    Returns estimated PnL for this trade.
    """
    # The 5-min market pays $1 if BTC goes up (or down) in the next 5 min
    # Before the move, the market is ~50¢ (uncertain)
    # After BTC moves 0.3%+, the market should be 60-70¢ for the correct direction
    # During the lag, it's still ~50¢
    
    entry_price = 0.50  # Buy during lag (market hasn't moved yet)
    
    # After the market adjusts, the correct direction trades at higher price
    # Magnitude of 0.3% → ~55-60¢, 0.5%+ → 65-75¢
    if move.magnitude >= 0.005:
        exit_price = 0.70  # Strong move → high confidence
    elif move.magnitude >= 0.004:
        exit_price = 0.65
    elif move.magnitude >= 0.003:
        exit_price = 0.58
    else:
        exit_price = 0.53
    
    return {
        "entry": entry_price,
        "exit": exit_price,
        "direction": move.direction,
        "btc_move": move.magnitude,
        "edge": exit_price - entry_price,
    }


def backtest_cex_lag(candle_path: str = "/tmp/btc_1m_candles.json",
                      capital: float = 1000.0,
                      kelly_frac: float = 0.50,
                      threshold: float = 0.003,
                      fee_rate: float = 0.002) -> dict:
    """Backtest CEX lag strategy against real Binance candle data.
    
    For each detected move:
    1. Assume we enter at ~50¢ (market hasn't adjusted)
    2. Market adjusts to fair value based on move magnitude
    3. We exit at adjusted price, minus fees
    """
    with open(candle_path) as f:
        candles = json.load(f)
    
    moves = detect_moves_from_candles(candles, threshold)
    
    trades = []
    running_capital = capital
    equity_curve = [capital]
    
    for move in moves:
        sim = simulate_polymarket_5min(candles, move)
        
        # Kelly sizing
        edge = sim["edge"]
        if edge <= 0:
            continue
        
        # Size: fraction of capital
        size = running_capital * kelly_frac * min(edge * 5, 0.15)  # Scale by edge
        size = min(size, running_capital * 0.15)  # Cap at 15%
        size = max(size, 5)  # Minimum $5
        
        # Entry
        entry_cost = size  # Buy shares at entry_price
        shares = (size - size * fee_rate) / sim["entry"]
        
        # Exit
        pnl = shares * (sim["exit"] - sim["entry"]) - size * fee_rate
        
        running_capital += pnl
        equity_curve.append(running_capital)
        
        trades.append({
            "direction": sim["direction"],
            "btc_move": f"{move.magnitude*100:.2f}%",
            "entry": sim["entry"],
            "exit": sim["exit"],
            "size": size,
            "pnl": pnl,
            "capital_after": running_capital,
        })
    
    total_pnl = running_capital - capital
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    
    return {
        "period": f"{len(candles)} 1-min candles ({len(candles)/60:.1f} hours)",
        "total_moves": len(moves),
        "trades": len(trades),
        "total_pnl": total_pnl,
        "total_return": total_pnl / capital,
        "win_rate": len(wins) / max(len(trades), 1),
        "avg_pnl": np.mean([t["pnl"] for t in trades]) if trades else 0,
        "best_trade": max(t["pnl"] for t in trades) if trades else 0,
        "worst_trade": min(t["pnl"] for t in trades) if trades else 0,
        "final_capital": running_capital,
        "trades_detail": trades,
        "equity_curve": equity_curve,
    }


async def fetch_live_btc_price():
    """Get current BTC price from multiple sources for reliability."""
    async with httpx.AsyncClient(timeout=5) as c:
        prices = []
        # Binance.US
        try:
            resp = await c.get("https://api.binance.us/api/v3/ticker/price", params={"symbol": "BTCUSD"})
            prices.append(float(resp.json()["price"]))
        except:
            pass
        # Coinbase
        try:
            resp = await c.get("https://api.coinbase.com/v2/prices/BTC-USD/spot")
            prices.append(float(resp.json()["data"]["amount"]))
        except:
            pass
        
        if prices:
            return np.mean(prices)
        return None
