"""10-minute live simulation against real Polymarket.

Scans real markets every 30s, generates signals, simulates execution,
tracks paper P&L in real-time. Shows exactly what the bot would do.

Usage: python3 live_sim.py [--minutes 10] [--interval 30]
"""
import asyncio
import json
import logging
import sys
import time
from datetime import datetime

import httpx
import numpy as np

from src.strategies.unified_strategy import unified_strategy
from src.strategies.market_filter import should_trade, classify_market

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("sim")

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"


class PaperTrader:
    def __init__(self, capital=1000.0):
        self.initial_capital = capital
        self.capital = capital
        self.positions = {}  # token_id → position
        self.closed_trades = []
        self.scan_count = 0
        self.signal_count = 0
        self.price_cache = {}  # token_id → last price
    
    @property
    def unrealized_pnl(self):
        pnl = 0
        for tid, pos in self.positions.items():
            current = self.price_cache.get(tid, pos["entry"])
            if pos["side"] == "buy":
                pnl += (current - pos["entry"]) * pos["shares"]
            else:
                pnl += (pos["entry"] - current) * pos["shares"]
        return pnl
    
    @property
    def total_equity(self):
        return self.capital + sum(p["usd"] for p in self.positions.values()) + self.unrealized_pnl
    
    @property
    def realized_pnl(self):
        return sum(t["pnl"] for t in self.closed_trades)
    
    def status_line(self):
        eq = self.total_equity
        ret = (eq - self.initial_capital) / self.initial_capital
        ur = self.unrealized_pnl
        n_open = len(self.positions)
        n_closed = len(self.closed_trades)
        wins = sum(1 for t in self.closed_trades if t["pnl"] > 0)
        wr = wins / n_closed * 100 if n_closed else 0
        return (
            f"💰 Equity: ${eq:.2f} ({ret:+.2%}) | "
            f"Open: {n_open} | Closed: {n_closed} (WR {wr:.0f}%) | "
            f"Realized: ${self.realized_pnl:.2f} | Unrealized: ${ur:.2f}"
        )


async def fetch_markets():
    async with httpx.AsyncClient(timeout=15) as c:
        resp = await c.get(f"{GAMMA}/markets", params={
            "limit": 60, "active": True, "order": "liquidity", "ascending": False
        })
        resp.raise_for_status()
        out = []
        for m in resp.json():
            tokens = json.loads(m.get("clobTokenIds", "[]")) if isinstance(m.get("clobTokenIds"), str) else m.get("clobTokenIds", [])
            prices = json.loads(m.get("outcomePrices", "[]")) if isinstance(m.get("outcomePrices"), str) else m.get("outcomePrices", [])
            if len(tokens) >= 2 and len(prices) >= 2:
                q = m.get("question", "")
                yp = float(prices[0])
                liq = float(m.get("liquidity", 0))
                ok, reason = should_trade(q, yp, liq)
                if ok:
                    out.append({"q": q, "token": tokens[0], "price": yp, "liq": liq, "type": classify_market(q)})
        return out


async def run_sim(minutes=10, interval=30):
    trader = PaperTrader(capital=1000.0)
    end_time = time.time() + minutes * 60
    
    log.info("=" * 65)
    log.info(f"  LIVE SIMULATION — {minutes} minutes, scanning every {interval}s")
    log.info(f"  Capital: ${trader.capital:.0f} | Strategy: unified | Markets: filtered")
    log.info("=" * 65)
    
    while time.time() < end_time:
        trader.scan_count += 1
        remaining = (end_time - time.time()) / 60
        log.info(f"\n--- Scan #{trader.scan_count} | {remaining:.1f} min remaining ---")
        
        try:
            markets = await fetch_markets()
            log.info(f"  {len(markets)} tradeable markets")
            
            # Update price cache and check exits
            for tid in list(trader.positions.keys()):
                pos = trader.positions[tid]
                # Find current price
                for m in markets:
                    if m["token"] == tid:
                        trader.price_cache[tid] = m["price"]
                        break
                
                current = trader.price_cache.get(tid, pos["entry"])
                if pos["side"] == "buy":
                    ur = (current - pos["entry"]) * pos["shares"]
                else:
                    ur = (pos["entry"] - current) * pos["shares"]
                
                pos["hold_scans"] = pos.get("hold_scans", 0) + 1
                
                # Exit conditions (same as backtester)
                should_exit = (
                    ur < -pos["usd"] * 0.015 or       # SL 1.5%
                    pos["hold_scans"] >= 3 or           # Max 3 scans
                    ur > pos["usd"] * 0.025             # TP 2.5%
                )
                
                if should_exit:
                    pnl = ur - pos["usd"] * 0.002  # Fee
                    trader.capital += pos["usd"] + pnl
                    trader.closed_trades.append({
                        "market": pos["market"],
                        "side": pos["side"],
                        "entry": pos["entry"],
                        "exit": current,
                        "pnl": pnl,
                        "hold": pos["hold_scans"],
                    })
                    emoji = "✅" if pnl > 0 else "❌"
                    log.info(f"  {emoji} CLOSED {pos['side'].upper()} {pos['market'][:40]} | PnL: ${pnl:.2f}")
                    del trader.positions[tid]
            
            # Scan for new signals
            for m in markets:
                if m["token"] in trader.positions:
                    continue
                if len(trader.positions) >= 5:  # Max 5 concurrent positions
                    break
                
                mp = m["price"]
                if mp <= 0.03 or mp >= 0.97:
                    continue
                
                prev = trader.price_cache.get(m["token"], mp)
                trader.price_cache[m["token"]] = mp
                
                # Strategy uses model_prob = slight perturbation of market
                # In real deployment this would be a real ML model
                for offset in [0.02, -0.02, 0.03, -0.03]:
                    model = max(0.01, min(0.99, mp + offset))
                    signal = unified_strategy(model, mp, trader.capital, prev)
                    
                    if signal and signal.get("size_usd", 0) > 5:
                        size = min(signal["size_usd"], trader.capital * 0.15)
                        fee = size * 0.002
                        slippage = mp * 0.001
                        entry = mp + slippage if signal["side"] == "buy" else mp - slippage
                        
                        trader.capital -= size
                        shares = (size - fee) / entry
                        trader.positions[m["token"]] = {
                            "market": m["q"][:50],
                            "side": signal["side"],
                            "entry": entry,
                            "usd": size,
                            "shares": shares,
                            "hold_scans": 0,
                        }
                        trader.signal_count += 1
                        log.info(f"  🔵 OPEN {signal['side'].upper()} ${size:.0f} on {m['q'][:45]} @ {entry:.3f}")
                        break
            
            log.info(f"  {trader.status_line()}")
            
        except Exception as e:
            log.error(f"  Error: {e}")
        
        await asyncio.sleep(interval)
    
    # Final report
    log.info("\n" + "=" * 65)
    log.info("  SIMULATION COMPLETE")
    log.info("=" * 65)
    log.info(f"  Duration: {minutes} minutes | Scans: {trader.scan_count}")
    log.info(f"  Signals generated: {trader.signal_count}")
    log.info(f"  Trades closed: {len(trader.closed_trades)}")
    if trader.closed_trades:
        wins = sum(1 for t in trader.closed_trades if t["pnl"] > 0)
        log.info(f"  Win rate: {wins}/{len(trader.closed_trades)} ({wins/len(trader.closed_trades)*100:.0f}%)")
        log.info(f"  Realized PnL: ${trader.realized_pnl:.2f}")
    log.info(f"  Open positions: {len(trader.positions)}")
    log.info(f"  {trader.status_line()}")
    log.info("=" * 65)


if __name__ == "__main__":
    minutes = 10
    interval = 30
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--minutes" and i < len(sys.argv) - 1:
            minutes = int(sys.argv[i + 1])
        elif arg == "--interval" and i < len(sys.argv) - 1:
            interval = int(sys.argv[i + 1])
    
    asyncio.run(run_sim(minutes, interval))
