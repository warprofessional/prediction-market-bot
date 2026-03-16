"""Live simulation using REAL signals from Polymarket data.

No ML model. Uses the same signals the Twitter research identified:
- Price momentum (from historical price changes)
- Orderbook imbalance (from real orderbook)
- Crowd bias detection (from price extremes)
- Mean-reversion (from price distance to 50%)
- Cross-check YES+NO pricing (arb detection)

The strategy IS the model. The 7 signals ARE the edge.

Usage: python3 live_sim.py [--minutes 10] [--interval 30]
"""
import asyncio
import json
import logging
import math
import sys
import time
from datetime import datetime

import httpx

from src.strategies.market_filter import should_trade, classify_market

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("sim")

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"


def compute_model_prob_from_signals(market_price, prev_price, book, yes_no_total):
    """Compute model probability using REAL signals — no ML needed.
    
    This is what the Twitter research taught us:
    1. OBI from real orderbook → which direction has more pressure
    2. Momentum from price change → is the market trending
    3. YES+NO mispricing → arb opportunity
    4. Crowd bias at extremes → fade overconfidence
    5. Mean-reversion tendency → prices revert to fundamentals
    """
    signals = {}
    adjustment = 0.0
    
    # --- SIGNAL: Orderbook Imbalance ---
    # @MarikWeb3: OBI explains ~65% of short-term price variance
    if book:
        bid_d = book["bid_depth"]
        ask_d = book["ask_depth"]
        total = bid_d + ask_d
        if total > 0:
            obi = (bid_d - ask_d) / total  # +1 = all bids, -1 = all asks
            # Strong bid pressure → price should go up
            if obi > 0.2:
                adjustment += 0.01 * obi
                signals["obi"] = f"+{obi:.2f} (buy pressure)"
            elif obi < -0.2:
                adjustment += 0.01 * obi
                signals["obi"] = f"{obi:.2f} (sell pressure)"
    
    # --- SIGNAL: Momentum ---
    # @db_polybot (live Rust bot): "waits for momentum confirmation before sizing"
    if prev_price and prev_price != market_price:
        price_change = market_price - prev_price
        # If price is trending, model should agree with trend slightly
        if abs(price_change) > 0.005:
            adjustment += price_change * 0.3  # Follow momentum partially
            signals["momentum"] = f"{price_change:+.3f}"
    
    # --- SIGNAL: YES+NO Arb ---
    # Gabagool ladder bot: if YES+NO < 1.00, there's free money
    if yes_no_total < 0.995:
        arb_edge = 1.0 - yes_no_total
        # If there's arb, price should be higher (market undervalues)
        adjustment += arb_edge * 0.5
        signals["arb"] = f"gap={arb_edge:.3f}"
    
    # --- SIGNAL: Optimism Tax Fade ---
    # Becker (2026): "Takers overpay for YES"
    # When crowd is extremely confident (>85%), fade slightly
    if market_price > 0.85:
        fade = (market_price - 0.85) * 0.15  # Small fade of extreme confidence
        adjustment -= fade
        signals["optimism_fade"] = f"-{fade:.3f} (crowd too bullish)"
    elif market_price < 0.15:
        fade = (0.15 - market_price) * 0.15
        adjustment += fade
        signals["optimism_fade"] = f"+{fade:.3f} (crowd too bearish)"
    
    # --- SIGNAL: Mean-Reversion ---
    # @frostikkkk: EMA ±8% deviation mean-reversion
    # Prices far from 50% tend to revert (on binary markets without strong info)
    distance_from_mid = market_price - 0.5
    if abs(distance_from_mid) > 0.3:
        reversion = -distance_from_mid * 0.05  # Gentle pull toward center
        adjustment += reversion
        signals["mean_rev"] = f"{reversion:+.3f}"
    
    # --- SIGNAL: Spread Quality ---
    # Wide spread = uncertain market = bigger potential edge
    if book and book["spread"] < 0.03:
        # Tight spread = efficient market, reduce confidence
        adjustment *= 0.7
        signals["spread"] = f"tight ({book['spread']:.3f}), reduced"
    
    model_prob = max(0.01, min(0.99, market_price + adjustment))
    return model_prob, signals


class PaperTrader:
    def __init__(self, capital=1000.0):
        self.initial_capital = capital
        self.capital = capital
        self.positions = {}
        self.closed_trades = []
        self.scan_count = 0
        self.signal_count = 0
        self.price_cache = {}
    
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
        n_open = len(self.positions)
        n_closed = len(self.closed_trades)
        wins = sum(1 for t in self.closed_trades if t["pnl"] > 0)
        wr = wins / n_closed * 100 if n_closed else 0
        return (
            f"💰 ${eq:.2f} ({ret:+.2%}) | "
            f"Open:{n_open} Closed:{n_closed} WR:{wr:.0f}% | "
            f"Real:${self.realized_pnl:.2f} Unreal:${self.unrealized_pnl:.2f}"
        )


async def fetch_markets():
    async with httpx.AsyncClient(timeout=15) as c:
        resp = await c.get(f"{GAMMA}/markets", params={
            "limit": 80, "active": True, "order": "liquidity", "ascending": False
        })
        resp.raise_for_status()
        out = []
        for m in resp.json():
            tokens = json.loads(m.get("clobTokenIds", "[]")) if isinstance(m.get("clobTokenIds"), str) else m.get("clobTokenIds", [])
            prices = json.loads(m.get("outcomePrices", "[]")) if isinstance(m.get("outcomePrices"), str) else m.get("outcomePrices", [])
            if len(tokens) >= 2 and len(prices) >= 2:
                q = m.get("question", "")
                yp = float(prices[0])
                np_ = float(prices[1])
                liq = float(m.get("liquidity", 0))
                ok, reason = should_trade(q, yp, liq)
                if ok:
                    out.append({
                        "q": q, "token": tokens[0], "price": yp,
                        "no_price": np_, "total": yp + np_,
                        "liq": liq, "type": classify_market(q),
                    })
        return out


async def fetch_book(token_id):
    async with httpx.AsyncClient(timeout=8) as c:
        try:
            resp = await c.get(f"{CLOB}/book", params={"token_id": token_id})
            if resp.status_code == 200:
                data = resp.json()
                bids = data.get("bids", [])
                asks = data.get("asks", [])
                return {
                    "best_bid": float(bids[0]["price"]) if bids else 0,
                    "best_ask": float(asks[0]["price"]) if asks else 1,
                    "bid_depth": sum(float(b["size"]) * float(b["price"]) for b in bids[:5]),
                    "ask_depth": sum(float(a["size"]) * float(a["price"]) for a in asks[:5]),
                    "spread": (float(asks[0]["price"]) - float(bids[0]["price"])) if bids and asks else 1,
                }
        except:
            pass
    return None


async def run_sim(minutes=10, interval=30):
    from src.strategies.unified_strategy import unified_strategy
    
    trader = PaperTrader(capital=1000.0)
    end_time = time.time() + minutes * 60
    
    log.info("=" * 70)
    log.info(f"  LIVE SIM — {minutes}min, every {interval}s | REAL SIGNALS (no ML)")
    log.info(f"  Signals: OBI + Momentum + Arb + Optimism Fade + Mean-Rev")
    log.info(f"  Capital: ${trader.capital:.0f}")
    log.info("=" * 70)
    
    while time.time() < end_time:
        trader.scan_count += 1
        remaining = (end_time - time.time()) / 60
        log.info(f"\n--- Scan #{trader.scan_count} | {remaining:.1f}min left ---")
        
        try:
            markets = await fetch_markets()
            log.info(f"  {len(markets)} tradeable markets")
            
            # Update prices & manage exits
            for tid in list(trader.positions.keys()):
                pos = trader.positions[tid]
                for m in markets:
                    if m["token"] == tid:
                        trader.price_cache[tid] = m["price"]
                        break
                
                current = trader.price_cache.get(tid, pos["entry"])
                if pos["side"] == "buy":
                    ur = (current - pos["entry"]) * pos["shares"]
                else:
                    ur = (pos["entry"] - current) * pos["shares"]
                
                pos["hold"] = pos.get("hold", 0) + 1
                
                should_exit = (
                    ur < -pos["usd"] * 0.015 or
                    pos["hold"] >= 3 or
                    ur > pos["usd"] * 0.025
                )
                
                if should_exit:
                    pnl = ur - pos["usd"] * 0.002
                    trader.capital += pos["usd"] + pnl
                    trader.closed_trades.append({"market": pos["market"], "pnl": pnl, "side": pos["side"]})
                    e = "✅" if pnl > 0 else "❌"
                    log.info(f"  {e} CLOSE {pos['side']} {pos['market'][:40]} PnL:${pnl:.2f}")
                    del trader.positions[tid]
            
            # Scan for new entries using REAL signals
            for m in markets:
                if m["token"] in trader.positions or len(trader.positions) >= 5:
                    continue
                
                mp = m["price"]
                if mp <= 0.03 or mp >= 0.97:
                    continue
                
                prev = trader.price_cache.get(m["token"], mp)
                trader.price_cache[m["token"]] = mp
                
                # Get real orderbook
                book = await fetch_book(m["token"])
                
                # Compute model prob from REAL signals
                model_prob, sigs = compute_model_prob_from_signals(
                    mp, prev, book, m["total"]
                )
                
                # Run through the full strategy
                signal = unified_strategy(model_prob, mp, trader.capital, prev)
                
                if signal and signal.get("size_usd", 0) > 5:
                    size = min(signal["size_usd"], trader.capital * 0.15)
                    fee = size * 0.002
                    slippage = mp * 0.001
                    entry = mp + slippage if signal["side"] == "buy" else mp - slippage
                    
                    trader.capital -= size
                    shares = (size - fee) / max(entry, 0.001)
                    trader.positions[m["token"]] = {
                        "market": m["q"][:50], "side": signal["side"],
                        "entry": entry, "usd": size, "shares": shares, "hold": 0,
                    }
                    trader.signal_count += 1
                    sig_str = ", ".join(f"{k}:{v}" for k, v in sigs.items())
                    log.info(f"  🔵 {signal['side'].upper()} ${size:.0f} {m['q'][:40]} @ {entry:.3f}")
                    log.info(f"     Signals: {sig_str}")
                
                await asyncio.sleep(0.05)
            
            log.info(f"  {trader.status_line()}")
        
        except Exception as e:
            log.error(f"  Error: {e}")
        
        await asyncio.sleep(interval)
    
    # Final
    log.info("\n" + "=" * 70)
    log.info("  SIMULATION COMPLETE")
    log.info("=" * 70)
    log.info(f"  Duration: {minutes}min | Scans: {trader.scan_count} | Signals: {trader.signal_count}")
    log.info(f"  Closed: {len(trader.closed_trades)} | Open: {len(trader.positions)}")
    if trader.closed_trades:
        wins = sum(1 for t in trader.closed_trades if t["pnl"] > 0)
        log.info(f"  Win rate: {wins}/{len(trader.closed_trades)} ({wins/len(trader.closed_trades)*100:.0f}%)")
    log.info(f"  {trader.status_line()}")
    log.info("=" * 70)


if __name__ == "__main__":
    minutes = 10
    interval = 30
    for i, arg in enumerate(sys.argv):
        if arg == "--minutes" and i + 1 < len(sys.argv):
            minutes = int(sys.argv[i + 1])
        elif arg == "--interval" and i + 1 < len(sys.argv):
            interval = int(sys.argv[i + 1])
    asyncio.run(run_sim(minutes, interval))
