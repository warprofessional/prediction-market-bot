"""Live simulation of ALL strategies running simultaneously.

Runs 5 strategies + 3 portfolios in parallel against real Polymarket data.
Each gets $1,000 paper capital. Tracks P&L independently.

Usage: python3 live_all_strategies.py [--minutes 20] [--interval 30]
"""
import asyncio
import json
import logging
import math
import sys
import time
from datetime import datetime
from collections import defaultdict

import httpx
import numpy as np

from src.strategies.unified_strategy import unified_strategy
from src.strategies.portfolio import cash_like_strategy, growth_strategy, alpha_strategy
from src.strategies.market_filter import should_trade, classify_market

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("multi")

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"


# ── CEX Price Tracking ──────────────────────────────────────────
class CEXTracker:
    """Track BTC/ETH prices from Binance.US + Coinbase."""
    def __init__(self):
        self.prices = {}       # symbol → current price
        self.prev_prices = {}  # symbol → previous price
        self.history = defaultdict(list)  # symbol → [prices]
    
    async def update(self):
        async with httpx.AsyncClient(timeout=5) as c:
            for symbol, url, parse in [
                ("BTC", "https://api.binance.us/api/v3/ticker/price?symbol=BTCUSD", lambda r: float(r.json()["price"])),
                ("ETH", "https://api.binance.us/api/v3/ticker/price?symbol=ETHUSD", lambda r: float(r.json()["price"])),
            ]:
                try:
                    resp = await c.get(url)
                    price = parse(resp)
                    self.prev_prices[symbol] = self.prices.get(symbol, price)
                    self.prices[symbol] = price
                    self.history[symbol].append(price)
                except:
                    pass
    
    def get_move(self, symbol):
        """Return % move since last check."""
        if symbol in self.prices and symbol in self.prev_prices:
            prev = self.prev_prices[symbol]
            if prev > 0:
                return (self.prices[symbol] - prev) / prev
        return 0.0


# ── Paper Trader (per strategy) ─────────────────────────────────
class StrategyRunner:
    def __init__(self, name, capital=1000.0):
        self.name = name
        self.initial = capital
        self.capital = capital
        self.positions = {}
        self.closed = []
        self.signals_total = 0
    
    @property
    def equity(self):
        return self.capital + sum(p["usd"] for p in self.positions.values())
    
    @property
    def pnl(self):
        return self.equity - self.initial
    
    @property
    def ret(self):
        return self.pnl / self.initial
    
    @property
    def wr(self):
        if not self.closed: return 0.0
        return sum(1 for t in self.closed if t["pnl"] > 0) / len(self.closed)
    
    def status(self):
        w = sum(1 for t in self.closed if t["pnl"] > 0)
        return (
            f"${self.equity:>8.2f} ({self.ret:>+6.2%}) | "
            f"O:{len(self.positions)} C:{len(self.closed)} W:{w} | "
            f"PnL:${self.pnl:>+.2f}"
        )


# ── Market Data ─────────────────────────────────────────────────
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
                yp, np_ = float(prices[0]), float(prices[1])
                liq = float(m.get("liquidity", 0))
                ok, _ = should_trade(q, yp, liq)
                if ok:
                    out.append({"q": q, "token": tokens[0], "price": yp, "no_price": np_,
                                "total": yp + np_, "liq": liq, "type": classify_market(q)})
        return out


async def fetch_book(token_id):
    async with httpx.AsyncClient(timeout=8) as c:
        try:
            resp = await c.get(f"{CLOB}/book", params={"token_id": token_id})
            if resp.status_code == 200:
                data = resp.json()
                bids, asks = data.get("bids", []), data.get("asks", [])
                return {
                    "bid_depth": sum(float(b["size"]) * float(b["price"]) for b in bids[:5]),
                    "ask_depth": sum(float(a["size"]) * float(a["price"]) for a in asks[:5]),
                    "spread": (float(asks[0]["price"]) - float(bids[0]["price"])) if bids and asks else 1,
                }
        except:
            pass
    return None


# ── Signal Generation ───────────────────────────────────────────
def compute_model_prob(mp, prev_price, book, yes_no_total):
    """Compute model probability from real signals (no ML)."""
    adj = 0.0
    
    # OBI
    if book:
        total = book["bid_depth"] + book["ask_depth"]
        if total > 0:
            obi = (book["bid_depth"] - book["ask_depth"]) / total
            if abs(obi) > 0.2:
                adj += 0.01 * obi
    
    # Momentum
    if prev_price and prev_price != mp:
        change = mp - prev_price
        if abs(change) > 0.005:
            adj += change * 0.3
    
    # Arb
    if yes_no_total < 0.995:
        adj += (1.0 - yes_no_total) * 0.5
    
    # Optimism fade
    if mp > 0.85:
        adj -= (mp - 0.85) * 0.15
    elif mp < 0.15:
        adj += (0.15 - mp) * 0.15
    
    # Mean-reversion
    dist = mp - 0.5
    if abs(dist) > 0.3:
        adj -= dist * 0.05
    
    return max(0.01, min(0.99, mp + adj))


def cex_lag_signal(btc_move, capital):
    """CEX lag: trade when BTC moved >0.3%."""
    if abs(btc_move) < 0.003:
        return None
    
    if abs(btc_move) >= 0.005: edge = 0.20
    elif abs(btc_move) >= 0.004: edge = 0.15
    else: edge = 0.08
    
    size = capital * 0.10 * min(abs(btc_move) * 100, 1.0)
    size = min(size, capital * 0.15)
    if size < 5:
        return None
    
    return {
        "side": "buy" if btc_move > 0 else "sell",
        "size_usd": size,
        "signal_count": 3,
        "edge": edge,
        "source": f"BTC {btc_move:+.2%}",
    }


def arb_signal(yes_no_total, capital):
    """Arb: YES+NO < $1.00."""
    gap = 1.0 - yes_no_total
    if gap < 0.005:
        return None
    size = min(capital * 0.05, capital * gap * 10)
    return {"side": "buy", "size_usd": max(size, 5), "signal_count": 5, "edge": gap, "source": f"arb gap={gap:.3f}"}


# ── Main Simulation Loop ────────────────────────────────────────
async def run_all(minutes=20, interval=30):
    cex = CEXTracker()
    price_cache = {}
    scan_count = 0
    
    # Initialize all runners
    runners = {
        "CEX Lag":       StrategyRunner("CEX Lag"),
        "Arb Scanner":   StrategyRunner("Arb Scanner"),
        "Ensemble":      StrategyRunner("Ensemble"),
        "Cash-Like":     StrategyRunner("Cash-Like"),
        "Alpha":         StrategyRunner("Alpha"),
        "Port:Safe":     StrategyRunner("Port:Safe"),
        "Port:Balanced": StrategyRunner("Port:Balanced"),
        "Port:Aggro":    StrategyRunner("Port:Aggro"),
    }
    
    strategy_fns = {
        "Ensemble":  unified_strategy,
        "Cash-Like": cash_like_strategy,
        "Alpha":     alpha_strategy,
    }
    
    # Portfolio allocations
    portfolios = {
        "Port:Safe":     {"Arb Scanner": 0.40, "Ensemble": 0.60},
        "Port:Balanced": {"CEX Lag": 0.40, "Ensemble": 0.35, "Alpha": 0.25},
        "Port:Aggro":    {"CEX Lag": 0.60, "Ensemble": 0.25, "Alpha": 0.15},
    }
    
    end_time = time.time() + minutes * 60
    
    log.info("=" * 75)
    log.info(f"  ALL STRATEGIES LIVE — {minutes}min, every {interval}s")
    log.info(f"  8 runners: 5 strategies + 3 portfolios × $1,000 each")
    log.info("=" * 75)
    
    while time.time() < end_time:
        scan_count += 1
        remaining = (end_time - time.time()) / 60
        log.info(f"\n{'─'*75}")
        log.info(f"  Scan #{scan_count} | {remaining:.1f}min left | {datetime.now().strftime('%H:%M:%S')}")
        log.info(f"{'─'*75}")
        
        try:
            # Update CEX prices
            await cex.update()
            btc_move = cex.get_move("BTC")
            btc_price = cex.prices.get("BTC", 0)
            if btc_price:
                log.info(f"  BTC: ${btc_price:,.0f} ({btc_move:+.3%})")
            
            # Fetch Polymarket
            markets = await fetch_markets()
            log.info(f"  Markets: {len(markets)} tradeable")
            
            # ── Process exits for ALL runners ──
            for rname, runner in runners.items():
                for tid in list(runner.positions.keys()):
                    pos = runner.positions[tid]
                    current = price_cache.get(tid, pos["entry"])
                    for m in markets:
                        if m["token"] == tid:
                            current = m["price"]
                            price_cache[tid] = current
                            break
                    
                    if pos["side"] == "buy":
                        ur = (current - pos["entry"]) * pos["shares"]
                    else:
                        ur = (pos["entry"] - current) * pos["shares"]
                    
                    pos["hold"] = pos.get("hold", 0) + 1
                    
                    if ur < -pos["usd"] * 0.015 or pos["hold"] >= 3 or ur > pos["usd"] * 0.025:
                        pnl = ur - pos["usd"] * 0.002
                        runner.capital += pos["usd"] + pnl
                        runner.closed.append({"pnl": pnl})
                        del runner.positions[tid]
            
            # ── Generate signals ──
            
            # 1. CEX Lag
            if abs(btc_move) >= 0.003:
                sig = cex_lag_signal(btc_move, runners["CEX Lag"].capital)
                if sig and len(runners["CEX Lag"].positions) < 3:
                    # Use a pseudo token for tracking
                    ptk = f"cex_{int(time.time())}"
                    entry = 0.50 + 0.01  # Simulated entry during lag
                    size = sig["size_usd"]
                    runners["CEX Lag"].capital -= size
                    runners["CEX Lag"].positions[ptk] = {
                        "side": sig["side"], "entry": entry, "usd": size,
                        "shares": (size * 0.998) / entry, "hold": 0,
                    }
                    runners["CEX Lag"].signals_total += 1
                    log.info(f"  ⚡ CEX LAG: {sig['side'].upper()} ${size:.0f} on {sig['source']}")
                    
                    # Also for portfolio runners
                    for pname, alloc in portfolios.items():
                        if "CEX Lag" in alloc:
                            prunner = runners[pname]
                            psig = cex_lag_signal(btc_move, prunner.capital * alloc["CEX Lag"])
                            if psig and len(prunner.positions) < 5:
                                pk = f"cex_{pname}_{int(time.time())}"
                                sz = psig["size_usd"]
                                prunner.capital -= sz
                                prunner.positions[pk] = {
                                    "side": psig["side"], "entry": entry, "usd": sz,
                                    "shares": (sz * 0.998) / entry, "hold": 0,
                                }
                                prunner.signals_total += 1
            
            # 2. Arb Scanner
            for m in markets:
                if m["total"] < 0.995:
                    sig = arb_signal(m["total"], runners["Arb Scanner"].capital)
                    if sig and m["token"] not in runners["Arb Scanner"].positions:
                        sz = sig["size_usd"]
                        runners["Arb Scanner"].capital -= sz
                        runners["Arb Scanner"].positions[m["token"]] = {
                            "side": "buy", "entry": m["price"], "usd": sz,
                            "shares": (sz * 0.998) / max(m["price"], 0.01), "hold": 0,
                        }
                        runners["Arb Scanner"].signals_total += 1
                        log.info(f"  🔍 ARB: gap={1-m['total']:.3f} on {m['q'][:40]}")
            
            # 3. Signal-based strategies (Ensemble, Cash-Like, Alpha)
            for m in markets[:15]:  # Top 15 by liquidity
                if m["price"] <= 0.03 or m["price"] >= 0.97:
                    continue
                
                prev = price_cache.get(m["token"], m["price"])
                price_cache[m["token"]] = m["price"]
                
                book = await fetch_book(m["token"])
                model_prob = compute_model_prob(m["price"], prev, book, m["total"])
                
                for sname, sfn in strategy_fns.items():
                    runner = runners[sname]
                    if m["token"] in runner.positions or len(runner.positions) >= 5:
                        continue
                    
                    sig = sfn(model_prob, m["price"], runner.capital, prev)
                    if sig and sig.get("size_usd", 0) > 5:
                        sz = min(sig["size_usd"], runner.capital * 0.15)
                        entry = m["price"] + 0.001 if sig["side"] == "buy" else m["price"] - 0.001
                        runner.capital -= sz
                        runner.positions[m["token"]] = {
                            "side": sig["side"], "entry": entry, "usd": sz,
                            "shares": (sz * 0.998) / max(entry, 0.01), "hold": 0,
                        }
                        runner.signals_total += 1
                
                # Portfolio runners (signal-based portion)
                for pname, alloc in portfolios.items():
                    prunner = runners[pname]
                    if m["token"] in prunner.positions or len(prunner.positions) >= 5:
                        continue
                    for sname, pct in alloc.items():
                        if sname in ("CEX Lag", "Arb Scanner"):
                            continue
                        sfn = strategy_fns.get(sname)
                        if not sfn:
                            continue
                        sig = sfn(model_prob, m["price"], prunner.capital * pct, prev)
                        if sig and sig.get("size_usd", 0) > 3:
                            sz = min(sig["size_usd"], prunner.capital * 0.10)
                            entry = m["price"] + 0.001 if sig["side"] == "buy" else m["price"] - 0.001
                            prunner.capital -= sz
                            prunner.positions[m["token"]] = {
                                "side": sig["side"], "entry": entry, "usd": sz,
                                "shares": (sz * 0.998) / max(entry, 0.01), "hold": 0,
                            }
                            prunner.signals_total += 1
                            break
                
                await asyncio.sleep(0.03)
            
            # ── Scoreboard ──
            log.info(f"\n  {'Strategy':<16} {'Equity':>10} {'Return':>8} {'Open':>5} {'Closed':>7} {'Wins':>5} {'PnL':>9}")
            log.info(f"  {'─'*65}")
            for rname, runner in runners.items():
                w = sum(1 for t in runner.closed if t["pnl"] > 0)
                log.info(
                    f"  {rname:<16} ${runner.equity:>8.2f} {runner.ret:>+7.2%} "
                    f"{len(runner.positions):>5} {len(runner.closed):>7} {w:>5} ${runner.pnl:>+8.2f}"
                )
        
        except Exception as e:
            log.error(f"  Error: {e}")
            import traceback; traceback.print_exc()
        
        await asyncio.sleep(interval)
    
    # ── Final Report ──
    log.info("\n" + "=" * 75)
    log.info("  SIMULATION COMPLETE")
    log.info("=" * 75)
    log.info(f"  Duration: {minutes}min | Scans: {scan_count}")
    log.info(f"\n  {'Strategy':<16} {'Final $':>10} {'Return':>8} {'Trades':>7} {'Wins':>5} {'WR':>6} {'PnL':>9}")
    log.info(f"  {'═'*65}")
    
    sorted_runners = sorted(runners.items(), key=lambda x: x[1].ret, reverse=True)
    for rname, r in sorted_runners:
        w = sum(1 for t in r.closed if t["pnl"] > 0)
        wr = w / len(r.closed) * 100 if r.closed else 0
        medal = "🥇" if rname == sorted_runners[0][0] else "🥈" if rname == sorted_runners[1][0] else "🥉" if rname == sorted_runners[2][0] else "  "
        log.info(
            f"{medal}{rname:<16} ${r.equity:>8.2f} {r.ret:>+7.2%} "
            f"{len(r.closed):>7} {w:>5} {wr:>5.0f}% ${r.pnl:>+8.2f}"
        )
    
    log.info(f"\n  Best: {sorted_runners[0][0]} ({sorted_runners[0][1].ret:+.2%})")
    log.info(f"  Worst: {sorted_runners[-1][0]} ({sorted_runners[-1][1].ret:+.2%})")
    log.info("=" * 75)


if __name__ == "__main__":
    minutes = 20
    interval = 30
    for i, arg in enumerate(sys.argv):
        if arg == "--minutes" and i + 1 < len(sys.argv):
            minutes = int(sys.argv[i + 1])
        elif arg == "--interval" and i + 1 < len(sys.argv):
            interval = int(sys.argv[i + 1])
    asyncio.run(run_all(minutes, interval))
