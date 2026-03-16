"""Live paper trading harness.

Connects to real Polymarket API, runs the unified strategy in DRY_RUN mode,
and logs what it WOULD trade. No real money at risk.

Usage:
    python3 live_paper_trade.py              # Run once (scan + signal)
    python3 live_paper_trade.py --loop 60    # Loop every 60 seconds
"""
import asyncio
import json
import logging
import sys
import time
from datetime import datetime

import httpx

from src.strategies.unified_strategy import unified_strategy
from src.strategies.portfolio import cash_like_strategy, growth_strategy, alpha_strategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("paper_trade")

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"


async def fetch_active_markets(limit=50):
    """Fetch active markets with liquidity."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{GAMMA_URL}/markets", params={
            "limit": limit, "active": True, "order": "liquidity", "ascending": False
        })
        resp.raise_for_status()
        markets = []
        for m in resp.json():
            tokens = json.loads(m.get("clobTokenIds", "[]")) if isinstance(m.get("clobTokenIds"), str) else m.get("clobTokenIds", [])
            prices = json.loads(m.get("outcomePrices", "[]")) if isinstance(m.get("outcomePrices"), str) else m.get("outcomePrices", [])
            if len(tokens) >= 2 and len(prices) >= 2:
                markets.append({
                    "question": m.get("question", ""),
                    "slug": m.get("slug", ""),
                    "yes_token": tokens[0],
                    "no_token": tokens[1],
                    "yes_price": float(prices[0]),
                    "no_price": float(prices[1]),
                    "liquidity": float(m.get("liquidity", 0)),
                    "volume": m.get("volume"),
                })
        return markets


async def fetch_orderbook(token_id):
    """Fetch orderbook for a token."""
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(f"{CLOB_URL}/book", params={"token_id": token_id})
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
        except Exception as e:
            logger.debug(f"Orderbook error for {token_id[:20]}: {e}")
    return None


async def run_paper_scan(capital=1000.0, strategy_name="unified"):
    """Scan all active markets and generate paper trade signals."""
    strategies = {
        "unified": unified_strategy,
        "cash": cash_like_strategy,
        "growth": growth_strategy,
        "alpha": alpha_strategy,
    }
    strategy_fn = strategies.get(strategy_name, unified_strategy)
    
    logger.info(f"{'='*60}")
    logger.info(f"PAPER TRADE SCAN — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Strategy: {strategy_name} | Capital: ${capital:.0f}")
    logger.info(f"{'='*60}")
    
    markets = await fetch_active_markets(limit=50)
    logger.info(f"Fetched {len(markets)} active markets")
    
    signals = []
    for mkt in markets:
        market_price = mkt["yes_price"]
        if market_price <= 0.01 or market_price >= 0.99:
            continue
        
        # Get orderbook for real spread/depth
        book = await fetch_orderbook(mkt["yes_token"])
        
        # Simple model: use market price as base, add small perturbation
        # In production, this would be replaced with a real ML model
        # For paper trading, we just check if the strategy would signal
        # on small deviations from market price
        for model_offset in [-0.03, -0.02, -0.01, 0.01, 0.02, 0.03]:
            model_prob = max(0.01, min(0.99, market_price + model_offset))
            signal = strategy_fn(model_prob, market_price, capital, market_price)
            
            if signal and signal.get("size_usd", 0) > 1:
                spread_info = f"spread={book['spread']:.3f}" if book else "no book"
                signals.append({
                    "market": mkt["question"][:55],
                    "side": signal["side"],
                    "size": signal["size_usd"],
                    "market_price": market_price,
                    "model_prob": model_prob,
                    "ev": model_prob - market_price,
                    "signals": signal.get("signal_count", "?"),
                    "spread": book["spread"] if book else None,
                })
                break  # One signal per market
        
        await asyncio.sleep(0.05)
    
    # Report
    if signals:
        logger.info(f"\n{'SIGNALS FOUND':=^60}")
        total_size = 0
        for s in sorted(signals, key=lambda x: abs(x["ev"]), reverse=True):
            side_emoji = "🟢 BUY " if s["side"] == "buy" else "🔴 SELL"
            spread_str = f"spread={s['spread']:.3f}" if s['spread'] else ""
            logger.info(
                f"  {side_emoji} ${s['size']:>6.0f} | EV={s['ev']:+.1%} | "
                f"price={s['market_price']:.3f} model={s['model_prob']:.3f} | "
                f"sigs={s['signals']} {spread_str} | {s['market']}"
            )
            total_size += s["size"]
        logger.info(f"\n  Total signals: {len(signals)} | Total size: ${total_size:.0f} / ${capital:.0f} capital")
    else:
        logger.info("No signals found — markets are efficient right now")
    
    return signals


async def main():
    loop_interval = None
    strategy = "unified"
    capital = 1000.0
    
    for arg in sys.argv[1:]:
        if arg.startswith("--loop"):
            loop_interval = int(sys.argv[sys.argv.index(arg) + 1])
        elif arg in ("cash", "growth", "alpha", "unified"):
            strategy = arg
        elif arg.startswith("--capital"):
            capital = float(sys.argv[sys.argv.index(arg) + 1])
    
    if loop_interval:
        logger.info(f"Starting paper trading loop (every {loop_interval}s)")
        while True:
            try:
                await run_paper_scan(capital, strategy)
            except Exception as e:
                logger.error(f"Scan error: {e}")
            await asyncio.sleep(loop_interval)
    else:
        await run_paper_scan(capital, strategy)


if __name__ == "__main__":
    asyncio.run(main())
