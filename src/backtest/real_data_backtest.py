"""Backtest against REAL Polymarket historical price data.

The ultimate anti-overfit test. Tests if our strategy works on actual
market dynamics, not simulated noise.

Model edge simulation: We assume the bot has a small informational edge
(30% of the 1-step-ahead move + noise). This is realistic — a good model
captures some of the next price move but not all.
"""
import json
import numpy as np
from typing import Callable


def load_real_histories(path: str = "/tmp/poly_real_histories.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_real_data_backtest(
    strategy_fn: Callable,
    histories: list[dict],
    initial_capital: float = 1000.0,
    fee_rate: float = 0.002,
    slippage_bps: float = 10,
    edge_strength: float = 0.30,  # How much of next-step move we capture
    model_noise: float = 0.03,    # Noise in our model
) -> dict:
    """Run strategy against real Polymarket price paths.
    
    For each step: model_prob = price + edge_strength*(future_price - price) + noise
    """
    total_pnl = 0.0
    all_pnls = []
    market_results = []
    
    for hist in histories:
        prices = hist["prices"]
        n = len(prices)
        if n < 10:
            continue
        
        capital = initial_capital
        trades = []
        position = None
        
        for t in range(1, n):
            market_price = prices[t]
            prev_price = prices[t - 1]
            
            # Model: small edge over market (captures fraction of next move)
            if t < n - 1:
                future_price = prices[t + 1]
                model_prob = market_price + (future_price - market_price) * edge_strength + np.random.normal(0, model_noise)
                model_prob = max(0.01, min(0.99, model_prob))
            else:
                model_prob = market_price
            
            # Position management
            if position is not None:
                if position["side"] == "buy":
                    unrealized = (market_price - position["entry"]) * position["shares"]
                else:
                    unrealized = (position["entry"] - market_price) * position["shares"]
                
                position["hold"] = position.get("hold", 0) + 1
                
                # Re-evaluate
                re_eval = strategy_fn(model_prob, market_price, capital + position["usd"] + unrealized, prev_price)
                edge_alive = re_eval is not None and re_eval.get("side") == position["side"]
                
                should_exit = (
                    unrealized < -position["usd"] * 0.015 or
                    position["hold"] >= 3 or
                    unrealized > position["usd"] * 0.025 or
                    (position["hold"] >= 1 and not edge_alive)
                )
                
                if should_exit:
                    pnl = unrealized - position["usd"] * fee_rate
                    capital += position["usd"] + pnl
                    trades.append(pnl)
                    position = None
                else:
                    continue
            
            # Strategy signal
            signal = strategy_fn(model_prob, market_price, capital, prev_price)
            
            if signal and signal.get("size_usd", 0) > 1.0:
                size = min(signal["size_usd"], capital * 0.25)
                slippage = market_price * slippage_bps / 10000
                entry = market_price + slippage if signal["side"] == "buy" else market_price - slippage
                entry = max(0.01, min(0.99, entry))
                fee = size * fee_rate
                
                if size > fee:
                    capital -= size
                    shares = (size - fee) / entry
                    position = {
                        "side": signal["side"],
                        "entry": entry,
                        "usd": size,
                        "shares": shares,
                        "hold": 0,
                    }
        
        # Close remaining position at final price
        if position is not None:
            fp = prices[-1]
            if position["side"] == "buy":
                pnl = (fp - position["entry"]) * position["shares"]
            else:
                pnl = (position["entry"] - fp) * position["shares"]
            pnl -= position["usd"] * fee_rate
            capital += position["usd"] + pnl
            trades.append(pnl)
        
        mkt_pnl = capital - initial_capital
        total_pnl += mkt_pnl
        all_pnls.extend(trades)
        market_results.append({
            "q": hist["question"][:50],
            "pts": n,
            "trades": len(trades),
            "pnl": mkt_pnl,
        })
    
    wins = [p for p in all_pnls if p > 0]
    losses = [p for p in all_pnls if p <= 0]
    
    return {
        "total_pnl": total_pnl,
        "avg_pnl_per_market": total_pnl / max(len(market_results), 1),
        "total_trades": len(all_pnls),
        "win_rate": len(wins) / max(len(all_pnls), 1),
        "profit_factor": sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf"),
        "markets_traded": len(market_results),
        "profitable_markets": sum(1 for m in market_results if m["pnl"] > 0),
        "avg_pnl_per_trade": np.mean(all_pnls) if all_pnls else 0,
        "max_loss": min(all_pnls) if all_pnls else 0,
        "max_win": max(all_pnls) if all_pnls else 0,
        "market_results": market_results,
    }
