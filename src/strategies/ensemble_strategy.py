"""Ensemble strategy: combine Conservative + Aggressive signals.

When both agree → highest conviction trade (large position).
When only one fires → moderate position with the signaling preset.
When they disagree → skip (conflicting signals = uncertainty).

This avoids overfitting to a single parameter set.
"""
from typing import Optional

from src.strategies.presets import (
    CONSERVATIVE, AGGRESSIVE, make_strategy_from_preset,
)

_conservative_fn = make_strategy_from_preset(CONSERVATIVE)
_aggressive_fn = make_strategy_from_preset(AGGRESSIVE)


def ensemble_strategy(model_prob: float, market_price: float, capital: float,
                      prev_market_price: float = 0.5) -> Optional[dict]:
    """Ensemble of Conservative + Aggressive presets."""
    
    sig_c = _conservative_fn(model_prob, market_price, capital, prev_market_price)
    sig_a = _aggressive_fn(model_prob, market_price, capital, prev_market_price)
    
    # Both agree → highest conviction
    if sig_c is not None and sig_a is not None and sig_c["side"] == sig_a["side"]:
        # Average the sizes, bias toward conservative
        size = sig_c["size_usd"] * 0.4 + sig_a["size_usd"] * 0.6
        return {"side": sig_c["side"], "size_usd": size}
    
    # Only aggressive fires → moderate position
    if sig_a is not None and sig_c is None:
        return {"side": sig_a["side"], "size_usd": sig_a["size_usd"] * 0.5}
    
    # Only conservative fires → small position (rare but reliable)
    if sig_c is not None and sig_a is None:
        return {"side": sig_c["side"], "size_usd": sig_c["size_usd"]}
    
    # Disagree or both None → skip
    return None
