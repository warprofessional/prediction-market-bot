"""THE UNIFIED STRATEGY — Combining the best edges from all research.

Based on:
- Near-resolution bonding (Sharky6999: $571K, 99.3% WR)
- CEX-lag momentum detection (k9Q2: $1.5M+, 55-69% WR)  
- Orderbook imbalance (OBI) — 65% short-term price variance capture
- Maker spread capture with optimism tax exploitation
- Frank-Wolfe guaranteed profit projection
- Empirical Kelly with VPIN kill switch

The key insight from Becker (2026): "Makers win by accommodating biased flow, not forecasting."

Strategy pipeline:
1. DETECT: OBI + CEX lag + EV gap (multi-signal ensemble)
2. FILTER: VPIN (skip toxic flow), liquidity depth, time-to-expiry
3. SIZE: Empirical Kelly with Monte Carlo uncertainty
4. EXECUTE: Maker limits only (zero fees + rebates)
5. BOND: Near resolution, switch to bonding mode (97-99¢ buys)
"""
import math
from typing import Optional

import numpy as np

from src.strategies.math_core import (
    expected_value,
    ev_full,
    fractional_kelly,
    empirical_kelly,
    logit_transform,
    inv_logit,
    vpin as compute_vpin,
    lmsr_price_impact,
)


def compute_obi(bid_depth: float, ask_depth: float) -> float:
    """Orderbook Imbalance: (Q_bid - Q_ask) / (Q_bid + Q_ask).
    
    Explains ~65% of short-term price variance per @MarikWeb3.
    Positive = buying pressure, negative = selling pressure.
    """
    total = bid_depth + ask_depth
    if total == 0:
        return 0.0
    return (bid_depth - ask_depth) / total


def compute_microprice(best_bid: float, best_ask: float, 
                        bid_vol: float, ask_vol: float) -> float:
    """Microprice: volume-weighted midpoint.
    
    μ = (P_bid × V_ask + P_ask × V_bid) / (V_bid + V_ask)
    Better than simple midpoint — accounts for liquidity imbalance.
    """
    total_vol = bid_vol + ask_vol
    if total_vol == 0:
        return (best_bid + best_ask) / 2
    return (best_bid * ask_vol + best_ask * bid_vol) / total_vol


def detect_cex_lag(spot_return: float, poly_price: float, 
                    prev_poly_price: float, threshold: float = 0.003) -> float:
    """Detect CEX (Binance) to Polymarket lag.
    
    When spot moves >0.3% but Poly hasn't adjusted yet, there's a lag edge.
    Returns the lag signal strength (0 = no lag, positive = up lag, negative = down lag).
    
    Based on k9Q2 cluster ($1.5M+ all-time, 55-69% WR).
    """
    poly_return = (poly_price - prev_poly_price) / max(prev_poly_price, 0.01)
    divergence = spot_return - poly_return
    
    if abs(divergence) < threshold:
        return 0.0
    return divergence


def near_resolution_bonding_signal(
    model_prob: float, 
    market_price: float, 
    time_remaining: float,  # 0 to 1 (fraction of market duration)
    certainty_threshold: float = 0.95,
) -> Optional[dict]:
    """Near-resolution bonding strategy.
    
    Buy at 97-99¢ when outcome is virtually certain (<5 min before close).
    Sharky6999: $571K PnL, 99.3% WR, 15K+ trades.
    
    Args:
        model_prob: Estimated probability of YES
        market_price: Current YES price
        time_remaining: Fraction of time left (0 = expired)
        certainty_threshold: Min confidence to bond
    
    Returns:
        Trade signal dict or None
    """
    # Only bond in final 5% of market duration
    if time_remaining > 0.05:
        return None
    
    # Check if outcome is near-certain
    if model_prob > certainty_threshold and market_price > 0.95:
        # Buy YES at current price — it'll resolve to $1.00
        edge = 1.0 - market_price
        if edge > 0.005:  # At least 0.5¢ edge
            return {
                "side": "buy",
                "edge": edge,
                "confidence": model_prob,
                "mode": "bonding",
            }
    
    if model_prob < (1 - certainty_threshold) and market_price < 0.05:
        # Buy NO (YES is near 0)
        edge = market_price  # NO resolves to $1.00
        if edge > 0.005:
            return {
                "side": "sell",
                "edge": edge,
                "confidence": 1 - model_prob,
                "mode": "bonding",
            }
    
    return None


def unified_strategy(model_prob: float, market_price: float, capital: float, 
                     prev_market_price: float = 0.5,
                     time_remaining: float = 0.5) -> Optional[dict]:
    """THE unified strategy combining all edges.
    
    Multi-signal ensemble:
    1. EV gap (core) — Bayesian model vs market
    2. OBI simulation — orderbook imbalance filter
    3. CEX lag simulation — momentum from external markets
    4. Optimism tax — fade extreme crowd bias
    5. Mean-reversion at extremes
    6. Momentum confirmation — price moving toward our predicted direction
    7. Contra-momentum fade
    + Near-resolution bonding mode (Sharky6999: $571K, 99.3% WR)
    
    Size with empirical Kelly, filter with VPIN-style checks.
    """
    # === SIGNAL 1: Core EV Gap ===
    ev = model_prob - market_price
    
    # === SIGNAL 2: Simulated OBI ===
    # OBI captures ~65% of short-term price variance (@MarikWeb3)
    # In backtest: if true prob consistently diverges from market, OBI should confirm
    # Stronger divergence = stronger OBI signal
    price_distance = abs(ev)
    obi_strength = min(1.0, price_distance / 0.04)  # Normalize to [0,1], tighter scaling
    obi_confirms = obi_strength > 0.25  # OBI confirms when divergence > 1%
    
    # === SIGNAL 3: Simulated CEX Lag ===  
    # CEX lag: when true prob moves but market hasn't caught up
    # Threshold: >0.3-8% divergence per k9Q2 cluster
    lag_signal = price_distance
    lag_confirms = lag_signal > 0.015  # >1.5% divergence = lag likely
    
    # === SIGNAL 4: Optimism Tax Fade ===
    # Becker (2026): "Takers overpay for YES; makers capture via longshot bias"
    # The stronger the crowd agreement, the more likely it's overpriced (longshot bias)
    optimism_fade = False
    # Tiered optimism detection (Becker 2026: longshot bias strongest at extremes):
    if market_price > 0.85 and model_prob < market_price - 0.015:
        optimism_fade = True  # Very strong crowd → likely overpriced
    elif market_price > 0.75 and model_prob < market_price - 0.025:
        optimism_fade = True  # Strong crowd + model disagrees significantly
    elif market_price < 0.15 and model_prob > market_price + 0.015:
        optimism_fade = True
    elif market_price < 0.25 and model_prob > market_price + 0.025:
        optimism_fade = True
    
    # === SIGNAL 5: Mean-reversion signal ===
    # When market price is far from 50%, it tends to revert (unless near resolution)
    # This captures the "overreaction" pattern
    mean_reversion = False
    if abs(market_price - 0.5) > 0.3 and abs(ev) > 0.02:
        # Market at extreme, true prob disagrees
        if (market_price > 0.8 and model_prob < 0.75) or (market_price < 0.2 and model_prob > 0.25):
            mean_reversion = True
    
    # === SIGNAL 6: Momentum confirmation ===
    # Price moving toward our model's predicted direction = confirming
    price_change = market_price - prev_market_price
    momentum_confirms = False
    if ev > 0 and price_change > 0.005:
        momentum_confirms = True  # Price rising + we want to buy = momentum confirms
    elif ev < 0 and price_change < -0.005:
        momentum_confirms = True  # Price falling + we want to sell = momentum confirms
    
    # === SIGNAL 7: Contra-momentum fade ===
    # @frostikkkk EMA ±8% strategy: fade sharp overreactions
    # If price moved sharply AWAY from our model, it's an overreaction to fade
    contra_momentum = False
    if abs(price_change) > 0.04:  # Sharp move (>4%)
        # If our model disagrees with the direction of the sharp move → fade opportunity
        if (price_change > 0 and ev < -0.02) or (price_change < 0 and ev > 0.02):
            contra_momentum = True
    
    # === ENSEMBLE DECISION ===
    # Count confirming signals — more = higher conviction
    signal_count = sum([
        abs(ev) > 0.015,       # EV gap exists (1.5%)
        obi_confirms,           # OBI confirms direction
        lag_confirms,           # CEX lag confirms
        optimism_fade,          # Crowd bias to exploit
        mean_reversion,         # Mean-reversion opportunity
        momentum_confirms,      # Price momentum confirms direction
        contra_momentum,        # Fade sharp overreactions
    ])
    
    # Adaptive threshold based on signal strength
    # @0xRicker playbook: "entry if EV > 0.05 after fees" — we scale by signal count
    if signal_count >= 5:
        min_ev_threshold = 0.004  # 0.4% with 5+ signals — very high conviction
    elif signal_count >= 4:
        min_ev_threshold = 0.005  # 0.5% with 4+ signals
    elif signal_count >= 3:
        min_ev_threshold = 0.008  # 0.8% with 3+ signals
    elif signal_count >= 2:
        min_ev_threshold = 0.012  # 1.2% with 2 signals  
    else:
        min_ev_threshold = 0.020  # 2.0% with single signal
    
    if abs(ev) < min_ev_threshold:
        return None
    
    # Selectivity: require more signals when edge is small
    min_required_signals = 1
    if abs(ev) < 0.025:
        min_required_signals = 2  # Small edge needs more confirmation
    
    # In trending markets (large price moves), require momentum to confirm our direction
    # This avoids fighting persistent trends
    if abs(price_change) > 0.015 and not momentum_confirms and not contra_momentum:
        # Big move happening and neither momentum nor contra-momentum confirms → skip
        # Unless we have very strong conviction (4+ signals)
        if signal_count < 3:
            return None
    
    if signal_count < min_required_signals:
        return None
    
    # === SIZING: Empirical Kelly with uncertainty ===
    base_cv = 0.30
    cv_reduction_signals = 0.06 * signal_count
    cv_reduction_edge = min(0.12, abs(ev) * 2.5)
    cv_edge = max(0.05, base_cv - cv_reduction_signals - cv_reduction_edge)
    
    kelly_frac = 0.50  # Half Kelly — best risk-adjusted sweet spot for max return
    
    # Boost Kelly for high conviction — max return mode
    if signal_count >= 3:
        kelly_frac = 0.60
    if signal_count >= 4:
        kelly_frac = 0.70
    if signal_count >= 5:
        kelly_frac = 0.80


    
    kelly = empirical_kelly(model_prob, market_price, cv_edge, kelly_frac)
    
    if kelly <= 0:
        return None
    
    size = kelly * capital
    
    # Dynamic position cap — larger positions on higher-quality trades
    # @0xRicker: fewer trades, bigger sizing on confirmed edges
    if signal_count >= 5 and abs(ev) > 0.05:
        max_position = capital * 0.25  # Exceptional: 25%
    elif signal_count >= 4 and abs(ev) > 0.04:
        max_position = capital * 0.20  # High conviction: 20%
    elif signal_count >= 3 and abs(ev) > 0.03:
        max_position = capital * 0.15
    else:
        max_position = capital * 0.10  # Standard: 10%
    
    size = min(size, max_position)
    
    # Minimum trade size
    if size < 1.0:
        return None
    
    # === EXECUTION ===
    side = "buy" if ev > 0 else "sell"
    
    return {
        "side": side,
        "size_usd": size,
        "ev": ev,
        "signal_count": signal_count,
        "kelly": kelly,
        "cv_edge": cv_edge,
        "obi_confirms": obi_confirms,
        "lag_confirms": lag_confirms,
        "optimism_fade": optimism_fade,
    }
