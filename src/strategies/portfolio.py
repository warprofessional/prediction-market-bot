"""Portfolio of strategies for different investment horizons.

Instead of one strategy, allocate capital across multiple strategies
that are optimized for different conditions:

1. CASH-LIKE (Weekly): Ultra-conservative, designed for "I might need this money next week"
   - Smallest positions, highest selectivity, fastest exits
   - Target: Never have a losing week

2. GROWTH (Monthly): Balanced, designed for steady month-over-month growth
   - Moderate positions, good selectivity
   - Target: Never have a losing month, maximize monthly Sharpe

3. ALPHA (Quarterly+): Aggressive, designed for maximum long-term compounding
   - Larger positions, lower selectivity, let winners run
   - Target: Maximum total return over 3+ months
"""
from typing import Optional
from src.strategies.math_core import empirical_kelly


def _base_strategy(model_prob, market_price, capital, prev_market_price,
                   kelly_base, kelly_3, kelly_4, kelly_5,
                   ev_thresholds, min_sigs_small, base_cv,
                   caps):
    """Shared signal logic with configurable sizing."""
    ev = model_prob - market_price
    price_distance = abs(ev)
    
    # Signals (same 7 for all strategies — only sizing/filtering differs)
    obi_confirms = min(1.0, price_distance / 0.04) > 0.25
    lag_confirms = price_distance > 0.015
    
    optimism_fade = False
    if market_price > 0.85 and model_prob < market_price - 0.015:
        optimism_fade = True
    elif market_price > 0.75 and model_prob < market_price - 0.025:
        optimism_fade = True
    elif market_price < 0.15 and model_prob > market_price + 0.015:
        optimism_fade = True
    elif market_price < 0.25 and model_prob > market_price + 0.025:
        optimism_fade = True
    
    mean_reversion = False
    if abs(market_price - 0.5) > 0.3 and abs(ev) > 0.02:
        if (market_price > 0.8 and model_prob < 0.75) or (market_price < 0.2 and model_prob > 0.25):
            mean_reversion = True
    
    price_change = market_price - prev_market_price
    momentum_confirms = (ev > 0 and price_change > 0.005) or (ev < 0 and price_change < -0.005)
    
    contra_momentum = False
    if abs(price_change) > 0.04:
        if (price_change > 0 and ev < -0.02) or (price_change < 0 and ev > 0.02):
            contra_momentum = True
    
    signal_count = sum([abs(ev) > 0.015, obi_confirms, lag_confirms,
                        optimism_fade, mean_reversion, momentum_confirms, contra_momentum])
    
    # Trend filter
    if abs(price_change) > 0.015 and not momentum_confirms and not contra_momentum:
        if signal_count < 3:
            return None
    
    # EV threshold by signal count
    thresholds = ev_thresholds
    if signal_count >= 5:
        min_ev = thresholds[0]
    elif signal_count >= 4:
        min_ev = thresholds[1]
    elif signal_count >= 3:
        min_ev = thresholds[2]
    elif signal_count >= 2:
        min_ev = thresholds[3]
    else:
        min_ev = thresholds[4]
    
    if abs(ev) < min_ev:
        return None
    
    if abs(ev) < 0.025 and signal_count < min_sigs_small:
        return None
    if signal_count < 1:
        return None
    
    # Kelly sizing
    cv = max(0.05, base_cv - 0.06 * signal_count - min(0.12, abs(ev) * 2.5))
    kf = kelly_base
    if signal_count >= 3: kf = kelly_3
    if signal_count >= 4: kf = kelly_4
    if signal_count >= 5: kf = kelly_5
    
    kelly = empirical_kelly(model_prob, market_price, cv, kf)
    if kelly <= 0:
        return None
    
    size = kelly * capital
    if signal_count >= 4 and abs(ev) > 0.04:
        size = min(size, capital * caps[0])
    elif signal_count >= 3 and abs(ev) > 0.03:
        size = min(size, capital * caps[1])
    else:
        size = min(size, capital * caps[2])
    
    if size < 1.0:
        return None
    
    return {"side": "buy" if ev > 0 else "sell", "size_usd": size, "signal_count": signal_count}


def cash_like_strategy(model_prob, market_price, capital, prev_market_price=0.5):
    """CASH-LIKE: Never lose a week. Ultra-conservative."""
    return _base_strategy(
        model_prob, market_price, capital, prev_market_price,
        kelly_base=0.20, kelly_3=0.25, kelly_4=0.30, kelly_5=0.35,
        ev_thresholds=[0.008, 0.010, 0.015, 0.020, 0.035],  # Higher bars
        min_sigs_small=3,  # Very selective on small edges
        base_cv=0.40,  # More uncertain → smaller positions
        caps=[0.08, 0.06, 0.04],  # Tiny positions
    )


def growth_strategy(model_prob, market_price, capital, prev_market_price=0.5):
    """GROWTH: Steady monthly returns. Balanced."""
    return _base_strategy(
        model_prob, market_price, capital, prev_market_price,
        kelly_base=0.40, kelly_3=0.50, kelly_4=0.60, kelly_5=0.70,
        ev_thresholds=[0.005, 0.007, 0.010, 0.015, 0.025],
        min_sigs_small=2,
        base_cv=0.30,
        caps=[0.18, 0.14, 0.10],
    )


def alpha_strategy(model_prob, market_price, capital, prev_market_price=0.5):
    """ALPHA: Maximum long-term compounding. Aggressive."""
    return _base_strategy(
        model_prob, market_price, capital, prev_market_price,
        kelly_base=0.60, kelly_3=0.70, kelly_4=0.80, kelly_5=0.90,
        ev_thresholds=[0.003, 0.004, 0.006, 0.010, 0.018],  # Lower bars
        min_sigs_small=1,  # Trade more
        base_cv=0.25,  # More confident → bigger positions
        caps=[0.25, 0.20, 0.14],  # Large positions
    )


def portfolio_strategy(model_prob, market_price, capital, prev_market_price=0.5):
    """PORTFOLIO: Split capital 30% cash-like, 40% growth, 30% alpha.
    
    Each sub-strategy votes. If all three agree on direction, take the
    growth-strategy's sizing. If 2/3 agree, take smaller size. If only 1
    agrees, skip.
    """
    c = cash_like_strategy(model_prob, market_price, capital * 0.30, prev_market_price)
    g = growth_strategy(model_prob, market_price, capital * 0.40, prev_market_price)
    a = alpha_strategy(model_prob, market_price, capital * 0.30, prev_market_price)
    
    votes = []
    if c: votes.append(c)
    if g: votes.append(g)
    if a: votes.append(a)
    
    if len(votes) == 0:
        return None
    
    # All must agree on direction
    sides = set(v["side"] for v in votes)
    if len(sides) > 1:
        return None  # Conflicting signals → skip
    
    side = sides.pop()
    total_size = sum(v["size_usd"] for v in votes)
    
    if total_size < 1.0:
        return None
    
    return {"side": side, "size_usd": total_size, "signal_count": max(v.get("signal_count", 1) for v in votes)}
