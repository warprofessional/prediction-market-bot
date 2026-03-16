"""Strategy presets for different risk appetites.

Conservative → Moderate → Aggressive → YOLO

Each preset tunes:
- Kelly fraction (position sizing aggressiveness)
- Minimum EV threshold (trade selectivity)
- Signal count requirements (entry strictness)
- Position caps (max % per trade)
- Stop loss / take profit levels
"""
from dataclasses import dataclass
from typing import Optional

from src.strategies.unified_strategy import unified_strategy as _base_strategy
from src.strategies.math_core import empirical_kelly


@dataclass
class StrategyPreset:
    """Configuration for a strategy preset."""
    name: str
    description: str
    
    # Kelly sizing
    kelly_base: float          # Base Kelly fraction (0.15-0.75)
    kelly_3sig: float          # Kelly with 3+ signals
    kelly_4sig: float          # Kelly with 4+ signals
    kelly_5sig: float          # Kelly with 5+ signals
    
    # Entry thresholds
    min_ev_1sig: float         # Min EV with 1 signal
    min_ev_2sig: float         # Min EV with 2 signals
    min_ev_3sig: float         # Min EV with 3 signals
    min_ev_4sig: float         # Min EV with 4+ signals
    min_signals_small_edge: int  # Min signals when edge < 2.5%
    
    # Position caps
    cap_standard: float        # Max % of capital per standard trade
    cap_medium: float          # Max % with 3+ signals and 3%+ edge
    cap_high: float            # Max % with 4+ signals and 4%+ edge
    
    # Exit rules
    stop_loss_pct: float       # Stop loss as % of position
    take_profit_pct: float     # Take profit as % of position
    max_hold_steps: int        # Max steps to hold
    
    # Risk limits
    max_portfolio_risk: float  # Max total capital deployed
    max_daily_loss: float      # Max daily loss before stopping


# === THE PRESETS ===

CONSERVATIVE = StrategyPreset(
    name="Conservative",
    description="Capital preservation first. Low drawdown, steady returns. For large accounts or risk-averse traders.",
    kelly_base=0.15,
    kelly_3sig=0.20,
    kelly_4sig=0.25,
    kelly_5sig=0.30,
    min_ev_1sig=0.030,
    min_ev_2sig=0.020,
    min_ev_3sig=0.015,
    min_ev_4sig=0.010,
    min_signals_small_edge=3,   # Very selective on small edges
    cap_standard=0.05,
    cap_medium=0.08,
    cap_high=0.10,
    stop_loss_pct=0.010,        # 1% stop loss
    take_profit_pct=0.015,      # 1.5% take profit
    max_hold_steps=2,
    max_portfolio_risk=0.30,
    max_daily_loss=0.03,
)

MODERATE = StrategyPreset(
    name="Moderate",
    description="Balanced risk/return. The default preset. Good for most traders.",
    kelly_base=0.25,
    kelly_3sig=0.33,
    kelly_4sig=0.40,
    kelly_5sig=0.50,
    min_ev_1sig=0.020,
    min_ev_2sig=0.012,
    min_ev_3sig=0.008,
    min_ev_4sig=0.006,
    min_signals_small_edge=2,   # Standard selectivity
    cap_standard=0.08,
    cap_medium=0.12,
    cap_high=0.15,
    stop_loss_pct=0.015,        # 1.5% stop loss
    take_profit_pct=0.025,      # 2.5% take profit
    max_hold_steps=2,
    max_portfolio_risk=0.50,
    max_daily_loss=0.05,
)

AGGRESSIVE = StrategyPreset(
    name="Aggressive",
    description="Higher returns, higher risk. Larger positions, lower entry bars. For experienced traders.",
    kelly_base=0.35,
    kelly_3sig=0.45,
    kelly_4sig=0.55,
    kelly_5sig=0.65,
    min_ev_1sig=0.015,
    min_ev_2sig=0.008,
    min_ev_3sig=0.005,
    min_ev_4sig=0.004,
    min_signals_small_edge=1,   # Less selective
    cap_standard=0.12,
    cap_medium=0.18,
    cap_high=0.22,
    stop_loss_pct=0.020,        # 2% stop loss (wider)
    take_profit_pct=0.035,      # 3.5% take profit
    max_hold_steps=3,
    max_portfolio_risk=0.70,
    max_daily_loss=0.08,
)

YOLO = StrategyPreset(
    name="YOLO",
    description="Maximum returns, maximum risk. For small accounts you can afford to lose. Not financial advice.",
    kelly_base=0.50,
    kelly_3sig=0.65,
    kelly_4sig=0.75,
    kelly_5sig=0.85,
    min_ev_1sig=0.010,
    min_ev_2sig=0.006,
    min_ev_3sig=0.004,
    min_ev_4sig=0.003,
    min_signals_small_edge=1,   # Trade everything with any signal
    cap_standard=0.18,
    cap_medium=0.25,
    cap_high=0.30,
    stop_loss_pct=0.030,        # 3% stop loss (very wide)
    take_profit_pct=0.050,      # 5% take profit
    max_hold_steps=4,
    max_portfolio_risk=0.90,
    max_daily_loss=0.15,
)

ALL_PRESETS = {
    "conservative": CONSERVATIVE,
    "moderate": MODERATE,
    "aggressive": AGGRESSIVE,
    "yolo": YOLO,
}


def make_strategy_from_preset(preset: StrategyPreset):
    """Create a strategy function configured with the given preset.
    
    Returns a callable with signature (model_prob, market_price, capital, prev_market_price) -> dict|None
    """
    p = preset  # shorthand
    
    def strategy(model_prob: float, market_price: float, capital: float,
                 prev_market_price: float = 0.5) -> Optional[dict]:
        ev = model_prob - market_price
        price_distance = abs(ev)
        
        # OBI
        obi_strength = min(1.0, price_distance / 0.04)
        obi_confirms = obi_strength > 0.25
        
        # CEX lag
        lag_confirms = price_distance > 0.015
        
        # Optimism fade (tiered)
        optimism_fade = False
        if market_price > 0.85 and model_prob < market_price - 0.015:
            optimism_fade = True
        elif market_price > 0.75 and model_prob < market_price - 0.025:
            optimism_fade = True
        elif market_price < 0.15 and model_prob > market_price + 0.015:
            optimism_fade = True
        elif market_price < 0.25 and model_prob > market_price + 0.025:
            optimism_fade = True
        
        # Mean reversion
        mean_reversion = False
        if abs(market_price - 0.5) > 0.3 and abs(ev) > 0.02:
            if (market_price > 0.8 and model_prob < 0.75) or (market_price < 0.2 and model_prob > 0.25):
                mean_reversion = True
        
        # Momentum
        price_change = market_price - prev_market_price
        momentum_confirms = False
        if ev > 0 and price_change > 0.005:
            momentum_confirms = True
        elif ev < 0 and price_change < -0.005:
            momentum_confirms = True
        
        # Contra-momentum
        contra_momentum = False
        if abs(price_change) > 0.04:
            if (price_change > 0 and ev < -0.02) or (price_change < 0 and ev > 0.02):
                contra_momentum = True
        
        # Signal count
        signal_count = sum([
            abs(ev) > 0.015,
            obi_confirms,
            lag_confirms,
            optimism_fade,
            mean_reversion,
            momentum_confirms,
            contra_momentum,
        ])
        
        # Adaptive EV threshold from preset
        if signal_count >= 4:
            min_ev = p.min_ev_4sig
        elif signal_count >= 3:
            min_ev = p.min_ev_3sig
        elif signal_count >= 2:
            min_ev = p.min_ev_2sig
        else:
            min_ev = p.min_ev_1sig
        
        if abs(ev) < min_ev:
            return None
        
        # Selectivity
        if abs(ev) < 0.025 and signal_count < p.min_signals_small_edge:
            return None
        
        if signal_count < 1:
            return None
        
        # Kelly sizing from preset
        base_cv = 0.35
        cv_reduction = 0.06 * signal_count + min(0.10, abs(ev) * 2)
        cv_edge = max(0.05, base_cv - cv_reduction)
        
        kelly_frac = p.kelly_base
        if signal_count >= 3:
            kelly_frac = p.kelly_3sig
        if signal_count >= 4:
            kelly_frac = p.kelly_4sig
        if signal_count >= 5:
            kelly_frac = p.kelly_5sig
        
        kelly = empirical_kelly(model_prob, market_price, cv_edge, kelly_frac)
        if kelly <= 0:
            return None
        
        size = kelly * capital
        
        # Position cap from preset
        if signal_count >= 4 and abs(ev) > 0.04:
            max_pos = capital * p.cap_high
        elif signal_count >= 3 and abs(ev) > 0.03:
            max_pos = capital * p.cap_medium
        else:
            max_pos = capital * p.cap_standard
        
        size = min(size, max_pos)
        if size < 1.0:
            return None
        
        side = "buy" if ev > 0 else "sell"
        return {"side": side, "size_usd": size}
    
    return strategy
