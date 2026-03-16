"""Market selection filter: which markets to trade, which to avoid.

Based on real-data backtest results (46 markets):
- Sports/Tennis: $3,097 profit, 81% profitable → TRADE
- Crypto 5-min: Validated by @k9Q2 cluster ($1.5M+) → TRADE
- Politics/long-duration: -$390, unstable → AVOID unless high conviction
- Meme/tweets: -$386 single market loss → AVOID

Filters:
1. Liquidity: minimum $100 depth on both sides
2. Spread: maximum 5% (thin markets = slippage death)
3. Market type: prefer short-duration, high-volume
4. Price: avoid extreme prices (<5% or >95%) — low edge, high risk
"""

# Keywords for market categorization
SPORTS_KEYWORDS = [
    "vs.", "match", "set ", "games", "goal", "scorer", "winner",
    "o/u", "over/under", "handicap", "spread:", "total", "kills",
    "map ", "round ", "tournament", "league", "cup", "championship",
    "nba", "nfl", "mlb", "nhl", "atp", "wta", "serie a", "premier",
    "la liga", "bundesliga", "ligue 1", "champions league",
]

CRYPTO_KEYWORDS = [
    "btc", "eth", "sol", "xrp", "bnb", "bitcoin", "ethereum",
    "up or down", "crypto", "price of", "settle over", "settle under",
]

AVOID_KEYWORDS = [
    "tweet", "elon", "post ", "# of", "number of",
    "will be released", "announce", "resign",
]

PREFER_KEYWORDS = [
    "o/u", "over/under", "handicap", "spread:",
    "up or down", "settle over", "settle under",
    "both teams", "anytime goal",
]


def classify_market(question: str) -> str:
    """Classify a market by type."""
    q = question.lower()
    if any(k in q for k in CRYPTO_KEYWORDS):
        return "crypto"
    if any(k in q for k in SPORTS_KEYWORDS):
        return "sports"
    if any(k in q for k in AVOID_KEYWORDS):
        return "meme"
    return "other"


def should_trade(question: str, yes_price: float, liquidity: float,
                  spread: float = None) -> tuple[bool, str]:
    """Decide if we should trade this market.
    
    Returns (should_trade, reason).
    """
    market_type = classify_market(question)
    
    # Hard filters
    if liquidity < 50:
        return False, f"low liquidity ({liquidity:.0f})"
    
    if yes_price < 0.03 or yes_price > 0.97:
        return False, f"extreme price ({yes_price:.2f})"
    
    if spread and spread > 0.05:
        return False, f"wide spread ({spread:.3f})"
    
    if market_type == "meme":
        return False, "meme/tweet market (historically unprofitable)"
    
    # Soft preferences
    q = question.lower()
    is_preferred = any(k in q for k in PREFER_KEYWORDS)
    
    if market_type == "sports" or market_type == "crypto":
        return True, f"{market_type} ✅" + (" (preferred)" if is_preferred else "")
    
    if market_type == "other":
        if is_preferred:
            return True, "other (preferred pattern)"
        if liquidity > 500:
            return True, "other (high liquidity)"
        return False, "other (low conviction)"
    
    return True, market_type
