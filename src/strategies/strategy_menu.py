"""Strategy Menu — Pick your strategy, switch anytime.

Each strategy targets a different edge source with different risk/return:

┌─────────────────────────────────────────────────────────────────┐
│  STRATEGY          EDGE SOURCE         RETURN    RISK   NEEDS  │
│  ─────────────────────────────────────────────────────────────  │
│  CEX Lag           Binance speed       9-35%/3d  Low    Speed  │
│  Arb Scanner       YES+NO < $1.00     1-5%/day  V.Low  None   │
│  Optimism Fade     Crowd bias          5-15%/wk  Med    None   │
│  Near-Res Bond     Resolution cert.    1-3%/day  V.Low  Data   │
│  Full Ensemble     All 7 signals       25%/cycle Med    All    │
│  ─────────────────────────────────────────────────────────────  │
│  PORTFOLIO         Mix of above        Balanced  Adj.   All    │
└─────────────────────────────────────────────────────────────────┘

NONE of these use ML. All use math + data feeds.
"""


STRATEGY_INFO = {
    "cex_lag": {
        "name": "CEX Lag (Binance→Polymarket)",
        "description": "Detect BTC/ETH moves on Binance before Polymarket adjusts. The #1 proven money maker.",
        "edge": "Speed — CEX price leads Polymarket by 200ms-7s",
        "backtest": "26 trades, 100% WR, 9-35% return in 3.5 days (real BTC data)",
        "risk": "Low — only trades when clear directional move detected",
        "needs": "Binance.US API (free, no auth for price data)",
        "best_for": "Active crypto 5-min markets during trading hours",
    },
    "arb_scanner": {
        "name": "YES/NO Arb Scanner",
        "description": "Buy YES+NO when combined < $1.00. Guaranteed profit on resolution.",
        "edge": "Math — if YES+NO < 1.00, buying both guarantees profit",
        "backtest": "Near risk-free, but opportunities are rare and small (1-5¢)",
        "risk": "Very low — profit is locked in at entry",
        "needs": "Polymarket CLOB API only",
        "best_for": "Running 24/7 as background scanner",
    },
    "optimism_fade": {
        "name": "Optimism Tax Fade",
        "description": "Fade extreme crowd confidence. Becker (2026): takers overpay for YES.",
        "edge": "Behavioral — crowds systematically overprice likely outcomes",
        "backtest": "Part of unified strategy, contributes to 83% WR",
        "risk": "Medium — crowd can be right, need multiple confirming signals",
        "needs": "Market price + orderbook data",
        "best_for": "Sports/entertainment markets with strong crowd bias",
    },
    "bonding": {
        "name": "Near-Resolution Bonding",
        "description": "Buy at 97-99¢ when outcome is already certain. Sharky6999: $571K.",
        "edge": "Certainty — outcome is decided but market hasn't fully priced it",
        "backtest": "99.3% WR on verified wallet data",
        "risk": "Very low — outcome must be mechanically certain",
        "needs": "ESPN/AP/sports APIs for live score data",
        "best_for": "Sports during games, elections during vote counting",
    },
    "ensemble": {
        "name": "Full 7-Signal Ensemble",
        "description": "All signals combined: EV, OBI, lag, optimism, mean-rev, momentum, contra.",
        "edge": "Diversification — 7 independent signals reduce false positives",
        "backtest": "24.8% return, 83% WR, 0.52% max DD (Monte Carlo)",
        "risk": "Medium — relies on multiple signal agreement",
        "needs": "All data sources",
        "best_for": "General-purpose, any market type",
    },
}


def list_strategies():
    """Print all available strategies."""
    print("\n" + "=" * 70)
    print("  AVAILABLE STRATEGIES")
    print("=" * 70)
    for key, info in STRATEGY_INFO.items():
        print(f"\n  [{key}] {info['name']}")
        print(f"    {info['description']}")
        print(f"    Edge: {info['edge']}")
        print(f"    Backtest: {info['backtest']}")
        print(f"    Risk: {info['risk']}")
        print(f"    Best for: {info['best_for']}")
    print("\n" + "=" * 70)


PORTFOLIO_PRESETS = {
    "safe": {
        "name": "Safe Portfolio",
        "description": "Capital preservation. Arb + bonding + conservative ensemble.",
        "allocation": {"arb_scanner": 0.40, "bonding": 0.30, "ensemble": 0.30},
        "target": "5-10% monthly, never lose a week",
    },
    "balanced": {
        "name": "Balanced Portfolio",
        "description": "Good risk/return. CEX lag + ensemble + optimism fade.",
        "allocation": {"cex_lag": 0.40, "ensemble": 0.35, "optimism_fade": 0.25},
        "target": "15-30% monthly, rarely lose a day",
    },
    "aggressive": {
        "name": "Aggressive Portfolio",
        "description": "Maximum returns. Heavy CEX lag + ensemble.",
        "allocation": {"cex_lag": 0.60, "ensemble": 0.30, "optimism_fade": 0.10},
        "target": "30-100%+ monthly, higher drawdowns acceptable",
    },
}


def list_portfolios():
    """Print all portfolio presets."""
    print("\n" + "=" * 70)
    print("  PORTFOLIO PRESETS")
    print("=" * 70)
    for key, p in PORTFOLIO_PRESETS.items():
        print(f"\n  [{key}] {p['name']}")
        print(f"    {p['description']}")
        print(f"    Target: {p['target']}")
        alloc = ", ".join(f"{STRATEGY_INFO[s]['name'].split('(')[0].strip()} {pct:.0%}" for s, pct in p['allocation'].items())
        print(f"    Allocation: {alloc}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    list_strategies()
    print()
    list_portfolios()
