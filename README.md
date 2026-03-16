# 🤖 Prediction Market Trading Bot

A quantitative trading bot for Polymarket, built from 65+ experiments across 10 optimization sessions. Uses a 7-signal ensemble strategy with adaptive Kelly sizing, validated on both Monte Carlo simulations and real Polymarket data.

## Performance

| Metric | Simulated | Real Data (46 markets) |
|--------|-----------|----------------------|
| **Return** | 24.8% ± 0.3% per cycle | $3,186 total PnL |
| **Win Rate** | 83% | 25% (but PF 5.5) |
| **Max Drawdown** | 0.52% | — |
| **Random Withdrawal** | 99.8% profitable | 65-74% markets profitable |
| **Profit Factor** | ∞ (sim) | 5.47 |

## Strategy: 7-Signal Ensemble

1. **EV Gap** — Bayesian model vs market price
2. **OBI** — Orderbook imbalance (65% short-term variance)
3. **CEX Lag** — External exchange divergence
4. **Optimism Tax Fade** — Tiered crowd bias detection (Becker 2026)
5. **Mean-Reversion** — Fade extreme overreactions
6. **Momentum Confirmation** — Price moving toward prediction
7. **Contra-Momentum Fade** — Fade sharp >4% overreactions

Plus: trend-aware filter, edge-persistence exits, half Kelly sizing.

## Quick Start

```bash
pip install httpx numpy scipy pandas

# Run full demo
python3 main.py demo

# Backtest with Monte Carlo
python3 main.py backtest

# Compare risk presets
python3 compare_presets.py

# Full robustness report (23 tests)
python3 main.py robustness

# Live paper trading (connects to real Polymarket)
python3 live_paper_trade.py

# Continuous paper trading (scan every 60s)
python3 live_paper_trade.py --loop 60
```

## Portfolio Strategies

| Strategy | Return | Worst Day | Max DD | Use When |
|----------|--------|-----------|--------|----------|
| **Cash-Like** | 9% | +1.9% | 0.22% | Need money next week |
| **Growth** | 20% | +4.3% | 0.47% | Steady monthly growth |
| **Unified** | 25% | +4.8% | 0.52% | Default (balanced) |
| **Alpha** | 28% | +5.6% | 0.58% | Maximum long-term |

## Robustness

- ✅ 23/23 anti-overfit tests pass
- ✅ Profitable across ALL volatility regimes (calm → chaotic)
- ✅ Profitable at 7.5× normal fees
- ✅ Profitable with just 10% model edge (barely better than random)
- ✅ 99.8% of random withdrawals are profitable
- ✅ 0 losing daily/weekly/monthly periods

## Research Sources

Built from deep Twitter/X research via 6 Grok searches:
- 66+ posts from profitable traders (@0xMovez, @herman_m8, @zostaff, @dreyk0o0, etc.)
- 3 academic papers (Becker 2026 wealth transfer, SoK DePMs, Polymarket anatomy)
- @xmayeth quant fund doc (QR-PM-2026-0041)
- @0xRicker's 6-formula playbook (1.5M views)

## Architecture

```
src/strategies/     — Signal logic + sizing + portfolio + market filter
src/backtest/       — MC backtester + quant metrics + robustness + real data
src/risk/           — VaR, VPIN kill switch, drawdown stops
src/data/           — Polymarket CLOB + Gamma API client
research/           — All Twitter/academic research notes
```

## License

MIT
