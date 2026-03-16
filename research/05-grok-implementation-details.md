# Implementation Details — Deep Technical (Grok Search #5)

## KEY INSIGHT: 90%+ of "bot" threads are FAKE
- Admitted fakes, Claude/Grok dashboards, or hype with zero executable code
- Dynamic taker fees + 250-500ms maker advantage KILLED pure latency takers
- Survivors: maker-only, spread/arb optimization, hybrid lag + certainty plays

## Near-Resolution Bonding (Technical Details)
- **Timing**: Last 60-90s or <5 min before close
- **Certainty detection**: Cross-check resolution source (UMA oracle rules + public data)
- **Entry price**: 0.97-0.999 (never earlier)
- **Sizing**: Fixed small % or fractional Kelly (no full Kelly near expiry — slippage)
- **Risk**: Low-liq slippage or sudden oracle flip even at 99¢

## Optimism Tax Exploitation
- Takers overpay in biased flow (longshot/optimism bias)
- Makers quote mirrored YES/NO limits around midpoint in thin books (depth < $2K)
- Takers hit one side → collect rebate + settlement edge
- Bid/ask imbalance where crowd piles one side (85%+ hype) → spread widens 2-3¢

## OBI for 5-min Markets
- **Microprice**: μ = (P_bid × V_ask + P_ask × V_bid) / (V_bid + V_ask)
- **Classic OBI**: (Q_bid – Q_ask) / (Q_bid + Q_ask)
- At φ levels (~0.618), bid depth collapses → reversal signal

## Binance-Polymarket Lag Detection
- Dual WS: Binance spot 1s ticks + Polymarket CLOB
- **Threshold**: >0.3-8% divergence or 0.15% impulse in <45s while Poly lags 200ms-7s
- Bot fires in <100ms window
- Near expiry (last 20-40s): tighten toward dominant side

## Frank-Wolfe Market Maker (Rust — from @RohOnChain)
```rust
fn exploit_polymarket() {
    let markets = fetch_markets();
    loop {
        for market in markets {
            if detect_mispricing(&market) {
                let (z_0, u, sigma_ext) = init_fw(&market);
                let (mu_t, theta_t) = barrier_fw(z_0, u, sigma_ext);
                let profit = bregman_divergence(mu_t, theta) - fw_gap(mu_t);
                if profit >= 0.9 * bregman_divergence(mu_t, theta) {
                    execute_trade(theta_t);
                }
            }
        }
        sleep(Duration::from_millis(1));
    }
}
```

## 5-min BTC Spread Bot Logic
1. Auto-detect new 5-min market
2. Place mirrored limits at ~0.46 both sides (or 0.98 – current for hedge)
3. Monitor book, adjust closer
4. Risk-min exit if opposite move
5. Limit orders only = zero fees + rebates
6. ~$0.30/trade, hundreds/day

## THE SINGLE BEST COMBINED STRATEGY
**Maker-only Frank-Wolfe Optimized Lag + Spread + Near-Res Bonding**

Pipeline:
1. WS dual feed: Binance spot + Poly CLOB (every 1ms)
2. Detect: mispricing (YES+NO <1.00 or >0.3-8% lag div) + OBI/microprice filter + low volume
3. Init FW/Bregman: compute guaranteed α-profit projection
4. Size: fractional Kelly + EV check + b-parameter (skip thin pools)
5. Execute: maker limits only (FOK/IOC hedge on partial). Near expiry (last 60s): bond obvious side at 97-99¢
6. Monitor: cancel/reprice on momentum, collect rebates + settlement
7. Risk: 0.5% per trade, 2% daily cap, shadow mode first

## Risks That Kill Strategies
- Dynamic taker fees + 250-500ms maker delay
- Edge decay: Competition tightens windows (290ms → <100ms)
- Drawdowns: Failed hedges, oracle changes, low-liq slippage
- Max DD examples: ~3.8% in sims
- Kyle λ spike = informed flow → EXIT immediately
- API rate limits, fees eating small edges
