# Technical Alpha: Advanced Strategies (from Grok X Search #3)

## 1) Polymarket CLOB Orderbook Mechanics
- Polymarket migrated from AMM (LMSR) to CLOB in late 2024
- Price = best outstanding limit order (not AMM curve)
- Slippage = pure order-book impact
- Edge = spotting thin books + order-flow exhaustion before UI updates
- **@Mnilax arb bot architecture**:
  - Scans every market via py-clob-client every few seconds
  - Checks YES + NO sum < $1.00 (guaranteed arb)
  - Computes edge on VWAP from live order book
  - FOK/IOC/TTL order verification
  - Three modes: paper → shadow → live + recovery layer
  - $100 → $119 in shadow mode over 2 days

## 2) Avellaneda-Stoikov for Prediction Markets
**@0xMovez core thread — the gold standard**

Reservation price: `r = s − qγσ²(T−t)`
Optimal spread: `δ = γσ²(T−t) + (2/γ)ln(1 + γ/κ)`
PM probability adaptation: `logit(p) = ln(p/(1−p))`

- Bots quote around reservation and widen on inventory
- @gemchange_ltd: Full Stoikov IAQF 2023 lecture + GLFT inventory bounds
- Glosten-Milgrom adverse selection + VPIN kill switches
- Poisson assumptions break in real LOB → inventory pressure is non-linear

## 3) Kelly Criterion with Real Drawdown Data
Simplified PM Kelly: `f = (p − P) / (1 − P)`
- p = your estimated prob, P = market price

**@0xPhasma real results (30 live trades)**:
- Full Kelly: −42% drawdown (even at 65% WR!)
- Half Kelly: same edge/profit, drawdown −8%
- "Variance was destroying my account… cut sizes and profit stayed identical"

Empirical adjustment: `f_empirical = f_kelly × (1 − CV_edge)`
- Most pros run ¼–½ Kelly

## 4) Cross-Platform Arb Architecture
**@zostaff** — $100 → $5,214 in 24h, 847 zero-risk trades:
- Millisecond gap scanner (same event, different pricing)
- Buy YES Polymarket @0.61, sell YES Kalshi @0.67 instantly
- 3–8¢ edges typical

## 5) VPIN Flow Toxicity
`VPIN = |V_buy − V_sell| / (V_buy + V_sell)`
- Rolling volume buckets
- High VPIN = toxic informed flow → widen quotes or kill switch
- **Institutions pull ALL liquidity above VPIN ~0.6**
- PMs are "almost pure information games" — extreme info asymmetry

## 6) Monte Carlo Simulation
**@0xMovez code-included thread**:
- Sequential MC: logit random walk `logit(x_t) = logit(x_{t-1}) + ε_t, ε ~ N(0, σ²)`
- Tail events: importance sampling
- Correlated markets: upper/lower tail dependence λ_U, λ_L
- Agent-based sim (informed + noise + MM + bots)
- Run before every position → feeds empirical Kelly adjustment

## 7) Key GitHub Repos
| Repo | Stars | Language | Use |
|------|-------|----------|-----|
| py-clob-client | 842 | Python | Official CLOB SDK |
| agents framework | 2.4K | - | AI thesis → execution |
| clob-client | - | TypeScript | TS SDK |
| rs-clob-client | - | Rust | Low-latency |
| real-time-data-client | - | - | WS order flow |

## Production Stack Recipe
1. py-clob-client (market data + execution)
2. @0xMovez formulas (AS reservation + empirical Kelly + VPIN kill)
3. @Mnilax arb loop (scan → verify → execute → recover)
4. Monte-Carlo pre-trade simulation
5. Walk-forward validation

## Key Accounts for Ongoing Alpha
- @0xMovez (Avellaneda-Stoikov + Kelly + VPIN + Monte Carlo)
- @Mnilax (arb bot architecture)
- @zostaff (cross-platform arb)
- @gemchange_ltd (Stoikov/GLFT deep dives)
- @dunik_7 (repo aggregator)
- @0xPhasma (Kelly drawdown data)
