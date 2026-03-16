# Prediction Market Trading Strategies (from Grok X Search)

## Key Twitter Accounts to Follow
- @dreyk0o0 — LMSR + Bayesian AI agent, $1K → $3.3K (72% WR)
- @DaoMariowhales — Real-time news scan bot, +237% in 9 days
- @vijn_crypto — Full quant toolkit (LMSR, Kelly, KL-divergence)
- @frostikkkk — EMA mean-reversion bot (60-65% WR)
- @zostaff — Cross-platform arb bot, $100 → $5.2K in 24h
- @w1nklerr — Claude 5-min micro-arb, $2.3K first night
- @0x_Discover — 2000 bets/day bot, $250K in 3 days
- @slash1sol — NBA arb bot, $629K net
- @0xMovez — Weather bot (NOAA + Claude)
- @RohOnChain — KL-divergence math deep-dives
- @gabagool22 — Top arb trader
- @0x8dxd — Momentum ($300 → $650K)
- @noisyb0y1 — OpenClaw basket arb
- @anthonyt590361 — Pair trading / stat arb
- @AIexey_Stark — Cross-platform hedge

## Strategy Categories

### 1) Quantitative/Algorithmic
- **Core formulas**: LMSR pricing, Bayesian P(H|D), EV = p̂ − p, fractional Kelly f* = (p̂ × b − q)/b
- Half/quarter Kelly recommended to avoid ruin
- Scan for model prob > market prob
- EMA ±8% deviation mean-reversion for overreactions

### 2) Arbitrage (Dominant profitable category)
- **Cross-platform (Poly ↔ Kalshi)**: 3-8¢ gaps, bots close in ms
- **Intra-Polymarket**: Buy YES+NO when combined < $1, lock 1-5¢ guaranteed
- **Sports**: Reverse-engineered inefficiencies vs bookies
- **Weather**: NOAA forecast scan every 2min, buy <15¢, flip >45¢
- **Math**: KL-divergence + Bregman projection + Adaptive Frank-Wolfe

### 3) Market Making
- AI tournament on Kalshi 15-min BTC/ETH
- Order book bid/ask wall imbalance trading
- WARNING: Adverse fills, Qmin penalties, spoofing attacks make pure MM risky

### 4) AI/ML-Based
- Feed LMSR/Bayesian/Kelly to Claude/OpenClaw → auto-bot
- Pattern: $1K → 3x+ in days with 70%+ WR
- 5-min crypto + basket arb scalable

### 5) News-Based Event Trading
- Real-time news/polls → Bayesian prob updates → EV trade
- Avoid meme/subjective markets
- Focus: elections, CPI, geopolitics, oil

### 6) Hedging
- **Pair trading**: Map correlated Wall St assets to prediction events
- **Cross-platform hedge**: Bet win on Kalshi, bet loss on Poly → worst-case breakeven

## Key Takeaways
- Automation wins (edges are small/frequent)
- Common stack: Claude/OpenClaw + LMSR + Bayesian + Kelly
- Risks: fees/slippage, resolution disputes, spoofing, adverse selection
- Most advise small tests + backtesting before deploying
