# Prediction Market Bot — Strategy Playbook
## Goal: Beat hedge funds and HFT on Polymarket/Kalshi

---

## 🏆 TOP 15 STRATEGIES (Ranked by Risk-Adjusted Return)

### TIER 1: Near Risk-Free (Arb Strategies)

**Strategy 1: Intra-Platform YES/NO Arbitrage**
- Buy YES + NO when sum < $1.00 on same market
- Guaranteed 1-5¢ per cycle, compounding
- Win rate: ~100% (mechanical)
- Expected PnL: ~$144/hr on 4 markets (theoretical)
- Risk: Execution speed, fees, slippage
- Source: @w1nklerr, @0x_Discover, Gabagool Ladder bot

**Strategy 2: Cross-Platform Arbitrage (Polymarket ↔ Kalshi)**
- Same events, different pricing → 3-8¢ gaps
- Buy YES Poly @0.61, sell YES Kalshi @0.67 instantly
- Win rate: ~100% (zero directional risk)
- Real PnL: $100 → $5,214 in 24h (847 trades) — @zostaff
- Risk: Execution latency, platform settlement differences
- Architecture: millisecond gap scanner, simultaneous execution

**Strategy 3: Cross-Platform Hedge Lock**
- Bet win on Kalshi, use implied profit to bet loss on Poly
- Worst-case breakeven, best-case locked profit
- Source: @AIexey_Stark
- Risk: Resolution dispute between platforms

### TIER 2: High-EV Statistical (60-75% WR)

**Strategy 4: Bayesian EV Scanner**
- Compute P(H|D) with live data → compare to market price
- Trade only when EV = p̂ − p > threshold
- Kelly-size with ¼ Kelly
- Real PnL: $1,000 → $3,320 (72% WR, 1% max DD) — @dreyk0o0
- Risk: Model miscalibration

**Strategy 5: EMA Mean-Reversion Bot**
- Trade ±8% deviation from EMA on 5-15min markets
- Target 60-65% WR
- Backtest before deploying (slippage eats edge in fast markets)
- Source: @frostikkkk
- Risk: Regime changes, trending markets

**Strategy 6: Probability Surface Gradient + Entropy**
- Multi-dimensional probability modeling
- Real PnL: $1,200 → $6,740 in 3 days — @L1vsun
- Risk: Complex implementation, overfitting

**Strategy 7: News-Based Bayesian Trading**
- Real-time news/polls → Bayesian prob updates → EV trade
- Works best: geopolitics, oil, elections, CPI
- Real PnL: $1,200 → +$2,847 (+237%) in 9 days — @DaoMariowhales
- Risk: News parsing errors, hallucination in AI

### TIER 3: Market Making (Sophisticated)

**Strategy 8: Avellaneda-Stoikov Adaptive Market Making**
- Reservation price: r = s − qγσ²(T−t)
- Optimal spread: δ = γσ²(T−t) + (2/γ)ln(1 + γ/κ)
- VPIN kill switch at >0.6 (toxic flow protection)
- Source: @0xMovez, @gemchange_ltd
- Risk: Adverse selection, spoofing ($16K/day spoofing attacks documented)

**Strategy 9: Order Book Imbalance Trading**
- Trade bid/ask wall imbalances on 15-min BTC/ETH
- Source: @JR_OnChain AI tournament
- Risk: Spoofing, thin books

### TIER 4: Structural Edge (Medium-Term)

**Strategy 10: Longshot Bias Exploitation**
- Systematically SHORT overpriced low-probability events
- Crowds overweight unlikely outcomes → sell into them
- Source: @herman_m8 hedge fund desk analysis
- Risk: Black swan events (tail risk)

**Strategy 11: Sportsbook Lag Capture**
- Trade when sportsbooks move BEFORE prediction markets
- Speed arbitrage on information flow
- Source: @herman_m8
- Risk: Requires sportsbook data feeds

**Strategy 12: Pair Trading / Stat Arb on Events**
- Map correlated assets to events (e.g., "Rep win presidency" vs "Trump win")
- Divergence → long one/short other
- Source: @anthonyt590361
- Risk: Correlation breakdown

**Strategy 13: Weather Market NOAA Arbitrage**
- Scan NOAA forecast every 2min
- Buy undervalued temp buckets <15¢, flip at >45¢
- Source: @0xMovez Clawdbot
- Risk: Forecast revisions

**Strategy 14: Conditional Arbitrage Graphs**
- Find mispricing between logically related markets
- P(BTC > 95K) ≥ P(BTC > 100K) ≥ P(BTC > 105K) — if broken → arb
- Source: @herman_m8
- Risk: Low frequency of opportunities

**Strategy 15: Volatility-Aware Momentum**
- Edge score: ES = (p_model − p_market) / σ_market
- Filter trades by edge/vol ratio
- Real PnL: $100 → $1,074 then −$500 DD (improved with ES filter) — @gipppezkv
- Risk: Regime changes

---

## 📊 BACKTESTING FRAMEWORK

### Architecture
```
┌─────────────────────────────────────────────────┐
│                 DATA LAYER                       │
│  py-clob-client → Historical orderbooks          │
│  Kalshi REST API → Historical trades             │
│  NOAA/News APIs → Event data                     │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│              SIMULATION ENGINE                    │
│  Monte Carlo: logit(x_t) = logit(x_{t-1}) + ε   │
│  Agent-based: informed + noise + MM + bots       │
│  Importance sampling for tail events              │
│  Correlated markets: tail dependence λ_U, λ_L    │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│              STRATEGY EVALUATOR                   │
│  Walk-forward validation (no lookahead)           │
│  Robustness: param sensitivity ±20%              │
│  If only 1 combo works → REJECT (curve-fit)      │
│  Slippage model: VWAP + spread + fees            │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│              RISK METRICS                         │
│  Sharpe Ratio (target >2.0)                      │
│  Max Drawdown (target <15%)                      │
│  VaR: Portfolio × Z × σ × √T                    │
│  Win Rate + Profit Factor                        │
│  Kelly fraction vs realized edge                 │
└─────────────────────────────────────────────────┘
```

### Eval Suite Metrics
| Metric | Target | Description |
|--------|--------|-------------|
| Win Rate | >60% | % of profitable trades |
| Profit Factor | >1.5 | Gross profit / gross loss |
| Sharpe Ratio | >2.0 | Risk-adjusted return |
| Max Drawdown | <15% | Worst peak-to-trough |
| Calmar Ratio | >3.0 | Annual return / max DD |
| Average Edge | >3¢ | Mean EV per trade |
| VPIN Correlation | Negative | P&L should be negative when VPIN high |
| Param Sensitivity | <20% impact | Robust to ±20% param changes |

### Walk-Forward Protocol
1. Split data: 70% train, 30% test (sliding window)
2. Optimize on window 1
3. Test on window 2 (out-of-sample)
4. Roll forward, repeat
5. If only one combo works → curve-fit → reject

---

## 🤖 BOT ARCHITECTURE

```
┌─────────────────────────────────────────────────────────┐
│                    PREDICTION MARKET BOT                  │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │  DATA    │  │ STRATEGY │  │ EXECUTION│  │  RISK   │ │
│  │  INGEST  │→ │  ENGINE  │→ │  ENGINE  │→ │ MONITOR │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
│       │              │              │             │      │
│  WebSocket      Bayesian       VWAP split      VaR     │
│  Order flow     EV calc        FOK/IOC/TTL     Kelly   │
│  News feeds     Kelly size     Gasless relay   VPIN    │
│  NOAA/sports    VPIN check     Auto-hedge      DD stop │
│                 Monte Carlo    Recovery         Kill sw │
└─────────────────────────────────────────────────────────┘
```

### Implementation Phases
1. **Phase 1**: Intra-Polymarket YES/NO arb (safest, prove infra works)
2. **Phase 2**: Add Bayesian EV scanner + Kelly sizing
3. **Phase 3**: Cross-platform arb (Poly ↔ Kalshi)
4. **Phase 4**: Avellaneda-Stoikov market making
5. **Phase 5**: News-based event trading + multi-strategy portfolio

### Tech Stack
- **Language**: Python (py-clob-client) + Rust (low-latency execution)
- **Data**: py-clob-client WebSocket, Kalshi REST, NOAA API, news APIs
- **Backtesting**: Pandas + vectorbt + custom Monte Carlo
- **Execution**: Builder gasless relayer, FOK/IOC orders
- **Monitoring**: Rich/Textual dashboard, kreo.app tracking
- **Infra**: 24/7 VPS, kill switches, rate limits

---

## 🎯 REALISTIC PnL EXPECTATIONS

### Conservative (Phase 1-2, first month)
- Capital: $1,000
- Strategy: YES/NO arb + Bayesian EV
- Expected: $50-200/week (5-20% weekly)
- Max DD: <5%

### Moderate (Phase 3-4, months 2-3)  
- Capital: $5,000
- Strategy: Multi-strat portfolio
- Expected: $500-2,000/week
- Max DD: <10%

### Aggressive (Phase 5, months 3+)
- Capital: $10,000+
- Strategy: Full stack + market making
- Expected: $2,000-10,000/week
- Max DD: <15%

### Reality Check
- Most Twitter PnL screenshots are cherry-picked best days
- Edges decay as more bots enter → must continuously iterate
- Fees/slippage eat 20-40% of gross edge
- Resolution disputes can nuke positions
- @0xPhasma: Even 65% WR with full Kelly → -42% DD

---

## 📚 KEY REFERENCES

### Twitter Accounts (Alpha)
@0xMovez, @herman_m8, @Mnilax, @zostaff, @dreyk0o0, @DaoMariowhales, @vijn_crypto, @0xPhasma, @gemchange_ltd, @dunik_7, @StarPlatinum_, @gabagool22, @w1nklerr, @L1vsun

### GitHub Repos
- polymarket/py-clob-client (842⭐)
- Gabagool2-2/polymarket-trading-bot-python
- echandsome/Polymarket-betting-bot
- polymarket/agents (2.4K⭐)

### Academic
- Avellaneda & Stoikov (2008): "High-frequency trading in a limit order book"
- Kelly (1956): "A New Interpretation of Information Rate"
- Easley, López de Prado, O'Hara: "Flow Toxicity and Liquidity" (VPIN)
- MIT OpenCourseWare: Financial Mathematics
