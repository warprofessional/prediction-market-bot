# Quant Fund Formulas (from @xmayeth tweet — QR-PM-2026-0041)

Source: https://x.com/xmayeth/status/2029943091528950248 (1.3M views, 8.1K likes)
Document: "Logarithmic Market Scoring Rule (LMSR): Pricing Mechanism & Inefficiency Detection"
+ "Real-Time Bayesian Signal Processing Agent Decision Architecture"
Classification: RESTRICTED, v2.3.1, 2026-02-14

## Page 1/7: LMSR Pricing

### 1. Cost Function
C(q) = b · ln(Σᵢ e^(qᵢ/b))
- b > 0 is liquidity parameter
- Larger b → tighter spreads, more liquidity, higher max MM loss

### Max Market Maker Loss
L_max = b · ln(n)
- For binary (n=2), b=100,000: L_max ≈ $69,315

### 2. Price Function (Softmax!)
pᵢ(q) = ∂C/∂qᵢ = e^(qᵢ/b) / Σⱼ e^(qⱼ/b)
- IDENTICAL to neural network softmax
- "The market is a neural network that prices beliefs"
- Σ pᵢ = 1 and pᵢ ∈ (0,1) ∀i

### 3. Cost of a Trade
Cost = C(q₁,...,qᵢ+δ,...,qₙ) - C(q₁,...,qᵢ,...,qₙ)

### 4. Inefficiency Signal — Entry Condition
(Page cut off — but this is the key section about when to enter)

## Page 3/7: Bayesian Signal Processing

### 1. Bayes' Theorem
P(H|D) = P(D|H) · P(H) / P(D)
"The traders who update fastest and most accurately win. Period."

### Production Latency (CRITICAL)
| Component | Avg Latency | p99 |
|-----------|------------|-----|
| Data ingestion (API/websocket) | 120ms | 340ms |
| Bayesian posterior computation | 15ms | 28ms |
| LMSR price comparison | 3ms | 8ms |
| Order execution (CLOB) | 690ms | 1400ms |
| **Total cycle** | **828ms** | **1776ms** |

### 2. Sequential Bayesian Updating
P(H|D₁,...,Dₜ) ∝ P(H) Πₖ P(Dₖ|H)

Log-space (numerically stable):
log P(H|D) = log P(H) + Σₖ log P(Dₖ|H) - log Z

### 3. Expected Value & Position Sizing
EV = p̂ · (1-p) - (1-p̂) · p = p̂ - p

**Handwritten note: "NEVER full Kelly on 5min markets!"**

## Key Thread Insights
- "Budget $250-500/month minimum for infra" (RPC + VPS)
- "Public Polygon endpoints won't work — insanely high latency"
- "Need more memory for logs → self-improvement"
- 1.3M views confirms this is the most-seen quant PM doc on X

## What This Changes For Our Strategy
1. LMSR price = softmax → we should compute LMSR cost impact before trading
2. 828ms total cycle → our sim's 1-step ≈ this latency window
3. Sequential Bayesian in log-space → more numerically stable than our current implementation
4. "NEVER full Kelly on 5min" → confirms our ¼ Kelly approach
5. Inefficiency signal (page 4-7 unseen) likely contains the actual entry criteria
