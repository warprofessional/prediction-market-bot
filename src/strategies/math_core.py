"""Core mathematical formulas used across all strategies.

Based on research from @0xMovez, @herman_m8, @dreyk0o0, and academic papers:
- Avellaneda & Stoikov (2008)
- Kelly (1956)
- Easley, López de Prado, O'Hara (VPIN)
"""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ============================================================
# BAYESIAN UPDATING
# ============================================================

def bayesian_update(prior: float, likelihood: float, evidence: float) -> float:
    """Bayes' theorem: P(H|D) = P(D|H) × P(H) / P(D)
    
    Args:
        prior: P(H) — our prior probability
        likelihood: P(D|H) — probability of data given hypothesis
        evidence: P(D) — total probability of observing data
    
    Returns:
        Posterior probability P(H|D)
    """
    if evidence == 0:
        return prior
    return (likelihood * prior) / evidence


def sequential_bayesian_log(log_prior: float, log_likelihoods: list[float]) -> float:
    """Sequential Bayesian updating in log-space (numerically stable).
    
    From QR-PM-2026-0041 (page 3):
    log P(H|D) = log P(H) + Σ_k log P(D_k|H) - log Z
    
    Args:
        log_prior: log P(H) — log of prior probability
        log_likelihoods: list of log P(D_k|H) for each data point
    
    Returns:
        Posterior probability P(H|D) (NOT in log space)
    """
    log_posterior_unnorm = log_prior + sum(log_likelihoods)
    # For binary: P(H|D) + P(¬H|D) = 1
    # log Z = log(exp(log_post_H) + exp(log_post_not_H))
    # Simplified: just use sigmoid of unnormalized log posterior
    # This avoids overflow for large log values
    if log_posterior_unnorm > 20:
        return 1.0 - 1e-10
    elif log_posterior_unnorm < -20:
        return 1e-10
    return 1.0 / (1.0 + math.exp(-log_posterior_unnorm))


def bayesian_update_binary(prior: float, sensitivity: float, specificity: float) -> float:
    """Binary Bayesian update with sensitivity and specificity.
    
    Args:
        prior: P(event happens)
        sensitivity: P(signal | event happens) — true positive rate
        specificity: P(no signal | event doesn't happen) — true negative rate
    
    Returns:
        Updated probability given positive signal
    """
    # P(D) = P(D|H)P(H) + P(D|¬H)P(¬H)
    p_evidence = sensitivity * prior + (1 - specificity) * (1 - prior)
    if p_evidence == 0:
        return prior
    return (sensitivity * prior) / p_evidence


# ============================================================
# EXPECTED VALUE
# ============================================================

def expected_value(model_prob: float, market_price: float) -> float:
    """EV = p̂ − p (trade only if positive).
    
    Args:
        model_prob: Our estimated true probability
        market_price: Current market price (= implied probability)
    
    Returns:
        Expected value of buying at market_price
    """
    return model_prob - market_price


def ev_full(model_prob: float, buy_price: float) -> float:
    """Full EV calculation: EV = p × profit − (1−p) × loss.
    
    For binary market: profit = 1 - buy_price, loss = buy_price
    """
    profit = 1.0 - buy_price
    loss = buy_price
    return model_prob * profit - (1 - model_prob) * loss


# ============================================================
# KELLY CRITERION
# ============================================================

def kelly_fraction(model_prob: float, market_price: float) -> float:
    """Kelly criterion for binary prediction market.
    
    f* = (p̂ × b − q) / b
    where b = odds = (1-price)/price, q = 1-p̂
    
    Simplified for PM: f = (p - P) / (1 - P)
    where p = model_prob, P = market_price
    
    Returns fraction of capital to bet (0 to 1).
    """
    if market_price >= 1.0 or market_price <= 0.0:
        return 0.0
    
    f = (model_prob - market_price) / (1.0 - market_price)
    return max(0.0, min(1.0, f))


def fractional_kelly(model_prob: float, market_price: float, fraction: float = 0.25) -> float:
    """Fractional Kelly — reduces variance at cost of lower growth.
    
    Most pros use 1/4 Kelly. @0xPhasma showed:
    - Full Kelly at 65% WR → -42% drawdown
    - Half Kelly → -8% drawdown, same profit
    
    Args:
        fraction: 0.25 (quarter), 0.5 (half), 1.0 (full)
    """
    return kelly_fraction(model_prob, market_price) * fraction


def empirical_kelly(model_prob: float, market_price: float, cv_edge: float, 
                     fraction: float = 0.25) -> float:
    """Empirical Kelly adjusted for uncertainty in edge estimate.
    
    f_empirical = f_kelly × (1 − CV_edge)
    where CV_edge = coefficient of variation of edge estimate.
    Higher uncertainty → smaller position.
    
    Source: @0xPhasma, @herman_m8
    """
    f = kelly_fraction(model_prob, market_price)
    return max(0.0, f * (1.0 - cv_edge) * fraction)


# ============================================================
# AVELLANEDA-STOIKOV MARKET MAKING
# ============================================================

@dataclass
class ASQuotes:
    """Avellaneda-Stoikov optimal quotes."""
    reservation_price: float
    bid: float
    ask: float
    spread: float


def avellaneda_stoikov(
    mid_price: float,
    inventory: float,  # Positive = long, negative = short
    gamma: float,      # Risk aversion (higher = more conservative)
    sigma: float,      # Volatility estimate
    time_remaining: float,  # Fraction of period remaining (0 to 1)
    kappa: float = 1.5,     # Order arrival intensity
) -> ASQuotes:
    """Avellaneda-Stoikov optimal market making quotes.
    
    Reservation price: r = s − qγσ²(T−t)
    Optimal spread: δ = γσ²(T−t) + (2/γ)ln(1 + γ/κ)
    
    Source: @0xMovez, @gemchange_ltd, Avellaneda & Stoikov (2008)
    """
    # Reservation price (shifts away from inventory)
    reservation = mid_price - inventory * gamma * sigma**2 * time_remaining
    
    # Optimal spread
    spread = gamma * sigma**2 * time_remaining + (2.0 / gamma) * math.log(1 + gamma / kappa)
    
    bid = reservation - spread / 2
    ask = reservation + spread / 2
    
    # Clamp to valid range [0, 1] for prediction markets
    bid = max(0.001, min(0.999, bid))
    ask = max(0.001, min(0.999, ask))
    
    return ASQuotes(
        reservation_price=reservation,
        bid=bid,
        ask=ask,
        spread=ask - bid,
    )


def lmsr_price_impact(current_price: float, trade_size: float, 
                       liquidity_b: float = 100000.0) -> float:
    """LMSR price impact: how much price moves for a given trade size.
    
    From QR-PM-2026-0041: C(q) = b · ln(Σ e^(q_i/b)), price = softmax
    The cost to move price from p to p' is non-linear (convex).
    
    Returns the new price after buying `trade_size` shares.
    """
    # For binary market: p = e^(q/b) / (e^(q/b) + e^(q'/b))
    # Buying δ shares: new_p = e^((q+δ)/b) / (e^((q+δ)/b) + e^(q'/b))
    p = max(1e-10, min(1 - 1e-10, current_price))
    q = math.log(p / (1 - p)) * liquidity_b  # Invert softmax to get q
    q_prime = math.log((1 - p) / p) * liquidity_b
    
    # After buying δ shares of outcome 1
    new_q = q + trade_size
    new_price = math.exp(new_q / liquidity_b) / (math.exp(new_q / liquidity_b) + math.exp(q_prime / liquidity_b))
    return max(0.001, min(0.999, new_price))


def lmsr_trade_cost(current_price: float, trade_size: float,
                     liquidity_b: float = 100000.0) -> float:
    """Cost to execute a trade of `trade_size` shares via LMSR.
    
    Cost = C(q+δ) - C(q) where C(q) = b · ln(Σ e^(q_i/b))
    """
    p = max(1e-10, min(1 - 1e-10, current_price))
    q1 = math.log(p / (1 - p)) * liquidity_b
    q2 = math.log((1 - p) / p) * liquidity_b
    
    c_before = liquidity_b * math.log(math.exp(q1/liquidity_b) + math.exp(q2/liquidity_b))
    c_after = liquidity_b * math.log(math.exp((q1+trade_size)/liquidity_b) + math.exp(q2/liquidity_b))
    
    return c_after - c_before


def logit_transform(p: float) -> float:
    """logit(p) = ln(p / (1-p)) — maps probability to real line."""
    p = max(1e-10, min(1 - 1e-10, p))
    return math.log(p / (1 - p))


def inv_logit(x: float) -> float:
    """Inverse logit (sigmoid) — maps real line to probability."""
    return 1.0 / (1.0 + math.exp(-x))


# ============================================================
# VPIN (Volume-Synchronized Probability of Informed Trading)
# ============================================================

def vpin(buy_volume: float, sell_volume: float) -> float:
    """VPIN = |V_buy − V_sell| / (V_buy + V_sell)
    
    High VPIN (>0.6) = toxic informed flow → widen quotes or kill.
    Source: @0xMovez, Easley et al.
    """
    total = buy_volume + sell_volume
    if total == 0:
        return 0.0
    return abs(buy_volume - sell_volume) / total


def classify_trade_direction(price: float, mid: float) -> str:
    """Lee-Ready trade direction classification."""
    if price > mid:
        return "buy"
    elif price < mid:
        return "sell"
    return "unknown"


# ============================================================
# RISK METRICS
# ============================================================

def value_at_risk(portfolio_value: float, confidence: float, 
                   sigma: float, time_horizon: float = 1.0) -> float:
    """VaR = Portfolio × Z × σ × √T
    
    Source: @herman_m8 hedge fund desk
    """
    from scipy.stats import norm
    z = norm.ppf(confidence)
    return portfolio_value * z * sigma * math.sqrt(time_horizon)


def max_drawdown(equity_curve: list[float]) -> float:
    """Calculate maximum drawdown from equity curve."""
    if len(equity_curve) < 2:
        return 0.0
    
    peak = equity_curve[0]
    max_dd = 0.0
    
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak
        if dd > max_dd:
            max_dd = dd
    
    return max_dd


def sharpe_ratio(returns: list[float], risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    
    arr = np.array(returns)
    excess = arr - risk_free_rate / 252  # Daily risk-free
    
    if np.std(excess) == 0:
        return 0.0
    
    return np.mean(excess) / np.std(excess) * np.sqrt(252)


def profit_factor(wins: list[float], losses: list[float]) -> float:
    """Profit factor = gross profit / gross loss."""
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


# ============================================================
# MONTE CARLO SIMULATION
# ============================================================

def monte_carlo_probability_path(
    initial_prob: float,
    sigma: float,
    n_steps: int = 100,
    n_simulations: int = 10000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate probability paths using logit random walk.
    
    logit(x_t) = logit(x_{t-1}) + ε_t, ε ~ N(0, σ²)
    
    Source: @0xMovez Monte Carlo thread
    
    Returns:
        Array of shape (n_simulations, n_steps + 1)
    """
    rng = np.random.default_rng(seed)
    
    logit_init = logit_transform(initial_prob)
    
    # Generate noise
    noise = rng.normal(0, sigma, size=(n_simulations, n_steps))
    
    # Build logit paths
    logit_paths = np.zeros((n_simulations, n_steps + 1))
    logit_paths[:, 0] = logit_init
    
    for t in range(n_steps):
        logit_paths[:, t + 1] = logit_paths[:, t] + noise[:, t]
    
    # Convert back to probability space
    prob_paths = 1.0 / (1.0 + np.exp(-logit_paths))
    
    return prob_paths


def estimate_edge_uncertainty(
    model_prob: float,
    market_price: float,
    sigma: float,
    n_simulations: int = 5000,
) -> dict:
    """Estimate uncertainty of our edge via Monte Carlo.
    
    Returns dict with mean_edge, std_edge, cv_edge, prob_profitable.
    """
    rng = np.random.default_rng()
    
    # Simulate possible true probabilities around our estimate
    logit_p = logit_transform(model_prob)
    simulated_logits = rng.normal(logit_p, sigma, n_simulations)
    simulated_probs = 1.0 / (1.0 + np.exp(-simulated_logits))
    
    # Compute edge for each simulation
    edges = simulated_probs - market_price
    
    return {
        "mean_edge": float(np.mean(edges)),
        "std_edge": float(np.std(edges)),
        "cv_edge": float(np.std(edges) / abs(np.mean(edges))) if np.mean(edges) != 0 else float("inf"),
        "prob_profitable": float(np.mean(edges > 0)),
        "p5_edge": float(np.percentile(edges, 5)),
        "p95_edge": float(np.percentile(edges, 95)),
    }
