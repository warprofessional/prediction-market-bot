"""Bot configuration — all tunable parameters in one place."""
import os
from dotenv import load_dotenv

load_dotenv()

# === API Keys ===
POLYMARKET_PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "")
KALSHI_API_KEY = os.getenv("KALSHI_API_KEY", "")
KALSHI_API_SECRET = os.getenv("KALSHI_API_SECRET", "")

# === Polymarket ===
POLY_CLOB_HOST = "https://clob.polymarket.com"
POLY_CHAIN_ID = 137  # Polygon

# === Kalshi ===
KALSHI_BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"

# === Strategy Parameters ===

# Strategy 1: YES/NO Arb
ARB_MIN_EDGE = 0.005       # Minimum edge in $ (0.5¢)
ARB_MAX_POSITION = 500     # Max $ per arb cycle
ARB_SCAN_INTERVAL = 2.0    # Seconds between scans

# Strategy 2: Bayesian EV Scanner
EV_MIN_EDGE = 0.03         # Minimum EV gap (3¢)
EV_CONFIDENCE_THRESHOLD = 0.6  # Min confidence in our probability estimate

# Strategy 3: Cross-Platform Arb
XARB_MIN_EDGE = 0.02       # Minimum cross-platform gap (2¢)
XARB_MAX_POSITION = 1000   # Max $ per cross-arb trade

# Strategy 4: Avellaneda-Stoikov MM
AS_GAMMA = 0.1             # Risk aversion parameter
AS_KAPPA = 1.5             # Order arrival rate
AS_SIGMA = 0.05            # Volatility estimate

# === Risk Management ===
KELLY_FRACTION = 0.25      # Use quarter-Kelly
MAX_PORTFOLIO_RISK = 0.15  # 15% max drawdown → kill switch
VPIN_KILL_THRESHOLD = 0.6  # Pull all orders above this
VAR_CONFIDENCE = 0.99      # 99% VaR
MAX_SINGLE_POSITION = 0.10 # Max 10% of capital in one market
INITIAL_CAPITAL = 1000.0   # Starting capital in USD

# === Execution ===
ORDER_TYPE = "FOK"         # Fill-or-Kill for arb
SLIPPAGE_TOLERANCE = 0.01  # 1% max slippage
MAX_RETRIES = 3
RETRY_DELAY = 0.5          # seconds

# === Monitoring ===
LOG_LEVEL = "INFO"
DASHBOARD_REFRESH = 5      # seconds
PNL_TRACKING = True

# === Modes ===
DRY_RUN = True             # Paper trading (no real money)
SHADOW_MODE = False        # Track what would happen without trading
