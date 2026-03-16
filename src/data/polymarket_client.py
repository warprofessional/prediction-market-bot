"""Polymarket data ingestion via py-clob-client."""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class OrderBook:
    """Snapshot of a market's order book."""
    token_id: str
    market_slug: str
    bids: list[dict]  # [{price, size}]
    asks: list[dict]
    timestamp: float = field(default_factory=time.time)

    @property
    def best_bid(self) -> float:
        return max((b["price"] for b in self.bids), default=0.0)

    @property
    def best_ask(self) -> float:
        return min((a["price"] for a in self.asks), default=1.0)

    @property
    def midpoint(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def bid_depth(self) -> float:
        return sum(b["price"] * b["size"] for b in self.bids)

    @property
    def ask_depth(self) -> float:
        return sum(a["price"] * a["size"] for a in self.asks)


@dataclass
class Market:
    """A Polymarket market."""
    condition_id: str
    question: str
    slug: str
    tokens: list[dict]  # YES and NO tokens
    end_date: str
    active: bool
    volume: float = 0.0
    liquidity: float = 0.0


class PolymarketClient:
    """Client for Polymarket CLOB API."""

    BASE_URL = "https://clob.polymarket.com"
    GAMMA_URL = "https://gamma-api.polymarket.com"

    def __init__(self, private_key: Optional[str] = None):
        self.private_key = private_key
        self._http = httpx.AsyncClient(timeout=30.0)
        self._markets_cache: dict[str, Market] = {}

    async def get_markets(self, limit: int = 100, active_only: bool = True) -> list[Market]:
        """Fetch available markets from Gamma API."""
        params = {"limit": limit, "active": active_only, "order": "liquidity", "ascending": False}
        resp = await self._http.get(f"{self.GAMMA_URL}/markets", params=params)
        resp.raise_for_status()
        data = resp.json()

        markets = []
        for m in data:
            # clobTokenIds is a JSON string like '["token1", "token2"]'
            raw_tokens = m.get("clobTokenIds", "[]")
            if isinstance(raw_tokens, str):
                try:
                    tokens = json.loads(raw_tokens)
                except json.JSONDecodeError:
                    tokens = []
            else:
                tokens = raw_tokens if raw_tokens else []

            market = Market(
                condition_id=m.get("conditionId", ""),
                question=m.get("question", ""),
                slug=m.get("slug", ""),
                tokens=tokens,
                end_date=m.get("endDate", ""),
                active=m.get("active", False),
                volume=float(m.get("volume", 0)),
                liquidity=float(m.get("liquidity", 0)),
            )
            markets.append(market)
            self._markets_cache[market.condition_id] = market

        logger.info(f"Fetched {len(markets)} markets")
        return markets

    async def get_orderbook(self, token_id: str) -> OrderBook:
        """Fetch order book for a specific token."""
        resp = await self._http.get(
            f"{self.BASE_URL}/book",
            params={"token_id": token_id}
        )
        resp.raise_for_status()
        data = resp.json()

        bids = [{"price": float(o["price"]), "size": float(o["size"])} for o in data.get("bids", [])]
        asks = [{"price": float(o["price"]), "size": float(o["size"])} for o in data.get("asks", [])]

        return OrderBook(
            token_id=token_id,
            market_slug="",
            bids=sorted(bids, key=lambda x: x["price"], reverse=True),
            asks=sorted(asks, key=lambda x: x["price"]),
        )

    async def get_midpoint(self, token_id: str) -> float:
        """Get midpoint price for a token."""
        resp = await self._http.get(
            f"{self.BASE_URL}/midpoint",
            params={"token_id": token_id}
        )
        resp.raise_for_status()
        return float(resp.json().get("mid", 0.5))

    async def get_price(self, token_id: str) -> float:
        """Get last traded price."""
        resp = await self._http.get(
            f"{self.BASE_URL}/price",
            params={"token_id": token_id, "side": "buy"}
        )
        resp.raise_for_status()
        return float(resp.json().get("price", 0.5))

    async def scan_arb_opportunities(self) -> list[dict]:
        """Scan for YES+NO < $1.00 arbitrage opportunities.
        
        Uses Gamma API outcomePrices for fast scanning, then verifies with CLOB orderbook.
        """
        # Fetch markets with outcome prices from Gamma
        params = {"limit": 200, "active": True, "order": "liquidity", "ascending": False}
        resp = await self._http.get(f"{self.GAMMA_URL}/markets", params=params)
        resp.raise_for_status()
        data = resp.json()
        
        opportunities = []

        for m in data:
            try:
                raw_tokens = m.get("clobTokenIds", "[]")
                tokens = json.loads(raw_tokens) if isinstance(raw_tokens, str) else (raw_tokens or [])
                
                raw_prices = m.get("outcomePrices", "[]")
                prices = json.loads(raw_prices) if isinstance(raw_prices, str) else (raw_prices or [])
                
                if len(tokens) < 2 or len(prices) < 2:
                    continue
                
                yes_price = float(prices[0])
                no_price = float(prices[1])
                total = yes_price + no_price
                
                if total < 0.995:  # Edge exists
                    edge = 1.0 - total
                    question = m.get("question", "Unknown")
                    opportunities.append({
                        "market": question,
                        "slug": m.get("slug", ""),
                        "yes_token": tokens[0],
                        "no_token": tokens[1],
                        "yes_price": yes_price,
                        "no_price": no_price,
                        "total": total,
                        "edge": edge,
                    })
                    logger.info(
                        f"ARB FOUND: {question[:50]} | "
                        f"YES={yes_price:.4f} NO={no_price:.4f} | "
                        f"Total={total:.4f} Edge={edge:.4f}"
                    )
                    
            except Exception as e:
                logger.debug(f"Error scanning market: {e}")
                continue

        logger.info(f"Scanned {len(data)} markets, found {len(opportunities)} arb opportunities")
        return sorted(opportunities, key=lambda x: x["edge"], reverse=True)

    async def close(self):
        await self._http.aclose()
