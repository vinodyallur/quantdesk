"""The tradeable perpetual universe, and what it costs to be wrong on each contract.

Binance lists several hundred perpetual contracts. Hardcoding a handful means the desk
can only look where someone thought to point it, so the universe is fetched and cached.

The field that matters most here is ``maintMarginPercent``. It is the maintenance margin
requirement, and it decides the *only* leverage question worth asking: at what point does
the venue close the position for you? A stop placed further away than the liquidation
distance is not a stop, it is a decoration - the position is gone before price reaches it.
That number is what turns "how much leverage" from a preference into an arithmetic
constraint, and it is available without authentication. ``leverageBracket``, which gives
the venue's own tiered caps, requires an API key and is not used.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

log = logging.getLogger(__name__)

EXCHANGE_INFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

#: Contract specs change rarely - listings and delistings, not intraday - so an hour is
#: plenty and keeps a UI search box from hammering the venue on every keystroke.
CACHE_SECONDS = 3600.0


@dataclass(slots=True)
class PerpContract:
    """One perpetual contract, with what is needed to size and to search for it."""

    symbol: str
    """Venue symbol, e.g. ``BTCUSDT``."""
    base: str
    quote: str
    maint_margin_pct: float
    """Maintenance margin as a fraction. Sets the liquidation distance."""
    price_precision: int = 2
    qty_precision: int = 3
    onboard_ms: int = 0

    @property
    def desk_symbol(self) -> str:
        """The desk's own form, e.g. ``BTC/USDT``."""
        return f"{self.base}/{self.quote}"

    @property
    def max_venue_leverage(self) -> float:
        """Leverage at which the initial margin equals the maintenance requirement.

        The ceiling implied by the margin model itself: beyond it a position is liquidated
        the instant it opens. Real venue caps are lower and tiered by notional, so treat
        this as an upper bound rather than an allowance.
        """
        if self.maint_margin_pct <= 0:
            return 125.0
        return 1.0 / self.maint_margin_pct

    def matches(self, needle: str) -> bool:
        """Loose match for a search box, on either symbol form or the base asset."""
        n = needle.strip().upper().replace("/", "")
        if not n:
            return True
        return n in self.symbol or n in self.base or n in self.desk_symbol.replace("/", "")


class PerpUniverse:
    """Cached view of every tradeable perpetual contract."""

    def __init__(self, timeout: float = 25.0) -> None:
        self.timeout = timeout
        self._contracts: dict[str, PerpContract] = {}
        self._fetched_at = 0.0

    # ------------------------------------------------------------------ loading
    def load(self, refresh: bool = False) -> dict[str, PerpContract]:
        fresh = (time.monotonic() - self._fetched_at) < CACHE_SECONDS
        if self._contracts and fresh and not refresh:
            return self._contracts
        try:
            payload = self._get(EXCHANGE_INFO)
        except OSError as exc:
            if self._contracts:
                # Serve the stale copy rather than emptying the search box on a blip.
                log.warning("exchangeInfo unreachable, using cached universe: %s", exc)
                return self._contracts
            raise
        contracts: dict[str, PerpContract] = {}
        for raw in payload.get("symbols", []):
            if raw.get("contractType") != "PERPETUAL" or raw.get("status") != "TRADING":
                continue
            try:
                contract = PerpContract(
                    symbol=raw["symbol"],
                    base=raw["baseAsset"],
                    quote=raw["quoteAsset"],
                    maint_margin_pct=float(raw.get("maintMarginPercent", 0.0)) / 100.0,
                    price_precision=int(raw.get("pricePrecision", 2)),
                    qty_precision=int(raw.get("quantityPrecision", 3)),
                    onboard_ms=int(raw.get("onboardDate", 0) or 0),
                )
            except (KeyError, TypeError, ValueError) as exc:
                log.debug("skipping malformed contract %s: %s", raw.get("symbol"), exc)
                continue
            contracts[contract.symbol] = contract
        if contracts:
            self._contracts = contracts
            self._fetched_at = time.monotonic()
        return self._contracts

    def _get(self, url: str):
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise OSError(f"{url} returned HTTP {exc.code}") from exc

    # ------------------------------------------------------------------- lookup
    def get(self, symbol: str) -> PerpContract | None:
        """Look up by either venue form (BTCUSDT) or desk form (BTC/USD)."""
        from quantdesk.data.perps import _binance_symbol

        contracts = self.load()
        return contracts.get(_binance_symbol(symbol))

    def search(self, needle: str, quote: str = "USDT", limit: int = 40) -> list[PerpContract]:
        """Contracts matching a search string, most-established first.

        Ordered by listing date rather than alphabetically: a search for "BTC" should put
        BTCUSDT above a recently listed derivative that happens to share the letters.
        """
        contracts = [
            c for c in self.load().values()
            if (not quote or c.quote == quote) and c.matches(needle)
        ]
        needle_u = needle.strip().upper().replace("/", "")
        contracts.sort(key=lambda c: (c.base != needle_u, c.onboard_ms))
        return contracts[:limit]

    def quotes(self) -> list[str]:
        return sorted({c.quote for c in self.load().values()})

    @property
    def count(self) -> int:
        return len(self._contracts)


#: Process-wide universe. Contract specs are the same for everyone, and one cache keeps a
#: search box from refetching several hundred contracts per keystroke.
UNIVERSE = PerpUniverse()
